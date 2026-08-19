"""AgentLoop-compatible structured context with trajectory fold and artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import Message, ToolCall, ToolSchema
from ..llm.usage import MEASUREMENT_SCHEMA_VERSION
from ..tools.base import ToolResult, format_tool_result_for_llm
from .artifact_store import ArtifactStore
from .event_log import EventLog
from .models import (
    InteractionGroup,
    PlanStep,
    StructuredSession,
    TaskState,
    ToolState,
    Trajectory,
    WorkspaceExpected,
    utcnow,
)
from .fold_engine import FoldEngine, FoldEngineConfig, FoldResult
from .state_compact import StateCompactor, StateCompactConfig
from .state_merge import merge_task_delta, merge_tool_delta
from .token_counter import TokenCounter
from .trajectory_archive import TrajectoryArchive
from .workspace import WorkspaceFingerprint

TOOL_OUTPUT_CAPS: dict[str, int] = {
    "list_files": 1_000,
    "grep_search": 2_000,
    "read_file": 4_000,
    "run_shell": 3_000,
    "write_file": 1_000,
    "edit_file": 1_000,
    "subagent": 4_000,
    "search_files": 2_000,
}

# A fold batch is the oldest groups folded in one FoldEngine call; the fold
# loop repeats batches until the context estimate is at or below the target.
_FOLD_BATCH_MAX = 30


@dataclass
class StructuredContextConfig:
    max_context_tokens: int = 256_000
    # Folding is triggered when the estimated context usage passes this
    # fraction of the max window...
    fold_trigger_ratio: float = 0.70
    # ...and, once triggered, keeps folding the oldest groups until the
    # post-fold estimate is at or below this fraction of the pre-fold estimate
    # (e.g. 0.30 = compress the context to 30% of its current size).
    fold_target_ratio: float = 0.30
    protected_recent_groups: int = 3
    protected_window_ratio: float = 0.30
    tool_output_caps: dict[str, int] = field(default_factory=lambda: dict(TOOL_OUTPUT_CAPS))
    raw_result_threshold_tokens: int = 2_000
    fold_engine: FoldEngineConfig = field(default_factory=FoldEngineConfig)
    state_compact: StateCompactConfig = field(default_factory=StateCompactConfig)

    def trigger_tokens(self) -> int:
        return int(self.max_context_tokens * self.fold_trigger_ratio)

    def protected_window_tokens(self) -> int:
        return int(self.max_context_tokens * self.protected_window_ratio)


def compress_tool_observation(
    raw_text: str,
    *,
    tool_name: str,
    artifact_id: str,
    token_counter: TokenCounter,
    config: StructuredContextConfig | None = None,
) -> str:
    """Production head/tail compressor shared with deterministic replay."""

    selected = config or StructuredContextConfig()
    if token_counter.estimate_text(raw_text) <= selected.raw_result_threshold_tokens:
        return raw_text
    cap = selected.tool_output_caps.get(tool_name, 4_000)
    char_budget = max(500, cap * 3)
    head = raw_text[: char_budget // 2]
    tail = raw_text[-char_budget // 2 :] if len(raw_text) > char_budget else ""
    omitted = max(0, len(raw_text) - len(head) - len(tail))
    body = head
    if omitted:
        body += f"\n... [omitted {omitted} chars] ...\n"
    if tail:
        body += tail
    body += f"\n[artifact_id: {artifact_id}]"
    return body[: char_budget + 200]


class StructuredContext:
    """Structured conversation context implementing the AgentLoop context surface."""

    def __init__(
        self,
        *,
        system_prompt: str,
        session: StructuredSession,
        task_state: TaskState,
        tool_state: ToolState,
        trajectory: Trajectory,
        event_log: EventLog,
        artifact_store: ArtifactStore,
        workspace_fingerprint: WorkspaceFingerprint,
        token_counter: TokenCounter,
        trajectory_archive: TrajectoryArchive | None = None,
        config: StructuredContextConfig | None = None,
        prefix_hash: str = "",
        tools_hash: str = "",
        workspace_expected: WorkspaceExpected | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self.session = session
        self.task_state = task_state
        self.tool_state = tool_state
        self.trajectory = trajectory
        self.event_log = event_log
        self.artifact_store = artifact_store
        self.workspace_fingerprint = workspace_fingerprint
        self.token_counter = token_counter
        self.trajectory_archive = trajectory_archive
        self.config = config or StructuredContextConfig()
        self.prefix_hash = prefix_hash
        self.tools_hash = tools_hash
        self._trace = None
        self._checkpoint_callback = None
        self._post_fold_callback = None
        self._pending_batch_calls: list[ToolCall] = []
        self._current_group: InteractionGroup | None = None
        self._pending_raw_refs: list[dict[str, Any]] = []
        self._archived_group_ids: set[str] = set()
        self._current_turn_group_id: str | None = None
        self._fold_count = int(session.metrics.get("fold_count", 0))
        self._last_built_messages: list[Message] = []
        self._last_estimate = 0
        self._last_layers: dict[str, int] = {}
        self._workspace_expected = workspace_expected
        self._messages_dirty = True
        self._tool_schemas: list[ToolSchema] = []
        self._fold_engine: FoldEngine | None = None
        self._state_compactor = StateCompactor(self.token_counter, config=self.config.state_compact)
        self._actual_cache: dict[str, Any] | None = None
        self._actual_cache_key: tuple[Any, ...] | None = None
        self._planner_enabled = False
        self._invalidate_actual_cache()

    # ------------------------------------------------------------------ AgentLoop API

    @property
    def messages(self) -> list[Message]:
        # Ad-hoc inspection should not create a provider-request metric. The
        # AgentLoop uses snapshot_for_request(), which emits exactly once.
        self.prepare_for_chat(emit_trace=False)
        return self._last_built_messages

    def prepare_for_chat(
        self,
        *,
        parent_request_id: str | None = None,
        step: int | None = None,
        emit_trace: bool = False,
    ) -> None:
        """Ensure a fold decision has been made and the prompt plan is fresh."""
        self._maybe_fold(parent_request_id=parent_request_id, step=step)
        if self._messages_dirty or not self._last_built_messages:
            self._last_built_messages = self._build_messages()
            self._messages_dirty = False
        self._refresh_estimate(
            request_id=parent_request_id,
            step=step,
            emit_trace=emit_trace,
        )

    def snapshot_for_request(self, *, request_id: str, step: int) -> dict[str, Any]:
        """Build one request snapshot and its replay anchor.

        AgentLoop deep-copies the returned messages before sending them to the
        provider, so this method does not expose future context mutations.
        """

        self.prepare_for_chat(
            parent_request_id=request_id,
            step=step,
            emit_trace=True,
        )
        return {
            "messages": tuple(self._last_built_messages),
            "metadata": {
                "event_seq_anchor": self.event_log.last_seq,
                "epoch_id": self.trajectory.epoch_id,
                "layers_estimated": dict(self._last_layers),
            },
        }

    def _refresh_estimate(
        self,
        *,
        request_id: str | None = None,
        step: int | None = None,
        emit_trace: bool = False,
    ) -> None:
        layers = self._estimate_layers()
        self._last_layers = dict(layers)
        self._last_estimate = layers["total"]
        if self._trace is not None and emit_trace:
            self._trace.emit(
                "context_estimate",
                measurement_schema_version=MEASUREMENT_SCHEMA_VERSION,
                request_id=request_id,
                agent_role="main",
                step=step,
                event_seq_anchor=self.event_log.last_seq,
                epoch_id=self.trajectory.epoch_id,
                **layers,
            )

    def _invalidate_messages(self) -> None:
        self._messages_dirty = True

    def _invalidate_actual_cache(self) -> None:
        self._actual_cache = None
        self._actual_cache_key = None

    def _get_actual(self) -> dict[str, Any]:
        key = (
            int((self.session.runtime_cursor.get("position") or {}).get("step", 0)),
            self.trajectory.epoch_id,
            len(self.trajectory.groups),
        )
        if self._actual_cache is not None and self._actual_cache_key == key:
            return self._actual_cache
        self._actual_cache = self.workspace_fingerprint.actual()
        self._actual_cache_key = key
        return self._actual_cache

    def add(self, message: Message) -> None:
        self._ensure_group()
        assert self._current_group is not None
        self._current_group.messages.append(message)
        self._current_group.token_count += self.token_counter.estimate_message(message)
        self._invalidate_messages()

    def add_user(self, content: str) -> None:
        self._close_group()
        self._ensure_group()
        assert self._current_group is not None
        self._current_group.protected = True
        self._current_group.protected_reasons.append("current_user_turn")
        self._current_turn_group_id = self._current_group.group_id
        self.add(Message(role="user", content=content))
        self.event_log.append("user_message", {"group_id": self._current_group.group_id, "content": content})

    def add_system(self, content: str) -> None:
        # System-level runtime notes stay out of Recent Trajectory. AgentLoop does
        # not use this path today; plans go through set_plan().
        self.event_log.append("system_note", {"content": content})

    def set_fold_engine(self, engine: FoldEngine) -> None:
        self._fold_engine = engine

    def set_tool_schemas(self, schemas: list[ToolSchema] | tuple[ToolSchema, ...]) -> None:
        self._tool_schemas = list(schemas)
        self._invalidate_messages()

    def set_state_compactor(self, compactor: StateCompactor) -> None:
        self._state_compactor = compactor

    def set_planner_enabled(self, enabled: bool) -> None:
        self._planner_enabled = bool(enabled)
        self._invalidate_messages()

    def start_new_epoch(self, *, reason: str) -> int:
        old_epoch = self.trajectory.epoch_id
        self.trajectory.epoch_id = old_epoch + 1
        self.session.epoch_id = self.trajectory.epoch_id
        self.event_log.append(
            "epoch_start",
            {"epoch_from": old_epoch, "epoch_to": self.trajectory.epoch_id, "reason": reason},
        )
        self._invalidate_messages()
        self._invalidate_actual_cache()
        return self.trajectory.epoch_id

    def mark_findings_stale(self, affected_paths: set[str]) -> int:
        count = 0
        normalized = {path.replace("\\", "/") for path in affected_paths}
        for finding in self.task_state.key_findings:
            evidence_paths = {
                str(item.get("path", "")).replace("\\", "/")
                for item in finding.evidence
                if isinstance(item, dict) and item.get("path")
            }
            if evidence_paths & normalized and finding.status != "stale":
                finding.status = "stale"
                finding.updated_step = int((self.session.runtime_cursor.get("position") or {}).get("step", 0))
                count += 1
        self._invalidate_messages()
        return count

    def mark_verification_stale(self) -> None:
        verification = self.session.runtime_cursor.setdefault("verification", {})
        if verification.get("status") not in {"not_required", "stale"}:
            verification["status"] = "stale"
        self._invalidate_messages()

    def apply_external_file_change(
        self,
        *,
        path: str,
        operation: str = "subagent",
        depth: int = 0,
        source: str = "subagent",
    ) -> dict[str, Any] | None:
        """Record a file change made outside the current AgentLoop (e.g. subagent)."""
        if not path:
            return None
        actual = self.workspace_fingerprint.actual(extra_paths=[path])
        file_hash = (actual.get("extra_hashes") or {}).get(path)
        status = "??"
        for item in actual.get("tracked_changes") or []:
            if item.get("path") == path:
                status = str(item.get("status", "M"))
                break
        if status == "??":
            for item in actual.get("untracked") or []:
                if item.get("path") == path:
                    status = "??"
                    break
        entry = {
            "path": path,
            "status": status,
            "sha256": file_hash,
            "operation": operation,
            "depth": depth,
            "source": source,
        }
        expected = self._expected()
        expected.expected_dirty = [x for x in expected.expected_dirty if x.get("path") != path]
        expected.expected_untracked = [x for x in expected.expected_untracked if x.get("path") != path]
        target = expected.expected_untracked if status == "??" else expected.expected_dirty
        target.append(entry)
        expected.postconditions = [x for x in expected.postconditions if x.get("path") != path]
        expected.postconditions.append(dict(entry))
        expected.fingerprint = expected.recompute_fingerprint()
        self.event_log.append("file_change", entry)
        self._invalidate_actual_cache()
        self._invalidate_messages()
        return entry

    def add_assistant(
        self,
        content: str | None,
        tool_calls: list[ToolCall] | None = None,
        *,
        raw_content: list[dict[str, Any]] | None = None,
    ) -> None:
        self._ensure_group()
        assert self._current_group is not None
        message = Message(
            role="assistant", content=content, tool_calls=tool_calls or [], raw_content=raw_content
        )
        self.add(message)
        self.event_log.append(
            "assistant_message",
            {
                "group_id": self._current_group.group_id,
                "content": content,
                "tool_calls": [call.to_dict() for call in tool_calls or []],
                "raw_content": raw_content,
            },
        )

    def add_tool_result(self, call: ToolCall, result: ToolResult) -> None:
        self._ensure_group()
        assert self._current_group is not None
        raw_text = format_tool_result_for_llm(call.name, result)
        artifact_entry = self.artifact_store.save(
            call.name,
            raw_text,
            tool_call_id=call.id,
            arguments=call.arguments,
        )
        if self._trace is not None:
            self._trace.emit(
                "artifact_stored",
                artifact_id=artifact_entry["artifact_id"],
                tool=call.name,
                tool_call_id=call.id,
                size=artifact_entry["size"],
            )
        self._pending_raw_refs.append(
            {
                "tool_call_id": call.id,
                "name": call.name,
                "artifact_id": artifact_entry["artifact_id"],
            }
        )
        observation = self._compress_observation(call, result, raw_text, artifact_entry["artifact_id"])
        self.add(Message(role="tool", content=observation, tool_call_id=call.id, name=call.name, is_error=not result.success))
        self.event_log.append(
            "tool_result",
            {
                "group_id": self._current_group.group_id,
                "call_id": call.id,
                "name": call.name,
                "success": result.success,
                "error": result.error,
                "output_preview": observation[:500],
                "artifact_id": artifact_entry["artifact_id"],
            },
        )
        if result.success and call.name in {"write_file", "edit_file"}:
            self._record_file_change(call)
        elif call.name == "run_shell":
            self._record_shell_observation(call, result.success, result.error)

    def set_plan(self, plan: object) -> None:
        render = getattr(plan, "render", None)
        text = render() if callable(render) else str(plan)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        steps: list[PlanStep] = []
        for index, line in enumerate(lines):
            if line == "(no plan)":
                continue
            line = line.split(". ", 1)[1] if line[:2].strip(" .") and ". " in line else line
            steps.append(PlanStep(id=f"plan-{index + 1}", text=line, status="active" if index == 0 else "pending"))
        self.task_state.remaining = steps
        if steps:
            self.task_state.current = steps[0].text
        self.event_log.append("plan_updated", {"plan": [step.to_dict() for step in steps]})
        self._invalidate_messages()

    # ------------------------------------------------------------------ prompt building

    def _build_messages(self) -> list[Message]:
        from dataclasses import replace

        messages: list[Message] = [Message(role="system", content=self.system_prompt)]
        messages.append(Message(role="user", content=self._state_block_text()))
        trajectory_messages: list[Message] = []
        for group in self.trajectory.groups:
            trajectory_messages.extend(list(group.messages))
        if trajectory_messages:
            last = trajectory_messages[-1]
            trajectory_messages[-1] = replace(last, cache_control=True)
        messages.extend(trajectory_messages)
        messages.append(Message(role="user", content=self._agent_state_text()))
        return messages

    def _state_block_text(self) -> str:
        task = self.task_state.to_dict()
        tool = self.tool_state.to_dict()
        return (
            "<structured_state>\n"
            "<task_state>\n"
            f"{json.dumps(task, ensure_ascii=False, indent=2)}\n"
            "</task_state>\n"
            "<tool_state>\n"
            f"{json.dumps(tool, ensure_ascii=False, indent=2)}\n"
            "</tool_state>\n"
            "</structured_state>"
        )

    def _agent_state_text(self) -> str:
        actual = self._get_actual()
        git = actual.get("git") or {}
        task = self.task_state
        cursor = self.session.runtime_cursor
        position = cursor.get("position") or {}
        verification = cursor.get("verification") or {}
        limits = cursor.get("limits") or {}
        active = None
        for step in task.remaining:
            if step.status == "active":
                active = step.to_dict()
                break
        state = {
            "schema_version": "1.0",
            "position": {
                "user_turn": position.get("user_turn", 0),
                "step": position.get("step", 0),
                "epoch": self.trajectory.epoch_id,
                "phase": position.get("phase", "exploration"),
                "status": self.session.status,
                "attempt": position.get("attempt", 0),
            },
            "focus": {
                "task_id": task.task_id,
                "current_goal_ref": active.get("id") if active else None,
                "current_goal": task.current or active.get("text") if active else None,
                "success_criteria": task.success_criteria,
                "next_action": (active or {}).get("text") if active else None,
            },
            "todo": {
                "active": active,
                "next": [step.to_dict() for step in task.remaining[:3] if step.status != "active"],
                "blocked": [item.to_dict() for item in task.unresolved if item.blocked],
            },
            "workspace": {"cwd": str(self.workspace_fingerprint.root)},
            "git": {
                "branch": git.get("branch", ""),
                "head": git.get("head", ""),
                "dirty": bool(actual.get("tracked_changes") or actual.get("untracked")),
                "modified_files": [x["path"] for x in actual.get("tracked_changes") or [] if x.get("status", "") != "??"],
                "untracked_files": [x["path"] for x in actual.get("untracked") or []],
                "conflicted_files": [x["path"] for x in actual.get("tracked_changes") or [] if "U" in x.get("status", "")],
            },
            "working_set": {
                "focused_files": [x.get("path") for x in (self._workspace_expected.postconditions if self._workspace_expected else [])][:10],
                "recently_inspected": [
                    item.value.get("path", "") for item in self.tool_state.profiles.get("read", {}).get("useful_files", []) if item.value.get("path")
                ][:10],
                "recently_modified": [x.get("path") for x in (self._workspace_expected.postconditions if self._workspace_expected else [])][:10],
            },
            "tool_execution": {
                "pending_calls": [call.to_dict() for call in self._pending_batch_calls],
                "retry_count": 0,
                "last_error": None,
            },
            "verification": verification,
            "limits": limits,
            "context": {
                "epoch": self.trajectory.epoch_id,
                "window_limit_tokens": self.config.max_context_tokens,
                "prompt_tokens": self._last_estimate,
                "usage_ratio": round(self._last_estimate / self.config.max_context_tokens, 4) if self.config.max_context_tokens else 0,
                "pressure": "normal" if self._last_estimate < self.config.trigger_tokens() else "elevated",
                "fold_count": self._fold_count,
                "prefix_hash": self.prefix_hash,
                "tools_hash": self.tools_hash,
            },
            "planner": {"enabled": self._planner_enabled, "active_item_id": active.get("id") if active else None, "replan_required": False},
            "recovery": {
                "blocked": self.session.status == "blocked",
                "checkpoint_id": self.session.last_checkpoint.get("checkpoint_id"),
                "resumable": self.session.status != "blocked",
                "resume_action": "continue_next_action",
            },
        }
        return "<agent_state>\n" + json.dumps(state, ensure_ascii=False, indent=2) + "\n</agent_state>"

    # ------------------------------------------------------------------ tool observation

    def _compress_observation(self, call: ToolCall, result: ToolResult, raw_text: str, artifact_id: str) -> str:
        return compress_tool_observation(
            raw_text,
            tool_name=call.name,
            artifact_id=artifact_id,
            token_counter=self.token_counter,
            config=self.config,
        )

    def _record_file_change(self, call: ToolCall) -> None:
        path = str((call.arguments or {}).get("path", ""))
        if not path:
            return
        actual = self.workspace_fingerprint.actual(extra_paths=[path])
        file_hash = (actual.get("extra_hashes") or {}).get(path)
        status = "??"
        for item in actual.get("tracked_changes") or []:
            if item["path"] == path:
                status = item.get("status", "M")
                break
        if status == "??":
            for item in actual.get("untracked") or []:
                if item["path"] == path:
                    status = "??"
                    break
        entry = {
            "path": path,
            "status": status,
            "sha256": file_hash,
            "operation": call.name,
            "tool_call_id": call.id,
        }
        expected = self._expected()
        expected.expected_dirty = [x for x in expected.expected_dirty if x["path"] != path]
        expected.expected_untracked = [x for x in expected.expected_untracked if x["path"] != path]
        target = expected.expected_untracked if status == "??" else expected.expected_dirty
        target.append(entry)
        expected.postconditions = [x for x in expected.postconditions if x["path"] != path]
        expected.postconditions.append(dict(entry))
        expected.fingerprint = expected.recompute_fingerprint()
        self.event_log.append(
            "file_change",
            {
                "group_id": self._current_group.group_id if self._current_group else None,
                "tool_call_id": call.id,
                **entry,
                "before_sha256": "",
                "after_sha256": file_hash,
                "size": None,
            },
        )
        if self._current_group is not None:
            self._current_group.workspace_changes.append({"path": path, "operation": call.name, "after_sha256": file_hash})
        self._invalidate_actual_cache()
        self._invalidate_messages()

    def _record_shell_observation(self, call: ToolCall, success: bool, error: str | None) -> None:
        expected = self._expected()
        actual = self.workspace_fingerprint.actual()
        observed: list[dict[str, Any]] = []
        known = {(x["path"], x.get("sha256")) for x in expected.expected_dirty + expected.expected_untracked}
        for item in list(actual.get("tracked_changes") or []) + list(actual.get("untracked") or []):
            if (item["path"], item.get("sha256")) not in known:
                observed.append(item)
                if item.get("status") == "??":
                    expected.expected_untracked.append(dict(item))
                else:
                    expected.expected_dirty.append(dict(item))
        if observed:
            expected.fingerprint = expected.recompute_fingerprint()
            self.event_log.append(
                "shell_workspace_observation",
                {
                    "group_id": self._current_group.group_id if self._current_group else None,
                    "tool_call_id": call.id,
                    "command": (call.arguments or {}).get("command", ""),
                    "exit_code": 0 if success else None,
                    "error": error,
                    "observed_changes": observed,
                },
            )
        self._invalidate_actual_cache()
        self._invalidate_messages()

    def _expected(self) -> WorkspaceExpected:
        if self._workspace_expected is None:
            self._workspace_expected = self.workspace_fingerprint.expected_from_actual()
        return self._workspace_expected

    def set_workspace_expected(self, expected: WorkspaceExpected) -> None:
        self._workspace_expected = expected

    # ------------------------------------------------------------------ trajectory fold

    def _ensure_group(self) -> None:
        if self._current_group is not None:
            return
        self._pending_raw_refs = []
        group_id = f"g-{self.event_log.last_seq + 1:06d}"
        self._current_group = InteractionGroup(
            group_id=group_id,
            epoch_id=self.trajectory.epoch_id,
            created_step=int((self.session.runtime_cursor.get("position") or {}).get("step", 0)),
        )
        self.trajectory.groups.append(self._current_group)

    def _close_group(self) -> None:
        if self._current_group is not None:
            self._current_group.status = "complete"
            if self._current_group.source_events.get("last_seq") is None:
                self._current_group.source_events["last_seq"] = self.event_log.last_seq
            if (
                self.trajectory_archive is not None
                and self._current_group.group_id not in self._archived_group_ids
                and self._current_group.messages
            ):
                self.trajectory_archive.append_group(
                    self._current_group,
                    raw_tool_result_refs=self._pending_raw_refs,
                )
                self._archived_group_ids.add(self._current_group.group_id)
            if self._trace is not None:
                self._trace.emit(
                    "interaction_group_closed",
                    group_id=self._current_group.group_id,
                    epoch_id=self._current_group.epoch_id,
                    message_count=len(self._current_group.messages),
                )
            self._current_group = None
            self._pending_raw_refs = []

    def _maybe_fold(
        self,
        *,
        parent_request_id: str | None = None,
        step: int | None = None,
    ) -> None:
        if self.trajectory.epoch_id < 0:
            return
        before = self._estimate_layers()
        if before["total"] < self.config.trigger_tokens():
            return
        # Once triggered, keep folding the oldest groups (in bounded batches)
        # until the post-fold estimate is at or below `fold_target_ratio` x the
        # pre-fold estimate, or nothing foldable remains.  How many of the most
        # recent groups survive is derived from that single ratio, not from a
        # fixed count.
        target_tokens = max(1, int(before["total"] * self.config.fold_target_ratio))
        step = (
            step
            if step is not None
            else int((self.session.runtime_cursor.get("position") or {}).get("step", 0))
        )
        batches_folded = 0
        while True:
            eligible = [
                group
                for group in self.trajectory.groups
                if not group.protected
                and group.status == "complete"
                and group.group_id != self._current_turn_group_id
            ]
            if len(self.trajectory.groups) > self.config.protected_recent_groups:
                eligible = eligible[: max(1, len(self.trajectory.groups) - self.config.protected_recent_groups)]
            if not eligible:
                break
            batch = eligible
            if len(batch) > _FOLD_BATCH_MAX:
                batch = batch[:_FOLD_BATCH_MAX]

            batch_before = self._estimate_layers()
            old_epoch = self.trajectory.epoch_id
            engine = self._fold_engine or FoldEngine(
                None, token_counter=self.token_counter, config=self.config.fold_engine
            )
            fold_result: FoldResult = engine.fold(
                task_state=self.task_state,
                tool_state=self.tool_state,
                groups=batch,
                epoch_id=old_epoch + 1,
                artifact_store=self.artifact_store,
                parent_request_id=parent_request_id,
                step=step,
                event_seq_anchor=self.event_log.last_seq,
                request_epoch_id=old_epoch,
            )
            task_delta = fold_result.task_delta
            tool_delta = fold_result.tool_delta

            task_dict = self.task_state.to_dict()
            tool_dict = self.tool_state.to_dict()
            merge_task_delta(task_dict, task_delta)
            merge_tool_delta(tool_dict, tool_delta)
            self.task_state = TaskState.from_dict(task_dict)
            self.tool_state = ToolState.from_dict(tool_dict)

            # Capacity control runs after every fold/merge.
            compact_result = self._state_compactor.compact(
                self.task_state,
                self.tool_state,
                artifact_store=self.artifact_store,
            )
            if compact_result.evicted_task_items or compact_result.evicted_tool_items:
                self.event_log.append("state_compact", compact_result.to_dict())

            folded_ids = [group.group_id for group in batch]
            folded_tokens = sum(group.token_count for group in batch)
            task_delta_tokens = self.token_counter.estimate_text(json.dumps(task_delta, ensure_ascii=False))
            tool_delta_tokens = self.token_counter.estimate_text(json.dumps(tool_delta, ensure_ascii=False))
            self.trajectory.groups = [group for group in self.trajectory.groups if group.group_id not in folded_ids]
            self.trajectory.epoch_id = old_epoch + 1
            self.session.epoch_id = self.trajectory.epoch_id
            self._fold_count += 1
            self.session.metrics["fold_count"] = self._fold_count
            self.event_log.append(
                "state_delta_applied",
                {
                    "epoch_from": old_epoch,
                    "epoch_to": self.trajectory.epoch_id,
                    "folded_group_ids": folded_ids,
                    "task_delta": task_delta,
                    "tool_delta": tool_delta,
                },
            )
            self.event_log.append("trajectory_folded", {"folded_group_ids": folded_ids})

            self._invalidate_messages()
            self._invalidate_actual_cache()
            batches_folded += 1
            after = self._estimate_layers()
            residual_ratio = 0.0
            if folded_tokens > 0:
                residual_ratio = (task_delta_tokens + tool_delta_tokens) / folded_tokens
            fold_reduction_ratio = 1.0 - residual_ratio if folded_tokens > 0 else 0.0
            fold_event = {
                "fold_id": f"fold-{self._fold_count}",
                "ts": utcnow(),
                "trigger": {
                    "threshold_tokens": self.config.trigger_tokens(),
                    "threshold_ratio": self.config.fold_trigger_ratio,
                    "estimated_tokens_before": batch_before["total"],
                    "usage_ratio_before": round(batch_before["total"] / self.config.max_context_tokens, 6),
                },
                "before": batch_before,
                "after": after,
                "folded": {
                    "group_count": len(batch),
                    "group_ids": folded_ids,
                    "trajectory_tokens_removed": folded_tokens,
                    "task_delta_tokens": task_delta_tokens,
                    "tool_delta_tokens": tool_delta_tokens,
                    # Kept for schema compatibility; this is a residual ratio, not
                    # the end-to-end Context Reduction Ratio (CRR).
                    "compression_ratio": round(residual_ratio, 6),
                    "residual_ratio": round(residual_ratio, 6),
                    "fold_reduction_ratio": round(fold_reduction_ratio, 6),
                },
                "model": fold_result.model_stats(),
                "result": {
                    "target_tokens": target_tokens,
                    "target_met": after["total"] <= target_tokens,
                    "new_epoch": self.trajectory.epoch_id,
                },
            }
            self.event_log.append("fold_event", fold_event)
            self.session.metrics["last_fold"] = {
                "fold_id": fold_event["fold_id"],
                "ts": fold_event["ts"],
                "estimated_tokens_before": batch_before["total"],
                "estimated_tokens_after": after["total"],
                "compression_ratio": fold_event["folded"]["residual_ratio"],
                "residual_ratio": fold_event["folded"]["residual_ratio"],
                "fold_reduction_ratio": fold_event["folded"]["fold_reduction_ratio"],
                "target_met": fold_event["result"]["target_met"],
            }
            self.session.metrics["total_folded_trajectory_tokens"] = int(
                self.session.metrics.get("total_folded_trajectory_tokens", 0)
            ) + folded_tokens
            self.session.metrics["last_fold_residual_ratio"] = fold_event["folded"]["residual_ratio"]
            self.session.metrics["last_fold_reduction_ratio"] = fold_event["folded"]["fold_reduction_ratio"]
            self.session.metrics["last_compression_ratio"] = fold_event["folded"]["residual_ratio"]
            if self._trace is not None:
                self._trace.emit("fold_event", **fold_event)
            if after["total"] <= target_tokens:
                break

        # Fold is a safe cut point: persist a post_fold checkpoint once, after
        # the whole fold loop has converged to the target.
        if batches_folded:
            step = int((self.session.runtime_cursor.get("position") or {}).get("step", 0))
            if self._post_fold_callback is not None:
                self._post_fold_callback(step)
            elif self._checkpoint_callback is not None:
                self._checkpoint_callback(step)

    def _estimate_layers(self) -> dict[str, int]:
        state_block = self._state_block_text()
        agent_block = self._agent_state_text()
        stable_prefix = self.token_counter.estimate_text(self.system_prompt)
        tools = self.token_counter.estimate_tools(self._tool_schemas)
        task_tool_state = self.token_counter.estimate_text(state_block)
        recent_trajectory = sum(group.token_count for group in self.trajectory.groups)
        agent_state = self.token_counter.estimate_text(agent_block)
        raw_total = stable_prefix + tools + task_tool_state + recent_trajectory + agent_state
        coefficient = self.token_counter.calibration.coefficient
        total = max(1, int(raw_total * coefficient)) if raw_total > 0 else 1
        return {
            "stable_prefix": stable_prefix,
            "tools": tools,
            "task_tool_state": task_tool_state,
            "recent_trajectory": recent_trajectory,
            "agent_state": agent_state,
            "total": total,
        }

    def _estimate_current(self) -> int:
        return self._estimate_layers()["total"]

    # ------------------------------------------------------------------ snapshot helpers

    def snapshot_task_state(self) -> TaskState:
        return self.task_state

    def snapshot_tool_state(self) -> ToolState:
        return self.tool_state

    def snapshot_trajectory(self) -> Trajectory:
        return self.trajectory

    def workspace_expected(self) -> WorkspaceExpected:
        return self._expected()

    def set_checkpoint_callback(self, callback: Any) -> None:
        self._checkpoint_callback = callback

    def set_post_fold_callback(self, callback: Any) -> None:
        self._post_fold_callback = callback

    def set_trace(self, trace: Any) -> None:
        self._trace = trace

    def on_tool_batch_start(self, calls: list[ToolCall], step: int) -> None:
        self._pending_batch_calls = list(calls)
        self.event_log.append(
            "tool_batch_intent",
            {
                "step": step,
                "calls": [call.to_dict() for call in calls],
            },
        )

    def on_tool_batch_complete(self, step: int) -> None:
        self.event_log.append(
            "tool_batch_closed",
            {"step": step, "call_ids": [call.id for call in self._pending_batch_calls]},
        )
        self._pending_batch_calls = []
        self._close_group()
        self._invalidate_messages()
        self._invalidate_actual_cache()
        if self._checkpoint_callback is not None:
            self._checkpoint_callback(step)

    def record_usage(self, usage: dict[str, Any] | None) -> None:
        if usage and self._last_estimate:
            self.token_counter.record_usage(self._last_estimate, usage)

    def update_runtime_cursor(self, *, step: int, tool_calls_used: int, status: str) -> None:
        position = self.session.runtime_cursor.setdefault("position", {})
        position["step"] = step
        position["status"] = status
        limits = self.session.runtime_cursor.setdefault("limits", {})
        limits["steps_used"] = step
        limits["tool_calls_used"] = tool_calls_used
        self.session.status = status

    def finalize(self) -> None:
        self._close_group()
