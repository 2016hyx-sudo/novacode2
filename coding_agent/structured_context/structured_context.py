"""AgentLoop-compatible structured context with trajectory fold and artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import Message, ToolCall
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
)
from .state_merge import deterministic_fold_delta, merge_task_delta, merge_tool_delta
from .token_counter import TokenCounter
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


@dataclass
class StructuredContextConfig:
    max_context_tokens: int = 256_000
    fold_trigger_ratio: float = 0.70
    fold_target_ratio: float = 0.50
    protected_recent_groups: int = 3
    protected_window_ratio: float = 0.30
    tool_output_caps: dict[str, int] = field(default_factory=lambda: dict(TOOL_OUTPUT_CAPS))
    raw_result_threshold_tokens: int = 2_000

    def trigger_tokens(self) -> int:
        return int(self.max_context_tokens * self.fold_trigger_ratio)

    def target_tokens(self) -> int:
        return int(self.max_context_tokens * self.fold_target_ratio)

    def protected_window_tokens(self) -> int:
        return int(self.max_context_tokens * self.protected_window_ratio)


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
        self.config = config or StructuredContextConfig()
        self.prefix_hash = prefix_hash
        self.tools_hash = tools_hash
        self._checkpoint_callback = None
        self._pending_batch_calls: list[ToolCall] = []
        self._current_group: InteractionGroup | None = None
        self._current_turn_group_id: str | None = None
        self._fold_count = int(session.metrics.get("fold_count", 0))
        self._last_built_messages: list[Message] = []
        self._last_estimate = 0
        self._workspace_expected = workspace_expected

    # ------------------------------------------------------------------ AgentLoop API

    @property
    def messages(self) -> list[Message]:
        self._maybe_fold()
        self._last_built_messages = self._build_messages()
        self._last_estimate = self.token_counter.estimate_prompt(
            system_text=self.system_prompt,
            tools=None,
            messages=self._last_built_messages,
        )
        return self._last_built_messages

    def add(self, message: Message) -> None:
        self._ensure_group()
        assert self._current_group is not None
        self._current_group.messages.append(message)
        self._current_group.token_count += self.token_counter.estimate_message(message)

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

    def add_assistant(self, content: str | None, tool_calls: list[ToolCall] | None = None) -> None:
        self._ensure_group()
        assert self._current_group is not None
        message = Message(role="assistant", content=content, tool_calls=tool_calls or [])
        self.add(message)
        self.event_log.append(
            "assistant_message",
            {
                "group_id": self._current_group.group_id,
                "content": content,
                "tool_calls": [call.to_dict() for call in tool_calls or []],
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
        actual = self.workspace_fingerprint.actual()
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
                "recently_inspected": [],
                "recently_modified": [x.get("path") for x in (self._workspace_expected.postconditions if self._workspace_expected else [])][:10],
            },
            "tool_execution": {"pending_calls": [], "retry_count": 0, "last_error": None},
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
            "planner": {"enabled": True, "active_item_id": active.get("id") if active else None, "replan_required": False},
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
        cap = self.config.tool_output_caps.get(call.name, 4_000)
        if self.token_counter.estimate_text(raw_text) <= self.config.raw_result_threshold_tokens:
            return raw_text
        # Deterministic head/tail reduction; tool-specific compressors can replace this.
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
            self._current_group = None

    def _maybe_fold(self) -> None:
        if self.trajectory.epoch_id < 0:
            return
        estimate = self._estimate_current()
        if estimate < self.config.trigger_tokens():
            return
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
            return

        old_epoch = self.trajectory.epoch_id
        task_delta, tool_delta = deterministic_fold_delta(
            self.task_state, self.tool_state, eligible, epoch_id=old_epoch + 1
        )
        task_dict = self.task_state.to_dict()
        tool_dict = self.tool_state.to_dict()
        merge_task_delta(task_dict, task_delta)
        merge_tool_delta(tool_dict, tool_delta)
        self.task_state = TaskState.from_dict(task_dict)
        self.tool_state = ToolState.from_dict(tool_dict)

        folded_ids = [group.group_id for group in eligible]
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

    def _estimate_current(self) -> int:
        messages = self._build_messages()
        return self.token_counter.estimate_prompt(system_text=self.system_prompt, tools=None, messages=messages)

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
