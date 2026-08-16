"""Crash-recovery replay for events written after the latest checkpoint.

Replay reconstructs Recent Trajectory, Persistent State and WorkspaceExpected
in memory without appending new events.  It also closes tool batches that were
interrupted by a crash, synthesizing failed results for calls whose outcome was
never recorded.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..llm.base import Message, ToolCall
from ..tools.base import ToolResult
from .models import InteractionGroup, PlanStep, TaskState, ToolState
from .state_merge import merge_task_delta, merge_tool_delta


@dataclass
class ReplayResult:
    events_replayed: int = 0
    interrupted_batch: bool = False
    missing_tool_call_ids: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events_replayed": self.events_replayed,
            "interrupted_batch": self.interrupted_batch,
            "missing_tool_call_ids": list(self.missing_tool_call_ids),
            "notes": list(self.notes),
        }


class EventReplayer:
    """Idempotent in-memory application of suffix events to a context."""

    def replay(self, context: Any, events: list[dict[str, Any]]) -> ReplayResult:
        result = ReplayResult()
        pending_calls: dict[str, ToolCall] = {}
        recorded_results: set[str] = set()
        interrupted_batches: list[tuple[str, list[ToolCall]]] = []
        current_group_id = ""

        for event in events:
            current_group_id = self._apply(
                context,
                event,
                pending_calls=pending_calls,
                recorded_results=recorded_results,
                interrupted_batches=interrupted_batches,
                current_group_id=current_group_id,
            ) or current_group_id
            result.events_replayed += 1

        for group_id, calls in interrupted_batches:
            missing = [call.id for call in calls if call.id not in recorded_results]
            if missing:
                result.interrupted_batch = True
                result.missing_tool_call_ids.extend(missing)
                result.notes.append(f"closed interrupted batch with synthetic failures for {missing}")
                self._synthesize_missing_results(context, calls, recorded_results, group_id)

        if result.interrupted_batch:
            context.session.status = "recovery_pending"
            context.session.runtime_cursor.setdefault("recovery", {})["wals_recovered"] = True

        # Replayed complete groups belong in the never-folded archive as well.
        archive = getattr(context, "trajectory_archive", None)
        if archive is not None:
            existing = {str(group.get("group_id")) for group in archive.read_groups()}
            for group in context.trajectory.groups:
                if group.status == "complete" and group.group_id not in existing and group.messages:
                    archive.append_group(group)
        return result

    # ------------------------------------------------------------------ dispatch

    def _apply(
        self,
        context: Any,
        event: dict[str, Any],
        *,
        pending_calls: dict[str, ToolCall],
        recorded_results: set[str],
        interrupted_batches: list[tuple[str, list[ToolCall]]],
        current_group_id: str = "",
    ) -> str:
        event_type = str(event.get("type", ""))
        payload = event.get("payload") or {}
        if not isinstance(payload, dict):
            payload = {}

        if event_type == "user_message":
            return self._apply_user_message(context, payload)
        elif event_type == "assistant_message":
            return self._apply_assistant_message(context, payload)
        elif event_type == "tool_batch_intent":
            self._apply_tool_batch_intent(context, payload, pending_calls, interrupted_batches, current_group_id)
        elif event_type == "tool_result":
            self._apply_tool_result(context, payload, pending_calls, recorded_results)
        elif event_type == "tool_batch_closed":
            self._apply_tool_batch_closed(context, payload, pending_calls, interrupted_batches, current_group_id)
        elif event_type == "file_change":
            self._apply_file_change(context, payload)
        elif event_type == "shell_workspace_observation":
            self._apply_shell_observation(context, payload)
        elif event_type == "plan_updated":
            self._apply_plan_updated(context, payload)
        elif event_type == "state_delta_applied":
            self._apply_state_delta(context, payload)
        elif event_type == "trajectory_folded":
            self._apply_trajectory_folded(context, payload)
        elif event_type == "checkpoint_created":
            if payload.get("checkpoint_seq") is not None:
                current = dict(context.session.last_checkpoint or {})
                current["seq"] = int(payload["checkpoint_seq"])
                if payload.get("checkpoint_id"):
                    current["checkpoint_id"] = str(payload["checkpoint_id"])
                if payload.get("created_at"):
                    current["created_at"] = str(payload["created_at"])
                context.session.last_checkpoint = current
        elif event_type == "drift_detected":
            context.event_log  # ensure the log object exists; no in-memory state change
        elif event_type == "recovery_decision":
            if payload.get("action"):
                context.session.runtime_cursor.setdefault("recovery", {})["last_decision"] = dict(payload)
        elif event_type == "session_end":
            if payload.get("status"):
                context.session.status = str(payload["status"])
        return current_group_id

    # ------------------------------------------------------------------ messages

    def _apply_user_message(self, context: Any, payload: dict[str, Any]) -> str:
        group = self._group(context, str(payload.get("group_id", "")), protected=True)
        group.messages.append(Message(role="user", content=str(payload.get("content", ""))))
        group.token_count += context.token_counter.estimate_message(group.messages[-1])
        group.protected_reasons.append("replayed_user_turn")
        return group.group_id

    def _apply_assistant_message(self, context: Any, payload: dict[str, Any]) -> str:
        group = self._group(context, str(payload.get("group_id", "")))
        calls = [ToolCall.from_dict(item) for item in payload.get("tool_calls") or []]
        message = Message(role="assistant", content=payload.get("content"), tool_calls=calls)
        group.messages.append(message)
        group.token_count += context.token_counter.estimate_message(message)
        return group.group_id

    def _apply_tool_batch_intent(
        self,
        context: Any,
        payload: dict[str, Any],
        pending_calls: dict[str, ToolCall],
        interrupted_batches: list[tuple[str, list[ToolCall]]],
        current_group_id: str,
    ) -> None:
        calls = [ToolCall.from_dict(item) for item in payload.get("calls") or []]
        for call in calls:
            pending_calls[call.id] = call
        interrupted_batches.append((current_group_id, calls))

    def _apply_tool_result(
        self,
        context: Any,
        payload: dict[str, Any],
        pending_calls: dict[str, ToolCall],
        recorded_results: set[str],
    ) -> None:
        call_id = str(payload.get("call_id", ""))
        name = str(payload.get("name", ""))
        call = pending_calls.get(call_id)
        if call is None:
            args: dict[str, Any] = {}
            if context.artifact_store is not None:
                for entry in _artifact_index(context.artifact_store):
                    if entry.get("tool_call_id") == call_id:
                        args = dict(entry.get("arguments") or {})
                        break
            call = ToolCall(id=call_id, name=name, arguments=args)
            pending_calls[call_id] = call

        raw_text = str(payload.get("output_preview") or "")
        artifact_id = str(payload.get("artifact_id") or "")
        if artifact_id and context.artifact_store is not None:
            raw = context.artifact_store.read(artifact_id)
            if raw is not None:
                try:
                    raw_text = raw.decode("utf-8")
                except UnicodeDecodeError:
                    raw_text = raw.decode("utf-8", errors="replace")
        result = ToolResult(
            success=bool(payload.get("success", False)),
            output=raw_text if bool(payload.get("success", False)) else "",
            error=payload.get("error"),
        )
        observation = context._compress_observation(call, result, raw_text, artifact_id)
        group = self._group(context, str(payload.get("group_id", "")))
        message = Message(
            role="tool",
            content=observation,
            tool_call_id=call_id,
            name=name,
            is_error=not result.success,
        )
        group.messages.append(message)
        group.token_count += context.token_counter.estimate_message(message)
        recorded_results.add(call_id)

    def _apply_tool_batch_closed(
        self,
        context: Any,
        payload: dict[str, Any],
        pending_calls: dict[str, ToolCall],
        interrupted_batches: list[list[ToolCall]],
    ) -> None:
        call_ids = [str(item) for item in payload.get("call_ids") or []]
        group_id = str(payload.get("group_id") or "")
        if group_id:
            group = self._group(context, group_id)
            group.status = "complete"
        for call_id in call_ids:
            pending_calls.pop(call_id, None)

    def _synthesize_missing_results(
        self,
        context: Any,
        calls: list[ToolCall],
        recorded_results: set[str],
        group_id: str = "",
    ) -> None:
        for call in calls:
            if call.id in recorded_results:
                continue
            group = self._group(context, group_id, complete=True)
            message = Message(
                role="tool",
                content=(
                    "Tool batch interrupted before this call completed; its outcome is unknown. "
                    "Inspect the workspace before relying on any expected side effect."
                ),
                tool_call_id=call.id,
                name=call.name,
                is_error=True,
            )
            group.messages.append(message)
            group.token_count += context.token_counter.estimate_message(message)
            recorded_results.add(call.id)

    # ------------------------------------------------------------------ state / workspace

    def _apply_file_change(self, context: Any, payload: dict[str, Any]) -> None:
        expected = context.workspace_expected()
        path = str(payload.get("path", ""))
        entry = {
            "path": path,
            "status": str(payload.get("status", "??")),
            "sha256": payload.get("after_sha256") or payload.get("sha256"),
            "operation": str(payload.get("operation", "")),
            "tool_call_id": str(payload.get("tool_call_id", "")),
        }
        expected.expected_dirty = [x for x in expected.expected_dirty if x.get("path") != path]
        expected.expected_untracked = [x for x in expected.expected_untracked if x.get("path") != path]
        target = expected.expected_untracked if entry["status"] == "??" else expected.expected_dirty
        target.append(entry)
        expected.postconditions = [x for x in expected.postconditions if x.get("path") != path]
        expected.postconditions.append(dict(entry))
        expected.fingerprint = expected.recompute_fingerprint()

    def _apply_shell_observation(self, context: Any, payload: dict[str, Any]) -> None:
        expected = context.workspace_expected()
        known = {(x.get("path"), x.get("sha256")) for x in expected.expected_dirty + expected.expected_untracked}
        for item in payload.get("observed_changes") or []:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", ""))
            if (path, item.get("sha256")) in known:
                continue
            entry = dict(item)
            if entry.get("status") == "??":
                expected.expected_untracked.append(entry)
            else:
                expected.expected_dirty.append(entry)
        expected.fingerprint = expected.recompute_fingerprint()

    def _apply_plan_updated(self, context: Any, payload: dict[str, Any]) -> None:
        steps = []
        for item in payload.get("plan") or []:
            steps.append(PlanStep.from_dict(item) if isinstance(item, dict) else PlanStep(id="", text=str(item)))
        if steps:
            context.task_state.remaining = steps
            context.task_state.current = steps[0].text

    def _apply_state_delta(self, context: Any, payload: dict[str, Any]) -> None:
        task_dict = context.task_state.to_dict()
        tool_dict = context.tool_state.to_dict()
        merge_task_delta(task_dict, payload.get("task_delta") or {})
        merge_tool_delta(tool_dict, payload.get("tool_delta") or {})
        context.task_state = TaskState.from_dict(task_dict)
        context.tool_state = ToolState.from_dict(tool_dict)
        if payload.get("epoch_to") is not None:
            context.trajectory.epoch_id = int(payload["epoch_to"])
            context.session.epoch_id = int(payload["epoch_to"])

    def _apply_trajectory_folded(self, context: Any, payload: dict[str, Any]) -> None:
        folded = {str(item) for item in payload.get("folded_group_ids") or []}
        context.trajectory.groups = [group for group in context.trajectory.groups if group.group_id not in folded]

    # ------------------------------------------------------------------ group helper

    @staticmethod
    def _group(context: Any, group_id: str, *, protected: bool = False, complete: bool = False) -> InteractionGroup:
        for group in context.trajectory.groups:
            if group.group_id == group_id and group_id:
                return group
        if not group_id:
            group_id = f"g-{context.event_log.last_seq + 1:06d}"
        group = InteractionGroup(
            group_id=group_id,
            epoch_id=context.trajectory.epoch_id,
            created_step=int((context.session.runtime_cursor.get("position") or {}).get("step", 0)),
        )
        if protected:
            group.protected = True
            group.protected_reasons.append("replayed_user_turn")
        if complete:
            group.status = "complete"
        context.trajectory.groups.append(group)
        return group


def _artifact_index(artifact_store: Any):
    entries_method = getattr(artifact_store, "entries", None)
    if callable(entries_method):
        return entries_method()
    path = getattr(artifact_store, "index_path", None)
    if path is None or not path.exists():
        return []
    import json

    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return entries
