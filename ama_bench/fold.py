"""Offline construction of NovaCode structured memory from a trajectory.

``NovaCodeMemoryBuilder`` replays a parsed trajectory through NovaCode's own
components: steps become Interaction Groups, deterministic or LLM-assisted
Trajectory Fold distills them into Task State (durable findings, completed
work) and Tool State (reusable tool experience), then a capacity-control pass
evicts low-value entries exactly like the live harness does.

The builder is deliberately dependency-light: it requires only the
``coding_agent`` package that ships with NovaCode, plus (optionally) the
benchmark's ``ModelClient`` for the LLM-assisted fold path.
"""
from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coding_agent.llm.base import LLMProvider, Message, ToolCall
from coding_agent.structured_context.artifact_store import ArtifactStore
from coding_agent.structured_context.event_log import EventLog
from coding_agent.structured_context.fold_engine import (
    FoldEngine,
    FoldEngineConfig,
    FoldResult,
)
from coding_agent.structured_context.models import (
    InteractionGroup,
    TaskState,
    ToolState,
    Trajectory,
)
from coding_agent.structured_context.state_compact import (
    StateCompactConfig,
    StateCompactor,
)
from coding_agent.structured_context.state_merge import (
    merge_task_delta,
    merge_tool_delta,
)
from coding_agent.structured_context.token_counter import TokenCounter

from .memory import NovaCodeMemory
from .steps import Step

# A tool batch is one assistant message with tool calls plus the tool results,
# mirroring how the live harness groups interactions.  Small batches keep long
# trajectories fine-grained enough for meaningful folds.
_BATCH_CAP = 2
_MAX_GROUPS = 30

_MERGE_TABLE = {
    "progress.completed": ("progress", "completed"),
    "progress.current": ("progress", "current"),
    "progress.remaining": ("progress", "remaining"),
    "key_findings": ("key_findings",),
    "decisions": ("decisions",),
    "unresolved": ("unresolved",),
}
_EXPERIENCE_SLOTS = ("useful_scopes", "effective_queries", "known_error_patterns")
_READ_SLOTS = ("useful_files", "effective_ranges", "known_symbols")
_SHELL_SLOTS = ("effective_commands", "known_failures")
_TEST_SLOTS = ("effective_commands", "known_failures")


def _parse_tool_command(action: str) -> tuple[str, str] | None:
    """Return ``(tool_name, tool_arguments)`` for one plausible tool call."""
    stripped = action.strip()
    if not stripped:
        return None
    if stripped.startswith("{"):
        stripped = stripped.lstrip("{").strip()
        stripped = stripped.removeprefix('"')
        stripped = stripped.split('"', 1)[0].strip()
    if "(" not in stripped:
        return None
    name, _, rest = stripped.partition("(")
    name = name.strip().strip(".")
    if not name or not name.replace("_", "").isalnum():
        return None
    args = rest.rsplit(")", 1)[0].strip()
    return name, args


def _add_tool_experience(profile: dict[str, list[Any]], slot: str, value: str) -> None:
    """Append *value* to *slot* unless it duplicates an existing entry."""
    existing = [str(item.get("value") or "") for item in profile.get(slot, [])]
    if value and value not in existing and len(existing) < 40:
        experience_id = f"e-{len(existing) + 1:03d}"
        profile.setdefault(slot, []).append(
            {"id": experience_id, "kind": "learned", "value": {"text": value}, "status": "valid", "updated_step": 0}
        )


def _tool_argument_value(action: str) -> str:
    """Best-effort first argument of a tool call, for the ``read``/``write`` tools."""
    _, args = _parse_tool_command(action) or (None, None)
    if not args:
        return ""
    for token in args.replace(",", " ").split():
        stripped = token.strip("'\"")
        if stripped.startswith(("/", "./", "src", "tests", ".")):
            return stripped
    return args.split(",")[0].strip().strip("'\"")


def _normalize_slots(tool_state: ToolState) -> None:
    """Drop slot names that changed between schema drafts during live runs."""
    for profile in tool_state.profiles.values():
        for slot in tuple(profile):
            if slot not in _EXPERIENCE_SLOTS and slot not in _READ_SLOTS and slot not in _SHELL_SLOTS and slot not in _TEST_SLOTS:
                profile.pop(slot, None)


@dataclass
class MemoryBuildStats:
    """Transparent summary of one memory construction run.

    Beyond counts, the stats carry the audit trail needed to debug answers:
    a structural inventory of every pre-fold group, one record per fold
    epoch (what the model folded, deltas applied, fallback/errors), and token
    estimates before and after compression.  ``work_dir`` is populated only
    when ``keep_work_dir`` is set; it points at the surviving build directory
    (``groups.jsonl`` holds the full pre-fold content).
    """

    steps: int = 0
    groups: int = 0
    groups_folded: int = 0
    groups_kept: int = 0
    task_entries: int = 0
    tool_entries: int = 0
    model_folds: int = 0
    fallback_folds: int = 0
    fold_calls: int = 0
    fold_errors: int = 0
    pre_fold_tokens: int = 0
    post_fold_tokens: int = 0
    compact_task_evicted: list[dict[str, Any]] = field(default_factory=list)
    compact_tool_evicted: list[dict[str, Any]] = field(default_factory=list)
    work_dir: str | None = None
    notes: list[str] = field(default_factory=list)
    group_inventory: list[dict[str, Any]] = field(default_factory=list)
    fold_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "groups": self.groups,
            "groups_folded": self.groups_folded,
            "groups_kept": self.groups_kept,
            "task_entries": self.task_entries,
            "tool_entries": self.tool_entries,
            "model_folds": self.model_folds,
            "fallback_folds": self.fallback_folds,
            "fold_calls": self.fold_calls,
            "fold_errors": self.fold_errors,
            "pre_fold_tokens": self.pre_fold_tokens,
            "post_fold_tokens": self.post_fold_tokens,
            "compact_task_evicted": self.compact_task_evicted,
            "compact_tool_evicted": self.compact_tool_evicted,
            "work_dir": self.work_dir,
            "notes": list(self.notes),
            "group_inventory": [dict(item) for item in self.group_inventory],
            "fold_events": [dict(event) for event in self.fold_events],
        }


@dataclass
class BuilderConfig:
    max_context_tokens: int = 96_000
    fold_trigger_ratio: float = 0.70
    fold_target_ratio: float = 0.45
    protected_recent_groups: int = 2
    fold_max_input_chars: int = 60_000
    fold_max_attempts: int = 2
    compact_task_budget_tokens: int = 8_000
    compact_tool_budget_tokens: int = 8_000
    work_dir: Path | None = None
    keep_work_dir: bool = False
    groups_per_batch: int = _BATCH_CAP


class NovaCodeMemoryBuilder:
    """Build a :class:`NovaCodeMemory` from a trajectory using NovaCode's stack."""

    def __init__(
        self,
        *,
        fold_provider: LLMProvider | None = None,
        config: BuilderConfig | None = None,
        token_counter: TokenCounter | None = None,
    ) -> None:
        self.fold_provider = fold_provider
        self.config = config or BuilderConfig()
        self.token_counter = token_counter or TokenCounter()
        self.fold_engine = FoldEngine(
            fold_provider,
            token_counter=self.token_counter,
            config=FoldEngineConfig(
                max_attempts=self.config.fold_max_attempts,
                max_input_chars=self.config.fold_max_input_chars,
            ),
        )

    # ------------------------------------------------------------------ construction

    def build(
        self,
        steps: list[Step],
        *,
        task: str = "",
        objective: str = "",
        task_id: str = "ama-episode",
    ) -> NovaCodeMemory:
        stats = MemoryBuildStats()
        stats.steps = len(steps)
        fold_dir = self._work_dir()
        artifact_store = ArtifactStore(fold_dir / "artifacts", fsync=False)
        event_log = EventLog(fold_dir / "events.jsonl", fsync=False)
        task_state = TaskState.new(task_id=task_id, objective=objective or task or "Trajectory memory construction")
        tool_state = ToolState.new()
        trajectory = Trajectory(epoch_id=0)
        if not steps:
            stats.notes.append("empty trajectory; nothing to fold")
            return NovaCodeMemory(task_state=task_state, tool_state=tool_state, trajectory=trajectory, stats=stats.to_dict())
        groups: list[InteractionGroup] = []

        for batch in self._batches(steps, self.config.groups_per_batch):
            group = self._batch_group(batch, group_id=f"g-{len(groups) + 1:06d}", epoch_id=trajectory.epoch_id)
            groups.append(group)
            trajectory.groups.append(group)
            stats.groups += 1
            stats.group_inventory.append(
                {
                    "id": group.group_id,
                    "message_count": len(group.messages),
                    "chars": sum(len(str(message.content or "")) for message in group.messages),
                    "turn_range": _group_turn_range(group),
                }
            )
            event_log.append("tool_batch_closed", {"step": batch[-1].turn_idx, "call_ids": []})
        stats.pre_fold_tokens = sum(
            self.token_counter.estimate_text(
                json.dumps(group.to_dict(), ensure_ascii=False, separators=(",", ":"))
            )
            for group in groups
        )
        # Full pre-fold content (all groups) — consumed by the audit trail when
        # keep_work_dir is enabled; deleted with the work dir otherwise.
        with (fold_dir / "groups.jsonl").open("w", encoding="utf-8") as handle:
            for group in groups:
                handle.write(
                    json.dumps(group.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n"
                )

        kept, folded, final_task, final_tool, final_trajectory = self._fold_all(
            groups=groups,
            task_state=task_state,
            tool_state=tool_state,
            artifact_store=artifact_store,
            epoch_id=0,
            event_log=event_log,
            stats=stats,
        )
        trajectory = final_trajectory
        task_state = final_task
        tool_state = final_tool

        compact_result = StateCompactor(
            self.token_counter,
            config=StateCompactConfig(
                task_budget_tokens=self.config.compact_task_budget_tokens,
                tool_budget_tokens=self.config.compact_tool_budget_tokens,
            ),
        ).compact(task_state, tool_state, artifact_store=artifact_store)
        stats.compact_task_evicted = compact_result.evicted_task_items
        stats.compact_tool_evicted = compact_result.evicted_tool_items
        if compact_result.evicted_task_items or compact_result.evicted_tool_items:
            stats.notes.append(f"state compact evicted {compact_result.evicted_task_items} task / {compact_result.evicted_tool_items} tool items")

        stats.task_entries = _count_task_entries(task_state)
        stats.tool_entries = _count_tool_entries(tool_state)
        stats.groups_folded = folded
        stats.groups_kept = len(kept)
        _normalize_slots(tool_state)
        stats.post_fold_tokens = self._estimate_memory_tokens(task_state, tool_state, trajectory)
        if self.config.keep_work_dir:
            stats.work_dir = str(fold_dir)
        if not self.config.keep_work_dir:
            for child in fold_dir.iterdir():
                if child.is_dir():
                    import shutil

                    shutil.rmtree(child, ignore_errors=True)
                else:
                    try:
                        child.unlink()
                    except OSError:
                        pass
            try:
                fold_dir.rmdir()
            except OSError:
                pass
        return NovaCodeMemory(
            task_state=task_state,
            tool_state=tool_state,
            trajectory=trajectory,
            stats=stats.to_dict(),
        )

    # ------------------------------------------------------------------ fold loop

    def _fold_all(
        self,
        *,
        groups: list[InteractionGroup],
        task_state: TaskState,
        tool_state: ToolState,
        artifact_store: ArtifactStore,
        epoch_id: int,
        event_log: EventLog,
        stats: MemoryBuildStats,
    ) -> tuple[list[InteractionGroup], int, TaskState, ToolState, Trajectory]:
        remaining = list(groups)
        folded_total = 0
        current_epoch = epoch_id
        task_dict = task_state.to_dict()
        tool_dict = tool_state.to_dict()
        while len(remaining) > self.config.protected_recent_groups:
            # Fold the oldest groups, keeping the most recent ones intact.
            foldable = remaining[: -self.config.protected_recent_groups]
            if len(foldable) > _MAX_GROUPS:
                foldable = foldable[:_MAX_GROUPS]
            task_state = TaskState.from_dict(task_dict)
            tool_state = ToolState.from_dict(tool_dict)
            result = self._fold_batch(
                task_state=task_state,
                tool_state=tool_state,
                groups=foldable,
                epoch_id=current_epoch + 1,
                artifact_store=artifact_store,
                event_log=event_log,
                stats=stats,
            )
            task_dict = result["task_dict"]
            tool_dict = result["tool_dict"]
            stats.fold_events.append(self._fold_event(current_epoch + 1, len(foldable), result["fold_result"]))
            remaining = remaining[len(foldable) :]
            folded_total += len(foldable)
            current_epoch += 1
        task_state = TaskState.from_dict(task_dict)
        tool_state = ToolState.from_dict(tool_dict)
        trajectory = Trajectory(epoch_id=current_epoch)
        trajectory.groups = remaining
        return remaining, folded_total, task_state, tool_state, trajectory

    def _estimate_memory_tokens(self, task_state: TaskState, tool_state: ToolState, trajectory: Trajectory) -> int:
        """Estimated tokens of the post-fold memory (state + kept trajectory)."""
        task_text = json.dumps(task_state.to_dict(), ensure_ascii=False, separators=(",", ":"))
        tool_text = json.dumps(tool_state.to_dict(), ensure_ascii=False, separators=(",", ":"))
        recent_groups = sum(self.token_counter.estimate_messages(group.messages) for group in trajectory.groups)
        return self.token_counter.estimate_text(task_text) + self.token_counter.estimate_text(tool_text) + recent_groups

    @staticmethod
    def _fold_event(epoch: int, groups_folded: int, fold_result: FoldResult) -> dict[str, Any]:
        task_delta = fold_result.task_delta or {}
        tool_delta = fold_result.tool_delta or {}
        return {
            "epoch": epoch,
            "groups_folded": groups_folded,
            "model_used": fold_result.model_used,
            "fallback_used": fold_result.fallback_used,
            "calls": fold_result.calls,
            "retries": fold_result.retries,
            "last_error": (fold_result.last_error or "")[:200],
            "request_ids": list(fold_result.request_ids),
            "logical_input_tokens": fold_result.logical_input_tokens,
            "output_tokens": fold_result.output_tokens,
            "task_delta_entries": len(task_delta.get("set") or {}) + len(task_delta.get("append") or []),
            "tool_delta_entries": len(tool_delta.get("set") or {}) + len(tool_delta.get("append") or []),
        }

    def _fold_batch(
        self,
        *,
        task_state: TaskState,
        tool_state: ToolState,
        groups: list[InteractionGroup],
        epoch_id: int,
        artifact_store: ArtifactStore,
        event_log: EventLog,
        stats: MemoryBuildStats,
    ) -> dict[str, Any]:
        fold_result: FoldResult = self.fold_engine.fold(
            task_state=task_state,
            tool_state=tool_state,
            groups=groups,
            epoch_id=epoch_id,
            artifact_store=artifact_store,
            step=0,
            event_seq_anchor=event_log.last_seq,
            request_epoch_id=epoch_id - 1,
        )
        stats.fold_calls += 1
        if fold_result.model_used:
            stats.model_folds += 1
        if fold_result.fallback_used:
            stats.fallback_folds += 1
            if fold_result.last_error:
                stats.fold_errors += 1
                stats.notes.append(f"fold {epoch_id} fallback: {fold_result.last_error[:200]}")
        task_dict = task_state.to_dict()
        tool_dict = tool_state.to_dict()
        merge_task_delta(task_dict, fold_result.task_delta)
        merge_tool_delta(tool_dict, fold_result.tool_delta)
        return {"task_dict": task_dict, "tool_dict": tool_dict, "fold_result": fold_result}

    # ------------------------------------------------------------------ helpers

    def _work_dir(self) -> Path:
        if self.config.work_dir is not None:
            self.config.work_dir.mkdir(parents=True, exist_ok=True)
            return self.config.work_dir
        return Path(tempfile.mkdtemp(prefix="novacode-ama-build-"))

    @staticmethod
    def _batches(steps: list[Step], size: int) -> list[list[Step]]:
        return [steps[index : index + size] for index in range(0, len(steps), size)]

    def _batch_group(
        self,
        batch: list[Step],
        *,
        group_id: str,
        epoch_id: int,
    ) -> InteractionGroup:
        calls: list[ToolCall] = []
        group = InteractionGroup(
            group_id=group_id,
            epoch_id=epoch_id,
            created_step=batch[0].turn_idx,
            status="complete",
        )
        for step in batch:
            tool_call = self._step_tool_call(step)
            calls.append(tool_call)
        if calls:
            group.messages.append(
                Message(
                    role="assistant",
                    content=None,
                    tool_calls=calls,
                )
            )
        for step in batch:
            tool_call = self._step_tool_call(step)
            content = f"Step {step.turn_idx}: {step.action}\n\n{step.observation}"[:2_000]
            group.messages.append(
                Message(
                    role="tool",
                    content=content,
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    is_error=False,
                )
            )
        for message in group.messages:
            group.token_count += self.token_counter.estimate_message(message)
        return group

    def _step_tool_call(self, step: Step) -> ToolCall:
        parsed = _parse_tool_command(step.action)
        if parsed is None:
            return ToolCall(id=f"call-{step.turn_idx:04d}", name="observe", arguments={"step": step.turn_idx})
        name, args = parsed
        arguments: dict[str, Any] = {"raw": step.action}
        lowered = name.lower()
        if lowered in {"read_file", "grep_search", "write_file", "edit_file"}:
            path = _tool_argument_value(step.action)
            if path:
                arguments["path"] = path
        elif lowered in {"run_shell", "shell"}:
            arguments["command"] = args
        elif lowered in {"search", "list_files"}:
            arguments["pattern"] = args
        return ToolCall(id=f"call-{step.turn_idx:04d}", name=name, arguments=arguments)


def _count_task_entries(task_state: TaskState) -> int:
    return (
        len(task_state.completed)
        + len(task_state.key_findings)
        + len(task_state.decisions)
        + len(task_state.unresolved)
        + len(task_state.key_sequences)
    )


def _count_tool_entries(tool_state: ToolState) -> int:
    return sum(len(entries) for profile in tool_state.profiles.values() for entries in profile.values())


def _group_turn_range(group: InteractionGroup) -> str:
    """First-last turn indices of a group, or '' when no turn markers exist."""
    first = last = ""
    for message in group.messages:
        content = str(message.content or "")
        match = re.search(r"Step\s+(\d+)", content)
        if match:
            first = first or match.group(1)
            last = match.group(1)
    return f"{first}-{last}" if first else ""
