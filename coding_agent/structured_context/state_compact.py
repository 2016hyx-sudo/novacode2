"""Deterministic Task State / Tool State capacity control.

The design baseline gives each persistent state a hard token budget.  This
module implements that budget with a conservative eviction order and an
optional artifact overflow so evicted entries are not lost from disk.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .artifact_store import ArtifactStore
from .models import TaskState, ToolState
from .token_counter import TokenCounter


@dataclass
class StateCompactConfig:
    task_budget_tokens: int = 8_000
    tool_budget_tokens: int = 8_000
    soft_ratio: float = 0.75
    max_overflow_items: int = 200

    def task_soft_limit(self) -> int:
        return int(self.task_budget_tokens * self.soft_ratio)

    def tool_soft_limit(self) -> int:
        return int(self.tool_budget_tokens * self.soft_ratio)


@dataclass
class StateCompactResult:
    task_tokens_before: int = 0
    task_tokens_after: int = 0
    tool_tokens_before: int = 0
    tool_tokens_after: int = 0
    evicted_task_items: list[dict[str, Any]] = field(default_factory=list)
    evicted_tool_items: list[dict[str, Any]] = field(default_factory=list)
    overflow_artifact_id: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_tokens_before": self.task_tokens_before,
            "task_tokens_after": self.task_tokens_after,
            "tool_tokens_before": self.tool_tokens_before,
            "tool_tokens_after": self.tool_tokens_after,
            "evicted_task_item_count": len(self.evicted_task_items),
            "evicted_tool_item_count": len(self.evicted_tool_items),
            "overflow_artifact_id": self.overflow_artifact_id,
            "notes": list(self.notes),
        }


class StateCompactor:
    """Evict low-value state entries until the configured token budgets hold.

    Eviction order is intentionally conservative:

    1. stale items first;
    2. oldest completed progress / findings / tool experiences;
    3. entries referenced by the active plan are protected;
    4. valid decisions are protected.
    """

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        *,
        config: StateCompactConfig | None = None,
    ) -> None:
        self.token_counter = token_counter or TokenCounter()
        self.config = config or StateCompactConfig()

    def compact(
        self,
        task_state: TaskState,
        tool_state: ToolState,
        *,
        artifact_store: ArtifactStore | None = None,
    ) -> StateCompactResult:
        task_dict = task_state.to_dict()
        tool_dict = tool_state.to_dict()
        result = StateCompactResult(
            task_tokens_before=self._tokens(task_dict),
            tool_tokens_before=self._tokens(tool_dict),
        )
        result.notes.append(
            f"task budget={self.config.task_budget_tokens} tool budget={self.config.tool_budget_tokens}"
        )

        protected_refs = self._protected_refs(task_dict)
        self._compact_task(task_dict, result, protected_refs)
        self._compact_tool(tool_dict, result)

        if (result.evicted_task_items or result.evicted_tool_items) and artifact_store is not None:
            result.overflow_artifact_id = self._spill_overflow(result, artifact_store)
            task_dict.setdefault("extensions", {})["state_compact_artifact_id"] = result.overflow_artifact_id
            tool_dict.setdefault("evidence_index", []).append(
                {"type": "state_compact_overflow", "artifact_id": result.overflow_artifact_id}
            )

        new_task_state = TaskState.from_dict(task_dict)
        new_tool_state = ToolState.from_dict(tool_dict)
        # Reflect the compaction back on the caller's mutable state.
        task_state.schema_version = new_task_state.schema_version
        task_state.task_id = new_task_state.task_id
        task_state.objective = new_task_state.objective
        task_state.constraints = new_task_state.constraints
        task_state.success_criteria = new_task_state.success_criteria
        task_state.completed = new_task_state.completed
        task_state.current = new_task_state.current
        task_state.remaining = new_task_state.remaining
        task_state.key_findings = new_task_state.key_findings
        task_state.decisions = new_task_state.decisions
        task_state.unresolved = new_task_state.unresolved
        task_state.key_sequences = new_task_state.key_sequences
        task_state.extensions = new_task_state.extensions
        tool_state.schema_version = new_tool_state.schema_version
        tool_state.profiles = new_tool_state.profiles
        tool_state.evidence_index = new_tool_state.evidence_index
        result.task_tokens_after = self._tokens(task_state.to_dict())
        result.tool_tokens_after = self._tokens(tool_state.to_dict())
        result.notes.append("compact complete")
        if len(result.notes) > 80:
            result.notes = result.notes[:80] + ["... (notes truncated)"]
        return result

    # ------------------------------------------------------------------ helpers

    def _tokens(self, value: dict[str, Any]) -> int:
        return self.token_counter.estimate_text(json.dumps(value, ensure_ascii=False))

    @staticmethod
    def _protected_refs(task_dict: dict[str, Any]) -> set[str]:
        refs: set[str] = set()
        progress = task_dict.get("progress") or {}
        remaining = progress.get("remaining") or []
        for step in remaining:
            if step.get("status") == "active":
                refs.update(str(ref) for ref in step.get("evidence_refs") or [])
        return refs

    def _compact_task(
        self,
        task_dict: dict[str, Any],
        result: StateCompactResult,
        protected_refs: set[str],
    ) -> None:
        budget = self.config.task_budget_tokens
        if self._tokens(task_dict) <= budget:
            return

        progress = task_dict.setdefault("progress", {})
        completed: list[dict[str, Any]] = progress.setdefault("completed", [])
        findings: list[dict[str, Any]] = task_dict.setdefault("key_findings", [])

        # Evict stale, unprotected findings first, then oldest completed progress.
        finding_candidates = sorted(
            [item for item in findings if item.get("status") == "stale" and item.get("id") not in protected_refs],
            key=lambda item: int(item.get("updated_step") or 0),
        )
        valid_finding_candidates = sorted(
            [item for item in findings if item.get("status") != "stale" and item.get("id") not in protected_refs],
            key=lambda item: int(item.get("updated_step") or 0),
        )
        progress_candidates = sorted(
            completed,
            key=lambda item: int(item.get("completed_step") or 0),
        )

        for item in [*finding_candidates, *valid_finding_candidates, *progress_candidates]:
            if self._tokens(task_dict) <= budget:
                break
            for collection in (findings, completed):
                if item in collection:
                    collection.remove(item)
                    result.evicted_task_items.append(dict(item))
                    result.notes.append(f"evict task item {item.get('id', '')}")
                    break

    def _compact_tool(self, tool_dict: dict[str, Any], result: StateCompactResult) -> None:
        budget = self.config.tool_budget_tokens
        if self._tokens(tool_dict) <= budget:
            return

        profiles = tool_dict.get("profiles") or {}
        all_items: list[tuple[str, str, dict[str, Any]]] = []
        for tool, entries_by_kind in profiles.items():
            for kind, entries in entries_by_kind.items():
                for entry in entries:
                    all_items.append((tool, kind, entry))

        stale = sorted(
            [item for item in all_items if item[2].get("status") == "stale"],
            key=lambda item: int(item[2].get("updated_step") or 0),
        )
        valid = sorted(
            [item for item in all_items if item[2].get("status") != "stale"],
            key=lambda item: int(item[2].get("updated_step") or 0),
        )

        for tool, kind, entry in [*stale, *valid]:
            if self._tokens(tool_dict) <= budget:
                break
            collection = profiles.get(tool, {}).get(kind)
            if collection and entry in collection:
                collection.remove(entry)
                result.evicted_tool_items.append({"tool": tool, "kind": kind, **dict(entry)})
                result.notes.append(f"evict tool item {tool}.{kind}:{entry.get('id', '')}")

        # Remove now-empty profile buckets to keep the serialized state small.
        for tool, entries_by_kind in list(profiles.items()):
            for kind, entries in list(entries_by_kind.items()):
                if not entries:
                    entries_by_kind.pop(kind, None)
            if not entries_by_kind:
                profiles.pop(tool, None)

    def _spill_overflow(self, result: StateCompactResult, artifact_store: ArtifactStore) -> str:
        payload = {
            "schema_version": "1.0",
            "task_items": result.evicted_task_items[: self.config.max_overflow_items],
            "tool_items": result.evicted_tool_items[: self.config.max_overflow_items],
        }
        entry = artifact_store.save(
            "state_compact",
            json.dumps(payload, ensure_ascii=False, indent=2),
            arguments={"kind": "state_compact_overflow"},
        )
        result.notes.append(f"spilled overflow to artifact {entry['artifact_id']}")
        return entry["artifact_id"]
