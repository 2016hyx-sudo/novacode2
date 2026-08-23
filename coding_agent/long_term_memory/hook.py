"""Lifecycle extraction hooks for autonomous memory harvesting on fold / session end."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from .models import MemoryEntry, MemoryHeader, MemoryType
from .store import MemoryStore
from .suppression import SuppressionEngine

if TYPE_CHECKING:
    from ..structured_context.models import TaskState, ToolState


class MemoryLifecycleHook:
    """Extracts high-value architecture decisions and verified fixes into long-term memories."""

    def __init__(
        self,
        store: MemoryStore,
        suppression: SuppressionEngine | None = None,
    ) -> None:
        self.store = store
        self.suppression = suppression or SuppressionEngine(store=store)

    def extract_from_task_state(self, task_state: TaskState) -> list[MemoryEntry]:
        """Extract durable architectural decisions (project memories) from TaskState."""
        entries: list[MemoryEntry] = []
        if not hasattr(task_state, "decisions"):
            return entries

        for dec in getattr(task_state, "decisions", []):
            if getattr(dec, "status", "valid") != "valid":
                continue

            raw_decision = getattr(dec, "decision", "").strip()
            raw_reason = getattr(dec, "reason", "").strip()
            if not raw_decision:
                continue

            # Slugify name
            clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_decision[:30]).strip("_").lower()
            name = f"arch_{clean_name}" if clean_name else "arch_decision"

            header = MemoryHeader(
                name=name,
                type=MemoryType.PROJECT,
                description=raw_decision[:80],
            )
            content_lines = [
                f"### Architectural Decision: {raw_decision}",
                "",
                f"**Rationale:** {raw_reason or 'Documented via structured context task decisions.'}",
            ]
            evidence = getattr(dec, "evidence", [])
            if evidence:
                content_lines.append("")
                content_lines.append(f"**Evidence / Context:** {', '.join(str(e) for e in evidence)}")

            entry = MemoryEntry(header=header, content="\n".join(content_lines))
            entries.append(entry)

        return entries

    def extract_from_tool_state(self, tool_state: ToolState) -> list[MemoryEntry]:
        """Extract verified error fixes (feedback) and effective commands (reference) from ToolState."""
        entries: list[MemoryEntry] = []
        if not hasattr(tool_state, "profiles"):
            return entries

        for tool_name, profile in tool_state.profiles.items():
            # Extract known error patterns / known failures
            error_patterns = profile.get("known_error_patterns", []) or profile.get("known_failures", [])
            for exp in error_patterns:
                if getattr(exp, "status", "valid") != "valid":
                    continue
                val = getattr(exp, "value", {})
                err_text = val.get("error", "") or val.get("pattern", "") or str(val)
                fix_text = val.get("fix", "") or val.get("solution", "") or val.get("workaround", "")
                if not err_text or not fix_text:
                    continue

                slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", f"{tool_name}_{err_text[:20]}").strip("_").lower()
                name = f"fix_{slug}"

                header = MemoryHeader(
                    name=name,
                    type=MemoryType.FEEDBACK,
                    description=f"{tool_name} fix: {fix_text[:60]}"[:80],
                )
                content = (
                    f"### Tool Troubleshooting: {tool_name}\n\n"
                    f"**Error / Symptom:** {err_text}\n\n"
                    f"**Verified Fix (Why & How):** {fix_text}\n"
                )
                entries.append(MemoryEntry(header=header, content=content))

        return entries

    def harvest_and_save(
        self,
        task_state: TaskState | None = None,
        tool_state: ToolState | None = None,
    ) -> list[MemoryEntry]:
        """Extract candidates, run negative suppression checks, and persist valid memories."""
        candidates: list[MemoryEntry] = []
        if task_state is not None:
            candidates.extend(self.extract_from_task_state(task_state))
        if tool_state is not None:
            candidates.extend(self.extract_from_tool_state(tool_state))

        saved_entries: list[MemoryEntry] = []
        for entry in candidates:
            verdict = self.suppression.validate_write(entry, is_subagent=False, is_update=False)
            if verdict.passed:
                saved = self.store.save_entry(entry)
                saved_entries.append(saved)

        return saved_entries

    def on_fold(
        self,
        task_state: TaskState | None,
        tool_state: ToolState | None,
    ) -> list[MemoryEntry]:
        """Triggered during Trajectory Fold."""
        return self.harvest_and_save(task_state, tool_state)

    def on_session_end(
        self,
        task_state: TaskState | None,
        tool_state: ToolState | None,
    ) -> list[MemoryEntry]:
        """Triggered when session finishes."""
        return self.harvest_and_save(task_state, tool_state)
