"""Memory manipulation tools for AgentLoop."""
from __future__ import annotations

from typing import Any, Literal

from ..tools.base import Tool, ToolErrorKind, ToolResult
from .models import MemoryEntry, MemoryHeader, MemoryType, QuotaConfig
from .store import MemoryStore
from .suppression import SuppressionEngine


class SaveMemoryTool:
    """Tool to save a new durable memory entry."""

    name = "save_memory"
    description = (
        "Save a new durable memory entry (user preference, critical feedback/fix, "
        "project architectural decision, or reference). Strict suppression rules apply."
    )
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Unique identifier for this memory (alphanumeric, underscores/dashes, <=40 chars).",
            },
            "type": {
                "type": "string",
                "enum": ["user", "feedback", "project", "reference"],
                "description": "Semantic category of the memory.",
            },
            "description": {
                "type": "string",
                "description": "Concise one-line summary (<=80 chars) used for semantic recall indexing.",
            },
            "content": {
                "type": "string",
                "description": "Markdown body detailing the knowledge, context, code snippet, or rules.",
            },
            "scope": {
                "type": "string",
                "enum": ["project", "global", "auto"],
                "description": "Storage scope: 'project' (current workspace) or 'global' (cross-project user preferences).",
                "default": "auto",
            },
        },
        "required": ["name", "type", "description", "content"],
    }

    def __init__(
        self,
        store: MemoryStore,
        suppression: SuppressionEngine | None = None,
        is_subagent: bool = False,
    ) -> None:
        self.store = store
        self.suppression = suppression or SuppressionEngine(store=store)
        self.is_subagent = is_subagent

    def execute(
        self,
        name: str,
        type: str,
        description: str,
        content: str,
        scope: Literal["project", "global", "auto"] = "auto",
        **kwargs: Any,
    ) -> ToolResult:
        if self.is_subagent:
            return ToolResult.fail(
                "Subagents are forbidden from modifying persistent memories.",
                kind=ToolErrorKind.WORKSPACE_VIOLATION,
            )

        try:
            m_type = MemoryType(type.strip().lower())
        except ValueError:
            return ToolResult.fail(
                f"Invalid memory type {type!r}. Must be one of: {[t.value for t in MemoryType]}",
                kind=ToolErrorKind.INVALID_ARGUMENTS,
            )

        header = MemoryHeader(
            name=name.strip(),
            type=m_type,
            description=description.strip()[: self.store.quota.max_description_length],
        )
        entry = MemoryEntry(header=header, content=content.strip())

        verdict = self.suppression.validate_write(entry, is_subagent=self.is_subagent, is_update=False)
        if not verdict.passed:
            hint = f"Rule #{verdict.rule_id} triggered. " if verdict.rule_id else ""
            if verdict.suggested_action == "use_update_memory":
                hint += "Please use update_memory to modify the existing memory."
            return ToolResult.fail(
                f"Memory write suppressed: {verdict.reason}",
                kind=ToolErrorKind.INVALID_ARGUMENTS,
                hint=hint.strip() or None,
            )

        saved = self.store.save_entry(entry, scope=scope)
        return ToolResult.ok(
            f"Successfully saved memory '{saved.name}' (Category: {saved.type.value}, Scope: {scope}). "
            f"Summary: {saved.description}",
            name=saved.name,
            type=saved.type.value,
            file_path=saved.file_path,
        )


class UpdateMemoryTool:
    """Tool to update or append content to an existing memory entry."""

    name = "update_memory"
    description = "Update or append content to an existing persistent memory entry."
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Identifier of the existing memory to update.",
            },
            "patch_mode": {
                "type": "string",
                "enum": ["replace", "append"],
                "description": "'replace' completely rewrites body content; 'append' appends to existing body.",
            },
            "description": {
                "type": "string",
                "description": "Optional updated summary (<=80 chars).",
            },
            "content": {
                "type": "string",
                "description": "New content to replace or append.",
            },
        },
        "required": ["name", "patch_mode", "content"],
    }

    def __init__(
        self,
        store: MemoryStore,
        suppression: SuppressionEngine | None = None,
        is_subagent: bool = False,
    ) -> None:
        self.store = store
        self.suppression = suppression or SuppressionEngine(store=store)
        self.is_subagent = is_subagent

    def execute(
        self,
        name: str,
        patch_mode: str,
        content: str,
        description: str | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        if self.is_subagent:
            return ToolResult.fail(
                "Subagents are forbidden from modifying persistent memories.",
                kind=ToolErrorKind.WORKSPACE_VIOLATION,
            )

        existing = self.store.load_entry(name)
        if existing is None:
            return ToolResult.fail(
                f"Memory '{name}' not found. Use save_memory to create a new entry.",
                kind=ToolErrorKind.NOT_FOUND,
            )

        mode = patch_mode.strip().lower()
        if mode == "append":
            new_content = f"{existing.content}\n\n{content.strip()}"
        elif mode == "replace":
            new_content = content.strip()
        else:
            return ToolResult.fail(
                f"Invalid patch_mode {patch_mode!r}. Must be 'replace' or 'append'.",
                kind=ToolErrorKind.INVALID_ARGUMENTS,
            )

        if description is not None and description.strip():
            existing.header.description = description.strip()[: self.store.quota.max_description_length]

        existing.content = new_content

        verdict = self.suppression.validate_write(existing, is_subagent=self.is_subagent, is_update=True)
        if not verdict.passed:
            return ToolResult.fail(
                f"Memory update suppressed: {verdict.reason}",
                kind=ToolErrorKind.INVALID_ARGUMENTS,
            )

        updated = self.store.save_entry(existing)
        return ToolResult.ok(
            f"Successfully updated memory '{updated.name}' (mode: {mode}).",
            name=updated.name,
            type=updated.type.value,
        )


class DeleteMemoryTool:
    """Tool to delete an obsolete memory entry."""

    name = "delete_memory"
    description = "Delete an obsolete or inaccurate persistent memory entry."
    parameters = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Identifier of the memory to delete.",
            },
            "reason": {
                "type": "string",
                "description": "Explicit reason for deleting this memory.",
            },
        },
        "required": ["name", "reason"],
    }

    def __init__(
        self,
        store: MemoryStore,
        is_subagent: bool = False,
    ) -> None:
        self.store = store
        self.is_subagent = is_subagent

    def execute(self, name: str, reason: str, **kwargs: Any) -> ToolResult:
        if self.is_subagent:
            return ToolResult.fail(
                "Subagents are forbidden from modifying persistent memories.",
                kind=ToolErrorKind.WORKSPACE_VIOLATION,
            )

        deleted = self.store.delete_entry(name)
        if not deleted:
            return ToolResult.fail(
                f"Memory '{name}' not found.",
                kind=ToolErrorKind.NOT_FOUND,
            )

        return ToolResult.ok(
            f"Successfully deleted memory '{name}'. Reason: {reason}",
            name=name,
        )


def create_memory_tools(
    store: MemoryStore,
    suppression: SuppressionEngine | None = None,
    is_subagent: bool = False,
) -> list[Tool]:
    """Factory creating memory tools for the main agent."""
    if is_subagent:
        return []
    suppr = suppression or SuppressionEngine(store=store)
    return [
        SaveMemoryTool(store, suppr, is_subagent=False),
        UpdateMemoryTool(store, suppr, is_subagent=False),
        DeleteMemoryTool(store, is_subagent=False),
    ]
