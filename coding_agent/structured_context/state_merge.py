"""Deterministic ID-based state delta merging and deterministic fold extraction."""
from __future__ import annotations

from typing import Any

from .models import InteractionGroup, TaskState, ToolState


TASK_LIST_TARGETS = {
    "progress.completed": ("progress", "completed"),
    "progress.remaining": ("progress", "remaining"),
    "key_findings": ("key_findings",),
    "decisions": ("decisions",),
    "unresolved": ("unresolved",),
    "key_sequences": ("key_sequences",),
}

TOOL_LIST_TARGETS = {
    "profiles.grep.useful_scopes": ("grep", "useful_scopes"),
    "profiles.grep.effective_queries": ("grep", "effective_queries"),
    "profiles.grep.known_error_patterns": ("grep", "known_error_patterns"),
    "profiles.read.useful_files": ("read", "useful_files"),
    "profiles.read.effective_ranges": ("read", "effective_ranges"),
    "profiles.read.known_symbols": ("read", "known_symbols"),
    "profiles.shell.effective_commands": ("shell", "effective_commands"),
    "profiles.shell.known_failures": ("shell", "known_failures"),
    "profiles.test.effective_commands": ("test", "effective_commands"),
    "profiles.test.known_failures": ("test", "known_failures"),
}


class StateMergeError(ValueError):
    pass


def _get_list(state: dict[str, Any], target: str, table: dict[str, tuple[str, ...]]) -> list[dict[str, Any]]:
    keys = table[target]
    current: Any = state
    for key in keys[:-1]:
        current = current.setdefault(key, {})
    return current.setdefault(keys[-1], [])


def _validate_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise StateMergeError(f"Delta item must be an object, got {type(item).__name__}")
    return item


def merge_delta(state: dict[str, Any], delta: dict[str, Any], *, table: dict[str, tuple[str, ...]]) -> list[str]:
    """Apply a State Delta to a state dict in-place. Returns trace notes."""
    notes: list[str] = []
    set_values = delta.get("set") or {}
    if not isinstance(set_values, dict):
        raise StateMergeError("delta.set must be an object")
    for key, value in set_values.items():
        if key in {"progress.current"}:
            state.setdefault("progress", {})["current"] = value
            notes.append(f"set {key}")
        else:
            notes.append(f"ignored top-level set target {key}")

    for operation in ("remove", "mark_stale"):
        for entry in delta.get(operation) or []:
            entry = _validate_item(entry)
            target = str(entry.get("target", ""))
            item_id = str(entry.get("id", ""))
            if target not in table:
                raise StateMergeError(f"Unknown delta target {target!r}")
            items = _get_list(state, target, table)
            found = False
            for item in items:
                if str(item.get("id", "")) == item_id:
                    found = True
                    if operation == "remove":
                        items.remove(item)
                        notes.append(f"remove {target}:{item_id}")
                    else:
                        item["status"] = "stale"
                        item["stale_reason"] = str(entry.get("reason", "delta"))
                        notes.append(f"mark_stale {target}:{item_id}")
                    break
            if not found:
                notes.append(f"missing {target}:{item_id} for {operation}")

    for entry in delta.get("upsert") or []:
        entry = _validate_item(entry)
        target = str(entry.get("target", ""))
        if target not in table:
            raise StateMergeError(f"Unknown delta target {target!r}")
        item = _validate_item(entry.get("value"))
        item_id = str(item.get("id") or entry.get("id") or "")
        if not item_id:
            raise StateMergeError("upsert requires an id")
        item["id"] = item_id
        items = _get_list(state, target, table)
        replaced = False
        for index, old in enumerate(items):
            if str(old.get("id", "")) == item_id:
                items[index] = item
                replaced = True
                notes.append(f"upsert {target}:{item_id}")
                break
        if not replaced:
            items.append(item)
            notes.append(f"append {target}:{item_id}")

    for entry in delta.get("append") or []:
        entry = _validate_item(entry)
        target = str(entry.get("target", ""))
        if target not in table:
            raise StateMergeError(f"Unknown delta target {target!r}")
        item = _validate_item(entry.get("item"))
        dedupe_key = str(entry.get("dedupe_key") or "id")
        dedupe_value = item.get(dedupe_key)
        items = _get_list(state, target, table)
        duplicate = dedupe_value is not None and any(str(x.get(dedupe_key)) == str(dedupe_value) for x in items)
        if not duplicate:
            items.append(item)
            notes.append(f"append {target}:{item.get('id', '')}")
        else:
            notes.append(f"dedupe append {target}:{dedupe_value}")

    return notes


def merge_task_delta(task_state: dict[str, Any], delta: dict[str, Any]) -> list[str]:
    return merge_delta(task_state, delta, table=TASK_LIST_TARGETS)


def merge_tool_delta(tool_state: dict[str, Any], delta: dict[str, Any]) -> list[str]:
    return merge_delta(tool_state, delta, table=TOOL_LIST_TARGETS)


def _summarize_group(group: InteractionGroup) -> dict[str, Any]:
    tools: list[str] = []
    success_count = 0
    failure_count = 0
    output_summaries: list[str] = []
    for message in group.messages:
        if message.role == "assistant":
            for call in message.tool_calls:
                tools.append(call.name)
        if message.role == "tool":
            if message.is_error:
                failure_count += 1
            else:
                success_count += 1
            text = (message.content or "").strip()
            if text:
                output_summaries.append(text[:160])
    return {
        "group_id": group.group_id,
        "created_step": group.created_step,
        "tools": tools,
        "success_count": success_count,
        "failure_count": failure_count,
        "output_summaries": output_summaries,
    }


def deterministic_fold_delta(
    task_state: TaskState,
    tool_state: ToolState,
    groups: list[InteractionGroup],
    *,
    epoch_id: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Best-effort deterministic Task/Tool State delta for folded groups.

    This is intentionally conservative. It preserves durable facts that can be
    extracted without an LLM: completed tool steps, modified paths, successful
    commands and read/grep evidence pointers.
    """
    task_delta: dict[str, Any] = {"set": {}, "upsert": [], "append": [], "remove": [], "mark_stale": []}
    tool_delta: dict[str, Any] = {"set": {}, "upsert": [], "append": [], "remove": [], "mark_stale": []}

    completed = task_state.completed if task_state is not None else []
    completed_ids = {item.id for item in completed}
    command_ids = {item.id for profile in (tool_state.profiles.values() if tool_state else {}) for entries in profile.values() for item in entries}

    for index, group in enumerate(groups):
        summary = _summarize_group(group)
        if summary["success_count"]:
            text = "; ".join(summary["output_summaries"][:3]) or f"tool group {group.group_id} completed"
            progress_id = f"p-{epoch_id}-{group.group_id}"
            if progress_id not in completed_ids:
                task_delta["append"].append(
                    {
                        "target": "progress.completed",
                        "dedupe_key": "id",
                        "item": {
                            "id": progress_id,
                            "text": text[:240],
                            "completed_step": group.created_step,
                            "evidence_refs": [],
                        },
                    }
                )

        # Preserve recently modified workspace paths as task facts.
        for change in group.workspace_changes:
            fact_id = f"f-{epoch_id}-{index}-{change.get('path', 'x').replace('/', '-')}"
            task_delta["upsert"].append(
                {
                    "target": "key_findings",
                    "id": fact_id,
                    "value": {
                        "id": fact_id,
                        "fact": f"modified {change.get('path')}",
                        "evidence": [
                            {"type": "file", "path": change.get("path"), "file_hash": change.get("after_sha256")}
                        ],
                        "status": "valid",
                        "updated_step": group.created_step,
                    },
                }
            )

        for message in group.messages:
            if message.role != "assistant":
                continue
            for call in message.tool_calls:
                args = dict(call.arguments or {})
                if call.name in {"grep_search", "read_file", "run_shell", "search_files"}:
                    command = str(args.get("query") or args.get("path") or args.get("command") or "")
                    if not command:
                        continue
                    profile = "grep" if call.name == "grep_search" else ("read" if call.name == "read_file" else ("shell" if call.name == "run_shell" else "test"))
                    kind = (
                        "effective_queries"
                        if call.name == "grep_search"
                        else ("useful_files" if call.name == "read_file" else "effective_commands")
                    )
                    value = (
                        {"query": command, "scope": str(args.get("path", "."))}
                        if call.name == "grep_search"
                        else (
                            {"path": command, "file_hash": ""}
                            if call.name == "read_file"
                            else {"command": command, "purpose": ""}
                        )
                    )
                    item_id = f"{profile}-{epoch_id}-{len(command_ids)}-{command[:20]}"
                    if item_id not in command_ids:
                        command_ids.add(item_id)
                        tool_delta["append"].append(
                            {
                                "target": f"profiles.{profile}.{kind}",
                                "dedupe_key": "id",
                                "item": {
                                    "id": item_id,
                                    "kind": kind,
                                    "value": value,
                                    "status": "valid",
                                    "updated_step": group.created_step,
                                },
                            }
                        )
    return task_delta, tool_delta
