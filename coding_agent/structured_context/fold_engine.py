"""LLM-assisted Trajectory Fold with a deterministic fallback.

The model is asked to turn a set of completed Interaction Groups into two
State Deltas.  Its output is parsed, validated against the same target tables
used by :mod:`state_merge`, and only then merged.  Any failure degrades to the
existing deterministic extractor.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..llm.base import LLMProvider, Message
from .models import InteractionGroup, TaskState, ToolState
from .state_merge import TASK_LIST_TARGETS, TOOL_LIST_TARGETS, deterministic_fold_delta
from .token_counter import TokenCounter

_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")


class FoldError(RuntimeError):
    pass


class FoldDeltaValidationError(FoldError):
    pass


@dataclass
class FoldResult:
    task_delta: dict[str, Any] = field(default_factory=dict)
    tool_delta: dict[str, Any] = field(default_factory=dict)
    model_used: bool = False
    calls: int = 0
    retries: int = 0
    fallback_used: bool = False
    last_error: str = ""
    notes: list[str] = field(default_factory=list)

    def model_stats(self) -> dict[str, Any]:
        return {
            "used": self.model_used,
            "calls": self.calls,
            "retries": self.retries,
            "fallback_used": self.fallback_used,
            "last_error": self.last_error,
        }


@dataclass
class FoldEngineConfig:
    max_attempts: int = 2
    max_input_chars: int = 60_000
    max_assistant_chars: int = 2_000
    max_tool_chars: int = 1_000
    max_output_chars: int = 24_000
    max_groups_per_request: int = 30


class FoldEngine:
    """Generate Task/Tool State Deltas for folded trajectory groups.

    ``provider`` is the same LLM used by the main agent, per the design
    baseline.  A provider may be omitted (or may fail) in which case
    :func:`deterministic_fold_delta` is used directly.
    """

    def __init__(
        self,
        provider: LLMProvider | None,
        *,
        token_counter: TokenCounter | None = None,
        config: FoldEngineConfig | None = None,
    ) -> None:
        self.provider = provider
        self.token_counter = token_counter or TokenCounter()
        self.config = config or FoldEngineConfig()

    def fold(
        self,
        *,
        task_state: TaskState,
        tool_state: ToolState,
        groups: list[InteractionGroup],
        epoch_id: int,
        artifact_store: Any | None = None,
    ) -> FoldResult:
        if self.provider is None:
            return self._fallback(task_state, tool_state, groups, epoch_id, "no fold provider configured")

        last_error = ""
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                text = self._request_delta(task_state, tool_state, groups)
                task_delta, tool_delta = self._parse_delta(text)
                self._validate_delta(task_delta, task_state.to_dict(), TASK_LIST_TARGETS, "task")
                self._validate_delta(tool_delta, tool_state.to_dict(), TOOL_LIST_TARGETS, "tool")
                self._validate_artifact_refs(task_delta, tool_delta, artifact_store)
                return FoldResult(
                    task_delta=task_delta,
                    tool_delta=tool_delta,
                    model_used=True,
                    calls=attempt,
                    retries=attempt - 1,
                    fallback_used=False,
                    notes=[f"llm fold succeeded on attempt {attempt}"],
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.config.max_attempts:
                    break

        return self._fallback(
            task_state,
            tool_state,
            groups,
            epoch_id,
            last_error or "llm fold failed",
            calls=self.config.max_attempts,
            retries=self.config.max_attempts - 1,
        )

    # ------------------------------------------------------------------ model call

    def _request_delta(
        self,
        task_state: TaskState,
        tool_state: ToolState,
        groups: list[InteractionGroup],
    ) -> str:
        system = (
            "You are the fold compressor of a coding agent's structured context. "
            "Read the current Task State, Tool State and the completed Interaction "
            "Groups that are about to be removed from the prompt. Produce a compact, "
            "faithful JSON object with two fields: \"task_delta\" and \"tool_delta\".\n"
            "Allowed operations: set, upsert, append, remove, mark_stale.\n"
            "Rules:\n"
            "- Preserve durable task facts (what was changed/decided/verified), not chat history.\n"
            "- Preserve reusable tool experience (commands, queries, useful files, known failures).\n"
            "- For task delta: \"set\" may only contain {\"progress.current\": string}.\n"
            "- remove and mark_stale entries must reference an existing id in the current state.\n"
            "- Every upsert value must have an \"id\".\n"
            "- Do not invent file contents. Refer to artifact_id values when evidence is already an artifact.\n"
            "- Return only one JSON object. No markdown, no explanation."
        )
        model_groups = [self._group_for_model(group) for group in groups]
        base = {
            "instruction": (
                "Fold the following completed Interaction Groups into the current states. "
                "Return {\"task_delta\": {...}, \"tool_delta\": {...}}."
            ),
            "allowed_task_targets": sorted(TASK_LIST_TARGETS),
            "allowed_tool_targets": sorted(TOOL_LIST_TARGETS),
            "task_state": task_state.to_dict(),
            "tool_state": tool_state.to_dict(),
        }
        selected = model_groups[: self.config.max_groups_per_request]
        user = ""
        while selected:
            user = json.dumps({**base, "groups": selected}, ensure_ascii=False)
            if len(user) <= self.config.max_input_chars:
                break
            selected.pop()
        if not selected:
            raise FoldError("fold input exceeds configured size budget")
        messages = [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]
        response = self.provider.chat(messages, tools=[])
        text = (response.text or "").strip()
        if not text:
            raise FoldError("fold model returned empty text")
        return text[: self.config.max_output_chars]

    def _group_for_model(self, group: InteractionGroup) -> dict[str, Any]:
        data = group.to_dict()
        for message in data.get("messages") or []:
            content = message.get("content")
            if isinstance(content, str):
                if message.get("role") == "assistant":
                    message["content"] = content[: self.config.max_assistant_chars]
                elif message.get("role") == "tool":
                    message["content"] = content[: self.config.max_tool_chars]
            calls = message.get("tool_calls") or []
            for call in calls:
                args = call.get("arguments") or {}
                try:
                    text = json.dumps(args, ensure_ascii=False)
                    call["arguments"] = json.loads(text[:1_000]) if text[:1_000].strip() else {}
                except (TypeError, ValueError, json.JSONDecodeError):
                    call["arguments"] = {}
        data["protected"] = False
        data["protected_reasons"] = []
        return data

    # ------------------------------------------------------------------ parsing / validation

    @staticmethod
    def _parse_delta(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
        cleaned = _FENCE_RE.sub("", text).strip()
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        candidate = cleaned[start : end + 1] if start >= 0 and end > start else cleaned
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise FoldDeltaValidationError(f"fold model returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict):
            raise FoldDeltaValidationError("fold output must be a JSON object")
        if "task_delta" not in data or "tool_delta" not in data:
            # Accept a single delta shape only when it is unambiguous.
            if {"task_state", "tool_state"}.issubset(data):
                data = {"task_delta": data["task_state"], "tool_delta": data["tool_state"]}
            else:
                raise FoldDeltaValidationError("fold output must contain task_delta and tool_delta")
        task_delta = data.get("task_delta") or {}
        tool_delta = data.get("tool_delta") or {}
        if not isinstance(task_delta, dict) or not isinstance(tool_delta, dict):
            raise FoldDeltaValidationError("task_delta and tool_delta must be objects")
        return task_delta, tool_delta

    @staticmethod
    def _validate_delta(
        delta: dict[str, Any],
        current_state: dict[str, Any],
        table: dict[str, tuple[str, ...]],
        label: str,
    ) -> None:
        allowed_keys = {"set", "upsert", "append", "remove", "mark_stale"}
        unknown = set(delta) - allowed_keys
        if unknown:
            raise FoldDeltaValidationError(f"{label} delta has unsupported keys: {sorted(unknown)}")

        set_values = delta.get("set") or {}
        if not isinstance(set_values, dict):
            raise FoldDeltaValidationError(f"{label} delta.set must be an object")
        if label == "task":
            for key in set_values:
                if key != "progress.current":
                    raise FoldDeltaValidationError(f"unsupported {label} set target {key!r}")
        elif set_values:
            raise FoldDeltaValidationError(f"{label} delta does not support set operations")

        for operation in ("remove", "mark_stale"):
            for raw_entry in delta.get(operation) or []:
                if not isinstance(raw_entry, dict):
                    raise FoldDeltaValidationError(f"{label} {operation} entry must be an object")
                target = str(raw_entry.get("target", ""))
                item_id = str(raw_entry.get("id", ""))
                if target not in table:
                    raise FoldDeltaValidationError(f"unknown {label} delta target {target!r}")
                items = _lookup_list(current_state, target, table)
                if not any(str(item.get("id", "")) == item_id for item in items):
                    raise FoldDeltaValidationError(f"{label} {operation} references missing id {item_id!r}")

        for operation in ("upsert", "append"):
            for raw_entry in delta.get(operation) or []:
                if not isinstance(raw_entry, dict):
                    raise FoldDeltaValidationError(f"{label} {operation} entry must be an object")
                target = str(raw_entry.get("target", ""))
                if target not in table:
                    raise FoldDeltaValidationError(f"unknown {label} delta target {target!r}")
                value = raw_entry.get("value" if operation == "upsert" else "item")
                if not isinstance(value, dict):
                    raise FoldDeltaValidationError(f"{label} {operation} entry must contain an object")
                item_id = str(value.get("id") or raw_entry.get("id") or "")
                if operation == "upsert" and not item_id:
                    raise FoldDeltaValidationError(f"{label} upsert requires an id")

    @staticmethod
    def _validate_artifact_refs(
        task_delta: dict[str, Any],
        tool_delta: dict[str, Any],
        artifact_store: Any | None,
    ) -> None:
        if artifact_store is None or not hasattr(artifact_store, "find"):
            return
        for value in _iter_objects(task_delta):
            for evidence in value.get("evidence") or []:
                if isinstance(evidence, dict) and evidence.get("artifact_id"):
                    if artifact_store.find(str(evidence["artifact_id"])) is None:
                        raise FoldDeltaValidationError(
                            f"artifact reference does not exist: {evidence['artifact_id']}"
                        )

    # ------------------------------------------------------------------ fallback

    def _fallback(
        self,
        task_state: TaskState,
        tool_state: ToolState,
        groups: list[InteractionGroup],
        epoch_id: int,
        reason: str,
        *,
        calls: int = 0,
        retries: int = 0,
    ) -> FoldResult:
        task_delta, tool_delta = deterministic_fold_delta(
            task_state, tool_state, groups, epoch_id=epoch_id
        )
        return FoldResult(
            task_delta=task_delta,
            tool_delta=tool_delta,
            model_used=False,
            calls=calls,
            retries=retries,
            fallback_used=True,
            last_error=reason,
            notes=["deterministic fallback used"],
        )


def _lookup_list(
    state: dict[str, Any],
    target: str,
    table: dict[str, tuple[str, ...]],
) -> list[dict[str, Any]]:
    keys = table[target]
    current: Any = state
    for key in keys[:-1]:
        current = current.get(key) if isinstance(current, dict) else None
        if current is None:
            return []
    return current.get(keys[-1], []) if isinstance(current, dict) else []


def _iter_objects(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _iter_objects(child)
