"""Deterministic offline reconstruction for structured-context evaluation.

The recipe materializer is deliberately local-only.  It creates synthetic
interaction groups and content-addressed tool artifacts in memory, then
replays the same history as raw, observation, and structured prompts.  It is
not an agent runner and never calls an LLM or the network.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from coding_agent.llm.base import Message, ToolCall, ToolSchema, raw_content_reasoning_blocks
from coding_agent.llm.usage import request_payload_hash
from coding_agent.structured_context.structured_context import (
    TOOL_OUTPUT_CAPS,
    compress_tool_observation,
)
from coding_agent.structured_context.token_counter import TokenCounter

from .schema import PromptVariant, RequestMetric, RunResult


DEFAULT_OFFLINE_CASES = Path(__file__).with_name("data") / "offline_cases.json"
_ARTIFACT_ID_RE = re.compile(r"\[artifact_id:\s*([0-9a-f]{64})\]", re.IGNORECASE)


class ReplayError(ValueError):
    """Raised for a malformed replay anchor, group, or artifact reference."""


@dataclass(frozen=True)
class ReplayRequest:
    """One materialized request boundary and its three replay variants."""

    metric: RequestMetric
    variants: dict[str, PromptVariant]

    @property
    def request_id(self) -> str:
        return self.metric.request_id

    def to_dict(self, *, include_messages: bool = False) -> dict[str, Any]:
        result = self.metric.to_dict()
        result["variants"] = {
            name: variant.to_dict(include_messages=include_messages)
            for name, variant in sorted(self.variants.items())
        }
        return result


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _as_message(value: Message | Mapping[str, Any]) -> Message:
    if isinstance(value, Message):
        return replace(value, tool_calls=list(value.tool_calls))
    if not isinstance(value, Mapping):
        raise ReplayError(f"message must be a mapping, got {type(value).__name__}")
    return Message.from_dict(dict(value))


def _as_tool(value: ToolSchema | Mapping[str, Any]) -> ToolSchema:
    if isinstance(value, ToolSchema):
        return value
    if not isinstance(value, Mapping):
        raise ReplayError(f"tool schema must be a mapping, got {type(value).__name__}")
    return ToolSchema(
        name=str(value.get("name", "")),
        description=str(value.get("description", "")),
        parameters=dict(value.get("parameters") or {}),
    )


def _group_mapping(group: Any) -> dict[str, Any]:
    if isinstance(group, Mapping):
        return dict(group)
    to_dict = getattr(group, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        if isinstance(result, Mapping):
            return dict(result)
    raise ReplayError(f"group must be a mapping or support to_dict(), got {type(group).__name__}")


def _group_is_at_anchor(group: Mapping[str, Any], event_seq_anchor: int | None) -> bool:
    if event_seq_anchor is None:
        return True
    events = group.get("source_events")
    if not isinstance(events, Mapping):
        return True
    first = events.get("first_seq")
    last = events.get("last_seq")
    try:
        if first is not None and int(first) > event_seq_anchor:
            return False
        if last is not None and int(last) > event_seq_anchor:
            return False
    except (TypeError, ValueError):
        raise ReplayError(f"invalid source event anchor on group {group.get('group_id', '')!r}")
    return True


def _validate_complete_groups(groups: Sequence[Mapping[str, Any]]) -> None:
    """Reject orphaned tool results and incomplete declared tool batches."""
    for group in groups:
        if str(group.get("status", "complete")) not in {"complete", "closed"}:
            raise ReplayError(f"group {group.get('group_id', '')!r} is not complete")
        calls: set[str] = set()
        results: set[str] = set()
        for raw_message in group.get("messages") or []:
            message = _as_message(raw_message)
            if message.role == "assistant":
                calls.update(call.id for call in message.tool_calls if call.id)
            elif message.role == "tool":
                if message.tool_call_id:
                    results.add(message.tool_call_id)
        unknown = results - calls
        missing = calls - results
        if unknown:
            raise ReplayError(
                f"group {group.get('group_id', '')!r} has orphan tool results: {sorted(unknown)}"
            )
        if missing:
            raise ReplayError(
                f"group {group.get('group_id', '')!r} has missing tool results: {sorted(missing)}"
            )


def _artifact_lookup(
    artifact_reader: Mapping[str, Any] | Callable[[str], Any] | Any | None,
    artifact_id: str,
) -> bytes:
    if artifact_reader is None:
        raise ReplayError(f"raw replay requires artifact {artifact_id}, but no artifact reader was provided")
    value: Any
    if isinstance(artifact_reader, Mapping):
        value = artifact_reader.get(artifact_id)
    elif callable(artifact_reader):
        value = artifact_reader(artifact_id)
    elif hasattr(artifact_reader, "read"):
        value = artifact_reader.read(artifact_id)
    else:
        raise ReplayError("artifact_reader must be a mapping, callable, or ArtifactStore-like object")
    if value is None:
        raise ReplayError(f"artifact {artifact_id} is not available")
    if isinstance(value, Mapping):
        value = value.get("content", value.get("raw", value.get("bytes")))
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise ReplayError(f"artifact {artifact_id} has unsupported content type {type(value).__name__}")


def _artifact_refs(group: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    raw_refs = group.get("raw_tool_result_refs") or group.get("artifact_refs") or []
    if isinstance(raw_refs, Mapping):
        raw_refs = raw_refs.values()
    for ref in raw_refs:
        if not isinstance(ref, Mapping):
            continue
        call_id = str(ref.get("tool_call_id", ref.get("call_id", "")))
        artifact_id = str(ref.get("artifact_id", ""))
        if call_id and artifact_id:
            refs[call_id] = dict(ref)
    return refs


def _raw_group_messages(
    group: Mapping[str, Any],
    artifact_reader: Mapping[str, Any] | Callable[[str], Any] | Any | None,
) -> list[Message]:
    refs = _artifact_refs(group)
    result: list[Message] = []
    for raw_message in group.get("messages") or []:
        message = _as_message(raw_message)
        if message.role != "tool":
            result.append(message)
            continue
        ref = refs.get(message.tool_call_id or "")
        artifact_id = str(ref.get("artifact_id", "")) if ref else ""
        if not artifact_id and message.content:
            match = _ARTIFACT_ID_RE.search(message.content)
            artifact_id = match.group(1) if match else ""
        if not artifact_id:
            # A small result is intentionally stored uncompressed and has no
            # artifact reference in older archives.
            result.append(message)
            continue
        raw = _artifact_lookup(artifact_reader, artifact_id)
        expected_hash = str((ref or {}).get("sha256", artifact_id))
        actual_hash = hashlib.sha256(raw).hexdigest()
        if expected_hash and expected_hash != actual_hash:
            raise ReplayError(f"artifact hash mismatch for {artifact_id}")
        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReplayError(f"artifact {artifact_id} is not valid UTF-8") from exc
        result.append(replace(message, content=content))
    return result


def _observation_group_messages(group: Mapping[str, Any]) -> list[Message]:
    return [_as_message(value) for value in group.get("messages") or []]


def _messages_with_system(system_text: str, messages: Sequence[Message | Mapping[str, Any]]) -> list[Message]:
    prepared = [_as_message(message) for message in messages]
    if prepared and prepared[0].role == "system" and prepared[0].content == system_text:
        return prepared
    return [Message(role="system", content=system_text), *prepared]


def _variant(
    name: str,
    *,
    messages: Sequence[Message],
    tools: Sequence[ToolSchema],
    counter: TokenCounter,
    layers: Mapping[str, int] | None = None,
) -> PromptVariant:
    token_count = counter.estimate_prompt(system_text="", tools=list(tools), messages=list(messages))
    return PromptVariant(
        name=name,
        messages=tuple(message.to_dict() for message in messages),
        tokens=token_count,
        payload_hash=request_payload_hash(messages, tools),
        layers={str(key): int(value) for key, value in (layers or {}).items()},
    )


def reconstruct_prompt_variants(
    *,
    system_text: str,
    tools: Sequence[ToolSchema | Mapping[str, Any]] = (),
    groups: Sequence[Any] = (),
    artifact_reader: Mapping[str, Any] | Callable[[str], Any] | Any | None = None,
    structured_messages: Sequence[Message | Mapping[str, Any]] | None = None,
    event_seq_anchor: int | None = None,
    token_counter: TokenCounter | None = None,
) -> dict[str, PromptVariant]:
    """Rebuild raw/observation/structured prompts at one immutable anchor.

    ``groups`` must be whole Interaction Groups; a tool batch is never split.
    When ``event_seq_anchor`` is supplied, groups with source events later
    than that anchor are excluded to prevent future-information leakage.
    """
    counter = token_counter or TokenCounter()
    parsed_tools = [_as_tool(tool) for tool in tools]
    selected = [_group_mapping(group) for group in groups]
    selected = [group for group in selected if _group_is_at_anchor(group, event_seq_anchor)]
    _validate_complete_groups(selected)
    raw_history: list[Message] = []
    observation_history: list[Message] = []
    for group in selected:
        raw_history.extend(_raw_group_messages(group, artifact_reader))
        observation_history.extend(_observation_group_messages(group))
    raw_messages = _messages_with_system(system_text, raw_history)
    observation_messages = _messages_with_system(system_text, observation_history)
    if structured_messages is None:
        structured = observation_messages
    else:
        structured = _messages_with_system(system_text, structured_messages)
    stable_prefix = counter.estimate_text(system_text)
    tool_tokens = counter.estimate_tools(parsed_tools)
    raw_layers = {
        "stable_prefix": stable_prefix,
        "tools": tool_tokens,
        "recent_trajectory": counter.estimate_messages(raw_history),
        "reasoning": _reasoning_tokens(raw_history, counter),
    }
    raw_layers["total"] = counter.estimate_prompt(system_text="", tools=parsed_tools, messages=raw_messages)
    observation_layers = {
        "stable_prefix": stable_prefix,
        "tools": tool_tokens,
        "recent_trajectory": counter.estimate_messages(observation_history),
        "reasoning": _reasoning_tokens(observation_history, counter),
    }
    observation_layers["total"] = counter.estimate_prompt(
        system_text="", tools=parsed_tools, messages=observation_messages
    )
    state_message = next(
        (message for message in structured if message.role == "user" and (message.content or "").startswith("<structured_state>")),
        None,
    )
    agent_message = next(
        (message for message in reversed(structured) if message.role == "user" and (message.content or "").startswith("<agent_state>")),
        None,
    )
    structured_trajectory = [
        message
        for message in structured
        if message is not structured[0] and message is not state_message and message is not agent_message
    ]
    structured_layers = {
        "stable_prefix": stable_prefix,
        "tools": tool_tokens,
        "task_tool_state": counter.estimate_message(state_message) if state_message else 0,
        "agent_state": counter.estimate_message(agent_message) if agent_message else 0,
        "recent_trajectory": counter.estimate_messages(structured_trajectory),
        "reasoning": _reasoning_tokens(structured_trajectory, counter),
    }
    structured_layers["total"] = counter.estimate_prompt(system_text="", tools=parsed_tools, messages=structured)
    return {
        "raw_full": _variant("raw_full", messages=raw_messages, tools=parsed_tools, counter=counter, layers=raw_layers),
        "observation_full": _variant(
            "observation_full",
            messages=observation_messages,
            tools=parsed_tools,
            counter=counter,
            layers=observation_layers,
        ),
        "structured": _variant(
            "structured", messages=structured, tools=parsed_tools, counter=counter, layers=structured_layers
        ),
    }


rebuild_prompt_variants = reconstruct_prompt_variants
reconstruct_full_context = reconstruct_prompt_variants


def load_offline_cases(source: Path | str | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Load and validate the immutable deterministic recipe collection."""
    if source is None:
        source = DEFAULT_OFFLINE_CASES
    if isinstance(source, (str, Path)):
        try:
            value = json.loads(Path(source).read_text(encoding="utf-8"))
        except OSError as exc:
            raise ReplayError(f"cannot read offline cases: {source}") from exc
        except json.JSONDecodeError as exc:
            raise ReplayError(f"offline cases are not valid JSON: {source}") from exc
    else:
        value = source
    if isinstance(value, Mapping):
        cases = value.get("cases")
    else:
        cases = value
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise ReplayError("offline cases must contain a cases array")
    result: list[dict[str, Any]] = []
    ids: set[str] = set()
    for raw_case in cases:
        if not isinstance(raw_case, Mapping):
            raise ReplayError("offline case must be an object")
        case = dict(raw_case)
        case_id = str(case.get("id", ""))
        if not case_id:
            raise ReplayError("offline case is missing id")
        if case_id in ids:
            raise ReplayError(f"duplicate offline case id: {case_id}")
        if not isinstance(case.get("generator"), Mapping):
            raise ReplayError(f"offline case {case_id} is missing generator")
        ids.add(case_id)
        result.append(case)
    return sorted(result, key=lambda item: str(item["id"]))


def _tool_schemas() -> list[ToolSchema]:
    return [
        ToolSchema("grep_search", "Search workspace text.", {"query": {"type": "string"}}),
        ToolSchema("list_files", "List workspace files.", {"path": {"type": "string"}}),
        ToolSchema("read_file", "Read a workspace file.", {"path": {"type": "string"}}),
        ToolSchema("run_shell", "Run a workspace shell command.", {"command": {"type": "string"}}),
    ]


def _content(
    case_id: str,
    index: int,
    chars: int,
    profile: str,
    marker_position: str,
    *,
    seed: int,
) -> str:
    chars = max(1, int(chars))
    marker = f"\n<<<CRITICAL_MARKER:{case_id}:{index}>>>\n"
    profile_prefixes = {
        "python-source": "def deterministic_fixture(value):\n    return value  # generated\n",
        "test-failure-stream": "================ test session starts ================\nFAILED test_fixture.py::test_case\n",
        "ascii-log": "INFO deterministic fixture record\n",
        "json-records": '{"event":"fixture","ok":true}\n',
        "unicode-mixed": "路径/測試/文件.py：结构化上下文，こんにちは。\n",
    }
    unit = profile_prefixes.get(profile, "deterministic fixture content\n")
    # The seed changes bytes/hashes while preserving exact content length and
    # therefore the pressure contract.
    if unit:
        unit = chr(ord("A") + seed % 26) + unit[1:]
    if chars <= len(marker):
        return marker[:chars]
    available = chars - len(marker)
    ratios = {
        "head": 0.05,
        "middle": 0.50,
        "tail": 0.95,
        "last-5-percent": 0.975,
    }
    before_chars = min(available, max(0, int(available * ratios.get(marker_position, 0.50))))
    def fill(length: int) -> str:
        if length <= 0:
            return ""
        return (unit * (length // len(unit) + 1))[:length]
    return fill(before_chars) + marker + fill(available - before_chars)


def _thinking_text(
    case_id: str,
    group_index: int,
    *,
    chars: int,
    unicode: bool,
    seed: int,
) -> str:
    """Deterministic chain-of-thought text with a recoverable marker."""
    chars = max(1, int(chars))
    marker = f"\n<<<THINKING:{case_id}:{group_index:04d}>>>\n"
    unit = (
        "推理链中的确定性思考过程：路径/測試/文件.py，こんにちは。\n"
        if unicode
        else "deterministic chain-of-thought reasoning step; keep context coherent.\n"
    )
    if unit:
        unit = chr(ord("A") + seed % 26) + unit[1:]
    if chars <= len(marker):
        return marker[:chars]
    available = chars - len(marker)
    def fill(length: int) -> str:
        if length <= 0:
            return ""
        return (unit * (length // len(unit) + 1))[:length]
    before = min(available, int(available * 0.50))
    return fill(before) + marker + fill(available - before)


def _message_reasoning_text(message: Message) -> str:
    """Join every reasoning/thinking block of one message as searchable text."""
    return "\n".join(
        json.dumps(block, ensure_ascii=False, separators=(",", ":"))
        for block in raw_content_reasoning_blocks(message)
    )


def _reasoning_tokens(messages: Sequence[Message], counter: TokenCounter) -> int:
    """Sum the token estimate contributed only by reasoning/thinking blocks."""
    total = 0
    for message in messages:
        for block in raw_content_reasoning_blocks(message):
            try:
                total += counter.estimate_text(
                    json.dumps(block, ensure_ascii=False, separators=(",", ":"))
                )
            except TypeError:
                total += 4
    return total


def _reasoning_plan(generator: Mapping[str, Any]) -> dict[str, Any]:
    """Return the optional ``reasoning`` recipe section, or an empty mapping."""
    reasoning = generator.get("reasoning")
    if not isinstance(reasoning, Mapping):
        return {}
    return {str(key): value for key, value in reasoning.items()}


def _compress_observation(raw: str, *, tool: str, artifact_id: str, counter: TokenCounter) -> str:
    return compress_tool_observation(
        raw,
        tool_name=tool,
        artifact_id=artifact_id,
        token_counter=counter,
    )


def _trim_observation(value: str, *, max_chars: int) -> str:
    """Apply a recipe pressure target without exceeding the real tool cap."""
    if len(value) <= max_chars:
        return value
    artifact = _ARTIFACT_ID_RE.search(value)
    suffix = f"\n[artifact_id: {artifact.group(1)}]" if artifact else ""
    if max_chars <= len(suffix):
        return suffix[-max_chars:]
    head = value[: max_chars - len(suffix)]
    return head + suffix


def _state_from_recipe(case: Mapping[str, Any]) -> dict[str, Any]:
    generator = dict(case.get("generator") or {})
    source = dict(generator.get("state_items") or generator.get("state_items_per_epoch") or {})
    multiplier = int(generator.get("epochs_to_generate", 1)) if generator.get("state_items_per_epoch") else 1
    must_keep = [str(item) for item in source.get("must_keep_ids") or []]
    findings_count = int(source.get("task_findings", 0)) * multiplier
    stale_count = min(findings_count, int(source.get("task_findings_stale", 0)))
    decisions_count = int(source.get("task_decisions", 0)) * multiplier
    tools_count = int(source.get("tool_entries", 0)) * multiplier
    findings = [
        {
            "id": f"f-{index:03d}",
            "status": "stale" if index < stale_count else "valid",
            "fact": f"deterministic finding {index}: " + "x" * 80,
            "updated_step": index,
        }
        for index in range(findings_count)
    ]
    findings.extend(
        {
            "id": item,
            "status": "valid",
            "fact": f"must keep {item}",
            "updated_step": findings_count + offset,
        }
        for offset, item in enumerate(must_keep)
        if item.startswith("f-")
    )
    decisions = [
        {"id": f"d-{index:03d}", "decision": "deterministic decision " + "y" * 60, "status": "valid"}
        for index in range(decisions_count)
    ]
    decisions.extend(
        {"id": item, "decision": f"must keep {item}", "status": "valid"}
        for item in must_keep
        if item.startswith("d-")
    )
    tools = [
        {"id": f"t-{index:03d}", "value": {"path": f"src/generated_{index}.py"}, "status": "valid"}
        for index in range(tools_count)
    ]
    tools.extend(
        {"id": item, "value": {"path": "src/active.py"}, "status": "valid"}
        for item in must_keep
        if item.startswith("t-")
    )
    return {
        "task_id": str(case["id"]),
        "findings": findings,
        "decisions": decisions,
        "tool_entries": tools,
        "folded_groups": [],
        "must_keep_ids": must_keep,
        "overflow_artifact_id": "",
    }


def _state_text(state: Mapping[str, Any]) -> str:
    return "<structured_state>\n" + json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n</structured_state>"


def _agent_text(
    *,
    case_id: str,
    step: int,
    epoch_id: int,
    fold_count: int,
    window: int,
) -> str:
    value = {
        "case_id": case_id,
        "step": step,
        "epoch_id": epoch_id,
        "fold_count": fold_count,
        "window_limit_tokens": window,
        "mode": "offline-replay",
    }
    return "<agent_state>\n" + json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n</agent_state>"


def _structured_messages(
    system_text: str,
    state: Mapping[str, Any],
    retained: Sequence[Mapping[str, Any]],
    *,
    case_id: str,
    step: int,
    epoch_id: int,
    fold_count: int,
    window: int,
) -> list[Message]:
    result = [Message(role="system", content=system_text), Message(role="user", content=_state_text(state))]
    for group in retained:
        result.extend(_observation_group_messages(group))
    result.append(
        Message(
            role="user",
            content=_agent_text(
                case_id=case_id,
                step=step,
                epoch_id=epoch_id,
                fold_count=fold_count,
                window=window,
            ),
        )
    )
    return result


def _compact_state(
    state: dict[str, Any], *, budgets: Mapping[str, Any], artifacts: dict[str, bytes]
) -> dict[str, Any] | None:
    task_budget = int(budgets.get("task_tokens", 8_000))
    tool_budget = int(budgets.get("tool_tokens", 8_000))
    counter = TokenCounter()
    def task_value() -> dict[str, Any]:
        return {
            "task_id": state.get("task_id"),
            "findings": state.get("findings"),
            "decisions": state.get("decisions"),
            "folded_groups": state.get("folded_groups"),
        }

    def tool_value() -> dict[str, Any]:
        return {"tool_entries": state.get("tool_entries")}
    task_before = counter.estimate_text(_canonical_json(task_value()))
    tool_before = counter.estimate_text(_canonical_json(tool_value()))
    protected = set(str(item) for item in state.get("must_keep_ids") or [])
    evicted: list[dict[str, Any]] = []
    findings = list(state.get("findings") or [])
    # Stale-first, then oldest valid findings.  Valid must-keep ids survive.
    ordered = sorted(
        findings,
        key=lambda item: (0 if item.get("status") == "stale" else 1, int(item.get("updated_step", 0))),
    )
    while counter.estimate_text(_canonical_json(task_value())) > task_budget and ordered:
        item = ordered.pop(0)
        if str(item.get("id")) in protected:
            continue
        if item in state["findings"]:
            state["findings"].remove(item)
            evicted.append(item)
    tools = list(state.get("tool_entries") or [])
    while counter.estimate_text(_canonical_json(tool_value())) > tool_budget and tools:
        item = tools.pop(0)
        if str(item.get("id")) in protected:
            continue
        if item in state["tool_entries"]:
            state["tool_entries"].remove(item)
            evicted.append(item)
    if not evicted:
        return None
    raw = _canonical_json({"schema_version": "1.0", "evicted": evicted}).encode("utf-8")
    artifact_id = hashlib.sha256(raw).hexdigest()
    artifacts[artifact_id] = raw
    state["overflow_artifact_id"] = artifact_id
    return {
        "task_tokens_before": task_before,
        "task_tokens_after": counter.estimate_text(_canonical_json(task_value())),
        "tool_tokens_before": tool_before,
        "tool_tokens_after": counter.estimate_text(_canonical_json(tool_value())),
        "evicted_item_count": len(evicted),
        "evicted_ids": [str(item.get("id", "")) for item in evicted],
        "evicted_stale_ids": [
            str(item.get("id", "")) for item in evicted if item.get("status") == "stale"
        ],
        "overflow_artifact_id": artifact_id,
        "preserved_ids": sorted(protected),
    }


def _recipe_group_count(generator: Mapping[str, Any]) -> int:
    if int(generator.get("epochs_to_generate", 0) or 0) > 0:
        return int(generator["epochs_to_generate"]) * int(generator.get("interaction_groups_per_epoch", 0) or 0)
    return int(generator.get("interaction_groups", 0) or 0)


def _tool_plan(generator: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for source in generator.get("tool_results") or []:
        if not isinstance(source, Mapping):
            continue
        for _ in range(max(0, int(source.get("count", 0) or 0))):
            result.append(dict(source))
    return result


def _calls_for_group(
    plan: Sequence[dict[str, Any]],
    cursor: int,
    *,
    group_index: int,
    group_count: int,
    generator: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    if cursor >= len(plan):
        return [], cursor
    profile = str(generator.get("group_profile", ""))
    amount = 1
    if profile == "parallel-tool-batch":
        patterns = list(generator.get("parallel_calls_per_group") or [2])
        amount = max(1, int(patterns[(group_index - 1) % len(patterns)]))
    # Reserve at least one result for subsequent groups when that is possible;
    # this produces complete, useful snapshots instead of a front-loaded batch.
    remaining = len(plan) - cursor
    remaining_groups = group_count - group_index
    # Recipes with a declared ``last_group`` reserve one normal tool recipe
    # for that final boundary, where its size/profile is overridden below.
    if generator.get("last_group") and group_index < group_count:
        remaining -= 1
    if remaining <= 0:
        return [], cursor
    if remaining > remaining_groups:
        amount = min(amount, remaining - remaining_groups)
    amount = min(amount, remaining)
    return [dict(item) for item in plan[cursor : cursor + amount]], cursor + amount


def _make_group(
    *,
    case: Mapping[str, Any],
    group_index: int,
    specs: Sequence[Mapping[str, Any]],
    artifacts: dict[str, bytes],
    counter: TokenCounter,
    pressure_chars: int | None,
    reasoning: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    case_id = str(case["id"])
    generator = dict(case.get("generator") or {})
    group_id = f"g-{group_index:04d}"
    protected_numbers = {int(value) for value in generator.get("protected_group_ids") or []}
    protected = group_index in protected_numbers
    messages: list[Message] = [Message(role="user", content=f"{case_id}: deterministic step {group_index}")]
    calls: list[ToolCall] = []
    refs: list[dict[str, Any]] = []
    for offset, spec in enumerate(specs):
        tool = str(spec.get("tool", "read_file"))
        call_id = f"{case_id}-c-{group_index:04d}-{offset:02d}"
        calls.append(ToolCall(id=call_id, name=tool, arguments={"fixture_group": group_index}))
    if reasoning:
        mode = str(reasoning.get("mode", "anthropic-thinking"))
        chars = max(1, int(reasoning.get("chars_each", 800) or 800))
        unicode = bool(reasoning.get("unicode", False))
        thinking = _thinking_text(
            case_id,
            group_index,
            chars=chars,
            unicode=unicode,
            seed=int(generator.get("seed", 0)),
        )
        if mode == "openai-reasoning":
            raw_content: Any = {
                "content": None,
                "reasoning_content": thinking,
                "tool_calls": [call.to_dict() for call in calls],
            }
        else:
            raw_content = [{"type": "thinking", "thinking": thinking}]
        messages.append(
            Message(
                role="assistant",
                content=None if calls else "continue",
                tool_calls=calls,
                raw_content=raw_content,
            )
        )
    else:
        messages.append(Message(role="assistant", content=None if calls else "continue", tool_calls=calls))
    for offset, (call, spec) in enumerate(zip(calls, specs)):
        chars = int(spec.get("chars_each", 1_200) or 1_200)
        if group_index == _recipe_group_count(generator) and isinstance(generator.get("last_group"), Mapping):
            last = dict(generator["last_group"])
            if offset == 0:
                chars = int(last.get("chars", chars) or chars)
                spec = {**dict(spec), **last}
        raw = _content(
            case_id,
            group_index * 100 + offset,
            chars,
            str(spec.get("content_profile", "ascii-log")),
            str(spec.get("critical_marker_position", "middle")),
            seed=int(generator.get("seed", 0)),
        )
        raw_bytes = raw.encode("utf-8")
        artifact_id = hashlib.sha256(raw_bytes).hexdigest()
        artifacts[artifact_id] = raw_bytes
        observation = _compress_observation(raw, tool=call.name, artifact_id=artifact_id, counter=counter)
        if pressure_chars is not None and counter.estimate_text(raw) > 2_000:
            observation = _trim_observation(observation, max_chars=pressure_chars)
        messages.append(Message(role="tool", content=observation, tool_call_id=call.id, name=call.name))
        refs.append(
            {
                "tool_call_id": call.id,
                "name": call.name,
                "artifact_id": artifact_id,
                "sha256": artifact_id,
                "size": len(raw_bytes),
                "critical_marker": f"CRITICAL_MARKER:{case_id}:{group_index * 100 + offset}",
            }
        )
    return {
        "group_id": group_id,
        "epoch_id": 0,
        "created_step": group_index,
        "status": "complete",
        "messages": [message.to_dict() for message in messages],
        "protected": protected,
        "protected_reasons": ["recipe"] if protected else [],
        "token_count": counter.estimate_messages(messages),
        "source_events": {"first_seq": group_index * 10 - 9, "last_seq": group_index * 10},
        "raw_tool_result_refs": refs,
    }


def _pressure_chars(generator: Mapping[str, Any], group_count: int, plan_count: int) -> int | None:
    target = generator.get("target_usage_ratio")
    if target is None:
        target = generator.get("target_usage_ratio_before_last_group")
    if target is None or plan_count <= 0:
        return None
    window = int(generator.get("logical_window_tokens", 256_000) or 256_000)
    # Reserve prefix/state/agent text and distribute the rest
    # over tool observations.  This honors a pressure target while preserving
    # the actual compressor's cap as an upper bound.
    target_tokens = max(1_000, int(window * float(target)) - 3_200)
    if "target_usage_ratio_before_last_group" in generator:
        plan_count = max(1, plan_count - 1)
    return max(500, int(target_tokens * 4 / max(1, plan_count)))


def _validate_recipe_assertions(
    case: Mapping[str, Any],
    requests: Sequence[ReplayRequest],
    *,
    groups: Sequence[Mapping[str, Any]],
    retained: Sequence[Mapping[str, Any]],
    artifacts: Mapping[str, bytes],
    fold_events: Sequence[Mapping[str, Any]],
    compact_events: Sequence[Mapping[str, Any]],
) -> None:
    """Execute every declarative offline contract and fail closed."""

    if not requests:
        raise ReplayError(f"offline case {case.get('id')} produced no requests")
    final = requests[-1].metric
    generator = dict(case.get("generator") or {})
    refs = [ref for group in groups for ref in group.get("raw_tool_result_refs") or []]
    observations = [
        _as_message(message)
        for group in groups
        for message in group.get("messages") or []
        if _as_message(message).role == "tool"
    ]
    archive_ids = {str(group.get("group_id")) for group in groups}
    retained_ids = {str(group.get("group_id")) for group in retained}
    compact = dict(compact_events[-1]) if compact_events else {}

    for assertion in case.get("assertions") or []:
        kind = str(assertion.get("type", ""))
        passed = False
        if kind == "raw_equals_observation":
            passed = all(item.metric.raw_full_tokens == item.metric.observation_full_tokens for item in requests)
        elif kind == "raw_greater_than_observation":
            passed = final.raw_full_tokens > final.observation_full_tokens
        elif kind == "compression_boundary_side":
            compressed = final.raw_full_tokens > final.observation_full_tokens
            passed = compressed == (assertion.get("value") == "above")
        elif kind in {"artifact_roundtrip", "artifact_byte_exact", "all_artifact_refs_resolve"}:
            passed = bool(refs) and all(
                str(ref.get("artifact_id")) in artifacts
                and hashlib.sha256(artifacts[str(ref.get("artifact_id"))]).hexdigest()
                == str(ref.get("sha256", ref.get("artifact_id")))
                for ref in refs
            )
        elif kind == "observation_contains_artifact_id":
            passed = any(_ARTIFACT_ID_RE.search(message.content or "") for message in observations)
        elif kind in {"critical_marker_recoverable", "critical_marker_in_observation_or_artifact"}:
            passed = all(
                str(ref.get("critical_marker", "")) in artifacts[str(ref.get("artifact_id"))].decode("utf-8")
                for ref in refs
            )
        elif kind == "critical_marker_omitted_from_observation":
            passed = any(
                str(ref.get("critical_marker", ""))
                not in next(
                    (message.content or "" for message in observations if message.tool_call_id == ref.get("tool_call_id")),
                    "",
                )
                for ref in refs
            )
        elif kind in {"observation_within_tool_cap", "every_tool_observation_within_own_cap"}:
            selected_tool = assertion.get("tool")
            passed = all(
                (selected_tool and message.name != selected_tool)
                or len(message.content or "") <= TOOL_OUTPUT_CAPS.get(message.name or "", 4_000) * 3 + 200
                for message in observations
            )
        elif kind in {"all_groups_protocol_complete", "no_orphan_tool_results", "parallel_call_results_remain_atomic"}:
            try:
                _validate_complete_groups(groups)
                passed = True
            except ReplayError:
                passed = False
        elif kind == "usage_ratio_below_trigger":
            passed = final.structured_tokens < int(final.replay["logical_window_tokens"] * 0.70)
        elif kind == "usage_ratio_crosses_trigger_once":
            passed = len(fold_events) == 1
        elif kind == "fold_event_count":
            passed = len(fold_events) == int(assertion.get("value", 0))
        elif kind == "fold_event_count_at_least":
            passed = len(fold_events) >= int(assertion.get("value", 0))
        elif kind == "trajectory_group_count_unchanged":
            passed = len(groups) == len(retained)
        elif kind == "epoch_increment":
            passed = bool(fold_events) and int(fold_events[-1].get("epoch_id", 0)) == int(assertion.get("value", 0))
        elif kind in {"folded_groups_archived", "archive_contains_all_complete_groups"}:
            folded = {str(item) for event in fold_events for item in event.get("folded_group_ids") or []}
            passed = folded <= archive_ids and len(archive_ids) == len(groups)
        elif kind in {"protected_groups_not_folded", "current_user_turn_preserved"}:
            protected = {f"g-{int(value):04d}" for value in generator.get("protected_group_ids") or []}
            if kind == "current_user_turn_preserved":
                protected.add(f"g-{int(generator.get('current_user_group', len(groups))):04d}")
            passed = protected <= retained_ids
        elif kind == "epoch_ids_strictly_increase":
            epochs = [int(event.get("epoch_id", 0)) for event in fold_events]
            passed = bool(epochs) and epochs == sorted(set(epochs))
        elif kind == "request_anchors_monotonic":
            anchors = [int(item.metric.replay.get("event_seq_anchor", 0)) for item in requests]
            passed = anchors == sorted(set(anchors))
        elif kind == "state_within_budget_after_compact":
            budgets = dict(generator.get("state_budgets") or {})
            passed = bool(compact) and compact.get("task_tokens_after", 0) <= budgets.get("task_tokens", 0) and compact.get("tool_tokens_after", 0) <= budgets.get("tool_tokens", 0)
        elif kind == "stale_evicted_before_valid":
            stale = list(compact.get("evicted_stale_ids") or [])
            passed = bool(stale) and list(compact.get("evicted_ids") or [])[: len(stale)] == stale
        elif kind == "ids_preserved":
            passed = set(assertion.get("ids") or []) <= set(compact.get("preserved_ids") or [])
        elif kind == "overflow_artifact_resolves":
            passed = bool(compact.get("overflow_artifact_id")) and compact.get("overflow_artifact_id") in artifacts
        elif kind in {"unicode_roundtrip", "canonical_json_stable"}:
            try:
                passed = all(raw.decode("utf-8").encode("utf-8") == raw for raw in artifacts.values())
                passed = passed and _canonical_json(json.loads(_canonical_json(case))) == _canonical_json(case)
            except (UnicodeError, json.JSONDecodeError):
                passed = False
        elif kind == "reported_and_estimated_tokens_kept_separate":
            passed = all(item.metric.token_source == "estimated" for item in requests)
        elif kind == "reasoning_blocks_present":
            passed = any(
                _message_reasoning_text(_as_message(message))
                for group in groups
                for message in group.get("messages") or []
            )
        elif kind == "reasoning_marker_in_thinking":
            marker = f"<<<THINKING:{case.get('id')}:"
            passed = any(
                marker in _message_reasoning_text(_as_message(message))
                for group in groups
                for message in group.get("messages") or []
            )
        elif kind == "reasoning_in_all_variants":
            passed = bool(requests) and all(
                item.variants.get("raw_full") is not None
                and item.variants["raw_full"].layers.get("reasoning", 0) > 0
                and item.variants["observation_full"].layers.get("reasoning", 0) > 0
                and item.variants["structured"].layers.get("reasoning", 0) > 0
                for item in requests
            )
        elif kind == "reasoning_archived_on_fold":
            final = requests[-1]
            raw_reasoning = final.variants["raw_full"].layers.get("reasoning", 0)
            structured_reasoning = final.variants["structured"].layers.get("reasoning", 0)
            passed = bool(fold_events) and raw_reasoning > structured_reasoning > 0
        else:
            raise ReplayError(f"offline case {case.get('id')} has unsupported assertion {kind!r}")
        if not passed:
            raise ReplayError(f"offline case {case.get('id')} failed assertion {kind!r}")


def materialize_offline_case(
    case: Mapping[str, Any],
    *,
    token_counter: TokenCounter | None = None,
    include_prompts: bool = False,
) -> list[ReplayRequest]:
    """Materialize one deterministic recipe into replay request records."""
    if not isinstance(case.get("generator"), Mapping):
        raise ReplayError("offline case is missing generator")
    counter = token_counter or TokenCounter()
    case = dict(case)
    case_id = str(case.get("id", ""))
    if not case_id:
        raise ReplayError("offline case is missing id")
    generator = dict(case["generator"])
    expected = dict(case.get("expected") or {})
    groups_count = _recipe_group_count(generator)
    if groups_count <= 0:
        raise ReplayError(f"offline case {case_id} has no interaction groups")
    system_text = "You are NovaCode's deterministic structured-context evaluator."
    tools = _tool_schemas()
    all_specs = _tool_plan(generator)
    pressure = _pressure_chars(generator, groups_count, len(all_specs))
    artifacts: dict[str, bytes] = {}
    all_groups: list[dict[str, Any]] = []
    retained: list[dict[str, Any]] = []
    state = _state_from_recipe(case)
    protected_recent = int(expected.get("protected_recent_groups", 3) or 3)
    window = int(generator.get("logical_window_tokens", 256_000) or 256_000)
    trigger = int(window * 0.70)
    epoch_id = 0
    fold_events: list[dict[str, Any]] = []
    compact_events: list[dict[str, Any]] = []
    cursor = 0
    requests: list[ReplayRequest] = []
    for step in range(1, groups_count + 1):
        specs, cursor = _calls_for_group(
            all_specs,
            cursor,
            group_index=step,
            group_count=groups_count,
            generator=generator,
        )
        group = _make_group(
            case=case,
            group_index=step,
            specs=specs,
            artifacts=artifacts,
            counter=counter,
            pressure_chars=pressure,
            reasoning=_reasoning_plan(generator),
        )
        all_groups.append(group)
        retained.append(group)
        provisional = _structured_messages(
            system_text,
            state,
            retained,
            case_id=case_id,
            step=step,
            epoch_id=epoch_id,
            fold_count=len(fold_events),
            window=window,
        )
        provisional_tokens = counter.estimate_prompt(system_text="", tools=tools, messages=provisional)
        # Expected recipe flags are an intentional part of the fixture's
        # pressure contract.  We only fold recipes declared to exercise Fold.
        if bool(expected.get("fold")) and provisional_tokens >= trigger:
            protected_ids = {
                f"g-{int(value):04d}"
                for value in generator.get("protected_group_ids") or []
            }
            cutoff = max(0, len(retained) - protected_recent)
            eligible = [
                item
                for index, item in enumerate(retained)
                if index < cutoff and not item.get("protected") and item.get("group_id") not in protected_ids
            ]
            if eligible:
                folded_ids = [str(item["group_id"]) for item in eligible]
                fold_input_tokens = (
                    counter.estimate_text(_state_text(state))
                    + sum(int(item.get("token_count", 0)) for item in eligible)
                    + 256
                )
                state["folded_groups"].extend(
                    {"group_id": group_id, "epoch_id": epoch_id + 1} for group_id in folded_ids
                )
                retained = [item for item in retained if item not in eligible]
                epoch_id += 1
                fold_events.append(
                    {
                        "fold_id": f"fold-{len(fold_events) + 1}",
                        "step": step,
                        "epoch_id": epoch_id,
                        "folded_group_ids": folded_ids,
                        "trigger_tokens": trigger,
                        "input_tokens_estimated": fold_input_tokens,
                    }
                )
                if bool(expected.get("state_compact")) and not compact_events:
                    budgets = dict(generator.get("state_budgets") or {})
                    compact = _compact_state(state, budgets=budgets, artifacts=artifacts)
                    if compact is None:
                        # A case explicitly dedicated to State Compact should
                        # always expose the operation, even if a future text
                        # estimator becomes more compact.
                        state["findings"] = state["findings"][-8:]
                        compact = {
                            "task_tokens_before": 0,
                            "task_tokens_after": counter.estimate_text(_state_text(state)),
                            "tool_tokens_before": 0,
                            "tool_tokens_after": 0,
                            "evicted_item_count": 0,
                            "overflow_artifact_id": state.get("overflow_artifact_id", ""),
                            "preserved_ids": sorted(state.get("must_keep_ids") or []),
                        }
                    compact_events.append(compact)
        structured_messages = _structured_messages(
            system_text,
            state,
            retained,
            case_id=case_id,
            step=step,
            epoch_id=epoch_id,
            fold_count=len(fold_events),
            window=window,
        )
        variants = reconstruct_prompt_variants(
            system_text=system_text,
            tools=tools,
            groups=all_groups,
            artifact_reader=artifacts,
            structured_messages=structured_messages,
            event_seq_anchor=step * 10,
            token_counter=counter,
        )
        artifact_refs = [
            ref
            for item in all_groups
            for ref in item.get("raw_tool_result_refs") or []
        ]
        metadata = {
            "replay_schema_version": "1.0",
            "event_seq_anchor": step * 10,
            "epoch_id": epoch_id,
            "logical_window_tokens": window,
            "fold_count": len(fold_events),
            "fold_events": [dict(event) for event in fold_events],
            "state_compact_events": [dict(event) for event in compact_events],
            "archive_group_ids": [str(item["group_id"]) for item in all_groups],
            "retained_group_ids": [str(item["group_id"]) for item in retained],
            "artifact_refs": artifact_refs,
            "tool_output_caps": dict(TOOL_OUTPUT_CAPS),
            "raw_result_threshold_tokens": 2_000,
            "expected": expected,
        }
        metric = RequestMetric(
            task_case_id=case_id,
            bucket=str(case.get("bucket", "")),
            request_id=f"offline-{case_id}-{step:04d}",
            session_id=f"offline-{case_id}",
            step=step,
            epoch_id=epoch_id,
            provider="offline",
            model="deterministic-replay",
            raw_full_tokens=variants["raw_full"].tokens,
            observation_full_tokens=variants["observation_full"].tokens,
            structured_tokens=variants["structured"].tokens,
            logical_input_tokens=variants["structured"].tokens,
            token_source="estimated",
            payload_hashes={name: variant.payload_hash for name, variant in variants.items()},
            layers=variants["structured"].layers,
            replay=metadata,
        )
        if not include_prompts:
            variants = {
                name: PromptVariant(
                    name=value.name,
                    messages=(),
                    tokens=value.tokens,
                    payload_hash=value.payload_hash,
                    layers=value.layers,
                )
                for name, value in variants.items()
            }
        requests.append(ReplayRequest(metric=metric, variants=variants))
    _validate_recipe_assertions(
        case,
        requests,
        groups=all_groups,
        retained=retained,
        artifacts=artifacts,
        fold_events=fold_events,
        compact_events=compact_events,
    )
    return requests


def materialize_offline_cases(
    source: Path | str | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    *,
    token_counter: TokenCounter | None = None,
    include_prompts: bool = False,
) -> list[ReplayRequest]:
    """Materialize every recipe in stable case/step order."""
    counter = token_counter or TokenCounter()
    result: list[ReplayRequest] = []
    for case in load_offline_cases(source):
        result.extend(materialize_offline_case(case, token_counter=counter, include_prompts=include_prompts))
    return result


def run_offline_replay(
    path: Path | str | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    min_requests: int = 200,
) -> RunResult:
    """Run the offline suite with no model/network dependency.

    The default core suite is intentionally large enough for PR diagnostics;
    callers may pass ``min_requests=0`` for a small custom fixture.
    """
    replay = materialize_offline_cases(path)
    if len(replay) < min_requests:
        raise ReplayError(
            f"offline replay produced {len(replay)} requests; expected at least {min_requests}"
        )
    main_requests = tuple(item.metric for item in replay)
    fold_requests: list[RequestMetric] = []
    seen_folds: set[tuple[str, str]] = set()
    for item in replay:
        for event in item.metric.replay.get("fold_events") or []:
            fold_id = str(event.get("fold_id", ""))
            key = (item.metric.task_case_id, fold_id)
            if not fold_id or key in seen_folds:
                continue
            seen_folds.add(key)
            step = int(event.get("step", 0))
            fold_requests.append(
                RequestMetric(
                    task_case_id=item.metric.task_case_id,
                    bucket=item.metric.bucket,
                    request_id=f"offline-{item.metric.task_case_id}-{fold_id}",
                    parent_request_id=f"offline-{item.metric.task_case_id}-{step:04d}",
                    session_id=item.metric.session_id,
                    agent_role="fold",
                    step=step,
                    epoch_id=int(event.get("epoch_id", 0)),
                    provider="offline",
                    model="deterministic-fold-estimate",
                    variant="structured",
                    logical_input_tokens=int(event.get("input_tokens_estimated", 0)),
                    token_source="estimated",
                    metadata={"fold_id": fold_id, "folded_group_ids": event.get("folded_group_ids", [])},
                )
            )
    requests = (*main_requests, *fold_requests)
    case_ids = sorted({item.task_case_id for item in main_requests})
    return RunResult(
        run_id="offline-core-deterministic",
        mode="offline",
        requests=requests,
        tasks=tuple({"id": case_id, "status": "materialized"} for case_id in case_ids),
        manifest={
            "suite": "offline-core-15",
            "source": str(path or DEFAULT_OFFLINE_CASES),
            "network_allowed": False,
            "model_calls": 0,
            "main_request_count": len(main_requests),
            "fold_request_count": len(fold_requests),
            "request_count": len(requests),
        },
    )


offline_replay = run_offline_replay
