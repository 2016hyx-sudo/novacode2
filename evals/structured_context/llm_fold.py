"""LLM-fold evaluation mode: replay offline recipes through the real FoldEngine.

The deterministic ``offline`` subcommand simulates Trajectory Fold with rules
only: eligibility by window pressure plus estimated token counts, no state
deltas, no LLM.  This module runs the same recipes but at every fold point
adapts the synthetic state and groups into real ``TaskState`` / ``ToolState`` /
``InteractionGroup`` objects, calls the production :class:`FoldEngine`
(optionally backed by a live provider), and merges the returned task/tool
deltas back into the replay state, so later structured prompts reflect what
the fold actually produced.  Fold requests carry provider-reported usage.

Everything outside the fold calls stays deterministic and mirrors the pure
offline replay; this mode is never used to gate CI (no committed baseline).
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Sequence

from coding_agent.structured_context.fold_engine import FoldEngine
from coding_agent.structured_context.models import (
    InteractionGroup,
    ToolExperience,
    ToolState,
)
from coding_agent.structured_context.token_counter import TokenCounter

from .full_context import (
    DEFAULT_OFFLINE_CASES,
    ReplayError,
    ReplayRequest,
    RunResult,
    load_offline_cases,
    materialize_offline_case,
)
from .schema import RequestMetric

_EMPTY_PROGRESS = {"completed": [], "current": "", "remaining": []}


def _usage_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# state adapters
#
# The round trip must stay stable in both directions: the recipe dict only
# gains keys (LLM-written progress / evidence / extra tool slots), and
# `_compact_state` only ever inspects ids and statuses, so surplus keys are
# harmless.  Recipe entries carry id/status/fact — the model from_dict
# implementations default every other field.
# ---------------------------------------------------------------------------


def _recipe_task_dict(state: Mapping[str, Any], case_id: str) -> dict[str, Any]:
    """Adapt the recipe state dict into a TaskState-shaped plain dict."""
    return {
        "schema_version": "1.0",
        "task_id": str(state.get("task_id", "")),
        "objective": f"offline case {case_id} (deterministic replay)",
        "constraints": [],
        "success_criteria": [],
        "progress": dict(state.get("progress") or _EMPTY_PROGRESS),
        "key_findings": list(state.get("findings") or []),
        "decisions": list(state.get("decisions") or []),
        "unresolved": list(state.get("unresolved") or []),
        "key_sequences": list(state.get("key_sequences") or []),
        "extensions": {},
    }


def _is_empty_merge_value(key: str, value: Any) -> bool:
    if key == "progress":
        return dict(value) == _EMPTY_PROGRESS
    return not value


def _task_state_to_recipe(state: dict[str, Any], merged: dict[str, Any]) -> None:
    """Write a merged TaskState-shaped dict back into the recipe state dict."""
    state["findings"] = list(merged.get("key_findings") or [])
    state["decisions"] = list(merged.get("decisions") or [])
    for key in ("progress", "unresolved", "key_sequences"):
        value = merged.get(key)
        if value is not None and not _is_empty_merge_value(key, value):
            state[key] = copy.deepcopy(value)


def _recipe_tool_state(state: Mapping[str, Any]) -> ToolState:
    """Flatten recipe ``tool_entries`` into a ToolState by profile slot."""
    tool_state = ToolState.new()
    for entry in state.get("tool_entries") or []:
        if not isinstance(entry, Mapping):
            continue
        profile = str(entry.get("profile") or "read")
        kind = str(entry.get("kind") or "useful_files")
        slots = tool_state.profiles.setdefault(profile, {})
        slots.setdefault(kind, []).append(ToolExperience.from_dict(dict(entry)))
    return tool_state


def _tool_state_to_recipe(state: dict[str, Any], merged: dict[str, Any]) -> None:
    """Flatten a merged ToolState-shaped dict back into ``tool_entries``.

    The ``profile`` key disambiguates same-named slots across tools (shell and
    test both keep ``effective_commands``), which the recipe schema cannot
    express otherwise.
    """
    entries: list[dict[str, Any]] = []
    for profile, slots in (merged.get("profiles") or {}).items():
        for kind, items in slots.items():
            for item in items:
                entries.append({**dict(item), "profile": str(profile)})
    state["tool_entries"] = entries


def _groups_from_recipe(groups: Sequence[Mapping[str, Any]]) -> list[InteractionGroup]:
    """Adapt eligible recipe groups into real InteractionGroup objects.

    ``raw_tool_result_refs`` is a replay-only key stripped before
    ``InteractionGroup.from_dict``.
    """
    result: list[InteractionGroup] = []
    for group in groups:
        data = dict(group)
        data.pop("raw_tool_result_refs", None)
        result.append(InteractionGroup.from_dict(data))
    return result


# ---------------------------------------------------------------------------
# replay entry points
# ---------------------------------------------------------------------------


def materialize_llm_fold_cases(
    source: Path | str | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    *,
    fold_engine: Any,
    token_counter: TokenCounter | None = None,
    include_prompts: bool = False,
) -> list[ReplayRequest]:
    """Materialize every recipe with the real FoldEngine at each fold point."""
    counter = token_counter or TokenCounter()
    result: list[ReplayRequest] = []
    for case in load_offline_cases(source):
        result.extend(
            materialize_offline_case(
                case,
                token_counter=counter,
                include_prompts=include_prompts,
                fold_engine=fold_engine,
            )
        )
    return result


def run_llm_fold_replay(
    path: Path | str | Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None,
    *,
    fold_engine: Any,
    min_requests: int = 200,
    include_prompts: bool = False,
) -> RunResult:
    """Replay the offline suite, folding through the real FoldEngine.

    Unlike the pure offline replay, failures are isolated per case: a
    ReplayError inside one recipe is recorded in ``manifest["case_errors"]``
    and the remaining recipes still run.  Callers must treat
    ``failed_case_count > 0`` as a failed run — each failed case already cost
    fold calls.
    """
    if fold_engine is None:
        raise ReplayError(
            "offline-llm-fold requires a FoldEngine; the rule-only simulation "
            "stays in the offline subcommand"
        )
    counter = TokenCounter()
    replay: list[ReplayRequest] = []
    case_errors: list[dict[str, Any]] = []
    for case in load_offline_cases(path):
        try:
            replay.extend(
                materialize_offline_case(
                    case,
                    token_counter=counter,
                    include_prompts=include_prompts,
                    fold_engine=fold_engine,
                )
            )
        except ReplayError as exc:
            case_errors.append({"case_id": str(case.get("id", "")), "error": str(exc)})
    if len(replay) < min_requests:
        raise ReplayError(
            f"offline-llm-fold replay produced {len(replay)} requests; "
            f"expected at least {min_requests}"
        )
    main_requests = tuple(item.metric for item in replay)
    fold_requests = _fold_rows(replay, fold_engine)
    requests = (*main_requests, *fold_requests)
    model_calls = sum(_usage_int(row.metadata.get("calls")) for row in fold_requests)
    case_ids = sorted({item.task_case_id for item in main_requests})
    return RunResult(
        run_id="offline-llm-fold",
        mode="offline-llm-fold",
        requests=requests,
        tasks=tuple({"id": case_id, "status": "materialized"} for case_id in case_ids),
        manifest={
            "suite": "offline-core-15",
            "source": str(path or DEFAULT_OFFLINE_CASES),
            "network_allowed": True,
            "model_calls": model_calls,
            "llm_folds": sum(1 for row in fold_requests if row.metadata.get("model_used")),
            "fallback_folds": sum(1 for row in fold_requests if row.metadata.get("fallback_used")),
            "fold_mode": "llm",
            "main_request_count": len(main_requests),
            "fold_request_count": len(fold_requests),
            "request_count": len(requests),
            "failed_case_count": len(case_errors),
            "case_errors": case_errors,
        },
    )


def _fold_rows(replay: Sequence[ReplayRequest], fold_engine: Any) -> list[RequestMetric]:
    """Synthesize fold request rows from enriched fold events (deduplicated).

    Mirrors the pure offline row shape so metrics and reporting treat both
    modes identically; usage comes from the engine's real fold calls.  A
    fallback fold (no provider call) carries the estimated input tokens
    instead, with an honest ``estimated`` source.
    """
    rows: list[RequestMetric] = []
    seen: set[tuple[str, str]] = set()
    for item in replay:
        for event in item.metric.replay.get("fold_events") or []:
            fold_id = str(event.get("fold_id", ""))
            key = (item.metric.task_case_id, fold_id)
            if not fold_id or key in seen:
                continue
            seen.add(key)
            step = int(event.get("step", 0))
            model_used = bool(event.get("model_used"))
            logical = _usage_int(
                event.get("logical_input_tokens")
                if model_used
                else event.get("input_tokens_estimated")
            )
            rows.append(
                RequestMetric(
                    task_case_id=item.metric.task_case_id,
                    bucket=item.metric.bucket,
                    request_id=f"offline-{item.metric.task_case_id}-{fold_id}",
                    parent_request_id=f"offline-{item.metric.task_case_id}-{step:04d}",
                    session_id=item.metric.session_id,
                    agent_role="fold",
                    step=step,
                    epoch_id=int(event.get("epoch_id", 0)),
                    provider=str(getattr(fold_engine, "provider_name", "") or "fold"),
                    model=str(getattr(fold_engine, "model", "") or "unknown"),
                    variant="structured",
                    logical_input_tokens=logical,
                    cache_hit_tokens=_usage_int(event.get("cache_hit_tokens")),
                    fresh_processed_input_tokens=_usage_int(
                        event.get("fresh_processed_input_tokens")
                    ),
                    token_source="provider_reported" if model_used else "estimated",
                    metadata={
                        "fold_id": fold_id,
                        "folded_group_ids": list(event.get("folded_group_ids") or []),
                        "model_used": model_used,
                        "fallback_used": bool(event.get("fallback_used")),
                        "calls": _usage_int(event.get("calls")),
                        "retries": _usage_int(event.get("retries")),
                        "last_error": str(event.get("last_error", "") or ""),
                        "request_ids": list(event.get("request_ids") or []),
                        "input_tokens_estimated": _usage_int(
                            event.get("input_tokens_estimated")
                        ),
                        "output_tokens": _usage_int(event.get("output_tokens")),
                        "raw_usage": {
                            "output_tokens": _usage_int(event.get("output_tokens"))
                        },
                        "task_delta": dict(event.get("task_delta") or {}),
                        "tool_delta": dict(event.get("tool_delta") or {}),
                    },
                )
            )
    return rows


# ---------------------------------------------------------------------------
# environment wiring
# ---------------------------------------------------------------------------


def build_fold_engine_from_env(*, trace_dir: Path | None = None) -> FoldEngine:
    """Build the production FoldEngine from environment configuration.

    Requires a configured provider: with no usable API key this raises
    ValueError.  The CLI only calls this behind an explicit ``--provider-env``
    flag so an accidental invocation can never construct a live provider.
    """
    from config import LLMConfig
    from coding_agent.llm import create_provider

    config = LLMConfig.from_env()
    provider = create_provider(config)
    trace = None
    if trace_dir is not None:
        from coding_agent.runtime.trace import TraceWriter

        trace = TraceWriter(Path(trace_dir) / "traces")
    return FoldEngine(
        provider,
        trace=trace,
        provider_name=config.provider,
        model=config.model,
        reasoning_effort=config.secondary_reasoning_effort or "none",
    )


__all__ = [
    "_groups_from_recipe",
    "_recipe_task_dict",
    "_recipe_tool_state",
    "_task_state_to_recipe",
    "_tool_state_to_recipe",
    "build_fold_engine_from_env",
    "materialize_llm_fold_cases",
    "run_llm_fold_replay",
]
