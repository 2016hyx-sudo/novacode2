"""Pure metric and CI-gate functions for structured-context evaluation.

No function in this module reads files, calls a model, or depends on wall
clock time.  Keeping the arithmetic here makes offline replay suitable for a
strict, deterministic PR gate.
"""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .schema import GateResult, RequestMetric


Record = RequestMetric | Mapping[str, Any]


def _mapping(record: Record) -> Mapping[str, Any]:
    return record if isinstance(record, Mapping) else record.to_dict()


def _get(record: Record, *names: str, default: Any = None) -> Any:
    if isinstance(record, RequestMetric):
        for name in names:
            if hasattr(record, name):
                return getattr(record, name)
    data = _mapping(record)
    tokens = data.get("tokens")
    for name in names:
        if name in data:
            return data[name]
        if isinstance(tokens, Mapping):
            short_name = name.removesuffix("_tokens").removesuffix("_input")
            if short_name in tokens:
                return tokens[short_name]
    return default


def _number(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _tokens(record: Record, field: str) -> int:
    aliases = {
        "raw_full_tokens": ("raw_full_tokens", "raw_full_input_tokens"),
        "observation_full_tokens": (
            "observation_full_tokens",
            "observation_full_input_tokens",
        ),
        "structured_tokens": ("structured_tokens", "structured_input_tokens"),
        "logical_input_tokens": ("logical_input_tokens",),
        "fold_input_tokens": ("fold_input_tokens",),
    }
    return max(0, int(_number(_get(record, *aliases.get(field, (field,)), default=0))))


def _field_name(field: str) -> str:
    return {
        "raw_full": "raw_full_tokens",
        "observation_full": "observation_full_tokens",
        "structured": "structured_tokens",
    }.get(field, field)


def _is_main_success(record: Record) -> bool:
    role = str(_get(record, "agent_role", default="main") or "main")
    status = str(_get(record, "status", default="success") or "success")
    return role == "main" and status == "success"


def main_successful(records: Iterable[Record]) -> list[Record]:
    """Return the population used by primary offline/main-agent metrics."""
    return [record for record in records if _is_main_success(record)]


def nearest_rank(values: Iterable[int | float], percentile: float) -> int | float | None:
    """Compute a nearest-rank percentile, returning ``None`` for no samples.

    ``percentile`` accepts either a fraction (``0.95``) or a percent
    (``95``).  The formula is ``sorted(values)[ceil(q * N) - 1]``.
    """
    items = sorted(values)
    if not items:
        return None
    q = float(percentile)
    if 1 < q <= 100:
        q /= 100
    if not 0 < q <= 1:
        raise ValueError("percentile must be in (0, 1] or (0, 100]")
    index = max(0, math.ceil(q * len(items)) - 1)
    return items[index]


percentile_nearest_rank = nearest_rank


def distribution(values: Iterable[int | float]) -> dict[str, int | float | None]:
    """Return the p50/p95/max distribution without silently inventing zeros."""
    items = [value for value in values]
    return {
        "p50": nearest_rank(items, 0.50),
        "p95": nearest_rank(items, 0.95),
        "max": max(items) if items else None,
        "count": len(items),
    }


def weighted_reduction(
    records: Iterable[Record],
    *,
    baseline_field: str = "raw_full_tokens",
    candidate_field: str = "structured_tokens",
) -> float | None:
    """Calculate ``1 - sum(candidate) / sum(baseline)`` for valid baselines.

    Zero-baseline records have no reduction denominator and are intentionally
    excluded rather than treated as a zero-percent reduction.  A candidate
    larger than baseline is valid and yields a negative result.
    """
    baseline_total = 0.0
    candidate_total = 0.0
    for record in records:
        baseline = _tokens(record, _field_name(baseline_field))
        if baseline <= 0:
            continue
        baseline_total += baseline
        candidate_total += _tokens(record, _field_name(candidate_field))
    if baseline_total <= 0:
        return None
    return 1.0 - candidate_total / baseline_total


def weighted_crr(records: Iterable[Record]) -> float | None:
    return weighted_reduction(records)


def tool_reduction(records: Iterable[Record]) -> float | None:
    return weighted_reduction(records, candidate_field="observation_full_tokens")


def fold_incremental_reduction(records: Iterable[Record]) -> float | None:
    return weighted_reduction(
        records,
        baseline_field="observation_full_tokens",
        candidate_field="structured_tokens",
    )


def _task_key(record: Record) -> str:
    return str(_get(record, "task_case_id", "case_id", default="") or "")


def task_median_crr(records: Iterable[Record]) -> float | None:
    """Compute a weighted CRR per task, then take nearest-rank task p50."""
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        if _tokens(record, "raw_full_tokens") <= 0:
            continue
        grouped[_task_key(record)].append(record)
    values = [weighted_crr(group) for _, group in sorted(grouped.items())]
    return nearest_rank([value for value in values if value is not None], 0.50)


def request_crr(record: Record) -> float | None:
    baseline = _tokens(record, "raw_full_tokens")
    if baseline <= 0:
        return None
    return 1.0 - _tokens(record, "structured_tokens") / baseline


def negative_reduction_rate(records: Iterable[Record]) -> float | None:
    """Share of denominator-valid requests where structured exceeds raw."""
    count = 0
    negative = 0
    for record in records:
        baseline = _tokens(record, "raw_full_tokens")
        if baseline <= 0:
            continue
        count += 1
        if _tokens(record, "structured_tokens") > baseline:
            negative += 1
    return negative / count if count else None


def _fold_overhead(records: Sequence[Record]) -> float:
    explicit = sum(_tokens(record, "fold_input_tokens") for record in records if _is_main_success(record))
    # A separate fold request is normally how Phase 0 traces encode this
    # overhead.  Explicit per-main values and a separate fold row should not
    # coexist, but use the larger value if old data did record both.
    separate = sum(
        _tokens(record, "logical_input_tokens") or _tokens(record, "structured_tokens")
        for record in records
        if str(_get(record, "agent_role", default="main") or "main") == "fold"
    )
    return max(explicit, separate)


def net_crr(records: Iterable[Record], *, fold_input_tokens: int | float | None = None) -> float | None:
    """CRR after adding Fold-model input tokens to structured main input."""
    all_records = list(records)
    main = main_successful(all_records)
    baseline = sum(_tokens(record, "raw_full_tokens") for record in main if _tokens(record, "raw_full_tokens") > 0)
    if baseline <= 0:
        return None
    structured = sum(_tokens(record, "structured_tokens") for record in main if _tokens(record, "raw_full_tokens") > 0)
    overhead = _fold_overhead(all_records) if fold_input_tokens is None else max(0.0, _number(fold_input_tokens))
    return 1.0 - (structured + overhead) / baseline


def _distribution_for_records(records: Sequence[Record], field: str) -> dict[str, int | float | None]:
    return distribution([_tokens(record, field) for record in records])


def _variant_records(records: Sequence[Record], variant: str) -> list[Record]:
    return [
        record
        for record in records
        if _is_main_success(record)
        and str(_get(record, "variant", default="structured") or "structured") == variant
        and str(_get(record, "token_source", default="")) == "provider_reported"
    ]


def _logical_values(records: Sequence[Record]) -> list[int]:
    return [
        _tokens(record, "logical_input_tokens") or _tokens(record, "structured_tokens")
        for record in records
    ]


def _per_task_totals(records: Sequence[Record]) -> dict[str, int]:
    totals: dict[str, int] = defaultdict(int)
    for record, value in zip(records, _logical_values(records), strict=True):
        totals[_task_key(record)] += value
    return dict(totals)


def _paired_variant_reduction(
    baseline: Sequence[Record], candidate: Sequence[Record]
) -> float | None:
    if {_task_key(record) for record in baseline} != {_task_key(record) for record in candidate}:
        return None
    baseline_total = sum(_logical_values(baseline))
    if baseline_total <= 0 or not candidate:
        return None
    return 1.0 - sum(_logical_values(candidate)) / baseline_total


def _summarize_full_run(records: Sequence[Record]) -> dict[str, Any]:
    raw = _variant_records(records, "raw_full")
    observation = _variant_records(records, "observation_full")
    structured = _variant_records(records, "structured")
    raw_tasks = _per_task_totals(raw)
    structured_tasks = _per_task_totals(structured)
    paired_tasks = sorted(set(raw_tasks) & set(structured_tasks))
    pairing_valid = bool(raw_tasks) and set(raw_tasks) == set(structured_tasks)
    task_crr = [
        1.0 - structured_tasks[task] / raw_tasks[task]
        for task in paired_tasks
        if raw_tasks[task] > 0
    ]
    fold_input = sum(
        _tokens(record, "logical_input_tokens")
        for record in records
        if str(_get(record, "variant", default="")) == "structured"
        and str(_get(record, "agent_role", default="main")) == "fold"
    )
    raw_total = sum(_logical_values(raw))
    structured_total = sum(_logical_values(structured))
    net = (
        1.0 - (structured_total + fold_input) / raw_total
        if pairing_valid and raw_total > 0 and structured
        else None
    )
    all_main = main_successful(records)
    reported_main = [
        record
        for record in all_main
        if str(_get(record, "token_source", default="")) == "provider_reported"
    ]
    estimated_main = [record for record in all_main if record not in reported_main]
    failed_main = [
        record
        for record in records
        if str(_get(record, "agent_role", default="main") or "main") == "main"
        and str(_get(record, "status", default="success") or "success") != "success"
    ]
    return {
        "mode": "full-run",
        "sample": {
            "tasks": len({_task_key(record) for record in all_main if _task_key(record)}),
            "main_requests": len(all_main),
            "failed_requests": len(failed_main),
            "estimated_usage_count": sum(
                1
                for record in all_main
                if str(_get(record, "token_source", default="estimated")) != "provider_reported"
            ),
            "fold_requests": sum(
                1 for record in records if str(_get(record, "agent_role", default="main")) == "fold"
            ),
            "paired_tasks": len(paired_tasks),
            "pairing_valid": pairing_valid,
        },
        "context": {
            "crr_weighted": _paired_variant_reduction(raw, structured),
            "net_crr": net,
            "tool_reduction": _paired_variant_reduction(raw, observation),
            "fold_incremental": _paired_variant_reduction(observation, structured),
            "crr_task_median": nearest_rank(task_crr, 0.50) if pairing_valid else None,
            "negative_reduction_rate": (
                sum(value < 0 for value in task_crr) / len(task_crr)
                if pairing_valid and task_crr
                else None
            ),
        },
        "input_tokens": {
            "structured": distribution(_logical_values(structured)),
            "raw_full": distribution(_logical_values(raw)),
            "observation_full": distribution(_logical_values(observation)),
            "logical": distribution(_logical_values(reported_main)),
            "estimated_logical": distribution(_logical_values(estimated_main)),
            "per_task_cumulative": {
                "structured": distribution(structured_tasks.values()),
                "raw_full": distribution(raw_tasks.values()),
                "observation_full": distribution(_per_task_totals(observation).values()),
            },
        },
        "fold": {"input_tokens": fold_input, "request_count": sum(
            1 for record in records if str(_get(record, "agent_role", default="main")) == "fold"
        )},
        "usage": aggregate_usage(records),
    }


def aggregate_usage(records: Iterable[Record]) -> dict[str, Any]:
    """Aggregate provider-reported cache/usage fields across records.

    Cache hit rate uses the same ``cache_hit_tokens / logical_input_tokens``
    semantics as ``normalize_usage`` (``coding_agent.llm.usage``).  Output
    tokens are read from the record when exposed; otherwise they fall back to
    the raw provider usage stored in ``metadata.raw_usage``.
    """

    request_count = 0
    logical = 0
    cache_hit = 0
    fresh = 0
    output = 0
    for record in records:
        request_count += 1
        logical += int(_number(_get(record, "logical_input_tokens", default=0)))
        cache_hit += int(_number(_get(record, "cache_hit_tokens", default=0)))
        fresh += int(_number(_get(record, "fresh_processed_input_tokens", default=0)))
        record_output = int(_number(_get(record, "output_tokens", default=0)))
        if not record_output:
            raw_usage = _get(record, "metadata", default={})
            if isinstance(raw_usage, Mapping):
                raw_usage = raw_usage.get("raw_usage") or {}
            if isinstance(raw_usage, Mapping):
                record_output = int(
                    _number(raw_usage.get("output_tokens") or raw_usage.get("completion_tokens"))
                )
        output += record_output
    return {
        "request_count": request_count,
        "logical_input_tokens": logical,
        "cache_hit_tokens": cache_hit,
        "fresh_processed_input_tokens": fresh,
        "output_tokens": output,
        "cache_hit_rate": cache_hit / logical if logical else None,
    }


def summarize_metrics(records: Iterable[Record], *, mode: str = "offline") -> dict[str, Any]:
    """Create the stable, JSON-serializable summary consumed by ``report``."""
    all_records = list(records)
    if mode == "full-run":
        return _summarize_full_run(all_records)
    main = main_successful(all_records)
    folds = [
        record
        for record in all_records
        if str(_get(record, "agent_role", default="main") or "main") == "fold"
    ]
    tasks = {key for key in (_task_key(record) for record in main) if key}
    failed_main = [
        record
        for record in all_records
        if str(_get(record, "agent_role", default="main") or "main") == "main"
        and str(_get(record, "status", default="success") or "success") != "success"
    ]
    return {
        "mode": mode,
        "sample": {
            "tasks": len(tasks),
            "main_requests": len(main),
            "failed_requests": len(failed_main),
            "estimated_usage_count": sum(
                1 for record in main if str(_get(record, "token_source", default="estimated")) == "estimated"
            ),
            "fold_requests": len(folds),
        },
        "context": {
            "crr_weighted": weighted_crr(main),
            "net_crr": net_crr(all_records),
            "tool_reduction": tool_reduction(main),
            "fold_incremental": fold_incremental_reduction(main),
            "crr_task_median": task_median_crr(main),
            "negative_reduction_rate": negative_reduction_rate(main),
        },
        "input_tokens": {
            "structured": _distribution_for_records(main, "structured_tokens"),
            "raw_full": _distribution_for_records(main, "raw_full_tokens"),
            "observation_full": _distribution_for_records(main, "observation_full_tokens"),
            "logical": distribution(
                [
                    _tokens(record, "logical_input_tokens") or _tokens(record, "structured_tokens")
                    for record in main
                ]
            ),
        },
        "fold": {
            "input_tokens": _fold_overhead(all_records),
            "request_count": len(folds),
        },
        "usage": aggregate_usage(all_records),
    }


def _baseline_key(record: Record) -> tuple[str, int, int, int, str, str]:
    case = _task_key(record)
    step = int(_number(_get(record, "step", default=0)))
    attempt = int(_number(_get(record, "attempt", default=1)))
    repetition = int(_number(_get(record, "repetition", default=0)))
    role = str(_get(record, "agent_role", default="main") or "main")
    variant = str(_get(record, "variant", default="structured") or "structured")
    # Request ids are intentionally not part of the primary key: live retry
    # ids may be opaque while the recorded anchor remains the same.
    request = str(_get(record, "request_id", default="") or "") if not case else ""
    return (case, repetition, step, attempt, role, variant or request)


def compare_request_metrics(
    current: Iterable[Record],
    baseline: Iterable[Record],
    *,
    fields: Sequence[str] = (
        "raw_full_tokens",
        "observation_full_tokens",
        "structured_tokens",
        "logical_input_tokens",
    ),
    absolute_tolerance: int | float = 128,
    relative_tolerance: float = 0.02,
) -> GateResult:
    """Strictly compare deterministic replay rows by anchor identity.

    A growth fails when it exceeds ``max(absolute_tolerance,
    baseline * relative_tolerance)``. Missing or unexpected rows fail closed,
    which makes an approved baseline update an explicit review action.
    """
    if absolute_tolerance < 0 or relative_tolerance < 0:
        raise ValueError("gate tolerances must be non-negative")
    failures: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []

    def index_rows(rows: Iterable[Record], side: str) -> dict[tuple[str, int, int, int, str, str], Record]:
        indexed: dict[tuple[str, int, int, int, str, str], Record] = {}
        for record in rows:
            key = _baseline_key(record)
            if key in indexed:
                failures.append({"type": "duplicate_request_identity", "side": side, "key": list(key)})
                continue
            indexed[key] = record
        return indexed

    current_rows = index_rows(current, "current")
    baseline_rows = index_rows(baseline, "baseline")
    for key in sorted(set(current_rows) | set(baseline_rows)):
        before = baseline_rows.get(key)
        after = current_rows.get(key)
        if before is None:
            failures.append({"type": "unexpected_request", "key": list(key)})
            continue
        if after is None:
            failures.append({"type": "missing_request", "key": list(key)})
            continue
        for field in fields:
            normalized = _field_name(field)
            baseline_value = _tokens(before, normalized)
            current_value = _tokens(after, normalized)
            allowed_growth = max(float(absolute_tolerance), baseline_value * float(relative_tolerance))
            growth = current_value - baseline_value
            check = {
                "type": "request_token_growth",
                "key": list(key),
                "field": normalized,
                "baseline": baseline_value,
                "current": current_value,
                "growth": growth,
                "allowed_growth": allowed_growth,
                "passed": growth <= allowed_growth,
            }
            checks.append(check)
            if not check["passed"]:
                failures.append(dict(check))
    return GateResult(
        passed=not failures,
        failures=tuple(failures),
        checks=tuple(checks),
    )


def _path(value: Mapping[str, Any], *parts: str) -> Any:
    current: Any = value
    for part in parts:
        if not isinstance(current, Mapping):
            return None
        current = current.get(part)
    return current


def compare_summaries(
    current: Mapping[str, Any],
    baseline: Mapping[str, Any],
    *,
    crr_drop_limit: float = 0.03,
    structured_p50_regression_limit: float = 0.05,
    structured_p95_regression_limit: float = 0.08,
    require_non_negative_net_crr: bool = True,
) -> GateResult:
    """Compare aggregate gates without I/O; useful for nightly baselines."""
    checks: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def add_check(name: str, passed: bool, **details: Any) -> None:
        check = {"type": name, "passed": passed, **details}
        checks.append(check)
        if not passed:
            failures.append(dict(check))

    required = {
        "context.crr_weighted": ("context", "crr_weighted"),
        "context.net_crr": ("context", "net_crr"),
        "input_tokens.structured.p50": ("input_tokens", "structured", "p50"),
        "input_tokens.structured.p95": ("input_tokens", "structured", "p95"),
    }
    for name, path in required.items():
        for side, summary in (("current", current), ("baseline", baseline)):
            value = _path(summary, *path)
            if value is None:
                add_check(
                    "required_metric_present",
                    False,
                    side=side,
                    metric=name,
                )

    current_crr = _path(current, "context", "crr_weighted")
    baseline_crr = _path(baseline, "context", "crr_weighted")
    if current_crr is not None and baseline_crr is not None:
        add_check(
            "crr_weighted_drop",
            _number(current_crr) >= _number(baseline_crr) - crr_drop_limit,
            baseline=baseline_crr,
            current=current_crr,
            limit=crr_drop_limit,
        )

    for percentile, limit in (("p50", structured_p50_regression_limit), ("p95", structured_p95_regression_limit)):
        current_value = _path(current, "input_tokens", "structured", percentile)
        baseline_value = _path(baseline, "input_tokens", "structured", percentile)
        if current_value is not None and baseline_value is not None:
            allowed = _number(baseline_value) * (1 + limit)
            add_check(
                f"structured_{percentile}_regression",
                _number(current_value) <= allowed,
                baseline=baseline_value,
                current=current_value,
                limit=limit,
                allowed=allowed,
            )

    if require_non_negative_net_crr:
        current_net = _path(current, "context", "net_crr")
        if current_net is not None:
            add_check("net_crr_non_negative", _number(current_net) >= 0, current=current_net)
    if not checks:
        add_check("non_empty_gate", False)
    return GateResult(not failures, tuple(failures), tuple(checks))


def _records_from(value: Any) -> list[Record] | None:
    if isinstance(value, Mapping):
        rows = value.get("requests", value.get("request_metrics"))
        if isinstance(rows, Iterable) and not isinstance(rows, (str, bytes, Mapping)):
            return list(rows)
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        return list(value)
    return None


def compare_baseline(current: Any, baseline: Any, **kwargs: Any) -> GateResult:
    """Dispatch to row-level or summary-level baseline comparison.

    ``RunResult``, a sequence of request rows, and a serialized result mapping
    are all accepted.  This is deliberately a pure function so a CLI can
    decide how to display or persist a gate outcome.
    """
    if hasattr(current, "requests"):
        current = getattr(current, "requests")
    if hasattr(baseline, "requests"):
        baseline = getattr(baseline, "requests")
    current_rows = _records_from(current)
    baseline_rows = _records_from(baseline)
    if current_rows is not None and baseline_rows is not None:
        return compare_request_metrics(current_rows, baseline_rows, **kwargs)
    if isinstance(current, Mapping) and isinstance(baseline, Mapping):
        return compare_summaries(current, baseline, **kwargs)
    raise TypeError("current and baseline must both be request rows or summary mappings")


evaluate_pr_gate = compare_request_metrics
gate_against_baseline = compare_baseline
