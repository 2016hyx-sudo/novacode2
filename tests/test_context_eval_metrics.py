from __future__ import annotations

import pytest

from evals.structured_context.metrics import (
    aggregate_usage,
    compare_summaries,
    compare_request_metrics,
    negative_reduction_rate,
    nearest_rank,
    net_crr,
    summarize_metrics,
    task_median_crr,
    weighted_crr,
)
from evals.structured_context.schema import RequestMetric


def _metric(case: str, raw: int, structured: int, *, observation: int | None = None) -> RequestMetric:
    return RequestMetric(
        task_case_id=case,
        request_id=f"req-{case}-{raw}",
        raw_full_tokens=raw,
        observation_full_tokens=raw if observation is None else observation,
        structured_tokens=structured,
    )


def test_nearest_rank_handles_empty_repeats_and_boundaries() -> None:
    assert nearest_rank([], 0.50) is None
    assert nearest_rank([7], 0.95) == 7
    assert nearest_rank([1, 2], 0.50) == 1
    assert nearest_rank(list(range(1, 21)), 0.95) == 19
    assert nearest_rank([1, 1, 1, 9], 95) == 9
    with pytest.raises(ValueError):
        nearest_rank([1], 0)


def test_crr_formulas_skip_zero_denominators_and_keep_negative_samples() -> None:
    records = [
        _metric("a", 100, 50, observation=80),
        _metric("a", 100, 150, observation=100),
        _metric("zero", 0, 1),
        _metric("b", 1000, 100, observation=500),
    ]
    assert weighted_crr(records) == pytest.approx(1 - 300 / 1200)
    # Per-task weighted CRRs are 0.0 for task a and 0.9 for task b; nearest
    # rank p50 selects the lower value.
    assert task_median_crr(records) == pytest.approx(0.0)
    assert negative_reduction_rate(records) == pytest.approx(1 / 3)
    assert weighted_crr([_metric("zero", 0, 0)]) is None


def test_net_crr_counts_separate_fold_input_once() -> None:
    main = _metric("a", 1000, 400)
    fold = RequestMetric(
        task_case_id="a",
        request_id="fold-1",
        agent_role="fold",
        raw_full_tokens=0,
        logical_input_tokens=150,
    )
    assert net_crr([main, fold]) == pytest.approx(0.45)
    main_with_explicit = RequestMetric(
        task_case_id="a",
        request_id="main-2",
        raw_full_tokens=1000,
        structured_tokens=400,
        fold_input_tokens=200,
    )
    assert net_crr([main_with_explicit, fold]) == pytest.approx(0.4)


def test_pr_gate_uses_maximum_absolute_or_relative_tolerance() -> None:
    baseline = [_metric("case", 1_000, 1_000)]
    within = [_metric("case", 1_000, 1_128)]
    regressed = [_metric("case", 1_000, 1_129)]
    assert compare_request_metrics(within, baseline).passed is True
    result = compare_request_metrics(regressed, baseline)
    assert result.passed is False
    assert result.failures[0]["growth"] == 129


def test_full_run_summary_pairs_variants_and_counts_fold_overhead() -> None:
    records = [
        RequestMetric(
            task_case_id="case",
            request_id="raw-1",
            variant="raw_full",
            logical_input_tokens=1_000,
            raw_full_tokens=1_000,
            token_source="provider_reported",
        ),
        RequestMetric(
            task_case_id="case",
            request_id="structured-1",
            variant="structured",
            logical_input_tokens=600,
            structured_tokens=600,
            token_source="provider_reported",
        ),
        RequestMetric(
            task_case_id="case",
            request_id="fold-1",
            variant="structured",
            agent_role="fold",
            logical_input_tokens=100,
            token_source="provider_reported",
        ),
    ]
    summary = summarize_metrics(records, mode="full-run")
    assert summary["context"]["crr_weighted"] == pytest.approx(0.4)
    assert summary["context"]["net_crr"] == pytest.approx(0.3)
    assert summary["input_tokens"]["raw_full"]["p95"] == 1_000
    assert summary["input_tokens"]["structured"]["p50"] == 600


def test_full_run_summary_fails_closed_on_unpaired_tasks() -> None:
    raw_a = RequestMetric(task_case_id="a", request_id="raw-a", variant="raw_full", logical_input_tokens=100, token_source="provider_reported")
    raw_b = RequestMetric(task_case_id="b", request_id="raw-b", variant="raw_full", logical_input_tokens=100, token_source="provider_reported")
    structured_a = RequestMetric(task_case_id="a", request_id="structured-a", variant="structured", logical_input_tokens=50, token_source="provider_reported")

    summary = summarize_metrics([raw_a, raw_b, structured_a], mode="full-run")

    assert summary["sample"]["pairing_valid"] is False
    assert summary["context"]["crr_weighted"] is None
    assert summary["context"]["net_crr"] is None


def test_summary_gate_requires_metrics_and_row_gate_rejects_duplicates() -> None:
    gate = compare_summaries({"mode": "offline"}, {"mode": "offline"})
    assert not gate.passed
    assert all(item["type"] == "required_metric_present" for item in gate.failures)

    row = _metric("duplicate", 100, 50)
    duplicate_gate = compare_request_metrics([row, row], [row])
    assert not duplicate_gate.passed
    assert duplicate_gate.failures[0]["type"] == "duplicate_request_identity"


def test_aggregate_usage_sums_cache_fields_and_rate() -> None:
    cached = RequestMetric(
        task_case_id="a",
        request_id="r1",
        logical_input_tokens=2415,
        cache_hit_tokens=2368,
        fresh_processed_input_tokens=47,
        token_source="provider_reported",
        metadata={"raw_usage": {"output_tokens": 18}},
    )
    fresh = RequestMetric(
        task_case_id="a",
        request_id="r2",
        logical_input_tokens=51,
        cache_hit_tokens=0,
        fresh_processed_input_tokens=51,
        token_source="provider_reported",
        metadata={"raw_usage": {"output_tokens": 8}},
    )
    usage = aggregate_usage([cached, fresh])
    assert usage["request_count"] == 2
    assert usage["logical_input_tokens"] == 2466
    assert usage["cache_hit_tokens"] == 2368
    assert usage["fresh_processed_input_tokens"] == 98
    assert usage["output_tokens"] == 26
    assert usage["cache_hit_rate"] == pytest.approx(2368 / 2466)


def test_aggregate_usage_zero_denominator_has_none_rate() -> None:
    usage = aggregate_usage(
        [RequestMetric(task_case_id="a", request_id="r1", logical_input_tokens=0, cache_hit_tokens=0)]
    )
    assert usage["cache_hit_rate"] is None
    assert usage["logical_input_tokens"] == 0


def test_summaries_expose_usage_aggregation() -> None:
    records = [
        RequestMetric(
            task_case_id="a",
            request_id="r1",
            variant="structured",
            logical_input_tokens=100,
            cache_hit_tokens=80,
            fresh_processed_input_tokens=20,
            token_source="provider_reported",
        )
    ]
    for mode in ("offline", "full-run"):
        summary = summarize_metrics(records, mode=mode)
        assert summary["usage"]["cache_hit_tokens"] == 80
        assert summary["usage"]["cache_hit_rate"] == pytest.approx(0.8)
