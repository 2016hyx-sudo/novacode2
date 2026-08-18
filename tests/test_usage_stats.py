"""Session-level usage aggregation tests (UsageStats / UsageEventAggregator)."""
from __future__ import annotations

import pytest

from coding_agent.runtime.trace import TraceEvent
from coding_agent.structured_context.usage_stats import UsageEventAggregator, UsageStats


def _normalized(**overrides: int) -> dict:
    base = {
        "logical_input_tokens": 1000,
        "cache_hit_tokens": 900,
        "fresh_processed_input_tokens": 100,
        "output_tokens": 50,
        "provider_format": "anthropic",
    }
    base.update(overrides)
    return base


def test_usage_stats_accumulates_and_computes_hit_rate() -> None:
    stats = UsageStats()
    stats.record(_normalized(), latency_ms=2000)
    stats.record(_normalized(cache_hit_tokens=0, fresh_processed_input_tokens=500, provider_format="openai"))
    data = stats.to_dict()
    assert data["request_count"] == 2
    assert data["logical_input_tokens"] == 2000
    assert data["cache_hit_tokens"] == 900
    assert data["fresh_processed_input_tokens"] == 600
    assert data["output_tokens"] == 100
    assert data["cache_hit_rate"] == pytest.approx(0.45)
    assert data["avg_latency_ms"] == pytest.approx(1000.0)
    assert data["provider_formats"] == ["anthropic", "openai"]


def test_usage_stats_zero_denominator_rate_is_zero() -> None:
    stats = UsageStats()
    stats.record(_normalized(logical_input_tokens=0, cache_hit_tokens=0))
    assert stats.to_dict()["cache_hit_rate"] == 0.0
    assert stats.to_dict()["avg_latency_ms"] is None


def test_usage_stats_roundtrip_from_dict() -> None:
    stats = UsageStats()
    stats.record(_normalized(), latency_ms=1200)
    stats.record(_normalized(cache_hit_tokens=0))
    restored = UsageStats.from_dict(stats.to_dict())
    assert restored.to_dict() == stats.to_dict()


def test_aggregator_ignores_non_usage_events() -> None:
    aggregator = UsageEventAggregator()
    aggregator(TraceEvent(type="session_start", session_id="s", data={"task": "t"}))
    aggregator(
        TraceEvent(
            type="llm_request_finished",
            session_id="s",
            data={"normalized_usage": _normalized(), "latency_ms": 900},
        )
    )
    assert aggregator.stats.to_dict()["request_count"] == 1
    assert aggregator.stats.to_dict()["cache_hit_tokens"] == 900


def test_aggregator_tolerates_missing_usage_on_error_events() -> None:
    aggregator = UsageEventAggregator()
    aggregator(
        TraceEvent(
            type="llm_request_finished",
            session_id="s",
            data={"status": "provider_error", "normalized_usage": {}},
        )
    )
    data = aggregator.stats.to_dict()
    assert data["request_count"] == 1
    assert data["logical_input_tokens"] == 0
