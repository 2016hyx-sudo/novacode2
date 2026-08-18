"""Session-level usage aggregation from trace usage events.

A single ``UsageEventAggregator`` attaches to the ``TraceWriter`` listener
list and accumulates every ``llm_request_finished`` event — main agent calls,
fold calls and any future provider call share the same emission path, so the
aggregate covers the whole session.  ``run_task`` reads ``stats.to_dict()``
into ``session_metrics`` and the session's persisted ``metrics`` dict, which
makes cache hit rate part of both the trace log and the checkpointed session.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..llm.usage import normalize_usage


class UsageStats:
    """Pure accumulator over provider-normalized usage records."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.request_count = 0
        self.logical_input_tokens = 0
        self.cache_hit_tokens = 0
        self.fresh_processed_input_tokens = 0
        self.output_tokens = 0
        self.latency_sum_ms = 0.0
        self.provider_formats: set[str] = set()

    def record(self, normalized: dict[str, Any], *, latency_ms: int | None = None) -> None:
        """Accumulate one provider call; ``normalized`` is a ``normalize_usage`` view."""
        self.request_count += 1
        self.logical_input_tokens += int(normalized.get("logical_input_tokens") or 0)
        self.cache_hit_tokens += int(normalized.get("cache_hit_tokens") or 0)
        self.fresh_processed_input_tokens += int(normalized.get("fresh_processed_input_tokens") or 0)
        self.output_tokens += int(normalized.get("output_tokens") or 0)
        if latency_ms is not None:
            self.latency_sum_ms += float(latency_ms)
        provider_format = normalized.get("provider_format")
        if provider_format:
            self.provider_formats.add(str(provider_format))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "logical_input_tokens": self.logical_input_tokens,
            "cache_hit_tokens": self.cache_hit_tokens,
            "fresh_processed_input_tokens": self.fresh_processed_input_tokens,
            "output_tokens": self.output_tokens,
            "cache_hit_rate": (
                self.cache_hit_tokens / self.logical_input_tokens if self.logical_input_tokens else 0.0
            ),
            "avg_latency_ms": (
                round(self.latency_sum_ms / self.request_count, 1)
                if self.request_count and self.latency_sum_ms > 0
                else None
            ),
            "provider_formats": sorted(self.provider_formats),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "UsageStats":
        data = data or {}
        stats = cls()
        stats.request_count = int(data.get("request_count") or 0)
        stats.logical_input_tokens = int(data.get("logical_input_tokens") or 0)
        stats.cache_hit_tokens = int(data.get("cache_hit_tokens") or 0)
        stats.fresh_processed_input_tokens = int(data.get("fresh_processed_input_tokens") or 0)
        stats.output_tokens = int(data.get("output_tokens") or 0)
        stats.latency_sum_ms = float(data.get("avg_latency_ms") or 0) * stats.request_count
        stats.provider_formats = set(str(item) for item in data.get("provider_formats") or [])
        return stats


class UsageEventAggregator:
    """TraceWriter listener that folds ``llm_request_finished`` into ``UsageStats``."""

    def __init__(self) -> None:
        self.stats = UsageStats()

    def __call__(self, event: Any) -> None:
        if getattr(event, "type", "") != "llm_request_finished":
            return
        data = getattr(event, "data", None) or {}
        normalized = data.get("normalized_usage") or {}
        if not isinstance(normalized, dict):
            normalized = normalize_usage(None)
        self.stats.record(normalized, latency_ms=data.get("latency_ms"))
