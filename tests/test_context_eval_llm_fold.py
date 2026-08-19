"""Tests for the offline-llm-fold replay mode (real FoldEngine at fold points).

The deterministic ``offline`` mode simulates folds with rules; this mode runs
the production FoldEngine — with a fake provider in tests — and merges the
returned deltas into the replay state.  See ``baselines/README.md``: this mode
has no committed baseline and is never part of the PR gate.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_agent.llm.base import LLMError, LLMResponse
from coding_agent.runtime.trace import TraceWriter
from coding_agent.structured_context.fold_engine import FoldEngine, FoldEngineConfig

from evals.structured_context.__main__ import main
from evals.structured_context.full_context import (
    ReplayError,
    materialize_offline_case,
    run_offline_replay,
)
from evals.structured_context.llm_fold import run_llm_fold_replay

_SMALL_FOLD_CASE = {
    "id": "llm-fold-smoke",
    "bucket": "short-control",
    "generator": {
        "seed": 1,
        "interaction_groups": 10,
        "logical_window_tokens": 4096,
        "tool_results": [{"tool": "read_file", "count": 10, "chars_each": 1500}],
    },
    "expected": {"fold": True, "protected_recent_groups": 3},
    "assertions": [{"type": "fold_event_count_at_least", "count": 1}],
}

# key_findings upsert: the fold engine's "LLM answer" in most tests.
_DELTA_UPSERT = {
    "task_delta": {
        "set": {},
        "upsert": [
            {
                "target": "key_findings",
                "id": "kf-llm-1",
                "value": {
                    "id": "kf-llm-1",
                    "fact": "llm fold fact",
                    "evidence": [],
                    "status": "valid",
                    "updated_step": 0,
                },
            }
        ],
        "append": [],
        "remove": [],
        "mark_stale": [],
    },
    "tool_delta": {"set": {}, "upsert": [], "append": [], "remove": [], "mark_stale": []},
}


class _FoldLLM:
    """Fake fold provider; reports anthropic-shaped usage (input 250 / output 40)."""

    def __init__(self, text: str, *, fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0

    def chat(self, messages, tools=None, *, reasoning_effort=None):
        self.calls += 1
        if self.fail:
            raise LLMError("fold model unavailable")
        return LLMResponse(
            text=self.text,
            stop_reason="end_turn",
            usage={"input_tokens": 250, "output_tokens": 40},
        )


def _engine(llm: _FoldLLM | None = None, *, trace: TraceWriter | None = None) -> FoldEngine:
    # max_attempts=1 / retry_delay_s=0: never sleep in tests.
    return FoldEngine(
        llm,
        config=FoldEngineConfig(max_attempts=1, retry_delay_s=0),
        trace=trace,
        provider_name="test-provider",
        model="test-model",
    )


def _fold_events(replay) -> list[dict]:
    return list(replay[-1].metric.replay["fold_events"])


def test_llm_fold_state_evolves_with_delta() -> None:
    llm = _FoldLLM(json.dumps(_DELTA_UPSERT))
    replay = materialize_offline_case(
        _SMALL_FOLD_CASE, include_prompts=True, fold_engine=_engine(llm)
    )
    assert replay and llm.calls >= 1
    events = _fold_events(replay)
    assert events, "small case must cross the fold trigger"
    assert events[0]["model_used"] is True
    assert events[0]["logical_input_tokens"] == 250
    assert events[0]["output_tokens"] == 40
    # The merged finding survives into the last structured prompt's state.
    last = replay[-1].variants["structured"].messages
    assert "kf-llm-1" in json.dumps(last, ensure_ascii=False, default=str)


def test_fold_rows_carry_real_provider_usage() -> None:
    llm = _FoldLLM(json.dumps(_DELTA_UPSERT))
    result = run_llm_fold_replay([_SMALL_FOLD_CASE], fold_engine=_engine(llm), min_requests=0)
    main_rows = [r for r in result.requests if r.agent_role == "main"]
    fold_rows = [r for r in result.requests if r.agent_role == "fold"]
    assert len(main_rows) == 10
    assert fold_rows, "small case must fold at least once"
    row = fold_rows[0]
    assert row.token_source == "provider_reported"
    assert row.logical_input_tokens == 250
    assert row.cache_hit_tokens == 0
    assert row.provider == "test-provider"
    assert row.model == "test-model"
    assert row.request_id == f"offline-llm-fold-smoke-{row.metadata['fold_id']}"
    assert row.parent_request_id == f"offline-llm-fold-smoke-{int(row.step):04d}"
    assert row.metadata["raw_usage"] == {"output_tokens": 40}
    assert row.metadata["model_used"] is True
    assert row.metadata["calls"] == 1
    assert row.metadata["task_delta"] == _DELTA_UPSERT["task_delta"]
    assert result.manifest["model_calls"] == llm.calls
    assert result.manifest["llm_folds"] == len(fold_rows)
    assert result.manifest["fallback_folds"] == 0
    assert result.manifest["failed_case_count"] == 0
    # Main rows keep the deterministic replay semantics.
    assert all(r.token_source == "estimated" and r.provider == "offline" for r in main_rows)


def test_provider_error_falls_back_and_merges_deterministic_delta() -> None:
    llm = _FoldLLM("{}", fail=True)
    replay = materialize_offline_case(
        _SMALL_FOLD_CASE, include_prompts=True, fold_engine=_engine(llm)
    )
    events = _fold_events(replay)
    assert events, "small case must cross the fold trigger"
    event = events[0]
    assert event["model_used"] is False
    assert event["fallback_used"] is True
    assert event["last_error"] == "LLMError: fold model unavailable"
    assert event["logical_input_tokens"] == 0
    # The deterministic extractor still merged completed progress entries.
    # (Synthetic tool calls only carry fixture_group arguments, so no tool
    # path/query evidence is extractable — the tool delta stays empty.)
    assert event["task_delta"]["append"]
    assert event["tool_delta"]["append"] == []
    # The merged state is visible in a later request.
    assert any(
        "p-1-g-" in json.dumps(request.variants["structured"].messages, default=str)
        for request in replay
    )


def test_fold_engine_without_provider_uses_fallback() -> None:
    replay = materialize_offline_case(_SMALL_FOLD_CASE, fold_engine=_engine(None))
    event = _fold_events(replay)[0]
    assert event["model_used"] is False
    assert event["fallback_used"] is True
    assert event["last_error"] == "no fold provider configured"
    assert event["calls"] == 0
    assert event["task_delta"]["append"]


def test_run_llm_fold_replay_requires_engine() -> None:
    with pytest.raises(ReplayError, match="FoldEngine"):
        run_llm_fold_replay([_SMALL_FOLD_CASE], fold_engine=None, min_requests=0)


def test_cli_requires_provider_env_flag() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["offline-llm-fold"])
    assert excinfo.value.code == 2


def test_cli_fails_closed_without_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # Empty strings are already set in the environment, so load_env_file's
    # setdefault never fills them from .env.
    monkeypatch.setenv("NOVACODE_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("NOVACODE_API_KEY", "")
    with pytest.raises(SystemExit) as excinfo:
        main(["offline-llm-fold", "--provider-env"])
    assert excinfo.value.code == 2


def test_pure_offline_replay_unchanged() -> None:
    first = run_offline_replay()
    second = run_offline_replay()
    assert [item.to_dict() for item in first.requests] == [item.to_dict() for item in second.requests]
    assert first.manifest["main_request_count"] == 278
    assert first.manifest["model_calls"] == 0
    for row in first.requests:
        if row.agent_role == "fold":
            assert row.provider == "offline"
            assert row.token_source == "estimated"
            assert "model_used" not in row.metadata
            assert "logical_input_tokens" not in row.metadata


def test_llm_fold_preserves_scheduling() -> None:
    llm = _FoldLLM(json.dumps(_DELTA_UPSERT))
    replay = materialize_offline_case(_SMALL_FOLD_CASE, fold_engine=_engine(llm))
    last = replay[-1].metric
    events = last.replay["fold_events"]
    archive = set(last.replay["archive_group_ids"])
    retained = set(last.replay["retained_group_ids"])
    epochs = [int(event["epoch_id"]) for event in events]
    assert epochs == list(range(1, len(events) + 1))
    for event in events:
        folded = set(event["folded_group_ids"])
        assert folded <= archive
        assert not (folded & retained)
    assert int(last.epoch_id) == len(events)


def test_llm_fold_writes_trace_file(tmp_path: Path) -> None:
    trace = TraceWriter(tmp_path / "traces")
    llm = _FoldLLM(json.dumps(_DELTA_UPSERT))
    materialize_offline_case(_SMALL_FOLD_CASE, fold_engine=_engine(llm, trace=trace))
    written = list((tmp_path / "traces").glob("offline-llm-*.jsonl"))
    assert len(written) == 1
    events = [json.loads(line) for line in written[0].read_text(encoding="utf-8").splitlines()]
    kinds = {event["type"] for event in events}
    assert "llm_request_prepared" in kinds
    assert "llm_request_finished" in kinds
    finished = next(event for event in events if event["type"] == "llm_request_finished")
    assert finished["data"]["agent_role"] == "fold"
    assert finished["data"]["status"] == "success"
    assert finished["data"]["normalized_usage"]["logical_input_tokens"] == 250
