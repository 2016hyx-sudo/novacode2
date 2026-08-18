"""Tests for the standalone AMA-Bench runner (no AMA-Bench checkout needed)."""

from __future__ import annotations

from pathlib import Path

from ama_bench.run import (
    _query,
    detect_subset,
    exact_match_accuracy,
    filter_episodes,
    load_episodes,
    render_trajectory,
    run_episode,
)
from coding_agent.llm.base import LLMError, LLMResponse

EPISODE = {
    "episode_id": 7,
    "task": "collect a red apple",
    "trajectory": [
        {"turn_idx": 0, "action": "navigate(room=kitchen)", "observation": "kitchen with a red apple"},
        {"turn_idx": 1, "action": "grab(apple)", "observation": "holding red apple"},
    ],
    "qa_pairs": [
        {"question": "What did the agent grab?", "answer": "(A)"},
        {"question": "Where was the apple?", "answer": "(B)"},
    ],
}


class CannedProvider:
    """Fake LLM provider returning a fixed batch response."""

    def chat(self, messages, tools=None):
        return LLMResponse(text="Answer[1]: (A)\nAnswer[2]: (B)", stop_reason="end_turn", usage={})


class FlakyProvider:
    """Provider that fails (or returns empty) a few times, then succeeds."""

    def __init__(self, failures: int, *, empty: bool = False) -> None:
        self.failures = failures
        self.empty = empty
        self.calls = 0

    def chat(self, messages, tools=None):
        self.calls += 1
        if self.calls <= self.failures:
            if self.empty:
                return LLMResponse(text="", stop_reason="end_turn", usage={})
            raise LLMError("Connection error.", retryable=True)
        return LLMResponse(text="final answer", stop_reason="end_turn", usage={})


def test_query_retries_transient_failures() -> None:
    provider = FlakyProvider(failures=2)
    assert _query(provider, "q", max_tokens=64, base_delay=0.0) == "final answer"
    assert provider.calls == 3


def test_query_retries_empty_responses() -> None:
    provider = FlakyProvider(failures=1, empty=True)
    assert _query(provider, "q", max_tokens=64, base_delay=0.0) == "final answer"
    assert provider.calls == 2


def test_query_gives_up_after_max_retries() -> None:
    provider = FlakyProvider(failures=99)
    try:
        _query(provider, "q", max_tokens=64, max_retries=3, base_delay=0.0)
    except LLMError:
        assert provider.calls == 3
    else:  # pragma: no cover - must raise
        raise AssertionError("expected LLMError after retries exhausted")


def test_detect_subset(tmp_path: Path) -> None:
    assert detect_subset("dataset/test/mcq_set.jsonl") == "mcq"
    assert detect_subset("dataset/test/open_end_qa_set.jsonl") == "openend"


def test_load_and_filter_episodes(tmp_path: Path) -> None:
    path = tmp_path / "episodes.jsonl"
    path.write_text(json_line({"episode_id": 1}) + json_line({"episode_id": 2}) + json_line({"episode_id": 3}), encoding="utf-8")
    episodes = load_episodes(path)
    assert [episode["episode_id"] for episode in episodes] == [1, 2, 3]
    assert [episode["episode_id"] for episode in filter_episodes(episodes, episode_ids=[2])] == [2]
    sampled = filter_episodes(episodes, samples=2)
    assert len(sampled) == 2


def json_line(value: dict) -> str:
    import json

    return json.dumps(value) + "\n"


def test_render_trajectory_uses_turn_idx() -> None:
    text = render_trajectory(EPISODE)
    assert "Step 0:" in text
    assert "Action: grab(apple)" in text
    assert "Observation: holding red apple" in text


def test_run_episode_batch(tmp_path: Path) -> None:
    from ama_bench.method import NovaCodeMemoryMethod

    method = NovaCodeMemoryMethod()
    result = run_episode(method, CannedProvider(), EPISODE, subset="mcq", max_tokens=256, per_question=False)
    assert result["episode_id"] == 7
    assert result["answer_list"] == ["(A)", "(B)"]


def test_run_episode_per_question(tmp_path: Path) -> None:
    from ama_bench.method import NovaCodeMemoryMethod

    method = NovaCodeMemoryMethod()
    result = run_episode(method, CannedProvider(), EPISODE, subset="mcq", max_tokens=256, per_question=True)
    # The canned provider returns the same batch response each call, so both
    # per-question answers come from the first Answer[1] block.
    assert result["answer_list"] == ["(A)", "(A)"]


def test_exact_match_accuracy_mcq() -> None:
    results = [{"episode_id": 7, "answer_list": ["(A)", "(C)"]}]
    stats = exact_match_accuracy(results, [EPISODE], subset="mcq")
    assert stats["correct"] == 1
    assert stats["total"] == 2
    assert stats["accuracy"] == 0.5
