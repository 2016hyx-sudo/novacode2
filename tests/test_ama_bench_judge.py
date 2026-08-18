"""Tests for the self-contained LLM-as-judge evaluation module."""

from __future__ import annotations

from ama_bench.judge import (
    build_qa_results,
    compute_f1_score,
    evaluate_results,
    llm_as_judge,
    main,
)
from coding_agent.llm.base import LLMResponse

EPISODE = {
    "episode_id": 3,
    "task": "collect a red apple",
    "task_type": "embodied_ai",
    "domain": "game",
    "qa_pairs": [
        {"question": "What did the agent grab?", "answer": "a red apple", "type": "A"},
        {"question": "Where was the apple?", "answer": "kitchen", "type": "B"},
    ],
}


class JudgeProvider:
    """Fake judge provider returning a fixed response per call."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls = 0

    def chat(self, messages, tools=None):
        text = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return LLMResponse(text=text, stop_reason="end_turn", usage={})


def test_compute_f1_score_matches_official_semantics() -> None:
    assert compute_f1_score("apple pie", "apple pie") == 1.0
    assert compute_f1_score("banana", "apple") == 0.0
    assert compute_f1_score("", "") == 1.0
    assert 0.0 < compute_f1_score("apple pie", "apple") < 1.0


def test_llm_as_judge_parses_yes_no() -> None:
    provider = JudgeProvider(["yes", "no"])
    assert llm_as_judge("q", "gold", "pred", provider) == 1.0
    assert llm_as_judge("q", "gold", "pred", provider) == 0.0


def test_llm_as_judge_strips_think_and_takes_last() -> None:
    provider = JudgeProvider(["<think>the answer is yes</think> no", "not sure but yes"])
    assert llm_as_judge("q", "gold", "pred", provider) == 0.0  # last word wins
    assert llm_as_judge("q", "gold", "pred", provider) == 1.0


def test_llm_as_judge_falls_back_to_f1() -> None:
    provider = JudgeProvider(["hmm, uncertain", "I cannot tell"])
    # No yes/no tokens -> token-level F1 against the golden answer.
    assert llm_as_judge("q", "apple pie", "apple pie", provider) == 1.0
    assert llm_as_judge("q", "apple pie", "banana", provider) == 0.0


def test_build_qa_results_matches_answers_to_golden() -> None:
    results = [{"episode_id": 3, "answer_list": ["red apple", "kitchen"]}]
    qa = build_qa_results(results, [EPISODE])
    assert len(qa) == 2
    assert qa[0]["question"] == "What did the agent grab?"
    assert qa[0]["golden_answer"] == "a red apple"
    assert qa[0]["predicted_answer"] == "red apple"
    assert qa[0]["domain"] == "game"
    assert qa[0]["qa_type"] == "A"


def test_evaluate_results_stats() -> None:
    results = [{"episode_id": 3, "answer_list": ["a red apple", "kitchen"]}]
    qa = build_qa_results(results, [EPISODE])
    provider = JudgeProvider(["yes", "no"])
    summary = evaluate_results(qa, provider, judge_model="fake-judge", max_workers=2)
    assert summary["overall"]["total_questions"] == 2
    assert summary["overall"]["accuracy"] == 0.5
    assert summary["overall"]["avg_score"] == 0.5
    assert summary["by_domain"]["game"]["count"] == 2
    assert summary["by_qa_type"]["A"]["count"] == 1
    assert summary["config"]["judge_model"] == "fake-judge"


def test_cli_missing_files_return_error() -> None:
    assert main(["--answers-file", "nope.jsonl", "--test-file", "also-nope.jsonl", "--no-env"]) == 2
