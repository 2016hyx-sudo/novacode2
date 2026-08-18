"""Tests for the per-episode AMA-Bench audit trail (ama_bench.audit + run_episode)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ama_bench.audit import (
    build_audit_record,
    compact_memory_stats,
    record_question,
    write_audit,
)
from ama_bench.method import NovaCodeMemoryMethod
from ama_bench.run import _merge_memory_stats, render_trajectory, run_episode
from coding_agent.llm.base import LLMResponse

EPISODE = {
    "episode_id": 7,
    "task": "collect a red apple",
    "task_type": "interactive",
    "domain": "navigation",
    "trajectory": [
        {"turn_idx": 0, "action": "navigate(room=kitchen)", "observation": "kitchen with a red apple"},
        {"turn_idx": 1, "action": "grab(apple)", "observation": "holding red apple"},
    ],
    "qa_pairs": [
        {"question": "What did the agent grab?"},
        {"question": "Where was the apple?"},
    ],
}


class CannedProvider:
    def chat(self, messages, tools=None):
        return LLMResponse(
            text="Answer[1]: (A)\nAnswer[2]: (B)",
            stop_reason="end_turn",
            usage={"input_tokens": 120, "output_tokens": 8, "cache_read_input_tokens": 90},
        )


def _candidates() -> list[dict]:
    return [
        {"type": "group", "id": "g-1", "score": 4, "text": "A" * 900, "meta": {"step_range": "1-2"}},
        {"type": "finding", "id": "f-2", "score": 1, "text": "short fact", "meta": {"status": "confirmed"}},
    ]


def _memory() -> object:
    method = NovaCodeMemoryMethod(keep_work_dir=True)
    return method.memory_construction(render_trajectory(EPISODE), task=EPISODE["task"])


def test_record_question_compact_clips_content_and_keeps_scores() -> None:
    record = record_question(
        question="What did the agent grab?",
        candidates=_candidates(),
        prompt="Q: What did the agent grab?\n...",
        answer="(A)",
        usage={"request_count": 1},
        full=False,
    )
    retrieval = record["retrieval"]
    assert retrieval["evidence"] is None
    assert record["prompt"]["full"] is None
    assert retrieval["candidates"][0]["score"] == 4
    assert "[truncated 600 chars]" in retrieval["candidates"][0]["text"]
    assert len(retrieval["candidates"][0]["text"]) < 900
    assert record["prompt"]["hash"]  # sha256 prefix present
    assert record["prompt"]["chars"] == len(record["prompt"]["full"] or "Q: What did the agent grab?\n...")


def test_record_question_full_keeps_content() -> None:
    candidates = _candidates()
    record = record_question(
        question="Q?",
        candidates=candidates,
        prompt="full prompt",
        answer="(A)",
        usage={},
        full=True,
    )
    assert record["retrieval"]["evidence"].startswith("<evidence id=\"g-1\"")
    assert record["prompt"]["full"] == "full prompt"
    assert record["retrieval"]["candidates"][0]["text"] == "A" * 900


def test_build_audit_record_compact_shape() -> None:
    memory = _memory()
    outcome = {"answer_list": ["(A)", "(B)"], "reasoning_trace": "", "usage": {"request_count": 2}}
    record = build_audit_record(
        episode=EPISODE,
        memory=memory,
        questions=[{"question": "q1"}, {"question": "q2"}],
        outcome=outcome,
        full=False,
    )
    assert record["episode_id"] == 7
    assert record["input"]["question_count"] == 2
    assert record["memory"]["pre_fold_tokens"] > 0
    assert record["memory"]["post_fold_tokens"] > 0
    assert record["memory"]["residual_ratio"] == pytest.approx(
        record["memory"]["post_fold_tokens"] / record["memory"]["pre_fold_tokens"]
    )
    # Compact: no full pre-fold groups, no post-fold memory dump, counts only.
    assert "groups" not in record["pre_fold"]
    assert "post_fold_memory" not in record["memory"]
    assert record["pre_fold"]["group_inventory"]
    assert record["outcome"] == outcome


def test_build_audit_record_full_includes_content() -> None:
    memory = _memory()  # keep_work_dir=True -> groups.jsonl survives
    record = build_audit_record(
        episode=EPISODE,
        memory=memory,
        questions=[],
        outcome={"answer_list": []},
        full=True,
    )
    groups = record["pre_fold"]["groups"]
    assert groups  # full pre-fold groups read back from the work dir
    assert all("messages" in group for group in groups)
    assert "post_fold_memory" in record["memory"]
    assert record["memory"]["post_fold_summary"]["recent_groups"]


def test_write_audit_is_atomic_and_named_by_episode(tmp_path: Path) -> None:
    path = write_audit({"episode_id": 7, "payload": 1}, tmp_path)
    assert path == tmp_path / "7.json"
    assert json.loads(path.read_text(encoding="utf-8"))["payload"] == 1
    assert not list(tmp_path.glob("*.tmp"))
    write_audit({"episode_id": 7, "payload": 2}, tmp_path)
    assert json.loads(path.read_text(encoding="utf-8"))["payload"] == 2


def test_compact_memory_stats_counts_evictions() -> None:
    stats = {
        "pre_fold_tokens": 100,
        "post_fold_tokens": 40,
        "groups": 4,
        "groups_folded": 3,
        "groups_kept": 1,
        "model_folds": 2,
        "fallback_folds": 1,
        "fold_calls": 3,
        "fold_errors": 0,
        "compact_task_evicted": [{"id": "t1"}, {"id": "t2"}],
        "compact_tool_evicted": [],
        "work_dir": "/tmp/x",
        "notes": ["note"],
    }
    compact = compact_memory_stats(stats)
    assert compact["compact_task_evicted"] == 2
    assert compact["compact_tool_evicted"] == 0
    assert compact["residual_ratio"] == pytest.approx(0.4)
    assert compact["work_dir"] == "/tmp/x"


def test_merge_memory_stats_sums_across_episodes() -> None:
    merged = _merge_memory_stats(
        [
            {"pre_fold_tokens": 100, "post_fold_tokens": 40, "groups": 4, "groups_folded": 3,
             "model_folds": 2, "fallback_folds": 1},
            {"pre_fold_tokens": 200, "post_fold_tokens": 80, "groups": 6, "groups_folded": 2,
             "model_folds": 0, "fallback_folds": 2},
            None,
        ]
    )
    assert merged["episodes"] == 2
    assert merged["pre"] == 300
    assert merged["folded"] == 5
    assert merged["model"] == 2
    assert merged["fallback"] == 3


def test_run_episode_writes_audit_and_memory_summary(tmp_path: Path) -> None:
    audit_dir = tmp_path / "audit"
    result = run_episode(
        NovaCodeMemoryMethod(),
        CannedProvider(),
        EPISODE,
        subset="openend",
        max_tokens=256,
        per_question=True,
        audit_dir=audit_dir,
    )
    assert result["audit_path"] == str(audit_dir / "7.json")
    memory = result["memory"]
    assert memory["pre_fold_tokens"] > 0
    assert memory["groups"] == memory["groups_folded"] + memory["groups_kept"]

    record = json.loads(Path(result["audit_path"]).read_text(encoding="utf-8"))
    assert len(record["questions"]) == 2
    assert record["questions"][0]["question"] == "What did the agent grab?"
    assert record["questions"][0]["answer"] == "(A)"
    # Provider-reported usage flows through normalize_usage into the question record
    # (anthropic accounting: logical = input + cache_read + cache_creation).
    assert record["questions"][0]["usage"]["logical_input_tokens"] == 210
    assert record["questions"][0]["usage"]["cache_hit_tokens"] == 90
    assert record["outcome"]["answer_list"] == ["(A)", "(A)"]
    assert record["outcome"]["usage"]["request_count"] == 2


def test_run_episode_without_audit_keeps_plain_result() -> None:
    result = run_episode(
        NovaCodeMemoryMethod(),
        CannedProvider(),
        EPISODE,
        subset="mcq",
        max_tokens=256,
        per_question=False,
    )
    assert "audit_path" not in result
    assert "memory" not in result
