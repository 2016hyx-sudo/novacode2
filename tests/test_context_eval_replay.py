from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_agent.llm.base import Message, ToolCall

from evals.structured_context.full_context import (
    ReplayError,
    load_offline_cases,
    materialize_offline_case,
    reconstruct_prompt_variants,
    run_offline_replay,
)
from evals.structured_context.report import write_report


def _case(result, case_id: str):
    return [
        item
        for item in result.requests
        if item.task_case_id == case_id and item.agent_role == "main"
    ]


def test_offline_replay_is_deterministic_and_has_pr_sized_population() -> None:
    first = run_offline_replay()
    second = run_offline_replay()
    assert sum(item.agent_role == "main" for item in first.requests) >= 200
    assert [item.to_dict() for item in first.requests] == [item.to_dict() for item in second.requests]
    assert any(item.agent_role == "fold" for item in first.requests)
    assert first.manifest["main_request_count"] == 278


def test_threshold_and_fold_recipe_contracts() -> None:
    result = run_offline_replay()
    below = _case(result, "offline-threshold-below-02")[-1]
    above = _case(result, "offline-threshold-above-03")[-1]
    pre_fold = _case(result, "offline-pre-fold-07")[-1]
    boundary = _case(result, "offline-fold-boundary-08")[-1]
    protected = _case(result, "offline-protected-groups-09")[-1]

    assert below.raw_full_tokens == below.observation_full_tokens
    assert above.raw_full_tokens > above.observation_full_tokens
    assert above.replay["artifact_refs"]
    assert pre_fold.structured_tokens < int(pre_fold.replay["logical_window_tokens"] * 0.70)
    assert boundary.replay["fold_count"] == 1
    assert boundary.replay["fold_events"][0]["step"] == 21
    assert {"g-0022", "g-0023", "g-0024"} <= set(protected.replay["retained_group_ids"])


def test_state_compact_recipe_preserves_ids_and_stale_first_eviction() -> None:
    result = run_offline_replay()
    record = _case(result, "offline-state-compact-11")[-1]
    compact = record.replay["state_compact_events"][0]
    assert compact["task_tokens_after"] <= 8_000
    assert compact["tool_tokens_after"] <= 8_000
    assert {"f-current-contract", "d-public-api", "t-active"} <= set(compact["preserved_ids"])
    assert compact["evicted_stale_ids"]
    assert compact["evicted_ids"][: len(compact["evicted_stale_ids"])] == compact["evicted_stale_ids"]
    assert compact["overflow_artifact_id"]


def test_raw_reconstruction_reads_artifact_and_respects_anchor() -> None:
    raw = "head\nCRITICAL\ntail"
    artifact_id = __import__("hashlib").sha256(raw.encode()).hexdigest()
    group_one = {
        "group_id": "g-1",
        "status": "complete",
        "source_events": {"first_seq": 1, "last_seq": 2},
        "raw_tool_result_refs": [{"tool_call_id": "c1", "artifact_id": artifact_id, "sha256": artifact_id}],
        "messages": [
            Message(role="assistant", tool_calls=[ToolCall(id="c1", name="read_file")]).to_dict(),
            Message(role="tool", content=f"[artifact_id: {artifact_id}]", tool_call_id="c1", name="read_file").to_dict(),
        ],
    }
    group_future = {
        "group_id": "g-2",
        "status": "complete",
        "source_events": {"first_seq": 3, "last_seq": 4},
        "messages": [Message(role="user", content="future").to_dict()],
    }
    variants = reconstruct_prompt_variants(
        system_text="sys",
        groups=[group_one, group_future],
        artifact_reader={artifact_id: raw.encode()},
        event_seq_anchor=2,
    )
    raw_messages = variants["raw_full"].messages
    assert any(message.get("content") == raw for message in raw_messages)
    assert not any(message.get("content") == "future" for message in raw_messages)


def test_report_files_are_stable_and_atomic_at_call_boundary(tmp_path: Path) -> None:
    result = run_offline_replay()
    paths = write_report(result, tmp_path)
    first = {name: path.read_bytes() for name, path in paths.items()}
    write_report(result, tmp_path)
    assert {name: path.read_bytes() for name, path in paths.items()} == first
    assert {path.name for path in paths.values()} == {"requests.jsonl", "summary.json", "report.md"}


def test_reasoning_recipes_preserve_and_count_thinking_blocks() -> None:
    result = run_offline_replay()
    anthropic = _case(result, "offline-reasoning-anthropic-13")
    openai = _case(result, "offline-reasoning-openai-14")
    assert anthropic
    assert openai
    for record in (*anthropic, *openai):
        assert record.layers.get("reasoning", 0) > 0

    cases = {str(case["id"]): case for case in load_offline_cases()}
    fold_replay = materialize_offline_case(cases["offline-reasoning-fold-15"])
    assert fold_replay
    last = fold_replay[-1]
    assert last.metric.replay["fold_count"] >= 1
    # Folded thinking is archived: raw keeps all groups' reasoning while the
    # structured prompt only retains the recent protected groups' reasoning.
    assert last.variants["raw_full"].layers["reasoning"] > last.variants["structured"].layers["reasoning"] > 0


def test_reasoning_marker_survives_into_reconstructed_messages() -> None:
    cases = {str(case["id"]): case for case in load_offline_cases()}
    anthropic = materialize_offline_case(cases["offline-reasoning-anthropic-13"], include_prompts=True)
    openai = materialize_offline_case(cases["offline-reasoning-openai-14"], include_prompts=True)
    for replay in (anthropic, openai):
        for request in replay:
            for variant in request.variants.values():
                assert any(
                    "THINKING:" in json.dumps(message, ensure_ascii=False, default=str)
                    for message in variant.messages
                )


def test_unknown_recipe_assertion_fails_closed() -> None:
    case = {
        "id": "invalid-assertion",
        "bucket": "short-control",
        "generator": {"seed": 1, "interaction_groups": 1, "tool_results": []},
        "expected": {"fold": False},
        "assertions": [{"type": "not-a-real-contract"}],
    }

    with pytest.raises(ReplayError, match="unsupported assertion"):
        materialize_offline_case(case)
