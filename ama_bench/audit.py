"""Per-episode audit trail for AMA-Bench runs.

Assembles the full pipeline record of one episode — input trajectory, pre-fold
group structure, fold decisions, post-fold memory, per-question retrieval and
prompts, answers and usage — into one JSON file so a bad answer can be traced
back to the exact stage that failed (memory loss, retrieval miss, or answer
generation).

Size control: structural data (group inventory, fold events, retrieval scores,
prompt hashes) is always recorded.  The full pre-compression trajectory text
and the full post-fold memory dump are ALWAYS recorded too — they are the
before/after evidence needed to re-evaluate an episode later.  The remaining
heavy content (full pre-fold group objects, per-question evidence blocks and
prompts) is only stored when ``full`` is set; otherwise hashes and char counts
stand in.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .memory import NovaCodeMemory
from .retrieve import render_evidence


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[:limit] + f"... [truncated {len(text) - limit} chars]"


def summarize_post_fold_memory(memory: NovaCodeMemory) -> dict[str, Any]:
    """Compact structural summary of the post-fold memory (always recorded)."""
    return {
        "task_state": {
            "completed": len(memory.task_state.completed),
            "key_findings": len(memory.task_state.key_findings),
            "decisions": len(memory.task_state.decisions),
            "unresolved": len(memory.task_state.unresolved),
            "key_sequences": len(memory.task_state.key_sequences),
        },
        "tool_state_profiles": {
            tool: sum(len(entries) for entries in profile.values())
            for tool, profile in memory.tool_state.profiles.items()
        },
        "recent_groups": [
            {"id": group.group_id, "messages": len(group.messages), "epoch": group.epoch_id}
            for group in memory.trajectory.groups
        ],
    }


def compact_memory_stats(stats: Mapping[str, Any]) -> dict[str, Any]:
    """Compact compression projection for result records (always recorded).

    Same shape as the audit record's ``memory`` block minus the fold-event
    detail, so the results JSONL already carries the headline numbers and the
    audit file is only needed when drilling into one episode.
    """
    return {
        "pre_fold_tokens": stats.get("pre_fold_tokens", 0),
        "post_fold_tokens": stats.get("post_fold_tokens", 0),
        "residual_ratio": (
            stats["post_fold_tokens"] / stats["pre_fold_tokens"]
            if stats.get("pre_fold_tokens")
            else None
        ),
        "groups": stats.get("groups", 0),
        "groups_folded": stats.get("groups_folded", 0),
        "groups_kept": stats.get("groups_kept", 0),
        "model_folds": stats.get("model_folds", 0),
        "fallback_folds": stats.get("fallback_folds", 0),
        "fold_calls": stats.get("fold_calls", 0),
        "fold_errors": stats.get("fold_errors", 0),
        "compact_task_evicted": len(stats.get("compact_task_evicted") or []),
        "compact_tool_evicted": len(stats.get("compact_tool_evicted") or []),
        "work_dir": stats.get("work_dir"),
        "notes": list(stats.get("notes", [])),
    }


def build_audit_record(
    *,
    episode: dict[str, Any],
    memory: NovaCodeMemory,
    questions: list[dict[str, Any]],
    outcome: dict[str, Any],
    trajectory_text: str,
    full: bool,
) -> dict[str, Any]:
    """Assemble the per-episode audit record from stage artifacts.

    ``questions`` is a list of per-question records::

        {"index", "question", "candidates" (scored retrieval list),
         "evidence" (rendered block), "prompt", "answer", "usage"}

    ``outcome`` holds the final answers plus the merged usage summary.

    The full pre-compression trajectory text (``trajectory_text``, the exact
    input fed to memory construction) and the full post-compression memory dump
    (``memory.to_dict()``) are ALWAYS recorded, so an episode can be re-evaluated
    later without rebuilding memory.  Only the heavy per-question content (full
    evidence blocks, prompts) and the full pre-fold group objects stay gated
    behind ``full``.
    """
    episode_id = int(episode.get("episode_id", 0))
    stats = memory.stats
    record: dict[str, Any] = {
        "episode_id": episode_id,
        "task": str(episode.get("task", "")),
        "task_type": str(episode.get("task_type", "")),
        "domain": str(episode.get("domain", "")),
        "input": {
            "steps": stats.get("steps", 0),
            "trajectory_chars": len(json.dumps(episode, ensure_ascii=False)),
            "num_turns": int(episode.get("num_turns", 0)),
            "question_count": len(questions),
        },
        "memory": {
            **compact_memory_stats(stats),
            "fold_events": [dict(event) for event in stats.get("fold_events", [])],
            "state_compact": {
                "evicted_task_items": [dict(item) for item in stats.get("compact_task_evicted") or []],
                "evicted_tool_items": [dict(item) for item in stats.get("compact_tool_evicted") or []],
            },
            "post_fold_summary": summarize_post_fold_memory(memory),
            # Full compressed memory (task/tool state + kept trajectory groups).
            # Always recorded so later evaluation needs no rebuild.
            "post_fold_memory": memory.to_dict(),
        },
        "pre_fold": {
            "group_inventory": [dict(item) for item in stats.get("group_inventory", [])],
            # The exact pre-compression input text. Always recorded.
            "trajectory_text": trajectory_text,
        },
        "questions": questions,
        "outcome": outcome,
    }
    if full:
        record["pre_fold"]["groups"] = _read_prefold_groups(stats.get("work_dir"))
    return record


def _read_prefold_groups(work_dir: str | None) -> list[dict[str, Any]]:
    """Full pre-fold group content from the build directory (full mode only)."""
    if not work_dir:
        return []
    path = Path(work_dir) / "groups.jsonl"
    if not path.is_file():
        return []
    groups: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                groups.append(json.loads(line))
    return groups


def write_audit(record: dict[str, Any], audit_dir: Path) -> Path:
    """Atomically write one audit record as ``<audit_dir>/<episode_id>.json``."""
    audit_dir.mkdir(parents=True, exist_ok=True)
    path = audit_dir / f"{record['episode_id']}.json"
    data = json.dumps(record, ensure_ascii=False, indent=2) + "\n"
    temp = path.with_suffix(".json.tmp")
    temp.write_text(data, encoding="utf-8")
    temp.replace(path)
    return path


def record_question(
    *,
    question: str,
    candidates: list[dict[str, Any]],
    prompt: str,
    answer: str,
    usage: dict[str, Any],
    full: bool,
    evidence_budget: int = 3_000,
) -> dict[str, Any]:
    """One question's retrieval/answer stage, size-controlled by ``full``.

    ``evidence_budget`` mirrors the prompt builder's per-question render
    budget (3_000 single / 2_000 batch), so the recorded evidence matches
    what the model actually saw.
    """
    evidence = render_evidence(candidates, max_total_chars=evidence_budget)
    return {
        "question": question,
        "retrieval": {
            "candidates": [
                {
                    "type": item.get("type"),
                    "id": item.get("id"),
                    "score": item.get("score", 0),
                    "meta": item.get("meta"),
                    "text": item.get("text") if full else _clip(str(item.get("text", "")), 300),
                }
                for item in candidates
            ],
            "evidence_hash": _hash(evidence),
            "evidence_chars": len(evidence),
            "evidence": evidence if full else None,
        },
        "prompt": {
            "hash": _hash(prompt),
            "chars": len(prompt),
            "full": prompt if full else None,
        },
        "answer": answer,
        "usage": dict(usage),
    }
