"""Standalone AMA-Bench runner for NovaCode (no AMA-Bench checkout needed).

Reads the official dataset JSONL directly, drives ``NovaCodeMemoryMethod``,
answers all questions with NovaCode's own LLM provider, and writes results in
the AMA-Bench submission format (``episode_id`` / ``answer_list`` /
``reasoning_trace``).  For MCQ subsets with ground-truth answers it also
reports local exact-match accuracy; the official LLM-as-judge scoring still
comes from AMA-Bench's ``evaluate.py`` (run it on the same results file).

Usage:

    python -m ama_bench.run \\
        --dataset dataset/test/open_end_qa_set.jsonl \\
        --episode-ids 0,1,2 \\
        --output results/novacode_openend.jsonl \\
        --audit-dir results/audit \\
        --audit-full

Every episode writes an audit JSON (``<audit-dir>/<episode_id>.json``, compact
by default) tracing memory build -> fold -> retrieval -> answer, so a bad
answer can be debugged back to the exact stage.  ``--audit-full`` records full
pre-fold groups, evidence and prompts; ``--method-config`` points at a
JSON/YAML config (``max_context_tokens`` / ``fold_max_attempts`` /
``keep_work_dir``).

Provider configuration comes from the environment / ``.env`` exactly like the
main NovaCode CLI (``NOVACODE_PROVIDER``, ``NOVACODE_MODEL``, ``OPENAI_API_KEY``
or ``ANTHROPIC_API_KEY``).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from coding_agent.llm import LLMProvider, Message, create_provider
from coding_agent.llm.usage import normalize_usage
from config import LLMConfig, load_env_file

from .audit import build_audit_record, compact_memory_stats, record_question, write_audit
from .extract import parse_answer_blocks
from .io import iter_jsonl_records
from .memory import NovaCodeMemory
from .method import NovaCodeMemoryMethod
from .retrieve import score_candidates


def load_episodes(dataset: str | Path) -> list[dict[str, Any]]:
    """Load the benchmark JSONL into episode records (one dict per line)."""
    return list(iter_jsonl_records(dataset))


def filter_episodes(
    episodes: list[dict[str, Any]],
    *,
    episode_ids: list[int] | None = None,
    samples: int | None = None,
) -> list[dict[str, Any]]:
    """Keep episodes by id or random sample; ``episode_ids`` wins over ``samples``."""
    if episode_ids:
        wanted = set(episode_ids)
        return [episode for episode in episodes if int(episode.get("episode_id", -1)) in wanted]
    if samples is not None:
        chosen = random.Random(20260818).sample(episodes, min(samples, len(episodes)))
        return sorted(chosen, key=lambda episode: int(episode.get("episode_id", 0)))
    return episodes


def render_trajectory(episode: dict[str, Any]) -> str:
    """Render an episode's trajectory list into the benchmark text format."""
    trajectory = episode.get("trajectory") or []
    steps = []
    for index, step in enumerate(trajectory):
        if isinstance(step, str):
            steps.append(f"Step {index}:\nAction: \nObservation: {step}")
            continue
        if not isinstance(step, dict):
            continue
        turn = step.get("turn_idx", index)
        action = str(step.get("action", ""))
        observation = str(step.get("observation", ""))
        steps.append(f"Step {turn}:\nAction: {action}\nObservation: {observation}")
    return "\n\n".join(steps)


def detect_subset(dataset: str | Path) -> str:
    """Heuristic subset detection from the file name."""
    name = Path(dataset).name.lower()
    if "mcq" in name:
        return "mcq"
    if "open" in name:
        return "openend"
    raise ValueError(
        f"cannot detect subset from file name {name!r}; pass --subset explicitly"
    )


def episode_questions(episode: dict[str, Any]) -> list[str]:
    qa_pairs = episode.get("qa_pairs") or []
    return [str(pair.get("question", "")) for pair in qa_pairs if pair.get("question")]


def run_episode(
    method: NovaCodeMemoryMethod,
    provider: LLMProvider,
    episode: dict[str, Any],
    *,
    subset: str,
    max_tokens: int = 4096,
    per_question: bool = True,
    progress: Callable[[int, int], None] | None = None,
    audit_dir: str | Path | None = None,
    audit_full: bool = False,
) -> dict[str, Any]:
    """Run one episode: build memory, answer every question, return a result record.

    Per-question mode (default) is the reliable path: one focused LLM call per
    question, so answers are never dropped to a long-prompt timeout.  Batch
    mode sends all questions in one call and should only be used with models
    that reliably follow the ``Answer[i]:`` block format and tolerate long
    prompts.

    ``progress(done, total)`` is invoked after every answered question.  A
    failed per-question call records an empty answer and the episode continues,
    so one timeout does not abort the episode; the caller persists the episode
    once it completes.

    With ``audit_dir`` set, the episode's full pipeline record — pre-fold group
    inventory, fold events, post-fold memory, per-question retrieval and
    prompts — is written to ``<audit_dir>/<episode_id>.json`` and the result
    gains ``audit_path`` plus a compact ``memory`` compression summary.
    ``audit_full`` additionally records full pre-fold groups, evidence and
    prompts (larger files; also keeps the fold work directory).
    """
    episode_id = int(episode.get("episode_id", 0))
    task = str(episode.get("task", ""))
    questions = episode_questions(episode)
    trajectory_text = render_trajectory(episode)
    mcq_mode = subset == "mcq"

    memory = method.memory_construction(trajectory_text, task=task)
    audit_questions: list[dict[str, Any]] = []
    if not questions:
        base = {
            "episode_id": episode_id,
            "answer_list": [],
            "reasoning_trace": "",
            "usage": _merge_usage([]),
        }
        return _with_audit(base, memory, episode, audit_questions, audit_dir, audit_full)

    usages: list[dict[str, Any]] = []
    if per_question:
        answer_list: list[str] = []
        for index, question in enumerate(questions, start=1):
            answer = ""
            prompt = ""
            usage: dict[str, Any] = {}
            try:
                prompt = method.build_prompt(memory, [question], mcq_mode=mcq_mode)
                response, usage = _query_with_usage(provider, prompt, max_tokens=max_tokens)
                usages.append(usage)
                parsed = parse_answer_blocks(response, 1, mcq_mode=mcq_mode)
                answer = parsed[0] if parsed else ""
            except Exception as exc:  # one bad call must not lose the episode
                print(
                    f"[ama]   episode {episode_id} question {index} failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
            if audit_dir is not None:
                audit_questions.append(
                    record_question(
                        question=question,
                        candidates=score_candidates(memory, question),
                        prompt=prompt,
                        answer=answer,
                        usage=usage,
                        full=audit_full,
                        evidence_budget=3_000,
                    )
                )
            answer_list.append(answer)
            if progress is not None:
                progress(index, len(questions))
    else:
        prompt = method.build_prompt(memory, questions, mcq_mode=mcq_mode)
        response, usage = _query_with_usage(provider, prompt, max_tokens=max_tokens)
        usages.append(usage)
        answer_list = parse_answer_blocks(response, len(questions), mcq_mode=mcq_mode)
        if len(answer_list) != len(questions):
            answer_list = _repair_answers(answer_list, questions, mcq_mode)
        if audit_dir is not None:
            # One LLM call answered every question; each question record
            # shares the batch prompt and the call's usage (first record only,
            # the merged total lives in ``outcome.usage``).
            for index, (question, answer) in enumerate(zip(questions, answer_list), start=1):
                audit_questions.append(
                    record_question(
                        question=question,
                        candidates=score_candidates(memory, question),
                        prompt=prompt,
                        answer=answer,
                        usage=usage if index == 1 else {},
                        full=audit_full,
                        evidence_budget=2_000,
                    )
                )

    result = {
        "episode_id": episode_id,
        "answer_list": answer_list,
        "reasoning_trace": "",
        "usage": _merge_usage(usages),
    }
    return _with_audit(result, memory, episode, audit_questions, audit_dir, audit_full)


def _with_audit(
    result: dict[str, Any],
    memory: NovaCodeMemory,
    episode: dict[str, Any],
    audit_questions: list[dict[str, Any]],
    audit_dir: str | Path | None,
    audit_full: bool,
) -> dict[str, Any]:
    """Attach the per-episode audit trail to a result record (optional).

    Writes ``<audit_dir>/<episode_id>.json`` and augments ``result`` with
    ``audit_path`` and the compact ``memory`` compression summary so the
    results JSONL alone already shows how much the trajectory was compressed
    and where the audit file for each episode lives.
    """
    if audit_dir is None:
        return result
    record = build_audit_record(
        episode=episode,
        memory=memory,
        questions=audit_questions,
        outcome={
            "answer_list": result["answer_list"],
            "reasoning_trace": result["reasoning_trace"],
            "usage": result.get("usage") or {},
        },
        full=audit_full,
    )
    path = write_audit(record, Path(audit_dir))
    return {**result, "audit_path": str(path), "memory": compact_memory_stats(memory.stats)}


def _query_with_usage(
    provider: LLMProvider,
    prompt: str,
    *,
    max_tokens: int,
    max_retries: int = 4,
    base_delay: float = 3.0,
    max_delay: float = 60.0,
) -> tuple[str, dict[str, Any]]:
    """Call the provider, retrying transient failures and empty responses.

    Returns ``(text, normalized_usage)`` so results records carry per-call
    token and cache-hit accounting alongside the answer.

    The proxy endpoint intermittently drops connections or returns empty text,
    so retries happen on both ``LLMError``/``Exception`` (connection errors,
    timeouts) and on empty responses (the call succeeded but produced nothing).
    Backoff is exponential with jitter, capped at ``max_delay`` seconds, and
    the whole sequence is bounded by ``timeout_seconds`` so a stuck request
    cannot stall the run.
    """
    import random
    import time

    from coding_agent.llm.base import LLMError

    deadline = time.monotonic() + _QUERY_TIMEOUT_SECONDS
    for attempt in range(1, max_retries + 1):
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("query timed out after retries")
            response = provider.chat([Message(role="user", content=prompt)], tools=None)
            text = str(response.text or "")
            if text.strip():
                return text, normalize_usage(response.usage)
            reason = "empty response"
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
        if attempt >= max_retries:
            break
        delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
        delay = delay * (0.5 + random.random() * 0.5)  # jitter
        print(
            f"[ama]     retry {attempt}/{max_retries - 1} after {reason} (waiting {delay:.0f}s)",
            flush=True,
        )
        time.sleep(delay)
    raise LLMError(f"query failed after {max_retries} attempts: {reason}", retryable=True)


def _query(
    provider: LLMProvider,
    prompt: str,
    *,
    max_tokens: int,
    max_retries: int = 4,
    base_delay: float = 3.0,
    max_delay: float = 60.0,
) -> str:
    """Call the provider and return just the text (see ``_query_with_usage``)."""
    text, _ = _query_with_usage(
        provider,
        prompt,
        max_tokens=max_tokens,
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
    )
    return text


def _merge_usage(usages: list[dict[str, Any]]) -> dict[str, Any]:
    """Sum per-call normalized usage into one episode/run-level summary.

    Keeps the same keys as the session-level aggregation
    (``UsageStats.to_dict``) and the eval summary (``aggregate_usage``), so
    the cache hit rate has one definition everywhere.
    """
    logical = sum(int(item.get("logical_input_tokens") or 0) for item in usages)
    cache_hit = sum(int(item.get("cache_hit_tokens") or 0) for item in usages)
    fresh = sum(int(item.get("fresh_processed_input_tokens") or 0) for item in usages)
    output = sum(int(item.get("output_tokens") or 0) for item in usages)
    return {
        "request_count": len(usages),
        "logical_input_tokens": logical,
        "cache_hit_tokens": cache_hit,
        "fresh_processed_input_tokens": fresh,
        "output_tokens": output,
        "cache_hit_rate": cache_hit / logical if logical else 0.0,
    }


def _merge_memory_stats(memories: list[dict[str, Any] | None]) -> dict[str, int]:
    """Sum per-episode compact memory summaries into one run-level view."""
    merged = {"episodes": 0, "pre": 0, "post": 0, "groups": 0, "folded": 0, "model": 0, "fallback": 0}
    for memory in memories:
        if not memory:
            continue
        merged["episodes"] += 1
        merged["pre"] += int(memory.get("pre_fold_tokens") or 0)
        merged["post"] += int(memory.get("post_fold_tokens") or 0)
        merged["groups"] += int(memory.get("groups") or 0)
        merged["folded"] += int(memory.get("groups_folded") or 0)
        merged["model"] += int(memory.get("model_folds") or 0)
        merged["fallback"] += int(memory.get("fallback_folds") or 0)
    return merged


_QUERY_TIMEOUT_SECONDS = 180.0


def _repair_answers(answers: list[str], questions: list[str], mcq_mode: bool) -> list[str]:
    """Pad or truncate a batch answer list to match the question count."""
    if len(answers) < len(questions):
        return answers + [""] * (len(questions) - len(answers))
    return answers[: len(questions)]


def write_results(path: str | Path, results: list[dict[str, Any]]) -> Path:
    """Write result records in AMA-Bench submission format (JSONL).

    ``ensure_ascii=True`` keeps every record on exactly one line even when an
    answer contains exotic characters, so the file can be safely re-read with
    line-based loaders.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")
    return target


def write_episode_result(path: str | Path, episode_id: int, answer_list: list[str]) -> Path:
    """Append-or-replace one episode's answers in the results JSONL.

    Called after every answered question so progress survives crashes, timeouts
    and Ctrl-C.  Re-running the runner later rewrites this episode in place.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps(
        {"episode_id": episode_id, "answer_list": answer_list, "reasoning_trace": ""},
        ensure_ascii=True,
    )
    lines: list[str] = []
    replaced = False
    if target.exists():
        for raw in target.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            try:
                existing = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if existing.get("episode_id") == episode_id:
                lines.append(record)
                replaced = True
            else:
                lines.append(raw)
    if not replaced:
        lines.append(record)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def exact_match_accuracy(
    results: list[dict[str, Any]],
    episodes: list[dict[str, Any]],
    *,
    subset: str,
) -> dict[str, Any]:
    """Exact-match accuracy against ground truth; meaningful for MCQ subsets."""
    ground_truth = {int(episode.get("episode_id", 0)): episode for episode in episodes}
    mcq_mode = subset == "mcq"
    correct = 0
    total = 0
    per_episode: list[dict[str, Any]] = []
    for result in results:
        episode = ground_truth.get(int(result.get("episode_id", 0)), {})
        qa_pairs = episode.get("qa_pairs") or []
        predicted = [str(item) for item in result.get("answer_list", [])]
        episode_correct = 0
        episode_total = 0
        for index, pair in enumerate(qa_pairs):
            truth = str(pair.get("answer", "") or "")
            if not truth or index >= len(predicted):
                continue
            episode_total += 1
            if _answers_equal(predicted[index], truth, mcq_mode=mcq_mode):
                episode_correct += 1
        correct += episode_correct
        total += episode_total
        per_episode.append(
            {"episode_id": int(result.get("episode_id", 0)), "correct": episode_correct, "total": episode_total}
        )
    return {
        "subset": subset,
        "correct": correct,
        "total": total,
        "accuracy": round(correct / total, 4) if total else None,
        "per_episode": per_episode,
    }


def _answers_equal(predicted: str, truth: str, *, mcq_mode: bool) -> bool:
    predicted = predicted.strip()
    truth = truth.strip()
    if mcq_mode:
        import re

        letters = lambda value: sorted(re.findall(r"\(([A-D])\)", value))
        return letters(predicted) == letters(truth)
    return predicted.lower() == truth.lower()


def build_provider(args: argparse.Namespace) -> LLMProvider:
    llm_config = LLMConfig.from_env()
    if args.provider:
        llm_config.provider = args.provider  # type: ignore[assignment]
    if args.model:
        llm_config.model = args.model
    if args.api_key:
        llm_config.api_key = args.api_key
    if args.base_url:
        llm_config.base_url = args.base_url
    if args.max_tokens:
        llm_config.max_tokens = args.max_tokens
    return create_provider(llm_config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Standalone AMA-Bench evaluation of NovaCode memory (no AMA-Bench checkout needed).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Path to the dataset JSONL (e.g. dataset/test/mcq_set.jsonl)")
    parser.add_argument("--subset", choices=["mcq", "openend"], default=None, help="Dataset subset (auto-detected from file name)")
    parser.add_argument("--episode-ids", default=None, help="Comma-separated episode ids to run")
    parser.add_argument("--samples", type=int, default=None, help="Random sample of N episodes (seeded)")
    parser.add_argument("--output", default="results/novacode_results.jsonl", help="Results JSONL output path")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max output tokens per LLM call")
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Answer all questions of an episode in one LLM call (default: one call per question, which is more reliable)",
    )
    parser.add_argument(
        "--audit-dir",
        default=None,
        help="Per-episode audit JSON directory (default: <output parent>/audit)",
    )
    parser.add_argument(
        "--audit-full",
        action="store_true",
        help="Record full pre-fold groups, evidence and prompts in audits (larger files)",
    )
    parser.add_argument(
        "--method-config",
        default=None,
        help="NovaCode memory method config (JSON/YAML), e.g. max_context_tokens / fold_max_attempts / keep_work_dir",
    )
    parser.add_argument(
        "--keep-work-dir",
        action="store_true",
        help="Keep the fold build directory (groups.jsonl) on disk; implied by --audit-full",
    )
    parser.add_argument("--provider", default=None, help="Overrides NOVACODE_PROVIDER")
    parser.add_argument("--model", default=None, help="Overrides NOVACODE_MODEL")
    parser.add_argument("--api-key", default=None, help="Overrides the provider API key")
    parser.add_argument("--base-url", default=None, help="Overrides NOVACODE_BASE_URL")
    parser.add_argument("--no-env", action="store_true", help="Skip .env loading")
    args = parser.parse_args(argv)

    if not args.no_env:
        loaded = load_env_file()
        if loaded is not None:
            print(f"[env] loaded {loaded}")

    subset = args.subset or detect_subset(args.dataset)
    dataset_path = Path(args.dataset)
    if not dataset_path.is_file():
        print(f"error: dataset not found: {dataset_path}", file=sys.stderr)
        return 2
    episodes = load_episodes(dataset_path)
    episode_ids = [int(item) for item in args.episode_ids.split(",") if item.strip()] if args.episode_ids else None
    selected = filter_episodes(episodes, episode_ids=episode_ids, samples=args.samples)
    if not selected:
        print("error: no episodes matched the filter", file=sys.stderr)
        return 2
    print(f"[ama] subset={subset} episodes={len(selected)}")

    audit_dir = Path(args.audit_dir) if args.audit_dir else (Path(args.output).parent / "audit")
    keep_work_dir = bool(args.keep_work_dir or args.audit_full)
    provider = build_provider(args)
    method = NovaCodeMemoryMethod(
        config_path=args.method_config,
        keep_work_dir=keep_work_dir,
        # Standalone runner drives the LLM-assisted fold with the same
        # provider as answering; without this the fold falls back to the
        # deterministic extractor and key_sequences is never produced.
        fold_provider=provider,
    )
    mode = "per-question" if not args.batch else "batch"
    print(f"[ama] mode={mode}")
    print(f"[ama] audit: {audit_dir}" + (" (full)" if args.audit_full else " (compact)"))

    results = []
    for index, episode in enumerate(selected, start=1):
        episode_id = int(episode.get("episode_id", 0))
        result = run_episode(
            method,
            provider,
            episode,
            subset=subset,
            max_tokens=args.max_tokens,
            per_question=not args.batch,
            audit_dir=audit_dir,
            audit_full=args.audit_full,
            progress=(
                (lambda done, total, eid=episode_id: print(
                    f"[ama]   episode {eid}: answered {done}/{total}", flush=True
                ))
                if not args.batch
                else None
            ),
        )
        results.append(result)
        # Persist each completed episode immediately so a crash or Ctrl-C
        # after this point never loses the whole run.
        write_episode_result(args.output, episode_id, result["answer_list"])
        print(f"[ama] episode {episode_id} ({index}/{len(selected)}): {len(result['answer_list'])} answers -> saved")

    output = write_results(args.output, results)
    print(f"[ama] results written to {output}")

    total_usage = _merge_usage([result["usage"] for result in results if result.get("usage")])
    print(
        f"[ama] usage: {total_usage['request_count']} requests, "
        f"{total_usage['logical_input_tokens']} input tokens "
        f"({total_usage['cache_hit_tokens']} cache-hit, {total_usage['fresh_processed_input_tokens']} fresh), "
        f"{total_usage['output_tokens']} output, cache hit rate {total_usage['cache_hit_rate'] * 100:.1f}%"
    )

    memory_summary = _merge_memory_stats([result.get("memory") for result in results])
    if memory_summary["episodes"]:
        ratio = (
            memory_summary["post"] / memory_summary["pre"] * 100
            if memory_summary["pre"]
            else 0.0
        )
        print(
            f"[ama] memory: pre={memory_summary['pre']} tok, post={memory_summary['post']} tok, "
            f"residual {ratio:.1f}%, folded {memory_summary['folded']}/{memory_summary['groups']} groups "
            f"({memory_summary['model']} model, {memory_summary['fallback']} fallback folds)"
        )
    print(f"[ama] audit files written to {audit_dir}")

    stats = exact_match_accuracy(results, selected, subset=subset)
    if stats["accuracy"] is None:
        print("[ama] no ground-truth answers in the dataset; skipping local accuracy")
    else:
        print(f"[ama] exact-match accuracy: {stats['accuracy']} ({stats['correct']}/{stats['total']})")
        print("[ama] note: official scoring uses AMA-Bench's LLM-as-judge on the same results file")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
