"""LLM-as-judge evaluation for NovaCode AMA-Bench results (self-contained).

Faithfully reimplements AMA-Bench's official judge so scores are comparable to
the leaderboard:

* the judge prompt is byte-identical to ``utils/evaluation_metrics.py``;
* the answer is the *last* ``yes``/``no`` token in the judge response (thinking
  tags stripped first), falling back to token-level F1 when neither appears;
* summary statistics (overall / by domain / by qa type) match ``evaluate.py``.

The only difference: the judge runs on NovaCode's own LLM provider (same
``.env`` / ``NOVACODE_*`` configuration as everything else), so no AMA-Bench
checkout, ``ModelClient`` or YAML config is required.

Usage:

    python -m ama_bench.judge \\
        --answers-file results/novacode_results.jsonl \\
        --test-file dataset/test/open_end_qa_set.jsonl \\
        --output-file results/evaluation.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from coding_agent.llm import LLMProvider, Message, create_provider
from config import LLMConfig, load_env_file

from .io import iter_jsonl_records

# --------------------------------------------------------------------------- prompt


def _judge_prompt(
    question: str,
    golden_answer: str,
    predicted_answer: str,
    *,
    task_description: str = "",
    task_type: str = "",
    episode_id: str = "",
) -> str:
    context_parts = []
    if task_type:
        context_parts.append(f"Task Type: {task_type}")
    if episode_id:
        context_parts.append(f"Episode ID: {episode_id}")
    if task_description:
        context_parts.append(f"Task Context: {task_description}")
    context_str = "\n".join(context_parts) if context_parts else ""
    return f"""You are an expert evaluator. You will be given a question, a reference answer, and a predicted answer.
Your task is to determine if the predicted answer is correct based on:
1. Factual correctness compared to the reference
2. Completeness of the answer
3. Relevance to the question

{context_str}

Question: {question}

Reference Answer: {golden_answer}

Predicted Answer: {predicted_answer}

Is the predicted answer correct? Respond with ONLY "yes" or "no". Do not include any thinking process, explanation, or additional text.

Answer:<think></think>"""


# --------------------------------------------------------------------------- lexical helpers (official semantics)


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, remove a/an/the, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def compute_f1_score(predicted: str, golden: str) -> float:
    """Token-level F1, exactly as AMA-Bench computes it (fallback metric)."""
    pred_tokens = normalize_text(predicted).split()
    gold_tokens = normalize_text(golden).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    return 2 * (precision * recall) / (precision + recall)


# --------------------------------------------------------------------------- judge call


def llm_as_judge(
    question: str,
    golden_answer: str,
    predicted_answer: str,
    provider: LLMProvider,
    *,
    task_description: str = "",
    task_type: str = "",
    episode_id: str = "",
    max_tokens: int = 2048,
) -> float:
    """Ask the judge whether the prediction is correct; returns 1.0 or 0.0."""
    prompt = _judge_prompt(
        question,
        golden_answer,
        predicted_answer,
        task_description=task_description,
        task_type=task_type,
        episode_id=episode_id,
    )
    response = provider.chat([Message(role="user", content=prompt)], tools=None)
    raw = str(response.text or "")
    cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()
    lowered = cleaned.lower()
    yes_matches = list(re.finditer(r"\byes\b", lowered))
    no_matches = list(re.finditer(r"\bno\b", lowered))
    last_yes = yes_matches[-1].start() if yes_matches else -1
    last_no = no_matches[-1].start() if no_matches else -1
    if last_yes > last_no:
        return 1.0
    if last_no > last_yes:
        return 0.0
    return compute_f1_score(predicted_answer, golden_answer)


# --------------------------------------------------------------------------- batch evaluation


def build_qa_results(
    results: Sequence[dict[str, Any]],
    episodes: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Match predicted answers to golden answers, mirroring evaluate_from_files."""
    original = {episode.get("episode_id"): episode for episode in episodes}
    qa_results: list[dict[str, Any]] = []
    for episode in results:
        episode_id = episode.get("episode_id")
        answer_list = episode.get("answer_list") or []
        source = original.get(episode_id, {})
        qa_pairs = source.get("qa_pairs") or []
        for predicted, pair in zip(answer_list, qa_pairs):
            qa_results.append(
                {
                    "episode_id": episode_id,
                    "task_type": source.get("task_type", "unknown"),
                    "domain": source.get("domain", "unknown"),
                    "task_description": source.get("task", ""),
                    "question": pair.get("question", ""),
                    "golden_answer": pair.get("answer", ""),
                    "predicted_answer": predicted,
                    "qa_type": pair.get("type") or "unknown",
                }
            )
    return qa_results


def evaluate_results(
    qa_results: Sequence[dict[str, Any]],
    provider: LLMProvider,
    *,
    judge_model: str = "",
    max_workers: int = 3,
    max_tokens: int = 2048,
    progress: Callable[[int], None] | None = None,
) -> dict[str, Any]:
    """Judge every QA pair and aggregate statistics (official summary shape)."""
    evaluated: list[dict[str, Any]] = []

    def judge_one(result: dict[str, Any]) -> dict[str, Any]:
        score = llm_as_judge(
            result["question"],
            result["golden_answer"],
            result["predicted_answer"],
            provider,
            task_description=result.get("task_description", ""),
            task_type=result.get("task_type", ""),
            episode_id=str(result.get("episode_id", "")),
            max_tokens=max_tokens,
        )
        item = dict(result)
        item["score"] = score
        return item

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(judge_one, result) for result in qa_results]
        for index, future in enumerate(as_completed(futures)):
            evaluated.append(future.result())
            if progress is not None:
                progress(index + 1)

    def group_stats(grouped: dict[str, list[float]]) -> dict[str, dict[str, Any]]:
        return {
            key: {
                "count": len(scores),
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "accuracy": sum(1 for s in scores if s == 1.0) / len(scores) if scores else 0,
            }
            for key, scores in grouped.items()
        }

    by_task_type: dict[str, list[float]] = {}
    by_domain: dict[str, list[float]] = {}
    by_qa_type: dict[str, list[float]] = {}
    for item in evaluated:
        by_task_type.setdefault(item.get("task_type", "unknown"), []).append(item["score"])
        by_domain.setdefault(item.get("domain", "unknown"), []).append(item["score"])
        by_qa_type.setdefault(item.get("qa_type", "unknown"), []).append(item["score"])

    return {
        "config": {"judge_model": judge_model or ""},
        "overall": {
            "total_questions": len(evaluated),
            "avg_score": sum(item["score"] for item in evaluated) / len(evaluated) if evaluated else 0,
            "accuracy": sum(1 for item in evaluated if item["score"] == 1.0) / len(evaluated) if evaluated else 0,
        },
        "by_task_type": group_stats(by_task_type),
        "by_domain": group_stats(by_domain),
        "by_qa_type": group_stats(by_qa_type),
        "results": evaluated,
    }


def print_summary(summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    print("\nOverall Performance:")
    print(f"  Total questions: {overall['total_questions']}")
    print(f"  Average score: {overall['avg_score']:.4f}")
    print(f"  Accuracy: {overall['accuracy']:.4f}")
    print("\nBy Domain:")
    for domain, stats in sorted(summary.get("by_domain", {}).items()):
        print(f"  {domain}:")
        print(f"    Accuracy: {stats['accuracy']:.4f} ({stats['count']} questions)")
    print("\nBy QA Type:")
    for qa_type, stats in sorted(summary.get("by_qa_type", {}).items()):
        print(f"  Type {qa_type}:")
        print(f"    Accuracy: {stats['accuracy']:.4f} ({stats['count']} questions)")
    print("=" * 70)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl_records(path))


def judge_config_from_yaml(path: str | Path | None) -> dict[str, Any]:
    """Read AMA-Bench-style judge YAML (provider/model/api_key/base_url/max_tokens)."""
    if not path:
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "--judge-config needs pyyaml; install it or use --provider/--model/--api-key instead"
        ) from exc
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return {}
    return {
        key: data[key]
        for key in ("provider", "model", "api_key", "base_url", "max_tokens")
        if data.get(key) is not None
    }


def build_provider(args: argparse.Namespace, yaml_config: dict[str, Any]) -> LLMProvider:
    llm_config = LLMConfig.from_env()
    provider = args.provider or yaml_config.get("provider")
    model = args.model or yaml_config.get("model")
    api_key = args.api_key or yaml_config.get("api_key")
    base_url = args.base_url or yaml_config.get("base_url")
    max_tokens = args.max_tokens or yaml_config.get("max_tokens")
    if provider:
        llm_config.provider = provider  # type: ignore[assignment]
    if model:
        llm_config.model = model
    if api_key:
        llm_config.api_key = api_key
    if base_url:
        llm_config.base_url = base_url
    if max_tokens:
        llm_config.max_tokens = int(max_tokens)
    return create_provider(llm_config)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="LLM-as-judge evaluation of NovaCode AMA-Bench results (self-contained).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--answers-file", required=True, help="Results JSONL produced by ama_bench.run")
    parser.add_argument("--test-file", required=True, help="Original dataset JSONL (ground truth)")
    parser.add_argument("--judge-config", default=None, help="Optional AMA-style judge YAML (provider/model/api_key/...)")
    parser.add_argument("--output-file", default=None, help="Write the full evaluation JSON here")
    parser.add_argument("--max-workers", type=int, default=3, help="Concurrent judge calls")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens per judge call")
    parser.add_argument("--provider", default=None, help="Overrides judge provider")
    parser.add_argument("--model", default=None, help="Overrides judge model")
    parser.add_argument("--api-key", default=None, help="Overrides judge API key")
    parser.add_argument("--base-url", default=None, help="Overrides judge base URL")
    parser.add_argument("--no-env", action="store_true", help="Skip .env loading")
    args = parser.parse_args(argv)

    answers_path = Path(args.answers_file)
    test_path = Path(args.test_file)
    if not answers_path.is_file():
        print(f"error: answers file not found: {answers_path}", file=sys.stderr)
        return 2
    if not test_path.is_file():
        print(f"error: test file not found: {test_path}", file=sys.stderr)
        return 2
    if not args.no_env:
        loaded = load_env_file()
        if loaded is not None:
            print(f"[env] loaded {loaded}")

    yaml_config = judge_config_from_yaml(args.judge_config)
    provider = build_provider(args, yaml_config)
    judge_model = args.model or yaml_config.get("model") or getattr(provider, "model", "")
    print(f"[judge] provider={getattr(provider, 'provider_name', getattr(provider, 'config', None) and getattr(provider.config, 'provider', '?'))} model={judge_model}")

    qa_results = build_qa_results(load_jsonl(answers_path), load_jsonl(test_path))
    if not qa_results:
        print("error: no QA pairs to evaluate (answer list may not match qa_pairs)", file=sys.stderr)
        return 2
    print(f"[judge] evaluating {len(qa_results)} QA pairs (max_workers={args.max_workers})")

    summary = evaluate_results(
        qa_results,
        provider,
        judge_model=str(judge_model),
        max_workers=max(args.max_workers, 1),
        max_tokens=args.max_tokens,
        progress=lambda done: print(f"[judge] {done}/{len(qa_results)}", end="\r"),
    )
    if args.output_file:
        output = Path(args.output_file)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[judge] results written to {output}")
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
