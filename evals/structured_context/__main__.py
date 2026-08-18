"""Command line entry points for structured-context evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .full_context import run_offline_replay
from .metrics import compare_baseline
from .report import load_requests, write_report


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_comparison_input(path: str | Path) -> Any:
    source = Path(path)
    if source.is_dir():
        requests = source / "requests.jsonl"
        if requests.is_file():
            return load_requests(requests)
        source = source / "summary.json"
    if source.suffix == ".jsonl":
        return load_requests(source)
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"comparison JSON must be an object: {source}")
    return value


def _offline(args: argparse.Namespace) -> int:
    from .report import build_summary

    result = run_offline_replay(args.cases, min_requests=args.min_requests)
    paths = write_report(result, args.output, baseline=args.baseline)
    summary = build_summary(result, baseline=args.baseline)
    gate_passed = bool((summary.get("gates") or {}).get("passed"))
    sample = dict(summary.get("sample") or {})
    print(_json({"mode": result.mode, "main_requests": sample.get("main_requests", 0), "fold_requests": sample.get("fold_requests", 0), "gate_passed": gate_passed, "output": {key: str(value) for key, value in paths.items()}}))
    return 0 if gate_passed else 1


def _run(args: argparse.Namespace) -> int:
    from .report import build_summary
    from .runner import ContextEvaluationRunner, RunnerConfig, ScriptedProvider
    from .scenarios import load_scenario_suite

    suite = load_scenario_suite()
    scenarios = suite.filter(ids=args.scenario or ("short-01",), buckets=args.bucket)
    if not scenarios:
        raise ValueError("scenario filter selected no tasks")
    if not args.scripted:
        raise ValueError(
            "the CLI never constructs a live provider; use --scripted for the offline smoke run "
            "or inject provider_factory through ContextEvaluationRunner"
        )
    if any(item.id != "short-01" for item in scenarios):
        raise ValueError("the built-in scripted dialogue is defined only for scenario short-01")
    runner = ContextEvaluationRunner(
        config=RunnerConfig(work_root=args.work_root, keep_workspaces=args.keep_workspaces),
        provider_factory=ScriptedProvider.inclusive_total_smoke,
    )
    result = runner.run_many(scenarios, variants=args.variant or ("structured",))
    paths = write_report(result, args.output, baseline=args.baseline)
    summary = build_summary(result, baseline=args.baseline)
    print(_json({"mode": result.mode, "passed": result.metadata.get("passed", False), "summary": summary, "output": {key: str(value) for key, value in paths.items()}}))
    gate_passed = bool((summary.get("gates") or {}).get("passed"))
    return 0 if result.metadata.get("passed", False) and gate_passed else 1


def _compare(args: argparse.Namespace) -> int:
    current = _load_comparison_input(args.result)
    baseline = _load_comparison_input(args.baseline)
    gate = compare_baseline(current, baseline)
    payload = gate.to_dict()
    if not args.verbose:
        payload = {
            "passed": gate.passed,
            "check_count": len(gate.checks),
            "failures": [dict(item) for item in gate.failures],
        }
    print(_json(payload))
    return 0 if gate.passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NovaCode structured-context evaluation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    offline = subparsers.add_parser("offline", help="run deterministic three-way replay")
    offline.add_argument("--cases", type=Path, default=None, help="offline_cases.json override")
    offline.add_argument("--output", type=Path, default=Path(".eval-results/offline"))
    offline.add_argument("--baseline", type=Path, default=None)
    offline.add_argument("--min-requests", type=int, default=200)
    offline.set_defaults(handler=_offline)

    run = subparsers.add_parser("run", help="run a complete isolated scripted conversation")
    run.add_argument("--scripted", action="store_true", help="use the deterministic no-network provider")
    run.add_argument("--scenario", action="append", default=None)
    run.add_argument("--bucket", action="append", default=[])
    run.add_argument("--variant", action="append", choices=("raw_full", "observation_full", "structured"), default=None)
    run.add_argument("--output", type=Path, default=Path(".eval-results/scripted"))
    run.add_argument("--work-root", type=Path, default=None)
    run.add_argument("--keep-workspaces", action="store_true")
    run.add_argument("--baseline", type=Path, default=None)
    run.set_defaults(handler=_run)

    compare = subparsers.add_parser("compare", help="compare request rows or summaries")
    compare.add_argument("--result", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--verbose", action="store_true")
    compare.set_defaults(handler=_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
