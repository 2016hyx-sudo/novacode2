"""Stable JSONL/JSON/Markdown reporting for structured-context evaluations."""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .metrics import compare_baseline, summarize_metrics
from .schema import GateResult, RequestMetric, RunResult


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _coerce_metric(value: RequestMetric | Mapping[str, Any]) -> RequestMetric:
    return value if isinstance(value, RequestMetric) else RequestMetric.from_dict(value)


def _coerce_result(value: RunResult | Iterable[RequestMetric | Mapping[str, Any]]) -> RunResult:
    if isinstance(value, RunResult):
        return value
    return RunResult(
        run_id="adhoc",
        mode="offline",
        requests=tuple(_coerce_metric(item) for item in value),
    )


def _atomic_write(path: Path, data: bytes) -> None:
    """Write a complete replacement in the target directory before rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _percent(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def _bucket_rows(requests: Sequence[RequestMetric]) -> list[tuple[str, int, int | float | None, int | float | None]]:
    grouped: dict[str, list[RequestMetric]] = {}
    for request in requests:
        if request.agent_role != "main" or request.status != "success":
            continue
        grouped.setdefault(request.bucket or "(unbucketed)", []).append(request)
    rows: list[tuple[str, int, int | float | None, int | float | None]] = []
    for bucket, items in sorted(grouped.items()):
        replay_rows = [item for item in items if item.raw_full_tokens and item.structured_tokens]
        if replay_rows:
            structured = sorted(item.structured_tokens for item in replay_rows)
            raw = sorted(item.raw_full_tokens for item in replay_rows)
        else:
            structured = sorted(
                item.effective_logical_input_tokens for item in items if item.variant == "structured"
            )
            raw = sorted(
                item.effective_logical_input_tokens for item in items if item.variant == "raw_full"
            )
        raw_index = max(0, (95 * len(raw) + 99) // 100 - 1)
        structured_index = max(0, (95 * len(structured) + 99) // 100 - 1)
        rows.append(
            (
                bucket,
                len(items),
                raw[raw_index] if raw else None,
                structured[structured_index] if structured else None,
            )
        )
    return rows


def render_report(summary: Mapping[str, Any], requests: Sequence[RequestMetric]) -> str:
    """Render a deterministic human-readable report without timestamps."""
    context = dict(summary.get("context") or {})
    input_tokens = dict(summary.get("input_tokens") or {})
    structured = dict(input_tokens.get("structured") or {})
    raw = dict(input_tokens.get("raw_full") or {})
    usage = dict(summary.get("usage") or {})
    sample = dict(summary.get("sample") or {})
    quality = dict(summary.get("quality") or {})
    gates = dict(summary.get("gates") or {})
    lines = [
        "# Structured Context Evaluation",
        "",
        f"Mode: {summary.get('mode', 'unknown')}",
        f"Run: {summary.get('run_id', 'unknown')}",
        "",
        "## Context",
        "",
        f"- CRR weighted: {_percent(context.get('crr_weighted'))}",
        f"- Net CRR: {_percent(context.get('net_crr'))}",
        f"- Tool reduction: {_percent(context.get('tool_reduction'))}",
        f"- Fold incremental: {_percent(context.get('fold_incremental'))}",
        f"- Task median CRR: {_percent(context.get('crr_task_median'))}",
        f"- Negative reduction rate: {_percent(context.get('negative_reduction_rate'))}",
        "",
        "## Input tokens",
        "",
        "| variant | p50 | p95 | max | count |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| raw_full | {_number(raw.get('p50'))} | {_number(raw.get('p95'))} | {_number(raw.get('max'))} | {_number(raw.get('count'))} |",
        f"| structured | {_number(structured.get('p50'))} | {_number(structured.get('p95'))} | {_number(structured.get('max'))} | {_number(structured.get('count'))} |",
        "",
        "## Usage",
        "",
        f"- Requests: {_number(usage.get('request_count'))}",
        f"- Logical input tokens: {_number(usage.get('logical_input_tokens'))}",
        f"- Cache-hit tokens: {_number(usage.get('cache_hit_tokens'))}",
        f"- Fresh input tokens: {_number(usage.get('fresh_processed_input_tokens'))}",
        f"- Output tokens: {_number(usage.get('output_tokens'))}",
        f"- Cache hit rate: {_percent(usage.get('cache_hit_rate'))}",
        "",
        "## Sample",
        "",
        f"- Tasks: {_number(sample.get('tasks'))}",
        f"- Successful main requests: {_number(sample.get('main_requests'))}",
        f"- Failed main requests: {_number(sample.get('failed_requests'))}",
        f"- Estimated-usage requests: {_number(sample.get('estimated_usage_count'))}",
        "",
        "## Quality",
        "",
        f"- Completed rate: {_percent(quality.get('completed_rate'))}",
        f"- Oracle pass rate: {_percent(quality.get('oracle_pass_rate'))}",
        "",
        "## Buckets",
        "",
        "| bucket | requests | raw p95 | structured p95 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for bucket, count, raw_p95, structured_p95 in _bucket_rows(requests):
        lines.append(f"| {bucket} | {count} | {_number(raw_p95)} | {_number(structured_p95)} |")
    lines.extend(["", "## Largest structured requests", "", "| case | step | raw | observation | structured | epoch | hash |", "| --- | ---: | ---: | ---: | ---: | ---: | --- |"])
    ranked = sorted(
        (
            request
            for request in requests
            if request.agent_role == "main"
            and request.status == "success"
            and request.variant == "structured"
        ),
        key=lambda item: (-item.structured_input_tokens, item.sort_key()),
    )[:20]
    for request in ranked:
        short_hash = request.payload_hashes.get("structured", "")
        lines.append(
            "| {case} | {step} | {raw} | {observation} | {structured} | {epoch} | {payload} |".format(
                case=request.task_case_id,
                step=request.step,
                raw=request.raw_full_tokens,
                observation=request.observation_full_tokens,
                structured=request.structured_input_tokens,
                epoch=request.epoch_id,
                payload=short_hash,
            )
        )
    lines.extend(["", "## Gates", "", f"- Passed: {bool(gates.get('passed', True))}"])
    for failure in gates.get("failures") or []:
        lines.append(f"- Failure: {_canonical_json(failure)}")
    return "\n".join(lines) + "\n"


def _load_baseline(value: Any) -> Any:
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_dir():
            requests_path = path / "requests.jsonl"
            if requests_path.is_file():
                return load_requests(requests_path)
            path = path / "summary.json"
        if path.suffix == ".jsonl":
            return load_requests(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            return payload
        raise ValueError("baseline JSON must be an object")
    return value


def build_summary(
    result: RunResult | Iterable[RequestMetric | Mapping[str, Any]],
    *,
    baseline: Any | None = None,
) -> dict[str, Any]:
    """Aggregate a result and optionally attach a pure baseline gate outcome."""
    run = _coerce_result(result)
    summary = summarize_metrics(run.sorted_requests(), mode=run.mode)
    summary["run_id"] = run.run_id
    summary["manifest"] = dict(run.manifest)
    summary["task_count"] = len(run.tasks)
    if run.mode == "full-run" and run.tasks:
        completed = sum(1 for task in run.tasks if task.get("status") == "completed")
        passed = sum(1 for task in run.tasks if bool(task.get("passed")))
        summary["quality"] = {
            "completed_rate": completed / len(run.tasks),
            "oracle_pass_rate": passed / len(run.tasks),
        }
    if baseline is None:
        gate = GateResult(True)
    else:
        baseline_value = _load_baseline(baseline)
        # Row-level baseline data is preferred for deterministic PR replay.
        if isinstance(baseline_value, Mapping) and not any(
            key in baseline_value for key in ("requests", "request_metrics")
        ):
            gate = compare_baseline(summary, baseline_value)
        else:
            gate = compare_baseline(run, baseline_value)
    summary["gates"] = gate.to_dict()
    return summary


def write_report(
    result: RunResult | Iterable[RequestMetric | Mapping[str, Any]],
    output_dir: Path | str,
    *,
    baseline: Any | None = None,
) -> dict[str, Path]:
    """Atomically write ``requests.jsonl``, ``summary.json`` and ``report.md``.

    Sorting and canonical JSON make repeated writes of the same result
    byte-for-byte identical.  Only the three fixed filenames are written.
    """
    run = _coerce_result(result)
    directory = Path(output_dir)
    if directory.exists() and not directory.is_dir():
        raise ValueError(f"output path is not a directory: {directory}")
    requests = run.sorted_requests()
    summary = build_summary(run, baseline=baseline)
    jsonl = "".join(_canonical_json(request.to_dict()) + "\n" for request in requests).encode("utf-8")
    summary_bytes = (_canonical_json(summary) + "\n").encode("utf-8")
    markdown = render_report(summary, requests).encode("utf-8")
    paths = {
        "requests": directory / "requests.jsonl",
        "summary": directory / "summary.json",
        "report": directory / "report.md",
    }
    _atomic_write(paths["requests"], jsonl)
    _atomic_write(paths["summary"], summary_bytes)
    _atomic_write(paths["report"], markdown)
    return paths


def load_requests(path: Path | str) -> list[RequestMetric]:
    """Read a requests JSONL file written by :func:`write_report`."""
    result: list[RequestMetric] = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSONL at line {number}") from exc
        if not isinstance(value, Mapping):
            raise ValueError(f"JSONL line {number} is not an object")
        result.append(RequestMetric.from_dict(value))
    return result


write_run_report = write_report
