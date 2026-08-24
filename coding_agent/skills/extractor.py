"""Grounded, deterministic dual-granularity candidate extraction."""
from __future__ import annotations

import re
from typing import Any

from .candidates import SkillCandidate, SkillRule


def _safe_name(value: str, fallback: str) -> str:
    words = re.findall(r"[a-z0-9]+", value.lower())
    name = "_".join(words[:6]).strip("_")
    if len(name) < 2 or not name[0].isalpha():
        name = fallback
    return name[:64]


def _deidentify(value: str) -> str:
    text = re.sub(r"(?<!\w)(?:[A-Za-z]:[\\/]|/)[^\s'\"]+", "<PATH>", value)
    text = re.sub(
        r"(?i)\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*[^\s,;]+",
        "<REDACTED>",
        text,
    )
    return text


class SkillExtractor:
    """Extract only evidence-backed rules; never writes TaskState or ToolState."""

    def extract_event_driven(
        self,
        *,
        episode_id: str,
        outcome_version: int,
        failure: dict[str, Any],
        fix: dict[str, Any],
        verification: dict[str, Any],
    ) -> SkillCandidate | None:
        if bool(failure.get("success", True)) or not bool(verification.get("success", False)):
            return None
        refs = [
            str(item.get("artifact_id") or item.get("call_id") or item.get("ref") or "")
            for item in (failure, fix, verification)
        ]
        if any(not ref for ref in refs):
            return None
        trigger = _deidentify(
            str(failure.get("error") or failure.get("output_preview") or "tool failure")
        )
        action = _deidentify(
            str(fix.get("summary") or fix.get("name") or "apply the verified corrective action")
        )
        verify = _deidentify(
            str(verification.get("summary") or verification.get("name") or "repeat verification")
        )
        base = str(failure.get("name") or "tool") + "_failure_resolver"
        return SkillCandidate.new(
            episode_id=episode_id,
            outcome_version=outcome_version,
            name=_safe_name(base, "verified_failure_resolver"),
            title=f"Resolve recurring {failure.get('name', 'tool')} failures",
            granularity="event-driven",
            when_to_apply=f"When the same failure pattern recurs: {trigger[:240]}",
            constraints_and_style=["Do not claim success until the original verification is repeated."],
            workflow_rules=[
                SkillRule(f"Inspect and reproduce the failure before changing state: {trigger[:300]}", [refs[0]]),
                SkillRule(f"Apply the smallest verified corrective action: {action[:300]}", [refs[1]]),
                SkillRule(f"Repeat the matching verification and require success: {verify[:300]}", [refs[2]]),
            ],
            evidence_refs=refs,
            source="verified_event",
        )

    def extract_task_level(
        self,
        *,
        episode_id: str,
        outcome_version: int,
        objective: str,
        summary: str,
        evidence_refs: list[str],
        episode_succeeded: bool,
    ) -> SkillCandidate | None:
        if not episode_succeeded or not evidence_refs or not summary.strip():
            return None
        safe_objective = _deidentify(objective)
        safe_summary = _deidentify(summary)
        return SkillCandidate.new(
            episode_id=episode_id,
            outcome_version=outcome_version,
            name=_safe_name(safe_objective, "verified_task_workflow"),
            title=f"Verified workflow for {safe_objective[:100]}",
            granularity="task-level",
            when_to_apply=f"When a task has the same transferable objective family: {safe_objective[:240]}",
            constraints_and_style=["Verify workspace changes after the final modification."],
            workflow_rules=[SkillRule(safe_summary[:1000], list(dict.fromkeys(evidence_refs)))],
            evidence_refs=list(dict.fromkeys(evidence_refs)),
            source="episode_succeeded",
            episode_succeeded=True,
        )
