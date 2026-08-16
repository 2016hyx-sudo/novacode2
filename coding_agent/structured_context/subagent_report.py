"""Structured report parsing and rendering for bounded subagents."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

_FENCE_RE = re.compile(r"```(?:json)?\s*|\s*```")


@dataclass
class SubagentReport:
    summary: str = ""
    findings: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "findings": [dict(item) for item in self.findings],
            "evidence": [dict(item) for item in self.evidence],
            "blockers": [str(item) for item in self.blockers],
            "next_action": self.next_action,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> SubagentReport:
        data = data or {}
        return cls(
            summary=str(data.get("summary") or ""),
            findings=[dict(item) for item in data.get("findings") or [] if isinstance(item, dict)],
            evidence=[dict(item) for item in data.get("evidence") or [] if isinstance(item, dict)],
            blockers=[str(item) for item in data.get("blockers") or []],
            next_action=str(data.get("next_action") or ""),
        )

    def render(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


REPORT_TEMPLATE = """Return a single JSON object with this schema:
{
  "summary": "one concise self-contained summary",
  "findings": [{"fact": "...", "evidence": ["path or artifact reference"]}],
  "evidence": [{"type": "file|command|artifact", "path": "...", "value": "..."}],
  "blockers": ["anything that prevented completion"],
  "next_action": "what the main agent should do next"
}
Do not include markdown fences around the JSON."""


def parse_subagent_report(text: str | None) -> SubagentReport:
    """Parse a subagent final answer into a structured report.

    If parsing or schema validation fails, the complete text is preserved in
    ``summary`` and parsing is considered degraded rather than fatal.
    """
    clean = (text or "").strip()
    if not clean:
        return SubagentReport(summary="(subagent returned no output)")

    candidate = _FENCE_RE.sub("", clean).strip()
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
        if isinstance(data, dict) and "summary" in data:
            report = SubagentReport.from_dict(data)
            if report.summary:
                return report
    except json.JSONDecodeError:
        pass

    return SubagentReport(summary=clean)
