"""Static tool protocol for explicit per-turn task disposition."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from ..runtime.episode import CriterionEvidence, TurnDisposition
from .base import ToolErrorKind, ToolResult


class ReportTaskOutcomeTool:
    name = "report_task_outcome"
    description = (
        "Report this turn's disposition for the current multi-turn task episode. "
        "completion_proposed requests deterministic completion review; it never directly marks success."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "disposition": {
                "type": "string",
                "enum": ["continue", "waiting_user", "completion_proposed", "blocked"],
            },
            "summary": {"type": "string"},
            "open_questions": {"type": "array", "items": {"type": "string"}},
            "criteria_evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "criterion_id": {"type": "string"},
                        "evidence_refs": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["criterion_id", "evidence_refs"],
                },
            },
        },
        "required": ["disposition", "summary", "open_questions", "criteria_evidence"],
    }

    def __init__(self, submit: Callable[[TurnDisposition], None]) -> None:
        self._submit = submit

    def execute(
        self,
        disposition: str,
        summary: str,
        open_questions: list[str],
        criteria_evidence: list[dict[str, Any]],
    ) -> ToolResult:
        if disposition not in {"continue", "waiting_user", "completion_proposed", "blocked"}:
            return self._invalid("invalid disposition")
        if not isinstance(summary, str) or not summary.strip():
            return self._invalid("summary must not be empty")
        if not isinstance(open_questions, list) or not all(isinstance(x, str) for x in open_questions):
            return self._invalid("open_questions must be a list of strings")
        if not isinstance(criteria_evidence, list) or not all(isinstance(x, dict) for x in criteria_evidence):
            return self._invalid("criteria_evidence must be a list of objects")
        parsed = [CriterionEvidence.from_dict(item) for item in criteria_evidence]
        if any(not item.criterion_id for item in parsed):
            return self._invalid("every criteria_evidence item needs criterion_id")
        if len({item.criterion_id for item in parsed}) != len(parsed):
            return self._invalid("criterion_id entries must be unique")
        report = TurnDisposition(
            disposition=disposition,  # type: ignore[arg-type]
            summary=summary.strip(),
            open_questions=[item.strip() for item in open_questions if item.strip()],
            criteria_evidence=parsed,
        )
        self._submit(report)
        return ToolResult.ok(
            "Task outcome recorded. Continue to a concise user-facing response; the harness will apply the completion gate.",
            disposition=disposition,
        )

    @staticmethod
    def _invalid(message: str) -> ToolResult:
        return ToolResult.fail(message, metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value})
