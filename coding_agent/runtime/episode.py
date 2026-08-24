"""Multi-turn task episodes and the deterministic completion gate."""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

EpisodeStatus = Literal[
    "active",
    "waiting_user",
    "completion_pending",
    "succeeded",
    "blocked",
    "cancelled",
    "superseded",
    "dormant",
]
Disposition = Literal["continue", "waiting_user", "completion_proposed", "blocked"]


@dataclass
class CriterionEvidence:
    criterion_id: str
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"criterion_id": self.criterion_id, "evidence_refs": list(self.evidence_refs)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CriterionEvidence:
        return cls(
            criterion_id=str(data.get("criterion_id", "")).strip(),
            evidence_refs=[str(ref) for ref in data.get("evidence_refs") or [] if str(ref)],
        )


@dataclass
class TurnDisposition:
    disposition: Disposition
    summary: str
    open_questions: list[str] = field(default_factory=list)
    criteria_evidence: list[CriterionEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "disposition": self.disposition,
            "summary": self.summary,
            "open_questions": list(self.open_questions),
            "criteria_evidence": [item.to_dict() for item in self.criteria_evidence],
        }


@dataclass
class SuccessCriterion:
    criterion_id: str
    description: str
    required: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "description": self.description,
            "required": self.required,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuccessCriterion:
        return cls(
            criterion_id=str(data.get("criterion_id", "")),
            description=str(data.get("description", "")),
            required=bool(data.get("required", True)),
        )


@dataclass
class TaskEpisode:
    episode_id: str
    session_id: str
    objective: str
    status: EpisodeStatus = "active"
    started_turn: int = 1
    completed_turn: int | None = None
    start_event_seq: int = 0
    success_criteria: list[SuccessCriterion] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    evidence_refs: list[str] = field(default_factory=list)
    outcome_version: int = 1
    last_summary: str = ""

    @classmethod
    def new(
        cls,
        *,
        session_id: str,
        objective: str,
        started_turn: int = 1,
        start_event_seq: int = 0,
        success_criteria: list[str] | None = None,
    ) -> TaskEpisode:
        criteria = [
            SuccessCriterion(f"criterion-{index}", text)
            for index, text in enumerate(success_criteria or [], 1)
        ]
        return cls(
            episode_id=f"ep-{uuid.uuid4().hex[:16]}",
            session_id=session_id,
            objective=objective.strip(),
            started_turn=started_turn,
            start_event_seq=start_event_seq,
            success_criteria=criteria,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "objective": self.objective,
            "status": self.status,
            "started_turn": self.started_turn,
            "completed_turn": self.completed_turn,
            "start_event_seq": self.start_event_seq,
            "success_criteria": [item.to_dict() for item in self.success_criteria],
            "open_questions": list(self.open_questions),
            "unresolved": list(self.unresolved),
            "evidence_refs": list(self.evidence_refs),
            "outcome_version": self.outcome_version,
            "last_summary": self.last_summary,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TaskEpisode:
        return cls(
            episode_id=str(data["episode_id"]),
            session_id=str(data["session_id"]),
            objective=str(data.get("objective", "")),
            status=str(data.get("status", "active")),  # type: ignore[arg-type]
            started_turn=int(data.get("started_turn", 1)),
            completed_turn=(int(data["completed_turn"]) if data.get("completed_turn") is not None else None),
            start_event_seq=int(data.get("start_event_seq", 0)),
            success_criteria=[SuccessCriterion.from_dict(x) for x in data.get("success_criteria") or []],
            open_questions=[str(x) for x in data.get("open_questions") or []],
            unresolved=[str(x) for x in data.get("unresolved") or []],
            evidence_refs=[str(x) for x in data.get("evidence_refs") or []],
            outcome_version=int(data.get("outcome_version", 1)),
            last_summary=str(data.get("last_summary", "")),
        )


@dataclass(frozen=True)
class CompletionGateResult:
    passed: bool
    reasons: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


class EpisodeController:
    """Own episode transitions without mutating TaskState or ToolState."""

    _CORRECTION_RE = re.compile(
        r"\b(wrong|incorrect|broken|still fails?|not fixed|regression)\b|不对|错误|仍然|还是失败|没修好|有问题",
        re.IGNORECASE,
    )

    def __init__(self, episode: TaskEpisode) -> None:
        self.episode = episode
        self.turn = max(episode.started_turn, episode.completed_turn or episode.started_turn)
        self.pending_report: TurnDisposition | None = None

    def begin_turn(self, user_message: str | None = None) -> Literal["continue", "reopened", "new_episode"]:
        self.turn += 1
        self.pending_report = None
        if self.episode.status == "succeeded" and user_message:
            if self._CORRECTION_RE.search(user_message):
                self.episode.status = "active"
                self.episode.completed_turn = None
                self.episode.outcome_version += 1
                self.episode.open_questions = []
                self.episode.unresolved = []
                return "reopened"
            return "new_episode"
        if self.episode.status in {"waiting_user", "blocked", "dormant", "completion_pending"}:
            self.episode.status = "active"
        return "continue"

    def submit(self, report: TurnDisposition) -> None:
        self.pending_report = report

    def apply_turn(
        self,
        *,
        evidence: list[dict[str, Any]],
        unresolved: list[str] | None = None,
        plan_steps: list[dict[str, Any]] | None = None,
        current_objective: str | None = None,
    ) -> CompletionGateResult:
        report = self.pending_report
        if report is None:
            self.episode.status = "active"
            return CompletionGateResult(False, ("turn ended without report_task_outcome",))
        self.episode.last_summary = report.summary
        self.episode.open_questions = list(report.open_questions)
        self.episode.unresolved = list(unresolved or [])
        refs = [ref for item in report.criteria_evidence for ref in item.evidence_refs]
        self.episode.evidence_refs = list(dict.fromkeys([*self.episode.evidence_refs, *refs]))
        if report.disposition == "waiting_user":
            self.episode.status = "waiting_user"
            return CompletionGateResult(False, ("waiting for user input",), tuple(refs))
        if report.disposition == "blocked":
            self.episode.status = "blocked"
            return CompletionGateResult(False, ("episode is blocked",), tuple(refs))
        if report.disposition == "continue":
            self.episode.status = "active"
            return CompletionGateResult(False, ("more work remains",), tuple(refs))

        self.episode.status = "completion_pending"
        verdict = self.completion_gate(
            report=report,
            evidence=evidence,
            unresolved=self.episode.unresolved,
            plan_steps=plan_steps or [],
            current_objective=current_objective,
        )
        if verdict.passed:
            self.episode.status = "succeeded"
            self.episode.completed_turn = self.turn
        else:
            self.episode.status = "active"
            self.episode.unresolved = list(dict.fromkeys([*self.episode.unresolved, *verdict.reasons]))
        return verdict

    def completion_gate(
        self,
        *,
        report: TurnDisposition,
        evidence: list[dict[str, Any]],
        unresolved: list[str],
        plan_steps: list[dict[str, Any]],
        current_objective: str | None = None,
    ) -> CompletionGateResult:
        reasons: list[str] = []
        if current_objective is not None and current_objective.strip() != self.episode.objective:
            reasons.append("episode objective was overwritten")
        by_ref: dict[str, dict[str, Any]] = {}
        current_events: list[dict[str, Any]] = []
        for item in evidence:
            seq = int(item.get("seq", 0))
            if seq < self.episode.start_event_seq:
                continue
            event_episode = str(item.get("episode_id", self.episode.episode_id))
            if event_episode != self.episode.episode_id:
                continue
            current_events.append(item)
            for key in ("ref", "call_id", "artifact_id"):
                if item.get(key):
                    by_ref[str(item[key])] = item

        supplied = {item.criterion_id: item.evidence_refs for item in report.criteria_evidence}
        accepted_refs: list[str] = []
        for criterion in self.episode.success_criteria:
            if not criterion.required:
                continue
            refs = supplied.get(criterion.criterion_id, [])
            valid = [ref for ref in refs if ref in by_ref]
            accepted_refs.extend(valid)
            if not refs:
                reasons.append(f"required criterion {criterion.criterion_id} has no evidence")
            elif len(valid) != len(refs):
                reasons.append(f"required criterion {criterion.criterion_id} has invalid evidence refs")
        if report.open_questions:
            reasons.append("open questions remain")
        if unresolved:
            reasons.append("unresolved items remain")
        for step in plan_steps:
            if not bool(step.get("required", True)):
                continue
            status = str(step.get("status", "pending"))
            if status in {"completed", "done"}:
                continue
            if status == "skipped" and str(step.get("reason", "")).strip():
                continue
            reasons.append(f"plan step {step.get('id', '?')} is not completed or explicitly skipped")

        modifications = [int(x.get("seq", 0)) for x in current_events if x.get("type") == "file_change"]
        verifications = [
            int(x.get("seq", 0))
            for x in current_events
            if x.get("type") == "tool_result"
            and x.get("name") == "run_shell"
            and bool(x.get("success"))
            and bool(x.get("matches_task", True))
        ]
        if modifications and not any(seq > max(modifications) for seq in verifications):
            reasons.append("workspace changes lack a successful verification after the latest modification")
        return CompletionGateResult(not reasons, tuple(reasons), tuple(dict.fromkeys(accepted_refs)))

    def cancel(self) -> None:
        self.episode.status = "cancelled"

    def supersede(self) -> None:
        self.episode.status = "superseded"

    def dormant(self) -> None:
        if self.episode.status not in {"succeeded", "cancelled", "superseded"}:
            self.episode.status = "dormant"
