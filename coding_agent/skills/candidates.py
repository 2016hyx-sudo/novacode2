"""Pending candidate window and append-only evolution provenance."""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..structured_context.models import utcnow

CandidateStatus = Literal[
    "pending", "verified", "promoted", "dropped", "stale", "needs_revision", "failed"
]


@dataclass
class SkillRule:
    rule: str
    evidence_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"rule": self.rule, "evidence_refs": list(self.evidence_refs)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillRule:
        return cls(str(data.get("rule", "")), [str(x) for x in data.get("evidence_refs") or []])


@dataclass
class SkillCandidate:
    candidate_id: str
    episode_id: str
    outcome_version: int
    name: str
    title: str
    granularity: Literal["task-level", "event-driven"]
    when_to_apply: str
    constraints_and_style: list[str]
    workflow_rules: list[SkillRule]
    evidence_refs: list[str]
    source: str
    status: CandidateStatus = "pending"
    episode_succeeded: bool = False
    independently_verified: bool = False
    has_scripts: bool = False
    verification: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, **kwargs: Any) -> SkillCandidate:
        return cls(candidate_id=f"cand-{uuid.uuid4().hex[:16]}", **kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "episode_id": self.episode_id,
            "outcome_version": self.outcome_version,
            "name": self.name,
            "title": self.title,
            "granularity": self.granularity,
            "when_to_apply": self.when_to_apply,
            "constraints_and_style": list(self.constraints_and_style),
            "workflow_rules": [item.to_dict() for item in self.workflow_rules],
            "evidence_refs": list(self.evidence_refs),
            "source": self.source,
            "status": self.status,
            "episode_succeeded": self.episode_succeeded,
            "independently_verified": self.independently_verified,
            "has_scripts": self.has_scripts,
            "verification": dict(self.verification),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillCandidate:
        return cls(
            candidate_id=str(data["candidate_id"]),
            episode_id=str(data["episode_id"]),
            outcome_version=int(data.get("outcome_version", 1)),
            name=str(data.get("name", "")),
            title=str(data.get("title", "")),
            granularity=str(data.get("granularity", "event-driven")),  # type: ignore[arg-type]
            when_to_apply=str(data.get("when_to_apply", "")),
            constraints_and_style=[str(x) for x in data.get("constraints_and_style") or []],
            workflow_rules=[SkillRule.from_dict(x) for x in data.get("workflow_rules") or []],
            evidence_refs=[str(x) for x in data.get("evidence_refs") or []],
            source=str(data.get("source", "")),
            status=str(data.get("status", "pending")),  # type: ignore[arg-type]
            episode_succeeded=bool(data.get("episode_succeeded", False)),
            independently_verified=bool(data.get("independently_verified", False)),
            has_scripts=bool(data.get("has_scripts", False)),
            verification=dict(data.get("verification") or {}),
        )


class PendingCandidateStore:
    def __init__(self, skill_root: Path, *, fsync: bool = True) -> None:
        self.root = Path(skill_root) / ".evolution"
        self.candidates_dir = self.root / "candidates"
        self.history_dir = self.root / "history"
        self.provenance_path = self.root / "provenance.jsonl"
        self.fsync = fsync

    def save(self, candidate: SkillCandidate, *, event: str = "candidate_saved", **metadata: Any) -> Path:
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        path = self.candidates_dir / f"{candidate.candidate_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
        self.audit(event, candidate, **metadata)
        return path

    def get(self, candidate_id: str) -> SkillCandidate | None:
        if Path(candidate_id).name != candidate_id:
            raise ValueError("invalid candidate id")
        path = self.candidates_dir / f"{candidate_id}.json"
        if not path.is_file():
            return None
        return SkillCandidate.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list(self, *, status: str | None = None) -> list[SkillCandidate]:
        if not self.candidates_dir.is_dir():
            return []
        values = [
            SkillCandidate.from_dict(json.loads(path.read_text(encoding="utf-8")))
            for path in sorted(self.candidates_dir.glob("cand-*.json"))
        ]
        return [item for item in values if status is None or item.status == status]

    def mark_episode_stale(self, episode_id: str, *, older_than_version: int) -> int:
        count = 0
        for candidate in self.list():
            if (
                candidate.episode_id == episode_id
                and candidate.outcome_version < older_than_version
                and candidate.status not in {"dropped", "stale"}
            ):
                candidate.status = "stale"
                self.save(candidate, event="candidate_stale", reason="episode_reopened")
                count += 1
        return count

    def audit(self, event: str, candidate: SkillCandidate, **metadata: Any) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": utcnow(),
            "event": event,
            "candidate_id": candidate.candidate_id,
            "episode_id": candidate.episode_id,
            "outcome_version": candidate.outcome_version,
            "status": candidate.status,
            "metadata": metadata,
        }
        with self.provenance_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            if self.fsync:
                handle.flush()
                os.fsync(handle.fileno())
