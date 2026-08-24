"""Promotion decisions, semantic merge, versioning and history."""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .bank import SkillBank
from .candidates import PendingCandidateStore, SkillCandidate, SkillRule
from .models import SkillEntry
from .verifier import SurrogateVerifier


@dataclass(frozen=True)
class MaintenanceDecision:
    action: Literal["add", "merge", "drop"]
    reason: str
    target: str | None = None


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_]+", text.lower()))


class SkillMaintainer:
    def __init__(
        self,
        bank: SkillBank,
        candidates: PendingCandidateStore,
        *,
        verifier: SurrogateVerifier | None = None,
    ) -> None:
        self.bank = bank
        self.candidates = candidates
        self.verifier = verifier or SurrogateVerifier()

    def decide(self, candidate: SkillCandidate, *, current_outcome_version: int) -> MaintenanceDecision:
        if candidate.status == "stale" or candidate.outcome_version != current_outcome_version:
            return MaintenanceDecision("drop", "candidate outcome version is stale")
        if candidate.granularity == "task-level" and not candidate.episode_succeeded:
            return MaintenanceDecision("drop", "task-level candidate lacks episode_succeeded evidence")
        if candidate.granularity == "event-driven" and not (
            candidate.episode_succeeded or candidate.independently_verified
        ):
            return MaintenanceDecision("drop", "event-driven candidate has not passed a promotion gate")
        entries = self.bank.list()
        if any(entry.name == candidate.name for entry in entries):
            return MaintenanceDecision("merge", "same capability identity", candidate.name)
        candidate_tokens = _tokens(f"{candidate.title} {candidate.when_to_apply}")
        scored: list[tuple[float, SkillEntry]] = []
        for entry in entries:
            other = _tokens(f"{entry.manifest.title} {entry.manifest.when_to_apply}")
            score = len(candidate_tokens & other) / max(1, len(candidate_tokens | other))
            scored.append((score, entry))
        scored.sort(key=lambda item: item[0], reverse=True)
        if scored and scored[0][0] >= 0.72:
            return MaintenanceDecision("merge", "high semantic overlap", scored[0][1].name)
        return MaintenanceDecision("add", "distinct durable capability")

    def promote(
        self,
        candidate: SkillCandidate,
        *,
        current_outcome_version: int,
        package_dir: Path | None = None,
    ) -> MaintenanceDecision:
        if candidate.status == "stale" or candidate.outcome_version != current_outcome_version:
            candidate.status = "dropped"
            self.candidates.save(candidate, event="candidate_dropped", reason="stale outcome")
            return MaintenanceDecision("drop", "candidate outcome version is stale")
        if candidate.granularity == "task-level" and not candidate.episode_succeeded:
            candidate.status = "dropped"
            self.candidates.save(candidate, event="candidate_dropped", reason="episode not succeeded")
            return MaintenanceDecision("drop", "task-level candidate lacks episode_succeeded evidence")
        verification = self.verifier.verify_with_refinement(candidate, package_dir)
        candidate.verification = verification.to_dict()
        if not verification.passed:
            candidate.status = "failed"
            self.candidates.save(candidate, event="verification_failed")
            return MaintenanceDecision("drop", "surrogate/script verification failed")
        candidate.independently_verified = True
        decision = self.decide(candidate, current_outcome_version=current_outcome_version)
        if decision.action == "drop":
            candidate.status = "dropped"
            self.candidates.save(candidate, event="candidate_dropped", reason=decision.reason)
            return decision
        if decision.action == "add":
            self._write_new(candidate, package_dir)
        else:
            self._merge(candidate, decision.target or candidate.name)
        candidate.status = "promoted"
        self.candidates.save(candidate, event="candidate_promoted", action=decision.action)
        return decision

    def _write_new(self, candidate: SkillCandidate, package_dir: Path | None) -> None:
        target = self.bank.project_dir / candidate.name
        if target.exists():
            raise FileExistsError(f"skill already exists: {candidate.name}")
        target.mkdir(parents=True)
        (target / "SKILL.md").write_text(self._render(candidate, version="0.1.0"), encoding="utf-8")
        if package_dir is not None:
            for name in ("scripts", "references", "assets"):
                source = Path(package_dir) / name
                if source.is_dir():
                    shutil.copytree(source, target / name)

    def _merge(self, candidate: SkillCandidate, target_name: str) -> None:
        existing = self.bank.get(target_name)
        if existing is None:
            raise FileNotFoundError(f"merge target not found: {target_name}")
        old = existing.directory / "SKILL.md"
        history = self.candidates.history_dir / target_name
        history.mkdir(parents=True, exist_ok=True)
        shutil.copy2(old, history / f"{existing.manifest.version}.SKILL.md")
        major, minor, patch = (int(x) for x in existing.manifest.version.split("."))
        version = f"{major}.{minor}.{patch + 1}"
        old_rules = [
            SkillRule(line[2:].strip(), [])
            for line in existing.body.splitlines()
            if line.strip().startswith("- ")
        ]
        seen: set[str] = set()
        merged_rules: list[SkillRule] = []
        for rule in [*old_rules, *candidate.workflow_rules]:
            key = " ".join(rule.rule.lower().split())
            if key and key not in seen:
                seen.add(key)
                merged_rules.append(rule)
        merged = SkillCandidate.from_dict(candidate.to_dict())
        merged.name = target_name
        merged.title = existing.manifest.title
        merged.when_to_apply = existing.manifest.when_to_apply
        merged.workflow_rules = merged_rules
        old.write_text(
            self._render(
                merged,
                version=version,
                evolution_notes=[
                    *existing.manifest.evolution_notes,
                    f"{version}: merged candidate {candidate.candidate_id}",
                ],
            ),
            encoding="utf-8",
        )

    @staticmethod
    def _render(
        candidate: SkillCandidate,
        *,
        version: str,
        evolution_notes: list[str] | None = None,
    ) -> str:
        def inline(values: list[str]) -> str:
            return json.dumps(values, ensure_ascii=False)

        rules = "\n".join(
            f"{index}. {rule.rule} (Evidence: {', '.join(rule.evidence_refs)})"
            for index, rule in enumerate(candidate.workflow_rules, 1)
        )
        constraints = "\n".join(f"- {item}" for item in candidate.constraints_and_style)
        return (
            "---\n"
            f"name: {candidate.name}\n"
            f"title: {json.dumps(candidate.title, ensure_ascii=False)}\n"
            f"granularity: {candidate.granularity}\n"
            f"version: {version}\n"
            "context: inline\n"
            f"when_to_apply: {json.dumps(candidate.when_to_apply, ensure_ascii=False)}\n"
            "user_invocable: true\n"
            "allowed_tools: []\n"
            "tags: []\n"
            f"evolution_notes: {inline(evolution_notes or [])}\n"
            "---\n\n# Goal\n"
            f"{candidate.title}\n\n# Constraints & Style\n{constraints}\n\n"
            f"# Workflow & Key Rules\n{rules}\n"
        )
