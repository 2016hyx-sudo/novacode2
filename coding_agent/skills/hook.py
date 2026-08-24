"""Read-only consumers for fold, verified-event and episode lifecycle events."""
from __future__ import annotations

from typing import Any

from ..runtime.episode import TaskEpisode
from .candidates import PendingCandidateStore, SkillCandidate
from .extractor import SkillExtractor
from .maintainer import SkillMaintainer


class SkillLifecycleHook:
    def __init__(
        self,
        store: PendingCandidateStore,
        extractor: SkillExtractor | None = None,
        maintainer: SkillMaintainer | None = None,
    ) -> None:
        self.store = store
        self.extractor = extractor or SkillExtractor()
        self.maintainer = maintainer

    def on_fold(self, *, episode: TaskEpisode, evidence: list[dict[str, Any]]) -> list[SkillCandidate]:
        """Fold is only a pending-ingestion boundary; never a promotion boundary."""
        created: list[SkillCandidate] = []
        for index in range(1, len(evidence) - 1):
            failure, fix, verification = evidence[index - 1 : index + 2]
            candidate = self.extractor.extract_event_driven(
                episode_id=episode.episode_id,
                outcome_version=episode.outcome_version,
                failure=failure,
                fix=fix,
                verification=verification,
            )
            if candidate is not None:
                self.store.save(candidate, event="candidate_from_fold")
                created.append(candidate)
        return created

    def on_verified_event(
        self,
        *,
        episode: TaskEpisode,
        failure: dict[str, Any],
        fix: dict[str, Any],
        verification: dict[str, Any],
    ) -> SkillCandidate | None:
        candidate = self.extractor.extract_event_driven(
            episode_id=episode.episode_id,
            outcome_version=episode.outcome_version,
            failure=failure,
            fix=fix,
            verification=verification,
        )
        if candidate is not None:
            self.store.save(candidate, event="candidate_from_verified_event")
        return candidate

    def on_episode_succeeded(self, episode: TaskEpisode) -> SkillCandidate | None:
        candidate = self.extractor.extract_task_level(
            episode_id=episode.episode_id,
            outcome_version=episode.outcome_version,
            objective=episode.objective,
            summary=episode.last_summary,
            evidence_refs=episode.evidence_refs,
            episode_succeeded=episode.status == "succeeded",
        )
        if candidate is not None:
            self.store.save(candidate, event="candidate_from_episode_succeeded")
        for pending in self.store.list(status="pending"):
            if pending.episode_id == episode.episode_id and pending.outcome_version == episode.outcome_version:
                pending.episode_succeeded = True
                self.store.save(pending, event="candidate_episode_evidence_added")
                if self.maintainer is not None:
                    self.maintainer.promote(
                        pending,
                        current_outcome_version=episode.outcome_version,
                    )
        if candidate is not None and self.maintainer is not None:
            self.maintainer.promote(
                candidate,
                current_outcome_version=episode.outcome_version,
            )
        return candidate
