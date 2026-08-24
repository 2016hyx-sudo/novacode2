"""Skill evolution status dashboard and replay-friendly JSON output."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from .bank import SkillBank
from .candidates import PendingCandidateStore


def skill_evaluation_status(*, project_dir: Path, user_dir: Path | None = None) -> dict[str, Any]:
    bank = SkillBank(project_dir=project_dir, user_dir=user_dir)
    store = PendingCandidateStore(project_dir)
    entries = bank.scan(strict=False)
    candidates = store.list()
    statuses = Counter(candidate.status for candidate in candidates)
    provenance_events: Counter[str] = Counter()
    malformed_provenance = 0
    if store.provenance_path.is_file():
        for raw in store.provenance_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(raw)
                provenance_events[str(record.get("event", "unknown"))] += 1
            except (TypeError, json.JSONDecodeError):
                malformed_provenance += 1
    return {
        "skills": len(entries),
        "project_skills": sum(item.source == "project" for item in entries.values()),
        "user_skills": sum(item.source == "user" for item in entries.values()),
        "candidates": len(candidates),
        "candidate_status": dict(sorted(statuses.items())),
        "skill_names": sorted(entries),
        "provenance_events": dict(sorted(provenance_events.items())),
        "malformed_provenance": malformed_provenance,
    }
