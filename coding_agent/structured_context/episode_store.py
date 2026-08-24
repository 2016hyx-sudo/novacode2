"""Append-only episode transition storage beside a structured session."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..runtime.episode import TaskEpisode
from .models import utcnow


class EpisodeStore:
    def __init__(self, session_dir: Path, *, fsync: bool = True) -> None:
        self.path = Path(session_dir) / "episodes.jsonl"
        self.fsync = fsync

    def append(self, event_type: str, episode: TaskEpisode, **metadata: Any) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": utcnow(),
            "type": event_type,
            "episode_id": episode.episode_id,
            "outcome_version": episode.outcome_version,
            "episode": episode.to_dict(),
            "metadata": metadata,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            if self.fsync:
                handle.flush()
                os.fsync(handle.fileno())
        return record

    def records(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        records: list[dict[str, Any]] = []
        for raw in self.path.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                records.append(json.loads(raw))
        return records

    def latest(self) -> TaskEpisode | None:
        records = self.records()
        if not records:
            return None
        return TaskEpisode.from_dict(records[-1]["episode"])
