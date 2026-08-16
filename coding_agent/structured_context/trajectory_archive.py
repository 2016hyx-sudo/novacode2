"""Append-only archive of complete, never-folded Interaction Groups."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import InteractionGroup


class TrajectoryArchive:
    def __init__(self, path: Path, *, fsync: bool = True) -> None:
        self.path = Path(path)
        self.fsync = fsync

    def append_group(
        self,
        group: InteractionGroup,
        *,
        raw_tool_result_refs: list[dict[str, Any]] | None = None,
    ) -> None:
        entry = group.to_dict()
        entry["raw_tool_result_refs"] = [dict(ref) for ref in raw_tool_result_refs or []]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            if self.fsync:
                handle.flush()
                os.fsync(handle.fileno())

    def read_groups(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        groups: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line:
                    groups.append(json.loads(line))
        return groups
