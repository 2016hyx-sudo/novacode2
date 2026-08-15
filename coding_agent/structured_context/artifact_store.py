"""Content-addressed raw tool result artifacts."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .models import utcnow


class ArtifactStore:
    def __init__(self, directory: Path, *, fsync: bool = True) -> None:
        self.directory = Path(directory)
        self.index_path = self.directory.parent / "artifact-index.jsonl"
        self.fsync = fsync
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        tool: str,
        content: str | bytes,
        *,
        tool_call_id: str = "",
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        raw = content.encode("utf-8") if isinstance(content, str) else bytes(content)
        artifact_id = hashlib.sha256(raw).hexdigest()
        tool_dir = self.directory / tool
        tool_dir.mkdir(parents=True, exist_ok=True)
        target = tool_dir / f"{artifact_id}.artifact"
        if not target.exists():
            tmp = tool_dir / f".{artifact_id}.tmp"
            tmp.write_bytes(raw)
            if self.fsync:
                with tmp.open("rb") as handle:
                    os.fsync(handle.fileno())
            os.replace(tmp, target)
        entry = {
            "seq": self._next_index_seq(),
            "artifact_id": artifact_id,
            "tool": tool,
            "tool_call_id": tool_call_id,
            "arguments": dict(arguments or {}),
            "created_at": utcnow(),
            "size": len(raw),
            "sha256": artifact_id,
            "encoding": "utf-8",
            "path": str(target.relative_to(self.directory.parent)) if target.is_relative_to(self.directory.parent) else str(target),
            "sensitive_filter_applied": False,
        }
        self._append_index(entry)
        return entry

    def _next_index_seq(self) -> int:
        if not self.index_path.exists():
            return 1
        last = 0
        with self.index_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    last = max(last, int(json.loads(line)["seq"]))
                except (KeyError, ValueError, json.JSONDecodeError):
                    continue
        return last + 1

    def _append_index(self, entry: dict[str, Any]) -> None:
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")
            if self.fsync:
                handle.flush()
                os.fsync(handle.fileno())

    def read(self, artifact_id: str) -> bytes | None:
        entry = self.find(artifact_id)
        if entry is None:
            return None
        path = self.directory / entry["tool"] / f"{artifact_id}.artifact"
        if not path.exists():
            return None
        return path.read_bytes()

    def find(self, artifact_id: str) -> dict[str, Any] | None:
        if not self.index_path.exists():
            return None
        with self.index_path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("artifact_id") == artifact_id:
                    return entry
        return None

    def last_index_seq(self) -> int:
        return self._next_index_seq() - 1
