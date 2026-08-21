"""Storage manager for NovaCode long-term memory system."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Literal

from .models import (
    MemoryEntry,
    MemoryHeader,
    MemoryType,
    QuotaConfig,
    current_iso_timestamp,
)


class MemoryStore:
    """Manages disk persistence, namespace isolation, and index caching for memories."""

    def __init__(
        self,
        project_dir: Path | str | None = None,
        global_dir: Path | str | None = None,
        quota: QuotaConfig | None = None,
    ) -> None:
        self.quota = quota or QuotaConfig()
        
        # Project-specific directory (defaults to <cwd>/.agent/memories)
        if project_dir is not None:
            self.project_dir = Path(project_dir).expanduser().resolve()
        else:
            self.project_dir = (Path.cwd() / ".agent" / "memories").resolve()

        # Global user directory (defaults to ~/.novacode/memories/global)
        if global_dir is not None:
            self.global_dir = Path(global_dir).expanduser().resolve()
        else:
            self.global_dir = (Path.home() / ".novacode" / "memories" / "global").resolve()

        # Ensure directories exist
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.global_dir.mkdir(parents=True, exist_ok=True)

    def _get_target_dir(self, entry_type: MemoryType | str, scope: Literal["project", "global", "auto"] = "auto") -> Path:
        if scope == "global":
            return self.global_dir
        if scope == "project":
            return self.project_dir
        # Auto: 'user' type can go to global if desired, but default is project unless specified
        if isinstance(entry_type, str):
            try:
                entry_type = MemoryType(entry_type)
            except ValueError:
                entry_type = MemoryType.USER
        if entry_type == MemoryType.USER:
            return self.global_dir
        return self.project_dir

    def _sanitize_name(self, name: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())
        return clean[: self.quota.max_name_length] or "memory"

    def save_entry(
        self,
        entry: MemoryEntry,
        scope: Literal["project", "global", "auto"] = "auto",
    ) -> MemoryEntry:
        """Persist a memory entry to disk as a YAML Frontmatter + Markdown file."""
        entry.header.name = self._sanitize_name(entry.header.name)
        target_dir = self._get_target_dir(entry.header.type, scope)
        file_path = target_dir / f"{entry.header.name}.md"

        now = current_iso_timestamp()
        if not entry.header.created_at:
            entry.header.created_at = now
        entry.header.updated_at = now
        entry.file_path = str(file_path)
        entry.header.file_path = str(file_path)

        # Truncate content if necessary
        truncated_content = entry.truncate_content_if_needed(self.quota.max_content_bytes)
        entry.content = truncated_content

        file_path.write_text(entry.to_markdown(), encoding="utf-8")
        self._sync_index_entry(target_dir, entry.header)
        return entry

    def load_entry(self, name: str) -> MemoryEntry | None:
        """Load a memory entry by name, searching project first then global."""
        clean_name = self._sanitize_name(name)
        
        # Check project dir first
        p_path = self.project_dir / f"{clean_name}.md"
        if p_path.is_file():
            content = p_path.read_text(encoding="utf-8")
            return MemoryEntry.from_markdown(content, file_path=str(p_path))

        # Check global dir
        g_path = self.global_dir / f"{clean_name}.md"
        if g_path.is_file():
            content = g_path.read_text(encoding="utf-8")
            return MemoryEntry.from_markdown(content, file_path=str(g_path))

        return None

    def delete_entry(self, name: str) -> bool:
        """Delete a memory entry from disk and remove it from the index."""
        clean_name = self._sanitize_name(name)
        deleted = False

        for target_dir in (self.project_dir, self.global_dir):
            file_path = target_dir / f"{clean_name}.md"
            if file_path.is_file():
                file_path.unlink()
                self._remove_index_entry(target_dir, clean_name)
                deleted = True

        return deleted

    def list_headers(self, limit: int = 200) -> list[MemoryHeader]:
        """Scan memory headers fast (<5ms) using index or fast frontmatter scan."""
        headers: list[MemoryHeader] = []
        seen_names: set[str] = set()

        for target_dir in (self.project_dir, self.global_dir):
            dir_headers = self._load_dir_headers(target_dir)
            for h in dir_headers:
                if h.name not in seen_names:
                    seen_names.add(h.name)
                    headers.append(h)

        # Sort by updated_at descending
        headers.sort(key=lambda x: x.updated_at or x.created_at, reverse=True)
        return headers[: max(1, min(limit, self.quota.max_scan_pool_size))]

    def _load_dir_headers(self, target_dir: Path) -> list[MemoryHeader]:
        """Load headers for a directory with fallback to scanning top 30 lines."""
        if not target_dir.is_dir():
            return []

        index_file = target_dir / "index.jsonl"
        if index_file.is_file():
            try:
                headers: list[MemoryHeader] = []
                for line in index_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    headers.append(MemoryHeader.from_dict(data))
                if headers:
                    return headers
            except Exception:
                pass

        # Fallback: fast scan top 30 lines of .md files
        headers = []
        md_files = sorted(target_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in md_files[: self.quota.max_scan_pool_size]:
            try:
                # Read at most 30 lines
                lines: list[str] = []
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    for _ in range(30):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line)
                partial_text = "".join(lines)
                entry = MemoryEntry.from_markdown(partial_text, file_path=str(p))
                headers.append(entry.header)
            except Exception:
                continue

        # Re-save index in background
        self._rebuild_index(target_dir, headers)
        return headers

    def _sync_index_entry(self, target_dir: Path, header: MemoryHeader) -> None:
        """Add or update an entry in index.jsonl."""
        index_file = target_dir / "index.jsonl"
        headers_map: dict[str, MemoryHeader] = {}
        if index_file.is_file():
            try:
                for line in index_file.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        h = MemoryHeader.from_dict(json.loads(line))
                        headers_map[h.name] = h
            except Exception:
                pass

        headers_map[header.name] = header
        self._write_index(target_dir, list(headers_map.values()))

    def _remove_index_entry(self, target_dir: Path, name: str) -> None:
        """Remove an entry from index.jsonl."""
        index_file = target_dir / "index.jsonl"
        if not index_file.is_file():
            return
        headers_map: dict[str, MemoryHeader] = {}
        try:
            for line in index_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    h = MemoryHeader.from_dict(json.loads(line))
                    if h.name != name:
                        headers_map[h.name] = h
            self._write_index(target_dir, list(headers_map.values()))
        except Exception:
            pass

    def _rebuild_index(self, target_dir: Path, headers: list[MemoryHeader]) -> None:
        self._write_index(target_dir, headers)

    def _write_index(self, target_dir: Path, headers: list[MemoryHeader]) -> None:
        index_file = target_dir / "index.jsonl"
        lines = [json.dumps(h.to_dict(), ensure_ascii=False) for h in headers]
        try:
            index_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        except Exception:
            pass
