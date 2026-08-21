"""Data models and serialization for the NovaCode long-term memory system."""
from __future__ import annotations

import datetime
from dataclasses import asdict, dataclass, field
from enum import Enum
import re
from typing import Any

import yaml


class MemoryType(str, Enum):
    """Structured semantic taxonomy for persistent memories."""
    USER = "user"          # User technical background, coding habits, toolchain preferences
    FEEDBACK = "feedback"  # Corrected errors, root cause explanations, and verified fixes
    PROJECT = "project"    # Cross-session architectural decisions, milestone constraints
    REFERENCE = "reference"# External pointers, URLs, ports, fixed commands


@dataclass(frozen=True)
class QuotaConfig:
    """Multi-tier quota limits for long-term memory."""
    max_content_bytes: int = 4 * 1024       # 4 KB per entry
    max_recalled_items: int = 5             # Max 5 recalled memories per turn
    max_session_injected_bytes: int = 60 * 1024 # 60 KB cumulative limit per session
    max_scan_pool_size: int = 200           # Max 200 items in candidate scan pool
    max_name_length: int = 40               # Max 40 chars for memory name
    max_description_length: int = 80        # Max 80 chars for description


def current_iso_timestamp() -> str:
    """Return the current UTC timestamp formatted as ISO-8601 string."""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


@dataclass
class MemoryHeader:
    """Lightweight metadata header for quick scanning and side-query arbitration."""
    name: str
    type: MemoryType
    description: str
    created_at: str = field(default_factory=current_iso_timestamp)
    updated_at: str = field(default_factory=current_iso_timestamp)
    file_path: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.type, str) and not isinstance(self.type, MemoryType):
            self.type = MemoryType(self.type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "file_path": self.file_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryHeader:
        raw_type = data.get("type", "user")
        try:
            m_type = MemoryType(raw_type)
        except ValueError:
            m_type = MemoryType.USER

        return cls(
            name=str(data.get("name", "")),
            type=m_type,
            description=str(data.get("description", "")),
            created_at=str(data.get("created_at", "") or current_iso_timestamp()),
            updated_at=str(data.get("updated_at", "") or current_iso_timestamp()),
            file_path=str(data.get("file_path", "")),
        )

    def to_yaml_frontmatter(self) -> str:
        data = {
            "name": self.name,
            "type": self.type.value,
            "description": self.description,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        dumped = yaml.safe_dump(data, sort_keys=False, allow_unicode=True).strip()
        return f"---\n{dumped}\n---"


@dataclass
class MemoryEntry:
    """A full memory entry consisting of a metadata header and Markdown body."""
    header: MemoryHeader
    content: str
    file_path: str = ""

    def __post_init__(self) -> None:
        if self.file_path and not self.header.file_path:
            self.header.file_path = self.file_path

    @property
    def name(self) -> str:
        return self.header.name

    @property
    def type(self) -> MemoryType:
        return self.header.type

    @property
    def description(self) -> str:
        return self.header.description

    def truncate_content_if_needed(self, max_bytes: int = 4 * 1024) -> str:
        """Truncate content if it exceeds max_bytes, appending a notice."""
        encoded = self.content.encode("utf-8")
        if len(encoded) <= max_bytes:
            return self.content
        
        truncated_bytes = encoded[:max_bytes]
        # Avoid partial utf-8 characters
        truncated_text = truncated_bytes.decode("utf-8", errors="ignore")
        return truncated_text + "\n\n[... truncated, memory file too large ...]"

    def to_markdown(self) -> str:
        """Serialize entry to YAML Frontmatter + Markdown Body."""
        frontmatter = self.header.to_yaml_frontmatter()
        body = self.content.strip()
        return f"{frontmatter}\n\n{body}\n"

    @classmethod
    def from_markdown(cls, text: str, file_path: str = "") -> MemoryEntry:
        """Parse YAML Frontmatter + Markdown Body."""
        text = text.strip()
        if not text.startswith("---"):
            # Plain text fallback
            header = MemoryHeader(
                name=file_path.split("/")[-1].replace(".md", "") if file_path else "unnamed",
                type=MemoryType.USER,
                description="",
                file_path=file_path,
            )
            return cls(header=header, content=text, file_path=file_path)

        parts = re.split(r"^---\s*$", text, maxsplit=2, flags=re.MULTILINE)
        if len(parts) >= 3:
            raw_frontmatter = parts[1].strip()
            body = parts[2].strip()
            try:
                data = yaml.safe_load(raw_frontmatter) or {}
            except Exception:
                data = {}
            if file_path:
                data["file_path"] = file_path
            header = MemoryHeader.from_dict(data)
            return cls(header=header, content=body, file_path=file_path)
        else:
            # Malformed frontmatter fallback
            header = MemoryHeader(
                name=file_path.split("/")[-1].replace(".md", "") if file_path else "unnamed",
                type=MemoryType.USER,
                description="",
                file_path=file_path,
            )
            return cls(header=header, content=text, file_path=file_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "header": self.header.to_dict(),
            "content": self.content,
            "file_path": self.file_path or self.header.file_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        header = MemoryHeader.from_dict(data.get("header") or {})
        content = str(data.get("content", ""))
        file_path = str(data.get("file_path", "") or header.file_path)
        return cls(header=header, content=content, file_path=file_path)
