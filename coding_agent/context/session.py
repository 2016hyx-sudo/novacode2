"""Session persistence: one JSON file per agent task/conversation."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..llm.base import Message


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_session_id() -> str:
    return f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{uuid4().hex[:6]}"


@dataclass
class Session:
    id: str
    created_at: str
    updated_at: str
    provider: str
    model: str
    user_task: str
    messages: list[Message] = field(default_factory=list)
    status: str = "running"
    plan: list[str] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, *, user_task: str, provider: str, model: str) -> Session:
        now = _now()
        return cls(
            id=new_session_id(),
            created_at=now,
            updated_at=now,
            provider=provider,
            model=model,
            user_task=user_task,
            status="running",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "provider": self.provider,
            "model": self.model,
            "user_task": self.user_task,
            "messages": [message.to_dict() for message in self.messages],
            "status": self.status,
            "plan": self.plan,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Session:
        return cls(
            id=str(data["session_id"]),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            user_task=str(data.get("user_task", "")),
            messages=[Message.from_dict(item) for item in data.get("messages", [])],
            status=str(data.get("status", "running")),
            plan=[str(step) for step in data.get("plan") or []] or None,
            metadata=dict(data.get("metadata") or {}),
        )


class SessionStore:
    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    def save(self, session: Session) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{session.id}.json"
        tmp = self.directory / f".{session.id}.json.tmp"
        tmp.write_text(
            json.dumps(session.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, target)
        return target

    def load(self, session_id: str) -> Session:
        if Path(session_id).name != session_id:
            raise ValueError(f"Invalid session id: {session_id!r}")
        target = self.directory / f"{session_id}.json"
        if not target.exists():
            raise FileNotFoundError(f"Session not found: {session_id}")
        data = json.loads(target.read_text(encoding="utf-8"))
        return Session.from_dict(data)

    def list_sessions(self) -> list[Path]:
        if not self.directory.exists():
            return []
        return sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
