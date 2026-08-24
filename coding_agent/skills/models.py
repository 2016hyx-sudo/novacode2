"""Data models for a multi-file NovaCode skill package."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

SkillGranularity = Literal["task-level", "event-driven"]
SkillContext = Literal["inline", "fork"]

_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{1,63}$")
_VERSION_RE = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")


@dataclass(frozen=True)
class SkillManifest:
    name: str
    title: str
    granularity: SkillGranularity
    version: str
    context: SkillContext
    when_to_apply: str
    user_invocable: bool = True
    allowed_tools: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    evolution_notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SkillManifest:
        required = ("name", "title", "granularity", "version", "context", "when_to_apply")
        missing = [key for key in required if not str(data.get(key, "")).strip()]
        if missing:
            raise ValueError(f"missing manifest fields: {', '.join(missing)}")
        name = str(data["name"]).strip()
        if not _NAME_RE.fullmatch(name):
            raise ValueError("name must be 2-64 lowercase snake_case characters")
        granularity = str(data["granularity"]).strip()
        if granularity not in {"task-level", "event-driven"}:
            raise ValueError("granularity must be task-level or event-driven")
        version = str(data["version"]).strip().lstrip("v")
        if not _VERSION_RE.fullmatch(version):
            raise ValueError("version must be semantic major.minor.patch")
        context = str(data["context"]).strip()
        if context not in {"inline", "fork"}:
            raise ValueError("context must be inline or fork")

        def strings(value: Any, field_name: str) -> tuple[str, ...]:
            if value is None:
                return ()
            if not isinstance(value, (list, tuple)):
                raise TypeError(f"{field_name} must be a list")
            values = tuple(str(item).strip() for item in value if str(item).strip())
            if len(values) != len(set(values)):
                raise ValueError(f"{field_name} must not contain duplicates")
            return values

        user_invocable = data.get("user_invocable", True)
        if not isinstance(user_invocable, bool):
            raise TypeError("user_invocable must be true or false")
        allowed_tools = strings(data.get("allowed_tools"), "allowed_tools")
        if any(not _NAME_RE.fullmatch(tool) for tool in allowed_tools):
            raise ValueError("allowed_tools entries must be lowercase snake_case names")
        return cls(
            name=name,
            title=str(data["title"]).strip(),
            granularity=granularity,  # type: ignore[arg-type]
            version=version,
            context=context,  # type: ignore[arg-type]
            when_to_apply=str(data["when_to_apply"]).strip(),
            user_invocable=user_invocable,
            allowed_tools=allowed_tools,
            tags=strings(data.get("tags"), "tags"),
            evolution_notes=strings(data.get("evolution_notes"), "evolution_notes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "granularity": self.granularity,
            "version": self.version,
            "context": self.context,
            "when_to_apply": self.when_to_apply,
            "user_invocable": self.user_invocable,
            "allowed_tools": list(self.allowed_tools),
            "tags": list(self.tags),
            "evolution_notes": list(self.evolution_notes),
        }


@dataclass(frozen=True)
class SkillEntry:
    manifest: SkillManifest
    body: str
    directory: Path
    source: Literal["project", "user"]
    files: tuple[Path, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def skill_dir(self) -> str:
        return str(self.directory.resolve())

    def render(self, arguments: dict[str, Any] | None = None) -> str:
        import json

        rendered = self.body.replace("${SKILL_DIR}", self.skill_dir)
        rendered = rendered.replace(
            "${ARGUMENTS}", json.dumps(arguments or {}, ensure_ascii=False, sort_keys=True)
        )
        return rendered
