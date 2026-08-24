"""Read-only project/user skill discovery with deterministic precedence."""
from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from .models import SkillEntry, SkillManifest


class SkillBankError(ValueError):
    pass


def _parse_scalar(text: str) -> Any:
    value = text.strip()
    if not value:
        return ""
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError) as exc:
                raise SkillBankError(f"invalid frontmatter list: {value}") from exc
        if not isinstance(parsed, list):
            raise SkillBankError(f"expected a list: {value}")
        return parsed
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        try:
            return ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return value[1:-1]
    return value


def parse_skill_markdown(text: str) -> tuple[dict[str, Any], str]:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise SkillBankError("SKILL.md must start with YAML frontmatter")
    try:
        end = next(index for index in range(1, len(lines)) if lines[index].strip() == "---")
    except StopIteration as exc:
        raise SkillBankError("SKILL.md frontmatter is not closed") from exc
    data: dict[str, Any] = {}
    pending_list: str | None = None
    for number, raw in enumerate(lines[1:end], 2):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if raw[:1].isspace() and stripped.startswith("- ") and pending_list is not None:
            data[pending_list].append(_parse_scalar(stripped[2:]))
            continue
        pending_list = None
        if raw[:1].isspace() or ":" not in raw:
            raise SkillBankError(f"unsupported YAML at frontmatter line {number}")
        key, value = raw.split(":", 1)
        key = key.strip()
        if not key or key in data:
            raise SkillBankError(f"invalid or duplicate frontmatter key at line {number}")
        data[key] = _parse_scalar(value)
        if not value.strip():
            data[key] = []
            pending_list = key
    body = "\n".join(lines[end + 1 :]).strip()
    if not body:
        raise SkillBankError("SKILL.md body must not be empty")
    return data, body


class SkillBank:
    """Scan two immutable roots; project entries override user entries by name."""

    def __init__(self, *, project_dir: Path, user_dir: Path | None = None) -> None:
        self.project_dir = Path(project_dir)
        self.user_dir = Path(user_dir).expanduser() if user_dir is not None else None

    def scan(self, *, strict: bool = True) -> dict[str, SkillEntry]:
        entries: dict[str, SkillEntry] = {}
        errors: list[str] = []
        roots = []
        if self.user_dir is not None:
            roots.append(("user", self.user_dir))
        roots.append(("project", self.project_dir))
        for source, root in roots:
            if not root.is_dir():
                continue
            for directory in sorted(root.iterdir(), key=lambda path: path.name):
                if not directory.is_dir() or directory.is_symlink() or directory.name.startswith("."):
                    continue
                path = directory / "SKILL.md"
                if not path.is_file():
                    errors.append(f"{directory}: missing SKILL.md")
                    continue
                try:
                    data, body = parse_skill_markdown(path.read_text(encoding="utf-8"))
                    manifest = SkillManifest.from_dict(data)
                    if manifest.name != directory.name:
                        raise SkillBankError(
                            f"manifest name {manifest.name!r} does not match directory {directory.name!r}"
                        )
                    files = tuple(
                        child for child in directory.rglob("*") if child.is_file() and not child.is_symlink()
                    )
                    entries[manifest.name] = SkillEntry(
                        manifest=manifest,
                        body=body,
                        directory=directory,
                        source=source,  # type: ignore[arg-type]
                        files=files,
                    )
                except (OSError, UnicodeError, TypeError, ValueError) as exc:
                    errors.append(f"{path}: {exc}")
        if strict and errors:
            raise SkillBankError("invalid skill bank:\n- " + "\n- ".join(errors))
        return entries

    def get(self, name: str) -> SkillEntry | None:
        return self.scan().get(name)

    def list(self) -> list[SkillEntry]:
        return [entry for _, entry in sorted(self.scan().items())]
