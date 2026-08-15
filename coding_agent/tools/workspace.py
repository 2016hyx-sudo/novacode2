"""Unified workspace path resolution.

Every filesystem tool funnels its path argument through this module so that
absolute paths, ``..`` traversal and symlink escapes are rejected in one place.
"""
from __future__ import annotations

import os
from pathlib import Path

from .base import ToolError, ToolErrorKind


def _ensure_workspace(workspace: Path) -> Path:
    try:
        return workspace.expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise ToolError(
            f"Workspace does not exist: {workspace}",
            kind=ToolErrorKind.NOT_FOUND,
        ) from exc


def resolve_workspace_path(workspace: Path, raw: str | Path) -> Path:
    """Resolve *raw* against *workspace* and guarantee the result stays inside.

    Raises ToolError(WORKSPACE_VIOLATION) for escape attempts.
    """
    root = _ensure_workspace(workspace)
    text = os.path.expanduser(str(raw))

    if not text.strip():
        return root

    candidate_raw = Path(text)
    if candidate_raw.is_absolute():
        raise ToolError(
            f"Absolute paths are not allowed: {text}",
            kind=ToolErrorKind.WORKSPACE_VIOLATION,
            hint="Use a path relative to the workspace.",
        )

    normalized = os.path.normpath(text)
    if normalized == ".." or normalized.startswith(f"..{os.sep}"):
        raise ToolError(
            f"Path escapes the workspace: {text}",
            kind=ToolErrorKind.WORKSPACE_VIOLATION,
            hint="Use a path inside the workspace.",
        )

    # Resolve component by component so symlinks are checked even when the
    # final path component does not exist yet.
    current = root
    candidate = Path(os.path.abspath(root / normalized))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ToolError(
            f"Path escapes the workspace: {text}",
            kind=ToolErrorKind.WORKSPACE_VIOLATION,
        ) from exc

    for part in relative.parts:
        current = current / part
        try:
            current = current.resolve(strict=False)
        except OSError as exc:
            raise ToolError(
                f"Cannot resolve path {text!r}: {exc}",
                kind=ToolErrorKind.WORKSPACE_VIOLATION,
            ) from exc
        if current != root and not current.is_relative_to(root):
            raise ToolError(
                f"Path escapes the workspace through a symlink: {text}",
                kind=ToolErrorKind.WORKSPACE_VIOLATION,
                hint="Resolve or remove the symlink, then retry.",
            )

    return current
