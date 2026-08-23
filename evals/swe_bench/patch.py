"""Reliable model-patch export for SWE-bench predictions."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path


class PatchExportError(RuntimeError):
    """Raised when a workspace cannot produce an applicable Git patch."""


def _git(workspace: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        ["git", "-C", str(workspace), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise PatchExportError(
            f"git {' '.join(args)} failed with {completed.returncode}: "
            f"{(completed.stdout + completed.stderr).strip()}"
        )
    return completed


def export_patch(workspace: Path, base_commit: str, output_path: Path) -> str:
    """Export tracked, staged, committed and non-ignored untracked changes.

    The resulting patch is checked against a detached worktree at ``base_commit``
    before it is returned.
    """

    workspace = workspace.expanduser().resolve()
    output_path = output_path.expanduser().resolve()
    _git(workspace, "rev-parse", "--verify", f"{base_commit}^{{commit}}")

    untracked_raw = subprocess.run(
        [
            "git",
            "-C",
            str(workspace),
            "ls-files",
            "-z",
            "--others",
            "--exclude-standard",
        ],
        capture_output=True,
        check=False,
    )
    if untracked_raw.returncode != 0:
        raise PatchExportError(untracked_raw.stderr.decode("utf-8", errors="replace"))
    untracked = [
        item.decode("utf-8", errors="surrogateescape")
        for item in untracked_raw.stdout.split(b"\0")
        if item
    ]
    if untracked:
        _git(workspace, "add", "--intent-to-add", "--", *untracked)

    diff = _git(
        workspace,
        "diff",
        "--binary",
        "--no-ext-diff",
        base_commit,
        "--",
    ).stdout
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(diff, encoding="utf-8", errors="surrogateescape")
    if diff:
        _verify_patch(workspace, base_commit, output_path)
    return diff


def _verify_patch(workspace: Path, base_commit: str, patch_path: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="novacode-swe-patch-check-", dir=workspace.parent) as raw:
        check_root = Path(raw)
        _git(workspace, "worktree", "add", "--detach", str(check_root), base_commit)
        try:
            completed = subprocess.run(
                ["git", "-C", str(check_root), "apply", "--check", str(patch_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                raise PatchExportError(
                    f"exported patch does not apply cleanly to {base_commit}: "
                    f"{(completed.stdout + completed.stderr).strip()}"
                )
        finally:
            _git(workspace, "worktree", "remove", "--force", str(check_root), check=False)
