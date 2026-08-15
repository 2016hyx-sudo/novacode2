"""Workspace / git fingerprint scanning and expected-vs-actual diffing."""
from __future__ import annotations

import fnmatch
import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

from .models import DriftReport, WorkspaceExpected, sha256_text


def _run_git(root: Path, *args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def _hash_file(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _status_pairs(root: Path) -> list[tuple[str, str]]:
    """Return (status, path) pairs for tracked and untracked files."""
    root = Path(root)
    output = _run_git(root, "status", "--porcelain=v1", "--untracked-files=all")
    if output is None:
        return []
    pairs: list[tuple[str, str]] = []
    for raw_line in output.splitlines():
        if len(raw_line) < 4:
            continue
        status = raw_line[:2].strip()
        path = raw_line[3:].strip()
        if path.endswith("/") and status == "??":
            # Untracked directory: list its files directly.
            dir_path = root / path.rstrip("/")
            if dir_path.exists():
                for item in sorted(dir_path.rglob("*")):
                    if item.is_file():
                        rel = item.relative_to(root).as_posix()
                        pairs.append(("??", rel))
            continue
        if path and status:
            pairs.append((status, path))
    return pairs


class WorkspaceFingerprint:
    def __init__(self, workspace_root: Path, *, ignore_patterns: list[str] | None = None) -> None:
        self.root = Path(workspace_root).resolve()
        self.ignore_patterns = list(ignore_patterns or [])

    def is_ignored(self, rel_path: str) -> bool:
        rel = rel_path.replace(os.sep, "/")
        for pattern in self.ignore_patterns:
            if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, f"{pattern.rstrip('/')}/*"):
                return True
        return False

    def actual(self, extra_paths: list[str] | None = None, *, scan_all_non_git: bool = False) -> dict[str, Any]:
        branch = (_run_git(self.root, "rev-parse", "--abbrev-ref", "HEAD") or "").strip()
        head = (_run_git(self.root, "rev-parse", "HEAD") or "").strip()
        raw_status = _run_git(self.root, "status", "--porcelain=v1", "--untracked-files=all") or ""
        status_hash = sha256_text(raw_status)
        tracked: list[dict[str, Any]] = []
        untracked: list[dict[str, Any]] = []
        for status, rel in _status_pairs(self.root):
            if self.is_ignored(rel):
                continue
            item = {"path": rel, "status": status}
            file_hash = _hash_file(self.root / rel)
            if file_hash is not None:
                item["sha256"] = file_hash
            if status == "??":
                untracked.append(item)
            else:
                tracked.append(item)

        git_present = bool(head or branch)
        if scan_all_non_git and not git_present and not tracked and not untracked:
            for item in self.root.rglob("*"):
                if not item.is_file() or self.is_ignored(item.relative_to(self.root).as_posix()):
                    continue
                rel = item.relative_to(self.root).as_posix()
                if rel.startswith(".agent/") or rel.startswith(".git/"):
                    continue
                file_hash = _hash_file(item)
                untracked.append({"path": rel, "status": "??", "sha256": file_hash})

        hashes: dict[str, str | None] = {}
        for rel in extra_paths or []:
            if not self.is_ignored(rel):
                hashes[rel] = _hash_file(self.root / rel)
        return {
            "workspace_root": str(self.root),
            "git": {
                "present": bool(head or branch),
                "branch": branch,
                "head": head,
                "status_hash": status_hash,
            },
            "tracked_changes": tracked,
            "untracked": untracked,
            "extra_hashes": hashes,
        }

    def diff(self, expected: WorkspaceExpected, actual: dict[str, Any] | None = None) -> DriftReport:
        expected_extra_paths = [
            str(item["path"])
            for item in expected.expected_dirty + expected.expected_untracked + expected.postconditions
        ]
        actual = actual or self.actual(extra_paths=expected_extra_paths)
        report = DriftReport()
        actual_git = actual.get("git") or {}
        if expected.workspace_root and Path(expected.workspace_root).resolve() != self.root:
            report.git_divergence = {
                "expected_workspace_root": expected.workspace_root,
                "actual_workspace_root": str(self.root),
            }
            report.summary = "workspace root changed"
            report.severity = "STRUCTURAL"
            return report
        if expected.head and actual_git.get("head") and expected.head != actual_git["head"]:
            report.git_divergence = {
                "expected_head": expected.head,
                "actual_head": actual_git.get("head"),
                "expected_branch": expected.branch,
                "actual_branch": actual_git.get("branch"),
            }
        if expected.branch and actual_git.get("branch") and expected.branch != actual_git["branch"]:
            report.git_divergence = {
                "expected_head": expected.head,
                "actual_head": actual_git.get("head"),
                "expected_branch": expected.branch,
                "actual_branch": actual_git.get("branch"),
            }

        expected_paths: dict[str, dict[str, Any]] = {}
        for item in expected.expected_dirty + expected.expected_untracked + expected.postconditions:
            expected_paths[str(item["path"])] = dict(item)

        actual_paths: dict[str, dict[str, Any]] = {}
        for item in actual.get("tracked_changes") or []:
            actual_paths[str(item["path"])] = dict(item)
        for item in actual.get("untracked") or []:
            actual_paths[str(item["path"])] = dict(item)
        for path, file_hash in (actual.get("extra_hashes") or {}).items():
            existing = actual_paths.get(path)
            if existing is None:
                existing = {"path": path, "status": "??"}
                actual_paths[path] = existing
            if file_hash is not None:
                existing["sha256"] = file_hash

        for path, expected_item in expected_paths.items():
            actual_item = actual_paths.get(path)
            if actual_item is None:
                report.missing_expected_changes.append({"path": path, "expected": expected_item})
                continue
            expected_hash = expected_item.get("sha256")
            actual_hash = actual_item.get("sha256") or self._hash_path(path)
            if expected_hash and actual_hash and expected_hash != actual_hash:
                report.hash_mismatches.append(
                    {
                        "path": path,
                        "expected_sha256": expected_hash,
                        "actual_sha256": actual_hash,
                        "expected_status": expected_item.get("status"),
                        "actual_status": actual_item.get("status"),
                    }
                )

        for path, actual_item in actual_paths.items():
            if path not in expected_paths:
                report.unexpected_changes.append({"path": path, "actual": actual_item})

        if report.git_divergence:
            report.severity = "STRUCTURAL"
        elif report.hash_mismatches or report.missing_expected_changes or report.unexpected_changes:
            report.severity = "HIGH"
        else:
            report.severity = "NONE"
        report.summary = (
            f"drift={report.severity}; "
            f"unexpected={len(report.unexpected_changes)}; "
            f"missing={len(report.missing_expected_changes)}; "
            f"hash_mismatches={len(report.hash_mismatches)}"
        )
        return report

    def _hash_path(self, rel: str) -> str | None:
        return _hash_file(self.root / rel)

    def expected_from_actual(self, *, tracked: list[dict[str, Any]] | None = None, untracked: list[dict[str, Any]] | None = None) -> WorkspaceExpected:
        actual = self.actual(scan_all_non_git=True)
        git = actual["git"]
        tracked = tracked if tracked is not None else actual.get("tracked_changes") or []
        untracked = untracked if untracked is not None else actual.get("untracked") or []
        expected = WorkspaceExpected(
            workspace_root=str(self.root),
            branch=str(git.get("branch", "")),
            head=str(git.get("head", "")),
            status_hash=str(git.get("status_hash", "")),
            expected_dirty=[dict(x) for x in tracked],
            expected_untracked=[dict(x) for x in untracked],
            postconditions=[dict(x) for x in tracked],
        )
        expected.fingerprint = expected.recompute_fingerprint()
        return expected
