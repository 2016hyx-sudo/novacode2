"""Shell tool with basic safety controls.

This is a best-effort guard, not a sandbox: commands run with cwd fixed to the
workspace and with a timeout and output cap.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

from .base import FunctionTool, ToolErrorKind, ToolResult
from .workspace import _ensure_workspace

_DENY_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+(-[a-zA-Z]+\s+)*/"),
    re.compile(r"\brm\s+-rf\s+[^ ]*\s+[^ ]*"),
    re.compile(r"\bmkfs(\.[a-z0-9]+)?"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bshutdown\b"),
    re.compile(r"\breboot\b"),
    re.compile(r"\bhalt\b"),
    re.compile(r"\bpoweroff\b"),
    re.compile(r"\bdd\b[^\n]*\bof=/dev/"),
    re.compile(r":\(\)\s*\{"),
    re.compile(r"\bgit\s+push\b[^\n]*--force"),
    # Best-effort workspace boundary: obvious relative-path escapes.
    re.compile(r"(?:^|[;&|]\s*)cd\s+\.\."),
    re.compile(r"(?:^|[\s;&|])\.\./"),
]


def build_shell_tool(
    workspace: Path,
    *,
    shell_timeout: float = 60.0,
    max_output_chars: int = 20_000,
) -> FunctionTool:
    root = _ensure_workspace(workspace)

    def _truncate(text: str) -> str:
        if len(text) <= max_output_chars:
            return text
        omitted = len(text) - max_output_chars
        return text[:max_output_chars] + f"\n... [output truncated, {omitted} chars omitted]"

    def run_shell(command: str, timeout: float | None = None) -> ToolResult:
        if not command or not command.strip():
            return ToolResult.fail(
                "command must not be empty",
                metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value},
            )

        for pattern in _DENY_PATTERNS:
            if pattern.search(command):
                return ToolResult.fail(
                    f"Shell command rejected by safety policy: {command[:200]}",
                    metadata={
                        "kind": ToolErrorKind.WORKSPACE_VIOLATION.value,
                        "hint": "Avoid destructive commands and do not reference paths outside the workspace.",
                    },
                )

        effective_timeout = shell_timeout
        if timeout is not None:
            try:
                effective_timeout = min(max(float(timeout), 0.1), shell_timeout)
            except (TypeError, ValueError):
                return ToolResult.fail(
                    f"Invalid timeout value: {timeout!r}",
                    metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value},
                )

        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(root),
                capture_output=True,
                text=True,
                timeout=effective_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            partial = _truncate(
                ((exc.stdout or "") if isinstance(exc.stdout, str) else "")
                + ((exc.stderr or "") if isinstance(exc.stderr, str) else "")
            )
            return ToolResult.fail(
                f"Shell command timed out after {effective_timeout:.1f}s",
                output=partial,
                metadata={
                    "kind": ToolErrorKind.SHELL_TIMEOUT.value,
                    "hint": "Split the work into smaller commands or increase the timeout.",
                    "timeout": effective_timeout,
                },
            )
        except OSError as exc:
            return ToolResult.fail(
                f"Failed to start shell command: {exc}",
                metadata={"kind": ToolErrorKind.UNEXPECTED.value},
            )

        output = _truncate((completed.stdout or "") + (completed.stderr or ""))
        if completed.returncode != 0:
            return ToolResult.fail(
                f"Shell command exited with code {completed.returncode}",
                output=output,
                metadata={
                    "kind": ToolErrorKind.SHELL_FAILED.value,
                    "exit_code": completed.returncode,
                    "hint": "Read stdout/stderr above, fix the issue, and retry with a corrected command.",
                },
            )
        return ToolResult.ok(
            output or "(command completed successfully with no output)",
            exit_code=0,
            cwd=str(root),
        )

    return FunctionTool(
        name="run_shell",
        description=(
            "Run a shell command inside the workspace directory. The command has a "
            "timeout and its output length is limited. Use it for tests, git, build "
            "commands and read-only inspections. Never run destructive commands."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to execute."},
                "timeout": {"type": "number", "description": "Optional timeout in seconds."},
            },
            "required": ["command"],
        },
        func=run_shell,
    )
