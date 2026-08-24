"""Shell tool with basic safety controls.

This is a best-effort guard, not a sandbox: commands run with cwd fixed to the
workspace and with a timeout and output cap.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

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


@dataclass(frozen=True)
class ShellExecutionResult:
    """Provider-neutral result of executing one shell command."""

    returncode: int
    stdout: str = ""
    stderr: str = ""
    cwd: str = ""


@runtime_checkable
class ShellRunner(Protocol):
    """Execution backend used by :func:`build_shell_tool`."""

    def run(
        self,
        command: str,
        *,
        workspace: Path,
        timeout: float,
    ) -> ShellExecutionResult:
        """Execute *command* for *workspace* and return captured output."""


class LocalShellRunner:
    """Default backend: execute commands directly in the local workspace."""

    def run(
        self,
        command: str,
        *,
        workspace: Path,
        timeout: float,
    ) -> ShellExecutionResult:
        env = dict(os.environ)
        # Keep the lexical path: virtualenv executables are symlinks whose
        # resolved parent (/usr/bin) does not contain the ``python`` shim.
        interpreter_dir = str(Path(sys.executable).parent)
        current_path = env.get("PATH", "")
        env["PATH"] = f"{interpreter_dir}{os.pathsep}{current_path}" if current_path else interpreter_dir
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        return ShellExecutionResult(
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            cwd=str(workspace),
        )


def build_shell_tool(
    workspace: Path,
    *,
    shell_timeout: float = 60.0,
    max_output_chars: int = 20_000,
    protected_names: list[str] | None = None,
    runner: ShellRunner | None = None,
) -> FunctionTool:
    root = _ensure_workspace(workspace)
    shell_runner = runner or LocalShellRunner()
    protected_patterns: list[re.Pattern[str]] = []
    for name in protected_names or []:
        clean = name.strip("/")
        if clean:
            protected_patterns.append(re.compile(rf"(?:^|[\s;&|])\.?{re.escape(clean)}(?:/|\s|$)"))
    protected_patterns.append(re.compile(r"(?:^|[\s;&|])\.?agent(?:/|\s|$)"))

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

        patterns = [*_DENY_PATTERNS, *protected_patterns]
        for pattern in patterns:
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
            completed = shell_runner.run(
                command,
                workspace=root,
                timeout=effective_timeout,
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
            cwd=completed.cwd or str(root),
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
