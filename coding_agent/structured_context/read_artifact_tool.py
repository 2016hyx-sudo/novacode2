"""Read-only access to raw tool-result artifacts for the current session."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

from ..tools.base import ToolErrorKind, ToolResult


class ReadArtifactTool:
    name = "read_artifact"
    description = (
        "Read a raw tool-result artifact from the current session by artifact_id. "
        "Use an optional 1-based line range. Historical raw output is available even "
        "after the model-visible Tool Observation was compressed."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string", "description": "sha256 artifact identifier shown in a Tool Observation."},
            "start_line": {"type": "integer", "description": "First line to read, 1-based."},
            "end_line": {"type": "integer", "description": "Last line to read, inclusive."},
        },
        "required": ["artifact_id"],
    }

    def __init__(self, store_provider: Callable[[], Any], *, max_lines: int = 1000, max_chars: int = 32_000) -> None:
        self._store_provider = store_provider
        self.max_lines = max(1, max_lines)
        self.max_chars = max(1_000, max_chars)

    def execute(self, artifact_id: str, start_line: int = 1, end_line: int | None = None) -> ToolResult:
        if not artifact_id or not artifact_id.strip():
            return ToolResult.fail(
                "artifact_id must not be empty",
                metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value},
            )
        store = self._store_provider()
        if store is None:
            return ToolResult.fail(
                "No active structured session is bound to read_artifact",
                metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value},
            )
        entry = store.find(artifact_id.strip())
        if entry is None:
            return ToolResult.fail(
                f"Artifact not found in current session: {artifact_id}",
                metadata={"kind": ToolErrorKind.NOT_FOUND.value},
            )
        if entry.get("tool") == "write_file":
            return ToolResult.fail(
                "write_file artifacts are not exposed to the model",
                metadata={"kind": ToolErrorKind.WORKSPACE_VIOLATION.value},
            )
        raw = store.read(artifact_id.strip())
        if raw is None:
            return ToolResult.fail(
                f"Artifact content missing: {artifact_id}",
                metadata={"kind": ToolErrorKind.NOT_FOUND.value},
            )
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if not lines:
            return ToolResult.ok("(empty artifact)", artifact_id=artifact_id.strip(), lines=0)
        start_line = max(1, start_line)
        if end_line is None:
            end_line = len(lines)
        if end_line < start_line:
            return ToolResult.fail(
                f"end_line ({end_line}) must be >= start_line ({start_line})",
                metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value},
            )
        selected = lines[start_line - 1 : min(end_line, len(lines))]
        content = "\n".join(selected)
        if len(selected) > self.max_lines or len(content) > self.max_chars:
            return ToolResult.fail(
                f"Requested artifact range exceeds read_artifact limits "
                f"({self.max_lines} lines / {self.max_chars} chars). Narrow the range.",
                metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value},
            )
        suffix = ""
        if end_line < len(lines):
            suffix = f"\n... [{len(lines) - end_line} more line(s)]"
        return ToolResult.ok(
            content + suffix,
            artifact_id=artifact_id.strip(),
            start_line=start_line,
            end_line=min(end_line, len(lines)),
            total_lines=len(lines),
        )
