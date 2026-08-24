"""Workspace-bounded filesystem tools."""
from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .base import FunctionTool, Tool, ToolError, ToolErrorKind, ToolResult
from .workspace import resolve_workspace_path


def _tool_error_to_result(func: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
    @functools.wraps(func)
    def wrapper(**kwargs: Any) -> ToolResult:
        try:
            return func(**kwargs)
        except ToolError as exc:
            return ToolResult.fail(
                exc.message,
                metadata={"kind": exc.kind.value, "hint": exc.hint or ""},
            )
        except FileNotFoundError as exc:
            return ToolResult.fail(
                str(exc) or "File not found",
                metadata={"kind": ToolErrorKind.NOT_FOUND.value},
            )
        except IsADirectoryError as exc:
            return ToolResult.fail(
                f"Is a directory: {exc}",
                metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value},
            )
        except PermissionError as exc:
            return ToolResult.fail(
                f"Permission denied: {exc}",
                metadata={"kind": ToolErrorKind.WORKSPACE_VIOLATION.value},
            )
        except OSError as exc:
            return ToolResult.fail(
                f"Filesystem error: {exc}",
                metadata={"kind": ToolErrorKind.UNEXPECTED.value},
            )
        except Exception as exc:  # defensive: tool failures must not crash the loop
            return ToolResult.fail(
                f"Unexpected filesystem tool error: {exc}",
                metadata={"kind": ToolErrorKind.UNEXPECTED.value},
            )

    return wrapper


def build_filesystem_tools(workspace: Path, *, protected_rel: list[str] | None = None) -> list[Tool]:
    root = workspace
    protected = [Path(part.strip("/")) for part in protected_rel or [] if part.strip("/")]

    def is_protected(target: Path) -> bool:
        if not protected:
            return False
        try:
            rel = target.relative_to(root)
        except ValueError:
            return False
        return any(rel == item or rel.is_relative_to(item) for item in protected)

    def guard_protected(target: Path) -> None:
        if is_protected(target):
            raise ToolError(
                f"Path is protected by the structured context runtime: {display(target)}",
                kind=ToolErrorKind.WORKSPACE_VIOLATION,
                hint="Use read_artifact for historical tool output; do not access runtime state files.",
            )

    def display(path: Path) -> str:
        try:
            return str(path.relative_to(root)) or "."
        except ValueError:
            return str(path)

    @_tool_error_to_result
    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> ToolResult:
        target = resolve_workspace_path(root, path)
        guard_protected(target)
        if not target.exists():
            raise ToolError(
                f"File not found: {display(target)}",
                kind=ToolErrorKind.NOT_FOUND,
                hint="Use list_files to inspect the actual directory layout.",
            )
        if target.is_dir():
            raise ToolError(
                f"Path is a directory, not a file: {display(target)}",
                kind=ToolErrorKind.INVALID_ARGUMENTS,
            )
        text = target.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        if not lines:
            return ToolResult.ok("(empty file)")
        start_line = max(start_line, 1)
        if end_line is None:
            end_line = len(lines)
        if end_line < start_line:
            raise ToolError(
                f"end_line ({end_line}) must be >= start_line ({start_line})",
                kind=ToolErrorKind.INVALID_ARGUMENTS,
            )
        selected = lines[start_line - 1 : min(end_line, len(lines))]
        suffix = ""
        if end_line < len(lines):
            suffix = f"\n... [{len(lines) - end_line} more line(s)]"
        return ToolResult.ok(
            "\n".join(selected) + suffix,
            path=display(target),
            start_line=start_line,
            end_line=min(end_line, len(lines)),
        )

    @_tool_error_to_result
    def write_file(path: str, content: str) -> ToolResult:
        target = resolve_workspace_path(root, path)
        guard_protected(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        # Validation: the write must actually have happened.
        written = target.read_text(encoding="utf-8")
        if written != content:
            raise ToolError(
                f"Write verification failed for {display(target)}",
                kind=ToolErrorKind.UNEXPECTED,
            )
        return ToolResult.ok(
            f"Wrote {len(content)} characters to {display(target)}",
            path=display(target),
            bytes_written=len(content.encode("utf-8")),
        )

    @_tool_error_to_result
    def edit_file(
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
    ) -> ToolResult:
        if old_string == new_string:
            raise ToolError(
                "old_string and new_string are identical; no change would be made",
                kind=ToolErrorKind.INVALID_ARGUMENTS,
                hint="Provide a different new_string for an actual edit.",
            )
        target = resolve_workspace_path(root, path)
        guard_protected(target)
        if not target.exists():
            raise ToolError(
                f"File not found: {display(target)}",
                kind=ToolErrorKind.NOT_FOUND,
                hint="Use list_files or read_file first.",
            )
        if target.is_dir():
            raise ToolError(
                f"Path is a directory, not a file: {display(target)}",
                kind=ToolErrorKind.INVALID_ARGUMENTS,
            )
        text = target.read_text(encoding="utf-8", errors="replace")
        occurrences = text.count(old_string)
        if occurrences == 0:
            raise ToolError(
                f"old_string not found in {display(target)}",
                kind=ToolErrorKind.INVALID_ARGUMENTS,
                hint="Read the file and use the exact current text as old_string.",
            )
        if occurrences > 1 and not replace_all:
            raise ToolError(
                f"old_string found {occurrences} times; set replace_all=true to replace all",
                kind=ToolErrorKind.INVALID_ARGUMENTS,
                hint="Make old_string more specific or pass replace_all=true.",
            )
        updated = text.replace(old_string, new_string) if replace_all else text.replace(old_string, new_string, 1)
        target.write_text(updated, encoding="utf-8")
        # Validation: edit must produce an actual change.
        if target.read_text(encoding="utf-8") != updated:
            raise ToolError(
                f"Edit verification failed for {display(target)}",
                kind=ToolErrorKind.UNEXPECTED,
            )
        return ToolResult.ok(
            f"Edited {display(target)} ({occurrences if replace_all else 1} replacement(s))",
            path=display(target),
            replacements=occurrences if replace_all else 1,
        )

    @_tool_error_to_result
    def list_files(path: str = ".", recursive: bool = False, max_entries: int = 200) -> ToolResult:
        target = resolve_workspace_path(root, path)
        guard_protected(target)
        if not target.exists():
            raise ToolError(
                f"Path not found: {display(target)}",
                kind=ToolErrorKind.NOT_FOUND,
            )
        if target.is_file():
            return ToolResult.ok(display(target), entries=1)
        entries: list[str] = []
        iterator = target.rglob("*") if recursive else target.glob("*")
        for item in iterator:
            if is_protected(item):
                continue
            rel = display(item)
            if item.is_dir():
                rel += "/"
            entries.append(rel)
            if len(entries) >= max_entries:
                entries.append(f"... [truncated at {max_entries} entries]")
                break
        entries.sort()
        if not entries:
            return ToolResult.ok("(empty directory)", entries=0)
        return ToolResult.ok("\n".join(entries), entries=len(entries))

    @_tool_error_to_result
    def search_files(
        path: str = ".",
        pattern: str = "*",
        recursive: bool = True,
        max_results: int = 100,
    ) -> ToolResult:
        target = resolve_workspace_path(root, path)
        guard_protected(target)
        if not target.exists():
            raise ToolError(
                f"Path not found: {display(target)}",
                kind=ToolErrorKind.NOT_FOUND,
            )
        matches: list[str] = []
        iterator = target.rglob(pattern) if recursive else target.glob(pattern)
        for item in iterator:
            if is_protected(item):
                continue
            matches.append(display(item))
            if len(matches) >= max_results:
                matches.append(f"... [truncated at {max_results} results]")
                break
        matches.sort()
        if not matches:
            return ToolResult.ok("(no matching files)", matches=0)
        return ToolResult.ok("\n".join(matches), matches=len(matches))

    @_tool_error_to_result
    def grep_search(
        query: str,
        path: str = ".",
        file_pattern: str = "*",
        case_sensitive: bool = False,
        max_results: int = 100,
    ) -> ToolResult:
        if not query:
            raise ToolError("query must not be empty", kind=ToolErrorKind.INVALID_ARGUMENTS)
        target = resolve_workspace_path(root, path)
        guard_protected(target)
        if not target.exists():
            raise ToolError(
                f"Path not found: {display(target)}",
                kind=ToolErrorKind.NOT_FOUND,
            )
        needle = query if case_sensitive else query.lower()
        results: list[str] = []
        truncated = False
        for item in sorted(target.rglob(file_pattern)):
            if not item.is_file() or is_protected(item):
                continue
            try:
                for lineno, line in enumerate(item.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                    haystack = line if case_sensitive else line.lower()
                    if needle in haystack:
                        results.append(f"{display(item)}:{lineno}: {line}")
                        if len(results) >= max_results:
                            truncated = True
                            break
            except OSError:
                continue
            if truncated:
                break
        if truncated:
            results.append(f"... [truncated at {max_results} results]")
        if not results:
            return ToolResult.ok("(no matches)", matches=0)
        return ToolResult.ok("\n".join(results), matches=len(results))

    return [
        FunctionTool(
            name="read_file",
            description=(
                "Read a UTF-8 text file inside the workspace. Supports optional "
                "1-based line ranges."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path."},
                    "start_line": {"type": "integer", "description": "First line to read (1-based)."},
                    "end_line": {"type": "integer", "description": "Last line to read, inclusive."},
                },
                "required": ["path"],
            },
            func=read_file,
        ),
        FunctionTool(
            name="write_file",
            description="Create or overwrite a UTF-8 text file inside the workspace. Parent directories are created.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path."},
                    "content": {"type": "string", "description": "Complete new file content."},
                },
                "required": ["path", "content"],
            },
            func=write_file,
        ),
        FunctionTool(
            name="edit_file",
            description=(
                "Apply a precise textual edit to a file: replace old_string with "
                "new_string. The old_string must match exactly."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Workspace-relative file path."},
                    "old_string": {"type": "string", "description": "Exact text to find."},
                    "new_string": {"type": "string", "description": "Replacement text."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence."},
                },
                "required": ["path", "old_string", "new_string"],
            },
            func=edit_file,
        ),
        FunctionTool(
            name="list_files",
            description="List files and directories in a workspace-relative path.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory path, default workspace root."},
                    "recursive": {"type": "boolean", "description": "Recurse into subdirectories."},
                    "max_entries": {"type": "integer", "description": "Maximum entries to return."},
                },
            },
            func=list_files,
        ),
        FunctionTool(
            name="search_files",
            description="Find files by glob pattern, e.g. '*.py' or 'test_*.py'.",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to search."},
                    "pattern": {"type": "string", "description": "Glob pattern such as *.py."},
                    "recursive": {"type": "boolean", "description": "Recurse into subdirectories."},
                    "max_results": {"type": "integer", "description": "Maximum results."},
                },
                "required": ["pattern"],
            },
            func=search_files,
        ),
        FunctionTool(
            name="grep_search",
            description="Search file contents for a literal string and return file:line matches.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Literal text to search for."},
                    "path": {"type": "string", "description": "Directory to search."},
                    "file_pattern": {"type": "string", "description": "Glob for files to inspect."},
                    "case_sensitive": {"type": "boolean", "description": "Case-sensitive search."},
                    "max_results": {"type": "integer", "description": "Maximum results."},
                },
                "required": ["query"],
            },
            func=grep_search,
        ),
    ]
