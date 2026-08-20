"""Tool construction helpers."""
from __future__ import annotations

from pathlib import Path

from config import Constraints

from .base import (
    FunctionTool,
    Tool,
    ToolCallResult,
    ToolError,
    ToolErrorKind,
    ToolResult,
)
from .filesystem import build_filesystem_tools
from .registry import ToolRegistry
from .shell import ShellRunner, build_shell_tool
from .subagent import SubagentTool


def build_tool_registry(
    workspace: Path,
    constraints: Constraints,
    *,
    subagent_tool: SubagentTool | None = None,
    protected_rel: list[str] | None = None,
    shell_runner: ShellRunner | None = None,
) -> ToolRegistry:
    """Build a registry with the standard tools for one agent nesting depth."""
    registry = ToolRegistry()
    for tool in build_filesystem_tools(workspace, protected_rel=protected_rel):
        registry.register(tool)
    registry.register(
        build_shell_tool(
            workspace,
            shell_timeout=constraints.shell_timeout,
            max_output_chars=constraints.max_output_chars,
            protected_names=protected_rel,
            runner=shell_runner,
        )
    )
    if subagent_tool is not None:
        registry.register(subagent_tool)
    return registry


__all__ = [
    "FunctionTool",
    "ShellRunner",
    "SubagentTool",
    "Tool",
    "ToolCallResult",
    "ToolError",
    "ToolErrorKind",
    "ToolRegistry",
    "ToolResult",
    "build_tool_registry",
]
