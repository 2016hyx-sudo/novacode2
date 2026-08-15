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
from .shell import build_shell_tool
from .subagent import SubagentTool


def build_tool_registry(
    workspace: Path,
    constraints: Constraints,
    *,
    subagent_tool: SubagentTool | None = None,
) -> ToolRegistry:
    """Build a registry with the standard tools for one agent nesting depth."""
    registry = ToolRegistry()
    for tool in build_filesystem_tools(workspace):
        registry.register(tool)
    registry.register(
        build_shell_tool(
            workspace,
            shell_timeout=constraints.shell_timeout,
            max_output_chars=constraints.max_output_chars,
        )
    )
    if subagent_tool is not None:
        registry.register(subagent_tool)
    return registry


__all__ = [
    "FunctionTool",
    "SubagentTool",
    "Tool",
    "ToolCallResult",
    "ToolError",
    "ToolErrorKind",
    "ToolRegistry",
    "ToolResult",
    "build_tool_registry",
]
