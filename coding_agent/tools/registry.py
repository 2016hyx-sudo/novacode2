"""Simple tool registry."""
from __future__ import annotations

from collections.abc import Iterable

from ..llm.base import ToolSchema
from .base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not getattr(tool, "name", ""):
            raise ValueError("Tool must have a name")
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def schemas(self) -> list[ToolSchema]:
        return [
            ToolSchema(
                name=tool.name,
                description=tool.description,
                parameters=dict(tool.parameters or {}),
            )
            for tool in self._tools.values()
        ]

    def clone(self, *, exclude: Iterable[str] = ()) -> ToolRegistry:
        clone = ToolRegistry()
        excluded = set(exclude)
        for name, tool in self._tools.items():
            if name not in excluded:
                clone.register(tool)
        return clone

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
