"""Offline runnable demo of the NovaCode harness.

This demo uses a small deterministic ScriptedDemoProvider so it can run without
API keys. It exercises Planner Mode, the AgentLoop, filesystem tools, run_shell,
validation, session persistence and JSONL traces end to end.

For a real model run:
    OPENAI_API_KEY=... python main.py "add a health check function" --planner
"""
from __future__ import annotations

import tempfile
from collections.abc import Sequence
from pathlib import Path

from coding_agent import create_harness
from coding_agent.llm.base import (
    LLMResponse,
    Message,
    ToolCall,
    ToolSchema,
)
from coding_agent.tui import TUI
from config import AgentConfig, LLMConfig

APP_BEFORE = '''"""Tiny demo application."""


def greet(name: str) -> str:
    return f"Hello, {name}!"


if __name__ == "__main__":
    print(greet("world"))
'''

APP_AFTER = '''"""Tiny demo application."""


def greet(name: str) -> str:
    return f"Hello, {name}!"


def health_check() -> dict[str, str]:
    return {"status": "ok"}


if __name__ == "__main__":
    print(greet("world"))
'''


class ScriptedDemoProvider:
    """A deterministic LLM stand-in used only by demo.py, not by the harness."""

    def chat(
        self,
        messages: Sequence[Message],
        tools: Sequence[ToolSchema] | None = None,
        *,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        if not tools:
            # Planner call.
            return LLMResponse(
                text=(
                    "1. Inspect the demo application\n"
                    "2. Add a health_check function\n"
                    "3. Verify the file compiles"
                ),
                stop_reason="end_turn",
            )

        assistant_turns = sum(1 for message in messages if message.role == "assistant")
        if assistant_turns == 0:
            return LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="call_1", name="list_files", arguments={"path": "."}),
                    ToolCall(id="call_2", name="read_file", arguments={"path": "app.py"}),
                ],
                stop_reason="tool_calls",
            )
        if assistant_turns == 1:
            return LLMResponse(
                text=None,
                tool_calls=[
                    ToolCall(id="call_3", name="write_file", arguments={"path": "app.py", "content": APP_AFTER}),
                    ToolCall(
                        id="call_4",
                        name="run_shell",
                        arguments={"command": "python3 -m py_compile app.py"},
                    ),
                ],
                stop_reason="tool_calls",
            )
        return LLMResponse(
            text=(
                "Added health_check() to app.py and verified the change with "
                "`python3 -m py_compile app.py`. Task completed."
            ),
            stop_reason="end_turn",
        )


def main() -> int:
    workspace = Path(tempfile.mkdtemp(prefix="novacode-demo-"))
    (workspace / "app.py").write_text(APP_BEFORE, encoding="utf-8")

    config = AgentConfig(
        llm=LLMConfig(provider="openai", model="scripted-demo", api_key="demo"),
        workspace=workspace,
        planner_enabled=True,
        session_dir=workspace / ".sessions",
        trace_dir=workspace / ".traces",
    )

    # Swap the real provider for the deterministic demo provider before wiring.
    tui = TUI()
    harness = create_harness(config, listeners=[tui.handle_event])
    harness.provider = ScriptedDemoProvider()  # type: ignore[assignment]
    harness.planner.llm = harness.provider  # type: ignore[union-attr]

    session = harness.new_session("给 app.py 添加一个 health check 函数")
    result = harness.run_task(session)

    print()
    print(f"Demo workspace kept at: {workspace}")
    print(f"Final status: {result.status}")
    print(f"Final answer: {result.text}")
    print(f"Sessions: {list((workspace / '.sessions').glob('*.json'))}")
    print(f"Traces:   {list((workspace / '.traces').glob('*.jsonl'))}")
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
