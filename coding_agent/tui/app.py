"""A deliberately simple Rich terminal UI."""
from __future__ import annotations

import json
import os
from typing import Any

# Load GNU readline before using builtin input(). Without it, backspacing over
# multi-byte characters (e.g. Chinese) can split UTF-8 bytes.
try:
    import readline as _readline  # noqa: F401
    _READLINE_AVAILABLE = True
except ImportError:  # Windows / environments without readline
    _READLINE_AVAILABLE = False

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text

from ..runtime.trace import TraceEvent


def _brief(value: Any, limit: int = 140) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# Collapsed thinking summary length shown inline in llm_response. The full
# reasoning text is kept and expanded on demand via /think.
THINKING_PREVIEW_LIMIT = 180


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a 1/true/yes/on style boolean from the environment."""
    value = os.environ.get(name, "")
    if not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class TUI:
    def __init__(
        self,
        console: Console | None = None,
        *,
        show_thinking: bool | None = None,
    ) -> None:
        self.console = console or Console()
        self._last_subagent_depth = 0
        # Reasoning/thinking preview in llm_response. Defaults to
        # NOVACODE_SHOW_THINKING; callers may override programmatically.
        self.show_thinking = (
            _env_bool("NOVACODE_SHOW_THINKING") if show_thinking is None else show_thinking
        )
        # Full thinking blocks recorded on llm_response, expanded via /think.
        self._thinking_blocks: list[str] = []
        # True while the interactive command loop is running (/think hint).
        self._interactive = False

    # ------------------------------------------------------------- event sink

    def handle_event(self, event: TraceEvent) -> None:
        data = event.data
        kind = event.type

        if kind == "session_start":
            body = Text()
            body.append("Session: ", style="bold")
            body.append(str(data.get("session_id", "")), style="bold")
            body.append("\nProvider/Model: ")
            body.append(f"{data.get('provider')}/{data.get('model')}")
            body.append("\nWorkspace: ")
            body.append(str(data.get("workspace")))
            body.append("\nTask: ")
            body.append(_brief(data.get("task")))
            self.console.print(Panel(body, title="NovaCode", border_style="blue"))
        elif kind == "session_resume":
            body = Text("Resumed session ", style="bold")
            body.append(str(data.get("session_id", "")), style="bold")
            body.append("\nTask: ")
            body.append(_brief(data.get("task")))
            self.console.print(Panel(body, border_style="blue"))
        elif kind == "planner_result":
            steps = data.get("steps") or []
            body = "\n".join(f"  {i}. {step}" for i, step in enumerate(steps, 1))
            self.console.print(Panel(body or "(no steps)", title="Plan", border_style="magenta"))
        elif kind == "planner_update":
            self.console.print(
                Text(f"[plan] correction: {_brief(data.get('correction'))}", style="magenta")
            )
        elif kind == "step_start":
            self.console.print(
                Rule(f"step {data.get('step')} · {data.get('agent', 'main')}", style="dim")
            )
        elif kind == "llm_retry":
            self.console.print(
                Text(f"[llm] retry {data.get('attempt')}: {_brief(data.get('error'))}", style="yellow")
            )
        elif kind == "llm_response":
            thinking = (data.get("thinking") or "").strip()
            if self.show_thinking and thinking:
                # Collapsed reasoning summary, dim so it never drowns the final
                # answer. The full text is kept and expanded via /think in
                # interactive mode. Shown even on tool-call turns.
                self._thinking_blocks.append(thinking)
                preview = _brief(thinking, THINKING_PREVIEW_LIMIT)
                hint = "  (/think to expand)" if self._interactive and preview.endswith("…") else ""
                self.console.print(Text(f"[thinking] {preview}{hint}", style="dim yellow"))
            # The assistant message text accompanies tool calls too; show it
            # instead of jumping straight to the tool call.
            text = data.get("text") or ""
            if text.strip():
                self.console.print(Text("Agent:", style="bold green"))
                self.console.print(Text(text.strip(), style="green"))
        elif kind == "tool_call":
            args = _brief(data.get("arguments"), 120)
            self.console.print(
                Text(f"[tool] {data.get('name')}({args})", style="cyan")
            )
        elif kind == "tool_retry":
            self.console.print(
                Text(
                    f"[retry] {data.get('name')} attempt {data.get('attempt')}: {_brief(data.get('error'))}",
                    style="yellow",
                )
            )
        elif kind == "tool_result":
            if data.get("success"):
                icon = "✓"
                style = "bright_black"
            else:
                icon = "✗"
                style = "red"
            preview = _brief(data.get("output_preview") or data.get("error"), 220)
            self.console.print(Text(f"  {icon} {data.get('name')}: {preview}", style=style))
        elif kind == "validation_failed":
            self.console.print(
                Text(f"[validate] ✗ {_brief(data.get('feedback'))}", style="yellow")
            )
        elif kind == "validation_passed":
            self.console.print(
                Text(f"[validate] ✓ {', '.join(data.get('checks') or [])}", style="green")
            )
        elif kind == "subagent_start":
            self._last_subagent_depth = int(data.get("depth", 1))
            self.console.print(
                Text(
                    f"[subagent] start (depth {data.get('depth')}, max_steps {data.get('max_steps')}): {_brief(data.get('task'))}",
                    style="magenta",
                )
            )
        elif kind == "subagent_end":
            ok = data.get("ok", False)
            icon = "✓" if ok else "✗"
            self.console.print(
                Text(
                    f"[subagent] {icon} depth {data.get('depth')} finished: "
                    f"status={data.get('status')}, steps={data.get('steps_used')}, tools={data.get('tool_calls_used')}",
                    style="magenta" if ok else "red",
                )
            )
        elif kind == "run_finished":
            status = data.get("status", "completed")
            icon = "✓" if status == "completed" else ("⚠" if status == "stopped" else "✗")
            style = "green" if status == "completed" else ("yellow" if status == "stopped" else "red")
            self.console.print(
                Panel(
                    f"{icon} Task finished with status [bold]{status}[/]\n"
                    f"steps={data.get('steps_used')} tool_calls={data.get('tool_calls_used')}",
                    title="Result",
                    border_style=style,
                )
            )
        elif kind == "error":
            self.console.print(Text(f"[error] {_brief(data.get('error'))}", style="red"))

    def _show_thinking(self) -> None:
        """Expand all recorded thinking blocks in full (/think)."""
        if not self._thinking_blocks:
            self.console.print(Text("No thinking recorded.", style="dim"))
            return
        for block in self._thinking_blocks:
            self.console.print(Text("[thinking]", style="bold yellow"))
            self.console.print(Text(block, style="dim yellow"))

    def _ask(self) -> str:
        """Read one line with a Unicode-safe, readline-aware prompt.

        The prompt is passed to builtin input() so readline knows the visible
        prompt width. ``\001`` / ``\002`` mark the ANSI color bytes as
        non-printing; without this, redrawing the last CJK character can make
        readline overwrite the ``>`` prompt itself.
        """
        if _READLINE_AVAILABLE and os.name == "posix":
            prompt = "\001\x1b[1;32m\002> \001\x1b[0m\002"
        else:
            prompt = "> "
        return input(prompt)

    # ------------------------------------------------------------ entry points

    def run_once(self, harness: Any, task: str, session_id: str | None = None) -> Any:
        try:
            session = harness.load_session(session_id) if session_id else harness.new_session(task)
        except FileNotFoundError as exc:
            self.console.print(Text(str(exc), style="red"))
            return None
        return self.run_task(harness, session, task)

    def run_interactive(self, harness: Any, session_id: str | None = None) -> None:
        self._interactive = True
        self.console.print(
            Panel(
                "Type a coding task. Commands: /new /session /plan /think /exit",
                title="NovaCode",
                border_style="blue",
            )
        )
        session = None
        if session_id:
            try:
                session = harness.load_session(session_id)
            except FileNotFoundError as exc:
                self.console.print(Text(str(exc), style="red"))
                return

        while True:
            try:
                raw = self._ask()
            except (EOFError, KeyboardInterrupt):
                self.console.print()
                break
            task = raw.strip()
            if not task:
                continue
            if task.startswith("/"):
                if task in {"/exit", "/quit"}:
                    break
                if task == "/new":
                    session = None
                    self.console.print(Text("Next task will start a new session.", style="dim"))
                elif task == "/session":
                    if session is None:
                        self.console.print(Text("No active session.", style="dim"))
                    else:
                        self.console.print(Text(f"Session ID: {session.id}", style="bold"))
                elif task == "/plan":
                    if session is None or not getattr(session, "plan", None):
                        self.console.print(Text("No plan yet.", style="dim"))
                    else:
                        body = "\n".join(f"  {i}. {step}" for i, step in enumerate(session.plan, 1))
                        self.console.print(Panel(body, title="Current Plan", border_style="magenta"))
                elif task == "/think":
                    self._show_thinking()
                else:
                    self.console.print(Text("Commands: /new /session /plan /think /exit", style="dim"))
                continue
            if session is None:
                session = harness.new_session(task)
            self.run_task(harness, session, task)

    def run_task(self, harness: Any, session: Any, task: str | None = None) -> Any:
        harness.trace.bind(session.id)
        try:
            result = harness.run_task(session, task)
        except Exception as exc:
            self.console.print(Text(f"Harness error: {exc}", style="red"))
            return None
        self.console.print(Text(f"Session ID: {session.id}", style="bright_black"))
        return result
