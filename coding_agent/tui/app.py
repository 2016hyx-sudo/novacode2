"""A feature-rich Rich & prompt_toolkit terminal UI for NovaCode."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

# Optional prompt_toolkit for advanced line editing, multiline input, and history.
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import HTML
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.key_binding import KeyBindings

    _PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:
    _PROMPT_TOOLKIT_AVAILABLE = False

# Fallback to GNU readline before using builtin input().
try:
    import readline as _readline  # noqa: F401

    _READLINE_AVAILABLE = True
except ImportError:
    _READLINE_AVAILABLE = False

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from ..runtime.trace import TraceEvent


def _brief(value: Any, limit: int = 140) -> str:
    """Serialize and collapse value into a single-line summary."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, default=str)
        except TypeError:
            text = str(value)
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


# Collapsed thinking summary length shown inline in llm_response.
THINKING_PREVIEW_LIMIT = 180


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a 1/true/yes/on style boolean from the environment."""
    value = os.environ.get(name, "")
    if not value.strip():
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class TUI:
    """Enhanced Terminal User Interface for NovaCode."""

    def __init__(
        self,
        console: Console | None = None,
        *,
        show_thinking: bool | None = None,
    ) -> None:
        self.console = console or Console()
        self._last_subagent_depth = 0
        self.show_thinking = (
            _env_bool("NOVACODE_SHOW_THINKING") if show_thinking is None else show_thinking
        )
        # Full thinking blocks recorded on llm_response, expanded via /think.
        self._thinking_blocks: list[str] = []
        # True while the interactive command loop is running.
        self._interactive = False
        self._prompt_session: Any = None
        # Streaming buffer and state
        self._streaming_text: bool = False
        self._streaming_thinking: bool = False
        self._streamed_text_buffer: list[str] = []
        self._streamed_thinking_buffer: list[str] = []
        # Track cumulative token usage across the session.
        self._total_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }

    # ------------------------------------------------------------- event sink

    def handle_event(self, event: TraceEvent) -> None:
        data = event.data
        kind = event.type
        depth = int(data.get("depth") or self._last_subagent_depth or 0)
        indent = "  " * depth if depth > 0 else ""

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
            if data.get("message_count"):
                body.append(f"\nHistory messages: {data.get('message_count')}")
            self.console.print(Panel(body, border_style="blue"))
        elif kind == "planner_result":
            steps = data.get("steps") or []
            body = "\n".join(f"  {i}. {step}" for i, step in enumerate(steps, 1))
            self.console.print(Panel(body or "(no steps)", title="Plan", border_style="magenta"))
        elif kind == "planner_update":
            self.console.print(
                Text(f"{indent}[plan] correction: {_brief(data.get('correction'))}", style="magenta")
            )
        elif kind == "step_start":
            agent = data.get("agent", "main")
            step = data.get("step")
            self.console.print(
                Rule(f"{indent}step {step} · {agent}", style="dim")
            )
        elif kind == "llm_retry":
            self.console.print(
                Text(f"{indent}[llm] retry {data.get('attempt')}: {_brief(data.get('error'))}", style="yellow")
            )
        elif kind == "llm_chunk":
            delta_text = data.get("delta_text")
            delta_thinking = data.get("delta_thinking")

            if delta_thinking and self.show_thinking:
                if not self._streaming_thinking:
                    self._streaming_thinking = True
                    self.console.print(Text(f"{indent}[thinking] ", style="dim yellow"), end="")
                self.console.print(Text(delta_thinking, style="dim yellow"), end="")
                self._streamed_thinking_buffer.append(delta_thinking)

            if delta_text:
                if self._streaming_thinking:
                    self._streaming_thinking = False
                    self.console.print()
                if not self._streaming_text:
                    self._streaming_text = True
                    agent_label = "Agent:" if depth == 0 else f"Subagent (depth {depth}):"
                    self.console.print(Text(f"{indent}{agent_label}", style="bold green"))
                self.console.print(Text(delta_text, style="green"), end="")
                self._streamed_text_buffer.append(delta_text)
        elif kind == "llm_response":
            streamed_anything = self._streaming_text or self._streaming_thinking
            if streamed_anything:
                self.console.print()

            thinking = (data.get("thinking") or "".join(self._streamed_thinking_buffer)).strip()
            if self.show_thinking and thinking:
                self._thinking_blocks.append(thinking)
                if not streamed_anything:
                    preview = _brief(thinking, THINKING_PREVIEW_LIMIT)
                    hint = "  (/think to expand)" if self._interactive and preview.endswith("…") else ""
                    self.console.print(Text(f"{indent}[thinking] {preview}{hint}", style="dim yellow"))

            # Track token usage from response
            usage = data.get("usage") or data.get("normalized_usage")
            if isinstance(usage, dict):
                p = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                c = usage.get("completion_tokens") or usage.get("output_tokens") or 0
                t = usage.get("total_tokens") or (p + c)
                self._total_usage["prompt_tokens"] += int(p)
                self._total_usage["completion_tokens"] += int(c)
                self._total_usage["total_tokens"] += int(t)
                self._total_usage["calls"] += 1

            text = data.get("text") or ""
            if text.strip() and not streamed_anything:
                agent_label = "Agent:" if depth == 0 else f"Subagent (depth {depth}):"
                self.console.print(Text(f"{indent}{agent_label}", style="bold green"))
                # Render markdown with rich formatting and syntax highlighting
                self.console.print(Markdown(text.strip()))

            self._streaming_text = False
            self._streaming_thinking = False
            self._streamed_text_buffer.clear()
            self._streamed_thinking_buffer.clear()
        elif kind == "tool_call":
            args = _brief(data.get("arguments"), 120)
            self.console.print(
                Text(f"{indent}[tool] {data.get('name')}({args})", style="cyan")
            )
        elif kind == "tool_retry":
            self.console.print(
                Text(
                    f"{indent}[retry] {data.get('name')} attempt {data.get('attempt')}: {_brief(data.get('error'))}",
                    style="yellow",
                )
            )
        elif kind == "tool_result":
            success = data.get("success", False)
            icon = "✓" if success else "✗"
            style = "bright_black" if success else "red"
            raw_output = data.get("output_preview") or data.get("error") or ""
            name = data.get("name", "tool")

            # Check if output is a unified diff
            if (
                isinstance(raw_output, str)
                and ("\n@@ " in raw_output or raw_output.startswith("@@ ") or "\n--- " in raw_output)
            ):
                self.console.print(Text(f"{indent}  {icon} {name} diff:", style=style))
                self.console.print(
                    Syntax(raw_output.strip(), "diff", theme="monokai", line_numbers=False)
                )
            else:
                preview = _brief(raw_output, 220)
                self.console.print(Text(f"{indent}  {icon} {name}: {preview}", style=style))
        elif kind == "validation_failed":
            self.console.print(
                Text(f"{indent}[validate] ✗ {_brief(data.get('feedback'))}", style="yellow")
            )
        elif kind == "validation_passed":
            checks = ", ".join(data.get("checks") or [])
            self.console.print(
                Text(f"{indent}[validate] ✓ {checks}", style="green")
            )
        elif kind == "memory_surfaced":
            name = data.get("name", "unnamed")
            mtype = data.get("type", "memory")
            self.console.print(
                Text(f"{indent}  🧠 [memory] Surfaced: {name} ({mtype})", style="bright_cyan")
            )
        elif kind == "fold_event":
            event_name = data.get("event") or data.get("action") or "compacted"
            before = data.get("before_tokens")
            after = data.get("after_tokens")
            savings = data.get("savings_ratio")
            if before is not None and after is not None:
                ratio_str = f" (saved {savings:.1%})" if savings is not None else ""
                msg = f"Context {event_name}: {before:,} → {after:,} tokens{ratio_str}"
            else:
                msg = f"{event_name}: {_brief(data, 120)}"
            self.console.print(Text(f"{indent}  ⚡ [fold] {msg}", style="bright_magenta"))
        elif kind == "session_metrics":
            p = data.get("total_prompt_tokens") or data.get("prompt_tokens") or 0
            c = data.get("total_completion_tokens") or data.get("completion_tokens") or 0
            t = data.get("total_tokens") or (p + c)
            if t > 0:
                self.console.print(
                    Text(
                        f"  📊 Session Metrics: {t:,} total tokens (prompt: {p:,}, completion: {c:,})",
                        style="dim cyan",
                    )
                )
        elif kind == "subagent_start":
            self._last_subagent_depth = int(data.get("depth", 1))
            self.console.print(
                Text(
                    f"{indent}[subagent] start (depth {data.get('depth')}, max_steps {data.get('max_steps')}): {_brief(data.get('task'))}",
                    style="magenta",
                )
            )
        elif kind == "subagent_end":
            ok = data.get("ok", False)
            icon = "✓" if ok else "✗"
            depth_val = int(data.get("depth", 1))
            self.console.print(
                Text(
                    f"{indent}[subagent] {icon} depth {depth_val} finished: "
                    f"status={data.get('status')}, steps={data.get('steps_used')}, tools={data.get('tool_calls_used')}",
                    style="magenta" if ok else "red",
                )
            )
            self._last_subagent_depth = max(0, depth_val - 1)
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
            self.console.print(Text(f"{indent}[error] {_brief(data.get('error'))}", style="red"))

    # -------------------------------------------------------- slash commands

    def _show_thinking(self) -> None:
        """Expand all recorded thinking blocks in full (/think)."""
        if not self._thinking_blocks:
            self.console.print(Text("No thinking recorded.", style="dim"))
            return
        for i, block in enumerate(self._thinking_blocks, 1):
            title = f"[thinking #{i}]" if len(self._thinking_blocks) > 1 else "[thinking]"
            self.console.print(Text(title, style="bold yellow"))
            self.console.print(Text(block, style="dim yellow"))

    def _show_help(self) -> None:
        """Render help table with all available interactive commands."""
        table = Table(
            title="NovaCode Interactive Commands",
            show_header=True,
            header_style="bold blue",
            border_style="dim",
        )
        table.add_column("Command", style="bold cyan")
        table.add_column("Description", style="white")
        table.add_row("/help", "Show this commands overview")
        table.add_row("/new", "Start a new session and reset conversation history")
        table.add_row("/session", "Show details and metadata of the active session")
        table.add_row("/plan", "Display current execution plan steps")
        table.add_row("/think", "Expand and view all recorded reasoning/thinking blocks")
        table.add_row("/tokens, /usage", "Show cumulative token usage statistics")
        table.add_row("/diff, /status", "Inspect git status and workspace changes")
        table.add_row("/skill-eval", "Show skill bank and evolution candidate status")
        table.add_row("/clear, /cls", "Clear terminal screen")
        table.add_row("/exit, /quit", "Exit NovaCode interactive mode")
        self.console.print(table)
        self.console.print(
            Text(
                "Shortcuts: Alt+Enter / Esc+Enter for multiline · Ctrl+C to cancel task · Ctrl+D to exit",
                style="dim italic",
            )
        )

    def _show_session(self, session: Any) -> None:
        """Display metadata and current state of the active session."""
        if session is None:
            self.console.print(Text("No active session.", style="dim"))
            return
        table = Table(
            title="Active Session",
            show_header=True,
            header_style="bold blue",
            border_style="dim",
        )
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Session ID", str(getattr(session, "id", "unknown")))
        table.add_row(
            "Provider / Model",
            f"{getattr(session, 'provider', '-')}/{getattr(session, 'model', '-')}",
        )
        table.add_row("Status", str(getattr(session, "status", "-")))
        msg_count = len(getattr(session, "messages", []))
        table.add_row("Messages in Context", str(msg_count))
        task = getattr(session, "user_task", None)
        if task:
            table.add_row("Task", _brief(task, 80))
        self.console.print(table)

    def _show_plan(self, session: Any) -> None:
        """Display current plan steps."""
        plan = getattr(session, "plan", None) if session else None
        if not plan:
            self.console.print(Text("No plan yet.", style="dim"))
            return
        body = "\n".join(f"  {i}. {step}" for i, step in enumerate(plan, 1))
        self.console.print(Panel(body, title="Current Plan", border_style="magenta"))

    def _show_tokens(self) -> None:
        """Display cumulative token consumption."""
        calls = self._total_usage.get("calls", 0)
        p = self._total_usage.get("prompt_tokens", 0)
        c = self._total_usage.get("completion_tokens", 0)
        t = self._total_usage.get("total_tokens", 0)
        if calls == 0 and t == 0:
            self.console.print(Text("No LLM token usage recorded in this session.", style="dim"))
            return
        table = Table(
            title="Token Usage Summary",
            show_header=True,
            header_style="bold magenta",
            border_style="dim",
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right", style="green")
        table.add_row("LLM Calls", str(calls))
        table.add_row("Prompt Tokens", f"{p:,}")
        table.add_row("Completion Tokens", f"{c:,}")
        table.add_row("Total Tokens", f"{t:,}")
        self.console.print(table)

    def _show_git_status(self, workspace_path: Any = None) -> None:
        """Display git workspace modified files and diff stat."""
        cwd = str(workspace_path) if workspace_path else os.getcwd()
        try:
            status_proc = subprocess.run(
                ["git", "status", "-s"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if status_proc.returncode != 0:
                self.console.print(Text("Not a git repository or git unavailable.", style="dim yellow"))
                return
            status_out = status_proc.stdout.strip()
            if not status_out:
                self.console.print(Text("Working tree clean (no changes).", style="green"))
                return

            self.console.print(
                Panel(status_out, title="Git Status (Modified Files)", border_style="cyan")
            )

            diff_proc = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            diff_out = diff_proc.stdout.strip()
            if diff_out:
                self.console.print(Panel(diff_out, title="Diff Summary", border_style="dim cyan"))
        except Exception as exc:
            self.console.print(Text(f"Failed to get git status: {exc}", style="red"))

    def _show_skill_eval(self, harness: Any) -> None:
        from ..skills.eval import skill_evaluation_status

        bank = getattr(harness, "skill_bank", None)
        if bank is None:
            self.console.print(Text("Skills are disabled.", style="dim yellow"))
            return
        status = skill_evaluation_status(project_dir=bank.project_dir, user_dir=bank.user_dir)
        table = Table(title="Skill Evolution Status", header_style="bold blue")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")
        table.add_row("Skills", str(status["skills"]))
        table.add_row("Project / User", f"{status['project_skills']} / {status['user_skills']}")
        table.add_row("Candidates", str(status["candidates"]))
        table.add_row("Candidate status", _brief(status["candidate_status"], 300))
        self.console.print(table)

    # ------------------------------------------------------------ input layer

    def _get_prompt_session(self) -> Any:
        """Initialize or return the prompt_toolkit session with persistent history."""
        if not _PROMPT_TOOLKIT_AVAILABLE or not sys.stdin.isatty():
            return None
        if self._prompt_session is None:
            try:
                history_dir = os.path.expanduser("~/.novacode")
                os.makedirs(history_dir, exist_ok=True)
                history = FileHistory(os.path.join(history_dir, "history"))
                kb = KeyBindings()

                @kb.add("escape", "enter")
                def _insert_newline(event: Any) -> None:
                    event.current_buffer.insert_text("\n")

                self._prompt_session = PromptSession(
                    history=history,
                    key_bindings=kb,
                    multiline=False,
                )
            except Exception:
                self._prompt_session = None
        return self._prompt_session

    def _ask(self) -> str:
        """Read one command or task with prompt_toolkit or readline fallback."""
        ps = self._get_prompt_session()
        if ps is not None:
            try:
                return ps.prompt(HTML("<ansigreen><b>&gt; </b></ansigreen>"))
            except (EOFError, KeyboardInterrupt):
                raise
            except Exception:
                # Fallback to standard readline/input
                self._prompt_session = None

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
                "Type a coding task. Commands: /help /new /session /plan /think /tokens /diff /skill-eval /clear /exit",
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
            except KeyboardInterrupt:
                self.console.print()
                continue
            except EOFError:
                self.console.print()
                break

            task = raw.strip()
            if not task:
                continue

            # Support multiline input enclosed by triple quotes in basic terminal mode
            if (task.startswith('"""') and (task.count('"""') % 2 != 0 or len(task) == 3)) or (
                task.startswith("'''") and (task.count("'''") % 2 != 0 or len(task) == 3)
            ):
                quote = task[:3]
                lines = [task[3:]]
                while True:
                    try:
                        line = input("... ")
                    except (EOFError, KeyboardInterrupt):
                        break
                    if quote in line:
                        lines.append(line.replace(quote, ""))
                        break
                    lines.append(line)
                task = "\n".join(lines).strip()

            if task.startswith("/"):
                cmd = task.split()[0].lower()
                if cmd in {"/exit", "/quit"}:
                    if session is not None and hasattr(harness, "dormancy_episode"):
                        harness.dormancy_episode(session)
                    break
                if cmd == "/help":
                    self._show_help()
                elif cmd == "/new":
                    if session is not None and hasattr(harness, "supersede_episode"):
                        harness.supersede_episode(session)
                    session = None
                    self._thinking_blocks.clear()
                    self._total_usage = {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "calls": 0,
                    }
                    self.console.print(Text("Next task will start a new session.", style="dim"))
                elif cmd == "/session":
                    self._show_session(session)
                elif cmd == "/plan":
                    self._show_plan(session)
                elif cmd == "/think":
                    self._show_thinking()
                elif cmd in {"/tokens", "/usage"}:
                    self._show_tokens()
                elif cmd in {"/diff", "/status"}:
                    ws = getattr(harness, "workspace", None)
                    self._show_git_status(ws)
                elif cmd == "/skill-eval":
                    self._show_skill_eval(harness)
                elif cmd in {"/clear", "/cls"}:
                    self.console.clear()
                else:
                    self.console.print(
                        Text(
                            "Commands: /help /new /session /plan /think /tokens /diff /skill-eval /clear /exit",
                            style="dim",
                        )
                    )
                continue

            if session is None:
                session = harness.new_session(task)
            self.run_task(harness, session, task)

    def run_task(self, harness: Any, session: Any, task: str | None = None) -> Any:
        harness.trace.bind(session.id)
        try:
            result = harness.run_task(session, task)
        except KeyboardInterrupt:
            if hasattr(harness, "cancel_episode"):
                harness.cancel_episode(session)
            self.console.print(Text("\n[Task cancelled by user]", style="yellow bold"))
            return None
        except Exception as exc:
            self.console.print(Text(f"Harness error: {exc}", style="red"))
            return None
        self.console.print(Text(f"Session ID: {session.id}", style="bright_black"))
        return result
