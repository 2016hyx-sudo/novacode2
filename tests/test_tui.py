"""TUI tests: reasoning/thinking display toggle.

Thinking is surfaced on ``llm_response`` events only when
``NOVACODE_SHOW_THINKING`` (or the ``show_thinking`` constructor flag) is on,
and it is shown even on tool-call turns where the TUI otherwise prints nothing.
"""
from __future__ import annotations

import io

from rich.console import Console

from coding_agent.runtime.trace import TraceEvent
from coding_agent.tui.app import TUI


def _make_response_event(*, text=None, thinking=None, tool_calls=None) -> TraceEvent:
    return TraceEvent(
        type="llm_response",
        session_id="test",
        data={
            "text": text,
            "thinking": thinking,
            "tool_calls": tool_calls or [],
        },
    )


def _capture(*, show_thinking=None) -> tuple[TUI, str]:
    out = io.StringIO()
    # Wide console so lines never wrap; substrings stay contiguous in output.
    tui = TUI(
        console=Console(file=out, width=1000, force_terminal=False),
        show_thinking=show_thinking,
    )
    return tui, out


def test_thinking_hidden_by_default(monkeypatch) -> None:
    monkeypatch.delenv("NOVACODE_SHOW_THINKING", raising=False)
    tui, out = _capture()
    tui.handle_event(_make_response_event(text="Answer", thinking="deep thoughts"))
    rendered = out.getvalue()
    assert "[thinking]" not in rendered
    assert "deep thoughts" not in rendered
    assert "Answer" in rendered


def test_thinking_collapsed_when_long() -> None:
    tui, out = _capture(show_thinking=True)
    long_text = "start of reasoning " + "x" * 500 + " end"
    tui.handle_event(_make_response_event(text="Answer", thinking=long_text))
    rendered = out.getvalue()
    assert "[thinking]" in rendered
    assert "start of reasoning" in rendered
    assert long_text not in rendered  # collapsed, not the full text
    assert "Answer" in rendered


def test_think_command_expands_full_text() -> None:
    tui, out = _capture(show_thinking=True)
    long_text = "full reasoning " + "x" * 500
    tui.handle_event(_make_response_event(thinking=long_text))
    assert long_text not in out.getvalue()  # collapsed on the event
    tui._show_thinking()
    expanded = out.getvalue()
    assert "[thinking]" in expanded
    assert long_text in expanded


def test_collapsed_hint_only_in_interactive_mode() -> None:
    long_text = "reasoning " + "x" * 500

    # Single-shot run: no /think command, so no expand hint.
    tui, out = _capture(show_thinking=True)
    tui.handle_event(_make_response_event(thinking=long_text))
    assert "(/think to expand)" not in out.getvalue()

    # Interactive mode, long thinking: hint shown.
    tui2, out2 = _capture(show_thinking=True)
    tui2._interactive = True
    tui2.handle_event(_make_response_event(thinking=long_text))
    assert "(/think to expand)" in out2.getvalue()

    # Interactive mode, short thinking: fits fully, no hint.
    tui3, out3 = _capture(show_thinking=True)
    tui3._interactive = True
    tui3.handle_event(_make_response_event(thinking="short"))
    assert "(/think to expand)" not in out3.getvalue()


def test_thinking_shown_on_tool_call_turn() -> None:
    tui, out = _capture(show_thinking=True)
    tui.handle_event(_make_response_event(thinking="read first", tool_calls=["read_file"]))
    rendered = out.getvalue()
    assert "[thinking]" in rendered
    assert "read first" in rendered
    # No text on this turn, so no "Agent:" block.
    assert "Agent:" not in rendered


def test_text_shown_on_tool_call_turn() -> None:
    # Even with thinking off, the assistant message text that accompanies a
    # tool call must be shown rather than skipped.
    tui, out = _capture()
    tui.handle_event(_make_response_event(text="Let me read the file first", tool_calls=["read_file"]))
    rendered = out.getvalue()
    assert "Agent:" in rendered
    assert "Let me read the file first" in rendered


def test_empty_tool_turn_prints_nothing() -> None:
    # A tool call with no accompanying text prints nothing from llm_response;
    # the tool_call event renders the call itself.
    tui, out = _capture()
    tui.handle_event(_make_response_event(tool_calls=["read_file"]))
    assert "Agent:" not in out.getvalue()


def test_thinking_enabled_via_env(monkeypatch) -> None:
    monkeypatch.setenv("NOVACODE_SHOW_THINKING", "1")
    tui, out = _capture()
    tui.handle_event(_make_response_event(thinking="from env"))
    assert "from env" in out.getvalue()


def test_memory_surfaced_rendered() -> None:
    tui, out = _capture()
    tui.handle_event(
        TraceEvent(
            type="memory_surfaced",
            session_id="test",
            data={"name": "react_patterns.md", "type": "rule"},
        )
    )
    rendered = out.getvalue()
    assert "[memory]" in rendered
    assert "react_patterns.md" in rendered
    assert "rule" in rendered


def test_fold_event_rendered() -> None:
    tui, out = _capture()
    tui.handle_event(
        TraceEvent(
            type="fold_event",
            session_id="test",
            data={
                "event": "compacted",
                "before_tokens": 12000,
                "after_tokens": 3000,
                "savings_ratio": 0.75,
            },
        )
    )
    rendered = out.getvalue()
    assert "[fold]" in rendered
    assert "12,000" in rendered
    assert "3,000" in rendered
    assert "75.0%" in rendered


def test_session_metrics_rendered() -> None:
    tui, out = _capture()
    tui.handle_event(
        TraceEvent(
            type="session_metrics",
            session_id="test",
            data={
                "total_prompt_tokens": 8500,
                "total_completion_tokens": 1500,
                "total_tokens": 10000,
            },
        )
    )
    rendered = out.getvalue()
    assert "Session Metrics" in rendered
    assert "10,000" in rendered


def test_tool_result_diff_rendering() -> None:
    tui, out = _capture()
    diff_text = "--- a/foo.py\n+++ b/foo.py\n@@ -1,2 +1,3 @@\n+def bar(): pass\n"
    tui.handle_event(
        TraceEvent(
            type="tool_result",
            session_id="test",
            data={"name": "edit_file", "success": True, "output_preview": diff_text},
        )
    )
    rendered = out.getvalue()
    assert "diff:" in rendered
    assert "def bar(): pass" in rendered


def test_markdown_rendering_in_llm_response() -> None:
    tui, out = _capture()
    markdown_content = "### Solution\nHere is the code:\n```python\nx = 42\n```"
    tui.handle_event(_make_response_event(text=markdown_content))
    rendered = out.getvalue()
    assert "Agent:" in rendered
    assert "Solution" in rendered
    assert "x = 42" in rendered


def test_subagent_depth_visualization() -> None:
    tui, out = _capture()
    tui.handle_event(
        TraceEvent(
            type="subagent_start",
            session_id="test",
            data={"depth": 2, "max_steps": 5, "task": "subtask"},
        )
    )
    tui.handle_event(
        TraceEvent(
            type="step_start",
            session_id="test",
            data={"step": 1, "agent": "subagent_1"},
        )
    )
    tui.handle_event(
        TraceEvent(
            type="subagent_end",
            session_id="test",
            data={"depth": 2, "ok": True, "status": "completed", "steps_used": 1, "tool_calls_used": 1},
        )
    )
    rendered = out.getvalue()
    assert "[subagent]" in rendered
    assert "subtask" in rendered
    assert "subagent_1" in rendered


def test_show_help_table() -> None:
    tui, out = _capture()
    tui._show_help()
    rendered = out.getvalue()
    assert "NovaCode Interactive Commands" in rendered
    assert "/help" in rendered
    assert "/new" in rendered
    assert "/tokens" in rendered
    assert "/diff" in rendered


def test_token_tracking_and_show_tokens() -> None:
    tui, out = _capture()
    tui.handle_event(
        TraceEvent(
            type="llm_response",
            session_id="test",
            data={
                "text": "Done",
                "usage": {"prompt_tokens": 150, "completion_tokens": 50, "total_tokens": 200},
            },
        )
    )
    assert tui._total_usage["prompt_tokens"] == 150
    assert tui._total_usage["completion_tokens"] == 50
    assert tui._total_usage["total_tokens"] == 200

    tui._show_tokens()
    rendered = out.getvalue()
    assert "Token Usage Summary" in rendered
    assert "150" in rendered
    assert "50" in rendered
    assert "200" in rendered

