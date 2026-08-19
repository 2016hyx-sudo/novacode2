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
