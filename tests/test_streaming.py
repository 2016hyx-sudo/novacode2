"""Streaming tests: token-by-token streaming and live TUI rendering."""
from __future__ import annotations

import io
from types import SimpleNamespace

from rich.console import Console

from coding_agent.llm import anthropic as anthropic_mod
from coding_agent.llm import openai as openai_mod
from coding_agent.llm.base import Message, StreamChunk
from coding_agent.runtime.trace import TraceEvent
from coding_agent.tui.app import TUI
from config import LLMConfig


def _capture_tui(*, show_thinking: bool = False) -> tuple[TUI, io.StringIO]:
    out = io.StringIO()
    tui = TUI(
        console=Console(file=out, width=1000, force_terminal=False),
        show_thinking=show_thinking,
    )
    return tui, out


# ------------------------------------------------------------- TUI streaming tests


def test_tui_renders_streamed_text_chunks() -> None:
    tui, out = _capture_tui()

    # Emit streaming chunks
    tui.handle_event(
        TraceEvent(type="llm_chunk", session_id="test", data={"delta_text": "Hello "})
    )
    tui.handle_event(
        TraceEvent(type="llm_chunk", session_id="test", data={"delta_text": "World!"})
    )
    # Finish response
    tui.handle_event(
        TraceEvent(
            type="llm_response",
            session_id="test",
            data={"text": "Hello World!", "usage": {"total_tokens": 10}},
        )
    )

    rendered = out.getvalue()
    assert "Agent:" in rendered
    assert "Hello World!" in rendered


def test_tui_renders_streamed_thinking_chunks() -> None:
    tui, out = _capture_tui(show_thinking=True)

    # Emit thinking chunks
    tui.handle_event(
        TraceEvent(type="llm_chunk", session_id="test", data={"delta_thinking": "step 1... "})
    )
    tui.handle_event(
        TraceEvent(type="llm_chunk", session_id="test", data={"delta_thinking": "step 2."})
    )
    # Emit answer chunks
    tui.handle_event(
        TraceEvent(type="llm_chunk", session_id="test", data={"delta_text": "Result is 42."})
    )
    # Finish response
    tui.handle_event(
        TraceEvent(
            type="llm_response",
            session_id="test",
            data={"text": "Result is 42.", "thinking": "step 1... step 2."},
        )
    )

    rendered = out.getvalue()
    assert "[thinking]" in rendered
    assert "step 1... step 2." in rendered
    assert "Agent:" in rendered
    assert "Result is 42." in rendered


# -------------------------------------------------------- Provider streaming tests


def test_openai_provider_streams_chunks_and_accumulates(monkeypatch) -> None:
    def make_chunk(content=None, reasoning=None, tool_calls=None, finish_reason=None, usage=None):
        delta = SimpleNamespace(
            content=content,
            reasoning_content=reasoning,
            tool_calls=tool_calls,
        )
        choice = SimpleNamespace(delta=delta, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice], usage=usage)

    chunks = [
        make_chunk(reasoning="thinking first "),
        make_chunk(reasoning="then act\n"),
        make_chunk(content="Hello "),
        make_chunk(content="world!"),
        make_chunk(finish_reason="stop", usage=SimpleNamespace(prompt_tokens=5, completion_tokens=10, total_tokens=15)),
    ]

    class FakeChatCompletions:
        def create(self, **kwargs):
            assert kwargs.get("stream") is True
            return iter(chunks)

    class FakeOpenAI:
        chat = SimpleNamespace(completions=FakeChatCompletions())

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(openai_mod.openai, "OpenAI", FakeOpenAI)

    config = LLMConfig(provider="openai", model="gpt-4o-mini", stream=True)
    provider = openai_mod.OpenAIProvider(config)

    emitted_chunks: list[StreamChunk] = []
    response = provider.chat(
        [Message(role="user", content="hi")],
        on_chunk=lambda c: emitted_chunks.append(c),
    )

    assert response.text == "Hello world!"
    assert response.thinking == "thinking first then act\n"
    assert response.stop_reason == "stop"
    assert response.usage["total_tokens"] == 15
    assert len(emitted_chunks) == 4
    assert emitted_chunks[0].delta_thinking == "thinking first "
    assert emitted_chunks[2].delta_text == "Hello "


def test_openai_provider_accumulates_streamed_tool_calls(monkeypatch) -> None:
    def make_tc_chunk(idx, call_id, name, args):
        fn = SimpleNamespace(name=name, arguments=args)
        return SimpleNamespace(index=idx, id=call_id, function=fn)

    chunks = [
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content=None,
                        tool_calls=[make_tc_chunk(0, "call_1", "read_file", '{"path":')],
                    ),
                    finish_reason=None,
                )
            ],
            usage=None,
        ),
        SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(
                        content=None,
                        reasoning_content=None,
                        tool_calls=[make_tc_chunk(0, None, "", ' "main.py"}')],
                    ),
                    finish_reason="tool_calls",
                )
            ],
            usage=None,
        ),
    ]

    class FakeChatCompletions:
        def create(self, **kwargs):
            return iter(chunks)

    class FakeOpenAI:
        chat = SimpleNamespace(completions=FakeChatCompletions())

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(openai_mod.openai, "OpenAI", FakeOpenAI)

    config = LLMConfig(provider="openai", model="gpt-4o-mini", stream=True)
    provider = openai_mod.OpenAIProvider(config)

    response = provider.chat([Message(role="user", content="read")], on_chunk=lambda c: None)
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].name == "read_file"
    assert response.tool_calls[0].arguments == {"path": "main.py"}


def test_anthropic_provider_streams_events_and_accumulates(monkeypatch) -> None:
    events = [
        SimpleNamespace(
            type="message_start",
            message=SimpleNamespace(
                usage=SimpleNamespace(input_tokens=12, cache_read_input_tokens=0, cache_creation_input_tokens=0)
            ),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=0,
            delta=SimpleNamespace(type="thinking_delta", thinking="anthropic thinking "),
        ),
        SimpleNamespace(
            type="content_block_delta",
            index=1,
            delta=SimpleNamespace(type="text_delta", text="anthropic text output"),
        ),
        SimpleNamespace(
            type="message_delta",
            delta=SimpleNamespace(stop_reason="end_turn"),
            usage=SimpleNamespace(output_tokens=8),
        ),
    ]

    class FakeMessages:
        def create(self, **kwargs):
            assert kwargs.get("stream") is True
            return iter(events)

    class FakeAnthropic:
        messages = FakeMessages()

        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr(anthropic_mod.anthropic, "Anthropic", FakeAnthropic)

    config = LLMConfig(provider="anthropic", model="claude-3-5-sonnet", stream=True)
    provider = anthropic_mod.AnthropicProvider(config)

    emitted_chunks: list[StreamChunk] = []
    response = provider.chat(
        [Message(role="user", content="hi")],
        on_chunk=lambda c: emitted_chunks.append(c),
    )

    assert response.text == "anthropic text output"
    assert response.thinking == "anthropic thinking "
    assert response.stop_reason == "end_turn"
    assert response.usage["input_tokens"] == 12
    assert response.usage["output_tokens"] == 8
    assert len(emitted_chunks) == 2
