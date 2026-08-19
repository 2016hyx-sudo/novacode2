"""Provider adapter tests: thinking/reasoning preservation and effort control.

The adapters build a real SDK client in __init__, so these tests monkeypatch
``anthropic.Anthropic`` / ``openai.OpenAI`` with fakes that capture kwargs and
return canned responses.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from config import LLMConfig
from coding_agent.llm import anthropic as anthropic_mod
from coding_agent.llm import openai as openai_mod
from coding_agent.llm.base import Message, ToolCall


def make_anthropic_client(response, *, fail_with=None):
    calls = []

    class FakeMessages:
        def create(self, **kwargs):
            calls.append(kwargs)
            if fail_with is not None:
                raise fail_with
            return response

    class FakeClient:
        messages = FakeMessages()

        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    return FakeClient, calls


def make_usage(**kw):
    return SimpleNamespace(
        input_tokens=kw.get("input_tokens", 10),
        output_tokens=kw.get("output_tokens", 20),
        cache_read_input_tokens=kw.get("cache_read_input_tokens", 0),
        cache_creation_input_tokens=kw.get("cache_creation_input_tokens", 0),
    )


def thinking_block(text="think hard", signature="sig-123"):
    return SimpleNamespace(type="thinking", thinking=text, signature=signature)


def text_block(text):
    return SimpleNamespace(type="text", text=text)


def tool_use_block(call_id="call-1", name="read_file", arguments=None):
    return SimpleNamespace(
        type="tool_use", id=call_id, name=name, input=arguments or {"path": "a.py"}
    )


def anthropic_message(*blocks, stop_reason="end_turn", usage=None):
    return SimpleNamespace(
        content=list(blocks), stop_reason=stop_reason, usage=usage or make_usage()
    )


@pytest.fixture
def anthropic_config():
    return LLMConfig(provider="anthropic", model="deepseek-v4-flash", max_tokens=2000)


# ------------------------------------------------------------------ Anthropic


def test_anthropic_parses_thinking_and_text(monkeypatch, anthropic_config):
    response = anthropic_message(
        thinking_block("first step"),
        text_block("Answer: 2"),
        stop_reason="end_turn",
    )
    fake_client, calls = make_anthropic_client(response)
    monkeypatch.setattr(anthropic_mod.anthropic, "Anthropic", fake_client)

    result = anthropic_mod.AnthropicProvider(anthropic_config).chat(
        [Message(role="user", content="1+1=?")]
    )

    assert result.text == "Answer: 2"
    assert result.thinking == "first step"
    assert result.raw_content == [
        {"type": "thinking", "thinking": "first step", "signature": "sig-123"},
        {"type": "text", "text": "Answer: 2"},
    ]
    # Plain request: no thinking/effort params unless configured.
    assert "thinking" not in calls[0]
    assert "output_config" not in calls[0]


def test_anthropic_tool_turn_keeps_blocks_and_replays_verbatim(
    monkeypatch, anthropic_config
):
    response = anthropic_message(
        thinking_block("need to read"),
        tool_use_block(),
        stop_reason="tool_use",
    )
    fake_client, calls = make_anthropic_client(response)
    monkeypatch.setattr(anthropic_mod.anthropic, "Anthropic", fake_client)

    provider = anthropic_mod.AnthropicProvider(anthropic_config)
    result = provider.chat([Message(role="user", content="read a.py")])

    assert result.text is None  # thinking + tool_use only, no text
    assert [c.name for c in result.tool_calls] == ["read_file"]
    assert result.raw_content == [
        {"type": "thinking", "thinking": "need to read", "signature": "sig-123"},
        {"type": "tool_use", "id": "call-1", "name": "read_file", "input": {"path": "a.py"}},
    ]

    # Second turn: the stored assistant message (raw blocks) plus the tool
    # result must be replayed unchanged, including the thinking signature.
    history = [
        Message(role="user", content="read a.py"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[ToolCall(id="call-1", name="read_file", arguments={"path": "a.py"})],
            raw_content=result.raw_content,
        ),
        Message(role="tool", content="file contents", tool_call_id="call-1", name="read_file"),
    ]
    final = anthropic_message(text_block("done"))
    final_client, final_calls = make_anthropic_client(final)
    monkeypatch.setattr(anthropic_mod.anthropic, "Anthropic", final_client)
    anthropic_mod.AnthropicProvider(anthropic_config).chat(history)

    assistant_payload = final_calls[0]["messages"][1]["content"]
    assert assistant_payload == [
        {"type": "thinking", "thinking": "need to read", "signature": "sig-123"},
        {"type": "tool_use", "id": "call-1", "name": "read_file", "input": {"path": "a.py"}},
    ]
    assert final_calls[0]["messages"][2]["content"][0]["tool_use_id"] == "call-1"


def test_anthropic_reasoning_effort_params(monkeypatch, anthropic_config):
    fake_client, calls = make_anthropic_client(anthropic_message(text_block("ok")))
    monkeypatch.setattr(anthropic_mod.anthropic, "Anthropic", fake_client)
    provider = anthropic_mod.AnthropicProvider(anthropic_config)
    msg = [Message(role="user", content="hi")]

    provider.chat(msg, reasoning_effort="none")
    assert calls[-1]["thinking"] == {"type": "disabled"}

    provider.chat(msg, reasoning_effort="low")
    assert calls[-1]["output_config"] == {"effort": "low"}

    provider.chat(msg, reasoning_effort="high")
    assert calls[-1]["output_config"] == {"effort": "high"}

    # Explicit call-level effort beats the config default.
    provider.chat(msg, reasoning_effort="none")
    assert calls[-1]["thinking"] == {"type": "disabled"}


def test_anthropic_config_effort_default(monkeypatch, anthropic_config):
    config = LLMConfig(
        provider="anthropic",
        model="deepseek-v4-flash",
        max_tokens=2000,
        reasoning_effort="low",
        secondary_reasoning_effort="none",
    )
    fake_client, calls = make_anthropic_client(anthropic_message(text_block("ok")))
    monkeypatch.setattr(anthropic_mod.anthropic, "Anthropic", fake_client)
    provider = anthropic_mod.AnthropicProvider(config)
    msg = [Message(role="user", content="hi")]

    provider.chat(msg)  # main agent: config value
    assert calls[-1]["output_config"] == {"effort": "low"}

    provider.chat(msg, reasoning_effort="none")  # subagent/fold override
    assert calls[-1]["thinking"] == {"type": "disabled"}


# ------------------------------------------------------------------ OpenAI


def make_openai_client(response):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return response

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

        def __init__(self, *args, **kwargs):
            pass

    return FakeClient, calls


def openai_choice(content="Answer: 2", tool_calls=None, reasoning_content=None):
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            tool_calls=tool_calls,
            reasoning_content=reasoning_content,
        ),
        finish_reason="stop",
    )


def openai_tool_call(call_id, name, arguments):
    """Fake SDK tool-call object; pydantic objects expose model_dump()."""
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
        model_dump=lambda: {
            "id": call_id,
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        },
    )


def test_openai_parses_reasoning_and_replays(monkeypatch):
    config = LLMConfig(provider="openai", model="deepseek-v4-flash", max_tokens=2000)
    response = SimpleNamespace(
        choices=[
            openai_choice(
                content="",
                tool_calls=[openai_tool_call("c1", "read_file", '{"path": "a.py"}')],
                reasoning_content="read first",
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=SimpleNamespace(cached_tokens=None),
        ),
    )
    fake_client, calls = make_openai_client(response)
    monkeypatch.setattr(openai_mod.openai, "OpenAI", fake_client)
    provider = openai_mod.OpenAIProvider(config)

    result = provider.chat([Message(role="user", content="read a.py")])
    assert result.thinking == "read first"
    assert result.raw_content == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
            }
        ],
        "reasoning_content": "read first",
    }

    # Replay passes the stored message through verbatim.
    history = [
        Message(role="user", content="read a.py"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
            raw_content=result.raw_content,
        ),
        Message(role="tool", content="ok", tool_call_id="c1", name="read_file"),
    ]
    fake2, calls2 = make_openai_client(SimpleNamespace(choices=[openai_choice()], usage=None))
    monkeypatch.setattr(openai_mod.openai, "OpenAI", fake2)
    openai_mod.OpenAIProvider(config).chat(history)
    assert calls2[0]["messages"][1] == result.raw_content


def test_openai_preserves_empty_reasoning_content(monkeypatch):
    # DeepSeek V4 returns reasoning_content as an empty string on some tool
    # turns; it must still be echoed back verbatim or the API returns a 400.
    config = LLMConfig(provider="openai", model="deepseek-v4-flash", max_tokens=2000)
    response = SimpleNamespace(
        choices=[
            openai_choice(
                content="",
                tool_calls=[openai_tool_call("c1", "read_file", '{"path": "a.py"}')],
                reasoning_content="",
            )
        ],
        usage=None,
    )
    fake_client, calls = make_openai_client(response)
    monkeypatch.setattr(openai_mod.openai, "OpenAI", fake_client)
    result = openai_mod.OpenAIProvider(config).chat([Message(role="user", content="read a.py")])
    assert result.raw_content["reasoning_content"] == ""

    history = [
        Message(role="user", content="read a.py"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="read_file", arguments={"path": "a.py"})],
            raw_content=result.raw_content,
        ),
        Message(role="tool", content="ok", tool_call_id="c1", name="read_file"),
    ]
    fake2, calls2 = make_openai_client(SimpleNamespace(choices=[openai_choice()], usage=None))
    monkeypatch.setattr(openai_mod.openai, "OpenAI", fake2)
    openai_mod.OpenAIProvider(config).chat(history)
    assert calls2[0]["messages"][1]["reasoning_content"] == ""


def test_openai_reasoning_effort_params(monkeypatch):
    config = LLMConfig(provider="openai", model="deepseek-v4-flash", max_tokens=2000)
    fake_client, calls = make_openai_client(
        SimpleNamespace(choices=[openai_choice()], usage=None)
    )
    monkeypatch.setattr(openai_mod.openai, "OpenAI", fake_client)
    provider = openai_mod.OpenAIProvider(config)
    msg = [Message(role="user", content="hi")]

    provider.chat(msg, reasoning_effort="none")
    assert calls[-1]["extra_body"] == {"thinking": {"type": "disabled"}}

    provider.chat(msg, reasoning_effort="max")
    assert calls[-1]["reasoning_effort"] == "max"


# ------------------------------------------------------------------ config


def test_reasoning_effort_env_parsing(monkeypatch):
    monkeypatch.setenv("NOVACODE_REASONING_EFFORT", "max")
    monkeypatch.setenv("NOVACODE_SECONDARY_REASONING_EFFORT", "low")
    config = LLMConfig.from_env()
    assert config.reasoning_effort == "max"
    assert config.secondary_reasoning_effort == "low"

    monkeypatch.delenv("NOVACODE_REASONING_EFFORT")
    monkeypatch.delenv("NOVACODE_SECONDARY_REASONING_EFFORT")
    config = LLMConfig.from_env()
    assert config.reasoning_effort is None
    assert config.secondary_reasoning_effort == "none"  # default


def test_reasoning_effort_env_invalid(monkeypatch):
    monkeypatch.setenv("NOVACODE_REASONING_EFFORT", "ultra")
    with pytest.raises(ValueError, match="invalid reasoning effort"):
        LLMConfig.from_env()


# ------------------------------------------------------ message round-trip


def test_message_raw_content_round_trip():
    raw = [
        {"type": "thinking", "thinking": "t", "signature": "s"},
        {"type": "tool_use", "id": "c1", "name": "read", "input": {"path": "x"}},
    ]
    message = Message(role="assistant", content=None, raw_content=raw)
    restored = Message.from_dict(message.to_dict())
    assert restored.raw_content == raw
    assert restored.content is None


def test_context_manager_counts_raw_content(monkeypatch):
    from coding_agent.context.manager import ContextManager

    manager = ContextManager("sys", max_context_tokens=1_000_000)
    manager.add_assistant(
        "hello world",
        raw_content=[{"type": "thinking", "thinking": "x" * 400, "signature": "s"}],
    )
    plain = ContextManager.estimate_tokens([Message(role="assistant", content="hello world")])
    with_raw = ContextManager.estimate_tokens(manager.messages)
    # Raw thinking text (400 chars ~ 101 tokens) is counted on top of content.
    assert with_raw > plain + 50


def test_context_manager_counts_openai_reasoning(monkeypatch):
    # OpenAI raw_content is a dict, not a block list; reasoning_content must
    # be counted rather than iterating the dict keys (the old bug).
    from coding_agent.context.manager import ContextManager

    manager = ContextManager("sys", max_context_tokens=1_000_000)
    manager.add_assistant(
        "hello world",
        raw_content={
            "role": "assistant",
            "content": "hello world",
            "reasoning_content": "deep " + "x" * 400,
        },
    )
    plain = ContextManager.estimate_tokens([Message(role="assistant", content="hello world")])
    with_raw = ContextManager.estimate_tokens(manager.messages)
    assert with_raw > plain + 50


# ------------------------------------------------------ token estimation


def test_token_counter_counts_openai_reasoning():
    from coding_agent.llm.base import ToolCall
    from coding_agent.structured_context.token_counter import TokenCounter

    content = "answer text here " * 20
    reasoning = "deep thinking " + "x" * 300
    with_reasoning = Message(
        role="assistant",
        content=content,
        raw_content={"role": "assistant", "content": content, "reasoning_content": reasoning},
    )
    no_reasoning = Message(role="assistant", content=content)
    delta = TokenCounter.estimate_message(with_reasoning) - TokenCounter.estimate_message(no_reasoning)
    # Reasoning text (~300 chars) is actually counted now, not the dict keys.
    assert delta > 30

    # raw_content.tool_calls duplicate Message.tool_calls and add nothing.
    tc = ToolCall(id="c1", name="read", arguments={"path": "a.py"})
    with_raw_tools = Message(
        role="assistant",
        content="",
        tool_calls=[tc],
        raw_content={
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a.py"}'}}
            ],
        },
    )
    only_message_tools = Message(role="assistant", content="", tool_calls=[tc])
    assert TokenCounter.estimate_message(with_raw_tools) == TokenCounter.estimate_message(only_message_tools)


def test_token_counter_anthropic_text_block_not_double_counted():
    from coding_agent.structured_context.token_counter import TokenCounter

    text = "the answer " * 30  # ~330 chars
    thinking = {"type": "thinking", "thinking": "x" * 400, "signature": "s"}
    plain = Message(role="assistant", content=text)
    with_text_block = Message(role="assistant", content=text, raw_content=[{"type": "text", "text": text}])
    # A text block duplicates Message.content → must not add tokens.
    assert TokenCounter.estimate_message(with_text_block) == TokenCounter.estimate_message(plain)

    with_thinking = Message(
        role="assistant",
        content=text,
        raw_content=[thinking, {"type": "text", "text": text}],
    )
    only_thinking = Message(role="assistant", content=text, raw_content=[thinking])
    # The thinking block adds tokens; the text block alongside it adds nothing.
    assert TokenCounter.estimate_message(with_thinking) > TokenCounter.estimate_message(plain) + 50
    assert TokenCounter.estimate_message(with_thinking) == TokenCounter.estimate_message(only_thinking)
