"""LLM provider factory."""
from __future__ import annotations

import os

from config import LLMConfig

from .anthropic import AnthropicProvider
from .base import LLMError, LLMProvider, LLMResponse, Message, ToolCall, ToolSchema
from .openai import OpenAIProvider


def create_provider(config: LLMConfig) -> LLMProvider:
    """The only place where concrete provider names are resolved."""
    if config.provider == "openai":
        env_key = "OPENAI_API_KEY"
    elif config.provider == "anthropic":
        env_key = "ANTHROPIC_API_KEY"
    else:
        raise ValueError(f"Unsupported provider: {config.provider!r}")

    if not config.api_key:
        generic_key = os.getenv("NOVACODE_API_KEY", "")
        if generic_key:
            config.api_key = generic_key
        elif not os.getenv(env_key):
            raise ValueError(
                f"Missing API key for provider {config.provider!r}. "
                f"Pass --api-key or set {env_key} / NOVACODE_API_KEY in .env or the environment."
            )

    if config.provider == "openai":
        return OpenAIProvider(config)
    return AnthropicProvider(config)


__all__ = [
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "Message",
    "ToolCall",
    "ToolSchema",
    "create_provider",
]
