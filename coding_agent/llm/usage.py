"""Provider-neutral request measurement helpers.

The provider adapters intentionally preserve their native usage dictionaries.
This module adds a normalized view for cross-provider context metrics without
discarding or rewriting those raw fields.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Sequence
from typing import Any

from .base import Message, ToolSchema

MEASUREMENT_SCHEMA_VERSION = "1.0"


def new_request_id() -> str:
    """Return an opaque ID for one actual provider attempt."""

    return f"req-{uuid.uuid4().hex}"


def normalize_usage(usage: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize provider usage while keeping cache semantics explicit.

    OpenAI reports cached tokens as a subset of prompt tokens. Anthropic
    reports ordinary input, cache reads and cache creation as separate fields,
    so its logical input is the sum of all three.
    """

    raw = usage or {}
    is_openai = "prompt_tokens" in raw or "completion_tokens" in raw
    is_anthropic = any(
        key in raw
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        )
    )

    if is_openai:
        logical = _non_negative_int(raw.get("prompt_tokens"))
        cache_hit = min(logical, _non_negative_int(raw.get("cached_tokens")))
        cache_creation = 0
        fresh = logical - cache_hit
        output = _non_negative_int(raw.get("completion_tokens"))
        provider_format = "openai"
    elif is_anthropic:
        ordinary = _non_negative_int(raw.get("input_tokens"))
        cache_hit = _non_negative_int(raw.get("cache_read_input_tokens"))
        cache_creation = _non_negative_int(raw.get("cache_creation_input_tokens"))
        logical = ordinary + cache_hit + cache_creation
        fresh = ordinary + cache_creation
        output = _non_negative_int(raw.get("output_tokens"))
        provider_format = "anthropic"
    else:
        logical = 0
        cache_hit = 0
        cache_creation = 0
        fresh = 0
        output = _non_negative_int(raw.get("output_tokens"))
        provider_format = "unknown"

    return {
        "provider_format": provider_format,
        "token_source": "provider_reported" if logical > 0 else "missing",
        "logical_input_tokens": logical,
        "cache_hit_tokens": cache_hit,
        "cache_creation_input_tokens": cache_creation,
        "fresh_processed_input_tokens": fresh,
        "output_tokens": output,
    }


def request_payload_hash(
    messages: Sequence[Message],
    tools: Sequence[ToolSchema] | None,
) -> str:
    """Hash the canonical provider-neutral request payload."""

    payload = {
        "messages": [message.to_dict() for message in messages],
        "tools": [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            }
            for tool in tools or ()
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _non_negative_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0
