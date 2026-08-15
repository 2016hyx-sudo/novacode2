"""Configuration dataclasses for NovaCode.

These are plain data holders. Behaviour lives in the runtime modules.
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


def load_env_file(path: str | Path | None = None, *, override: bool = False) -> Path | None:
    """Load KEY=VALUE entries from a simple .env file.

    - Blank lines and full-line ``#`` comments are ignored.
    - ``export KEY=...`` is accepted.
    - Values may be single- or double-quoted.
    - Existing OS environment variables win unless ``override=True``.
    """
    if path is not None:
        target = Path(path)
        if not target.is_file():
            return None
    else:
        target = None
        for directory in (Path.cwd(), *Path.cwd().parents):
            candidate = directory / ".env"
            if candidate.is_file():
                target = candidate
                break
        if target is None:
            return None

    for raw_line in target.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            try:
                value = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                value = value[1:-1]
        if override:
            os.environ[key] = value
        else:
            os.environ.setdefault(key, value)
    return target

ProviderName = Literal["openai", "anthropic"]

DEFAULT_SYSTEM_PROMPT = """You are NovaCode, a lightweight coding agent.
You work inside a single workspace directory. Use the provided tools to inspect,
modify and verify code. Prefer small, verifiable changes.

Important rules:
- Only access files inside the workspace.
- After modifying code, run an appropriate verification command before declaring success.
- If a tool fails, read the error message, diagnose the cause, and change your approach.
- Return a concise final summary when the task is truly complete."""


@dataclass
class LLMConfig:
    """Provider settings. Protocol differences are isolated behind the provider layer."""

    provider: ProviderName = "openai"
    model: str = "gpt-4o-mini"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    timeout: float = 120.0

    @classmethod
    def from_env(cls) -> LLMConfig:
        provider = os.getenv("NOVACODE_PROVIDER", "openai").strip().lower()
        if provider not in ("openai", "anthropic"):
            provider = "openai"
        default_model = "gpt-4o-mini" if provider == "openai" else "claude-3-5-sonnet-latest"
        key_env = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        return cls(
            provider=provider,  # type: ignore[arg-type]
            model=os.getenv("NOVACODE_MODEL", default_model),
            api_key=os.getenv(key_env, "") or os.getenv("NOVACODE_API_KEY", ""),
            base_url=os.getenv("NOVACODE_BASE_URL") or None,
            max_tokens=int(os.getenv("NOVACODE_MAX_TOKENS", "4096")),
            timeout=float(os.getenv("NOVACODE_LLM_TIMEOUT", "120")),
        )


@dataclass
class Constraints:
    """Runtime constraints for one Harness session (shared with subagents)."""

    max_steps: int = 30
    max_tool_calls: int = 80
    max_subagents: int = 3
    # Maximum nesting depth of subagents. Main agent is depth 0, its subagent is
    # depth 1. Default 1 means subagents may not spawn further subagents.
    max_subagent_depth: int = 1
    shell_timeout: float = 60.0
    tool_timeout: float = 120.0
    tool_max_retries: int = 1
    max_output_chars: int = 20_000
    max_correction_attempts: int = 2
    max_llm_retries: int = 1
    require_verification_after_edit: bool = True


@dataclass
class AgentConfig:
    """Top-level harness configuration."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    workspace: Path = field(default_factory=Path.cwd)
    planner_enabled: bool = False
    constraints: Constraints = field(default_factory=Constraints)
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    session_dir: Path = field(default_factory=lambda: Path(".sessions"))
    trace_dir: Path = field(default_factory=lambda: Path(".traces"))
    # ContextManager trim threshold, in estimated tokens.
    max_context_tokens: int = 100_000
