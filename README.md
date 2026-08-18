# NovaCode

A small, understandable, hackable coding agent harness in Python 3.11+.

NovaCode drives an LLM to complete coding tasks inside one workspace. It has a
thin AgentLoop, a unified LLM provider interface (OpenAI + Anthropic official
SDKs), eight standard tools, optional Planner Mode, bounded subagents, session
persistence and JSONL execution traces. It intentionally does **not** include a
workflow engine, DAGs, multi-agent coordination or reflection agents.

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .
```

Dependencies: `openai`, `anthropic`, `rich`.

## Quick start

```bash
# One-shot task
export OPENAI_API_KEY=...
python main.py "add a health check endpoint" --workspace ./myproject --planner

# Interactive TUI
python main.py --interactive

# Continue a session
python main.py "now also add /ready" --session <session_id>

# List sessions
python main.py --list-sessions
```

## .env configuration

Edit the included `.env` file to choose a provider and set credentials:

```bash
# provider: openai | anthropic
NOVACODE_PROVIDER=openai
NOVACODE_MODEL=gpt-4o-mini

OPENAI_API_KEY=sk-...
# Optional OpenAI-compatible base URL
NOVACODE_BASE_URL=

# For anthropic:
# NOVACODE_PROVIDER=anthropic
# ANTHROPIC_API_KEY=sk-ant-...
```

`python main.py ...` and the installed `novacode` command automatically look
for `.env` in this order:

1. `--env-file PATH` or `NOVACODE_ENV_FILE`
2. the current directory and its parent directories
3. the directory containing `main.py`

```bash
python main.py --env-file /path/to/other.env ...
```

Priority: **real shell environment variables > .env > built-in defaults**. At
startup the effective provider/model are printed as:

```text
[env] loaded /path/to/.env
[config] provider=anthropic model=claude-3-5-sonnet-latest
```

If `[config] provider` is not what your `.env` says, an already-exported shell
variable (for example `NOVACODE_PROVIDER=openai`) is overriding it.

All supported variables:

| Variable | Meaning |
|---|---|
| `NOVACODE_PROVIDER` | `openai` (default) or `anthropic` |
| `NOVACODE_MODEL` | model name |
| `NOVACODE_MAX_TOKENS` | max output tokens per LLM response; thinking/reasoning tokens share this budget with the final text |
| `NOVACODE_REASONING_EFFORT` | main-agent thinking effort: `none` \| `low` \| `high` \| `max` (empty = provider default, thinking on) |
| `NOVACODE_SECONDARY_REASONING_EFFORT` | effort for subagents and context folding (default `none`) |
| `OPENAI_API_KEY` | OpenAI key |
| `ANTHROPIC_API_KEY` | Anthropic key |
| `NOVACODE_API_KEY` | generic key for the selected provider |
| `NOVACODE_BASE_URL` | optional custom base URL |
| `NOVACODE_WORKSPACE` | default workspace |
| `NOVACODE_PLANNER` | `1/true/yes/on` enables Planner Mode |

## Offline demo

`demo.py` uses a deterministic scripted provider and needs no API key. It runs
the full runtime loop with Planner Mode, filesystem tools, `run_shell`,
validation, session persistence and traces:

```bash
python demo.py
```

## Project layout

```text
main.py                       CLI entry
config.py                     LLMConfig / AgentConfig / Constraints
coding_agent/
  agent.py                    AgentLoop
  llm/                        unified protocol + OpenAI/Anthropic adapters
  tools/                      Tool protocol, registry, executor, 8 tools, path guard
  context/                    ContextManager + JSON SessionStore
  runtime/                    planner, validator, constraints/budget, trace
  tui/                        Rich terminal UI
```

Sessions are written to `.sessions/<session_id>.json` and traces to
`.traces/<session_id>.jsonl`.

## Tools

`read_file`, `write_file`, `edit_file`, `list_files`, `search_files`,
`grep_search`, `run_shell`, `subagent`.

Filesystem tools resolve every path through one workspace guard that rejects
absolute paths, `..` traversal and symlink escapes. `run_shell` runs with the
workspace as cwd, enforces a timeout and output cap, and has a small deny list.
It is a best-effort guard, not a security sandbox.

## Structured context / checkpoint-resume (v3, opt-in)

A structured context mode is available behind a flag:

```bash
python main.py "add a health check endpoint" --structured-context --agent-dir .agent
```

It stores each session as a directory under `.agent/sessions/<session_id>/`
with `task-state.json`, `tool-state.json`, `trajectory.json`, `events.jsonl`,
raw tool-result artifacts and immutable checkpoints. Resume the same way as a
legacy session:

```bash
python main.py "now also add /ready" --structured-context --session <session_id>
```

Structured sessions also support crash replay of un-checkpointed events,
LLM-assisted trajectory folding with deterministic fallback, state capacity
control, workspace-drift RESUME / REPLAN / BLOCKED recovery, session locks and
legacy-session migration:

```bash
python main.py --structured-context --migrate-legacy-session <session_id>
```

Design baseline: `coding_agent_context_management_v3.md`; detailed schema and
implementation breakdown: `coding_agent_context_management_v3_schema.md`;
automated measurement design for Context Reduction Ratio and p50/p95 input
tokens: `structured_context_evaluation_plan.md`. The versioned evaluation task
data lives under `evals/structured_context/data/` (30 full-run scenarios and 12
offline replay recipes). Run deterministic replay with no model/network call:

```bash
python -m evals.structured_context offline --output .eval-results/offline
```

The isolated full-run framework requires an explicitly injected provider. Its
built-in end-to-end smoke path is also offline:

```bash
python -m evals.structured_context run --scripted --scenario short-01 \
  --variant raw_full --variant structured --output .eval-results/scripted
```

The CLI never constructs a live provider; nightly callers inject one through
`ContextEvaluationRunner`, so evaluation cannot silently make paid API calls.
Generated fixtures are trusted local test code: workspace copying, command
allowlists and hashes are regression isolation, not a hostile-code OS sandbox.

## Runtime loop

```text
User Task
→ optional Plan
→ LLM
→ Tool Call
→ ToolExecutor (timeout / retry / structured error)
→ Tool Result message
→ LLM
→ final-answer validation
→ correction feedback if needed
→ Final Answer
```

Subagents reuse the same provider and standard tools with an independent
context and max_steps. By default `max_subagent_depth = 1`, so a subagent may
not spawn another subagent.
