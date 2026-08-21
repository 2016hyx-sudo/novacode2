"""NovaCode CLI entry point."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from coding_agent import create_harness
from coding_agent.tui import TUI
from config import AgentConfig, Constraints, LLMConfig, load_env_file


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="novacode",
        description="NovaCode - a small, understandable coding agent harness",
    )
    parser.add_argument("task", nargs="?", help="Coding task to execute")
    parser.add_argument("--workspace", help="Workspace directory (default: current directory)")
    parser.add_argument("--provider", choices=["openai", "anthropic"], help="LLM provider")
    parser.add_argument("--model", help="Model name")
    parser.add_argument("--api-key", help="Provider API key")
    parser.add_argument("--base-url", help="Optional API base URL")
    parser.add_argument("--planner", action="store_true", default=None, help="Enable simple Planner Mode")
    parser.add_argument("--max-steps", type=int, help="Override max agent steps")
    parser.add_argument("--session", help="Continue existing session id")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive TUI")
    parser.add_argument("--list-sessions", action="store_true", help="List saved sessions and exit")
    parser.add_argument("--session-dir", default=None, help="Session storage directory")
    parser.add_argument("--trace-dir", default=None, help="Trace storage directory")
    parser.add_argument("--structured-context", action="store_true", default=None, help="Use structured context / checkpoint-resume storage")
    parser.add_argument("--migrate-legacy-session", metavar="SESSION_ID", default=None, help="Migrate one legacy .sessions/<id>.json session to structured storage")
    parser.add_argument("--legacy-session-dir", default=None, help="Directory containing the legacy .sessions/<id>.json file")
    parser.add_argument("--agent-dir", default=None, help="Structured state root (sessions and traces live below it)")
    parser.add_argument(
        "--storage-location",
        choices=["project", "user"],
        default=None,
        help="Storage root for sessions, traces, and memories: 'project' (inside workspace/.agent) or 'user' (inside ~/.novacode)",
    )
    parser.add_argument("--memory-dir", default=None, help="Project memory storage directory")
    parser.add_argument("--memory-global-dir", default=None, help="Global memory storage directory")
    parser.add_argument("--env-file", default=None, help="Path to .env file (default: ./.env)")
    return parser.parse_args(argv)


def resolve_env_file(args: argparse.Namespace) -> Path | None:
    """Find the env file in this order:
    1. --env-file / NOVACODE_ENV_FILE
    2. .env in the current directory or any parent directory
    3. .env next to main.py (useful when running from another cwd)
    """
    explicit = args.env_file or os.getenv("NOVACODE_ENV_FILE")
    if explicit:
        return load_env_file(explicit)

    env_path = load_env_file()
    if env_path is not None:
        return env_path

    script_dir = Path(__file__).resolve().parent
    if script_dir != Path.cwd():
        return load_env_file(script_dir / ".env")
    return None


def build_config(args: argparse.Namespace) -> AgentConfig:
    llm = LLMConfig.from_env()
    if args.provider:
        llm.provider = args.provider
    if args.model:
        llm.model = args.model
    if args.api_key:
        llm.api_key = args.api_key
    if args.base_url:
        llm.base_url = args.base_url

    constraints = Constraints()
    if args.max_steps:
        constraints.max_steps = args.max_steps

    workspace = Path(args.workspace or os.getenv("NOVACODE_WORKSPACE", ".")).expanduser().resolve()
    planner_enabled = (
        args.planner if args.planner is not None else _truthy(os.getenv("NOVACODE_PLANNER"))
    )
    structured_context = (
        args.structured_context
        if args.structured_context is not None
        else _truthy(os.getenv("NOVACODE_STRUCTURED_CONTEXT"))
    )

    storage_location = (
        args.storage_location
        or os.getenv("NOVACODE_STORAGE_LOCATION", "project")
    ).strip().lower()
    if storage_location not in ("project", "user"):
        storage_location = "project"

    import hashlib
    import re

    workspace_clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", workspace.name).strip("_") or "workspace"
    workspace_slug = f"{workspace_clean}_{hashlib.sha256(str(workspace).encode('utf-8')).hexdigest()[:8]}"

    session_explicit = bool(args.session_dir or os.getenv("NOVACODE_SESSION_DIR"))
    trace_explicit = bool(args.trace_dir or os.getenv("NOVACODE_TRACE_DIR"))

    if storage_location == "user":
        user_root = Path.home() / ".novacode" / "projects" / workspace_slug
        agent_dir = Path(args.agent_dir or os.getenv("NOVACODE_AGENT_DIR") or user_root)
        session_dir = Path(args.session_dir or os.getenv("NOVACODE_SESSION_DIR") or (agent_dir / "sessions"))
        trace_dir = Path(args.trace_dir or os.getenv("NOVACODE_TRACE_DIR") or (agent_dir / "traces"))
        memory_project_dir = Path(args.memory_dir or os.getenv("NOVACODE_MEMORY_DIR") or (agent_dir / "memories"))
        memory_global_dir = Path(args.memory_global_dir or os.getenv("NOVACODE_MEMORY_GLOBAL_DIR") or (Path.home() / ".novacode" / "memories" / "global"))
    else:
        # project mode
        agent_dir = Path(args.agent_dir or os.getenv("NOVACODE_AGENT_DIR") or (workspace / ".agent"))
        if structured_context:
            session_dir = Path(args.session_dir or os.getenv("NOVACODE_SESSION_DIR") or (agent_dir / "sessions"))
            trace_dir = Path(args.trace_dir or os.getenv("NOVACODE_TRACE_DIR") or (agent_dir / "traces"))
        else:
            session_dir = Path(args.session_dir or os.getenv("NOVACODE_SESSION_DIR") or (workspace / ".sessions"))
            trace_dir = Path(args.trace_dir or os.getenv("NOVACODE_TRACE_DIR") or (workspace / ".traces"))
        memory_project_dir = Path(args.memory_dir or os.getenv("NOVACODE_MEMORY_DIR") or (agent_dir / "memories"))
        memory_global_dir = Path(args.memory_global_dir or os.getenv("NOVACODE_MEMORY_GLOBAL_DIR") or (Path.home() / ".novacode" / "memories" / "global"))

    return AgentConfig(
        llm=llm,
        workspace=workspace,
        planner_enabled=planner_enabled,
        constraints=constraints,
        session_dir=session_dir,
        trace_dir=trace_dir,
        session_dir_explicit=session_explicit,
        trace_dir_explicit=trace_explicit,
        structured_context_enabled=structured_context,
        agent_dir=agent_dir,
        storage_location=storage_location,  # type: ignore[arg-type]
        memory_project_dir=memory_project_dir,
        memory_global_dir=memory_global_dir,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    env_path = resolve_env_file(args)
    if env_path is not None:
        print(f"[env] loaded {env_path}")
    else:
        print("[env] no .env file found; using OS environment variables only")
    config = build_config(args)
    print(f"[config] provider={config.llm.provider} model={config.llm.model}")
    tui = TUI()

    if args.list_sessions:
        if config.structured_context_enabled:
            from coding_agent.structured_context import StructuredSessionStore

            sessions = StructuredSessionStore(config.session_dir).list_sessions()
            if not sessions:
                tui.console.print("No saved sessions.")
            for path in sessions:
                tui.console.print(f"{path.name}  ·  {path}")
        else:
            from coding_agent.context.session import SessionStore

            sessions = SessionStore(config.session_dir).list_sessions()
            if not sessions:
                tui.console.print("No saved sessions.")
            for path in sessions:
                tui.console.print(f"{path.stem}  ·  {path}")
        return 0

    if args.migrate_legacy_session and not config.structured_context_enabled:
        tui.console.print("[red]Legacy session migration requires --structured-context[/red]")
        return 2

    try:
        harness = create_harness(config, listeners=[tui.handle_event])
    except Exception as exc:
        tui.console.print(f"[red]Failed to initialize harness: {exc}[/red]")
        return 1

    if args.migrate_legacy_session:
        if not hasattr(harness, "migrate_legacy_session"):
            tui.console.print("[red]Legacy session migration is unavailable in this harness[/red]")
            return 2
        legacy_dir = args.legacy_session_dir or os.getenv("NOVACODE_LEGACY_SESSION_DIR", ".sessions")
        try:
            session = harness.migrate_legacy_session(args.migrate_legacy_session, legacy_dir=legacy_dir)
        except (FileNotFoundError, FileExistsError, ValueError) as exc:
            tui.console.print(f"[red]Migration failed: {exc}[/red]")
            return 1
        tui.console.print(f"Migrated legacy session to structured session {session.id}")
        return 0

    if args.interactive:
        tui.run_interactive(harness, session_id=args.session)
        return 0

    task = (args.task or "").strip()
    if not task:
        print("No task provided. Use: novacode --interactive  or  novacode 'your task'")
        return 2

    result = tui.run_once(harness, task, session_id=args.session)
    if result is None or result.status != "completed":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
