"""CLI for NovaCode SWE-bench inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import load_env_file

from .runner import SingleRunConfig, run_single


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run NovaCode on one SWE-bench instance")
    parser.add_argument("run", nargs="?", default="run")
    parser.add_argument("--dataset", default="verified")
    parser.add_argument("--split", default="test")
    parser.add_argument("--instance", required=True)
    parser.add_argument("--output", type=Path, default=Path(".eval-results/swe-bench-single"))
    parser.add_argument("--work-root", type=Path, default=None)
    parser.add_argument("--keep-workspace", action="store_true")
    parser.add_argument("--image", default=None, help="override the official instance image")
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--docker-command", default="docker")
    parser.add_argument("--provider", choices=("openai", "anthropic"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-tool-calls", type=int, default=120)
    parser.add_argument("--shell-timeout", type=float, default=120.0)
    parser.add_argument("--planner", action="store_true")
    parser.add_argument("--structured-context", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.run != "run":
        parser.error("the only supported command is 'run'")
    if args.env_file:
        if load_env_file(args.env_file) is None:
            parser.error(f"env file not found: {args.env_file}")
    else:
        load_env_file()
    config = SingleRunConfig(
        dataset=args.dataset,
        split=args.split,
        instance_id=args.instance,
        output_dir=args.output,
        work_root=args.work_root,
        keep_workspace=args.keep_workspace,
        image=args.image,
        pull_image=not args.no_pull,
        docker_command=args.docker_command,
        provider=args.provider,
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        max_steps=args.max_steps,
        max_tool_calls=args.max_tool_calls,
        shell_timeout=args.shell_timeout,
        planner=args.planner,
        structured_context=args.structured_context,
    )
    try:
        result = run_single(config)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "instance_id": result.instance_id,
                "status": result.status,
                "patch_chars": len(result.model_patch),
                "elapsed_seconds": result.elapsed_seconds,
                "output": result.output_dir,
            },
            ensure_ascii=False,
        )
    )
    return 0 if result.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
