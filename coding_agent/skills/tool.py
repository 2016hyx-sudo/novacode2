"""The static meta-tool used to load or execute any discovered skill."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, ClassVar

from ..tools.base import AgentRunOutcome, ToolErrorKind, ToolResult
from .bank import SkillBank, SkillBankError
from .models import SkillEntry

ArtifactStoreProvider = Callable[[], Any | None]
ForkRunner = Callable[[SkillEntry, str], AgentRunOutcome]


class InvokeSkillTool:
    name = "invoke_skill"
    description = (
        "Load and apply one reusable NovaCode skill by name. Skill names are discovered from "
        "the project and user skill banks; the tool schema never changes when skills change."
    )
    parameters: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Exact snake_case skill name."},
            "args": {
                "type": "object",
                "description": "Optional JSON arguments interpolated into ${ARGUMENTS}.",
                "additionalProperties": True,
            },
        },
        "required": ["name"],
    }

    def __init__(
        self,
        bank: SkillBank,
        *,
        artifact_store: ArtifactStoreProvider | None = None,
        fork_runner: ForkRunner | None = None,
        inline_token_limit: int = 2_000,
    ) -> None:
        self.bank = bank
        self._artifact_store = artifact_store
        self._fork_runner = fork_runner
        self.inline_token_limit = max(100, int(inline_token_limit))

    def execute(self, name: str, args: dict[str, Any] | None = None) -> ToolResult:
        if not isinstance(name, str) or not name.strip():
            return ToolResult.fail(
                "skill name must not be empty",
                metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value},
            )
        if args is not None and not isinstance(args, dict):
            return ToolResult.fail(
                "skill args must be a JSON object",
                metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value},
            )
        try:
            entry = self.bank.get(name.strip())
        except SkillBankError as exc:
            return ToolResult.fail(
                str(exc), metadata={"kind": ToolErrorKind.INVALID_ARGUMENTS.value}
            )
        if entry is None:
            try:
                available = ", ".join(item.name for item in self.bank.list()) or "(none)"
            except SkillBankError:
                available = "(skill bank is invalid)"
            return ToolResult.fail(
                f"Unknown skill: {name}. Available skills: {available}",
                metadata={"kind": ToolErrorKind.NOT_FOUND.value},
            )

        rendered = entry.render(args)
        if entry.manifest.context == "fork":
            if self._fork_runner is None:
                return ToolResult.fail(
                    f"Skill {entry.name!r} requires fork execution, but no runner is configured",
                    metadata={"kind": ToolErrorKind.UNEXPECTED.value},
                )
            outcome = self._fork_runner(entry, rendered)
            report = getattr(outcome, "structured_report", None)
            output = json.dumps(report, ensure_ascii=False, indent=2) if isinstance(report, dict) else (
                getattr(outcome, "text", "") or "(skill subagent returned no report)"
            )
            status = str(getattr(outcome, "status", "failed"))
            return ToolResult(
                success=status == "completed",
                output=output,
                error=None if status == "completed" else f"Skill subagent ended with status {status!r}",
                metadata={"skill": entry.name, "mode": "fork", "version": entry.manifest.version},
            )

        estimated_tokens = max(1, (len(rendered) + 3) // 4)
        metadata = {
            "skill": entry.name,
            "mode": "inline",
            "version": entry.manifest.version,
            "estimated_tokens": estimated_tokens,
        }
        if estimated_tokens <= self.inline_token_limit:
            return ToolResult.ok(rendered, **metadata)
        store = self._artifact_store() if self._artifact_store is not None else None
        if store is None:
            return ToolResult.fail(
                f"Skill {entry.name!r} is too large for inline use and no ArtifactStore is active",
                metadata={**metadata, "kind": ToolErrorKind.UNEXPECTED.value},
            )
        artifact = store.save("invoke_skill", rendered, arguments={"name": entry.name, "args": args or {}})
        preview = rendered[:800].rstrip()
        output = (
            f"Skill {entry.name!r} was loaded, but its full content is stored as artifact "
            f"{artifact['artifact_id']} ({artifact['size']} bytes).\n\nPreview:\n{preview}\n\n"
            f"Use read_artifact with artifact_id={artifact['artifact_id']} for the full guidance."
        )
        return ToolResult.ok(output, **metadata, artifact_id=artifact["artifact_id"], compressed=True)
