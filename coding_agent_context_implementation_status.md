# Structured Context Implementation Status

Implementation follows `coding_agent_context_management_v3_schema.md`.

## Completed

- M0 data models: TaskState / ToolState / Trajectory / InteractionGroup / WorkspaceExpected / CheckpointManifest / StructuredSession.
- M1 EventLog with seq + hash chain + fsync, ArtifactStore with content hash, WorkspaceFingerprint with git-aware expected/actual diff.
- M2 TokenCounter with usage calibration; StructuredContext builds system + user(Task/Tool State) + native trajectory + user(Agent State); Anthropic cache_control on system, tools and trajectory tail; OpenAI usage exposes cached_tokens.
- M3 Tool results are always written to ArtifactStore; observations use 2K trigger and per-tool char caps with head/tail deterministic reduction + artifact_id pointer.
- M4 InteractionGroup management; deterministic Trajectory Fold at 70% with state delta merge; WAL tool_batch_intent / tool_batch_closed events.
- M5 Baseline, periodic (after each closed tool batch) and terminal checkpoints; immutable checkpoint manifests + CURRENT; load-time validation; drift detection; RESUME / recovery-pending / BLOCKED behavior; RESUME loads snapshot + validates event log anchor.
- M6 partial: StructuredHarness + CLI flags `--structured-context` / `--agent-dir`; `read_artifact` tool; runtime state directory is protected from filesystem/shell tools; unit tests.

## Remaining

- LLM-based Trajectory Fold extraction (currently deterministic fallback only).
- Full post-crash event replay / WAL recovery for interrupted tool batches.
- Old `.sessions/<id>.json` migration converter.
- Session lock file (`locks/<session_id>.lock`).
- Per-tool semantic compressors beyond head/tail reduction.
- Detailed metrics events for layer tokens and fold decisions.
- Structured SubagentReport schema enforcement.
