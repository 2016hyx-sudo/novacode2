# Structured Context Implementation Status

Implementation follows `coding_agent_context_management_v3_schema.md`.

## Completed

- M0 data models: TaskState / ToolState / Trajectory / InteractionGroup / WorkspaceExpected / CheckpointManifest / StructuredSession.
- M1 EventLog with seq + hash chain + fsync, ArtifactStore with content hash and cached in-memory index, WorkspaceFingerprint with git-aware expected/actual diff.
- M2 TokenCounter with usage calibration and calibration persistence in `session.json`; tool definitions are included in prompt token estimates; StructuredContext builds system + user(Task/Tool State) + native trajectory + user(Agent State); Anthropic cache_control on system, tools and trajectory tail; OpenAI usage exposes cached_tokens.
- M3 Tool results are always written to ArtifactStore; observations use 2K trigger and per-tool char caps with head/tail deterministic reduction + artifact_id pointer.
- M4 InteractionGroup management; LLM-assisted Trajectory Fold at 70% with deterministic fallback; post-fold checkpoint; state capacity control with stale-first eviction and artifact overflow; WAL tool_batch_intent / tool_batch_closed events; complete trajectory archive (`trajectory-archive.jsonl`).
- M5 Baseline, periodic (after each closed tool batch), post-fold and terminal checkpoints; immutable checkpoint manifests + CURRENT; load-time hash validation; crash-suffix event replay including interrupted tool batch synthesis; drift detection and RESUME / REPLAN / BLOCKED recovery wiring; stale propagation on impacted paths.
- M6 StructuredHarness + CLI flags `--structured-context` / `--agent-dir`; explicit `--session-dir` / `--trace-dir` override; `read_artifact` tool; runtime state directory protected from filesystem/shell tools; session lock files; legacy session migration (`--migrate-legacy-session`); structured subagent report parsing and subagent file-change propagation; unit tests.
- Evaluation Phase 0 instrumentation: one immutable payload snapshot per logical Agent request; unique request IDs for every provider attempt; `step` / `attempt` / `agent_role` / `event_seq_anchor` trace correlation; canonical payload hashes; paired `llm_request_prepared` / `llm_request_finished` events; OpenAI/Anthropic logical-input normalization; cache-aware Anthropic calibration; Fold request usage and parent-request linkage; explicit Fold residual/reduction ratios.
- Evaluation Phase 1/2 framework: deterministic 238-request three-way replay; CRR/Net CRR and p50/p95 reports; PR baseline gate; core-30 scenario validation; eight generated fixture families; isolated full-run workspaces; strict oracle allowlist; injected live-provider boundary and no-network scripted end-to-end smoke.

## Remaining

- Per-tool semantic compressors beyond head/tail reduction.
- Full structured subagent trajectory persistence (subagents currently use the lightweight legacy context; reports and file changes are propagated).
- Model hard-limit enforcement that stops a request before the provider rejects an oversized prompt.
- Artifact index compaction / garbage collection for orphaned artifacts.
- SchemaRegistry-based JSON Schema validation for every persisted file.
- Session metrics retention policy and event-log compaction.
- Three reviewed live-provider baselines and approved nightly thresholds (data collection; provider injection is implemented, but no paid API run was made here).
