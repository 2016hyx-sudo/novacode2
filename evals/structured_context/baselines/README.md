# Evaluation baselines

`offline-core-15-summary.json` and `offline-core-15-requests.jsonl` are the
reviewed deterministic synthetic replay baselines used by the PR workflow.
The workflow applies both aggregate and per-request gates. They contain
estimated tokens, not live provider usage. Update them only together with an
intentional evaluator, tokenizer, compressor, or dataset change and include
the generated Markdown diff in review.

The core-15 suite adds three reasoning-bearing recipes to the original twelve:
`offline-reasoning-anthropic-13` (Anthropic thinking blocks), 
`offline-reasoning-openai-14` (OpenAI `reasoning_content` with Unicode text),
and `offline-reasoning-fold-15` (thinking-heavy trajectory that crosses the
fold trigger, archiving folded reasoning). Each variant carries a `reasoning`
token layer so the PR gate can detect regressions in how thinking blocks are
preserved and counted.

Live provider/model baselines are deliberately not checked in yet. Nightly
runs must keep each provider/model snapshot separate and collect three
read-only runs before approving thresholds.

## `offline-llm-fold` mode

The `offline-llm-fold` subcommand replays the same core-15 recipes but runs
the real `FoldEngine` (optionally backed by a live provider) at every fold
point; LLM fold deltas mutate the replay state, so later prompts and token
estimates diverge from the deterministic simulation. Results are
non-deterministic and no baseline for this mode is committed:

- `--baseline` must point at a `summary.json` produced by a previous
  `offline-llm-fold` run (`compare_summaries` tolerance gate: crr 3% /
  p50 5% / p95 8%, `net_crr >= 0`). Never point it at
  `offline-core-15-summary.json` — state evolution guarantees p50/p95
  differences.
- The per-request gate (abs 128 / rel 2%) does not apply to this mode; the
  command exits non-zero when the summary gate fails or any recipe failed
  its assertions.
- The `usage` block mixes estimated main-request tokens with real
  provider-reported fold tokens: `fold.input_tokens` (in
  `requests.jsonl`) is the real fold cost for LLM folds (main rows are
  always estimates). Fallback folds carry the estimate instead, with
  `token_source: "estimated"` and the failure recorded in `last_error`.
- The FoldEngine enforces its production `max_input_chars` input budget at
  fold time. The synthetic core-15 eligible-group sets were sized for the
  rule simulation and often exceed it, so a large share of fold points
  degrade to the deterministic extractor (`calls: 0`, `fallback_used: true`,
  `last_error: "FoldError: fold input exceeds configured size budget"`).
  That is the mode's honest measurement, not a bug: shrink
  `tool_results[].chars_each` in custom fixtures to exercise the LLM path
  on more recipes.
- Fold request rows use the same `offline-{case}-{fold_id}` ids as the
  deterministic mode, so cross-mode `compare` runs are meaningless.
