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
