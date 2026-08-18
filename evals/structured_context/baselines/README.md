# Evaluation baselines

`offline-core-12-summary.json` and `offline-core-12-requests.jsonl` are the
reviewed deterministic synthetic replay baselines used by the PR workflow.
The workflow applies both aggregate and per-request gates. They contain
estimated tokens, not live provider usage. Update them only together with an
intentional evaluator, tokenizer, compressor, or dataset change and include
the generated Markdown diff in review.

Live provider/model baselines are deliberately not checked in yet. Nightly
runs must keep each provider/model snapshot separate and collect three
read-only runs before approving thresholds.
