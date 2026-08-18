# Structured Context Evaluation Dataset

This directory contains immutable evaluation source data. Merely reading or
editing these files does not run NovaCode or call a model; evaluator code lives
in the parent package.

## Files

- `manifest.json`: dataset version, bucket counts, shared fixture/content
  profiles, and reproducibility rules.
- `scenarios.json`: 30 full-run coding tasks. Each task includes a deterministic
  fixture recipe, context-pressure targets, budgets, and correctness oracles.
- `offline_cases.json`: 12 deterministic replay recipes for constructing
  `raw_full`, `observation_full`, and `structured` request snapshots later.

## Intended use

Offline evaluation consumes `offline_cases.json` and materializes synthetic
request histories without an LLM. Full-run evaluation consumes
`scenarios.json`, creates an isolated workspace from each fixture recipe, and
runs a complete Agent conversation through an explicitly injected provider.
The built-in scripted provider is deterministic and makes no network calls.
The checked-in PR baseline is synthetic. Real archive/artifact inputs can be
passed to `reconstruct_prompt_variants()`, but a reviewed production golden
trajectory corpus is intentionally not claimed by this dataset.

## Dataset guarantees

1. Every generated input has a fixed seed and no network dependency.
2. Every full-run task has at least one machine-checkable oracle.
3. Every case declares its intended context pressure; pressure targets are not
   correctness expectations.
4. A runner must materialize baseline and structured variants from the same
   fixture recipe and verify the resulting fixture hash before execution.
5. Provider/model results must never be written back into these source files;
   they belong under `.eval-results/<run_id>/`.

## Schema summary

Full-run scenario:

```text
id, bucket, title, task
fixture {template, seed, parameters}
pressure {expected_steps, raw_tool_output_chars, expect_fold, expect_state_compact}
limits {max_steps, max_tool_calls}
oracle {checks[]}
tags[]
```

Offline case:

```text
id, bucket, description
generator {seed, interaction_groups, tool_results[], state_items, language_mix}
expected {tool_compression, fold, state_compact, protected_recent_groups}
assertions[]
```

The `expected` and `pressure` fields describe which mechanism the case should
exercise. They must not contain a precomputed CRR or p50/p95 value; those values
are outputs of the future evaluator.
