# AMA-Bench integration

This directory (`ama_bench/`) lets NovaCode participate in
[AMA-Bench](https://github.com/AMA-Bench/AMA-Bench/), the long-horizon agent
memory benchmark (ICML 2026).  AMA-Bench feeds each method a rendered agent
trajectory plus questions, and scores how well the method builds memory from
the trajectory, retrieves evidence for a question, and answers open-ended /
MCQ questions (LLM-as-judge).

NovaCode's contribution is `NovaCodeMemoryMethod`: it exposes NovaCode's
structured-context stack — Interaction Groups, Trajectory Fold, Task/Tool
State, capacity control — through AMA-Bench's two-stage interface:

| AMA-Bench stage | NovaCode implementation |
|---|---|
| `memory_construction(traj_text, task)` | parse steps → Interaction Groups → deterministic / LLM-assisted Trajectory Fold → Task State + Tool State + Recent Trajectory |
| `memory_retrieve(memory, question)` | lexical scoring of groups / findings / decisions / tool experience → rendered `<evidence>` block (or a full batch answer prompt) |

The adapter runs with **zero additional runtime dependencies**: it only needs
the `coding_agent` package that ships with NovaCode.  The LLM-assisted fold
path optionally uses AMA-Bench's own `ModelClient` (the same model as the
run) and degrades to the deterministic fold automatically.

## Layout

| File | Purpose |
|---|---|
| `steps.py` | Parse AMA-Bench trajectory text (`Step N: / Action: / Observation:`) into uniform steps |
| `fold.py` | `NovaCodeMemoryBuilder`: steps → Interaction Groups → Trajectory Fold (deterministic or LLM) → state compaction |
| `memory.py` | `NovaCodeMemory`: the memory object (Task/Tool State + Recent Trajectory) |
| `retrieve.py` | `score_candidates` / `render_evidence`: question-gated retrieval and evidence rendering |
| `prompt.py` | `build_batch_prompt`: single-call `Answer[i]:` batch prompt (longcontext-style) |
| `extract.py` | `extract_final_answer` / `parse_answer_blocks`: AMA-Bench-compatible answer parsing |
| `method.py` | `NovaCodeMemoryMethod` (two-stage interface), `AMAClientFoldProvider` (ModelClient → LLMProvider adapter) |
| `run.py` | **Standalone runner** — dataset JSONL → method → NovaCode LLM provider → submission-format results (no AMA-Bench checkout) |
| `judge.py` | **Self-contained LLM-as-judge** — official judge prompt/semantics on NovaCode's provider, with summary stats |
| `register_ama.py` | One-time registration of the method inside the AMA-Bench repo (for Option B) |

## Running the evaluation

### Option A — standalone runner (recommended, no AMA-Bench checkout)

`python -m ama_bench.run` reads the dataset JSONL directly, drives the method,
answers questions with **NovaCode's own LLM provider** (same `.env` /
`NOVACODE_*` configuration as the main CLI), and writes results in the
AMA-Bench submission format.  For MCQ subsets with ground-truth answers it also
prints local exact-match accuracy.

```bash
# 1. Install NovaCode and download the dataset
python -m venv .venv && source .venv/bin/activate
pip install -e .
huggingface-cli download AMA-bench/AMA-bench --repo-type dataset --local-dir ./dataset

# 2. Configure the provider (.env or shell env)
#    NOVACODE_PROVIDER=openai  NOVACODE_MODEL=...  OPENAI_API_KEY=...

# 3. Run a few episodes
python -m ama_bench.run \
  --dataset dataset/test/open_end_qa_set.jsonl \
  --episode-ids 0,1,2 \
  --output results/novacode_openend.jsonl
```

CLI options: `--subset` (auto-detected from the file name), `--samples N`,
`--batch` (all questions of an episode in one LLM call instead of the default
one call per question), `--max-tokens`, `--method-config`, and provider
overrides `--provider / --model / --api-key / --base-url`.  Every run also
writes one audit JSON per episode (see below).

### Audit trail (debugging a run)

Each episode gets an audit file `<audit-dir>/<episode_id>.json` (default:
`<output parent>/audit`) tracing the full pipeline — input trajectory, the
pre-fold group inventory, every fold epoch (model used, deltas, fallback,
errors), the post-fold memory summary, per-question retrieval (top-k scores,
evidence hash), the exact prompt sent, the answer, and per-call usage
(cache-hit tokens included).  Results records carry `audit_path` plus a compact
`memory` summary (pre/post-fold tokens, residual ratio, fold counts) so the
results JSONL alone shows how much each trajectory was compressed.

```bash
python -m ama_bench.run \
  --dataset dataset/test/open_end_qa_set.jsonl \
  --episode-ids 0,1,2 \
  --output results/novacode_openend.jsonl \
  --audit-dir results/audit \
  --audit-full          # full pre-fold groups, evidence and prompts
```

`--audit-full` records full content (larger files; also implies
`--keep-work-dir` so `groups.jsonl` survives).  Without it the audit stores
hashes, char counts and scores — enough to spot *where* an answer went wrong
(memory loss vs retrieval miss vs generation), and the exact content is one
`--audit-full` re-run away.

### Judge the answers (self-contained LLM-as-judge)

`python -m ama_bench.judge` is a faithful reimplementation of AMA-Bench's
official LLM-as-judge (same prompt, same last-`yes`/`no` parsing, same F1
fallback, same summary shape), running on NovaCode's own provider — no
AMA-Bench checkout needed:

```bash
python -m ama_bench.judge \
  --answers-file results/novacode_openend.jsonl \
  --test-file dataset/test/open_end_qa_set.jsonl \
  --output-file results/evaluation.json
```

Judge provider defaults to the same `.env` config; override with
`--provider / --model / --api-key / --base-url`, or pass an AMA-style judge
YAML with `--judge-config configs/llm_judge.yaml` (needs `pyyaml`).
`--max-workers` controls concurrent judge calls.

### Downloading the dataset

The official dataset (`AMA-bench/AMA-bench`, single ~50 MB file
`test/open_end_qa_set.jsonl`, **open-end subset only** — there is no
`mcq_set.jsonl` in the repo) may be slow or blocked from some networks.  The
HuggingFace mirror is a reliable alternative:

```bash
# Official
huggingface-cli download AMA-bench/AMA-bench --repo-type dataset --local-dir ./dataset

# Mirror (when huggingface.co is unreachable)
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download AMA-bench/AMA-bench \
  --repo-type dataset --local-dir ./dataset

# ...or direct links for one file:
curl -L -o dataset/test/open_end_qa_set.jsonl \
  https://hf-mirror.com/datasets/AMA-bench/AMA-bench/resolve/main/test/open_end_qa_set.jsonl
```

The official LLM-as-judge scoring still uses AMA-Bench's `evaluate.py`; feed it
the same results file.

### Option B — official benchmark harness (AMA-Bench checkout)

`src/run.py` (and `scripts/run.sh`, `scripts/run_api.sh`) belong to the
**AMA-Bench repository**, not NovaCode — NovaCode only ships the method
package.  Clone the benchmark and run its harness from there:

```bash
# 1. Install NovaCode (editable, includes the ama_bench package)
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Clone AMA-Bench and install its requirements
git clone https://github.com/AMA-Bench/AMA-Bench.git
cd AMA-Bench
pip install -r requirements.txt

# 3. Register the method, pointing at the NovaCode checkout
python -m ama_bench.register_ama --novacode-root /path/to/novacode
# -> registered method: novacode

# 4. Run an episode with API-based LLM (fold + judge) — note: this is AMA-Bench's
#    src/run.py, run from inside the AMA-Bench checkout
python src/run.py \
  --llm-server api \
  --llm-config configs/gpt-5.2.yaml \
  --judge-config configs/llm_judge.yaml \
  --subset openend \
  --method novacode \
  --episode-ids 0,1,2
```

`register_ama` registers for the current interpreter process.  For persistent
registration, add the same two lines to a `sitecustomize.py` at the AMA-Bench
root:

```python
import sys
sys.path.insert(0, "/path/to/novacode")
from ama_bench.method import NovaCodeMemoryMethod
from src.method.base_method import BaseMethod
from src.method_register import register_method

class NovacodeAmaMethod(BaseMethod, NovaCodeMemoryMethod):
    pass

register_method("novacode", NovacodeAmaMethod)
```

### Method configuration

`--method-config` accepts a JSON/YAML path with any of:

| Key | Default | Meaning |
|---|---|---|
| `max_context_tokens` | `96000` | Fold trigger budget; larger = less aggressive folding |
| `fold_max_attempts` | `2` | LLM fold retries before deterministic fallback |
| `keep_work_dir` | `false` | Keep the temporary build directory (debugging) |

### Offline / deterministic runs

Without a provider (`client` not injected), fold uses NovaCode's deterministic
extractor (`deterministic_fold_delta`); only the answer stage needs an LLM.
Use the method directly from a script:

```python
from ama_bench.method import NovaCodeMemoryMethod

method = NovaCodeMemoryMethod()
memory = method.memory_construction(traj_text, task)
prompt = method.build_prompt(memory, [q1, q2], mcq_mode=True)
# send `prompt` to any LLM, then parse with:
from ama_bench.extract import parse_answer_blocks
answers = parse_answer_blocks(llm_response, question_count=2, mcq_mode=True)
```

## Local tests

```bash
python -m pytest tests/test_ama_bench.py -q
```

## Notes

* The dataset contains only a test split; there is no training set.
* The deterministic fold deliberately keeps only the most recent
  `protected_recent_groups` trajectory groups verbatim; everything older is
  folded into Task/Tool State, which is exactly the behavior the benchmark's
  memory formulation exercises.
* The benchmark's `run.sh`/vLLM path targets Linux/CUDA; the `api` server
  path works on any platform that can reach the LLM API.
