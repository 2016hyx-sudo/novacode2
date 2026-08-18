# NovaCode 结构化上下文自动化测评方案

> 目标：可重复地回答“结构化上下文实际减少了多少输入上下文”和“线上请求的 p50 / p95 input tokens 是多少”，并把结果接入回归门禁。
>
> 本方案基于当前仓库实现，重点覆盖 Tool Result Compress、Trajectory Fold、State Compact 和 provider cache。它不是只测某一次 Fold 的单元测试，而是测整个请求生命周期的净收益。

## 1. 当前系统分析

### 1.1 两条上下文链路

NovaCode 当前有两套上下文实现：

| 链路 | 实现 | 行为 | 适合作为 CRR 基线吗 |
|---|---|---|---|
| Legacy | `coding_agent/context/manager.py` | 累积原始消息，超过 `max_context_tokens` 后按消息组删除旧历史 | 不适合作为唯一分母。它的默认窗口为 100K，且直接丢历史，和结构化链路的 256K、可恢复折叠不是同一语义 |
| Structured | `coding_agent/structured_context/structured_context.py` | Stable Prefix + Task/Tool State + Recent Trajectory + Agent State；工具原文落 Artifact；70% 触发 Fold | 被测对象 |

结构化请求的物理布局为：

```text
system: Stable System Text
tools:  Native Tool Schemas
user:   Task State + Tool State
...:    Recent Trajectory（完整 Interaction Groups）
user:   Agent State
```

主要减量机制：

1. 所有 Raw Tool Result 先写 Artifact；超过估算 2K tokens 后，以 head/tail + `artifact_id` 进入 prompt。
2. Prompt 估算达到逻辑窗口 70% 后，旧 Interaction Groups 被折叠为 Task/Tool State Delta。
3. Fold merge 后执行 State Compact，限制 Task/Tool State 自身增长。
4. Stable Prefix、Tools 和轨迹尾部尽量保持稳定，以提高 provider cache 命中。

### 1.2 已有可观测数据

当前实现已有以下数据源：

- `context_estimate` trace：Stable Prefix、Tools、Task/Tool State、Recent Trajectory、Agent State 和 total 的本地估算。
- `llm_response.usage`：OpenAI 的 `prompt_tokens/cached_tokens`；Anthropic 的 `input_tokens/cache_read_input_tokens/cache_creation_input_tokens`。
- `fold_event`：Fold 前后分层 token、移除的轨迹 token、Delta token、模型/Fallback 信息。
- `trajectory-archive.jsonl`、`events.jsonl` 和 Artifact：可用于重建未折叠历史。
- `session.json.metrics`：Fold 数、累计被折叠轨迹 token 和 token calibration。

这些数据足够作为测评基础，但目前不能直接稳定计算端到端 CRR，原因见 1.3。

### 1.3 测量缺口与风险

1. **没有统一 request identity。** `context_estimate`、`llm_request` 和 `llm_response` 没有共同的 `request_id`；`llm_request` 也没有 `step`。重试发生时不能可靠 join。
2. **一次请求会重复产生估算事件。** `AgentLoop` 在 prepare、计算 `message_count`、传给 provider 时会多次读取 `context.messages`，当前每次读取都可能 emit `context_estimate`。
3. **Anthropic input token 会被低估。** 当前 calibration 只把 `usage.input_tokens` 当作 prompt tokens。启用缓存后，逻辑输入应按 adapter 的 usage 语义把 input、cache read、cache creation 分量归一化后再统计；否则 p50/p95 和校准系数都会显著偏低。
4. **当前 Fold compression ratio 不是 CRR。** `fold_event.folded.compression_ratio = (task_delta_tokens + tool_delta_tokens) / folded_tokens`，它是 Fold 输出相对输入的“残留比”。端到端 Reduction Ratio 还要计入 State/Agent State 开销、工具压缩、保留的近期轨迹和 Fold 模型额外调用。
5. **本地 token counter 是近似值。** 当前规则约为 `len(text) / 4`，中文、代码、JSON、tool schema 的误差不同。calibration 只校准 total，分层字段仍是未校准值。
6. **缺少请求边界快照。** Archive 能保存完整 group 和 raw refs，但没有记录“第 N 次 provider 请求看到的 event-log anchor”，无法无歧义重建每个历史 prompt。
7. **Fold 调用未计入主循环 usage。** `FoldEngine` 直接调用 provider；要同时报告主 Agent 的上下文收益和包含 Fold 开销后的净收益。
8. **工具 cap 的单位不完全一致。** 配置名和设计文档以 token 表达，实现里用 `cap * 3` 转成字符预算。测评需同时记录 raw/observation 的实际 token，不能把配置 cap 当实测值。
9. **历史 trace 不足以产出可信基线。** 仓库中的现有 trace 可用于验证 parser，但缺少 request snapshot/统一估算事件，不能用来设正式回归门限。

## 2. 指标口径

### 2.1 统计单位

最小单位是一次实际 provider API attempt，而不是 Agent step，也不是 Interaction Group。

每条记录至少有：

```text
run_id, task_case_id, repetition, variant
session_id, request_id, parent_request_id
agent_role(main|fold|subagent), step, attempt, epoch_id
provider, model, cache_mode(cold|warm|unknown)
status(success|provider_error|timeout)
```

主报告只统计 `agent_role=main AND status=success` 的请求；Fold、Subagent 和失败重试分别报告，并在“净收益”指标中加入 Fold 开销。

### 2.2 三路 prompt 重放

在同一个 request boundary 上重建三种 prompt，避免两个真实 Agent 运行因模型随机性、工具选择不同而失去可比性：

| 变体 | 内容 | 用途 |
|---|---|---|
| `raw_full` | Stable Prefix + Tools + 截止该请求的全部原始消息；Tool Result 从 Artifact 还原；不 Fold | 主基线，表示没有工具压缩和轨迹折叠时的上下文 |
| `observation_full` | 全部历史消息，但 Tool Result 使用实际进入 prompt 的压缩 observation；不 Fold | 隔离 Tool Result Compress 的收益 |
| `structured` | 实际 Task/Tool State + Recent Trajectory + Agent State | 被测值 |

三者使用相同的 system text、tool schema、请求边界和 token 计算器。Legacy ContextManager 的实跑结果作为产品对照项单列，不作为主 CRR 分母。

分解关系：

```text
raw_full ── Tool Result Compress ──> observation_full
                                     │
                                     └── Fold + State/Agent overhead ──> structured
```

### 2.3 Context Reduction Ratio

对请求 `r`：

```text
B_r = raw_full prompt tokens
O_r = observation_full prompt tokens
S_r = structured prompt tokens

CRR_r                 = 1 - S_r / B_r
ToolReduction_r       = 1 - O_r / B_r
FoldIncremental_r     = 1 - S_r / O_r
```

主指标采用 token 加权汇总：

```text
Context Reduction Ratio = 1 - sum(S_r) / sum(B_r)
```

使用加权汇总是为了表达整个 workload 实际少处理的 token；不能直接平均每个请求的百分比，因为短请求会被过度放大。报告同时给出：

- `crr_weighted`：主门禁指标；
- `crr_task_median`：先按 task 计算 CRR，再取 task 中位数，避免超长任务完全主导；
- `crr_request_p50/p95`：诊断项，不作为首要门禁；
- `negative_reduction_rate`：`S_r > B_r` 的请求占比，识别结构化 state 开销大于收益的短请求。

包含 Fold 模型输入开销后的净指标：

```text
Net CRR = 1 - (sum(S_main) + sum(Input_fold)) / sum(B_main)
```

`CRR` 与 `Net CRR` 必须同时展示。Fold output tokens、时延和成本另列，不混入 input token 定义。

注意：现有 `fold_event.folded.compression_ratio` 应重命名或在报告中标为 `fold_residual_ratio`。对应的单次 Fold reduction 是 `1 - fold_residual_ratio`，仍不能替代上述端到端 CRR。

### 2.4 input tokens 的 provider 归一化

主要分布指标使用“逻辑输入 token”，即请求完整 prompt 的 token 数，不因 cache 命中而变小：

| Provider | `logical_input_tokens` | `cache_hit_tokens` | `fresh_processed_input_tokens` |
|---|---:|---:|---:|
| OpenAI | `prompt_tokens` | `cached_tokens` | `prompt_tokens - cached_tokens` |
| Anthropic | `input_tokens + cache_read_input_tokens + cache_creation_input_tokens` | `cache_read_input_tokens` | `input_tokens + cache_creation_input_tokens` |

口径参考：[OpenAI API usage 定义](https://platform.openai.com/docs/api-reference/batch/object?api-mode=responses)、[Anthropic input/cache token 定义](https://docs.anthropic.com/en/docs/about-claude/pricing)。

所有 provider 原始 usage 字段必须原样保留。上述映射应在 adapter contract test 中用固定响应对象锁定；provider SDK/语义升级时先更新 contract，再更新基线。

如果实际 usage 缺失，记录 `token_source=estimated`，但不能把 estimated 和 provider-reported 样本混在同一个 p50/p95 门禁里。离线三路重放可统一使用同一 estimator；CRR 的比值会比绝对 token 更抗估算偏差。

### 2.5 p50 / p95 input tokens

主口径：

```text
population = 所有成功的 main-agent provider attempts
value      = logical_input_tokens
percentile = nearest-rank: sort(values)[ceil(q * N) - 1]
```

必须输出：

- `input_tokens_p50`、`input_tokens_p95`、`input_tokens_max`、`request_count`；
- `raw_full_input_tokens_p50/p95` 与 `structured_input_tokens_p50/p95`；
- 按 task bucket、provider/model、cold/warm cache、是否发生 Fold 分层；
- 每任务累计 logical input tokens 的 p50/p95，防止单请求降低但请求数大幅增加；
- `fresh_processed_input_tokens` 和 cache hit ratio，作为成本/缓存诊断，不替代 logical input 主指标。

少于 100 个请求的 slice 只展示、不做 p95 硬门禁。正式报告按 task 重采样做 bootstrap 95% CI，避免把同一任务内高度相关的多步请求当作完全独立样本。

## 3. 自动化测评架构

### 3.0 两种执行模式

本方案明确分为离线测评和正式测评，两者回答的问题不同：

| 模式 | 输入 | 是否跑完整对话 | 是否调用真实模型 | 主要回答 |
|---|---|---:|---:|---|
| Offline Replay | 已录制的 request anchors、events、trajectory archive、artifacts | 否 | 否 | 在完全相同的历史上，`raw_full`、`observation_full`、`structured` 分别有多少 token，CRR 是多少 |
| Full-run Evaluation | 任务定义 + 干净 workspace fixture + Agent 配置 | 是 | 是（也可用 scripted provider 做冒烟测试） | 完成整轮任务后，真实 p50/p95 input tokens、累计 token、成功率和质量是否改善 |

Offline Replay 是确定性、低成本的压缩算法回归测试。它直接读取已经存在的数据，对每个请求边界重建三种上下文，不执行工具、不改变 workspace，也不评价模型是否完成任务。

Full-run Evaluation 是正式端到端测评。Runner 为每个 scenario 创建隔离 workspace，从同一初始 fixture 分别运行 baseline/structured 变体，直到 completed、failed、stopped 或达到预算，然后收集完整对话、provider usage、验证结果和运行产物。

两层结果应分别输出，也应在总报告中合并：Offline 提供可严格配对的 CRR；Full-run 提供真实 p50/p95 和任务质量。不能只用 Full-run 的两次不同轨迹计算逐请求 CRR，也不能只用 Offline 结果推断任务完成质量。

### 3.1 建议目录

```text
evals/structured_context/
  __init__.py
  __main__.py                # offline/run/report/compare CLI
  schema.py                  # Scenario、RequestMetric、RunResult 数据模型
  metrics.py                 # CRR、Net CRR、nearest-rank p50/p95、bootstrap CI
  full_context.py            # 从 request anchor/archive/artifacts 重建 F_k
  runner.py                  # 隔离 workspace，执行完整 baseline/structured 任务
  scenarios.py              # 30 个任务定义、过滤与参数化
  report.py                 # requests.jsonl、summary.json、report.md
  data/
    manifest.json            # 数据集版本、bucket 数和 fixture/content profiles
    scenarios.json           # core-30 正式任务、fixture 配方和 oracle
    offline_cases.json       # offline-core-12 确定性重放配方
  fixtures/                  # 固定的小型代码仓库/合成工具输出
  golden_trajectories/       # 脱敏后的事件、archive、artifact manifest
  baselines/
    <provider>-<model>.json  # 已审核基线及环境指纹
tests/
  test_context_eval_metrics.py
  test_context_eval_replay.py
  test_context_eval_runner.py
```

首版可只使用标准库 JSONL/CSV，避免为 Parquet 引入运行时依赖。

模块职责：

- `metrics.py` 只接受结构化数值记录，不读取 trace、不执行任务，保证公式易测。
- `full_context.py` 负责生成某个请求边界的 `F_k/raw_full` 和 `observation_full`，并校验 Artifact/hash/Interaction Group。
- `runner.py` 负责复制 fixture、生成独立 session/trace 目录、运行 Agent、调用 correctness oracle 和清理临时 workspace。
- `scenarios.py` 负责声明任务，不包含测量逻辑；同一 scenario 可被 Offline fixture 生成器和 Full-run runner 使用。
- `report.py` 只消费标准 `RequestMetric/RunResult`，同一份输入应稳定生成 JSON 和 Markdown。

建议 CLI：

```bash
# 离线：读取 golden 数据，计算三路 token 和 CRR
python -m evals.structured_context offline --golden evals/structured_context/golden_trajectories

# 正式：执行 30 个完整任务，可按 bucket/provider 过滤
python -m evals.structured_context run --suite core-30 --variant raw_full --variant structured

# 与审核后的基线比较并生成门禁结果
python -m evals.structured_context compare --result .eval-results/latest --baseline evals/structured_context/baselines/<provider>-<model>.json
```

### 3.2 埋点改造

在真正调用 provider 前只生成一次不可变 request snapshot，并 emit：

```json
{
  "type": "llm_request_prepared",
  "request_id": "req-...",
  "step": 12,
  "attempt": 1,
  "agent_role": "main",
  "epoch_id": 2,
  "event_seq_anchor": 417,
  "layers_estimated": {
    "stable_prefix": 600,
    "tools": 1800,
    "task_tool_state": 7200,
    "recent_trajectory": 83000,
    "agent_state": 1100,
    "total": 93700
  },
  "replay_estimated": {
    "raw_full": 181000,
    "observation_full": 131000,
    "structured": 93700
  }
}
```

响应或异常时 emit 同一个 `request_id` 的 `llm_request_finished`，包含 `status`、原始 usage、归一化 usage、latency 和 error class。

实现约束：

- provider 接收的 messages/tools 必须来自该 snapshot，不能再次动态读取 context；
- `event_seq_anchor` 指向 prompt 生成时最后一个已提交事件；
- Fold provider 使用独立 `request_id`，并以 `parent_request_id` 指向触发它的主请求；
- raw prompt 默认不写 trace，避免体积和敏感信息扩散；只写 canonical hash、分层 token 和 replay anchor；
- calibration 使用归一化后的 `logical_input_tokens`，并按 provider/model/tokenizer version 隔离；
- 同一事件增加 `measurement_schema_version`，reporter 对未知版本 fail closed。

### 3.3 离线 deterministic replay

`python -m evals.structured_context offline` 调用 `full_context.py`，对每个 golden trajectory 的每个 request boundary：

1. 校验 event hash chain、archive group 完整性和 Artifact hash。
2. 按 `event_seq_anchor` 重建 raw messages。
3. 用 raw Artifact 替换压缩 observation，生成 `raw_full`。
4. 保留 observation 但取消 Fold，生成 `observation_full`。
5. 读取实际 snapshot，生成 `structured`。
6. 使用同一 tokenizer/estimator 计算三路 token，校验 snapshot hash。
7. 输出 `requests.jsonl`、`summary.json` 和 `report.md`。

该层不调用模型、无随机性、适合每个 PR 执行。它是 CRR 的主测量渠道。

### 3.4 端到端 paired run

`python -m evals.structured_context run` 调用 `runner.py`，对相同 case 运行：

- A：`raw_full/no_fold`；
- B：`structured`；
- 可选 C：`observation_full/no_fold`。

控制变量：相同 provider/model、system prompt、tools、workspace 初始 commit、max steps、planner 配置和输出上限；A/B 顺序随机化。模型支持 seed 时固定 seed，否则每个 case 至少重复 3 次，并按 task 聚合。

真实运行可能产生不同工具轨迹，因此它不用于逐 request CRR 配对，而用于验证：

- provider-reported p50/p95 input tokens；
- 每任务累计 input/output/cache tokens；
- 成功率、测试通过率、步数、工具调用数、超窗率；
- Fold 次数、Fallback 率、Artifact reread 率；
- TTFT/总时延和 provider error rate。

结构化方案只有在质量护栏通过时，token 改善才算有效。

### 3.5 任务定义接口

每个任务使用一个可序列化的 `Scenario`，最少包含：

```python
Scenario(
    id="tool-output-01",
    bucket="tool-output-heavy",
    task="定位测试失败原因，修复实现并运行验证。",
    fixture="fixtures/python-large-log",
    setup_commands=[],
    verification_commands=["pytest -q"],
    expected={
        "status": "completed",
        "files_exist": ["src/parser.py"],
        "tests_pass": True,
        "required_facts": ["failure is caused by malformed timestamp"],
    },
    limits={"max_steps": 30, "max_tool_calls": 80},
    tags=["large-shell-output", "artifact-reread"],
)
```

`setup_commands` 只在隔离 workspace 内运行。涉及随机数据时必须带固定 seed；fixture 初始内容必须有 hash。`expected` 是 correctness oracle，不依赖模型措辞，优先检查退出状态、测试、文件/AST 或结构化 report。

### 3.6 结果输出

每次执行输出到独立目录：

```text
.eval-results/<run_id>/
  manifest.json              # git SHA、provider/model、SDK、配置、suite、时间
  requests.jsonl             # 每个 provider attempt 的 token/层级/cache/Fold 数据
  tasks.jsonl                # 每个 scenario/variant 的状态、质量、累计 token
  summary.json               # 机器可读聚合指标和门禁结果
  report.md                  # 人读报告、bucket 对比、p95 top requests
  failures/
    <scenario_id>.json       # 失败 oracle、trace/artifact 引用，不复制敏感原文
```

`summary.json` 的最小结构：

```json
{
  "mode": "offline|full-run",
  "sample": {"tasks": 30, "main_requests": 240, "failed_requests": 1},
  "context": {
    "crr_weighted": 0.42,
    "net_crr": 0.39,
    "tool_reduction": 0.18,
    "fold_incremental": 0.29
  },
  "input_tokens": {
    "structured": {"p50": 18400, "p95": 96200, "max": 121300},
    "raw_full": {"p50": 25100, "p95": 181000, "max": 230400}
  },
  "quality": {"completed_rate": 0.97, "verification_pass_rate": 0.97},
  "gates": {"passed": true, "failures": []}
}
```

## 4. Workload 设计

### 4.1 必测 bucket

| Bucket | 数量 | 目的 | 建议构造 |
|---|---:|---|---|
| Short/control | 4 | 量化 State/Agent State 固定开销 | 1–3 steps、小输出、无需 Fold |
| Tool-output-heavy | 6 | 测 Tool Result Compress | 4K/20K/100K 的 read/grep/shell 输出，包含头尾关键信息 |
| Long-trajectory | 5 | 测 p95 与 Fold | 逐步追加完整 Interaction Groups，跨 70% trigger |
| State-churn | 4 | 测 State Compact | 大量 findings/decisions、stale/valid 混合、反复修改文件 |
| Protocol edge | 4 | 测安全裁剪 | 并行 tool calls、失败结果、correction、未闭合 group、retry |
| Resume/recovery | 3 | 测跨 epoch/恢复 | checkpoint resume、LOW/HIGH/STRUCTURAL drift |
| Multilingual/code | 2 | 测 estimator 偏差 | 中文说明、JSON、源码、日志、长路径和 Unicode |
| Subagent/fold overhead | 2 | 测净收益 | 主 Agent + subagent + LLM Fold/Fallback |
| **core-30 合计** | **30** | | |

每个 fixture 同时带 correctness oracle，例如预期文件 hash/AST 断言、测试命令、必须保留的事实和必须能通过 `read_artifact` 找回的原文片段。

任务构建分两步：先用确定性 fixture generator 生成仓库和大输出数据，再由 `scenarios.py` 绑定自然语言任务及 oracle。不要把 30 份大文件直接手写进 Python；共享 fixture 模板，通过 `size/seed/failure_mode` 参数展开场景。每个生成后的 fixture 保存 manifest 和内容 hash，Runner 启动前校验，保证 baseline/structured 的初始 workspace 完全一致。

### 4.2 执行分层

| 层级 | 频率 | 内容 | 是否用真实 API |
|---|---|---|---|
| PR quick | 每次 PR | 公式/adapter contract/unit tests + 合成 replay，至少 200 request snapshots | 否 |
| Nightly | 每晚 | golden replay 全集 + 端到端核心 case，至少 100 个成功 main requests/provider slice | 是 |
| Weekly | 每周/模型升级 | 全 bucket、A/B/C、多次重复、cold/warm cache 分层 | 是 |

真实 API 测评必须记录 model snapshot、SDK version、配置 hash、git SHA、运行区域和时间；不能把不同模型版本的结果合并。

## 5. 报告与门禁

### 5.1 标准报告

`report.md` 顶部固定输出：

```text
Quality: success rate / verification pass rate / protocol error rate
Context: CRR weighted / Net CRR / Tool Reduction / Fold Incremental
Input:   structured p50 / p95 / max / per-task cumulative p50/p95
Cache:   cache hit ratio / fresh processed p50/p95
Fold:    count / fallback rate / target-met rate / residual ratio
Sample:  tasks / successful requests / failed requests / estimated-usage count
```

随后给出按 bucket/provider/model 的分层表，以及 p95 最大的前 20 个 request（仅展示 case、step、各层 token、epoch、Fold 状态和 hash，不展示敏感 prompt）。

### 5.2 建议门禁策略

前三次 Nightly 只采集不阻断，用稳定数据生成经人工审核的 baseline。之后采用相对门禁：

- `crr_weighted` 相比基线下降不超过 3 个百分点；
- `Net CRR` 不得为负，且其 bootstrap 95% CI 下界不得跨过已审核下限；
- `structured input_tokens p50` 回归不超过 5%；
- `structured input_tokens p95` 回归不超过 8%；
- long-trajectory bucket 的 p95 不得超过配置 hard limit 的 90%；
- Fold `target_met` 率不得下降，Fallback 率不得上升超过 2 个百分点；
- correctness/verification pass rate 不得下降超过 2 个百分点；
- protocol error、不可恢复 Artifact、超限 provider request 必须为 0。

PR 离线 replay 无随机性，可采用更严格的精确门禁：同一 golden request 的 token 增长超过 `max(128 tokens, 2%)` 即输出 diff；只有被批准的 baseline update 才允许合入。

绝对目标（例如 CRR ≥ 35%）应在真实 workload 基线建立后再确定，不建议在没有样本的情况下写死。Short/control 应允许 CRR 为负，但必须单独展示；总体收益应主要来自 output-heavy 和 long-trajectory bucket。

## 6. 测试清单

### 6.1 指标单元测试

- nearest-rank p50/p95：空集、1/2/20/100 个样本和重复值。
- weighted CRR 与 task-median CRR，包含 `B=0`、`S>B`、大整数。
- OpenAI/Anthropic usage normalization 和缺失字段。
- retry join：同 step 多 attempt、失败无 usage、成功重试。
- main/fold/subagent role 隔离和 Net CRR 加总。

### 6.2 重建一致性测试

- 没发生压缩/Fold 时，`raw_full == observation_full`，structured 只允许有 state 固定开销差。
- 大 Tool Result 可从 Artifact byte-for-byte 恢复；压缩 observation 带正确 artifact id。
- 每个 request anchor 只包含当时已经发生的事件，禁止未来信息泄漏。
- Assistant tool call 与全部 Tool Result 作为完整 group 保留。
- Fold 前最后一个 structured snapshot 与 Fold 后第一个 snapshot 的 epoch/hash 连续。
- State Compact 后重要 valid finding 保留，stale-first 淘汰可解释。

### 6.3 端到端不变量

- provider 实际收到的 payload hash 等于 request snapshot hash。
- provider-reported logical input 与 calibrated estimate 的误差按 provider/model 分层；p50 误差建议 ≤ 10%，p95 误差建议 ≤ 15%。
- 任一 structured request 不超过 hard limit；超限时必须在 provider 调用前失败并可恢复。
- Fold 后 `after.total <= target_tokens`，未达到时必须有结构化原因，不能静默继续。
- 重放报告相同输入生成 byte-for-byte 相同的 `summary.json`（时间戳字段除外）。

## 7. 落地顺序

### Phase 0：修正埋点（已完成）

1. 引入 `request_id/step/attempt/agent_role/event_seq_anchor`。
2. provider 调用只消费一次构造的 immutable snapshot。
3. 增加 provider usage normalizer，并修正 Anthropic calibration。
4. FoldEngine 的调用也写 request usage；区分 `fold_residual_ratio` 和 CRR。

验收：一个带重试和一次 Fold 的测试 trace，所有 request 都能 1:1 join，没有重复 estimate。

实现状态：已加入 immutable request snapshot、逐 attempt request ID、主请求/Fold 父子关联、request payload hash、唯一 `context_estimate`、provider usage normalization，以及 Fold residual/reduction ratio 的明确字段。对应测试位于 `tests/test_phase0_instrumentation.py`。

### Phase 1：离线 replay 和 PR 门禁（已完成）

1. 实现 raw/observation/structured 三路重建。
2. 建立合成 fixtures 和至少 200 个 request snapshots。
3. 生成 JSONL + summary + Markdown diff，接入 CI。

验收：无 API key 可重复运行；连续两次输出一致；公式与 Artifact 重建测试全绿。

实现状态：默认 `offline-core-12` 生成 238 个确定性 request snapshots，支持三路
重建、CRR/Net CRR/p50/p95、逐请求 baseline gate，以及原子 JSONL/JSON/Markdown
报告；PR workflow 不使用 API key 或网络。合成 replay 与生产链路共享 Tool Result
compressor，并统一执行 recipe assertions；Fold/State 压力仍由确定性 recipe
materializer 生成。`reconstruct_prompt_variants()` 可接收真实 archive group 与 Artifact
reader，但脱敏后的生产 golden trajectory 尚未纳入仓库，因此当前 PR baseline 应标记为
synthetic，而不能替代后续 nightly live baseline。

### Phase 2：Nightly 真实 usage（执行框架已完成）

1. 按 provider/model 跑 paired suite。
2. 收集三次只读 baseline，审核异常 task。
3. 固化相对门禁和 bootstrap CI。

验收：正式报告同时包含 CRR、Net CRR、p50/p95、质量护栏和样本量。

实现状态：core-30 校验、8 类确定性 fixture、隔离 workspace、严格 oracle、
provider/harness 注入、provider-reported usage 汇总与 scripted 整轮冒烟已完成。
CLI 不会创建真实 provider；三轮真实只读 baseline 仍需在 nightly 环境显式注入
provider 后采集和人工审核，本次实现没有发起付费 API 请求。

### Phase 3：持续优化

用分层报告指导参数调优：2K 工具压缩阈值、各工具 cap、70% Fold trigger、50% target、protected group/window 和 State budget。每次只修改一组参数，用同一 golden replay 比较，避免把模型行为漂移误判为上下文策略收益。

## 8. 最小可交付定义

自动化测评可被认为“可用”必须同时满足：

1. 能从一次运行生成逐 request 的 raw/observation/structured token 记录；
2. 能输出严格定义的 weighted CRR 和 provider-reported p50/p95 logical input tokens；
3. Anthropic cache token 不会被漏算，OpenAI cached token 不会被重复算；
4. Fold 模型开销单列且进入 Net CRR；
5. 少样本、缺 usage、版本混用会显式标红，而不是静默参与聚合；
6. token 改善必须通过 correctness 和 protocol safety 护栏；
7. PR replay 无网络、确定性，Nightly live run 可按 provider/model 独立追踪趋势。
