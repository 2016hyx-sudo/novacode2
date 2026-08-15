# NovaCode 结构化上下文管理方案 v3

> 状态：主要设计决策已确认，少数实现参数标注为 TBD。
> 前置文档：`coding_agent_context_management_v2.docx`。
> 本文档汇总 v2 需求、Checkpoint/Resume 设计以及后续多轮讨论结论，作为 V1 的设计基线。

---

## 1. 设计目标

1. 最大化 KV Cache 复用：稳定前缀在 Epoch 内逐字节不变。
2. 控制 Prompt Token 增长：工具输出入口压缩 + Recent Trajectory 动态折叠。
3. 保持对话协议完整性：裁剪/折叠以 Interaction Group 为最小单位。
4. 支持长任务连续执行：通过 Task State / Tool State / Checkpoint 保存恢复点。
5. 保证可恢复性：Raw Tool Result 无损落盘；Prompt 可以有损，磁盘不能有损。
6. 漂移安全：Resume 时区分 Agent 自身修改与外部 workspace 漂移；漂移后按影响面决定恢复策略。

---

## 2. 术语定义

### 2.1 Context Epoch

Context Epoch 是两次状态发布之间的缓存稳定区间。

在一个 Epoch 内：

- Stable Prefix 冻结；
- Task State 冻结；
- Tool State 冻结；
- Recent Trajectory 只追加；
- Agent State 每轮重建，作为临时尾部。

已确认的新 Epoch 边界：

- Session 创建；
- 显式任务切换；
- 完成一次 Trajectory Fold；
- Checkpoint/Resume 检测到 HIGH 漂移并执行 REPLAN。

TBD：配置变化、工具 Schema 变化、模型切换等边界后续补充确认。

### 2.2 Tool Result Compress

单个 Raw Tool Result 进入 Prompt 前的压缩。

- 触发条件：Raw Result 估算 token 数 > 2K；
- 处理流程：先落盘 Artifact，再执行工具特定压缩；
- 输出约束：压缩结果不超过该工具的硬上限。

### 2.3 Trajectory Fold

全局历史轨迹的状态化折叠。

- 触发条件：总 Prompt Token 达到 256K 逻辑窗口的 70%；
- 处理对象：旧的 Recent Trajectory；
- 输出：Task State Delta 与 Tool State Delta；
- 合并后删除被折叠的旧轨迹，并开启新 Epoch。

### 2.4 State Compact

Task State / Tool State 自身的容量控制操作。

- 去重；
- 淘汰 stale 条目；
- 压缩低价值历史；
- 不删除 Recent Trajectory。

### 2.5 Interaction Group

Recent Trajectory 的协议原子单位。一个包含 Tool Call 的 Assistant 消息，必须与其全部 Tool Result、Correction 绑定为同一组，不可单独裁剪或保留。

---

## 3. 总体架构

### 3.1 逻辑上下文分层

```text
256K Logical Context
│
├── Stable Prefix
│     Stable System Text
│     Native Tool Definitions（独立 tools 参数）
│
├── Persistent State
│     Task State
│     Tool State
│
├── Recent Trajectory
│     User
│     Assistant
│     Tool Result
│
└── Agent State
      每轮重建的临时尾部
```

排列原则：越稳定的内容越靠前，越动态的内容越靠后。

### 3.2 Provider 请求结构

```text
system:
  Stable System Text

tools:
  原生 Tool Definitions（固定顺序、固定序列化）

messages:
  [user] Task State + Tool State
  Recent Trajectory
    [user] ...
    [assistant] ...
    [tool] ...
  [user] Agent State
```

角色规则：

- 只有 Stable System Text 使用 `system` 角色；
- Task State 与 Tool State 合并成一个 `user` 消息；
- Recent Trajectory 保留原始角色：`user` / `assistant` / `tool`；
- Agent State 是请求末尾单独的 `user` 消息，不进 Recent Trajectory。

### 3.3 Tool Definitions 的物理位置

采用 SDK 原生 `tools` 参数，不写入提示词正文。

- OpenAI：`chat.completions.create(..., tools=[...])`
- Anthropic：`messages.create(..., tools=[...])`
- Tool Definitions 顺序、description、JSON Schema 序列化必须固定；
- `prefix_hash` 只 hash Stable System Text；
- `tools_hash` 单独 hash 原生工具定义的规范化序列化；
- Token 预算必须包含原生 tools 的估算 token 数；
- SDK 版本与模型名纳入缓存指纹关联信息。

---

## 4. KV Cache 策略

### 4.1 Anthropic 显式缓存断点

设置三个缓存断点：

```text
断点 1：Stable System Text 末尾
断点 2：原生 Tools 末尾
断点 3：Recent Trajectory 最后一个消息块末尾，Agent State 之前
```

请求结构示意：

```text
[system Stable Text]        ← cache_control 1
[tools Definitions]         ← cache_control 2
[user Task/Tool State]
[Recent Trajectory A]
[Recent Trajectory B]
[Recent Trajectory C]
[Recent Trajectory D]       ← cache_control 3
[user Agent State]          ← 不缓存
```

下一轮追加 E：

```text
system、tools、A..D 全部命中缓存
仅 E 和新 Agent State 重新计算
```

Trajectory Fold 后：

- system、tools 缓存继续命中；
- Task State / Tool State / Recent Trajectory 变化，断点 3 整体失效并重新计算。

约束：

- 缓存片段需满足目标模型的最小可缓存 token 数；
- 默认使用 ephemeral cache；
- 具体 TTL、断点数量限制以目标模型文档为准。

### 4.2 OpenAI 缓存策略

OpenAI 无显式 `cache_control`，采用：

- 固定 system、tools、messages 的字节前缀；
- 保持相同模型与 SDK 序列化方式；
- 通过 `usage.prompt_tokens_details.cached_tokens` 观察命中；
- 不承诺强制命中，只保证不主动破坏缓存条件。

### 4.3 缓存指标

记录并区分：

```text
input_tokens
cache_read_input_tokens
cache_creation_input_tokens
```

---

## 5. Stable Prefix

### 5.1 内容

Stable System Text 包含：

- System Prompt；
- Safety / Constraints；
- Agent Instructions。

### 5.2 规则

- 使用 `system` 角色；
- Epoch 内逐字节不变；
- 不包含时间、cwd、git 分支、当前任务、剩余额度等动态信息；
- 不参与 Trajectory Fold；
- 不参与 State Compact。

---

## 6. Persistent State

Persistent State 只有 Task State 与 Tool State，二者都是“当前状态”，不是历史日志。

### 6.1 Task State：What

保存 Recent Trajectory 被折叠后，对当前任务仍有长期价值的信息。

```text
task_state:
  schema_version
  task_id
  objective
  constraints[]
  success_criteria[]

  progress:
    completed[]
    current
    remaining[]        # Plan 的权威位置

  key_findings[]
  decisions[]
  unresolved[]
  extensions{}
```

规则：

- Plan 的完整有序步骤保存在 `progress.remaining`；
- Agent State 只保存 `planner.active_item_id`；
- Epoch 内 Task State 冻结；
- 运行中的 plan correction 只进 Recent Trajectory / Agent State，Fold 时 merge 回 Task State；
- 代码事实、业务行为、决策归 Task State；
- 文件变更后相关 finding 标记 `stale`，重新读取后恢复 `valid`；
- 容量硬上限默认 8K tokens，可配置。

### 6.2 Tool State：How

保存可复用的工具使用经验，不保存逐次 Tool Call 日志。

```text
tool_state:
  schema_version

  profiles:
    grep:
      useful_scopes[]
      effective_queries[]
      known_error_patterns[]

    read:
      useful_files[]
      effective_ranges[]
      known_symbols[]

    shell:
      effective_commands[]
      known_failures[]
      environment_notes[]

    test:
      effective_commands[]
      known_failures[]

  evidence_index[]
```

规则：

- 只保存“如何有效使用工具”，不重复保存属于 Task State 的代码事实；
- 路径、行号、Symbol 引用携带 `file_hash` 或 `git_head`；
- 相同工具、相同目标、相同参数模式的重复经验合并；
- 容量硬上限默认 8K tokens，可配置。

### 6.3 State 容量与淘汰

- 固定硬上限 + 比例软上限；
- 淘汰顺序：`stale` 优先 → 低价值 `completed` → 过期 evidence → 重复项；
- 淘汰评分参考：是否 stale、最后引用轮次、是否被 active Todo 引用、是否为 decision/constraint；
- 超限详情可溢出到 Artifact，Prompt 中只保留索引。

---

## 7. Recent Trajectory

### 7.1 内容与角色

Recent Trajectory 保留最近对话内容，原始角色：

```text
user
assistant
tool
```

### 7.2 维护规则

- Epoch 内 append-only；
- 只能按完整 Interaction Group 追加、保留、删除；
- 包含 Tool Call 的 Assistant 消息与其全部 Tool Result 必须属于同一组；
- 批量 Tool Call 中未执行的调用必须生成合成 Tool Result。

### 7.3 Fold 参数

```text
Fold 触发阈值：70% × 256K = 179,200 tokens
保护窗口：≤ 30% Context
Fold 停止目标：受保护窗口 + 剩余 Recent Trajectory ≤ 50% Context
```

保护内容：

- 当前用户 turn 的全部 Interaction Group；
- 最近一次失败；
- 最近一次文件修改；
- 最近一次验证结果；
- 未完成调用；
- active Todo 的直接证据。

---

## 8. Agent State

### 8.1 定义

当前 Step 的小型实时快照，由 Runtime、Planner、工具事件共同重建。

- 使用请求末尾的独立 `user` 消息；
- 每轮重建；
- 不进入 Recent Trajectory；
- 不保存历史快照；
- 不参与缓存。

### 8.2 结构策略

采用 Core / Extended 双层：

- Core 字段始终显示；
- Extended 字段按需注入；
- Schema 级限制列表长度与字符串长度。

包含：position、focus、todo、workspace、git、working set、tool execution、verification、limits、context、planner、coordination、recovery。

### 8.3 持久化

只持久化运行必需游标：

```text
position
planner
verification
limits
recovery
```

其余字段 Resume 时从 Runtime、git、文件系统和事件日志重建。

---

## 9. Token 预算

### 9.1 窗口模型

采用方案 A：固定逻辑窗口 + 模型硬限制。

```text
逻辑窗口：256K tokens
Fold 分母：固定 256K
模型硬限制：min(256K, 模型实际窗口 − 输出预留 − 协议开销)
```

达到模型硬限制前必须停止发送，不能等待 Fold。

### 9.2 Token 计数

采用方案 C：本地估算 + provider 真实 usage 校准。

- 本地估算覆盖：Stable System Text、原生 tools、Task/Tool State、Recent Trajectory、Agent State、协议开销；
- 校准方式：最近 8 次 LLM 响应的加权平均；
- 初始校准系数：1.0；
- `input_tokens` 用于正文校准；
- `cache_read_input_tokens` / `cache_creation_input_tokens` 单独记录，不参与正文校准。

---

## 10. 两个压缩环节

### 10.1 Tool Result 入口压缩

规则：

```text
raw_tokens ≤ 2000
    → 原样进入 Recent Trajectory

raw_tokens > 2000
    → Raw Result 先落盘 Artifact
    → 执行工具特定压缩
    → 压缩结果 ≤ 该工具硬上限
```

2K 是入口触发阈值，不是压缩后上限。压缩结果允许大于 2K，但不得超过分工具硬上限。

默认分工具硬上限：

| Tool | 压缩后硬上限 |
|---|---|
| ls | 1K tokens |
| grep | 2K tokens |
| read | 4K tokens |
| shell | 3K tokens |
| test | 3K tokens |
| subagent | 4K tokens |
| write / edit | 1K tokens |

全部可配置。

压缩流程：

```text
确定性压缩
    ↓ 仍超限
可选 LLM 语义压缩
    ↓ 仍超限
强制截断为 head/tail + artifact_id
```

Read 的 exact code 策略：

- ≤ 2K：保留原文；
- > 2K：保留 file、range、file_hash、symbols、关键行为、必要 exact code；
- 修改文件前必须对目标范围执行精确 re-read，以当前文件 hash 为准。

### 10.2 Trajectory Fold

流程：

```text
1. 总 Prompt Token 达到 70%
2. 仅折叠旧的 Recent Trajectory
3. 生成 Task State Delta + Tool State Delta
4. Merge 到 Task State / Tool State
5. 删除被折叠的旧轨迹
6. 开启新 Epoch
7. 重新计数
```

约束：

- Stable Prefix、Task State、Tool State、Agent State 不作为本轮折叠对象；
- 只处理完整 Interaction Group；
- 达到停止条件即停止；
- 若 Persistent State 自身过大导致无法达到目标，执行 State Compact，禁止继续删除协议相关的近期轨迹；
- Fold 后若仍超过模型硬限制，停止本轮 LLM 调用并产生可恢复错误。

Fold 模型与降级：

- V1 使用与主 Agent 相同的模型；
- 失败最多重试一次；
- 随后使用确定性规则 Fallback；
- Fallback 必须保留当前用户请求、活动 Todo、最近错误、最近修改、最近验证及未完成工具协议组；
- 无法安全 Fold 时进入 BLOCKED，不发送超限请求。

---

## 11. State Delta 与合并

Delta 操作：

```text
set
upsert
append
remove
mark_stale
```

采用 ID-based Merge：

- 每个 finding / decision / evidence 有稳定 ID；
- `upsert` / `remove` / `mark_stale` 按 ID 操作；
- `set` 只允许修改固定顶层字段；
- Delta 必须通过 Schema 校验；
- 引用的 `artifact_id` 必须存在；
- `remove` / `mark_stale` 必须指向已有条目；
- 合并后重新执行容量与一致性检查；
- 禁止将 Fold 输出作为不断追加的 Summary，每次合并后的 State 都是当前唯一物化状态。

---

## 12. Artifact、存储与安全

### 12.1 目录布局

```text
.agent/
  sessions/<session_id>/
    session.json
    task-state.json
    tool-state.json
    artifacts/
    events.jsonl
  traces/<session_id>.jsonl
```

### 12.2 CLI / 环境变量

增加 `--agent-dir` 作为总开关：

```text
--agent-dir PATH
  session_dir = PATH/sessions
  trace_dir   = PATH/traces
```

显式 `--session-dir` / `--trace-dir` 覆盖总开关推导值。

环境变量：

```text
NOVACODE_AGENT_DIR
NOVACODE_SESSION_DIR
NOVACODE_TRACE_DIR
```

默认值：

```text
session_dir = .agent/sessions
trace_dir   = .agent/traces
```

### 12.3 Raw Artifact

原则：Context 可以有损，Disk 必须无损。

- 所有 Raw Tool Result 先落盘，再生成 Tool Observation；
- Artifact 内容不可变；
- 使用 Content Hash 作为 artifact_id；
- Artifact 索引记录 tool、arguments、created_at、size、sha256、encoding、path；
- Prompt 中的压缩结果必须能通过 `artifact_id` 回溯原始输出；
- V1 不引入数据库、向量库或 Embedding。

### 12.4 read_artifact 工具

模型可通过 `read_artifact` 读取历史 Raw Artifact：

```text
read_artifact(artifact_id, start_line, end_line)
```

约束：

- 只读历史 Raw Artifact；
- 支持行范围；
- 仅限当前 Session；
- 单次最多 1000 行或 32K chars；
- 默认套用与 Tool Result 相同的敏感信息过滤规则；
- `write_file` 的 Artifact 默认不开放给模型。

### 12.5 敏感信息

- `.agent` 目录使用受限权限；
- Raw Artifact 保留原样，Prompt Observation 执行敏感信息过滤；
- 支持用户自定义脱敏规则。

---

## 13. Checkpoint / Resume

### 13.1 核心原则

1. Checkpoint 是“一致性切点”，不是单个消息文件。
2. Agent 造成的 workspace 变化必须以事件日志为唯一可信来源。
3. 恢复决策由漂移影响面决定，不由漂移是否存在决定。

### 13.2 Checkpoint 内容

```text
Checkpoint
├── LogAnchor                事件日志锚点
├── StateRefs                上下文状态快照引用
├── RuntimeCursor            运行时游标
├── WorkspaceExpected        预期 workspace / git 指纹
├── ConfigFingerprint        配置与稳定前缀指纹
├── ArtifactAnchor           Artifact 索引锚点
└── RecoveryHint             恢复策略与上次可用 checkpoint
```

其中：

- LogAnchor：`last_event_seq`、`last_event_offset`、`last_event_hash`；
- StateRefs：Task State、Tool State、Trajectory Index 的内容寻址引用与 sha256；
- RuntimeCursor：position、planner、todo、verification、limits、last_error；`pending_calls` 必须为空；
- WorkspaceExpected：workspace root、git branch/HEAD、expected_dirty、expected_untracked、postconditions、workspace_fingerprint；
- ConfigFingerprint：config_version、prefix_hash、tools_hash、context_window_limit；
- RecoveryHint：checkpoint kind、resumable、resume_action、blocked_reason、last_good_checkpoint_seq。

不保存：Stable Prefix 全文、Tool Schema 全文、Raw Artifact 正文、旧 Agent State、事件日志副本。

### 13.3 Checkpoint 保存时机

安全切点条件：

```text
没有 in-flight LLM 请求
没有正在执行的 Tool Call
没有未关闭的 Interaction Group
没有 pending_calls
没有未提交的 Trajectory Fold
events.jsonl 已 fsync 到最后一个完成事件
```

必须保存：

- Session Baseline；
- Task Boundary；
- 每个完整 Interaction Group 关闭后；
- Trajectory Fold 完成后；
- 漂移检测与恢复状态确定后；
- Graceful Stop；
- Terminal / BLOCKED。

禁止保存：

- LLM 请求或工具执行中；
- 批量 Tool Call 未全部完成时；
- Fold 未通过校验时。

### 13.4 原子性协议

```text
1. 完成逻辑状态更新
2. 追加 Event Log 事件并 fsync
3. 写不可变快照文件：tmp → fsync → rename(content hash) → fsync 目录
4. 写 checkpoint manifest：tmp → fsync → atomic rename → fsync 目录
5. 更新 CURRENT 指针：tmp → fsync → atomic rename → fsync 目录
```

提交点是 manifest 的 atomic rename。

崩溃恢复：

| 崩溃位置 | 行为 |
|---|---|
| 事件未追加或写一半 | 旧 checkpoint 有效，半行事件隔离 |
| 事件已 fsync，manifest 未提交 | 旧 checkpoint + 回放事件后缀 |
| 快照已写，manifest 未提交 | 孤儿文件忽略，GC 清理 |
| manifest 已 rename，CURRENT 未更新 | 扫描 checkpoints 找最大有效 seq |
| 工具 intent 已写，result 未写 | 进入 WAL 工具恢复流程 |

加载校验：manifest hash、快照文件 hash、事件链 hash、配置指纹、workspace fingerprint。

---

## 14. 区分 Agent 修改与外部漂移

### 14.1 ExpectedWorkspace

```text
ExpectedWorkspace = BaselineFingerprint ⊕ 所有 Agent 文件变更事件
```

### 14.2 必须记录的事件

- 文件系统工具成功后的 `file_change`：path、operation、before/after sha256、size、status；
- Shell 执行前后的 `shell_workspace_observation`：command、exit_code、observed_changes；
- Subagent 文件变更必须上抛父 Event Log，并带 depth 与 subagent_tool_call_id。

### 14.3 Resume 漂移计算

```text
Expected = checkpoint.WorkspaceExpected

for event in events[last_event_seq + 1 .. end]:
    Expected.apply(event)

Actual = scan_workspace(current)

Drift = diff(Expected, Actual)
```

扫描范围：

- git branch / HEAD；
- `git status --porcelain=v2 -z` 规范化结果；
- expected_dirty、expected_untracked、postconditions 中的文件 hash；
- focused_files、recently_modified、verification inputs 中的文件 hash。

排除：

```text
.agent/**
.git/**
配置的 ignore patterns
```

### 14.4 已知边界

- Shell 执行窗口内无法完美区分内部与外部修改，统一记入 shell observation；
- 外部修改后又改回原内容，最终 hash 相同则无法检测，但不影响执行正确性；
- 大目录默认不纳入指纹，按需通过配置纳入。

---

## 15. 漂移恢复决策

### 15.1 漂移分级

| 级别 | 定义 |
|---|---|
| NONE | Actual == Expected |
| IGNORED | 仅在忽略范围内有差异 |
| LOW | 差异路径不在当前执行影响集内 |
| HIGH | 差异影响当前目标、Todo 证据、focused/modified 文件、postcondition 或 verification |
| STRUCTURAL | branch/HEAD 改变、conflict、workspace 缺失、checkpoint 不可恢复、大规模未知变更 |

### 15.2 影响集

```text
active_todo.evidence_paths
focused_files
recently_modified
postconditions
verification.inputs
Task State 中 status=valid 的 key_findings.evidence
当前 Interaction Group 中 tool_calls 的 path 参数
```

命中任意一个 → 至少 HIGH。

### 15.3 决策状态机

```text
checkpoint / event log 有效？
  否 → BLOCKED
  是 → 计算 Actual vs Expected

NONE / IGNORED → RESUME
LOW            → RESUME + 标记 stale
HIGH           → REPLAN
STRUCTURAL     → BLOCKED
REPLAN 失败    → BLOCKED
```

### 15.4 RESUME

适用于 NONE / IGNORED / LOW：

- 保留 Task State、Tool State、Recent Trajectory；
- 从当前 Runtime 重建 Agent State；
- LOW 时标记相关 finding / verification 为 stale；
- 继续 `next_action`；
- 创建 Recovery Transition checkpoint。

### 15.5 REPLAN

进入条件：

```text
checkpoint 或重建状态有效
workspace root 存在且可读
git branch / HEAD 未变化
无 conflict / rebase / detached 状态
漂移文件数 ≤ max_replan_drift_files
漂移路径均为可读普通文件
objective / constraints / success_criteria 仍存在
没有未知副作用的未完成工具调用
replan_attempts < max_replan_attempts
autonomous_replan_enabled = true
```

动作：

1. 开启新 Context Epoch；
2. 保留 objective、constraints、success_criteria；
3. 不恢复旧 pending_calls、next_action；
4. 生成 drift_report；
5. 标记受影响 State 条目为 stale；
6. verification 置为 stale；
7. Planner 基于 Task State + drift_report + Actual Agent State 重新规划；
8. 写入新 checkpoint；
9. 继续执行循环。

### 15.6 BLOCKED

进入条件：

```text
git branch / HEAD 改变
存在 merge conflict
workspace root 丢失或 .git 损坏
checkpoint 不可恢复且 Event Log 无法重建
工具批次中断且副作用不可判定
漂移规模超过 REPLAN 可处理阈值
连续 REPLAN 失败
策略要求用户确认
```

动作：

1. 不调用普通 LLM 任务；
2. 不执行写工具；
3. 只允许只读诊断；
4. 持久化 `kind=blocked` checkpoint 与 drift report；
5. `resume_action = await_user_decision`；
6. 向用户提供：接受 workspace 并 REPLAN、恢复 checkpoint 预期状态、切换分支/HEAD、丢弃外部改动。

### 15.7 策略配置

```text
checkpoint.every_n_groups = 1
checkpoint.min_interval_ms = 500

drift.ignore_patterns = [".agent/**", ".git/**", "node_modules/**", "__pycache__/**"]
drift.max_replan_files = 20
drift.autonomous_replan = true
drift.git_head_change_policy = "blocked"
drift.low_impact_policy = "resume_and_stale"

recovery.max_replan_attempts = 2
recovery.require_user_confirmation_for_structural = true
```

---

## 16. Subagent 上下文

- Subagent 使用独立 ContextManager / 独立上下文；
- 只回传结构化报告：

```text
findings
evidence
blockers
next_action
```

- 报告硬上限 4K tokens；
- 子 Agent 修改文件清单必须上抛父 Event Log；
- 子轨迹可落盘，父 Agent 通过 Artifact 按需查看；
- 不自动把子 Task State 合并进父 Task State。

---

## 17. Verification stale 传播

采用文件 hash 快照为主，涉及文件集合为辅。

- verification 通过时记录相关文件 hash 与涉及文件集合；
- 后续修改或 Resume 时比较 hash；
- hash 不一致 → verification 置为 `stale`；
- 路径级传播：只 stale 依赖漂移路径的 verification；
- 命令解析只做 best-effort，不作为唯一依据。

---

## 18. Metrics 与可观测性

采用 Trace 明细 + Fold 事件 + Session 聚合：

- 每个 LLM 请求一条明细事件：
  - 各层 token；
  - usage_ratio；
  - prefix_hash；
  - tools_hash；
  - cache_read / cache_creation；
- 每次 Fold 一条事件：
  - Fold 前后各层 token；
  - 压缩比；
  - State 变化；
  - 删除的 Interaction Group；
- session.json 保存聚合指标。

---

## 19. 旧 Session 迁移

- 旧 `.sessions/<id>.json` 做一次性转换；
- 转换内容：messages → Recent Trajectory，plan → Task State，metadata 保留；
- 转换后重新扫描 workspace baseline；
- 旧文件保留只读，新格式另存；
- 新格式必须带 `schema_version`，支持后续迁移链。

---

## 20. 非目标

- 不引入向量数据库、Embedding 或语义检索服务；
- 不实现跨项目共享长期 Tool State；
- 不实现多级摘要树或无限层次 Memory；
- 不允许模型直接修改 Runtime 权威字段；
- 不依赖 provider 服务端自动截断维持上下文可用性。

---

## 21. 决策记录

| 项 | 决策 |
|---|---|
| Context Window | 固定 256K 逻辑窗口 + 模型硬限制 |
| Token Counting | 本地估算 + provider usage 校准；区分 cache tokens |
| Tool Definitions | SDK 原生 tools 参数，不写入提示词 |
| KV Cache | Anthropic 三段 cache_control；OpenAI 保持前缀稳定并观察 cached_tokens |
| 角色映射 | 仅 Stable System Text 用 system；Task/Tool State 合并 user；Recent Trajectory 保留原角色；Agent State 尾部 user |
| Task/Tool State 边界 | Task State = What；Tool State = How |
| Plan 位置 | Task State.progress.remaining；Agent State 只放 active_item_id |
| Epoch 内更新 | Task/Tool State 冻结；plan correction 在 Fold 时 merge |
| State 容量 | 各 8K 硬上限 + 比例软上限；stale 优先淘汰；可溢出 Artifact |
| State Merge | ID-based Delta：set / upsert / append / remove / mark_stale |
| Tool Result 压缩 | 2K 触发；分工具硬上限；允许压缩后大于 2K |
| Read exact code | ≤2K 原文；>2K 保留关键 exact code；修改前精确 re-read |
| Trajectory Fold | 70% 触发；保护窗口 ≤30%；目标回落 ≤50% |
| Fold 模型 | 主 Agent 同模型；失败重试一次；确定性 Fallback；失败进入 BLOCKED |
| Artifact ID | Content Hash；不可变；Raw 无损落盘 |
| Artifact 读取 | read_artifact(artifact_id, start_line, end_line)；仅当前 Session；write_file Artifact 不开放 |
| 存储布局 | `.agent/sessions/<id>/` + `.agent/traces/<id>.jsonl` |
| CLI | `--agent-dir` 总开关；`--session-dir` / `--trace-dir` 可覆盖 |
| 敏感信息 | Raw 受限权限；Prompt 脱敏；用户自定义规则 |
| Subagent | 结构化报告 ≤4K；文件变更上抛父日志 |
| Verification | 文件 hash 快照为主，涉及文件集合为辅 |
| Metrics | Trace 明细 + Fold 事件 + Session 聚合 |
| 旧 Session | 一次性迁移，旧文件保留，schema_version |
| Checkpoint | LogAnchor + StateRefs + RuntimeCursor + WorkspaceExpected + ConfigFingerprint + ArtifactAnchor + RecoveryHint |
| Checkpoint 时机 | 安全切点；每个 Interaction Group 关闭后等 |
| 原子性 | Event Log 先落盘，manifest atomic rename 为提交点 |
| 漂移判定 | Actual Workspace − Expected Workspace − 可解释事件 |
| 恢复策略 | NONE/IGNORED/LOW → RESUME；HIGH → REPLAN；STRUCTURAL → BLOCKED |

---

## 22. 仍开放 / TBD

以下项目前仅剩实现期参数，不阻塞总体架构：

1. Context Epoch 的完整边界清单（配置变化、工具 Schema 变化、模型切换）；
2. Fold 选择算法中“最近一次失败/修改/验证”的精确定义；
3. 分工具压缩 Schema 的最终字段与示例；
4. State 淘汰评分的精确公式与权重；
5. Verification 涉及文件集合的精确记录方式；
6. `--agent-dir` 与 `--session-dir` / `--trace-dir` 的最终优先级细节；
7. 各 provider 实际缓存最小 token 数与 TTL 的运行时适配。
