# 技术方案规范：KV-Cache 友好型结构化长期记忆系统

## 1. 架构目标与核心原则

构建一个具备高信噪比、低延迟、零主链路阻塞，且与「结构化上下文管理」及「KV Cache 前缀缓存」深度协同的长期记忆系统：

1. **静态前缀绝对冻结（Prefix Invariance）**：系统提示词（System Prompt）保持 100% 静态，禁止混入动态记忆内容或动态索引，确保系统级 KV Cache 始终稳定命中。
2. **多轮历史单调递增（Append-Only Multi-turn）**：召回的记忆仅作为「当前轮次（Current Turn）」的动态插槽注入，严禁修改已生成的历史消息，确保多轮对话历史的 KV Cache 持续复用。
3. **带外异步两阶段检索（Out-of-Band Prefetch）**：检索过程脱离主对话上下文，由独立轻量小模型（Side Query）结合本地高速元数据扫描完成，主会话上下文纯净且无额外 Token 负担。
4. **全生命周期闭环治理（Full-Lifecycle Governance）**：构建包含「显式指令驱动 + 自主场景感知」的写入通道，并设立「强约束负向抑制网（一票否决）」与「时效性保鲜度警示」，防止记忆库膨胀与认知幻觉。

---

## 2. 记忆分类体系与边界治理规范（Taxonomy & Governance）

### 2.1 结构化分类维度（Taxonomy）

记忆库严格划分为以下 4 种独立语义类型，每种类型具有明确的生命周期与触发标准：

| 记忆类型 (Type) | 语义定义 | 触发与采集时机 | 典型内容示例 |
| :--- | :--- | :--- | :--- |
| **`user`**<br>(用户画像/偏好) | 用户的技术背景、特定编码习惯、工具链偏好、语言/框架选型倾向。 | 用户显式表达习惯，或连续多次纠正 AI 的输出风格。 | `代码风格：Python 必须开启严格类型注解；前端优先使用 TailwindCSS。` |
| **`feedback`**<br>(纠错与经验) | 用户指出的具体错误、踩坑教训，以及修正后的正确做法与原因。 | 发生工具调用失败后成功定位根因、逻辑 Bug 被纠正、特定环境避坑。 | `打包避坑：在 Windows 环境下构建时必须传 --no-daemon 参数，否则进程锁死。` |
| **`project`**<br>(项目架构决策) | 跨会话的关键技术决策、里程碑目标、核心模块设计原则。 | 用户阐述架构设计、确定重大重构方案或指定技术规范。 | `架构约束：认证模块必须走独立的 AuthMiddleware，禁止在 Controller 中直接鉴权。` |
| **`reference`**<br>(外部指针与规范) | 外部资源入口、API 文档指针、常用工具链指令集合。 | 涉及高频调用的外部服务、私有文档链接或固定调试命令。 | `监控入口：Prometheus 接口位于内网 :9090，健康检查接口统一为 /api/healthz。` |

---

### 2.2 严格存储边界治理（Boundary Governance）

为防止记忆库膨胀、信噪比下降及上下文污染，建立强约束的「负向存储清单」：

【坚决禁止存储（Negative Boundaries）】
- ❌ **代码库已有事实**：禁止存储函数签名、文件架构、目录树（应直接通过 `read_file` / `grep_search` 等工具读取代码库）。
- ❌ **版本控制日志**：禁止存储 Git 提交历史、分支变更（应调用 `git log` 查询）。
- ❌ **会话临时状态**：禁止存储当前单次任务的中间结果、待办进度（交由会话内的 Working Memory / TaskState 处理）。
- ❌ **模糊猜测**：禁止存储未经验证的假设，feedback 类型必须包含「为什么(Why) + 如何应用(How)」。
- ❌ **重复信息**：写入前必须检索去重，优先合并更新已有条目，禁止创建语义高度重叠的文件。
- ❌ **敏感凭证信息**：严禁存储 API Key、Token、私钥、密码等安全凭证。

---

## 3. 记忆写入机制与触发矩阵（Write Triggers & Negative Suppression）

```text
                              ┌──────────────────────────────────────────────┐
                              │                 用户交互与执行事件             │
                              └──────────────────────┬───────────────────────┘
                                                     │
                         ┌───────────────────────────┴───────────────────────────┐
                         ▼                                                       ▼
        【通道 1：用户显式指令】                                    【通道 2：模型自主感知】
     • "记住/偏好/踩坑/架构规范"                                 • 4 类高价值场景触发
                         │                                                       │
                         └───────────────────────────┬───────────────────────────┘
                                                     │
                                                     ▼
                                        ┌────────────────────────┐
                                        │  一票否决：负向抑制校验  │ ◄── 7 大禁写规则
                                        └────────────┬───────────┘
                                                     │ 校验通过
                                                     ▼
                                        ┌────────────────────────┐
                                        │  检索查重 (Header Scan) │ ◄── 检查是否存在同名/同语义条目
                                        └────────────┬───────────┘
                                                     │
                                      ┌──────────────┴──────────────┐
                                      ▼                             ▼
                            [新条目] save_memory          [已有条目] update_memory
```

### 3.1 通道一：用户显式指令触发（Explicit User Commands）

当用户明确发出指令要求模型“记住”、“保存”、“记录偏好”或“沉淀结论”时，模型最高优先级响应并调用工具落盘。

#### 1. 触发关键词与模式
- 偏好指令：“记住，我以后写 Python 都必须加严格类型注解”、“记录我的前端偏好：统一使用 TailwindCSS”。
- 避坑指令：“把这个 Bug 原因和排错命令记下来”、“记录这个踩坑点，下次构建注意”。
- 架构/规范：“沉淀一下这条架构决策”、“保存我们的数据库迁移规范”。
- 指针/配置：“记住测试环境的统一入口和 Token 格式”。

#### 2. 工具交互与落盘
模型通过调用 `save_memory` 或 `update_memory` 完成持久化，并在最终回答中给出显式确认反馈（例如：*“已将【TailwindCSS 前端偏好】保存至长期记忆（类别: user, ID: tailwind_preference）”*）。

---

### 3.2 通道二：模型自主感知触发（4 类高价值场景）

在用户未显式发出指令时，模型在解决问题或推进任务过程中自主识别高价值经验并沉淀：

| 场景类型 | 对应语义分类 | 自主感知触发时机（When） | 沉淀内容规范（What） |
| :--- | :--- | :--- | :--- |
| **1. 错误纠正与避坑突破** | `feedback` | 工具报错后经过排查找到根本原因并**验证修复成功**（Verified Fix）；或用户明确纠正了逻辑错误并给出正确做法。 | 记录错误症状、根因机制（Why）、验证有效的解法与命令（How）以及适用环境。 |
| **2. 重大架构与设计决策** | `project` | 讨论并确认了跨模块架构设计、技术选型方案、接口规范或重构约束，具有跨会话持久效力。 | 记录架构规则、决策背景、约束条件与弃用方案。 |
| **3. 环境与工具链隐式知识** | `reference` | 在执行命令/构建时发现了非标准端口、专用私有镜像源、特定环境参数或专用脚本用法。 | 记录服务用途、访问入口/指令、依赖前置条件。 |
| **4. 用户习惯被反复验证** | `user` | 用户在连续 2 次以上的交互中展现出一致的编码风格要求（如要求全部改用特定函数式写法）。 | 记录具体风格规则、适用语言框架、推荐写法与反例。 |

#### 自主感知的执行模式
- **实时 Tool 模式**：在问题验证解决的当前 Turn 中直接调用 `save_memory`。
- **生命周期切面提炼（Fold / End Hook）**：在 `Trajectory Fold` 或 Session 结束时，由后台轻量模型扫描 `TaskState.decisions` 与 `ToolState.known_error_patterns`，提炼并静默持久化。

---

### 3.3 通道三：负向抑制时机（7 项一票否决规则）

在任何写入触发时，若命中以下任意 1 项，**立即阻断写入**：

1. **❌ 严禁写入未经验证的临时假设**：Bug 尚未修复或测试未跑通前的临时猜测，绝不可作为 `feedback` 写入。
2. **❌ 严禁写入代码库已有事实**：函数签名、文件路径、目录结构等可通过代码工具随时检索的内容，严禁复制到记忆中。
3. **❌ 严禁写入单次任务的即时工作状态**：当前会话的 Todo、修改的行号、临时文件路径等只属于 Working Memory，严禁外溢。
4. **❌ 严禁泛化偶发性单次指令**：用户单次的临时测试要求（如“这次先打印前 5 行看看”）严禁泛化为全局用户偏好。
5. **❌ 严禁创建语义重复的冗余文件**：写入前必须执行 Header 查重，语义重叠度 > 80% 时必须转为 `update_memory` 合并更新。
6. **❌ 严禁存储敏感凭证（Secret Hard-Stop）**：包含 `sk-`、`ghp_`、私钥、密码、Token 等敏感字符串一律硬阻断。
7. **❌ 严禁子 Agent（Subagent）写入**：Subagent 运行在受限隔离上下文中，严禁赋予记忆写入工具，防止记忆库被垃圾碎片污染。

---

### 3.4 Memory Tools 接口规范

向 Agent 注册的标准记忆操作工具：

```python
# 1. 保存/新增记忆
save_memory(
    name: str,          # 唯一标识名 (字母数字下划线，40字符内)
    type: str,          # 取值: user | feedback | project | reference
    description: str,   # 单行精准摘要 (<=80字，召回核心依据)
    content: str        # Markdown 正文 (规则详情、代码示例、操作指导)
) -> ToolResult

# 2. 更新/合并已有记忆
update_memory(
    name: str,          # 目标记忆的唯一标识名
    patch_mode: str,    # "replace" (完全覆写) 或 "append" (追加正文)
    description: str | None = None, # 可选更新摘要
    content: str = ""   # 替换或追加的正文内容
) -> ToolResult

# 3. 删除过时记忆
delete_memory(
    name: str,          # 待删除记忆标识名
    reason: str         # 删除原因
) -> ToolResult
```

---

## 4. 存储与索引架构设计（Storage & Indexing）

### 4.1 数据建模：元数据头与正文解耦

每条记忆物理上采用单文件存储，文件格式为「YAML Frontmatter + Markdown Body」：

```yaml
---
name: string        # 唯一标识名 (字母数字下划线，40字符内)
type: enum          # 取值仅限: user | feedback | project | reference
description: string # 单行精准摘要 (不超过 80 字，召回决策的核心依据)
created_at: string  # ISO-8601 UTC 时间戳
updated_at: string  # ISO-8601 UTC 时间戳
---
[Markdown 正文内容]
详细规则、背景说明、具体代码示例或操作指导。
```

### 4.2 命名空间与物理隔离

物理存储路径按作用域进行隔离：
- **全局用户记忆（跨项目共享）**：`~/.novacode/memories/global/`（主要存放通用 `user` 偏好）。
- **项目专属记忆（项目级隔离）**：`<workspace>/.agent/memories/` 或 `~/.novacode/memories/projects/<workspace_hash>/`（存放特定代码库的 `project`, `feedback`, `reference`）。

### 4.3 带外全局索引自动同步

- **索引内容**：在记忆库根目录维护聚合索引清单 `index.jsonl`（包含所有条目的 `name`、`type`、`description` 及时间戳）。
- **切面更新**：记忆的新增、修改、删除通过底层存储切面自动触发索引更新。
- **解耦原则**：该索引仅供后台检索服务带外读取，**严禁硬编码至静态系统提示词前缀中**。

---

## 5. KV-Cache 友好的上下文管理与注入协议

```text
┌─────────────────────────────────────────────────────────────┐
│ 静态系统提示词 (System Prompt Prefix)                        │ ◄── 100% 保持静态 (命中系统级 KV Cache)
│ - Agent 角色准则 / 静态工具 Schema / 记忆存取规则说明         │
├─────────────────────────────────────────────────────────────┤
│ 历史轮次消息 (Turn 1 .. Turn N-1)                           │ ◄── 严格只读 Append-Only (命中多轮 KV Cache)
│ - User / Assistant / Tool 调用记录 (禁止回溯修改)           │
├─────────────────────────────────────────────────────────────┤
│ 当前轮次 (Current Turn N)                                   │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 动态记忆插槽 (Structured Memory Slot / <system-reminder>)│ │ ◄── 仅在此处注入本轮召回的记忆
│ │ - 附带时效性保鲜度警示 (Freshness Warning)              │ │
│ └─────────────────────────────────────────────────────────┘ │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 用户的最新消息 (User Query)                             │ │
│ └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 5.1 系统提示词静态规范

系统提示词中仅包含记忆系统的静态操作规约，示例标准文本如下：

```markdown
# Memory Operations
You have access to a persistent memory repository.
- Use dedicated memory tools (save_memory, update_memory, delete_memory) to persist durable user preferences, critical feedback, project constraints, and references.
- Always provide structured metadata (name, type, description) when creating memories.
- Treat recalled memories as contextual background, but always verify claims against live code.
```

### 5.2 结构化插槽注入格式

召回的记忆被封装在独立的 XML 标签中，置于**当前轮次 User 消息的最前端**：

```xml
<system-reminder>
[保鲜度警示语 (若存在)]
Memory ID: {name} (Category: {type}, Saved: {Relative Age})
{Body Content}
</system-reminder>

{User Query Content}
```

---

## 6. 带外两阶段召回管道（Out-of-Band Retrieval Pipeline）

```text
    UQ["用户输入 Query"] --> GATE{"触发门槛校验<br/>1. 非子Agent<br/>2. 包含有效关键词<br/>3. 会话内存配额 < 60KB<br/>4. 记忆库非空"}
    GATE -- "否" --> SKIP["跳过动态召回"]
    GATE -- "是" --> PREFETCH["执行两阶段检索 (Prefetch)"]

    subgraph "带外检索核心 (轻量独立)"
        PREFETCH --> S1["阶段一: 快速扫描元数据 Header<br/>• 极速 I/O 解析 YAML<br/>• 过滤当前会话已曝光项 (already_surfaced)<br/>• 组装轻量 Candidate Manifest"]
        S1 --> S2["阶段二: 独立轻量 Side Query / Lexical Fallback<br/>• 调用极小模型 (max_tokens=256)<br/>• 输入 Query + Manifest<br/>• 输出 Top-K 选中的 Memory Name 列表"]
        S2 --> S3["按需读取完整 Body<br/>• 单篇最大 4KB 截断"]
    end

    S3 --> FRESH{"时效性检测 (修改时间 > 1天?)"}
    FRESH -- "是" --> WARN["附加 Freshness Warning 警示"]
    FRESH -- "否" --> NORMAL["标注相对时间 (e.g. today)"]

    WARN --> INJECT["包装为 <system-reminder> 结构化插槽"]
    NORMAL --> INJECT
    INJECT --> MERGE["合并进当前轮次 User 消息前缀"]
    MERGE --> STATE["更新已曝光集合 & 累加会话已用字节数"]
```

### 6.1 检索门槛检查（Prefetch Gate）

在启动检索前必须满足以下条件，否则直接跳过以节约算力：
1. **角色检查**：仅主 Agent 触发，子 Agent（Subagent）严禁触发。
2. **输入长度**：必须包含有效关键词，单字符或简单控制命令（如 exit、clear）不触发。
3. **容量预算**：当前会话内已注入记忆总量未超过会话硬顶（默认 60 KB）。
4. **非空检查**：记忆库中存在有效条目。

### 6.2 阶段一：元数据轻量扫描（Header Scan）

- **极速 I/O**：仅读取记忆条目的前 30 行解析 YAML 元数据（或读取缓存的 `index.jsonl`）。
- **会话去重**：从候选池中剔除当前会话已经注入过的记忆（`already_surfaced` 集合）。
- **组装 Manifest**：将候选条目拼装为紧凑清单：
  `- [type] name (timestamp): description`

### 6.3 阶段二：独立小模型语义仲裁（Side Query）与 Fallback

调用独立的轻量模型（`max_tokens=256`），使用专用的决策提示词：

```markdown
[Side Query System Prompt]
You are a memory selection system. You are given a user's query and a list of available memories with their types, names, and one-line descriptions.
Select ONLY the memories that are directly and clearly relevant to helping answer the query.

Constraints:
- Return a JSON object: {"selected_memories": ["name1", "name2"]}
- Select at most 5 memories.
- If unsure whether a memory is helpful, DO NOT include it.
- If no memories are relevant, return {"selected_memories": []}.
```

- **离线 / Fallback 模式**：在未配置 Secondary Model 或无网络时，自动降级为基于 BM25 / 词法重叠度的确定性打分引擎，挑选 Top-K。

---

## 7. 时效性感知与安全防护机制（Safety & Quota Guards）

### 7.1 记忆保鲜度防御机制（Freshness Guard）

为了避免 Agent 盲目信任过时的历史代码信息，根据生成时间戳动态生成警示：

- **当天/次日生成**：标注 `(saved today)` 或 `(saved yesterday)`。
- **历史记忆（> 1 天）**：强制在记忆正文上方附加系统警告：
  > *⚠️ This memory is {N} days old. Memories are point-in-time observations, not live state — claims about code behavior may be outdated. Verify against current code before asserting as fact.*

### 7.2 多级容量熔断（Multi-tier Quota Limits）

| 层级 | 配额上限 | 超额处理策略 |
| :--- | :--- | :--- |
| **单条记忆正文** | 4 KB（约 1000-1500 Tokens） | 尾部截断并附带 `[... truncated, memory file too large ...]` 标识 |
| **单轮召回数量** | 最多 5 条 | 按小模型置信度与相关度硬截断 |
| **单会话累计注入** | 60 KB | 达到阈值后，该会话后续轮次自动熔断停止召回，保护上下文有效窗口 |
| **候选扫描池** | 最多 200 个条目 | 按修改时间倒序取最近活跃的 200 条参与扫描 |

---

## 8. 落地验收标准（Acceptance Criteria）

1. **KV Cache 命中验证**：System Prompt 与历史 Turn 保持 100% 不变，前缀 Hash 始终恒定。
2. **检索性能**：Header 扫描耗时 < 5ms；Side Query 耗时控制在 300ms 以内（或支持快速 Lexical Fallback）。
3. **写入与抑制准确性**：
   - 显式指令 100% 成功触发工具调用与落盘；
   - 命中 7 项负向抑制规则的内容被 100% 拦截；
   - 写入重复内容时自动触发 `update_memory` 合并而非新建文件。
4. **防幻觉与去重**：
   - 同一条记忆在同一个 Session 内绝不重复注入；
   - 超过 1 天的旧记忆注入时必须包含保鲜度告警。
