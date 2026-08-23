# NovaCode 长期记忆系统工程落地计划书（memory_plan.md）

> **依据规范**：`memory.md`（KV-Cache 友好型结构化长期记忆系统）  
> **目标**：在 `coding_agent/` 架构下以轻量、无损 KV Cache、强抑制防污染的方式实现跨会话长期记忆闭环。

---

## 一、 系统架构与模块规划

长期记忆模块作为独立子包建立在 `coding_agent/long_term_memory/`（或 `coding_agent/memory/`）下：

```text
coding_agent/
  long_term_memory/
    __init__.py
    models.py             # 数据模型 (MemoryEntry, MemoryHeader, MemoryType, QuotaConfig)
    store.py              # 存储管理器 (YAML Frontmatter 解析/落盘, index.jsonl 同步, 命名空间隔离)
    suppression.py        # 负向抑制引擎 (7 大一票否决规则校验, 敏感词检测, 查重逻辑)
    tools.py              # Memory Tools (save_memory, update_memory, delete_memory)
    retriever.py          # 带外两阶段检索器 (Prefetch Gate, Header Scan, Side Query / Lexical Fallback)
    injector.py           # 动态插槽组装器 (<system-reminder> 封装, Freshness Guard 告警计算)
    hook.py               # 生命周期提炼 Hook (Fold / Session 结束时自主感知提炼)
```

---

## 二、 核心数据模型与接口规范

### 1. 数据模型 (`models.py`)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Literal

class MemoryType(str, Enum):
    USER = "user"
    FEEDBACK = "feedback"
    PROJECT = "project"
    REFERENCE = "reference"

@dataclass
class MemoryHeader:
    name: str
    type: MemoryType
    description: str
    created_at: str
    updated_at: str

@dataclass
class MemoryEntry:
    header: MemoryHeader
    content: str
    file_path: str = ""

    def to_markdown(self) -> str:
        """序列化为 YAML Frontmatter + Markdown Body"""
        ...

    @classmethod
    def from_markdown(cls, text: str, file_path: str = "") -> "MemoryEntry":
        """解析 YAML Frontmatter + Markdown Body"""
        ...
```

### 2. 存储管理器 (`store.py`)

- **职责**：
  - 维护物理存储目录（全局 `~/.novacode/memories/global/` + 项目 `<workspace>/.agent/memories/`）。
  - 实现 YAML Frontmatter 高速读写（`save_entry`, `load_entry`, `list_headers`, `delete_entry`）。
  - 维护 `index.jsonl` 增量同步。
  - 按修改时间倒序限制候选扫描池最多 200 条。

### 3. 负向抑制引擎 (`suppression.py`)

- **职责**：
  - 校验是否包含 API Key / Token / 密码等敏感词（正则阻断）。
  - 校验是否为未验证假设（feedback 类型需验证标志）。
  - 校验是否为纯代码事实 / 目录结构（启发式关键词检测）。
  - 校验是否为单次会话临时 Todo / 工作状态。
  - 校验查重（与现有条目描述相似度 > 80% 触发合并建议）。

### 4. 检索器与注入器 (`retriever.py` & `injector.py`)

- **职责**：
  - `PrefetchGate.can_prefetch(query, session_stats, is_subagent)` 进行门槛放行。
  - `HeaderScanner.scan_manifest(already_surfaced_ids)` 执行 <5ms 极速扫描。
  - `SideQueryEngine.select(query, manifest, provider)` 发起 256-token 仲裁，支持 `LexicalScorer` 离线 Fallback。
  - `MemoryInjector.wrap_user_message(query, selected_memories)` 组装 `<system-reminder>` 与 Freshness 警示。

---

## 三、 分阶段实施路线图（Milestone Breakdown）

### 阶段 0：基础设施与存储引擎（M0）
- [ ] 创建 `coding_agent/long_term_memory/models.py` 数据类与序列化逻辑。
- [ ] 创建 `store.py` 实现文件 I/O、YAML Header 解析与 `index.jsonl` 同步。
- [ ] 编写 M0 单元测试：验证合法 Frontmatter 解析、超大文件截断与目录隔离。

### 阶段 1：负向抑制与 Memory Tools（M1）
- [ ] 实现 `suppression.py`：落地 7 项一票否决规则与敏感词拦截。
- [ ] 实现 `tools.py`：编写 `save_memory`, `update_memory`, `delete_memory`。
- [ ] 注册工具至 `ToolRegistry`（确保仅主 Agent 拥有，Subagent 排除）。
- [ ] 编写 M1 单元测试：测试显式指令写入、同名自动合并与 7 种恶意/无效写入拦截。

### 阶段 2：两阶段检索管道与动态注入（M2）
- [ ] 实现 `retriever.py`：Gate 检查 + Header Scan + Side Query 仲裁 + 词法 Fallback。
- [ ] 实现 `injector.py`：计算时间差生成保鲜度告警（>1天附加 Warning），生成 `<system-reminder>`。
- [ ] 接入 `AgentLoop` 与 `ContextManager` / `StructuredContext`。
- [ ] 编写 M2 单元测试：测试检索准确率、去重（`already_surfaced`）、60KB 会话硬顶熔断。

### 阶段 3：自主感知触发与生命周期 Hook（M3）
- [ ] 在 `FoldEngine` 与 Session 结束切面实现 `MemoryLifecycleHook`。
- [ ] 自动提炼 `TaskState.decisions` 与 `ToolState.known_error_patterns` 中的高价值条目。
- [ ] 编写 M3 单元测试：测试长任务折叠时的自动记忆沉淀。

### 阶段 4：配置与端到端联调验收（M4）
- [ ] 在 `config.py` 中增加 `NOVACODE_ENABLE_MEMORY`、`NOVACODE_MEMORY_DIR` 配置项。
- [ ] 在 `main.py` 和 `--interactive` TUI 中完成端到端打通。
- [ ] 验证系统级 KV Cache 前缀 Hash 稳定性与跨 Session 记忆生效。

---

## 四、 关键集成点与代码变更清单

| 文件路径 | 变更类型 | 变更内容说明 |
| :--- | :--- | :--- |
| `config.py` | 修改 | 新增 `AgentConfig.enable_long_term_memory` (默认 True) 与记忆存储路径配置。 |
| `coding_agent/agent.py` | 修改 | 在 `AgentLoop.run()` 处理用户输入前调用 `retriever.prefetch()` 注入插槽；注册 memory tools。 |
| `coding_agent/tools/registry.py` | 修改 | 支持主 Agent 注册 memory tools，Subagent 构造时默认过滤排除。 |
| `coding_agent/structured_context/structured_context.py` | 修改 | 在 `add_user` 时记录 `memory_surfaced` 事件至 `EventLog`；Fold 时调用提炼 Hook。 |

---

## 五、 测试与验收准则（Acceptance Matrix）

| 验收项 | 期望指标 / 行为 | 测试验证方式 |
| :--- | :--- | :--- |
| **KV-Cache 稳定性** | System Prompt 与历史 Turn 保持 100% 逐字节不变，Cache 持续命中 | 校验 Trace 中的 `prefix_hash` 与 `tools_hash` |
| **检索耗时** | Header 扫描 $<5	ext{ms}$，本地词法 Fallback $<10	ext{ms}$，Side Query $<300	ext{ms}$ | 运行性能基准测试脚本 |
| **负向抑制拦截率** | 敏感词、未验证假设、临时 Todo、纯代码事实拦截率 100% | 运行 7 种反例用例集 |
| **时效性保鲜度警示** | 超过 1 天的历史记忆在注入时 100% 携带保鲜度警告文本 | 模拟注入 2 天前保存的记忆 |
| **会话去重与熔断** | 同一会话内相同记忆不重复注入，累计达到 60KB 自动停止召回 | 连续 10 轮相关 Query 模拟测试 |
