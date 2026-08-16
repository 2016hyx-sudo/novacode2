# NovaCode 结构化上下文管理 v3：详细 Schema 与实现拆解草案

> 状态：第一版详细设计草案，用于逐节评审。
> 前置文档：`coding_agent_context_management_v3.md`。
> 约束：本阶段只做设计，不写实现代码。

---

## 1. 文档目的

本文档把 v3 已确认方案拆成：

1. 持久化文件与运行时数据结构的 Schema；
2. 模块职责与依赖关系；
3. 核心流程的详细拆解；
4. 实现里程碑与验收顺序。

---

## 2. 详细存储布局

在 v3 基础上细化 session 目录：

```text
.agent/
  sessions/<session_id>/
    session.json
    task-state.json
    tool-state.json
    trajectory.json
    trajectory-archive.jsonl
    events.jsonl
    artifact-index.jsonl
    artifacts/
      <tool_name>/
        <sha256>.artifact
    checkpoints/
      ckpt-000000.json
      ckpt-000001.json
      ...
    CURRENT
  traces/<session_id>.jsonl
  locks/<session_id>.lock
```

文件职责：

| 文件 | 职责 | 写入方式 |
|---|---|---|
| session.json | Session manifest、Runtime Cursor、聚合指标 | 原子替换 |
| task-state.json | Task State 物化快照 | 原子替换 |
| tool-state.json | Tool State 物化快照 | 原子替换 |
| trajectory.json | Recent Trajectory 物化快照 | 原子替换 |
| trajectory-archive.jsonl | 完整未折叠 Interaction Group 历史 | append-only |
| events.jsonl | 权威事件日志 / WAL | append-only |
| artifact-index.jsonl | Artifact 索引 | append-only |
| artifacts/ | Raw Tool Result 与必要原文 | 不可变写入 |
| checkpoints/ | Checkpoint manifest | 不可变写入 |
| CURRENT | 当前 checkpoint 指针 | 原子替换 |
| traces/ | 可观测性指标事件 | append-only |
| locks/ | Session 单写者锁 | lock file + OS lock |

原则：

- `events.jsonl` 是恢复的权威来源；
- `trajectory-archive.jsonl` 是完整未折叠对话历史，append-only；
- `session.json`、`task-state.json`、`tool-state.json`、`trajectory.json` 是从事件日志物化出的快照；
- Artifact 与 Checkpoint 不可变；
- 除事件日志和 Artifact 外，其他文件更新都必须经过 Checkpoint 提交协议。

---

## 3. 公共字段约定

### 3.1 版本

所有持久化对象包含：

```text
schema_version: string
```

### 3.2 ID 规则

| ID 类型 | 规则 |
|---|---|
| session_id | 时间戳 + 随机后缀 |
| group_id | `g-<event_seq>`，由关闭该组的事件 seq 决定 |
| step_id | `s-<step>` |
| tool_call_id | 保留 provider 返回 ID；缺失时生成 `call_<step>_<n>` |
| artifact_id | `sha256(raw bytes)` |
| finding_id | `f-<epoch>-<n>` |
| decision_id | `d-<epoch>-<n>` |
| evidence_id | `e-<epoch>-<n>` |
| todo_id | `t-<plan_step_id>` 或 `t-<epoch>-<n>` |
| checkpoint_id | `sha256(canonical manifest)` |

原则：ID 必须稳定、可校验、可被 State Delta 引用。

### 3.3 时间与序列

- 时间统一 ISO-8601 UTC；
- `event_seq` 从 1 开始单调递增，由 Event Log 分配；
- `step` 为 Agent Loop 内 LLM 步数；
- `epoch_id` 从 0 开始单调递增。

### 3.4 Hash

统一使用 SHA-256，十六进制表示。

---

## 4. session.json Schema

```json
{
  "schema_version": "1.0",
  "session_id": "20260815T120000-a1b2c3",
  "status": "running | completed | failed | blocked",
  "created_at": "2026-08-15T12:00:00Z",
  "updated_at": "2026-08-15T12:05:00Z",
  "provider": "anthropic",
  "model": "claude-sonnet-4-5",
  "workspace_root": "/abs/path",
  "config_fingerprint": {
    "config_version": "v3",
    "prefix_hash": "sha256:...",
    "tools_hash": "sha256:...",
    "sdk_version": "anthropic==0.122.0",
    "context_window_limit": 256000,
    "model_hard_limit": 190000
  },
  "epoch": {
    "epoch_id": 3,
    "start_event_seq": 810,
    "started_at": "2026-08-15T12:04:00Z",
    "started_by": "fold | task_boundary | session_start | replan"
  },
  "state_refs": {
    "task_state": {"sha256": "sha256:...", "size": 8231},
    "tool_state": {"sha256": "sha256:...", "size": 5210},
    "trajectory": {"sha256": "sha256:...", "size": 24000}
  },
  "runtime_cursor": {
    "position": {
      "user_turn": 3,
      "step": 17,
      "epoch": 3,
      "phase": "verification",
      "status": "running",
      "attempt": 0
    },
    "planner": {
      "plan_version": 2,
      "active_item_id": "plan-12"
    },
    "todo": {
      "active": {"id": "t-12", "text": "add health check", "status": "active"},
      "next": [],
      "blocked": []
    },
    "verification": {
      "required": true,
      "status": "passing",
      "last_command": "pytest tests/test_health.py",
      "failing_tests": [],
      "artifact_id": "sha256:..."
    },
    "limits": {
      "steps_used": 17,
      "max_steps": 30,
      "tool_calls_used": 23,
      "tool_calls_remaining": 57,
      "subagents_used": 1
    },
    "recovery": {
      "blocked": false,
      "checkpoint_id": "sha256:...",
      "resumable": true,
      "resume_action": "continue_next_action"
    }
  },
  "last_checkpoint": {
    "seq": 42,
    "checkpoint_id": "sha256:...",
    "created_at": "2026-08-15T12:04:58Z"
  },
  "metrics": {
    "total_steps": 17,
    "total_tool_calls": 23,
    "fold_count": 2,
    "last_usage_ratio": 0.42,
    "estimated_cache_hit_rate": 0.71
  }
}
```

字段约束：

- `runtime_cursor` 只保存继续执行所需的最小状态；
- 不保存完整 Agent State；
- `state_refs` 指向当前物化文件；
- `last_checkpoint` 必须与 `checkpoints/CURRENT` 一致；
- `config_fingerprint` 不包含任何动态信息。

---

## 5. task-state.json Schema

```json
{
  "schema_version": "1.0",
  "task_id": "task-20260815-001",
  "objective": "给 app.py 添加 health check 接口",
  "constraints": [
    "不引入新的第三方依赖",
    "修改后必须运行测试"
  ],
  "success_criteria": [
    "GET /health 返回 200",
    "现有测试全部通过"
  ],
  "progress": {
    "completed": [
      {
        "id": "p-1",
        "text": "阅读 app.py 现有结构",
        "completed_step": 2,
        "evidence_refs": ["artifact:sha256:abc"]
      }
    ],
    "current": "实现 /health 路由",
    "remaining": [
      {
        "id": "plan-12",
        "text": "添加 health check 函数",
        "status": "active",
        "depends_on": [],
        "evidence_refs": []
      }
    ]
  },
  "key_findings": [
    {
      "id": "f-1-1",
      "fact": "app.py 使用 FastAPI 框架",
      "evidence": [
        {
          "type": "artifact",
          "artifact_id": "sha256:abc",
          "range": "1-40"
        }
      ],
      "status": "valid",
      "updated_step": 3
    }
  ],
  "decisions": [
    {
      "id": "d-1-1",
      "decision": "health check 使用 FastAPI 原生路由实现",
      "reason": "项目已使用 FastAPI",
      "evidence": ["f-1-1"],
      "status": "valid"
    }
  ],
  "unresolved": [
    {
      "id": "u-1-1",
      "text": "是否需要 health check 访问数据库",
      "blocked": false
    }
  ],
  "extensions": {}
}
```

### 5.1 PlanStep

| 字段 | 类型 | 说明 |
|---|---|---|
| id | string | 稳定 ID，Agent State 通过 active_item_id 引用 |
| text | string | 单条可执行步骤 |
| status | enum | pending / active / done / blocked / skipped |
| depends_on | string[] | 前置 step id |
| evidence_refs | string[] | artifact_id 或 finding_id |

### 5.2 Finding

| 字段 | 类型 | 说明 |
|---|---|---|
| id | string | 稳定 ID |
| fact | string | 代码库或任务事实 |
| evidence | EvidenceRef[] | 必须引用 artifact、file hash 或 git head |
| status | enum | valid / stale |
| updated_step | integer | 最后更新 step |

### 5.3 EvidenceRef

```json
{
  "type": "artifact | file | git_head",
  "artifact_id": "可选",
  "path": "可选",
  "file_hash": "可选",
  "git_head": "可选",
  "range": "可选"
}
```

### 5.4 Task State 容量约束

- 硬上限：8K tokens；
- 软上限：6K tokens；
- `progress.completed` 只保留仍有价值的完成项；
- 超限时按 stale 优先、低价值 completed 次之的顺序淘汰。

---

## 6. tool-state.json Schema

```json
{
  "schema_version": "1.0",
  "profiles": {
    "grep": {
      "useful_scopes": [
        {
          "id": "g-1-1",
          "path": "src/auth",
          "file_pattern": "*.ts",
          "status": "valid",
          "updated_step": 5
        }
      ],
      "effective_queries": [
        {
          "id": "q-1-1",
          "query": "refreshToken",
          "scope": "src/auth",
          "hit_count": 2,
          "status": "valid",
          "updated_step": 5
        }
      ],
      "known_error_patterns": ["permission denied"]
    },
    "read": {
      "useful_files": [
        {
          "id": "r-1-1",
          "path": "src/auth/token.ts",
          "file_hash": "sha256:...",
          "status": "valid",
          "updated_step": 6
        }
      ],
      "effective_ranges": [
        {
          "id": "rr-1-1",
          "path": "src/auth/token.ts",
          "range": "42-88",
          "note": "refreshToken 实现",
          "file_hash": "sha256:..."
        }
      ],
      "known_symbols": [
        {
          "id": "s-1-1",
          "path": "src/auth/token.ts",
          "symbol": "refreshToken",
          "range": "42-88",
          "file_hash": "sha256:..."
        }
      ]
    },
    "shell": {
      "effective_commands": [
        {
          "id": "c-1-1",
          "command": "pytest tests/test_auth.py",
          "purpose": "auth 模块测试",
          "status": "valid",
          "updated_step": 8
        }
      ],
      "known_failures": [
        {
          "id": "e-1-1",
          "pattern": "ImportError",
          "note": "需要先安装测试依赖",
          "last_seen_step": 7
        }
      ],
      "environment_notes": ["python3 可用", "pytest 已安装"]
    },
    "test": {
      "effective_commands": [],
      "known_failures": []
    }
  },
  "evidence_index": [
    {
      "profile": "grep",
      "entry_id": "q-1-1",
      "artifact_id": "sha256:...",
      "path": "src/auth"
    }
  ]
}
```

### 6.1 Tool State 边界

- 不保存完整 Tool Call 日志；
- 不保存属于 Task State 的代码业务事实；
- `known_symbols` 只作为定位指针，业务含义归 Task State；
- 文件相关条目必须携带 `file_hash` 或 `git_head`；
- 硬上限 8K tokens，软上限 6K tokens。

---

## 7. trajectory.json 与 InteractionGroup Schema

### 7.1 trajectory.json

```json
{
  "schema_version": "1.0",
  "epoch_id": 3,
  "group_count": 6,
  "token_count_estimate": 42000,
  "first_group_id": "g-805",
  "last_group_id": "g-810",
  "groups": []
}
```

### 7.2 InteractionGroup

```json
{
  "group_id": "g-810",
  "epoch_id": 3,
  "created_step": 17,
  "user_input_ref": "user-turn-3",
  "status": "complete",
  "messages": [],
  "tool_calls_summary": {
    "call_ids": ["call_17_1"],
    "tools_used": ["write_file", "run_shell"]
  },
  "tool_results_summary": {
    "success_count": 1,
    "failure_count": 1,
    "artifact_ids": ["sha256:...", "sha256:..."]
  },
  "corrections": [],
  "workspace_changes": [
    {
      "path": "src/app.py",
      "operation": "edit",
      "after_sha256": "sha256:..."
    }
  ],
  "protected": true,
  "protected_reasons": ["recent_modification"],
  "token_count": 3200,
  "source_events": {
    "first_seq": 808,
    "last_seq": 810
  }
}
```

### 7.3 UnifiedMessage

```json
{
  "role": "user | assistant | tool",
  "content": "文本内容，可为 null",
  "tool_calls": [
    {
      "id": "call_17_1",
      "name": "write_file",
      "arguments": {}
    }
  ],
  "tool_call_id": "call_17_1",
  "name": "write_file",
  "is_error": false
}
```

约束：

- assistant tool_calls 消息与其全部 tool 结果必须在同一 group；
- 批量调用中未执行的调用必须生成合成 tool 结果；
- `protected_reasons` 只能是固定枚举；
- `token_count` 由 TokenCounter 估算，仅供 Fold 选择使用。

---

## 8. Agent State Schema

Agent State 是请求级临时对象，不单独落盘。每次构建时生成。

```json
{
  "schema_version": "1.0",
  "position": {
    "user_turn": 3,
    "step": 17,
    "epoch": 3,
    "phase": "verification",
    "status": "running",
    "attempt": 0
  },
  "focus": {
    "task_id": "task-20260815-001",
    "current_goal_ref": "plan-12",
    "current_goal": "添加 health check 函数",
    "success_criteria": ["GET /health 返回 200"],
    "next_action": "运行 pytest 验证"
  },
  "todo": {
    "active": {"id": "t-12", "text": "添加 health check", "status": "active"},
    "next": [],
    "blocked": []
  },
  "workspace": {
    "cwd": "/abs/path",
    "platform": "linux",
    "shell": "bash"
  },
  "git": {
    "branch": "main",
    "head": "9f3a...",
    "dirty": true,
    "staged_files": [],
    "modified_files": ["src/app.py"],
    "untracked_files": [],
    "conflicted_files": []
  },
  "working_set": {
    "focused_files": ["src/app.py"],
    "recently_inspected": ["src/app.py"],
    "recently_modified": ["src/app.py"]
  },
  "tool_execution": {
    "last_call": "call_17_1",
    "pending_calls": [],
    "retry_count": 0,
    "last_error": null
  },
  "verification": {
    "required": true,
    "status": "running",
    "last_command": "pytest tests/test_health.py",
    "failing_tests": [],
    "artifact_id": "sha256:..."
  },
  "limits": {
    "steps_used": 17,
    "max_steps": 30,
    "tool_calls_used": 23,
    "tool_calls_remaining": 57,
    "subagents_used": 1
  },
  "context": {
    "epoch": 3,
    "window_limit_tokens": 256000,
    "prompt_tokens": 120000,
    "usage_ratio": 0.47,
    "pressure": "normal",
    "fold_count": 2,
    "prefix_hash": "sha256:...",
    "tools_hash": "sha256:..."
  },
  "planner": {
    "enabled": true,
    "plan_version": 2,
    "active_item_id": "plan-12",
    "replan_required": false
  },
  "coordination": {
    "depth": 0,
    "active_subagents": []
  },
  "recovery": {
    "blocked": false,
    "checkpoint_id": "sha256:...",
    "resumable": true,
    "resume_action": "continue_next_action"
  }
}
```

### 8.1 Core / Extended 划分

| 层级 | 字段 |
|---|---|
| Core 始终出现 | position、focus.current_goal、todo.active、workspace.cwd、git.branch/head/dirty、verification.required/status、limits、context.pressure、planner、recovery |
| Extended 按需出现 | workspace.platform/shell、git 完整列表、working_set、tool_execution 详情、focus.success_criteria、todo.next/blocked |

### 8.2 大小约束

- 目标 1K–2K tokens；
- 列表 Top-N + `total_count`；
- 每条字符串限制最大长度；
- Core 优先级高于 Extended。

---

## 9. Checkpoint Manifest Schema

文件：`checkpoints/ckpt-<seq>.json`

```json
{
  "schema_version": "1.0",
  "checkpoint_seq": 42,
  "checkpoint_id": "sha256:canonical-manifest",
  "parent_checkpoint_seq": 41,
  "checkpoint_kind": "periodic",
  "created_at": "2026-08-15T12:04:58Z",
  "epoch_id": 3,
  "log_anchor": {
    "last_event_seq": 812,
    "last_event_offset": 153221,
    "last_event_hash": "sha256:...",
    "log_file": "events.jsonl"
  },
  "state_refs": {
    "task_state": {"sha256": "sha256:...", "size": 8231},
    "tool_state": {"sha256": "sha256:...", "size": 5210},
    "trajectory": {"sha256": "sha256:...", "size": 24000}
  },
  "runtime_cursor": {},
  "workspace_expected": {},
  "config_fingerprint": {},
  "artifact_anchor": {
    "last_artifact_index_seq": 210,
    "artifact_index_hash": "sha256:..."
  },
  "recovery": {
    "resumable": true,
    "resume_action": "continue_next_action",
    "blocked_reason": null,
    "last_good_checkpoint_seq": 41,
    "replan_attempts": 0
  }
}
```

`checkpoint_kind` 枚举：

```text
baseline
task_boundary
periodic
post_fold
recovery_transition
blocked
terminal
```

`runtime_cursor` 复用 session.json 中的结构；`pending_calls` 必须为空。

---

## 10. Event Log Schema

### 10.1 行格式

每行一个事件：

```json
{
  "seq": 812,
  "ts": "2026-08-15T12:04:58Z",
  "type": "file_change",
  "prev_hash": "sha256:event-811",
  "payload": {},
  "payload_hash": "sha256:canonical-payload"
}
```

约束：

- `seq` 单调递增；
- `prev_hash` 形成 hash chain；
- `payload_hash` 校验 payload 完整性；
- 追加后 fsync；
- 半行或 hash 不连续视为日志损坏。

### 10.2 事件类型

| type | 用途 |
|---|---|
| session_start | Session 创建与 baseline |
| epoch_start | 新 Epoch 开始 |
| task_boundary | 新任务 / 任务切换 |
| assistant_message | 模型回复进入轨迹 |
| tool_batch_intent | 批量工具执行前 WAL |
| tool_result | 单个工具结果 |
| tool_batch_closed | 批量工具组关闭 |
| file_change | Agent 文件修改 |
| shell_workspace_observation | Shell 前后 workspace 变化 |
| validation | 校验结果 |
| fold_start | Fold 开始 |
| state_delta_applied | State Delta 合并完成 |
| trajectory_folded | 旧轨迹删除完成 |
| checkpoint_created | Checkpoint 提交 |
| drift_detected | 漂移检测结果 |
| recovery_decision | 恢复决策 |
| session_end | 会话结束 |

### 10.3 关键事件 payload

#### tool_batch_intent

```json
{
  "group_id": "g-813",
  "step": 18,
  "calls": [
    {
      "call_id": "call_18_1",
      "name": "write_file",
      "arguments": {"path": "src/app.py", "content": "..."}
    }
  ]
}
```

#### tool_result

```json
{
  "group_id": "g-813",
  "call_id": "call_18_1",
  "name": "write_file",
  "success": true,
  "error": null,
  "output_preview": "Wrote 4321 characters to src/app.py",
  "artifact_id": "sha256:...",
  "duration_ms": 12,
  "retries_used": 0
}
```

#### file_change

```json
{
  "group_id": "g-813",
  "tool_call_id": "call_18_1",
  "path": "src/app.py",
  "operation": "edit",
  "before_sha256": "sha256:...",
  "after_sha256": "sha256:...",
  "size": 4321,
  "status": "M"
}
```

#### shell_workspace_observation

```json
{
  "group_id": "g-810",
  "tool_call_id": "call_17_2",
  "command": "pytest tests/test_health.py",
  "exit_code": 0,
  "pre_fingerprint": "sha256:...",
  "post_fingerprint": "sha256:...",
  "observed_changes": [
    {
      "path": ".pytest_cache/v/cache/nodeids",
      "status": "??",
      "sha256": "sha256:..."
    }
  ]
}
```

#### state_delta_applied

```json
{
  "epoch_from": 2,
  "epoch_to": 3,
  "fold_id": "fold-2",
  "task_delta": {},
  "tool_delta": {},
  "state_refs_after": {}
}
```

#### drift_detected

```json
{
  "checkpoint_seq": 42,
  "severity": "HIGH",
  "unexpected_changes": [],
  "missing_expected_changes": [],
  "hash_mismatches": [],
  "git_divergence": null
}
```

#### recovery_decision

```json
{
  "decision": "RESUME | REPLAN | BLOCKED",
  "reason": "HIGH drift on src/app.py",
  "drift_event_seq": 850
}
```

---

## 11. WorkspaceExpected Schema

```json
{
  "schema_version": "1.0",
  "workspace_root": "/abs/path",
  "git": {
    "present": true,
    "branch": "main",
    "head": "9f3a...",
    "status_hash": "sha256:porcelain-v2"
  },
  "expected_dirty": [
    {
      "path": "src/app.py",
      "status": "M",
      "sha256": "sha256:...",
      "source_event_seq": 810
    }
  ],
  "expected_untracked": [
    {
      "path": "out/report.txt",
      "sha256": "sha256:...",
      "source_event_seq": 805
    }
  ],
  "postconditions": [
    {
      "path": "src/app.py",
      "expected_status": "M",
      "sha256": "sha256:...",
      "tool_call_id": "call_17_1"
    }
  ],
  "workspace_fingerprint": "sha256:canonical-set"
}
```

更新规则：

- `file_change` 事件更新 `expected_dirty` 和 `postconditions`；
- `shell_workspace_observation` 的 observed_changes 更新 expected 集合；
- 同一路径以最后事件为准；
- Checkpoint 创建前必须重新计算 Actual 并与 Expected 比较。

---

## 12. State Delta Schema

```json
{
  "schema_version": "1.0",
  "fold_id": "fold-2",
  "epoch_from": 2,
  "epoch_to": 3,
  "task_delta": {
    "set": {
      "progress.current": "实现 /health 路由"
    },
    "upsert": [
      {
        "target": "key_findings",
        "id": "f-2-1",
        "value": {}
      }
    ],
    "append": [
      {
        "target": "progress.remaining",
        "item": {},
        "dedupe_key": "id"
      }
    ],
    "remove": [
      {"target": "progress.completed", "id": "p-1"}
    ],
    "mark_stale": [
      {
        "target": "key_findings",
        "id": "f-1-1",
        "reason": "file changed externally"
      }
    ]
  },
  "tool_delta": {
    "set": {},
    "upsert": [],
    "append": [],
    "remove": [],
    "mark_stale": []
  }
}
```

### 12.1 允许的 target

| target | 对象 |
|---|---|
| progress.completed | ProgressItem |
| progress.remaining | PlanStep |
| key_findings | Finding |
| decisions | Decision |
| unresolved | UnresolvedItem |
| profiles.grep.useful_scopes | ScopeEntry |
| profiles.grep.effective_queries | QueryEntry |
| profiles.read.useful_files | FileEntry |
| profiles.read.effective_ranges | RangeEntry |
| profiles.read.known_symbols | SymbolEntry |
| profiles.shell.effective_commands | CommandEntry |
| profiles.shell.known_failures | FailureEntry |
| profiles.test.effective_commands | CommandEntry |
| profiles.test.known_failures | FailureEntry |

### 12.2 合并顺序

```text
1. Schema 校验
2. 引用校验（artifact_id 必须存在）
3. 执行 set
4. 执行 remove
5. 执行 mark_stale
6. 执行 upsert
7. 执行 append 并去重
8. 执行容量约束
9. 一致性校验
```

规则：

- `remove` / `mark_stale` 指向不存在的 ID：忽略并写 Trace；
- `upsert` 无 ID 时由 Runtime 生成稳定 ID；
- `append` 根据 dedupe_key 去重；
- 合并结果必须仍满足 Task/Tool State 容量上限。

---

## 13. Artifact Index Schema

`artifact-index.jsonl` 每行：

```json
{
  "seq": 210,
  "artifact_id": "sha256:raw-bytes",
  "tool": "read",
  "tool_call_id": "call_16_1",
  "arguments": {
    "path": "src/app.py",
    "start_line": 1,
    "end_line": 120
  },
  "created_at": "2026-08-15T12:04:00Z",
  "size": 4821,
  "sha256": "sha256:raw-bytes",
  "encoding": "utf-8",
  "path": "artifacts/read/sha256:raw-bytes.artifact",
  "sensitive_filter_applied": false
}
```

约束：

- Artifact 内容不可变；
- `artifact_id = sha256(raw content)`；
- `artifact-index.jsonl` append-only；
- `read_artifact` 只允许读取当前 session 的 Artifact；
- `write_file` Artifact 默认不开放给模型。

---

## 14. Subagent Report Schema

```json
{
  "schema_version": "1.0",
  "status": "completed",
  "findings": [
    {
      "id": "sub-f-1",
      "fact": "token refresh 逻辑位于 src/auth/token.ts:42-88",
      "evidence": [
        {"type": "artifact", "artifact_id": "sha256:..."}
      ]
    }
  ],
  "evidence": [],
  "blockers": [],
  "next_action": "建议主 Agent 修改 token.ts 后运行 auth 测试",
  "modified_files": [
    {
      "path": "src/auth/token.ts",
      "after_sha256": "sha256:..."
    }
  ]
}
```

约束：

- 总量 ≤ 4K tokens；
- `modified_files` 必须上抛父 Event Log；
- 不自动合并子 Task State。

---

## 15. Metrics / Trace Schema

### 15.1 llm_request

```json
{
  "ts": "2026-08-15T12:04:58Z",
  "type": "llm_request",
  "step": 17,
  "layer_tokens": {
    "stable_system": 1200,
    "tools": 1800,
    "task_tool_state": 5200,
    "recent_trajectory": 41000,
    "agent_state": 1500,
    "protocol_overhead": 800
  },
  "total_estimate": 51500,
  "usage_ratio": 0.20,
  "prefix_hash": "sha256:...",
  "tools_hash": "sha256:...",
  "provider_usage": {
    "input_tokens": 52300,
    "cache_read_input_tokens": 48200,
    "cache_creation_input_tokens": 0,
    "output_tokens": 320
  }
}
```

### 15.2 fold_event

```json
{
  "type": "fold_event",
  "fold_id": "fold-2",
  "trigger_usage_ratio": 0.71,
  "after_usage_ratio": 0.48,
  "folded_group_ids": ["g-1", "g-2"],
  "retained_group_ids": ["g-3", "g-4"],
  "task_state_tokens_before": 8200,
  "task_state_tokens_after": 7600,
  "tool_state_tokens_before": 4800,
  "tool_state_tokens_after": 5100,
  "model_calls": 1,
  "retries": 0,
  "fallback_used": false
}
```

### 15.3 session_metrics

```json
{
  "total_steps": 30,
  "total_tool_calls": 41,
  "fold_count": 2,
  "total_llm_calls": 30,
  "estimated_cache_hit_rate": 0.68,
  "last_usage_ratio": 0.42
}
```

---

## 16. 实现模块拆解

### 16.1 模块清单

| 模块 | 职责 | 关键输入 | 关键输出 | 依赖 |
|---|---|---|---|---|
| SchemaRegistry | 所有 JSON Schema 定义与校验 | schema_version | validation result | 无 |
| ConfigFingerprint | 生成/校验 prefix_hash、tools_hash、config fingerprint | system text、tool schemas、SDK 版本 | config_fingerprint | ProviderAdapter |
| EventLog | 追加、fsync、hash chain、按 seq 读取 | 事件 | events.jsonl | 无 |
| ArtifactStore | Raw Result 落盘、索引、读取 | tool result | artifact_id、index | EventLog |
| WorkspaceFingerprint | 扫描 git/文件状态，生成 Actual fingerprint | workspace path、scope | WorkspaceActual | 无 |
| ExpectedWorkspace | 维护 expected dirty/untracked/postconditions | file_change、shell observation | WorkspaceExpected | EventLog |
| TokenCounter | 估算各层 token，校准系数 | PromptPlan、usage | token estimates | ProviderAdapter |
| ContextBuilder | 组装 system/tools/messages/Agent State | states、trajectory、agent state | PromptPlan | SchemaRegistry、AgentStateRebuilder |
| ProviderAdapter | 原生 tools、cache_control、消息转换、usage 解析 | PromptPlan | provider request/response | TokenCounter |
| AgentStateRebuilder | 每轮从 Runtime/git/事件重建 Agent State | runtime cursor、workspace、状态 | AgentState | WorkspaceFingerprint |
| InteractionGroupManager | 组创建、追加、关闭、保护标记 | assistant/tool 事件 | trajectory.json | EventLog |
| ToolObservationPipeline | 2K 判断、Artifact 落盘、工具压缩 | raw tool result | ToolObservation | ArtifactStore、TokenCounter |
| ToolCompressors | 各工具确定性压缩、可选 LLM 语义压缩 | raw result | compressed observation | TokenCounter |
| FoldEngine | 70% 检测、组选择、调用 Fold 模型、Fallback | trajectory、states | deltas、新 epoch | ContextBuilder、StateDeltaMerger |
| StateDeltaMerger | Delta 校验、合并、容量控制 | old states、deltas | new states | SchemaRegistry |
| CheckpointManager | 安全切点判断、快照、manifest、原子提交 | 全部状态 | checkpoint | EventLog、WorkspaceFingerprint |
| DriftDetector | Resume 时 Actual vs Expected | checkpoint、events、workspace | drift report | EventLog、WorkspaceFingerprint |
| RecoveryEngine | RESUME / REPLAN / BLOCKED 决策 | drift report、policy | recovery action | CheckpointManager、Planner |
| SessionManager | session 生命周期、锁、目录、CLI 路径 | config | session | CheckpointManager |
| MigrationService | 旧 session 一次性转换 | 旧 .sessions | 新 session 目录 | SchemaRegistry |
| MetricsEmitter | llm_request、fold_event、session_metrics | 各模块事件 | traces | 无 |

### 16.2 模块分层

```text
API / CLI
  │
  ├── SessionManager ── CheckpointManager ── RecoveryEngine
  │
  ├── AgentLoop
  │     ├── ContextBuilder ── ProviderAdapter ── LLM
  │     ├── TokenCounter
  │     ├── AgentStateRebuilder
  │     └── InteractionGroupManager
  │
  ├── ToolExecutor
  │     ├── ArtifactStore
  │     ├── ToolObservationPipeline
  │     └── ToolCompressors
  │
  ├── FoldEngine
  │     ├── StateDeltaMerger
  │     └── State Capacity Control
  │
  └── Storage
        ├── EventLog
        ├── WorkspaceFingerprint / ExpectedWorkspace
        ├── SchemaRegistry
        └── ArtifactStore
```

---

## 17. 核心流程拆解

### 17.1 每轮正常执行

```text
1. 检查预算与安全切点
2. AgentStateRebuilder 重建 Agent State
3. ContextBuilder 组装 PromptPlan
4. TokenCounter 估算 total_prompt_tokens
5. 若 usage >= 70%：进入 Fold 流程
6. ProviderAdapter 发送请求
7. InteractionGroupManager 追加 assistant_message
8. 若无 tool calls：
     validation / final answer
     关闭 Interaction Group
     触发 Checkpoint
9. 若有 tool calls：
     写 tool_batch_intent 事件
     逐个执行工具：
       Raw Result 落盘 Artifact
       ToolObservationPipeline 压缩
       写 tool_result 事件
     写 tool_batch_closed 事件
     关闭 Interaction Group
     触发 Checkpoint
```

### 17.2 Trajectory Fold 流程

```text
1. 计算可折叠组：
     从最老完整组开始
     跳过 protected 组
     直到“保护窗口 + 剩余轨迹 ≤ 50%”
2. 构造 Fold 输入：
     旧 Task State
     旧 Tool State
     选中的 Interaction Groups
3. 调用 Fold 模型生成 State Delta
4. 失败：重试一次
5. 仍失败：使用确定性规则 Fallback
6. StateDeltaMerger 校验并合并
7. 容量控制
8. 删除已折叠组
9. 写 state_delta_applied、trajectory_folded 事件
10. 开启新 Epoch
11. 重新计数
12. 若仍超过模型硬限制：停止并进入可恢复 BLOCKED
13. 创建 post_fold Checkpoint
```

### 17.3 Checkpoint 提交流程

```text
1. 判断是否满足安全切点
2. 计算 Actual workspace fingerprint
3. 比较 Actual 与 Expected
4. 不一致：
     写 drift_detected 事件
     进入 RecoveryEngine
     不创建常规 checkpoint
5. 一致：
     写 checkpoint 相关事件并 fsync
     写 state/trajectory 快照文件
     写 checkpoint manifest
     atomic rename manifest
     fsync 目录
     更新 CURRENT
```

### 17.4 Resume 流程

```text
1. 获取 session 锁
2. 读取 CURRENT 指向的 checkpoint
3. 校验 manifest、快照、事件链
4. 回放 last_event_seq 之后的事件
5. 若有未关闭 tool batch：
     进入 WAL 工具恢复流程
6. 计算 Actual workspace fingerprint
7. DriftDetector 生成 drift report
8. RecoveryEngine 决策：
     NONE / IGNORED / LOW → RESUME
     HIGH → REPLAN
     STRUCTURAL → BLOCKED
9. 写 recovery_decision 事件
10. 创建 recovery_transition checkpoint
```

---

## 18. 实现里程碑

| 里程碑 | 内容 | 退出标准 |
|---|---|---|
| M0 | Schema 与配置基座 | 所有 schema_version=1.0 文件可校验 |
| M1 | EventLog + ArtifactStore + WorkspaceFingerprint | 事件可 append/回放；Artifact 可落盘/读取；workspace 可扫描 |
| M2 | TokenCounter + ContextBuilder + ProviderAdapter | 三层缓存断点正确；usage 校准工作 |
| M3 | ToolObservationPipeline + 各工具压缩器 | 2K 阈值和分工具硬上限生效 |
| M4 | InteractionGroupManager + FoldEngine + StateDeltaMerger | 70/30/50 规则可执行；Fold 不破坏协议组 |
| M5 | CheckpointManager + DriftDetector + RecoveryEngine | 崩溃恢复、漂移恢复状态机可用 |
| M6 | SessionManager + MigrationService + MetricsEmitter | 旧 session 可迁移；指标可解释 |

建议严格按 M0→M6 顺序实施，M5 依赖 M1–M4 的完整事件语义。

---

## 19. 详细 Schema 阶段仍待确认项

1. 每个 Tool Compressor 的最终输出字段与示例；
2. State 淘汰评分的精确权重；
3. `read_artifact` 对二进制 Artifact 的行为；
4. Fold 模型输出 Delta 的 JSON Schema 严格程度；
5. Event Log payload 中 arguments 是否完整保存，还是大参数只存 Artifact 引用；
6. 事件日志的 compaction / retention 策略；
7. `--agent-dir` 与 `--session-dir` / `--trace-dir` 的最终优先级细节；
8. 旧 session 转换时 `plan` 到 `progress.remaining` 的映射规则。
