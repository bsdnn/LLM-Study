# Lecture 6 · Agent Design III：Memory and Reliability

> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/6_agent_design_3.pdf`](../slides/6_agent_design_3.pdf)

## 本讲要点

- **Agent memory**：超出 context window 的持久状态；设计是一系列 policy 决策（write → encode → store → retrieve → reflect → forget）
- Memory 的 failure modes 与评估方法（LoCoMo、LongMemEval、BEAM）
- **Agent reliability**：错误会复利累积；按来源分为 specification / harness / model / tool / environment 五类 failure

---

## 1. Agent Memory 概览

- **Context window**：临时 working state
- **Memory store**：跨交互的持久状态
- 结构：User ↔ Agent Harness ↔ LLM，Harness ↔ Memory Store

### 六个设计问题

| 阶段 | 问题 |
|---|---|
| WRITE OR SKIP（decision） | 值得保存吗？ |
| ENCODE / ORGANIZE（format） | 应该存成什么形式？ |
| STORE（backend） | 存在哪里？ |
| RETRIEVE（task） | 这个任务需要它吗？ |
| REFLECT / GENERALIZE（lesson） | 能得出什么更广的结论？ |
| KEEP / UPDATE / DELETE（status） | 它还有效吗？ |

外加一个问题：**谁拥有这份 memory？**

### 1.1 什么时候写入 Memory？

写入每次交互 → recall 高，但噪声大。写入策略（从高 recall 到高 precision）：
- 用户显式要求
- 确定性规则（heuristics）
- 小型 classifier
- 定期的 LLM review

### 1.2 如何组织 Memory？

| 形式 | 例子 | 特点 |
|---|---|---|
| Raw events / episodes | “We moved serving from vLLM to SGLang.” | 高保真，但大且冗余 |
| Unstructured summaries | “Uses SGLang for model serving.” | 紧凑，但可能丢 context |
| Structured state | `{"serving_engine": "SGLang"}` | 可查询，schema 有约束 |
| Knowledge graph | Project → USES → SGLang | 实体关联，可表达随时间变化 |

### 1.3 存在哪里？

| Backend | 例子 | 擅长 |
|---|---|---|
| Plain files | CLAUDE.md、scratchpad | 易检查、diff、搜索 |
| Key-value / relational DB | Redis、SQLite、Postgres JSONB | 精确查找、过滤、transactions |
| Vector store | pgvector、Chroma、Pinecone | 精确或近似 similarity search |
| Graph database | Neo4j、FalkorDB（Graphiti） | 关系查询、multi-hop traversal |

### 1.4 何时使用 Memory？

| 方式 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| Automatic | 每个 turn：search → LLM → add | 可预测、简单 | 增加 latency 和无关 context |
| Runtime-controlled | `if memory_needed: search()` | 有选择、高效 | 需要规则或 classifier |
| Model-controlled | LLM 调用 `search_memory()`、`add_memory()`、`update_memory()`、`forget_memory()` | 按需，由 LLM 决定 | 可能漏掉有用的读写 |

很多系统是混合的，读写策略也可以不同。

### 1.5 如何检索？

基础 pipeline：Query → Vector search → Top-k memories → Prompt。
但 semantic relevance 只是一个信号，还要看：
- **Scope**：同一 user、project、entity、task
- **Importance**：遗漏或包含它的代价
- **Validity**：此时此任务是否仍然成立
- **Provenance**：来源、置信度、证据

### 1.6 如何从观察中泛化（Reflection）？

例：用户问过 vLLM、SGLang、TRT-LLM、LLM scheduling → reflection → 假设“经常关注 LLM inference systems”，置信度 medium。
何时 consolidation：积累足够新证据后（每 N 个事件或 context 满时）；事实冲突或重复时；尽量放在后台任务中。

### 1.7 Forgetting 是一等操作

- 为什么忘：信息过时、错误或冲突、隐私与安全、存储和检索成本
- 生命周期控制：标记失效、永久删除、TTL 过期、分层并从 active memory 中 evict

### 1.8 Valid time 与 Provenance

- **Valid time**：哪个事实在哪段时间成立（2026 年 4 月用 vLLM —— superseded；6 月起用 SGLang —— current）
- **Provenance**：事实从哪里来（source: Episode 28）

### 1.9 谁拥有 Memory？

| | Agent-owned | Dedicated memory system |
|---|---|---|
| 形式 | harness 中的文件：CLAUDE.md、scratchpad、task notes | agent 调用的服务：Mem0、Graphiti、supermemory |
| 特点 | harness 把 memory 持久化到磁盘 | Memory 是一个 API，可在 agents 和 sessions 之间共享 |

**专门的 memory 系统**：
- **Mem0**（fact-centric）：输入按 user/agent/run 划分的 messages；vector search + filters，可选 graph memory
- **Graphiti**（temporal knowledge graph）：从 episodes 提取实体和带时间的事实；hybrid semantic + keyword + graph search
- **supermemory**（profile + memory + document）：维护 user profile，输入关联来源和时间的对话

### 1.10 设计 Checklist

Observe → **Select**（值得记吗）→ **Encode / organize**（什么表示能保留意义）→ **Store**（文件还是数据库）→ **Retrieve**（当前任务需要什么）→ **Reflect / consolidate**（是否更新或与旧记忆矛盾）→ **Forget**（何时过时）。每一步都是 policy 决策。

### 1.11 Failure Modes

| Failure | 描述 |
|---|---|
| Retrieval miss | 相关 memory 存在但没找到 |
| Context pollution | 太多无关 memory 进入 context |
| Memory hallucination | 推断被当成事实存储（“在 Harvard 学习” → “Harvard 的访问学生”） |
| Staleness | 旧事实覆盖了新信息 |
| Isolation failure | Memory 跨越了用户边界 |
| Overgeneralization | 一次事件变成了持久偏好 |
| Persistent prompt injection | untrusted 内容之后以 instruction 的形式回来 |
| Unbounded growth | 大小随生命周期增长，检索成本和质量下降 |

### 1.12 如何评估 Memory

- **Component 评估**：Retrieval（precision、recall）；Writes & update（precision、recall、stale-memory rate）；Governance（deletion compliance、scope isolation）；System（latency、token overhead、store growth）
- **Benchmarks**：LoCoMo（10 段长对话，1,540 题，ACL 2024）、LongMemEval（500 题，5 种记忆能力，ICLR 2025）、BEAM（2,000 题，最多 10M tokens，ICLR 2026）。主要测试 recall、temporal reasoning、updates、preferences、abstention

### 1.13 Memory Stack

- **独立 memory layer**：Agent runtime（Claude Code、Letta、Codex）→ Memory management（Mem0、Letta built-in、custom）→ Storage（vector、SQL、graph、object stores）
- **Harness 内部 memory**：memory management 在 runtime 内部，storage 是 markdown 文件

---

## 2. Agent Reliability

### 错误会复利累积

每步成功率 95%，步骤独立时：
- 5 步：77%；20 步：36%；**50 步：8%（约 1/13）**
- 一次运行只有每一步都成功才算成功；而 error propagation 让实际情况更糟

### 失败从哪里来？

User / task spec（specification failure）→ Agent harness（harness failure）→ LLM（model failure）→ Tool layer（tool failure）→ Environment（environment failure）

### 2.1 Task Specification Failure

| Failure | spec 说的 | agent 做的 | 修复 |
|---|---|---|---|
| Ambiguous goal | “Fix the login bug” | 修了三个中错误的那个 | 写明文件、症状、复现步骤 |
| Proxy objective | “Make the tests pass” | 删掉了失败的测试 | 禁止改测试；review diff |
| Unstated constraint | “Speed up search” | 破坏了 public API | 写明什么不能改 |
| No definition of done | “Clean up the parser” | 永远在重构 | 给出退出测试和停止规则 |
| Conflicting goals | “Minimal diff, every edge case” | 悄悄丢掉了 edge cases | 显式排序目标 |

### 2.2 Agent Harness Failure

| Failure | harness 做的 | 后果 | 修复 |
|---|---|---|---|
| Context truncation | 截掉最早的 turns | 忘了“保留 API”规则 | 关键规则固定在 window 外 |
| Memory error | 把过时笔记当事实 | 改了已经移走的文件 | 给 memory 加日期；使用前重新检查 |
| State corruption | 保留了半应用的 edit | 后续步骤建立在坏文件上 | Checkpoint；回滚失败步骤 |
| Control loop error | 没有 step 或 cost 预算 | 200 步没有进展 | 限制步数、成本、时间 |
| Blocking fan-out | 等待每个 subagent | 一个慢的子任务拖住整体 | 设置 timeout；使用部分结果 |

### 2.3 Model Failure

| Failure | 模型推理 | agent 做的 | 修复 |
|---|---|---|---|
| Hallucination | “`requests.fetch()` 存在” | 调用不存在的函数 | 对照真实文档检查调用 |
| Instruction violation | “速度比 API 重要” | 改了被要求保留的签名 | 重申约束；检查 diff |
| Faulty reasoning | “timeout 说明网络挂了” | 给 deadlock 加了重试 | 修复前先验证假设 |
| Reasoning–action mismatch | “我只改 parser.py” | 改了另外三个文件 | 每个 action 对照 plan |
| Unverified success | “测试应该过了” | 什么都没跑就宣布完成 | 要求测试输出作为证明 |
| Long-horizon degradation | “config 还需要改” | 撤销了自己第 12 步的修复 | 重新读状态；拆成短任务 |

### 2.4 Tool Failure

| Failure | tool 返回 | agent 做的 | 修复 |
|---|---|---|---|
| Truncated output | log 的前 100 行 | 漏掉第 4,000 行的错误 | 先返回错误；支持分页 |
| Silent failure | exit 0，空结果 | 以为没有匹配 | 区分 “empty” 和 “failed” |
| Ambiguous schema | 没有时区的日期 | 任务偏了 8 小时 | 使用 typed、显式的 schema |
| Flaky call | 第一次超时 | 宣布服务挂了 | 带 backoff 重试后再判断 |
| Misleading error | 拼写错误导致 “Permission denied” | 用 sudo 重试 | 返回精确、可操作的错误 |

### 2.5 Environment Failure

| Failure | 环境 | agent 做的 | 修复 |
|---|---|---|---|
| Stale state | 昨天的 cached build | 在旧代码上通过测试 | 从干净 cache 构建 |
| Version mismatch | Python 3.8 而非 3.12 | 交付了会崩溃的代码 | 用 lockfile 固定版本 |
| Random failure | 成功与否取决于环境 | 因为闰年失败 | 重跑失败；隔离 flaky tests |
| 网页布局变化 | checkout 页面重新设计 | 点错按钮 | 行动前重新读页面 |
| Rate limit | 50 次调用后 HTTP 429 | 迁移中途停止 | 限流并为长任务做 checkpoint |

### 2.6 为失败而设计

| 来源 | 原则 | 做法 |
|---|---|---|
| Task specification | 把 “done” 写成测试 | 预先写 acceptance tests；列出不能改的东西 |
| Agent harness | 限制每次运行 | step、cost、time 上限；checkpoint 与回滚 |
| Model | 要求证据 | 每个结论都要证明；verify；用更强的模型 |
| Tool | 大声失败 | typed schemas；错误与空结果区分开 |
| Environment | 固定并隔离 | 新鲜、固定版本的 sandbox；判断前重试 flaky |

> 我们对大规模 agent 的理解还很有限，几乎没有正式的 best practice。
