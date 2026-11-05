# Lecture 8 · Agent Design V：Cost-Effective Agents

> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/8_agent_design_5.pdf`](../slides/8_agent_design_5.pdf)

## 本讲要点

- **成本公式**：Cost = Users × Sessions/User × Turns/Session × Steps/Turn × Tokens/Step × Price/Token
- 逐项降低：turns per task、steps per turn、tokens per step（input/output）、price per token
- 每种节省都有 counter-cost；要设置 budget 和停止条件；先问“真的需要 LLM 吗”
- **让 agent 更快**：完成时间由 critical path 决定；parallelism、orchestration、speculative actions
- **Harness 架构**：Monolithic / Modular / Microkernel；八个真实 harness 的独特设计

---

## 1. 成本从哪里来？

一个 session = 多个 user turns；一个 turn = 多个 LLM + tool steps。

```
Cost = Users × Sessions/User × Turns/Session × Steps/Turn × Tokens/Step × Price/Token
        └─ 增长（adoption、engagement）─┘  └──── 缩小 agent trajectories ────┘  └ 缩小单价 ┘
```

（本讲只关注 LLM 成本，忽略其他计算成本。）

## 2. 减少 Turns per Task

- 实际使用数据（一门课学生的 Claude Code 日志）：68%–75% 的 session 只有一个 prompt（好习惯：每个任务开新 session）；少数“马拉松” session 超过 100 turns
- **第 k 个 turn 会重读 1…k−1 的所有内容，总成本随 turn 数平方增长**
- 用户回来的原因：没有计划、过早停止、没有检查
- 做法：
  - **Front-load planning**：行动前一次性确认目标和计划
  - **Structured task decomposition**：拆成 subtasks，每个有 done check
  - **Self-verification before handoff**：交还前跑测试并检查输出
- 目标：自主完成，不需要用户中途确认

## 3. 减少 Steps per Turn

- 大多数 turn 步数很少（effort 设为 medium/high）；但有的 turn 接近 1,000 步
- 每次 model call 都会重读整个 context，所以 **call 越少越便宜**
- 做法：
  - **Batch independent calls**：一步读 5 个文件，而不是 5 步
  - **Reuse plans and skills**：重放成功过的做法，而不是重新推导
  - **Load context up front**：开始时就读需要的文件
- 相关论文（Advanced）：Skill libraries（Voyager 2023、SkillRL 2026）、经验提炼（ExpeL 2023、ReasoningBank 2025）、Plan 模板（Test-Time Plan Caching 2025）、持久记忆（Mem0、LightMem、SimpleMem）。**只有任务会重复时复用才划算**，memory 维护和过时条目可能抵消节省

## 4. 减少 Tokens per Step

### 实际使用数据（Claude Code 日志）
- Main agent 每步 tokens：中位数 200K–300K，P90 600K–700K；Subagent 中位数约 80K，P90 约 200K
- Context 累积很快，第 ~500 步时约 650K–670K
- **97.6% 的 token 是 cache read**；output 比 fresh prefill（cache write）少 5 倍；fresh prefill 比 cache read 少 100 倍
- 典型价格比：**cache read : input : output = 1 : 10 : 50**

### 减少 Input Tokens：保持 context 精简

| 策略 | 做法 |
|---|---|
| Drop what's stale | 截断过期结果、mask 已用过的 tool outputs、compact 历史、把旧 turns 总结成 state |
| Shrink what enters | 压缩 observations（保留错误而不是完整日志）、offload 到文件（保存大输出，传路径） |
| Load only what's needed | 选择性检索（只要相关 tools 和 memory）、隔离探索（subagent 返回 summary） |

> 留在 context 里的 token，在之后每一步都要再付一次钱。

相关论文：ACON（peak context 减少 26–54%）、Active Context Compression（自我压缩，少 23% tokens）、TRACE（summary 会导致返工）、LLMLingua 系列（最多 20 倍压缩）、MemGPT（context 作为快速内存）、Scroll（无损外部日志）、Code execution with MCP（用代码过滤结果，150K → 2K tokens）、Tool Search（tool tokens 少 85%）、Programmatic Tool Calling（少 37%）。

### 减少 Output Tokens：要得更少

- **Prompt-level constraints**：“≤100 words”、“只返回 JSON”、“不要复述问题”、“不要解释推理”
- **Output-token caps**：`max_tokens=300`；简单但可能截断在句子中间；配合 concise prompt 使用；注意 reasoning 可能耗尽上限
- **Structured output**：只要必要字段的 schema（如 `{"action": "search", "argument": "KV cache"}`），限制 enum 和可选字段；对 tool-using agent 来说 tool call 周围的文字纯属成本
- **Adaptive reasoning effort**：lookup 用 `effort="low"`，debug 用 `effort="high"`；按调用/步骤设定
- 相关论文：Concise CoT、CCoT、Chain of Draft、TALE、Ares（每步选 effort）、s1、L1；Let Me Speak Freely?（严格格式可能损害推理）、XGrammar、JSONSchemaBench；AdaEdit（diff 或整文件）、CodeStruct（AST 级编辑）、Tool-making agents（output tokens 少 58%）

## 5. 降低 Price per Token

### Caching 与 Batching（Claude API 相对 base input 的价格）

| 类型 | 相对价格 |
|---|---|
| Output token | 5× |
| Base input token | 1× |
| Batch input | 0.5× |
| Cached input | 0.1× |
| Batch + cached input | 0.05× |

- **保持稳定的 prefix**：system prompt、tools、早期历史
- **能等的就 batch**：evals、backfills、批量任务

### Model Routing

| 方式 | 做法 | 实现 |
|---|---|---|
| Choose up front | 生成前为每个 turn 选模型 | 用偏好数据训练的 router；也可选 agent 角色（RouteLLM、MasRouter） |
| Escalate on failure | 先用便宜模型，检查失败再升级 | learned scorer、self-verification、execution feedback（FrugalGPT、AutoMix、EcoAssistant） |
| Re-route each step | 随 trajectory 增长重新选模型 | 按历史和预算路由；orchestrator 把模型当 tools 调用（ToolOrchestra、BAAR、MTRouter） |

难点：**一个便宜模型的错误规划可能浪费很多后续动作**。

### 后续课程的系统技术

- Model optimization：MoE / sparsity、quantization、pruning、distillation、针对 workload 的 fine-tuning
- Serving engine：speculative decoding、KV-cache 优化、更好的请求调度（vLLM PagedAttention、SGLang RadixAttention、Prompt Cache）
- Hardware：更快的 kernels、kernel scheduling、专用加速器

## 6. 每种节省都有代价

| 节省 | 代价 |
|---|---|
| Compaction | 被丢弃的文件要重读；summary 会重写 cache |
| Routing | 便宜模型的一个坏计划会增加很多步或导致失败 |
| Subagents | 每个 subagent 重读共享文件和 instructions（重复 context） |
| Speculation | 猜错浪费 tokens，并争用同样的 tools |

**Net saving = gross saving − counter-cost**，要在同一批任务上测量两者。

### Budget 与停止条件

- 每任务上限（示例）：$10、80 次 model calls、15 分钟 tool time、每步 2 次重试
- 何时停止：verifier 通过；N 步内没有新通过的测试；同样的错误重复；达到上限时带 summary 交还
- 根据 traced runs 设置上限，例如成功任务 P95 的 2 倍
- **带 summary 停止的任务比失控的任务便宜**

### 真的需要 LLM 吗？

| 情况 | 做法 |
|---|---|
| 每次步骤都一样 | 写脚本（让 LLM 写一次代码，之后直接跑） |
| 输入格式固定 | 用规则解析（regex、parser、模板，每次调用零成本） |
| 窄的标签或打分 | 训练小 classifier（如 GBM，gradient boosting machine） |
| 固定流程、几乎不需推理 | 把 workflow 脚本化，只在需要的步骤调用 LLM |

---

## 7. 让 Agent 更快

更快通常也更便宜，但不总是。手段：加速 LLM serving、改进 orchestration、加速 tool calls、并行化、speculative actions。

### Completion time 是 critical path，而不是所有步骤之和

- 串行：时间 ≈ 所有步骤之和（Queue、Setup、Prefill、Reasoning、Tools、Orchestration）
- 并行：时间 ≈ critical path（Plan → Worker A/B → Merge）
- **更便宜可能更慢**：小模型可能需要更多步骤
- **并行更贵**：等待更少，但总 tokens 更多

### 改进 Orchestration

- **Async execution**：tool call 运行时就开始下一步
- **Persistent sessions**：在多步之间保持 shell 或 browser 打开
- **Stream and pipeline**：LLM 回复结束前就开始 tool call
- **Colocate services**：tools 与模型在同一主机

### 加速 Tool Calls

Build、test、页面加载可能比 model call 还慢：缓存 tool 结果；重计算放在服务器上；更快、更便宜的 tools、有选择地调用；对卡住的调用设 timeout（fail fast）。**先测每步的 tool 时间再调模型。**

### 并行化

- Parallel tool calls：一步中发出独立调用
- Parallel sub-agents：拆分独立 subtasks，合并 summaries
- 只有独立的工作能并行；共享状态需要 merge 步骤

### Speculative Actions

1. **Predict**：便宜模型猜下一个 action
2. **Run early**：主模型思考时就执行猜测
3. **Compare**：与主模型的选择对比 —— 命中则直接用结果，未命中则丢弃（代价是一次浪费的调用）

适用：猜测常常正确且 action 可撤销。开放问题：在有限 GPU/CPU/API 预算下如何调度 speculative calls，使节省的 critical-path 时间超过浪费。

**推测什么**（范围从一次调用扩展到整个 trajectory）：Tool intent（SPORK）→ Tool call 含参数（PASTE、Speculative Tool Calls）→ 任意 action 或 API response（Speculative Actions）→ 多条分支（B-PASTE）→ 已执行的多步链（Speculative Macro Commit）

**预测从哪里来**：Model-based（小 draft 模型预测）；Self-speculative（同一模型的早期 fork 预测自己的 tool call，SPORK）；History-based（过去的 traces 预测下一个 tool，参数绑定到上一个结果，PASTE）。Lossless 方法总是先验证；lossy 方法（SPAgent）跳过部分检查。

### 循环优化

Run & Profile（记录每次调用、token、tool、等待）→ Measure（理解成本和延迟来源）→ Change one thing（compaction、routing、caching、parallelism）→ Re-evaluate（同样任务上的成功率、花费、P95 latency）→ Repeat。每次只改一个变量，才能知道收益来自哪里。

---

## 8. Agent Architecture

### 三种 Harness 架构

| | Monolithic | Modular | Microkernel |
|---|---|---|---|
| 设计原则 | 全部集成在一个 runtime | 接口抽象，一个 controller | 最小核心 + 外部服务 |
| 中心 controller | 有 | 有 | 有，但更小 |
| 替换组件 | 困难 | 容易 | 容易（协议匹配时） |
| 执行隔离 | 可选 | 可选 | 架构重点 |
| 部署 | 通常一起 | 一起或分开 | 通常独立 |
| 独立扩展 | 有限 | 可能 | 强 |
| 通信开销 | 低 | 低–中 | 可能更高 |
| 典型场景 | 独立 agent | Agent frameworks | Multi-agent 基础设施 |

- **Monolithic**：`CodingAgent` 自己持有 Memory、Tools、ContextManager、Planner、Executor。✅ 简单、易 debug、开销低、状态易共享；❌ 难替换、紧耦合、难跨 agent 共享、难隔离
- **Modular**：通过接口注入 provider（`RedisMemory()`、`MCPTools()`、`SlidingWindowContext()`、`DockerSandbox()`；Docker → Firecracker 只换一个 provider）。✅ 易替换、易测试、职责清晰、可接第三方；❌ controller 仍拥有生命周期、协调逻辑变复杂、故障域可能共享
- **Microkernel**：小 runtime 只负责 state、coordination、scheduling、security，通过标准协议调用 Memory / Tools / Execution 服务（可在不同 container 或机器）。✅ 独立扩展、故障隔离、共享基础设施；❌ 通信开销高、失败模式更多、端到端 debug 难、需要标准协议

例：运行一段 Python —— Monolithic 本地直接执行（hop 最少，隔离难）；Modular 调 `SandboxProvider.run(code)`；Microkernel 把请求发给 execution service，由它负责分配、隔离和 sandbox 生命周期。

### 八个 Harness 的独特设计

| Harness | 定位 | 独特之处 |
|---|---|---|
| **Claude Code**（Anthropic，TS，闭源） | Policy layer + typed hooks | — |
| **Codex CLI**（OpenAI，Rust） | Sandbox first, ask second | 命令在无网络的 sandbox 中运行，只有要出 sandbox 才请求批准；用 Lark grammar 定义 patch 格式，四轮匹配 hunks |
| **DeepSeek Harness**（TS） | 一切皆 plugin，连 loop 也是 | loop、adapters、tools、log、approvals 都是 plugin，可用 YAML 层替换；每个 request 是 session log 的冻结视图，harness 只追加 log；**compare-and-swap edits**：模型读后文件变了则编辑失败 |
| **OpenCode**（SST，TS/Bun） | Harness 是一个 server | TUI、SDK、Web app 共用同一个 OpenAPI HTTP server；九种 replacer 从 exact 到 anchor 匹配，拒绝过大的 fuzzy 匹配；每步用 shadow git repo 做 snapshot，`/undo` 能恢复文件 |
| **Pi**（TS） | 最小核心 | 只有 4 个 tools，没有 MCP、subagents、approvals，其余由 extensions 提供；只在预计节省 ≥ $0.05 时发送 keep-alive 做 cache warming；pi-ai 支持 42 个 provider |
| **oh-my-pi**（TS + Rust） | Pi 的 fork，内置 IDE 能力 | **Hashline edits**：用 4 位 hex 文件标签 + 行号编辑，模型无需重打旧文本；一个 shell 贯穿所有调用；原生 search、AST、LSP、debugger；snapcompact 把旧历史转成图片；TTSR 在 regex 匹配时停止重试 |
| **OpenClaw**（TS） | 面向聊天渠道的 gateway | 27 个聊天渠道，每个 session 一条 queue lane；compaction 前先静默写带日期的笔记，recall 结合全文检索和 vector search；fences 隔离 untrusted 文本，网络读取会 taint 该 turn，监控 secret sentinels |
| **Hermes Agent**（Nous Research，Python） | 自己写 skills 的 agent | 后台 fork 检查每个 turn 并编辑 memory 和 skills；**在 50% window 时就开始 compaction**（八个中最早）；任何副作用前先把 tool-call 写入 session 数据库 |
