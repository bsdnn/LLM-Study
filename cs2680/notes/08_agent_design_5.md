# Lecture 8 · Agent Design V：Cost-Effective Agents 与 Harness 架构

> Harvard CS 2680：Modern AI Systems: Agents and System Optimizations（Juncheng Yang，Fall 2026）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/8_agent_design_5.pdf`](../slides/8_agent_design_5.pdf)
> 预计阅读时间：18–22 分钟。预备知识：Lecture 2（prefix cache、compaction）、Lecture 4（context 管理）、Lecture 7（成本与延迟指标）。

## 本讲脉络

```
成本从哪里来：Users × Sessions × Turns × Steps × Tokens × Price
→ 逐项压缩：减少 turns（为什么成本随 turn 数平方增长）→ 减少 steps（批量调用、复用计划）
  → 减少每步 tokens（输入：丢旧的、缩小进来的、只加载需要的；输出：要求更少、结构化、按需调整推理强度）
  → 降低单价（cache、batch、模型路由）
→ 每种节省都有反作用 → 预算与停止条件 → 真的需要 LLM 吗？
→ 让 agent 更快：关键路径、编排、工具加速、并行、speculative actions
→ 在循环中优化：测量 → 改一处 → 重新评估
→ Harness 架构：monolithic、modular、microkernel → 八个真实 harness 的独特设计
```

**一句话总结**：Agent 的成本 = 用户数 × 每用户 session 数 × 每 session turn 数 × 每 turn step 数 × 每 step token 数 × 每 token 单价。前两项是业务增长，后四项都可以压缩。实测数据显示**超过 97% 的 token 是 cache read** —— 因为每一步都要重读整个 context。所以省钱的关键是**少调用、让 context 保持精简、让 cache 命中**。但每种节省都有反作用，必须在同一批任务上测量净收益。速度方面，**完成时间由关键路径决定**，而不是所有步骤之和。

（本讲只关注 LLM 的成本，忽略其他计算成本。）

---

## 1. 成本从哪里来

**回顾 session 的结构**：一个 session 包含多个用户 turn；一个 turn = 多个 LLM 调用和工具调用（step），最后回复用户。

```
Session
├─ Turn 1：User message → Model call → Tool call → Model call → … → Reply to user
├─ Turn 2：……
└─ Turn 3：……
```

**成本公式**：

$$
\text{Cost} = \underbrace{\text{Users} \times \frac{\text{Sessions}}{\text{User}}}_{\uparrow\ \text{增长：用户与参与度}} \times \underbrace{\frac{\text{Turns}}{\text{Session}} \times \frac{\text{Steps}}{\text{Turn}} \times \frac{\text{Tokens}}{\text{Step}}}_{\downarrow\ \text{缩小：agent 的轨迹}} \times \underbrace{\frac{\text{Price}}{\text{Token}}}_{\downarrow\ \text{缩小：单价}}
$$

前两项是业务希望增长的；后四项是工程师要压缩的。下面逐项讨论。

---

## 2. 减少 Turns per Task（每个任务的用户 turn 数）

### 2.1 实际使用数据

一门课学生的 Claude Code 日志显示：
- **大多数 session 只有一个 prompt**（作业 1、作业 2 分别是 68% 和 75%）—— 好习惯：每个任务开一个新 session
- 少数“马拉松” session 超过 100 个 turn

### 2.2 为什么成本随 turn 数平方增长

第 $k$ 个 turn 要**重读第 1 到 $k-1$ 个 turn 的全部内容**。假设每个 turn 往 context 中增加 $c$ 个 token，$T$ 个 turn 总共读取的 token 数约为

$$
\sum_{k=1}^{T} k \cdot c = c \cdot \frac{T(T+1)}{2} \propto T^2
$$

**算例**：每个 turn 增加 5K token。
- 10 个 turn：总共读取约 $5\text{K} \times 55 = 275$K token
- 20 个 turn：约 $5\text{K} \times 210 = 1.05$M token —— turn 数翻倍，**成本约变成 4 倍**

### 2.3 怎么减少用户 turn

**用户为什么要回来再说一次？** 因为 agent 没有计划、过早停止、或者没有检查自己的工作。

| 方法 | 做法 |
|---|---|
| **Front-load planning（预先规划）** | 行动之前，**一次性**确认目标和计划（有歧义就在开始时问清楚） |
| **Structured task decomposition（结构化分解）** | 拆成子任务，**每个子任务都有完成检查** |
| **Self-verification before handoff（交付前自检）** | 交还给用户之前，先跑测试、检查输出 |

**目标**：agent 自主完成任务，不需要用户中途确认。

---

## 3. 减少 Steps per Turn（每个 turn 的步数）

### 3.1 实际使用数据

- 大多数 turn 只需要很少的步数（因为 effort 设置为 medium 或 high）
- 但也有 turn 跑了接近 1,000 步

### 3.2 为什么少调用就省钱

**每一次模型调用都要重读整个 context**。所以同样的工作，调用次数越少越便宜。

```
一次调用做一个动作（4 次模型调用）：
Model → Tool → Model → Tool → Model → Tool → Model

先计划，再批量执行（2 次模型调用）：
Model → [Tool, Tool, Tool 并行] → Model
```

| 方法 | 做法 |
|---|---|
| **Batch independent calls（批量独立调用）** | 一步读 5 个文件，而不是分 5 步各读一个 |
| **Reuse plans and skills（复用计划和技能）** | 重放之前成功过的做法，而不是每次重新推导 |
| **Load context up front（预先加载 context）** | 开始时就读入需要的文件，而不是每步读一个 |

### 3.3 进阶：复用计划、技能和经验（相关论文）

| 方向 | 论文 |
|---|---|
| **Skill libraries（技能库）** | Voyager（2023）：把成功的做法存成可执行的技能；SkillRL（2026）：同时改进技能库和 agent 策略 |
| **从经验中提炼 insight** | ExpeL（2023）：从经验中提取可复用的 insight；ReasoningBank（2025）：同时保留成功和失败运行中的策略 |
| **Playbook 与计划模板** | Test-Time Plan Caching（2025）：复用并改编计划模板 |
| **持久的对话记忆** | Mem0（2025）：存储并更新事实；LightMem（2025）：离线整合；SimpleMem（2026）：压缩记忆 |

**只有任务会重复出现时，复用才划算**；维护记忆的成本和过时的条目可能会抵消节省。

---

## 4. 减少 Tokens per Step（每一步的 token 数）

### 4.1 实际使用数据

| 指标 | 数值 |
|---|---|
| 主 agent 每步 token 的中位数 | 200K–300K（作业 1：208K，作业 2：301K） |
| 主 agent 每步 token 的 P90 | 600K–700K |
| Subagent 每步 token 的中位数 | 约 80K（P90 约 200K） |
| 第 ~500 步时的 context 大小 | 约 650K–670K |
| **Cache read 占全部 token 的比例** | **97.6%** |

- Output 比 fresh prefill（cache write）小约 5 倍
- Fresh prefill 比 cache read 小约 100 倍
- 典型价格比：**cache read : input : output = 1 : 10 : 50**

**算例：一步要花多少钱**（假设 input $3 / 百万 token，cache read $0.3 / 百万，output $15 / 百万）。一步的 context 为 300K，其中 295K 命中 cache，5K 是新输入，输出 1K：

$$
295\text{K} \times 0.3 + 5\text{K} \times 3 + 1\text{K} \times 15 \;(\text{单位：美元 / 百万 token}) \approx \$0.089 + \$0.015 + \$0.015 = \$0.12
$$

- **cache read 占这一步成本的约 75%**
- 如果 cache 全部失效（按全价读 300K）：$0.90 一步，贵 7.5 倍
- 100 步下来：命中 cache 约 $12，不命中约 $90

结论：context 越长，每一步都越贵；**context 中的 token 会在之后的每一步被重复付费**。

### 4.2 减少输入 token：保持 context 精简

| 策略 | 具体做法 |
|---|---|
| **Drop what's stale（丢掉过时的）** | 截断过时的结果；把已经用过的工具输出 mask 掉；compact 历史；把旧的 turn 总结成一份状态 |
| **Shrink what enters（缩小进入的）** | 压缩 observation（**只保留错误，不要完整日志**）；offload 到文件（大输出存成文件，只传路径） |
| **Load only what's needed（只加载需要的）** | 有选择地检索（只给相关的工具和记忆）；隔离探索（subagent 只返回一个 summary） |

> **留在 context 中的 token，会在之后的每一步被再付一次钱。**

**相关论文**：

| 方向 | 论文与结果 |
|---|---|
| 压缩 agent 历史 | ACON（2025）：峰值 context 减少 26–54%；Active Context Compression（2026）：agent 自己压缩，token 少 23%；TRACE（2026）：发现 summary 会导致返工 |
| Prompt 压缩 | LLMLingua（2023）：最多 20 倍压缩；LongLLMLingua、LLMLingua-2（2024）；Prompt Compression in the Wild（2026）：只在长 prompt 上划算 |
| 把 context 当外部内存 | MemGPT（2023）：把 context 当作“快速内存”；Scroll（2026）：无损的外部日志，由代码决定模型看到什么 |
| 让工具数据不进 context | Code execution with MCP（2025）：在代码中过滤工具结果，150K → 2K token；Tool Search（2025）：工具定义的 token 少 85%；Programmatic Tool Calling：少 37% |

### 4.3 减少输出 token：要得更少

**Prompt 层面的约束**：明确说出长度、格式和要跳过什么。

```
Answer in ≤100 words.
Return only JSON.
One sentence. Top 3 items.
Don't restate the problem.
Don't explain your reasoning.
```

也可以在 system prompt 中写：“Be concise; omit background unless requested.”

**输出 token 上限**：在 API 调用中设硬上限，如 `max_tokens=300`。
- 简单，但可能在句子中间截断回答
- 应与“简洁输出”的 prompt 配合使用
- **注意**：对 reasoning model，思考 token 也计入上限，可能还没开始写答案就被用完了

**Structured output（结构化输出）**：要求只包含必要字段的 schema。

```json
{"action": "search", "argument": "KV cache"}
```

- 限制 enum 和可选字段，让模型无法添加多余的内容
- 特别适合使用工具的 agent：**工具调用周围的解释性文字纯属成本**

**Adaptive reasoning effort（自适应推理强度）**：让思考预算与当前步骤匹配。

```
lookup（查找）：effort="low"
debug（调试）：effort="high"
```

- 查找和格式化用低强度；规划和调试用高强度
- 可以按调用或按步骤设置（reasoning effort 或思考 token 预算）

**相关论文**：Concise CoT、CCoT（给 CoT 设字数限制）、Chain of Draft（简短草稿）、TALE（推理 token 预算）；Ares（每步选择 effort）、When to Plan（只在有用时才规划）、s1 与 L1（强制思考预算）；Let Me Speak Freely?（严格格式可能损害推理）、XGrammar、SGLang、JSONSchemaBench；AdaEdit（选择 diff 还是整文件）、CodeStruct（AST 级编辑）、Tool-making agents（制作可复用工具，输出 token 少 58%）。

---

## 5. 降低 Price per Token（单价）

### 5.1 Caching 与 Batching

相对于基础输入价格（Claude API）：

| 类型 | 相对价格 |
|---|---|
| Output token | 5× |
| Base input token | 1× |
| Batch input | 0.5× |
| Cached input | 0.1× |
| **Batch + cached input** | **0.05×** |

**保持前缀稳定**：prefix cache 只在**开头完全相同**时命中。system prompt、工具定义、早期的历史要保持不变。常见的“cache 杀手”：在 system prompt 开头放时间戳、动态改变工具列表的顺序、在历史中间插入内容。

**能等的就用 batch**：评测、数据回填、批量任务等不需要立即返回的工作，用 batch API（价格减半，通常在较长时间窗口内返回结果）。

### 5.2 模型路由（Model Routing）

| 方式 | 做法 | 怎么实现 |
|---|---|---|
| **Choose up front（预先选择）** | 生成之前为每个 turn 选一个模型 | 用偏好数据训练一个 router；也可以同时选择 agent 角色（RouteLLM、MasRouter） |
| **Escalate on failure（失败时升级）** | 先用便宜模型；检查不通过时换更强的 | 检查方式：学习的打分器、自我验证、执行反馈（FrugalGPT 的级联、AutoMix、EcoAssistant） |
| **Re-route each step（每步重新路由）** | 随着 agent 轨迹的推进重新选择模型 | 根据历史和预算路由；由一个 orchestrator 把各个模型当作工具来调用（ToolOrchestra、BAAR、MTRouter） |

**难点**：**一个便宜模型做出的错误规划，可能浪费后面很多步动作。** 省了一次调用的钱，却赔上了几十步。

### 5.3 后续讲次的系统技术

| 模型优化 | Serving 引擎优化 | 硬件 |
|---|---|---|
| MoE / 稀疏化、量化、剪枝、蒸馏、针对负载的微调 | Speculative decoding、KV cache 优化、更好的请求调度 | 更快的 kernel、更好的 kernel 调度、专用加速器 |

相关论文：vLLM（2023，PagedAttention）、SGLang（2024，RadixAttention 前缀缓存）、Prompt Cache（2024，模块化复用）。

---

## 6. 每种节省都有反作用

| 节省手段 | 反作用 |
|---|---|
| **Compaction（压缩）** | 被丢掉的文件后面可能要重新读；生成 summary 会**改写 cache**（之后的前缀变了，cache 失效） |
| **Routing（路由）** | 便宜模型的一个坏计划，可能导致多出很多步，甚至整个尝试失败 |
| **Subagents** | 每个 subagent 都要重新读共享的文件和指令（**context 重复**） |
| **Speculation（投机执行）** | 猜错了就浪费 token，还会与正常执行争用同样的工具 |

$$
\text{净节省} = \text{总节省} - \text{反作用成本}
$$

**必须在同一批任务上同时测量两者。**

### 6.1 给每个任务设预算，进展停滞时停止

| 每任务上限 | 示例 |
|---|---|
| 美元 | $10 |
| 模型调用次数 | 80 次 |
| 工具执行时间 | 15 分钟 |
| 每步重试次数 | 2 次 |

**何时停止**：
- verifier 通过
- 连续 N 步没有新通过的测试
- 同样的错误反复出现
- 触到上限：**带着 summary 交还给用户**

上限的设定方法：根据实际运行的记录（trace）来设，例如**成功任务 P95 的 2 倍**。

> 一个带着 summary 停下来的任务，比一个失控的任务便宜得多。

### 6.2 真的需要 LLM 吗？

| 情况 | 做法 |
|---|---|
| 每次都是同样的步骤？ | **写一个脚本**：让 LLM 写一次代码，以后直接运行 |
| 输入格式固定？ | **用规则解析**：正则、parser、模板，每次调用零成本 |
| 只是一个窄的分类或打分？ | **训练一个小分类器**（如梯度提升树） |
| 流程固定、几乎不需要推理？ | **把流程脚本化**，只在需要推理的那几步调用 LLM |

> 只有当某一步真正需要推理时，才动用 LLM。

---

## 7. 让 Agent 更快

更快通常也更便宜，但并不总是。手段：加速 LLM serving、改进编排、加速工具调用、并行化、speculative actions。

### 7.1 完成时间是关键路径，而不是所有步骤之和

```
串行：时间 ≈ 所有步骤之和
Queue → Setup → Prefill → Reasoning → Tools → Orchestration → …

并行：时间 ≈ 关键路径
Plan → [Worker A, Worker B 同时] → Merge
```

两个反直觉的事实：
- **更便宜可能更慢**：小模型可能需要更多步
- **并行更贵**：等待更少了，但总 token 更多

### 7.2 加速 LLM serving

前面所有省成本的方法也都能降延迟：更少的 turn、更少的步数、更少的 token、更便宜的 serving（cache、更小的模型）。

### 7.3 改进编排（Orchestration）

| 方法 | 例子 |
|---|---|
| **Async execution（异步执行）** | 工具调用运行的同时，就开始下一步 |
| **Persistent sessions（持久会话）** | 在多步之间保持同一个 shell 或浏览器打开（不用每次重新启动） |
| **Stream and pipeline（流式与流水线）** | LLM 回复还没结束，就开始执行它已经给出的工具调用 |
| **Colocate services（服务就近部署）** | 把工具和模型部署在同一台主机上，减少网络往返 |

### 7.4 加速工具调用

**构建、测试、加载网页的时间可能比模型调用还长。**

| 方法 | 说明 |
|---|---|
| **Cache tool results** | 重复的读取和搜索直接命中缓存 |
| **Faster hardware** | 重计算放在服务器上跑，而不是笔记本 |
| **Better tools** | 用更快、更便宜的工具做同样的事；有选择地调用 |
| **Fail fast** | 卡住的调用设超时 |

> **先测量每一步工具花的时间，再去调模型。**

### 7.5 并行化

```
串行：读文件 A → 读文件 B → 读文件 C
并行：读文件 A / 读文件 B / 读文件 C 同时   （同样的工作，等待时间只有三分之一）
```

- **Parallel tool calls**：一步中发出多个独立的调用
- **Parallel sub-agents**：拆分独立的子任务，合并它们的 summary
- **只有独立的工作才能并行**；有共享状态时需要一个合并步骤

### 7.6 Speculative Actions（投机执行）

**三步**：
1. **Predict（预测）**：一个便宜的模型猜测下一个动作
2. **Run early（提前执行）**：在主模型思考的同时，先执行这个猜测
3. **Compare（比较）**：把猜测与主模型的真实选择对比

| 猜对了 | 猜错了 |
|---|---|
| 结果已经准备好，省掉了等待 | 丢弃结果，代价是一次浪费的调用 |

**适用条件**：猜测经常正确，而且动作**可以撤销**（读文件、搜索可以投机执行；付款、删除不行）。

**开放问题**：在有限的 GPU、CPU 和 API 预算下，如何调度投机调用，使节省的关键路径时间超过浪费的工作？

**进阶：投机的对象**（范围从一次调用扩展到整条轨迹）：

| 投机对象 | 说明 | 代表工作 |
|---|---|---|
| Tool intent | 下一个要用哪个工具 | SPORK |
| Tool call | 工具 + 参数 | PASTE、Speculative Tool Calls |
| Next action | 任意动作或 API 响应 | Speculative Actions |
| Branches | 同时投机几条可能的路径 | B-PASTE |
| Trajectory | 一整串多步动作，已经执行好 | Speculative Macro Commit |

**进阶：预测从哪里来**：

| 来源 | 做法 | 代表工作 |
|---|---|---|
| Model-based | 一个小的 draft 模型预测大模型下一步会做什么 | Speculative Actions、Speculative Tool Calls |
| Self-speculative | 同一个模型的早期分支，提前预测它自己之后的工具调用 | SPORK |
| History-based | 从过去的轨迹中预测下一个工具，参数绑定到上一步的结果 | PASTE、Speculate with Memory |

**无损（lossless）**的方法总是先验证再保留工作；**有损（lossy）**的方法（如 SPAgent）会跳过部分检查。

**两个例子**：
- **PASTE**（从历史轨迹预测下一个工具）：一旦 `search(q)` 开始，历史模式表明下一步是 `visit(results[0].url)`，URL 可以从搜索结果中直接填入，于是在模型还在生成时就先去访问页面
- **SPORK**（模型预测自己的工具调用）：模型在推理时往往很早就“想好”了要用什么工具。一个早期的探测分支先发起预测的调用（如 `search(q)`），等主模型真正写出这个调用时，结果已经在等着了

### 7.7 在循环中优化

```
Run and Profile（运行并记录每次调用、token、工具、等待）
→ Measure（理解成本和延迟的来源）
→ Change one thing（只改一处：compaction、路由、缓存或并行）
→ Re-evaluate（在同一批任务上测成功率、花费、P95 延迟）
→ Repeat
```

每次只改一个变量，才知道收益来自哪里。（作业 4 会用到这个方法。）

---

## 8. Agent Harness 的架构

### 8.1 三种架构

**1）Monolithic harness（单体式）：所有东西都在一个 runtime 里**

```python
class CodingAgent:
    def __init__(self):
        self.memory   = Memory()
        self.tools    = Tools()
        self.context  = ContextManager()
        self.planner  = Planner()
        self.executor = Executor()

    def run(self, task):
        while not self.finished:
            a = self.planner.plan(task)
            r = self.executor.execute(a)
            self.context.update(r)
```

| 优点 | 缺点 |
|---|---|
| 实现和调试简单 | 难以替换某个组件 |
| 通信开销最小 | 组件之间紧密耦合 |
| 内部状态容易共享 | 难以在多个 agent 之间共享基础设施 |
| 执行开销低 | 难以强制隔离 |

关键特征：**agent 拥有一切，都在一个 runtime 中**。

**2）Modular harness（模块化）：agent 仍控制循环，但通过接口访问各组件**

```python
agent = CodingAgent(
    memory_provider=RedisMemory(),
    tool_provider=MCPTools(),
    context_provider=SlidingWindowContext(),
    sandbox_provider=DockerSandbox())
# Docker → Firecracker：只需替换一个 provider
```

| 优点 | 缺点 |
|---|---|
| 组件容易替换 | controller 仍然掌控整个生命周期 |
| 更好测试、更好扩展 | 协调逻辑变得复杂 |
| 职责清晰分离 | 故障域可能仍然共享（一个组件崩溃拖垮整体） |
| 第三方实现可以直接接入 | 要联合调优时仍需修改 controller |

关键特征：**一个 controller 通过接口委派工作**。

**3）Agent microkernel（微内核）：小 runtime 负责协调，外部服务提供能力**

```
Agent runtime（状态 · 协调 · 调度 · 安全）
        ↕ 标准化协议
Memory service   Tools service   Execution service
     ↕                ↕                 ↕
  Database         Tool APIs         Sandboxes
（各服务可以运行在不同的容器或机器上）
```

| 优点 | 缺点 |
|---|---|
| 各服务独立扩展 | 通信开销更高 |
| 更好的故障隔离 | 需要处理更多的失败模式 |
| 更强的隔离（如果真正落实） | 端到端调试更难 |
| 共享的、集中管理的基础设施 | 需要标准的协议和生命周期 |

关键特征：**runtime 只负责协调；外部服务提供能力**。（名字来自操作系统的微内核设计：内核只保留最核心的功能，其余放在用户态服务中。）

**例子：运行一段 Python 代码**

| 架构 | 过程 |
|---|---|
| Monolithic | Agent 直接在本地执行 Python。跳数最少，但隔离很难保证 |
| Modular | Agent 调用 `SandboxProvider.run(code)`。可以替换 provider 而不改 agent |
| Microkernel | Agent runtime 向 execution service 发送请求 → 服务分配 sandbox → 执行 → 返回结果。**由 execution service 负责分配、隔离和 sandbox 的生命周期** |

**对比**：

| 维度 | Monolithic | Modular | Microkernel |
|---|---|---|---|
| 设计原则 | 全部集成 | 接口抽象 | 最小核心 + 服务 |
| 中心 controller | 有 | 有 | 有，但更小 |
| 替换组件 | 困难 | 容易 | 容易（协议匹配即可） |
| 执行隔离 | 可选 | 可选 | 架构的重点 |
| 组件部署 | 通常在一起 | 一起或分开 | 通常独立 |
| 独立扩展 | 有限 | 可能 | 强 |
| 通信开销 | 低 | 低–中 | 可能较高 |
| 典型场景 | 独立的 agent | Agent 框架 | 多 agent 基础设施 |

**选择**：独立的 agent 用 monolithic，多 agent 基础设施用 microkernel。责任一步步从 agent 内部移到外面：**从代码，到接口，再到服务**。

### 8.2 八个真实 harness 的独特设计

| Harness | 开发者 · 语言 | 一句话定位 |
|---|---|---|
| Claude Code | Anthropic · TypeScript（闭源） | 在 Claude 外面包一层策略层和有类型的 hooks |
| Codex CLI | OpenAI · Rust | Sandbox 优先，需要时才询问 |
| DeepSeek Harness | DeepSeek AI · TypeScript | 一切皆插件，连循环本身也是 |
| OpenCode | Anomaly (SST) · TypeScript, Bun | 一个 session 服务器 + 多个瘦客户端 |
| Pi | Mario Zechner / Earendil · TypeScript | 一个可以用 TypeScript 扩展的小核心 |
| oh-my-pi | Can Bölük / Stencil Labs · TypeScript + Rust | Pi 的 fork，内置原生工具和 IDE 能力 |
| OpenClaw | OpenClaw Foundation · TypeScript | 面向聊天渠道的助手网关 |
| Hermes Agent | Nous Research · Python | 会自己写 skills 的 agent |

**Codex CLI**
- **先 sandbox，再审批**：命令在一个**没有网络访问**的 sandbox 中运行；只有需要走出 sandbox 时，agent 才请求批准。（对比 Claude Code 默认是逐个询问权限）
- **遵守语法的 patch**：用一个 Lark 语法定义 patch 的格式，模型输出的 patch 必须符合这个语法；应用 patch 时分四轮匹配 hunk（从精确匹配逐步放宽）

**DeepSeek Harness**
- **循环本身也是插件**：循环、适配器、工具、日志、审批全都是插件，可以通过 YAML 配置层替换
- **请求来自日志**：每个发给模型的请求，都是 session 日志的一个冻结视图；harness 只往日志里**追加**条目（便于重放和调试）
- **Compare-and-swap 编辑**：如果模型上次读取之后文件被改过，编辑就会失败（防止基于过时内容覆盖别人的修改，类似并发编程中的 CAS 操作）

**OpenCode**
- **Harness 是一个服务器**：TUI、SDK 和 Web 应用都调用同一个 OpenAPI HTTP 服务器
- **有大小限制的模糊编辑**：九种替换器从精确匹配逐步放宽到锚点匹配；如果模糊匹配的范围太大，harness 会拒绝（防止误改大片代码）
- **每一步一个快照**：用一个影子 Git 仓库记录每一步，`/undo` 能恢复**文件**，而不只是恢复聊天记录

**Pi**
- **最小核心**：核心只有四个工具，没有 MCP、没有 sub-agent、没有审批；所有其他功能都由扩展提供
- **带价格检查的 cache 预热**：prompt cache 有过期时间，一段时间不用就失效。Pi 会发送 keep-alive 请求来保持 cache 有效，但**只在预计节省 ≥ $0.05 时才发**（否则保活本身就不划算）
- **可被其他项目复用的库**：pi-ai 支持 10 种 wire API 上的 42 个 provider；有三个其他项目在使用 Pi 的代码

**oh-my-pi**
- **Hashline 编辑**：每次编辑给出一个 4 位十六进制的文件标签和行号，模型**不需要再输入一遍旧文本**（省输出 token，也避免锚点匹配失败）
- **IDE 就在进程里**：所有调用共用一个一直开着的 shell；原生工具提供搜索、AST、LSP 和调试器
- **特殊的 context 方法**：snapcompact 把旧的历史转换成**图片**（用视觉 token 压缩）；TTSR 在某个正则表达式匹配时停下并重试

**OpenClaw**
- **带队列通道的聊天网关**：27 个聊天渠道的消息被送入每个 session 的一条通道；队列模式决定运行期间新来的消息怎么处理
- **Compaction 之前先写记忆**：先进行一次“静默”的 turn，把带日期的笔记写下来，再做压缩；回忆时同时使用全文搜索和向量搜索
- **Fences、taint 和 sentinels**：用 fences（围栏标记）把不可信文本隔离开；一旦这个 turn 读取了网络内容，就把它标记为 tainted（被污染，之后的敏感操作要更谨慎）；agent 还会监控 secret sentinels（预先埋下的“诱饵秘密”，一旦它们出现在输出中，就说明发生了泄露）

**Hermes Agent**
- **一个会学习的 fork**：一个后台分支检查每个被提示过的 turn，然后修改 memory 和 skills —— agent 在使用中不断改进自己
- **最早开始 compaction**：context 用到 50% 就开始压缩，是八个 harness 中最早的
- **执行前先保存**：任何有副作用的操作执行之前，先把工具调用写入 session 数据库（类似数据库的 write-ahead log，崩溃后可以知道哪些操作已经发出）

---

## 9. 要点回顾

1. 成本 = 用户 × session × turn × step × token × 单价；后四项可以压缩
2. Turn 数让成本平方增长（每个 turn 重读之前的一切）：预先规划、结构化分解、交付前自检
3. 每次模型调用都重读整个 context：批量独立调用、复用计划、预先加载
4. 97.6% 的 token 是 cache read；context 中的每个 token 在之后每步都要再付钱：丢旧的、缩小进来的、只加载需要的；输出用约束、上限、结构化和自适应推理强度控制
5. 单价：保持前缀稳定以命中 cache（0.1×），能等的用 batch（0.5×），模型路由要小心便宜模型的坏计划
6. 每种节省都有反作用，在同一批任务上测量净节省；设预算与停止条件；不需要推理的步骤不用 LLM
7. 完成时间由关键路径决定；更便宜可能更慢，并行更贵；异步、持久会话、流水线、就近部署、工具缓存、投机执行
8. Harness 架构从 monolithic 到 modular 再到 microkernel，责任从代码移到接口再移到服务；真实 harness 在 sandbox、编辑方式、cache、记忆、安全上各有独特设计

---

## 10. 自测题

<details>
<summary>1. 每个 turn 往 context 增加 4K token，30 个 turn 的 session 总共读取多少 token？变成 60 个 turn 呢？</summary>

30 个 turn：4K × 30 × 31 / 2 = 1.86M；60 个 turn：4K × 60 × 61 / 2 = 7.32M，约为 4 倍。成本随 turn 数平方增长。
</details>

<details>
<summary>2. 一步的 context 是 200K，全部命中 cache，cache read 价格 $0.3/M，input $3/M。cache 命中和完全不命中时这一步的输入成本各是多少？</summary>

命中：200K × 0.3 / 1M = $0.06；不命中：200K × 3 / 1M = $0.60，贵 10 倍。
</details>

<details>
<summary>3. 列举两个会让 prefix cache 失效的常见做法。</summary>

在 system prompt 开头放当前时间戳；动态改变工具列表或顺序；在历史中间插入或修改内容。任何改变前缀的操作都会让之后的 cache 失效。
</details>

<details>
<summary>4. 为什么说 compaction 也有反作用？</summary>

被摘要掉的文件和细节之后可能要重新读取；生成的 summary 改变了 context 的前缀，导致之后的 prefix cache 失效。净节省要扣除这些成本。
</details>

<details>
<summary>5. Speculative action 适合用在哪些动作上？为什么？</summary>

适合可以撤销、没有副作用的动作，如读文件、搜索、访问网页；猜错时直接丢弃结果即可。不适合付款、删除、发邮件等不可逆操作。
</details>

<details>
<summary>6. Monolithic 和 microkernel 架构各适合什么场景？</summary>

Monolithic：独立的单个 agent，追求简单、低开销、易调试。Microkernel：多 agent 基础设施，需要独立扩展、故障隔离、共享集中管理的服务（如统一的 sandbox 服务），能接受更高的通信开销和调试难度。
</details>

<details>
<summary>7. DeepSeek Harness 的 compare-and-swap 编辑解决了什么问题？</summary>

如果模型读取文件之后，文件又被别人（或另一个 agent）修改了，基于旧内容的编辑可能覆盖新改动。CAS 编辑在文件已变化时直接失败，迫使模型重新读取最新内容后再改。
</details>
