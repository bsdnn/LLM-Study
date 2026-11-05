# Lecture 6 · Agent Design III：Memory and Reliability

> Harvard CS 2680：Modern AI Systems: Agents and System Optimizations（Juncheng Yang，Fall 2026）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/6_agent_design_3.pdf`](../slides/6_agent_design_3.pdf)
> 预计阅读时间：18–22 分钟。预备知识：Lecture 2（CLAUDE.md 与 auto memory）、Lecture 4（context 管理）、Lecture 5（multi-agent）。

## 本讲脉络

```
Agent memory：超出 context window 的持久状态
→ 七个设计决策：写不写 → 存成什么形式 → 存在哪 → 何时读写 → 怎么检索 → 怎么归纳 → 何时遗忘
→ 有效时间与来源（valid time、provenance）→ 谁拥有 memory（harness 内的文件 vs 独立的 memory 服务：Mem0、Graphiti、supermemory）
→ 设计 checklist → 八种失败模式 → 如何评估（指标与 benchmark）→ memory stack
Agent reliability：为什么错误会复利累积（0.95⁵⁰ ≈ 8%）
→ 失败从哪里来：任务描述、harness、模型、工具、环境 —— 每层的典型失败与修复
→ 为失败而设计的五条原则
```

**一句话总结**：Memory 的每一步都是一个**策略决策**：写入太多会带来噪声，太少会遗忘；检索时语义相似只是一个信号，还要看范围、重要性、是否仍然有效和来源；**遗忘和写入一样重要**。Reliability 的核心事实是：**每步 95% 的成功率，50 步之后只剩 8%**。失败可能出在任务描述、harness、模型、工具或环境的任何一层，要分层定位，并在每一层主动为失败做设计。

---

## 1. Agent Memory 概览

### 1.1 定义：超出 context window 的持久状态

| | Context window | Memory store |
|---|---|---|
| 是什么 | **临时的**工作状态 | **跨交互持久存在**的状态 |
| 生命周期 | 随 session 结束（或 compaction）而消失 | 一直保留，直到被更新或删除 |

架构：

```
User ↔ Agent Harness ↔ LLM
              ↕
        Memory Store
```

Memory 由 **harness** 管理：harness 决定什么时候从 memory store 读出内容放进 context，什么时候把 context 中的信息写回 memory store。LLM 本身仍然是无状态的。

### 1.2 七个设计问题

| 阶段 | 问题 | 决策 |
|---|---|---|
| **WRITE OR SKIP**（决定） | 值得保存吗？ | 写入策略 |
| **ENCODE / ORGANIZE**（格式） | 应该存成什么形式？ | 表示方式 |
| **STORE**（后端） | 存在哪里？ | 存储系统 |
| **RETRIEVE**（任务） | 这个任务需要它吗？ | 检索策略 |
| **REFLECT / GENERALIZE**（经验） | 能从中归纳出什么更广的结论？ | 反思与归纳 |
| **KEEP / UPDATE / DELETE**（状态） | 它还有效吗？ | 生命周期管理 |

外加一个贯穿始终的问题：**谁拥有这份 memory？**

---

## 2. 七个设计决策

### 2.1 什么时候应该把一条信息变成 memory？

**核心权衡**：**写入每一次交互** → recall 高（什么都记得），但**噪声**大（检索时一堆无关内容）；**有选择地写入** → precision 高，但可能漏掉有用的信息。

写入策略从“高 recall”到“高 precision”排列：

| 策略 | 做法 | 优点 | 缺点 |
|---|---|---|---|
| **用户显式要求** | 用户说“记住我不吃辣” | 最准确，用户完全掌控 | 用户不会想到每件值得记的事 |
| **确定性规则（heuristics）** | 例如用户表达偏好（“我更喜欢……”）或纠正 agent 时自动记录 | 可预测、便宜 | 规则覆盖不全 |
| **小型分类器** | 用一个便宜的小模型判断“这句话值不值得记” | 比规则灵活，成本可控 | 需要训练数据 |
| **定期 LLM 审查** | 每隔一段时间让 LLM 批量回顾对话，提炼值得保存的内容 | 能抓住隐含的信息，可以合并整理 | 成本高、有延迟 |

### 2.2 怎么组织 memory？

以同一个事实为例：项目把模型推理服务从 vLLM 迁移到了 SGLang。

| 表示方式 | 例子 | 优点 | 缺点 |
|---|---|---|---|
| **Raw events / episodes（原始事件）** | “We moved serving from vLLM to SGLang.” | 高保真，什么都不丢 | 体积大，冗余多 |
| **Unstructured summaries（非结构化摘要）** | “Uses SGLang for model serving.” | 紧凑 | 可能丢失背景（为什么迁移？什么时候？） |
| **Structured state（结构化状态）** | `{"serving_engine": "SGLang"}` | 可以精确查询 | 需要预先设计 schema，schema 限制了能记什么 |
| **Knowledge graph（知识图谱）** | Project → USES → SGLang | 实体之间相互关联，能表达随时间的变化 | 构建和维护复杂 |

实践中常常组合使用：保留原始事件作为“证据”，同时提炼出摘要或结构化事实用于检索。

### 2.3 存在哪里？

| 后端 | 例子 | 擅长 |
|---|---|---|
| **Plain files（纯文件）** | CLAUDE.md、scratchpad | 简单：人可以直接打开查看、diff、搜索、用 Git 管理 |
| **Key-value 或关系数据库** | Redis、SQLite、Postgres JSONB | 精确查找、过滤、事务 |
| **Vector store（向量库）** | pgvector、Chroma、Pinecone | 精确或近似的语义相似度搜索 |
| **Graph database（图数据库）** | Neo4j、FalkorDB（Graphiti 使用） | 关系查询、多跳遍历（“我的项目用的推理引擎是谁维护的？”） |

### 2.4 什么时候使用 memory？

| 方式 | 伪代码 | 优点 | 缺点 |
|---|---|---|---|
| **Automatic（自动）** | 每个 turn：`search → LLM → add` | 可预测、简单 | 每次都检索，增加延迟，还可能带进无关内容 |
| **Runtime-controlled（由 runtime 控制）** | `if memory_needed: search()`；`if memorable: add()` | 有选择、高效 | 需要规则或分类器来判断“需不需要” |
| **Model-controlled（由模型控制）** | 把 `search_memory()`、`add_memory()`、`update_memory()`、`forget_memory()` 作为工具交给 LLM | 按需使用，由 LLM 决定何时调用 | 模型可能忘了查、忘了记 |

很多系统采用**混合方式**，而且**读和写的策略可以不同**：例如每个 session 开始时自动加载 memory 索引（自动读），但只在模型判断需要时才写入（模型控制写）。Lecture 2 中 Claude Code 的 auto memory 就是这样。

### 2.5 怎么检索？

**基础流程**：

```
Query → Vector search → Top-k memories → 放进 Prompt
```

但**语义相关性只是一个信号**。还要考虑：

| 信号 | 含义 | 例子 |
|---|---|---|
| **Semantic match（语义匹配）** | 意思与查询相近 | “部署”与“上线”相关 |
| **Scope（范围）** | 是不是同一个用户、项目、实体或任务 | 用户 A 的偏好不能用在用户 B 身上 |
| **Importance（重要性）** | 漏掉它或误加它的代价有多大 | “用户对花生过敏”比“用户喜欢蓝色”重要得多 |
| **Validity（有效性）** | 在这个任务、这个时间点是否仍然成立 | “项目用 vLLM”在 6 月之后已经不成立 |
| **Provenance（来源）** | 来源、置信度和证据 | 用户亲口说的 vs 模型自己推测的 |

一个常见做法是把多个信号**加权组合**成检索分数。例如 Generative Agents（Park et al., 2023）用“最近程度（时间越近越高，按指数衰减）+ 重要性（由 LLM 打 1–10 分）+ 相关性（向量相似度）”三项相加来排序记忆。

### 2.6 怎么从观察中归纳（Reflection）？

例：用户先后问过 vLLM、SGLang、TRT-LLM 和 LLM 调度的问题。

```
问过 vLLM  ┐
问过 SGLang ├─ reflection → 假设：用户经常关注 LLM 推理系统   置信度：medium
问过 TRT-LLM│
问过调度    ┘
```

归纳出的是**带置信度的假设**，而不是确定的事实。

**什么时候做 consolidation（整合）**：
- 积累了足够多的新证据之后，例如每 N 个事件，或 context 快满时
- 事实之间出现**冲突**或**重复**时
- 尽可能放在**后台任务**中做，不影响交互延迟

### 2.7 遗忘是一等操作

**为什么要遗忘**：
- **信息过时**：项目已经不用 vLLM 了
- **信息错误或互相冲突**
- **隐私与安全**：用户要求删除；不该保留的敏感信息
- **存储和检索成本**：记忆越多，检索越慢，越容易检索到无关内容

**生命周期控制手段**：

| 手段 | 说明 |
|---|---|
| **Mark invalid（标记失效）** | 不删除，但标记为“已被取代”，检索时默认跳过，需要时还能追溯历史 |
| **Erase permanently（永久删除）** | 隐私要求时必须真正删除 |
| **Expire after a TTL（过期）** | 设置存活时间，到期自动失效（例如“本周在出差”） |
| **Tiers and eviction（分层与淘汰）** | 常用的放在活跃层，不常用的移到冷存储或淘汰出活跃 memory |

### 2.8 Memory 需要有效时间和来源

**Valid time（有效时间）**：哪个事实在什么时间段成立。

```
2026 年 4 月：Project serves models with vLLM    → superseded（已被取代）
2026 年 6 月：Project serves models with SGLang  → current（当前有效）
```

不要直接用新事实覆盖旧事实，而是记录每个事实的有效区间。这样既能回答“现在用什么”，也能回答“4 月份的时候用什么”。

**Provenance（来源）**：这个事实从哪里来。

```
Episode 28：“We moved serving from vLLM to SGLang.”
      ↓ 提取
Project → SERVES_WITH → SGLang     source: Episode 28
```

有了来源，就可以在出现冲突时判断该信谁，也能在原始事件被证明有误时，找到并修正所有由它派生出的记忆。

### 2.9 谁拥有 memory？

| | Agent-owned（agent 自己拥有） | Dedicated memory system（独立的 memory 系统） |
|---|---|---|
| 形式 | harness 中的文件：CLAUDE.md、agent scratchpad、任务笔记 | agent 调用的一个服务（库或独立服务）：Mem0、Graphiti、supermemory |
| 读写方式 | harness 直接读写磁盘上的文件 | 通过 API：search、update…… |
| 特点 | 简单、透明、可以用 Git 管理 | **Memory 是一个 API，可以在多个 agent 和 session 之间共享** |

### 2.10 几个独立的 memory 系统

| 系统 | 定位 | 输入 | 检索 | 例子 |
|---|---|---|---|---|
| **Mem0** | 以**事实**为中心 | 消息，按 user、agent 或 run 划分范围 | 带过滤的向量搜索；可选图记忆 | “Uses SGLang for model serving.” |
| **Graphiti** | **时间感知的知识图谱** | episodes（事件），从中提取实体和带时间的事实 | 语义 + 关键词 + 图遍历的混合搜索 | uses → SGLang，有效期 2026 年 4 月 → 至今 |
| **supermemory** | 用户画像 + 记忆 + 文档 | 对话，关联到来源和时间 | 画像、记忆和文档搜索 | 维护一份用户 profile |

（以 Mem0 为例，写入时会先从对话中提取候选事实，再与已有的相似记忆比较，由 LLM 决定执行 ADD（新增）、UPDATE（更新）、DELETE（删除）还是 NOOP（不变）。）

### 2.11 Memory 设计 checklist

```
Observe（观察）
 → Select：这值得记吗？
 → Encode / organize：什么表示方式能保留需要的含义？
 → Store：存在哪里？文件？数据库？
 → Retrieve：对当前任务有用的是什么？
 → Reflect / consolidate：它是否更新或与旧记忆矛盾？
 → Forget：什么时候信息过时或不再相关？
```

**每一步都是一个策略决策。**

### 2.12 八种失败模式

| 失败 | 描述 | 应对 |
|---|---|---|
| **Retrieval miss（检索遗漏）** | 相关的记忆存在，但没有被找到 | 改进检索（混合检索、更好的 query） |
| **Context pollution（context 污染）** | 太多无关的记忆进入 context | 提高检索阈值、按 scope 过滤 |
| **Memory hallucination（记忆幻觉）** | **推断被当作事实存储**。“在 Harvard 读书”被记成了“是 Harvard 的访问学生” | 区分“用户陈述”和“模型推断”，记录来源和置信度 |
| **Staleness（过时）** | 旧事实覆盖了新信息 | 记录有效时间；使用前重新核实 |
| **Isolation failure（隔离失败）** | 记忆跨越了用户边界（A 的信息出现在 B 的对话中） | 严格按 scope 隔离存储和检索 |
| **Overgeneralization（过度泛化）** | 一次偶然事件变成了长期偏好（用户某天说“今天想吃清淡的”，被记成“用户只吃清淡的”） | 归纳时要求多个证据，并附置信度 |
| **Persistent prompt injection（持久化的提示注入）** | 不可信内容被存进记忆，之后**以指令的形式**被读回来 | 不把外部内容当作指令存储；读回时标记为数据 |
| **Unbounded growth（无限增长）** | 大小随使用时间不断增长，检索成本上升、质量下降 | 遗忘、合并、分层 |

### 2.13 如何评估 memory

**组件级评估**：

| 方面 | 指标 |
|---|---|
| **Retrieval（检索）** | precision、recall |
| **Writes and updates（写入与更新）** | precision、recall，以及过时记忆的比例（stale-memory rate） |
| **Governance（治理）** | 删除请求是否真正执行（deletion compliance）、范围隔离是否可靠 |
| **System（系统）** | 延迟、额外消耗的 token、存储增长速度 |

**端到端 benchmark**：

| Benchmark | 规模 | 特点 |
|---|---|---|
| **LoCoMo**（ACL 2024） | 1,540 道题 | 10 段很长的多轮对话 |
| **LongMemEval**（ICLR 2025） | 500 道题 | 考察 5 种记忆能力：信息提取、跨 session 推理、时间推理、知识更新、拒答（不知道时说不知道） |
| **BEAM**（ICLR 2026） | 2,000 道题 | 对话长度最多 1,000 万 tokens |

这些 benchmark 主要测试 agent 能否在**用户与助手的长对话历史**上回忆和推理，覆盖回忆、时间推理、更新、偏好和拒答。

### 2.14 Agent memory stack

| 独立的 memory 层 | memory 在 harness 内部 |
|---|---|
| Agent runtime：Claude Code、Letta、Codex | Agent runtime：Claude Code、Letta、Codex |
| Memory management：Mem0、Letta 内置、自定义 | Memory management：在 runtime 内部完成 |
| Storage：向量库、SQL、图数据库、对象存储 | Storage：Markdown 文件 |

---

## 3. Agent Reliability（可靠性）

### 3.1 为什么可靠性是个问题：错误会复利累积

假设每一步**独立地**以 95% 的概率成功，一次运行只有**每一步都成功**才算成功：

$$
P(\text{整次运行无错误}) = 0.95^{n}
$$

| 步数 $n$ | 成功率 |
|---|---|
| 1 | 95% |
| 5 | 77% |
| 10 | 60% |
| 20 | 36% |
| **50** | **8%（约 13 次中只有 1 次）** |

更糟的是，真实运行中错误会**传播**：前一步的错误让后一步更容易出错（基于错误的前提继续推理），所以实际情况比这个独立假设还差。

**反过来想**：要让一个 50 步的任务有 90% 的成功率，每一步需要 $0.9^{1/50} \approx 99.8\%$ 的成功率。所以长任务的可靠性不能只靠“每步做得更好”，还需要**验证和纠错**机制：能在中途发现错误并修正，就打破了连乘。

### 3.2 失败从哪里来

```
User / task spec（任务描述：目标是否清楚？）     → specification failure
Agent harness（context、循环、调度）            → harness failure
LLM（推理与规划）                               → model failure
Tool layer（shell、浏览器、API）                → tool failure
Environment（代码仓库、操作系统、网页、服务）    → environment failure
```

### 3.3 任务描述的失败（Specification failure）

| 失败 | 描述写的是 | Agent 做的是 | 修复 |
|---|---|---|---|
| **Ambiguous goal（目标模糊）** | “Fix the login bug” | 修了三个 bug 中错误的那个 | 写明文件、症状和复现步骤 |
| **Proxy objective（代理目标）** | “Make the tests pass” | **把失败的测试删掉了**（测试确实“通过”了） | 禁止修改测试；review diff |
| **Unstated constraint（没说的约束）** | “Speed up search” | 破坏了公开 API | 写明什么不能改 |
| **No definition of done（没有完成标准）** | “Clean up the parser” | 没完没了地重构 | 给出退出测试和停止规则 |
| **Conflicting goals（目标冲突）** | “最小的 diff，覆盖所有边界情况” | 悄悄放弃了边界情况 | 明确目标的优先级 |

“Proxy objective”是最值得警惕的：agent 会优化你**说出来的**目标，而不是你**真正想要的**目标。

### 3.4 Harness 的失败

| 失败 | Harness 做了什么 | 后果 | 修复 |
|---|---|---|---|
| **Context truncation（截断）** | 截掉了最早的几轮对话 | 忘了“保留 API 不变”这条规则 | 把关键规则固定在可被截断的窗口之外（如 system prompt），compaction 后重新注入 |
| **Memory error（记忆错误）** | 把一条过时的笔记当作事实读出 | 编辑了一个已经被移走的文件 | 给记忆加日期；使用前重新核实 |
| **State corruption（状态损坏）** | 留下了一个只应用了一半的编辑 | 后续步骤都建立在坏掉的文件上 | Checkpoint；失败的步骤回滚 |
| **Control loop error（循环控制错误）** | 没有设置步数或成本上限 | 200 步毫无进展 | 限制步数、成本和时间 |
| **Blocking fan-out（阻塞的扇出）** | 等待每一个 sub-agent 都完成 | 一个慢的子任务拖住了整个运行 | 设置超时；使用部分结果 |

### 3.5 模型的失败

| 失败 | 模型的推理 | Agent 做的事 | 修复 |
|---|---|---|---|
| **Hallucination（幻觉）** | “`requests.fetch()` 存在” | 调用了一个不存在的函数 | 对照真实文档检查调用 |
| **Instruction violation（违反指令）** | “速度比 API 更重要” | 改了被要求保留的函数签名 | 重申约束；检查 diff |
| **Faulty reasoning（推理错误）** | “超时说明网络断了” | 给一个死锁加了重试 | 修复之前先验证假设 |
| **Reasoning-action mismatch（说一套做一套）** | “我只改 parser.py” | 改了另外三个文件 | 把每个动作与计划对照 |
| **Unverified success（未经验证就宣布成功）** | “测试应该通过了” | 什么都没运行就宣布完成 | **要求以测试输出作为证明** |
| **Long-horizon degradation（长程退化）** | “config 还需要修” | 撤销了自己第 12 步做的修复 | 重新读取当前状态；拆成短任务 |

### 3.6 工具的失败

| 失败 | 工具返回 | Agent 做的事 | 修复 |
|---|---|---|---|
| **Truncated output（输出被截断）** | 日志的前 100 行 | 错过了第 4,000 行的错误 | 先返回错误；支持分页 |
| **Silent failure（静默失败）** | 退出码 0，结果为空 | 以为“没有匹配” | **区分“空结果”和“执行失败”** |
| **Ambiguous schema（含糊的格式）** | 一个没有时区的日期 | 把任务安排错了 8 个小时 | 使用有类型的、明确的 schema |
| **Flaky call（不稳定的调用）** | 第一次请求超时 | 宣布服务挂了 | 带退避地重试后再下结论 |
| **Misleading error（误导性的错误）** | 拼写错误却返回 “Permission denied” | 用 sudo 重试 | 返回精确、可操作的错误信息 |

### 3.7 环境的失败

| 失败 | 环境中有什么 | Agent 做的事 | 修复 |
|---|---|---|---|
| **Stale state（过时的状态）** | 昨天的构建缓存 | 测试在旧代码上通过了 | 用干净的缓存重新构建 |
| **Version mismatch（版本不匹配）** | Python 3.8 而不是 3.12 | 交付了会崩溃的代码 | 用 lockfile 固定版本 |
| **Random failure（随机失败）** | 任务成败取决于环境 | 因为闰年而失败 | 重跑失败的任务；隔离不稳定的测试 |
| **Changed web page layout（网页改版）** | 结账页面重新设计了 | 点错了按钮 | 行动前重新读取页面 |
| **Rate limit（限流）** | 50 次调用后返回 HTTP 429 | 迁移做到一半停了 | 限速，并为长任务设置 checkpoint |

### 3.8 为失败而设计

| 层 | 原则 | 做法 |
|---|---|---|
| **Task specification** | **把“完成”写成测试** | 事先写好验收测试；列出不能改动的内容 |
| **Agent harness** | **给每次运行设上限** | 步数、成本、时间上限；checkpoint 与回滚 |
| **Model** | **要求证据** | 每个结论都要有证明；做验证；必要时使用更强的模型 |
| **Tool** | **大声地失败** | 有类型的 schema；错误与空结果区分开 |
| **Environment** | **固定并隔离** | 全新的、固定版本的 sandbox；下结论之前先重试不稳定的操作 |

> 我们对大规模 agent 的理解还很有限，几乎没有正式的最佳实践。

---

## 4. 要点回顾

1. Memory 是由 harness 管理、超出 context window 的持久状态；LLM 本身仍然无状态
2. 写入策略在 recall 与噪声之间权衡：用户要求、规则、分类器、定期 LLM 审查
3. 表示方式（原始事件、摘要、结构化状态、知识图谱）与存储后端（文件、数据库、向量库、图数据库）各有所长，常组合使用
4. 检索不能只看语义相似：还要看范围、重要性、有效性和来源
5. 遗忘是一等操作：标记失效、永久删除、TTL、分层淘汰；记录有效时间和来源，而不是直接覆盖
6. 八种失败：检索遗漏、context 污染、记忆幻觉、过时、隔离失败、过度泛化、持久化注入、无限增长
7. 错误复利：0.95⁵⁰ ≈ 8%；长任务需要中途验证和纠错，而不只是提高单步成功率
8. 失败分五层定位（任务描述、harness、模型、工具、环境），并对应五条设计原则：把完成写成测试、设上限、要证据、大声失败、固定并隔离

---

## 5. 自测题

<details>
<summary>1. 写入策略中，“写入每次交互”和“只在用户要求时写入”各有什么问题？</summary>

写入每次交互 recall 高但噪声大，检索时会带进大量无关内容，并且存储无限增长；只在用户要求时写入 precision 高，但用户不会想到每件值得记的事，很多有用信息会丢失。
</details>

<details>
<summary>2. 项目 4 月用 vLLM，6 月改用 SGLang。直接用新事实覆盖旧事实有什么问题？更好的做法是什么？</summary>

覆盖后就无法回答“4 月份用的是什么”，也无法追溯变化的原因和来源。更好的做法是给事实记录有效时间区间（4 月的标记为 superseded，6 月的为 current）并附上来源（如 Episode 28）。
</details>

<details>
<summary>3. 举一个 memory hallucination 的例子，并说明如何避免。</summary>

用户说“我在 Harvard 读书”，系统把推断“用户是 Harvard 的访问学生”当成事实存了下来。应区分“用户陈述”与“模型推断”，记录来源与置信度，推断不应被当作确定事实使用。
</details>

<details>
<summary>4. 每步成功率 99%，100 步任务的成功率约是多少？</summary>

0.99¹⁰⁰ ≈ 0.366，约 37%。即使单步很可靠，长任务的成功率也会大幅下降，所以需要中途验证和纠错。
</details>

<details>
<summary>5. 告诉 agent “Make the tests pass”，它删掉了失败的测试。这属于哪类失败？怎么修复？</summary>

属于任务描述层的 proxy objective（代理目标）失败：agent 优化了说出来的目标而不是真正的意图。修复：明确禁止修改测试，并在 review 中检查 diff。
</details>

<details>
<summary>6. 工具返回“退出码 0、结果为空”，agent 判断“没有匹配”，实际上命令执行失败了。如何从工具设计上避免？</summary>

让工具“大声失败”：把执行失败与空结果明确区分开（不同的状态码或字段），并返回精确、可操作的错误信息。
</details>

<details>
<summary>7. 为什么说 Persistent prompt injection 比普通的 prompt injection 更危险？</summary>

恶意内容被存进 memory 后，会在之后的每个 session 中被当作可信的记忆自动读回，甚至以指令的形式出现，影响长期存在、难以察觉。需要避免把外部内容当作指令存储，读回时也把它标记为数据。
</details>
