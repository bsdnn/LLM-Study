# Lecture 4 · Agent Design I：Context Management and Tool Design

> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/4_agent_design_1.pdf`](../slides/4_agent_design_1.pdf)

## 本讲要点

- **ReAct loop**：Thought → Action → Observation
- **Tool design**：search/read、edit、execution 各有多种实现，各有取舍
- **Context management**：prompt engineering → context engineering；四种管理手段：Admission、Offloading、Transformation、Compaction
- **Agent = Model × Harness**：同一个模型，换 harness 可以让准确率翻倍

---

## 1. ReAct Loop

| 步骤 | 内容 |
|---|---|
| Thought | 我知道什么、缺什么、下一步做什么 |
| Action | 发出 environment 可以执行的 structured call |
| Observation | environment 返回事实 |

- **只 reason**：没人检查推理，小错误累积成自信的 hallucination
- **只 act**：没有计划的 tool call，模型重复自己
- **ReAct**：thought 引导下一个 action，observation 纠正下一个 thought；直到 observation 回答了问题或触发 stopping rule

论文：<https://arxiv.org/pdf/2210.03629>

### 最简 Agent Loop

```python
messages = [{"role": "system", "content": "Solve the task using the tools."},
            {"role": "user", "content": user_request}]
for step in range(MAX_STEPS):
    response = call_model(messages, tools)
    messages.append(response.message)
    if response.final_answer:
        return response.final_answer
    for call in response.tool_calls:
        args = validate(call.arguments)
        result = execute_tool(call.name, args)
        messages.append({"role": "tool", "tool_call_id": call.id,
                         "content": serialize(result)})
raise RuntimeError("Agent exceeded its step limit")
```

作为 **harness designer** 你要做的：写 tool definitions、组装 context、validate request + 检查 permission + 执行、处理结果并管理 context。

### Coding Agent 的三个组件（共享一个 context window，要一起设计）

1. **Model + agent loop**：LLM 提出 action、看结果、重复
2. **Tools and guardrails**：search、read、edit、run tests、shell，配合 permissions、sandboxing、approval
3. **State：context and memory**

---

## 2. Tool Design

### 2.1 Search / Read：Agent 如何找到相关代码？

| 方法 | 原理 | 适合 | 主要成本 |
|---|---|---|---|
| **glob / grep** | 字面名称和路径 | 名字已知或可猜 | 每次搜索消耗 turns 和 context；lexical 匹配漏掉同义改写 |
| **Semantic index** | embedding chunks，按语义检索；或 structural（Aider 的 **repo map**：文件树 + 排序的 symbols） | 措辞对不上代码时 | 构建和同步 index |
| **Extrinsic signals** | stack trace、failing tests、git history | 已有报告指向代码（debug 时精度最高） | 只在部分任务中存在 |
| **Language server (LSP)** | compiler 的 symbol graph（definitions、references、types） | 编辑需跟随引用（rename 等） | 每种语言一个 live server，冷启动慢 |

实践：**先用 grep；名字不够用时再上 index 或 LSP**。

### 2.2 Edit：Agent 如何修改代码？

| 方法 | 优点 | 缺点 |
|---|---|---|
| **Whole-file rewrite** | 没有 patch 语法可写错，总能应用 | output 与文件大小成正比；未改动代码可能 drift |
| **Unified diff / patch** | 紧凑、可 review，人和 CI 都熟悉 | 行号或上下文漂移时 hunk 失败 |
| **Search and replace** | output 与改动大小成正比 | anchor 有歧义、过时或空白不对时失败 |
| **Structured / AST edits** | 不受格式影响，不会产生非法语法 | 需要每种语言的 tooling，可表达的编辑有限 |

两个维度权衡：**token cost** 与 **robustness**（编辑是否落在预期位置）。

### 2.3 Execution：本质上是一个 Shell

最强大也最危险的 tool：
- **Composition**：一次调用可以 chain、filter、loop、retry
- **Reach**：文件系统、网络、包管理器、二进制
- **Persistence**：文件和进程比单次调用活得久
- **Blast radius**：shell 能做的，agent 都能做

需要考虑：**Isolation**（在哪运行、能访问什么）、**Failure signal**（agent 能据此行动的错误）、**Output**（够用但别太多）、**Duration**（长任务和后台进程）、**Budgets**（何时停止重试并交还）。

### 2.4 其他常用 Tools

- Memory and notes：把发现写到文件，plan 在 compaction 后仍存在
- Subagents：exploration 返回 shortlist，而不是 40 次文件读取
- Web search / fetch：先读已安装的源码，再查网络
- Browser 与视觉反馈：截图渲染结果再反馈
- Asking the user：一个便宜的问题胜过一个自信的错误 patch

### 2.5 Tool Design Best Practices

- **Few, orthogonal tools**：重叠的 tools 会让模型犹豫
- **Purpose-built, not generic**：`search_orders(customer_id)` 胜过裸的 `run_sql`
- **Token-cheap output**：truncate、paginate、summarize（**tool design 就是 context design**）
- **Errors that teach**：返回修复提示，而不只是 “invalid argument”
- **Idempotent by default**：尽量可以安全重试

SWE-agent 消融实验（GPT-4，SWE-bench 解决率）：完整设计 18.0%；保留完整交互历史 15.0%；整文件查看 12.7%；只用 shell 11.0%；去掉专门 editor 10.3% —— 为 agent 专门设计的 tools（Agent-Computer Interface）明显优于直接给一个 shell。

---

## 3. Context Management

演进：**Prompt Engineering**（2023–24，怎么指示模型？）→ **Context Engineering**（2025–26，模型应该看到什么？）→ **Harness Engineering**（agent 如何运作？loop、tools、sandbox、retries、stopping）→ **Goal Engineering**（什么算成功？objectives、criteria、constraints）。

### 3.1 Prompt Engineering 技巧

| 技巧 | 做法 | 何时用 |
|---|---|---|
| **Zero-shot** | 只有 instruction | 常见任务、格式用文字说得清 |
| **Few-shot** | instruction + 几个例子 | 输出格式特殊、规则“示范比描述容易” |
| **Role & persona** | 给模型一个角色 | 语气或判断力重要、受众明确 |
| **Chain-of-thought (CoT)** | 先写推理步骤，最后给答案 | 任务确实有多步、错误来自跳步 |
| **Tree-of-thought (ToT)** | 分支、打分、剪枝 | 多条路径都合理且可以便宜地打分 |
| **Self-consistency** | 多次采样（如 T=0.7）后多数投票 | 有唯一正确答案、错误代价高（投票不够一致时转人工） |
| **Prompt chaining** | 每次调用一个任务，交接时校验 | 工作可分阶段且每阶段可校验（schema 失败只重试该阶段） |
| **Meta prompting** | 让模型改写自己的 prompt | 有 eval set、在调 prompt 本身 |

选择顺序：先简单（zero-shot / few-shot / role）→ 加推理（CoT / ToT / self-consistency）→ 系统工程（chaining / meta prompting）。注意：部分技巧对新模型已经“过时”，例如具备内置 reasoning 的模型不再需要手动要求 CoT，最好用自己的 eval 实测。

**Prompt 设计四原则**：从简单开始迭代；分节（context、instructions、tools、output spec）；具体简洁（花哨措辞没用）；大任务拆成子任务。**Prompting 建议因模型而异，换模型就要重新调。**

### 3.2 Context Engineering

> “Context engineering is the delicate art and science of filling the context window with just the right information for the next step.” —— Andrej Karpathy

**Context 是有限资源**：
- **Cost**：每一步都会重新发送整个 window
- **Latency**：prompt 越大，prefill 和 decode 越慢
- **Attention**：window 越满 recall 越差，中间部分容易被忽略
  - **Context Rot**（Chroma 的实验）：即使是简单查找任务，随着输入 token 增多准确率也持续下降；加入干扰信息后下降更明显
  - **Lost in the Middle**（Liu et al., TACL 2024）：把答案放在 20 篇文档中的不同位置，放在开头或结尾时准确率最高，放在中间时最低（U 形曲线）

**一个 agent turn 的 token budget 示意**：System prompt ~2K、Tool definitions ~6K、Project environment ~8K、Retrieved context ~25K、Message history ~80K 且持续增长。

### 3.3 四种管理手段（从预防性、无损 → 有损、不可逆）

1. **Admission**：守住入口
   - Targeted retrieval（窄查询，取字段而不是整个文档）
   - Tool output shaping（丢掉 DOM，只保留提取结果）
   - Truncation（shaping 不行时才截断）
   - Isolation（把原始 payload 交给 subagent）
   - 例：原始 DOM 48,000 tokens → 只保留价格和标题 900 tokens，**小 98%，答案相同**
2. **Offloading**：需要时才加载
   - Progressive tool definition、Tool search（保留 index，按需取 schema）
   - Tool output offload（返回 handle）
   - Externalised task list（plan 放磁盘）
   - Persistent memory（笔记写出去，保留小 index）
   - 例：60 个 tool schemas 42,000 tokens → index + 3 个 schema 3,100 tokens，**小 93%，60 个仍可用**
3. **Transformation**：有选择地丢弃历史
   - Deduplication（每个对象只保留最新状态）
   - Observation masking（去掉 N 步之前的 tool results）
   - Sliding window（只保留开头和最近的 turns）
   - 例：96,000 → 24,000 tokens，**小 75%，没有做 summary**
4. **Compaction**：LLM 作为最后手段
   - Rolling summary、Full summary、Auto memory（关键事实持久化到 window 外）
   - 注意：为 output 预留 headroom；保留 system prefix；compaction 后重新注入关键事实

### 3.4 Context 失败的五种方式（Breunig, 2025）

| 失败 | 描述 |
|---|---|
| **Poisoning** | hallucination 进入 window，之后被当作事实引用 |
| **Distraction** | 历史太长，agent 开始模仿自己过去的动作 |
| **Confusion** | 无关的 tools 和文档把答案带偏 |
| **Clash** | 新旧信息矛盾且未解决 |
| **Drift** | 真正目标被埋没，被换成相近的子目标 |

---

## 4. Agent = Model × Harness

- **Model capability 决定上限，Harness 决定实际表现**
- 同一模型换 harness：SWE-bench Verified 43/169 → 72/169；SWE-bench Pro 31/316 → 72/316
- 改了什么：压缩旧 tool outputs、保持最近 observations 可见、检测重复和停滞、卡住时干预
- 很多看起来像 “model failure” 的问题其实是 **context 和 control-loop failure**

---

## 5. Optional：ReAct 之外

| 方法 | 循环方式 |
|---|---|
| ReAct | 每一步都循环 Thought → Act → Observe |
| Reflexion | 失败后循环：Attempt → Evaluate → Critique → Retry |
| Plan-and-execute | 偏离时循环：Plan → Execute steps → Re-plan |
| Tree search | 不循环，一次宽搜索：并行探索 Branch A/B/C，只保留最好的 |
