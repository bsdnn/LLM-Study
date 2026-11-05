# Lecture 4 · Agent Design I：Context Management and Tool Design

> Harvard CS 2680：Modern AI Systems: Agents and System Optimizations
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/4_agent_design_1.pdf`](../slides/4_agent_design_1.pdf)

## 本讲脉络

```
Agent loop：ReAct（Thought → Action → Observation）与最简实现代码
→ Coding agent 的三个组件：model + loop、tools + guardrails、state
→ Tool 设计：怎么找代码（grep / 语义索引 / 外部信号 / LSP）、怎么改代码（整文件 / diff / 搜索替换 / AST）、
  怎么执行（shell）、其他常用工具、tool 设计原则
→ Context 管理：prompt engineering 的八种技巧 → context engineering
  → context 是有限资源 → 四种管理手段（admission、offloading、transformation、compaction）→ 五种失败方式
→ Agent = Model × Harness：换 harness 准确率翻倍
→（选读）ReAct 之外：Reflexion、plan-and-execute、tree search
```

一句话总结：**作为 harness 的设计者，你要做三件事：设计 tools、组装 context、控制循环。** Tool 的输出就是 context 的输入，所以 tool design 就是 context design。模型能力决定上限，harness 决定实际能发挥出多少 —— 同一个模型换一个更好的 harness，SWE-bench 的解决率可以翻倍。

---

## 1. Agent Loop

### 1.1 ReAct：推理与行动交替

ReAct（Reason + Act，https://arxiv.org/pdf/2210.03629）把一次任务拆成反复进行的三步：

| 步骤 | 内容 | 例子 |
|---|---|---|
| **Thought** | 我知道什么？缺什么？下一步做什么？ | “测试失败在 auth 模块，我需要看 retry 的逻辑” |
| **Action** | 发出一个 environment 可以执行的结构化调用 | `grep -rn "timeout" src/` |
| **Observation** | environment 返回事实 | “37 hits in 12 files” |

为什么要交替？

- **只推理（reason only）**：没有任何东西检查推理是否正确，小错误层层累积，变成**自信的 hallucination**
- **只行动（act only）**：工具调用背后没有计划，模型会重复自己，原地打转
- **ReAct**：thought 指导下一个 action；observation 纠正下一个 thought。不断循环，直到 observation 回答了问题，或者触发停止规则（如最大步数）

### 1.2 最简 agent loop 实现

```python
messages = [
    {"role": "system", "content": "Solve the task using the tools."},
    {"role": "user", "content": user_request},
]
for step in range(MAX_STEPS):
    response = call_model(messages, tools)        # 模型看到全部历史 + 工具定义
    messages.append(response.message)
    if response.final_answer:                     # 模型认为完成了
        return response.final_answer
    for call in response.tool_calls:              # 模型要调用工具
        args = validate(call.arguments)           # 校验参数
        result = execute_tool(call.name, args)    # 真正执行
        messages.append({                         # 结果追加到历史
            "role": "tool",
            "tool_call_id": call.id,
            "content": serialize(result),
        })
raise RuntimeError("Agent exceeded its step limit")   # 停止规则
```

几个要点：
- `messages` 就是 context，每一步都在增长
- 模型**只返回**要调用什么，执行由代码完成
- 一定要有**步数上限**，否则可能无限循环

### 1.3 作为 harness 设计者，你要做的事

| 环节 | 你的工作 |
|---|---|
| 模型调用之前 | **写 tool definitions**（名字、描述、参数 schema）；**组装 context** |
| 模型返回 tool call 之后 | **校验请求、检查权限、执行** |
| 工具返回之后 | **可选地处理结果**（截断、摘要）并**管理 context** |

### 1.4 Coding agent 的三个组件

三者共享一个 context window，必须一起设计：

| 组件 | 作用 | 要点 |
|---|---|---|
| **Model + agent loop** | LLM 提出动作、看结果、重复直到完成 | Agent = 模型 + 循环 |
| **Tools and guardrails** | 搜索、读、编辑、跑测试、shell，外加权限、sandbox、审批 | Tools 决定能力，guardrails 保证安全 |
| **State：context 与 memory** | 当前窗口里的内容，以及 agent 在步骤之间带着走的东西 | 组装好 context；保留需要跨 turn 存活的东西 |

---

## 2. Tool 设计

应该给 agent 提供哪些工具？大致四类：**Search / Read、Edit / Write、Execution、其他**。

### 2.1 Search / Read：agent 怎么找到相关代码？

**方法 1：glob 和 grep（Claude Code 的主要方式）**

- Agent 自己循环调用 ls、glob、grep —— 不需要预先构建任何东西，也不用保持同步
- 永远反映**实时的** repo 状态
- 代价：要花很多 turns 和 context；**字面匹配**会漏掉换了说法的代码

例：“fix the login timeout”
1. `grep -rn "timeout" src/` → 12 个文件中 37 处匹配
2. `glob src/**/auth/*.ts` → 缩小到 `auth/retry.ts`
3. 读文件并修改：`RETRY_MS = 500` → 已修复

**方法 2：语义搜索（semantic search）**

- **Embedding 检索**：把代码切成块并转成向量，按“意思”检索。措辞不同但意思相近的代码，向量也相近
- **结构化检索**：文件树 + 排好序的符号列表（如 Aider 的 **repo map**：列出每个文件中的类和函数签名，按重要性排序）
- 查询时很快；代价是**构建索引和保持同步**

例：“where do we rate-limit?” 第一次查询就返回 `throttle.ts` —— 即使代码里根本没有 “rate-limit” 这个词。

**方法 3：外部信号（extrinsic signals）**

- 利用源代码之外的证据：**stack trace、失败的测试、git history**
- debug 时**精度最高**的信号
- 只在这类信号存在时才有用

例：结账时崩溃的报告
1. 读 stack trace：`cart.ts:88`，列出 3 个调用栈帧
2. 打开这些帧：读 `tax.ts:20-60`，发现 `applyTax` 中的舍入问题
3. 修复并重跑 `pytest tests/tax.py` → 通过，全程没用搜索

**方法 4：Language Server（LSP）**

- 查询 compiler 的索引：定义、引用、符号、类型
- 能正确解析 import、重载、重命名 —— 这些 grep 只能猜
- 代价：每种语言要运行一个 live server；冷启动慢

例：把 `charge()` 重命名为 `capture()`
1. `textDocument/references` → 7 个文件中 42 处调用（包括用别名 import 的）
2. `textDocument/rename` → 全部改名
3. 类型检查 → 没有未解析的符号

**对比**：

| 方法 | 靠什么找代码 | 最适合 | 主要成本 |
|---|---|---|---|
| glob / grep | 字面名称和路径 | 名字已知或可以猜到 | 每次搜索都消耗 turns 和 context |
| 语义索引 | 代码块的意思 | 措辞对不上代码 | 构建和同步索引 |
| 外部信号 | trace、失败测试、git 历史 | 已有报告指向代码 | 只在部分任务中存在 |
| Language server | compiler 的符号图 | 编辑必须跟随引用 | 每种语言一个 live server |

实践建议：**先用 grep；名字不够用时再上索引或 LSP。**

### 2.2 Edit：agent 怎么改代码？

**方法 1：整文件重写（whole-file rewrite）**
- 模型重新生成整个文件，工具直接覆盖
- 没有 patch 语法可以写错 —— **每次编辑都能干净地应用**
- 代价：输出量与文件大小成正比；没改的代码也可能悄悄“漂移”（模型重写时不小心改了别的地方）
- 例：给 API client 加一个 retry —— 读入 120 行，重写全文件，实际只改了 4 行

**方法 2：Unified diff / patch**
- 模型输出一个 patch，由另一个工具应用（如 `git apply`）
- 紧凑、可以 review，是人和 CI 都熟悉的格式
- 代价：行号或上下文漂移时，hunk 会应用失败
- 例：在两个文件中提高 retry 上限 —— `@@ -41,3 +41,3 @@`，6 行，两个 hunk

**方法 3：搜索替换（search and replace，Claude Code 的 Edit 工具）**
- 模型给出一段精确的**锚点字符串**和它的替换内容
- 输出量与**改动大小**成正比，而不是文件大小
- 代价：锚点有歧义（出现多次）、过时、或空白字符不对时会失败
- 例：`retries = 3` → `retries = 5`，一次调用，文件其余部分不动

**方法 4：结构化 / AST 编辑**
- 模型指定一个针对符号的操作：重命名、添加参数……
- 不受格式影响；**不可能产生非法语法**
- 代价：每种语言都需要专门工具；能表达的编辑种类有限
- 例：给 `fetchUser()` 添加 `timeout` 参数（默认 5s）—— 工具改写函数签名**以及全部 18 个调用处**，解析 + 类型检查都通过

（AST = Abstract Syntax Tree，抽象语法树：代码被 parser 解析后的树形结构。）

**两个维度的权衡**：

| 方法 | Token 成本 | 鲁棒性（编辑是否落在预期位置） |
|---|---|---|
| 整文件重写 | 高，长文件会漂移 | 高 |
| 搜索替换 | 低 | 需要精确且唯一的匹配 |
| Unified diff | 低 | 行号过时就会失败 |
| AST 编辑 | 低 | 语言覆盖有限 |

### 2.3 Execution：本质上就是一个 shell

最强大，也最危险的工具：

| 特性 | 含义 |
|---|---|
| **Composition（组合）** | 一次调用可以串联、过滤、循环、重试（`cmd1 | grep x && cmd2`） |
| **Reach（触达范围）** | 文件系统、网络、包管理器、任何已安装的程序 |
| **Persistence（持久性）** | 文件和进程会比单次调用活得更久 |
| **Blast radius（爆炸半径）** | shell 能做的任何事，agent 都能做 |

设计 shell 工具时要考虑：
- **Isolation**：代码在哪里运行？能访问什么？
- **Failure signal**：返回 agent 能据此行动的错误（退出码、stderr）
- **Output**：够用但不要太多（截断长输出）
- **Duration**：长时间运行的任务和后台进程怎么处理
- **Budgets**：什么时候停止重试并交还给用户

### 2.4 另外五个常用工具

| 工具 | 价值 |
|---|---|
| **A. Memory 和笔记** | 把发现写进文件；计划在 compaction 后依然存在 |
| **B. Subagents / 委派** | 探索的结果以“候选清单”形式返回，而不是 40 次文件读取塞进主 context |
| **C. Web search 和 fetch** | 先读已安装的源代码，再查网络（本地代码与实际版本一致） |
| **D. 浏览器和视觉反馈** | 截图渲染结果并反馈给模型（前端开发必需） |
| **E. 询问用户** | 一个便宜的问题胜过一个自信但错误的 patch |

### 2.5 Tool 设计最佳实践

- **少而正交的工具**：功能重叠的工具会让模型犹豫不决
- **专用而非通用**：`search_orders(customer_id)` 胜过一个裸的 `run_sql` 工具（更容易用对，也更安全）
- **输出省 token**：截断、分页、摘要（**tool 设计就是 context 设计**）
- **会“教”的错误信息**：返回修复提示，而不只是 “invalid argument”。例如 “old_string matched 3 times; include more surrounding lines to make it unique”
- **默认幂等**：尽量让工具可以安全地重试

**SWE-agent 的消融实验**（GPT-4 在 SWE-bench 上的解决率）：

| 配置 | 解决率 |
|---|---|
| 完整设计（专门的 Agent-Computer Interface） | 18.0% |
| 保留完整交互历史 | 15.0% |
| 整文件查看 | 12.7% |
| 只用 shell | 11.0% |
| 没有专门的编辑器 | 10.3% |

结论：**为 agent 专门设计的工具明显优于直接给一个 shell。**（能不能只给 shell？在作业里亲自试试。）

---

## 3. Context 管理

### 3.1 从 prompt 到 agent 系统：术语的演进

| 时期 | 术语 | 关心的问题 | 内容 |
|---|---|---|---|
| 2023–2024 | **Prompt engineering** | “该怎么指示模型？” | 措辞、例子、输出格式 |
| 2025–2026 | **Context engineering** | “模型应该看到什么？” | 检索、memory、历史、compaction |
| 现在 → 下一步 | **Harness engineering** | “agent 应该怎么运作？” | 循环、工具、sandbox、重试、停止 |
| | **Goal engineering** | “什么算成功？” | 目标、标准、约束、权衡 |

（讲者提醒：大家很爱造新词。介绍这些术语是为了让你看到时知道它们指什么，不代表讲者认同或喜欢它们。）

### 3.2 Prompt engineering 的八种技巧

**Zero-shot：只给指令**

```
You are a support triage assistant.
Classify the ticket as one of: billing | bug | feature | other
Reply with JSON only: {"label": ...}
If unclear, use "other".
Ticket: "Crashes when I export a PDF."
→ {"label": "bug"}
```

- 适用：任务常见、广为人知；格式用文字说得清楚
- 何时换：规则是隐性的、说不清 → few-shot

**Few-shot：指令 + 几个示例**

```
Classify the ticket. Reply with JSON only.
"Charged twice this month."     → {"label": "billing"}
"Can you add dark mode?"        → {"label": "feature"}
"asdfgh"                        → {"label": "other"}
"Crashes when I export a PDF."  → ?
```

- 适用：输出格式不寻常；规则“示范比描述容易”
- 何时换：任务需要多步推理 → chain-of-thought

**Role & persona：给模型一个角色**

```
You are a senior SRE reviewing a postmortem for an exec audience.
Flag evidence gaps; never speculate. Be direct: at most five bullets.
→ "Timeline omits the 04:12 alert ack — evidence gap, not a cause."
```

- 适用：语气或判断力很重要；受众明确
- 何时换：需要步骤 → chain-of-thought

**Chain-of-Thought（CoT）：先写步骤，最后给答案**

```
Work through it in numbered steps, then end with: ANSWER: <value>
Q: 3 features/week for 4 weeks, then 5/week for 2 weeks. Total?
1. 3 × 4 = 12
2. 5 × 2 = 10
3. 12 + 10 = 22
ANSWER: 22
```

- 适用：任务确实有多个步骤；错误来自跳过了某一步
- 何时换：有多条可行路径 → tree-of-thought

**Tree-of-Thought（ToT）：分支、打分、剪枝**

```
Propose 3 ways to cut checkout latency. Score each 1-5 on impact and effort, then expand the best.
A cache warm-up      impact 3  effort 2
B drop 3rd-party     impact 5  effort 4
C lazy-load asset    impact 4  effort 1
Expand C, then B. Prune A.
```

- 适用：多条路线都说得通；能便宜地给它们打分
- 何时换：只有一个正确答案 → self-consistency

**Self-consistency：多次采样，投票**

```
同一个 prompt，采样 5 次，T = 0.7："Is this invoice a duplicate?"
yes | yes | no | yes | yes → 多数：yes（4/5 = 0.8）
阈值 0.8 → 给出答案；如果只有 3/5，就转给人工
```

- 适用：存在唯一正确答案；判断错误代价高
- 何时换：任务能拆成独立阶段 → prompt chaining

**Prompt chaining：每次调用只做一件事**

```
Stage 1 extract：         输入 email 文本 → 输出 {vendor, amount, date}
Stage 2 validate + enrich：→ 输出 {..., vendor_id}
Stage 3 draft reply：      → 输出 {subject, body}
某阶段输出不符合 schema → 只重试那一个阶段
```

- 适用：工作可以拆成阶段；每个阶段的输出可以校验
- 何时换：你在调的是 prompt 本身 → meta prompting

**Meta prompting：让模型改写 prompt**

```
Here is my prompt, 10 test cases and the 3 it fails.
Rewrite the prompt to fix those 3 without breaking the other 7. Return only the new prompt.
→ v2 增加了一条明确的拒答规则：评测集 8/10 → 10/10
```

- 适用：你有一个可以打分的评测集；prompt 本身就是你要优化的东西
- 何时换：收益停滞 → 重新审视你的方法选择

**如何选择**：

| 阶段 | 技巧 | 什么时候用 |
|---|---|---|
| 从简单开始 | Zero-shot | 任何常见任务的基线 |
| | Few-shot | 输出格式特殊或规则隐性 |
| | Role & persona | 语气或判断力重要 |
| 加入推理 | Chain-of-thought | 任务需要多步推理 |
| | Tree-of-thought | 有多条值得打分比较的路径 |
| | Self-consistency | 多问几次，取多数票 |
| 工程化系统 | Prompt chaining | 可分离的阶段，交接时校验 |
| | Meta prompting | prompt 本身就是产物 |

注意：部分技巧对新模型已经“过时”。例如带内置 reasoning 的模型不再需要你手动要求 CoT。最好用自己的评测集实测。

**设计 prompt 的四条最佳实践**：
1. **从简单开始，然后迭代**：只在需要时才加 context
2. **分节**：context、instructions、tools、output spec 分开写
3. **具体且简洁**：花哨的措辞没用
4. **把大任务拆成子任务**：不要一开始就堆复杂度

```
### Instruction ###
Translate the text below to Spanish:
Text: "hello!"
→ ¡Hola!
```

顺序：指令 → 分隔符 → 数据。**Prompt 建议因模型而异，换模型就要重新调。**

### 3.3 Context Engineering

> “Context engineering is the delicate art and science of filling the context window with just the right information for the next step.” —— Andrej Karpathy
> （Context engineering 是一门精细的艺术和科学：为下一步，在 context window 里恰好放入正确的信息。）

### 3.4 Context 是有限的资源

| 维度 | 原因 | 后果 |
|---|---|---|
| **Cost** | 每一步都要重新发送整个窗口 | 每个 turn 都要再付一次钱 |
| **Latency** | prompt 越大，prefill 和 decode 越慢 | 循环的每一轮都更慢 |
| **Attention** | 窗口越满，recall 越差；中间的内容最容易被忽略 | **更多 context ≠ 更好的答案** |

**长度和位置都会影响准确率**：
- **Context Rot**（Chroma 的实验）：即使是简单的查找任务，随着输入 token 增加，准确率也持续下降；加入干扰信息后下降得更明显
- **Lost in the Middle**（Liu et al., TACL 2024）：把答案放在 20 篇文档中的不同位置，放在开头或结尾时准确率最高，放在中间时最低（U 形曲线）

### 3.5 Context window 的构成

一个 agent turn 的 token 预算示意：

| 部分 | 大约 |
|---|---|
| System prompt | ~2K |
| Tool definitions | ~6K |
| Project environment | ~8K |
| Retrieved context | ~25K |
| Message history（用户 + LLM + tool 结果） | **~80K，并且持续增长** |

最大的、增长最快的是 message history，而其中大部分是 tool 结果。

### 3.6 Context 太大时怎么办：四种手段

从**预防性、无损**到**有损、不可逆**排列：

```
Admission → Offloading → Transformation → Compaction
（守住入口）  （不用不加载）  （选择性丢弃）     （LLM 总结，最后手段）
```

**1）Admission：守住进入 context 的东西**

| 方法 | 做法 |
|---|---|
| Targeted retrieval | 窄范围查询；只取需要的字段，而不是整个文档 |
| Tool output shaping | 丢掉 DOM，只保留提取出的内容 |
| Truncation | 只有在无法 shaping 时才截断 |
| Isolation | 把原始大块内容交给 subagent 处理 |

例：原始 DOM 直接放进来要 48,000 tokens；只保留价格和标题后只有 900 tokens —— **小 98%，答案相同**。在入口处就塑形。

**2）Offloading：不需要就不加载**

保持“可以取回”，而不是“常驻”：

| 方法 | 做法 |
|---|---|
| Progressive tool definition | 只在需要时才加载工具 schema |
| Tool search | 保留一个索引，按需取工具定义 |
| Tool output offload | 返回一个 handle（引用），需要时再加载内容 |
| Externalised task list | 把计划放在磁盘上 |
| Persistent memory | 笔记写到外面，只保留一个小索引 |

例：60 个工具 schema 常驻要 42,000 tokens；只放索引 + 3 个当前需要的 schema 只要 3,100 tokens —— **小 93%，60 个工具都仍然可以用**。

**3）Transformation：有选择地丢弃历史**

| 方法 | 做法 |
|---|---|
| Deduplication | 每个对象只保留最新状态（同一个文件读了 5 次，只留最后一次） |
| Observation masking | 把 N 步之前的 tool 结果去掉（替换成占位符） |
| Sliding window | 只保留开头和最近的几轮 |

例：保留每次结果要 96,000 tokens；去重 + mask 后只保留最新状态 24,000 tokens —— **小 75%，而且没有做任何摘要**。

**4）Compaction：LLM 作为最后手段**

| 方法 | 做法 |
|---|---|
| Rolling summary | 边做边把旧的 tool 结果总结掉 |
| Full summary | 到达上限时重写整个历史 |
| Auto memory | 把关键事实持久化到窗口之外 |

注意事项：
- **为输出预留空间**（headroom），否则模型没地方写回答
- **保留 system prefix**（也有利于 prefix cache 命中）
- **压缩后重新注入关键事实**（规则、目标）

### 3.7 Context 失败的五种方式

（Breunig，“How Contexts Fail and How to Fix Them”，2025）**给失败命名，修复方法就显而易见了。**

| 失败 | 描述 | 例子 |
|---|---|---|
| **Poisoning（投毒）** | 一个 hallucination 进入窗口，之后被当作事实引用 | 模型误以为某个函数存在，后面一直基于它写代码 |
| **Distraction（分心）** | 历史太长，agent 开始模仿自己过去的动作，而不是思考新方案 | 一直重复同样的修复尝试 |
| **Confusion（混淆）** | 无关的工具和文档把答案带偏 | 加载了 60 个工具，模型选了不相关的那个 |
| **Clash（冲突）** | 新信息与旧信息矛盾，两者都留在窗口里没有解决 | 早期说“用 v1 API”，后来说“改用 v2”，模型混用 |
| **Drift（漂移）** | 真正的目标被埋没，被换成一个相近的子目标 | 本来要修 bug，最后在重构无关代码 |

---

## 4. Agent = Model × Harness

> **模型能力决定上限；harness 决定实际表现。**

**换 harness，准确率翻倍**（Lewis，“Same Model, Different Harness: Different Coding-Agent Results”，arXiv:2608.26218，2026）：

| Benchmark | 基础 harness | 改进后的 harness |
|---|---|---|
| SWE-bench Verified | 43 / 169 | **72 / 169** |
| SWE-bench Pro | 31 / 316 | **72 / 316** |

改了什么：
- 压缩旧的 tool 输出
- 保持最近的 observations 可见
- 检测重复和停滞
- 卡住时进行干预

**很多看起来像“模型失败”的问题，其实是 context 和控制循环的失败。** Agent loop、prompt、工具、context 管理、失败处理都很重要。

---

## 5. 选读：ReAct 之外

| 方法 | 循环方式 | 流程 |
|---|---|---|
| **ReAct** | 每一步都循环 | Thought → Act → Observe → Thought … |
| **Reflexion** | 失败之后循环 | Attempt → Evaluate → Critique（反思哪里错了）→ Retry |
| **Plan-and-execute** | 偏离计划时循环 | Plan → Execute steps → 出现偏差则 Re-plan |
| **Tree search** | 不循环，一次宽搜索 | 并行探索分支 A、B、C，只保留最好的一个 |
