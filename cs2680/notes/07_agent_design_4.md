# Lecture 7 · Agent Design IV：Evaluation

> Harvard CS 2680：Modern AI Systems: Agents and System Optimizations（Juncheng Yang，Fall 2026）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/7_agent_design_4.pdf`](../slides/7_agent_design_4.pdf)
> 预计阅读时间：18–22 分钟。预备知识：Lecture 4（agent loop、tools）、Lecture 6（reliability）。

## 本讲脉络

```
怎么评估一个 agent：组件级 vs 端到端（成功率、成本、延迟）
→ 历史：benchmark 从“预测文本”一路演进到“在真实环境中完成任务”
→ 指标：perplexity、accuracy、win rate、pass@k、pass^k、成本、延迟
→ benchmark 评估循环：重置环境 → agent 交互 → 检查最终状态
→ 八个代表性 benchmark：GAIA、BFCL、WebArena、OSWorld、SWE-bench、τ-bench、Terminal-Bench、TheAgentCompany
→ 挑战：数据泄漏（四个层次）、单轮指令、成本与延迟、非确定性、副作用、指标设计
→ benchmark 饱和得越来越快 → 评估的演进方向 → 多维度与 Pareto 前沿
```

**一句话总结**：评估 agent 不能只看“答案对不对”，而要看它**在一个可重置的环境中，能否把环境改到正确的最终状态**，同时付出了多少**成本**和**时间**、是否**稳定**、有没有造成**副作用**。不同的 benchmark 考察不同的能力（检索推理、函数调用、网页操作、桌面操作、修代码、遵守业务规则、终端操作、完整的工作流）。由于 agent 能上网、能读环境，**数据泄漏**的途径比传统模型多得多；而 benchmark 也在以越来越快的速度被“刷满”。

---

## 1. 评估 agent 的两种方式

| Component-level evaluation（组件级） | End-to-end evaluation（端到端） |
|---|---|
| 看 agent **内部**的每个能力 | 看**任务整体**的结果 |
| **Planning**：任务分解是否合理 | **Success rate**：任务完成了吗？ |
| **Memory**：提取、整合、检索 | **Cost**：每个任务花多少钱 |
| **Tool use**：发现、选择、参数填写 | **Latency**：完成任务要多久 |
| **Error recovery**：出错后能否恢复 | |

组件级评估帮你**定位问题**（是规划差还是工具用错了），端到端评估告诉你**整体是否可用**。两者都需要。

---

## 2. 一段简史

| 时期 | 代表 benchmark | 关注点 |
|---|---|---|
| 2018 年以前 | Penn Treebank、WikiText、LAMBADA、SQuAD | 语言建模、窄任务 |
| 2018–2019 | GLUE、SuperGLUE、ARC、HellaSwag、WinoGrande | 迁移学习、通用的自然语言理解 |
| 2020–2022 | MMLU、BIG-bench、GSM8K、MATH、HumanEval、TruthfulQA、HELM | 基础模型的能力 |
| 2023–2024 | MT-Bench、Chatbot Arena、AlpacaEval、IFEval、LongBench、MMMU、GPQA、AgentBench、WebArena、GAIA、SWE-bench、OSWorld、τ-bench | agent 与工具使用、长 context、多模态 |
| 2024–2025 | MMLU-Pro、LiveBench、LiveCodeBench、MMMU-Pro、FrontierMath、Humanity's Last Exam | 难度、鲁棒性、题目新鲜度 |
| 2025–2026 | GDPval、BrowseComp、Terminal-Bench、ARC-AGI-3…… | 更大的任务范围、更少的具体指令、更真实的场景、更新的问题 |

（参考笔记：https://cs2680.com/notes/agent-evaluation）

### 评估目标的演进

| 评估目标 | 代表 benchmark | “成功”的含义 |
|---|---|---|
| 预测文本 | PTB、WikiText | 给真实出现的文本分配高概率 |
| 给出正确答案 | MMLU、GSM8K、GPQA、HLE | 与已知答案匹配 |
| 给出更受欢迎的回答 | Arena、AlpacaEval、MT-Bench | 得到人或模型的偏好 |
| 写出能运行的代码 | HumanEval、LiveCodeBench、SWE-bench | 通过可执行的测试 |
| **完成一个交互式任务** | WebArena、OSWorld、τ-bench、Terminal-Bench | **让环境达到要求的状态** |

趋势：从“说对”到“做对”。

---

## 3. 指标

**早期的 benchmark**：
- **Perplexity（困惑度）**：在留出的文本上，平均负对数似然的指数（越低越好）。直观上表示模型每一步平均在多少个词之间犹豫
- **Accuracy 或 F1**
- **Preference / win rate（偏好率 / 胜率）**：在两两比较中胜出的比例

**能力**：
- **pass@k**：k 个解中**至少一个**通过
- **pass^k**：k 个解**全部**通过（由 τ-bench 提出）

**算例**：一个 agent 每次独立地以 70% 的概率完成任务。
- pass@3 = $1 - 0.3^3 = 97.3\%$（试 3 次，至少成功一次）
- pass^3 = $0.7^3 = 34.3\%$（试 3 次，每次都成功）

两者差距巨大。pass@k 适合“能自动检验结果、可以多试几次”的场景；pass^k 衡量**可靠性** —— 真实部署的 agent 需要每次都做对。

**成本**：
- 主要是 LLM 调用的成本
- 常被忽视的：**工具调用**的成本，以及其他开销（例如 LLM 的 prompt cache 没命中时，输入 token 要按全价计费）

**延迟 / 时长**：
- **不只是 LLM 的时间**：工具执行（编译、跑测试、加载网页）往往占很大一部分

---

## 4. Benchmark 评估循环

```python
for task in benchmark:
    environment = reset_to_task_initial_state(task)        # 每个任务都从干净的初始状态开始
    while not terminated and within_budget:
        observation = environment.observe()                 # agent 看到环境的当前状态
        action = agent(task_instruction, observation, history, tools)
        environment.execute(action)                         # 动作真正改变环境
    score = task_evaluator(
        agent_final_answer,                                 # agent 给出的最终答案
        final_environment_state,                            # 环境的最终状态
        trajectory,                                         # 中间过程（有时也会检查）
    )
```

关键点：
- **可重置**：每个任务都从相同的初始状态开始，保证公平、可复现
- **有预算**：步数、时间或成本有上限
- **评分看结果**：大多数 agent benchmark 检查**最终答案**或**环境的最终状态**，而不是要求 agent 走某条特定的路径（因为完成任务的方式往往不止一种）

---

## 5. 八个代表性 benchmark

### 5.1 GAIA（General AI Assistant）：能否调研并给出正确答案？

- **466 道**人工编写的问题，需要网页浏览、读表格、读 PDF、看图片、听音频、做计算
- 答案很**短**：一个数字、一个字符串或一个列表（如 `17.4` 或 `4,381`）
- **不规定工具 API**：每个 agent 自带搜索、浏览器、Python、shell —— **测的是整个 agent 系统，而不只是 LLM**
- 难度分三级：

| 级别 | 步数 | 工具 |
|---|---|---|
| Level 1 | ≤ 5 步 | 0–1 个工具 |
| Level 2 | 5–10 步 | 多个工具 |
| Level 3 | 任意长 | 很多工具 |

**例题**：附件表格中有一份电影列表，找出满足两个条件的那部电影，查到它的导演，报出导演的出生地。

Agent 的过程可能是：读附件 → 搜索 → 打开网页 → 下载 PDF → 提取表格 → 运行 Python 计算 → 回答。**中间过程全部被评估器丢弃**，只有最终答案计分：

$$
\text{score} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}\big[\text{normalize}(\hat y_i) = \text{normalize}(y_i)\big]
$$

（normalize：统一大小写、去掉标点和单位等，避免格式差异导致误判。）

- ✅ 测量：推理、检索、工具选择和信息综合，而且评分干净（精确匹配）
- ❌ 不测量**动作**：发邮件、订机票、修改数据库、提交表单、更新 issue —— 只看最后给出的答案
- 参考：GAIA 发布时，人类答对约 92%，而当时配备插件的 GPT-4 只有约 15%

### 5.2 BFCL（Berkeley Function Calling Leaderboard）：模型的工具接口可靠吗？

**例子**：用户说 “Book the cheapest flight from Boston to Chicago next Monday.”，可用工具有 `search_flights(...)`、`book_flight(flight_id, passenger_id)`、`get_weather(...)`。

| 调用 | 评价 |
|---|---|
| ✅ `search_flights(origin="BOS", destination="CHI", date=...)` → 选出最便宜的 → `book_flight(flight_id=..., passenger_id=...)` | 正确：先搜索，再选最便宜的，再预订 |
| ❌ 自己编造一个 `passenger_id` | 应该**询问用户**，而不是编造参数 |
| ❌ 调用 `get_weather(...)` | 无关调用，订机票不需要 |

- **评分**：AST 检查（把调用解析成语法树，比较函数名和参数是否正确）或真正执行后检查结果；多轮场景下，每一轮都检查状态和回复
- V4 版本增加了网页搜索和 memory 等类别；**看分类得分，而不是只看综合分**（不同模型的强项不同）

### 5.3 WebArena：能完成一个网页工作流吗？

**例题**：在 GitLab 中创建一个标题为 “Update onboarding documentation” 的 issue，指派给 Alex，并加上 documentation 标签。

- **环境**：812 个任务，**自托管**的网站（购物网站、后台管理、Reddit 类论坛、GitLab、地图、Wikipedia）—— 每次都能重置成相同的初始状态，避免真实网站变化带来的干扰
- **观察**：网页的 **accessibility tree**（一种把页面元素组织成树的文本表示，每个元素有 ID）
- **动作**：对元素 ID 执行 click、type、stop
- **过程**：读 accessibility tree → `click [Projects]` → `click [project]` → 输入 issue 标题 → …… → `click [Create issue]` → `stop`
- **评分**：重置快照 → agent 产生轨迹 → 检查最终状态 → 成功 1 / 失败 0。上面的例题会检查：issue 是否存在、指派人和标签是否正确。其他任务可能检查最终的 URL、页面内容、后端状态或文字答案
- **WebArena-Verified**：要求结构化输出，去掉了 LLM-as-a-judge，改为对 HTTP 请求记录做确定性的检查，评分更可靠
- 参考：WebArena 发布时，最好的 GPT-4 agent 成功率约 14%，人类约 78%

### 5.4 OSWorld：能操作一台真正的电脑吗？

**例题**：在表格中添加一个计算列，排序，另存为 `result.xlsx`。

- **环境**：完整的桌面虚拟机，369 个任务，覆盖浏览器、办公软件、文件管理、系统应用，还有跨应用的工作流
- **观察**：**屏幕截图（像素）**
- **动作**：鼠标点击、键盘输入
- **过程**：截图 → 点击打开 LibreOffice → 截图 → 点击单元格 → 输入公式 → …… → 保存文件
- **评分**：每个任务有专门的初始化逻辑和评估函数，检查最终状态（单元格内容、公式、文件位置、系统设置、应用配置）；**中间过程不计分**
- **OSWorld 2.0（2026）**：108 个长工作流（常常超过 1 小时、数百步），每个约 27 个 checkpoint，可以给部分分
- 参考：OSWorld 发布时，最好的模型成功率约 12%，人类约 72%

### 5.5 SWE-bench：能修复真实代码仓库中的 issue 吗？

**例题**：用空索引调用某个库函数时会错误地抛出异常。要求修改为返回空结果，同时保持其他行为不变。

- **过程**：grep 仓库 → 查看实现 → 查看测试 → 运行测试 → 修改文件 → 再运行测试 → debug → 生成 patch
- **评分**：把 agent 的 patch 应用到仓库 → 运行测试（包括针对这个 issue 新增的测试，以及原有测试，确保没有破坏其他功能）→ resolved 或 not resolved

| 变体 | 特点 |
|---|---|
| **Verified** | 人工检查过题目描述清楚、测试合理 |
| **Lite** | 300 题的子集，便宜快速 |
| **Multilingual** | 9 种编程语言 |
| **Multimodal** | issue 中包含图片（如 UI 截图） |
| **Pro** | 更难，需要修改多个文件 |

### 5.6 τ-bench：能与用户交互、遵守业务规则、正确修改后端状态吗？

**两个交互的参与者 + 一套业务规则**：
- **Simulated user（模拟用户）**：一个扮演顾客的 LLM
- **Agent**：必须遵守航空公司的规则（policy）
- **Tools + 航空公司数据库**：API 调用会真正改变数据库

**例题**：“我想把明天的航班改成晚上那班。”

```
确认身份 → 查找预订 → 搜索航班 → 解释票价差额 → 获得用户确认 → change_booking(…) → 预订已更新
```

同时必须遵守规则，例如“basic economy（基础经济舱）不能改签” —— 如果用户的票是基础经济舱，正确的做法是拒绝改签并解释原因。

**评分**：原始的 reward 是

$$
r = r_{\text{action}} \times r_{\text{output}}
$$

- $r_{\text{action}} = 1$：只有当最终数据库状态与标注的目标状态**完全一致**时
- $r_{\text{output}} = 1$：只有当面向用户的回复包含了所需信息时
- 只读的调用可以随意（查多少次都行），但**任何非预期的写操作都会导致失败**

**pass^k**：同一个场景独立运行 k 次，**全部成功**的概率。比 pass@k（一次成功即可）严格得多。τ-bench 发现，即使单次成功率看起来不错的 agent，pass^k 也会随着 k 增大而迅速下降 —— 说明它们**不够稳定**。

最新的 τ³-bench 修正了一些任务，增加了更多领域、知识检索和语音交互的评估。

### 5.7 Terminal-Bench：能通过 shell 完成真实任务吗？

**任务领域**：端到端的工作，而不只是修 bug —— 编程、系统管理与服务器配置、数据处理、模型训练、安全、科学计算工作流。

**任务的打包与评分**：

```
instruction.md（任务说明）→ Docker 容器（Dockerfile）→ agent 通过 shell 操作 → 最终系统状态 → tests/test.sh：通过或失败
```

（agent 的操作轨迹不计分，只用测试检查最终的系统状态。）

- TB 2.0 有 89 个任务，TB 4.0 有 66 个任务（去掉了已被“刷满”的任务）；人类的参考解法是隐藏的；每个任务最长 8 小时
- 代表性任务：在一个正在运行的 KV 存储服务上切换到新的协议，要求**不断开任何连接**，并保持延迟

**介于 SWE-bench 和 OSWorld 之间**：

| | SWE-bench | Terminal-Bench | OSWorld |
|---|---|---|---|
| Agent 的动作 | 编辑一个仓库中的文件 | 在容器中执行 shell 命令 | 鼠标点击和按键 |
| Agent 的观察 | 代码和测试输出 | 终端输出 | 屏幕截图 |
| 任务范围 | 修一个 GitHub issue | 配置、调试、修复、构建 | 操作任何桌面应用 |
| 评分 | 仓库自带的测试 | 对最终系统状态的测试 | 对最终状态的检查 |

Terminal-Bench **不需要“看图找按钮”**（GUI grounding），专注考察长程推理和与系统的交互，而不用预测鼠标坐标。

### 5.8 TheAgentCompany：模拟一家公司

**公司环境**：
- **GitLab**：代码
- **Plane**：工单
- **ownCloud**：文档
- **RocketChat**：团队聊天
- **同事**：由 LLM 模拟

**一份完整的工作**：

```
读工单（Plane）→ 问同事（RocketChat）→ 查看代码库（GitLab）→ 写代码（工作区）→ 跑测试（终端）
→ 更新 issue（Plane）→ 汇报结果（RocketChat）
```

- 175 个任务，覆盖软件开发、项目管理、数据科学、HR、财务
- **部分得分**：每个任务有若干 checkpoint，按完成情况给分：

$$
S_{\text{partial}} = 0.5 \cdot \frac{\text{earned}}{\text{total}} + 0.5 \cdot S_{\text{full}}
$$

| 进度 | 计算 | 得分 |
|---|---|---|
| 8 个 checkpoint 完成 4 个，未完成 | 0.5 × 4/8 + 0 | 0.25 |
| 完成 7 个，未完成 | 0.5 × 7/8 + 0 | 0.44 |
| 全部 8 个，完成 | 0.5 × 1 + 0.5 × 1 | 1.00 |

设计意图：做了大部分工作值得一些分数，但**真正完成**才拿到另一半。

### 5.9 八个 benchmark 对比

| Benchmark | Agent 在哪里操作 | 典型动作 | 评什么 |
|---|---|---|---|
| GAIA | 网页 / 文件 / 代码工具 | 搜索、查看文件、计算 | 简短的最终答案 |
| WebArena | 自托管网站 | 点击、输入、导航 | 网站或后端状态 |
| OSWorld | 完整的桌面虚拟机 | 鼠标、键盘、应用操作 | 文件、应用、系统状态 |
| SWE-bench | 软件仓库 | 查看、编辑、运行测试 | patch 是否通过测试 |
| BFCL | 函数 / API 接口 | 结构化的函数调用 | 调用、参数及结果状态 |
| τ-bench | 客服 API 与对话 | 与用户对话、调用 API | 数据库状态与回复 |
| Terminal-Bench | 隔离的终端容器 | 任意 shell 命令 | 任务特定的验证测试 |
| TheAgentCompany | 模拟的软件公司 | 使用 Web 应用、代码、文件，与同事交流 | 工作流 checkpoint 与最终系统状态 |

---

## 6. Agent 评估的挑战

### 6.1 数据泄漏（Data Leakage）：四个层次

| 层次 | 怎么泄漏的 |
|---|---|
| **Training leakage（训练泄漏）** | 题目或人类给出的修复方案出现在训练数据中 |
| **Eval-time leakage（评估时泄漏）** | agent 在评估时上网搜索，**直接抄到了“未来”的修复方案**（例如 SWE-bench 的题目对应的 PR 早已公开） |
| **Environment leakage（环境泄漏）** | sandbox 中暴露了未来的 commit 或隐藏的测试（agent 用 `git log` 就能看到答案） |
| **Benchmark overfitting（对 benchmark 过拟合）** | prompt 和 scaffold 针对公开测试集反复调优 |

一个根本矛盾：agent 评估**需要**真实的工具（上网、读文件系统），而真实的工具又意味着**更大的泄漏面** —— benchmark 本身可能就是能被搜到的。

**泄漏可能发生在整个 agent 技术栈中**。现代 agent 的表现大致是

$$
P = f(\text{LLM},\ \text{prompt},\ \text{scaffold},\ \text{tools},\ \text{retriever},\ \text{memory},\ \text{environment})
$$

| 组件 | 可能的泄漏 |
|---|---|
| Base model | 预训练数据中包含 benchmark 的例子 |
| Fine-tuning | 用 benchmark 的解法作为 agent 的训练轨迹 |
| Prompt | few-shot 例子来自测试题 |
| Retrieval | RAG 语料中包含 benchmark 的答案 |
| Memory | 存了之前尝试同一 benchmark 的经验 |
| Web search | 搜到了已公开的解法 |
| Environment | 暴露了未来的 Git commit 或隐藏文件 |
| Scaffold | 针对 benchmark 手写的启发式规则 |
| Agent development | 研究者反复查看测试失败的案例，并据此调整系统 |

**任何一个组件都可能包含 benchmark 的信息。所以只说“模型没有在测试集上训练过”，对 agent 评估来说远远不够。**

### 6.2 单轮用户指令

- 大多数 benchmark 只给 agent **一条完整、明确的指令**：prompt 进去，最终状态出来，**没有用户参与**（这也是你在作业中使用 Claude Code 的方式）
- 但真实的工作是**模糊的、会修改、会被打断的**：缺少约束、中途纠正、目标变化
- **单轮评分奖励“自信地猜”**：问一个澄清问题要多花步数，却拿不到任何分数

应该评估的能力：

| 能力 | 问题 |
|---|---|
| **Clarification（澄清）** | 目标说得不清楚时，它会问吗？ |
| **Recovery（恢复）** | 任务中途被纠正后，它会重新规划吗？ |
| **Memory（记忆）** | 它能跨多轮记住约束吗？ |

### 6.3 其他挑战

| 挑战 | 说明 |
|---|---|
| **Cost and latency（成本与延迟）** | 只看成功率不够：一个 200 步、花 $40 才成功的运行，和一个 12 步、花 $0.30 成功的运行，价值完全不同 |
| **Non-determinism（非确定性）** | 同一个 agent、同一个任务，结果可能不同；**单次运行的分数是噪声，不是信号**。应多次运行，报告平均值和置信区间 |
| **Safety and side effects（安全与副作用）** | 在通往“通过”的路上删了文件、发了邮件、花了钱，**都不会被扣分** |
| **Metrics（指标设计）** | 只看结果的评分会让**侥幸或不安全的运行**通过；而“轨迹匹配”（要求按标准路径做）又会让**合理但不同的做法**失败 |

（讲者的研究组正在构建一个基于真实流量的 agent 评估研究平台。）

---

## 7. Benchmark 饱和得越来越快

| Benchmark | 提出年份 | 被追上的速度 |
|---|---|---|
| GLUE | 2018 | 14 个月达到人类水平（87.1） |
| MMLU | 2020 | 39 个月达到专家水平（89.8） |
| GPQA Diamond | 2023 | 10 个月超过博士专家（69.7） |
| SWE-bench Verified | 2024 | 22 个月从 33% 升到 95% |
| ARC-AGI-2 | 2025 | 11 个月达到人类平均水平（60） |
| ARC-AGI-3 | 2026 | 6 个月从 0.5% 升到 62.7% |

**已经饱和的**（前沿模型达到人类基线，或顶尖模型之间已经拉不开差距）：SQuAD 1.1、GLUE、SuperGLUE、ARC-AGI-1、MMLU、MATH、HumanEval、MT-Bench、SWE-bench Verified、GPQA Diamond……
**仍然开放的**：FrontierMath、Humanity's Last Exam、ARC-AGI-3、OSWorld 2.0、Terminal-Bench、GDPval、SWE-bench Pro、Remote Labor Index……

---

## 8. Agent 评估仍在演进

| 维度 | 早期的评估 | 当前前沿 |
|---|---|---|
| 评估对象 | Base model | 模型 + prompt + 工具 + agent harness |
| 交互 | 静态、单轮 | 有状态、长时程 |
| Grader（评分者） | 精确匹配或似然 | 测试、环境状态、人或 judge 模型 |
| 数据集 | 固定、公开 | 新鲜、隐藏、动态，或考虑了污染 |
| 主要指标 | 平均准确率 | 除成功率之外的系统效率：可靠性、质量、成本、延迟 |
| 主要关注 | 跨样本的泛化 | 在真实约束和安全要求下稳健地完成任务 |

### 真实 agent 需要考虑的更多维度

**Capability（能力）、Usability（易用性）、Cost（成本）、Speed（速度）**。

**Pareto dominance（帕累托支配）**：如果 A 在**所有**指标上都不比 B 差，并且**至少一个**指标更好，就说 A dominates（支配）B。

例：三个 agent

| Agent | 能力（越高越好） | 成本（越低越好） |
|---|---|---|
| A | 80% | $2 |
| B | 75% | $3 |
| C | 90% | $5 |

- A 支配 B（能力更高、成本更低）→ B 不在前沿上
- A 和 C 互不支配（C 能力更高但更贵）→ A、C 都在 **Pareto frontier（第一名的那一层）** 上
- 选 A 还是 C，取决于你更看重能力还是成本

（作业 3 会同时从多个指标评估。）

---

## 9. 要点回顾

1. 评估分组件级（规划、记忆、工具、纠错）和端到端（成功率、成本、延迟），两者互补
2. Benchmark 从“预测文本、答对题”演进到“在可重置环境中完成交互任务，并检查最终状态”
3. pass@k（至少一次成功）衡量能力上限，pass^k（每次都成功）衡量可靠性；成本要算上工具和 cache miss，延迟要算上工具执行
4. 八个 benchmark 各测一类能力：GAIA（调研回答）、BFCL（函数调用）、WebArena（网页）、OSWorld（桌面）、SWE-bench（修代码）、τ-bench（对话 + 规则 + 数据库）、Terminal-Bench（shell）、TheAgentCompany（完整工作流，部分得分）
5. 数据泄漏有训练、评估时、环境、过拟合四个层次，可能出现在 agent 栈的任何组件中
6. 单轮评估奖励自信地猜，忽视澄清、恢复和记忆；还要考虑成本、非确定性、副作用，以及“只看结果”与“轨迹匹配”之间的指标权衡
7. Benchmark 饱和越来越快；评估正走向“系统级、长时程、动态数据、多指标”
8. 用 Pareto 支配比较多维度的 agent

---

## 10. 自测题

<details>
<summary>1. 为什么 GAIA 测的是“整个 agent 系统”而不只是 LLM？它测不到什么？</summary>

GAIA 不规定工具 API，每个 agent 自带搜索、浏览器、代码执行等工具，最终答案是整个系统（模型 + 工具 + scaffold）共同作用的结果。它只评最终答案，测不到发邮件、修改数据库、提交表单这类“动作”。
</details>

<details>
<summary>2. τ-bench 中 agent 查询了三次数据库，并正确完成了改签。如果它还多执行了一次写操作，结果如何？</summary>

失败。只读调用可以随意，但任何非预期的写操作都会使最终数据库状态与目标状态不一致，r_action = 0，整体 reward = 0。
</details>

<details>
<summary>3. 一个 agent 单次成功率 90%，各次独立。pass^5 是多少？这说明什么？</summary>

0.9⁵ ≈ 0.59。单次 90% 看起来很好，但连续 5 次都成功只有约 59%，说明部署时用户会经常遇到失败，可靠性比单次成功率更能反映真实体验。
</details>

<details>
<summary>4. 举出 SWE-bench 中可能发生的“评估时泄漏”和“环境泄漏”的例子。</summary>

评估时泄漏：agent 上网搜索 issue 编号，找到了已合并的修复 PR 并照抄。环境泄漏：sandbox 中的 Git 仓库包含了修复之后的 commit，agent 用 git log 就能看到答案。
</details>

<details>
<summary>5. TheAgentCompany 中完成 6/8 个 checkpoint 但任务没完成，得分是多少？</summary>

0.5 × 6/8 + 0.5 × 0 = 0.375。
</details>

<details>
<summary>6. 为什么说单轮评估“奖励自信地猜”？</summary>

单轮评估只给一条指令、只看最终结果，agent 问澄清问题会多花步数却得不到任何分数；而直接按某种猜测去做，猜对了就能得分。这与真实工作中“不确定时应该先问”的期望相反。
</details>

<details>
<summary>7. 只看结果的评分和轨迹匹配评分各有什么问题？</summary>

只看结果：可能让侥幸成功或者过程中做了危险操作（删文件、乱花钱）的运行也通过。轨迹匹配：要求按标准路径完成，会让合理但走了不同路线的运行被判失败。
</details>
