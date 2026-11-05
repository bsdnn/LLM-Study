# Lecture 7 · Agent Design IV：Evaluation

> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/7_agent_design_4.pdf`](../slides/7_agent_design_4.pdf)

## 本讲要点

- Agent 评估分为 **component-level** 和 **end-to-end**（success rate、cost、latency）
- Benchmark 演进史：从 language modeling 到 agent 与真实工作流
- 指标：Perplexity、Accuracy/F1、win rate、**pass@k**、**pass^k**、cost、latency
- 代表性 agent benchmarks：GAIA、BFCL、WebArena、OSWorld、SWE-bench、τ-bench、Terminal-Bench、TheAgentCompany
- 挑战：**data leakage**、单轮指令、cost/latency、非确定性、安全副作用；benchmark 快速饱和

---

## 1. 评估 Agent 的不同方式

| Component-level（agent 内部） | End-to-end |
|---|---|
| Planning：task decomposition | **Success rate**：任务完成了吗？ |
| Memory：extraction、consolidation、retrieval | **Cost**：每个任务花费 |
| Tool use：discovery、selection、parameterization | **Latency**：完成任务的时间 |
| Error recovery | |

## 2. 简史

| 时期 | 代表 benchmark | 重点 |
|---|---|---|
| 2018 前 | Penn Treebank、WikiText、LAMBADA、SQuAD | Language modeling、窄任务 |
| 2018–2019 | GLUE、SuperGLUE、ARC、HellaSwag、WinoGrande | Transfer learning、通用 NLU |
| 2020–2022 | MMLU、BIG-bench、GSM8K、MATH、HumanEval、TruthfulQA、HELM | Foundation model 能力 |
| 2023–2024 | MT-Bench、Chatbot Arena、AlpacaEval、IFEval、LongBench、MMMU、GPQA、AgentBench、WebArena、GAIA、SWE-bench、OSWorld、τ-bench | Agents、tool use、context、多模态 |
| 2024–2025 | MMLU-Pro、LiveBench、LiveCodeBench、MMMU-Pro、FrontierMath、Humanity's Last Exam | 难度、robustness、新鲜度 |
| 2025–2026 | GDPval、BrowseComp、Terminal-Bench、ARC-AGI-3… | 更大 scope、更少具体指令、更真实场景 |

参考：<https://cs2680.com/notes/agent-evaluation>

### 评估目标

| 目标 | Benchmark | 成功的含义 |
|---|---|---|
| Predict text | PTB、WikiText | 给观测文本高概率 |
| Produce a correct answer | MMLU、GSM8K、GPQA、HLE | 匹配已知答案 |
| Produce a preferred response | Arena、AlpacaEval、MT-Bench | 获得人或模型的偏好 |
| Produce functioning code | HumanEval、LiveCodeBench、SWE-bench | 通过可执行测试 |
| Complete an interactive task | WebArena、OSWorld、τ-bench、Terminal-Bench | 达到要求的环境状态 |

## 3. Metrics

- 早期：**Perplexity**（held-out 文本上平均负对数似然的指数，越低越好）、Accuracy / F1、Preference / win rate
- 能力：
  - **pass@k**：k 次中至少一次通过
  - **pass^k**：k 次全部通过（τ-bench 引入，更严格，衡量一致性）
- Cost：主要是 LLM 成本；常被忽视的是 tool calls 和其他开销（如 LLM cache miss）
- Latency / duration：不只是 LLM

### Benchmark 评估循环

```python
for task in benchmark:
    environment = reset_to_task_initial_state(task)
    while not terminated and within_budget:
        observation = environment.observe()
        action = agent(task_instruction, observation, history, tools)
        environment.execute(action)
    score = task_evaluator(agent_final_answer, final_environment_state, trajectory)
```

---

## 4. 代表性 Agent Benchmarks

### GAIA（General AI Assistant）
- 466 道人写的问题，涉及 web browsing、spreadsheets、PDFs、图片、音频、计算；答案是短的数字/字符串/列表
- 不固定 tool API，每个 agent 自带 search、browser、Python、shell —— **测试整个 agent 系统而非仅 LLM**
- 难度：Level 1（≤ 5 步，0–1 个 tool）、Level 2（5–10 步，多个 tools）、Level 3（任意长）
- 打分：只看最终答案，`score = (1/N) Σ 1[normalize(ŷ) = normalize(y)]`
- 测量 reasoning、retrieval、tool selection、synthesis；**不测量 actions**（发邮件、订机票、改数据库…）

### BFCL（Berkeley Function Calling Leaderboard）
- 测 tool interface 是否可靠：调用是否正确、是否编造参数（如应该询问用户 `passenger_id` 却自己编了一个）、是否有无关调用（订机票时调用 `get_weather`）
- 打分：AST 或 execution 检查；多轮场景每轮检查 state + 回复
- V4 增加 web search、memory；**看分类得分而非综合分**

### WebArena
- 812 个任务，self-hosted 网站（shopping、admin、Reddit、GitLab、maps、Wikipedia）
- Observation：accessibility tree；Action：对 element ID click / type / stop
- 打分：reset snapshot → trajectory → **检查最终状态**（URL、页面内容、backend state、文字答案）
- **WebArena-Verified**：结构化输出，去掉 LLM-as-a-judge，对 HTTP traces 做确定性检查

### OSWorld
- 369 个任务，覆盖浏览器、office、文件、系统应用；可跨应用
- Observation：**截图（pixels）**；Action：鼠标、键盘
- 打分：evaluator 检查最终状态（单元格、公式、文件位置、设置、配置）；trajectory 不计分
- **OSWorld 2.0（2026）**：108 个长工作流（常超过 1 小时、数百步），约 27 个 checkpoint 给部分分

### SWE-bench
- 修复真实 repo 的 issue；agent：grep → 看实现 → 看测试 → 跑测试 → 改文件 → debug → 产出 patch
- 打分：patch 应用到 repo，跑测试，resolved or not
- 变体：**Verified**（人工检查）、**Lite**（300 子集）、**Multilingual**（9 种语言）、**Multimodal**（含图片的 issue）、**Pro**（更难，多文件）

### τ-bench
- Agent 要与**模拟用户**（LLM 扮演顾客）交互、遵守**策略**（如 basic economy 不能改签）、通过 API 改变数据库状态
- 打分：`r = r_action × r_output`
  - `r_action = 1`：最终数据库与标注目标状态一致（只读调用可以不同，**非预期写入导致失败**）
  - `r_output = 1`：面向用户的回复包含所需信息
- 引入 **pass^k**：同一场景 k 次独立运行全部成功的概率
- τ³-bench：修正任务、更多领域、knowledge retrieval、语音评估

### Terminal-Bench
- 在 Docker container 中通过 shell 完成端到端任务：编程、sysadmin、数据处理、模型训练、安全、科学工作流
- 打包：`instruction.md` + `Dockerfile` + `tests/test.sh`；对最终系统状态跑测试
- TB 2.0 89 题 → TB 4.0 66 题（剪掉饱和任务，8 小时超时）
- 介于 SWE-bench 和 OSWorld 之间：**没有 GUI grounding**，测长时程推理和系统交互

| | SWE-bench | Terminal-Bench | OSWorld |
|---|---|---|---|
| 动作 | 编辑一个 repo 的文件 | container 中的 shell 命令 | 鼠标点击和按键 |
| 观察 | 代码和测试输出 | 终端输出 | 截图 |
| 范围 | 修一个 GitHub issue | 配置、debug、修复、构建 | 操作任何桌面应用 |
| 打分 | repo 的测试 | 最终系统状态上的测试 | 最终状态检查 |

### TheAgentCompany
- 模拟公司：GitLab（代码）、Plane（工单）、ownCloud（文档）、RocketChat（聊天），以及 LLM 模拟的同事
- 175 个任务：软件、项目管理、数据科学、HR、财务
- **部分得分**：`S_partial = 0.5 · earned/total + 0.5 · S_full`（例：4/8 → 0.25；7/8 → 0.44；8/8 → 1.00）

### 总结

| Benchmark | 环境 | 典型动作 | 评什么 |
|---|---|---|---|
| GAIA | Web/文件/代码 tools | 搜索、看文件、计算 | 简短最终答案 |
| WebArena | 自托管网站 | 点击、输入、导航 | 网站或 backend 状态 |
| OSWorld | 完整桌面 VM | 鼠标、键盘 | 文件、应用、系统状态 |
| SWE-bench | 软件 repo | 查看、编辑、跑测试 | patch 通过测试 |
| BFCL | Function/API | 结构化 function calls | 调用、参数、结果状态 |
| τ-bench | 客服 API + 对话 | 与用户对话、调 API | 数据库状态与回复 |
| Terminal-Bench | 隔离的终端 container | 任意 shell 命令 | 任务特定的验证测试 |
| TheAgentCompany | 模拟软件公司 | Web 应用、代码、文件、与同事交流 | 工作流 checkpoint 与最终状态 |

---

## 5. Agent 评估的挑战

### Data Leakage 的四个层次

1. **Training leakage**：任务或人工修复出现在训练数据中
2. **Eval-time leakage**：agent 上网搜索并复制了“未来”的修复
3. **Environment leakage**：sandbox 暴露了未来的 commits 或隐藏测试
4. **Benchmark overfitting**：prompt 和 scaffold 针对公开测试集调优

Agent 评估需要真实 tools ⟺ 更大的 leakage 面（benchmark 可以被搜到）。

Agent 表现 ≈ `P = f(LLM, prompt, scaffold, tools, retriever, memory, environment)`，**每个组件都可能泄露**（pretraining 中的样例、用 benchmark 解法作为训练 trajectory、few-shot 例子来自测试、RAG 语料包含答案、memory 中存了以前的尝试、web 搜到解法、环境暴露未来 commit、手写 heuristics、研究者反复查看失败并调优）。所以“模型没在测试集上训练”远远不够。

### 单轮用户指令

- 大多数 benchmark 只给一个完全指定的指令，没有用户参与
- 真实工作是模糊的、会修改、会被打断（缺约束、中途纠正、目标变化）
- 单轮打分**奖励自信的猜测**：问澄清问题会消耗步数且不得分
- 应评估：**Clarification**（目标不清时会问吗）、**Recovery**（中途纠正后会重新规划吗）、**Memory**（会跨 turn 保持约束吗）

### 其他挑战

- **Cost & latency**：200 步 $40 vs 12 步 $0.30，只看 success rate 不够
- **Non-determinism**：同一 agent、同一任务结果不同，单次运行分数是噪声
- **Safety & side effects**：删除文件、发邮件、花钱在通过过程中不受惩罚
- **Metrics**：只看结果会让侥幸或不安全的运行通过；trajectory matching 会让合理的不同路径失败

### Benchmark 快速饱和

- GLUE（2018）14 个月达到人类水平；MMLU（2020）39 个月达到专家水平；GPQA Diamond（2023）10 个月超过 PhD 专家
- SWE-bench Verified：33% → 95% 用了 22 个月；ARC-AGI-2：11 个月达到人类平均；ARC-AGI-3：0.5% → 62.7% 用了 6 个月
- 已饱和：SQuAD、GLUE、SuperGLUE、MMLU、MATH、HumanEval、SWE-bench Verified、GPQA…；仍开放：FrontierMath、HLE、ARC-AGI-3、OSWorld 2.0、Terminal-Bench、GDPval、SWE-bench Pro、Remote Labor Index…

### 评估的演进

| 维度 | 早期 | 当前前沿 |
|---|---|---|
| 评估对象 | Base model | Model + prompt + tools + harness |
| 交互 | 静态、单轮 | 有状态、长时程 |
| Grader | Exact match 或 likelihood | 测试、环境状态、人、judge models |
| 数据集 | 固定、公开 | 新鲜、隐藏、动态、考虑污染 |
| 主要指标 | 平均准确率 | 成功率之外的系统效率、可靠性、质量、成本、延迟 |
| 主要关注 | 跨样本泛化 | 真实约束下的稳健完成、安全要求 |

### 真实 Agent 的多维度

Capability、Cost、Speed、Usability。**Pareto frontier**：如果 A 在所有指标上不差于 B，且至少一个指标更好，则 A dominates B；Rank 1 就是 Pareto frontier（即不被任何其他方案 dominate 的方案集合）。速度没有画进二维图，但同样重要。
