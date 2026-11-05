# Lecture 8 · LLM 评估：人工评分、规则指标、LLM-as-a-Judge 与 Benchmarks

> Stanford CME 295：Transformers & Large Language Models
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture8.pdf`](../slides/fall25-cme295-lecture8.pdf)

## 本讲脉络

```
“评估”指什么 → 人工评分（inter-rater agreement、Cohen's Kappa）
→ 规则指标（BLEU、ROUGE、METEOR）及其局限
→ LLM-as-a-Judge（做法、结构化输出、pointwise/pairwise、三种 bias、最佳实践）
→ 扩展：factuality 评估、agent 的失败模式
→ 常用 benchmarks（知识、推理、代码、安全、agent）→ Pareto frontier、数据污染、Goodhart's Law
```

一句话总结：LLM 输出的是自由文本，没有唯一正确答案，所以评估很难。**人工评分最可靠但又慢又贵；规则指标便宜但与人类判断相关性差；LLM-as-a-Judge 是目前的主力，但要注意它的偏差并用人工标注校准。** Benchmark 只是某个维度上的投影，不要过度迷信。

---

## 0. 回顾

| 普通 LLM 的短处 | 对应讲次 |
|---|---|
| 推理能力有限 | Lecture 6 |
| 知识是静态的、不能执行动作 | Lecture 7（RAG、tool calling、agents） |
| **难以评估** | **本讲** |

---

## 1. “评估”指什么

两大类：

| ✅ Output quality（输出质量） | ⚡ System performance（系统性能） |
|---|---|
| Instruction following（是否遵循指令） | Latency（延迟） |
| Coherence（连贯性） | Pricing（成本） |
| Factuality（事实性） | Reliability（可靠性） |
| 等等 | 等等 |

本讲主要讨论输出质量。难点在于：**LLM 生成的是自由文本，很难自动判断好坏。**

---

## 2. 人工评分（Human Rating）

### 2.1 理想情况

让人给 LLM 的输出打分 —— **最接近真相，是“金标准”**。

### 2.2 局限一：主观性

例：
- Prompt：“What birthday gift should I get?”
- 回答：“A teddy bear is almost always a sweet gift—just pick one that feels right to you.”
- 评价标准：**Usefulness**（有用性）

Rater 1 可能觉得“有用”（给了具体建议），Rater 2 可能觉得“没用”（没考虑送给谁、预算多少）。同一个回答，不同的人判断不同。

### 2.3 衡量评分者之间的一致性：Cohen's Kappa

**直接算“两人一致的比例”不够**：如果两个人都是随机乱打分，也会有一定比例的一致。我们想知道的是：

> 我们的一致程度，比“仅凭运气”（考虑到各评分者实际使用各类别的频率）预期的一致程度好多少？

```
κ = (p_o − p_e) / (1 − p_e)
```

- **p_o**（observed）：实际观察到的一致比例
- **p_e**（expected）：随机情况下预期的一致比例 = Σ_k （评分者 1 选类别 k 的比例 × 评分者 2 选类别 k 的比例）
- κ = 1：完全一致；κ = 0：和随机一样；κ < 0：比随机还差

**算例**：两人给 10 个回答打“有用 / 没用”。
- Rater 1 判 6 个有用，Rater 2 判 5 个有用；两人有 7 个判断一致（4 个都说有用，3 个都说没用）
- p_o = 7 / 10 = 0.7
- p_e = 0.6 × 0.5（都说有用）+ 0.4 × 0.5（都说没用）= 0.5
- κ = (0.7 − 0.5) / (1 − 0.5) = **0.4**

70% 的一致率看起来不错，但扣掉运气成分后只是“一般”（常用经验尺度：0.21–0.40 fair，0.41–0.60 moderate，0.61–0.80 substantial，0.81 以上 almost perfect）。

**变体**：
- **Cohen's Kappa**（Cohen, 1960）：两个评分者
- **Fleiss' Kappa**（Fleiss, 1971）：多个评分者
- **Krippendorff's alpha**：更通用，支持多评分者、缺失值、不同尺度（名义、序数、区间）

### 2.4 局限二：慢、贵

每次模型更新都要重新找人标注，无法快速迭代。

---

## 3. 规则指标（Rule-Based Metrics）

### 3.1 思路

**人工写一次参考答案（labels / references），以后都用规则指标把模型输出与参考答案比较。**

### 3.2 常见指标

**BLEU**（BiLingual Evaluation Understudy，Papineni et al., 2002）—— 偏 precision，常用于翻译

```
BLEU = BP · exp( Σ_{n=1..N} w_n · log p_n )
```

- p_n：**modified n-gram precision** —— 生成文本中的 n-gram 有多少出现在参考文本里（每个 n-gram 的计数会被“截断”到它在参考中出现的次数，防止重复同一个词刷分）
- 通常 N = 4，w_n = 1/4
- **BP（brevity penalty）**：生成文本比参考短时惩罚：c > r 时 BP = 1，否则 BP = exp(1 − r/c)（c = 生成长度，r = 参考长度）。否则只输出一两个确定的词就能拿到很高的 precision

**ROUGE**（Recall-Oriented Understudy for Gisting Evaluation，Lin, 2004）—— 偏 recall，常用于摘要

- **ROUGE-N** = 参考文本中被生成文本覆盖的 n-gram 数 / 参考文本中的 n-gram 总数
- **ROUGE-L**：基于**最长公共子序列（LCS）**，不要求连续，能反映句子层面的词序
- 有一整套变体指标

**METEOR**（Metric for Evaluation of Translation with Explicit ORdering，Banerjee et al., 2005）—— 改进与人类判断的相关性

- 匹配 unigram 时不仅看完全相同，还允许**词干相同**（reading / read）和**同义词**
- 综合 precision 和 recall，且 recall 权重更高：F_mean = 10·P·R / (R + 9·P)
- 加入**词序惩罚**：把匹配的词分成尽量少的连续“块”（chunks），块越碎说明词序越乱：Penalty = 0.5 · (chunks / matches)³
- METEOR = F_mean · (1 − Penalty)

### 3.3 规则指标的局限

**不考虑风格上的变化**。下面三句意思几乎相同，但词几乎不重叠，BLEU / ROUGE 分数会很低：
- “A plush teddy bear can comfort a child during bedtime.”
- “Soft stuffed bears often help kids feel safe as they fall asleep.”
- “Many youngsters rest more easily at night when they cuddle a gentle toy companion.”

此外：
- **与人类评分的相关性不太好**
- **仍然需要人工写参考答案**

---

## 4. LLM-as-a-Judge（LaaJ）

### 4.1 思路（Zheng et al., 2023）

**用 LLM 来评价回答的质量。**

```
Prompt：“Why are teddy bears comforting?”
Response：“They are comforting because they are soft and loyal companions.”
Criterion：Relevance
        ↓  ⚖ LLM-as-a-Judge
🎯 Score：PASS
💭 Rationale：Direct explanation, is on topic
```

Zheng et al. 发现，强 LLM（如 GPT-4）作为 judge 与人类偏好的一致率超过 80%，与人类之间的一致率相当。

**Judge prompt 示例**：

```
Evaluate how relevant the model's answer is to the user's prompt.
Prompt: {prompt}
Model Response: {model_response}
Return:
- Rationale (1–2 sentences)
- Score: 1 if mostly relevant, 0 if mostly irrelevant.
```

### 4.2 实践中强制输出格式

回顾 Lecture 3 的 guided decoding。各家 API（Gemini、OpenAI、Anthropic）都支持 **structured outputs**：

```python
# Step 1：定义期望的输出结构
class Response(BaseModel):
    rationale: str
    score: Literal[0, 1]

# Step 2：在模型调用中传入
response = client.responses.parse(
    model=model,
    input=input,
    text_format=Response,
)
```

这样 judge 的输出一定能被程序解析，便于大规模统计。

### 4.3 相比之前方法的好处

- **不需要参考答案 / 标签**
- **可解释**：rationale 说明了为什么给这个分。例：“The response does not contain any grammatical mistakes. It also is clear, easy to understand and is helpful with respect to the task.”

### 4.4 两种主要形式

| | Pointwise | Pairwise |
|---|---|---|
| 问法 | 评价这个回答的质量 | A 和 B 哪个更好？ |
| 输出 | 一个分数，如 “Very good” | 选择，如 “Response A” |
| 适合 | 绝对质量监控 | 比较两个模型或版本（更稳定，人和 LLM 都更擅长比较） |

### 4.5 三种常见 bias

**1）Position bias（位置偏差）**
- 症状：pairwise 时 judge 偏好某个位置（如总是选第一个）。把 A、B 交换顺序后，判断也跟着变
- 补救：**两种顺序各评一次取平均**（或要求两次结果一致才算数）；研究上也有调整 position embedding 的方法

**2）Verbosity bias（冗长偏差）**
- 症状：偏好更长的回答。“This response is short and correct.” 可能输给 “This response goes into details, a lot of which are not needed and is overall not very helpful.”
- 补救：明确的评分准则、few-shot 示例，以及/或对输出长度加惩罚

**3）Self-enhancement bias（自我偏好）**
- 症状：judge 偏好**自己生成**的回答，甚至胜过人工精心撰写、完美回答问题的答案
- 补救：**不要用同一个模型既当生成者又当 judge**

### 4.6 最佳实践

- **清晰明确的评分准则**（crisp guidelines）
- **二元尺度优于细粒度尺度**（pass/fail 比 1–10 分更稳定、更一致）
- **先写 rationale 再给分**（相当于 CoT，分数更有依据）
- **缓解各种 bias**（position、verbosity、self-enhancement 等）
- **用人工判断校准**：在一小批人工标注数据上检查 judge 与人的一致性
- **低 temperature**，保证可复现

### 4.7 修订后的评估工作流

```
🐢 人工标注一小批样本（慢，但是金标准）
   → 用它校准 LLM-as-a-Judge（调 prompt 直到与人的一致性足够高）
   → 🐇 用 LaaJ 大规模、快速地评估
```

### 4.8 通常要评估的维度

| 🧠 Task performance | 🗣 Alignment |
|---|---|
| 有用性、相关性、正确性、事实性等 | Tone（语气）、Style（风格）、Safety（安全） |

---

## 5. 扩展：Factuality（事实性）评估

**目标**：量化一段输出的事实性。

例：“Teddy bears, first created in the 1920s, were named after President Theodore Roosevelt after he proudly wanted to shoot a captured bear on a hunting trip.”

这段话**部分对、部分错**（teddy bear 诞生于 1902–1903 年；Roosevelt 是**拒绝**射杀那只被拴住的熊）。开放问题：如何量化这种细微差别？

**做法：Fact decomposition（事实分解）**（参考 Wei et al., 2024，“Long-form factuality in large language models”，其中的 SAFE 方法会对每个事实用搜索引擎核查）

1. 把原文拆成独立的原子事实
2. 逐条核查真假
3. 按重要性加权汇总：score = Σ wᵢ · 1[事实 i 正确] / Σ wᵢ

| 原子事实 | 重要性 wᵢ | 是否正确 |
|---|---|---|
| Teddy bears were first created in the 1920s. | 0.3 | ❌ |
| Teddy bears were named after President Theodore Roosevelt. | 0.4 | ✅ |
| Theodore Roosevelt was on a hunting trip where a bear was captured. | 0.2 | ✅ |
| Theodore Roosevelt proudly wanted to shoot the captured bear. | 0.1 | ❌ |

score = (0.4 + 0.2) / 1.0 = **0.60**

---

## 6. 扩展：Agent 的评估与失败模式

### 6.1 Agentic 视角

回顾 ReAct：Observe → Plan → Act 循环。一次典型的 agentic 调用会经历**多轮循环**，每轮都包含 Lecture 7 的 tool 三步：

1. LLM 为相关函数推断参数：“Find a bear near me!” → `find_teddy_bear(location=(37.42, -122.17))`
2. 执行函数调用：后端返回 `{"name": "Teddy", …}`
3. LLM 根据结果推出结论，生成回复

每一步都可能出错。

### 6.2 失败类型 1：Tool prediction error（工具预测错误）

| 症状 | 可能原因 | 补救 |
|---|---|---|
| **不使用工具**：直接回复 “Sorry, I don't know where I can find one.” | Tool router 出错（没把该工具选进来） | 重新训练 tool router |
| | 模型不知道怎么用这个工具 | 对模型做 SFT，或改进该 API 对应的 prompt 说明 |
| **幻想出不存在的工具**：调用 `find_bear()` | 模型太弱 | 升级模型 |
| | API 命名不合逻辑 | 重新设计 API |
| | 指令不清楚 | 迭代顶层指令 |
| **用错工具**：调用 `send_message(recipient=LocalBusiness, message="may I buy a teddy bear?")` | 模型选择了错误的工具 | 同上（升级模型、改进 API 命名和指令） |
| **参数推断错误**：`location = (0, -0)` | 参数无法从上下文推断（模型根本不知道用户位置） | 引入辅助工具（如获取用户位置的 tool），以及/或确保上下文中有所需信息 |

### 6.3 失败类型 2：Tool call error（工具调用错误）

| 症状 | 原因 | 补救 |
|---|---|---|
| **返回错误结果**：返回不正确的值，或抛出异常（如 `ValueError: could not...`） | 工具实现有 bug（注意：有时报错是合理的，但不是标准做法） | **修复工具实现！** |
| **没有返回**：工具什么也不返回，常导致最终回复是**幻觉编造**的 | bug，或工具实现的“特性” | 总要返回点东西，哪怕是空 JSON；一般原则：输出有意义的工具结果 |

### 6.4 失败类型 3：Response generation error（回复生成错误）

| 症状 | 原因 | 补救 |
|---|---|---|
| 最终回复没有正确传达工具的结果：工具找到了熊，回复却说 “Didn't find any bear!” | 模型缺乏 **grounding** 能力（不能忠实依据给定信息回答） | 升级负责综合回复的 LLM |
| | 工具返回的内容太多，塞满了 context window | 精简后端返回的信息 |
| | 工具输出的格式不能有意义地传达信息 | 让工具输出格式更具描述性 |

### 6.5 总结

| 阶段 | 失败 |
|---|---|
| Tool prediction | 不用工具 / 幻想工具 / 用错工具 / 参数错误 |
| Tool call | 结果错误 / 没有结果 |
| Response generation | 回复错误 |

问题来源归纳：
- **建模方面**：推理和 grounding 能力弱；context window 中东西太多；工具的建模（命名、描述、粒度）不对
- **工具方面**：工具本身有问题；工具输出不易理解

调试和修复这些问题需要格外的细心和耐心。

---

## 7. 常用 Benchmarks

### 7.1 Knowledge（知识）

- 准确陈述关于世界的事实；更重“广度”而非“深度”；反映**预训练质量**

**MMLU**（Massive Multitask Language Understanding，Hendrycks et al., 2020）
- 57 个任务（初等数学、美国历史、计算机科学、法律……），每题 4 个选项
- 评估：从 A/B/C/D 中选对，**硬匹配**

### 7.2 Reasoning（推理）

- 解决多步问题，包括“数学”和“常识”两个子领域；考察内部推理能力

**AIME**（American Invitational Mathematics Examination）
- 约 30 道数学题，涵盖几何、代数、分析等，需要深入推理
- 评估：给出正确的 **3 位数**答案（000–999），容易自动判分
- 例：AIME 2025

**PIQA**（Physical Interaction: Question Answering，Bisk et al., 2019）
- 日常场景中的物理常识，每题两个解法（Sol1 / Sol2），约 20,000 个样例
- 评估：选出正确的解法

### 7.3 Coding（代码）

- 生成语法正确的代码；考察编程能力；也是**工具使用能力的代理指标**

**SWE-bench**（Jimenez et al., 2023）
- 2,294 个来自真实 GitHub issue 的软件工程问题，涉及 12 个流行的 Python 仓库
- 每个问题包括：一个 base commit + 一个已合并、带测试的 PR
- 评估：模型生成的 patch 通过所有测试用例

### 7.4 Safety（安全）

- 防止有害、有毒、不当行为；部署前暴露漏洞；与自定义偏好对齐

**HarmBench**（Mazeika et al., 2024）
- 510 种独特的有害行为（400 种文本、110 种多模态），分为 standard、copyright、contextual、multimodal 四类
- 标准：违反法律或普遍接受的规范
- 评估：**Attack Success Rate（ASR）**，用一个分类器判断攻击是否成功诱导出有害输出（越低越安全）

### 7.5 Agents

**τ-bench**（Tool-Agent-User Interaction Benchmark，Yao et al., 2024）
- 给定数据库 schema、APIs 和业务策略，覆盖两个领域：
  - **Airline agent**：500 用户、300 航班、2,000 预订；约 10 个工具、50 个任务
  - **Retail agent**：500 用户、50 产品、1,000 订单；约 10 个工具、115 个任务
- Agent 要与 LLM 模拟的用户对话，并通过工具修改数据库
- 评估：最大化 reward，以及 **pass^k**

**pass^k**：k 次尝试**全部成功**的概率，衡量**一致性 / 可靠性**。对比 pass@k（至少一次成功）。给定 n 次尝试中 c 次成功，无偏估计为 C(c, k) / C(n, k)。对于真实部署的 agent，用户期待每次都对，所以 pass^k 比 pass@k 更重要；而且 pass^k 随 k 增大通常下降很快。

### 7.6 Benchmark 的作用与局限

- 新模型发布（如 2025 年 11 月的 Gemini 3）通常会报告 reasoning、coding、tool use、knowledge 等维度的 benchmark 成绩
- **Benchmark 是在某个轴上对表现的投影**
- 不同模型擅长不同的事情；每个人也有自己的经验“事实”（例如 Claude 的 Sonnet 模型擅长写代码，Gemini 的 Flash 模型便宜）

### 7.7 Pareto Frontier

**定义**：Pareto 曲线 = 在某种权衡下“最优”的解的集合 —— 在这条曲线上，想改进一个维度就必须牺牲另一个维度。

可能的权衡：
- 质量 vs 成本 / 延迟
- 质量 vs 安全
- 质量 vs context 长度

选模型时，应在自己关心的权衡维度上从 Pareto 前沿中挑选。

### 7.8 小心数据污染（Data Contamination）

**问题**：benchmark 的题目或答案线索可能出现在训练集中，导致分数虚高。

**预防措施**：
- **使用唯一标识符**：如 BIG-bench 在数据文件中放一个 canary 字符串（GUID），训练数据中若出现该字符串就说明被污染，应过滤
- **对工具使用 blocklist**：例如评估时禁止模型的搜索工具访问 benchmark 答案所在的网站（Gemini 3 的评估说明中提到）
- **在更新的测试版本上评估**（如每年新的 AIME 题目）

### 7.9 Goodhart's Law

> “When a measure becomes a target, it ceases to be a good measure.”（当一个指标变成目标时，它就不再是一个好指标。）

**教训**：
- 不要过度依赖 benchmark
- 需要更“自然”的视角来补全图景：如 Chatbot Arena（真实用户的盲测投票）
- ……自己亲手试几个模型！

---

## 参考论文

- Cohen, “A coefficient of agreement for nominal scales”, 1960；Fleiss, “Measuring nominal scale agreement among many raters”, 1971
- Papineni et al., “BLEU”, 2002；Lin, “ROUGE”, 2004；Banerjee et al., “METEOR”, 2005
- Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena”, 2023
- Wei et al., “Long-form factuality in large language models”, 2024
- Yao et al., “ReAct”, 2022
- Hendrycks et al., “Measuring Massive Multitask Language Understanding”, 2020
- Bisk et al., “PIQA”, 2019；Jimenez et al., “SWE-bench”, 2023
- Mazeika et al., “HarmBench”, 2024；Yao et al., “τ-bench”, 2024
- Strathern, “‘Improving ratings’: audit in the British University system”, 1997（Goodhart's Law 的常见表述）
