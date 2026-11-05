# Lecture 8 · LLM 评估：人工评分、规则指标、LLM-as-a-Judge 与 Benchmarks

> Stanford CME 295：Transformers & Large Language Models（Afshine Amidi & Shervine Amidi，Fall 2025）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture8.pdf`](../slides/fall25-cme295-lecture8.pdf)
> 预计阅读时间：18–22 分钟。预备知识：Lecture 1（BLEU、ROUGE）、Lecture 3（guided decoding）、Lecture 7（tool calling、ReAct）。

## 本讲脉络

```
“评估”指什么：输出质量 vs 系统性能
→ 人工评分：金标准，但主观（inter-rater agreement、Cohen's Kappa）、慢、贵
→ 规则指标：BLEU、ROUGE、METEOR，便宜但与人类判断相关性差
→ LLM-as-a-Judge：做法、结构化输出、pointwise / pairwise、三种 bias、最佳实践、与人工结合的工作流
→ 扩展 1：事实性（fact decomposition）
→ 扩展 2：agent 的评估 —— 按 tool 调用的三个阶段分析失败模式
→ Benchmarks：知识（MMLU）、推理（AIME、PIQA）、代码（SWE-bench）、安全（HarmBench）、agent（τ-bench，pass^k）
→ 如何看待 benchmark：能力画像、Pareto frontier、数据污染、Goodhart's Law
```

**一句话总结**：LLM 输出的是自由文本，没有唯一的正确答案，所以评估很难。**人工评分最可靠但又慢又贵；规则指标便宜但看不懂“换个说法”；LLM-as-a-Judge 是目前的主力，但它有位置、长度、自我偏好等偏差，必须用人工标注来校准。** Benchmark 只是在某一个维度上的投影 —— 不要过度迷信，最好自己动手试。

---

## 0. 回顾与本讲定位

上一讲：RAG（检索 → 增强 → 生成）、tool calling（模型输出调用参数，应用执行 API）、agent（Observe → Plan → Act 循环）。

| 普通 LLM 的短处 | 对应讲次 |
|---|---|
| 推理能力有限 | Lecture 6 |
| 知识是静态的、不能执行动作 | Lecture 7 |
| **难以评估** | **本讲** |

---

## 1. “评估”指什么

| ✅ Output quality（输出质量） | ⚡ System performance（系统性能） |
|---|---|
| Instruction following：是否遵循了指令（格式、长度、约束） | Latency：多快给出回答 |
| Coherence：是否连贯、通顺 | Pricing：每次调用花多少钱 |
| Factuality：事实是否正确 | Reliability：是否稳定可用（错误率、超时） |
| 等等 | 等等 |

系统性能比较容易测量（计时、计费、统计错误）。本讲主要讨论**输出质量** —— 难点在于 LLM 生成的是**自由文本**，同一个问题有无数种“好”的回答方式。

---

## 2. 人工评分（Human Rating）

### 2.1 理想情况

让人给 LLM 的输出打分 —— **最接近真相，是“金标准（gold standard）”**。实践中通常会：
- 写一份**评分指南（rubric）**，定义每个维度的含义和每个分数档的标准
- 培训标注者，先在少量样本上试标、讨论分歧
- 每个样本由多人独立标注

### 2.2 局限一：主观性

例：
- 📝 Prompt：“What birthday gift should I get?”
- 🤖 回答：“A teddy bear is almost always a sweet gift—just pick one that feels right to you.”
- 🔍 评价标准：**Usefulness（有用性）**

Rater 1 可能觉得有用（给了一个具体建议），Rater 2 可能觉得没用（没问送给谁、预算多少，建议很敷衍）。同一个回答，不同的人判断不同。

### 2.3 衡量标注者之间的一致性：Cohen's Kappa

**直接算“两人判断一致的比例”不够**。设想两个标注者完全随机地打分，也会有一定比例的“一致”。我们真正想问的是：

> 我们的一致程度，比“仅凭运气”预期的一致程度好多少？（考虑到每个标注者实际使用各个类别的频率。）

$$
\kappa = \frac{p_o - p_e}{1 - p_e}
$$

- $p_o$（observed）：实际观察到的一致比例
- $p_e$（expected）：如果两人各自按自己的类别频率随机打分，预期的一致比例：$p_e = \sum_k p_{1,k} \cdot p_{2,k}$
- $\kappa = 1$：完全一致；$\kappa = 0$：和随机一样；$\kappa < 0$：比随机还差

**算例**：两人给 10 个回答打“有用 / 没用”。
- Rater 1 判 6 个有用，Rater 2 判 5 个有用；两人有 7 个判断一致（4 个都说有用，3 个都说没用）
- $p_o = 7/10 = 0.7$
- $p_e = 0.6 \times 0.5$（都说有用的概率）$+ 0.4 \times 0.5$（都说没用的概率）$= 0.5$
- $\kappa = (0.7 - 0.5) / (1 - 0.5) = $ **0.4**

70% 的一致率听起来不错，扣掉运气成分后只是“一般”。常用的经验解读：0.21–0.40 一般（fair），0.41–0.60 中等（moderate），0.61–0.80 较好（substantial），0.81 以上几乎完全一致（almost perfect）。

**变体**：
- **Cohen's Kappa**（Cohen, 1960）：两个标注者
- **Fleiss' Kappa**（Fleiss, 1971）：多个标注者
- **Krippendorff's alpha**：最通用，支持多个标注者、缺失的标注、不同类型的尺度（类别、有序、数值）

κ 很低时，说明**评分指南本身不清楚**或任务太主观，应先改指南，而不是急着收集更多标注。

### 2.4 局限二：慢、贵

每次模型更新都要重新找人标注，几天才能拿到结果，无法支撑快速迭代。

---

## 3. 规则指标（Rule-Based Metrics）

### 3.1 思路

**人工只写一次参考答案（references），以后每次都用规则指标把模型输出与参考答案比较。**

### 3.2 三个常见指标

**BLEU**（BiLingual Evaluation Understudy，Papineni et al., 2002）—— 偏 **precision**，常用于翻译

$$
\text{BLEU} = BP \cdot \exp\Big(\sum_{n=1}^{N} w_n \log p_n\Big),\qquad
BP = \begin{cases} 1 & c > r \\ e^{\,1 - r/c} & c \le r \end{cases}
$$

- $p_n$：**modified n-gram precision** —— 生成文本中的 n-gram 有多少出现在参考文本中。每个 n-gram 的匹配次数被截断到它在参考中出现的次数（防止重复同一个词刷分）
- 通常 $N = 4$，$w_n = 1/4$
- $BP$（brevity penalty）：生成文本比参考短时惩罚（$c$ = 生成长度，$r$ = 参考长度），否则只输出几个最有把握的词就能拿高 precision
- （完整算例见 Lecture 1 笔记）

**ROUGE**（Recall-Oriented Understudy for Gisting Evaluation，Lin, 2004）—— 偏 **recall**，常用于摘要

- **ROUGE-N** = 参考文本中被生成文本覆盖的 n-gram 数 / 参考文本中的 n-gram 总数
- 例：参考 “the bear is reading”（4 个 unigram），生成 “the bear reads a book”：覆盖了 the、bear 两个 → ROUGE-1 recall = 2/4 = 0.5
- **ROUGE-L**：基于**最长公共子序列（LCS）**，不要求连续，能反映词序
- 是一整套变体指标

**METEOR**（Metric for Evaluation of Translation with Explicit ORdering，Banerjee et al., 2005）—— 为了与人类判断更相关

- 匹配 unigram 时，除了完全相同，还允许**词干相同**（reading / read）和**同义词**（cute / adorable）
- 同时考虑 precision 和 recall，并且 recall 权重更高：$F_{\text{mean}} = \frac{10PR}{R + 9P}$
- 加入**词序惩罚**：把匹配上的词分成尽量少的连续“块（chunks）”，块越碎说明词序越乱：$\text{Penalty} = 0.5\big(\frac{\text{chunks}}{\text{matches}}\big)^3$
- $\text{METEOR} = F_{\text{mean}}(1 - \text{Penalty})$

### 3.3 规则指标的局限

**不考虑风格上的变化**。下面三句话意思几乎相同，但用词几乎不重叠，BLEU / ROUGE 互相比较时分数都会很低：
- “A plush teddy bear can comfort a child during bedtime.”
- “Soft stuffed bears often help kids feel safe as they fall asleep.”
- “Many youngsters rest more easily at night when they cuddle a gentle toy companion.”

此外：
- **与人类评分的相关性不太好**：尤其对开放式生成（聊天、写作），一个好回答可能与参考答案完全不同
- **仍然需要人工写参考答案**

（改进方向之一是基于 embedding 的指标，例如用 BERT 的上下文向量比较语义相似度，能部分识别“换了说法但意思相同”的情况。）

---

## 4. LLM-as-a-Judge（LaaJ）

### 4.1 思路（Zheng et al., 2023）

**用一个 LLM 来评价回答的质量。**

```
📝 Prompt：“Why are teddy bears comforting?”
🤖 Response：“They are comforting because they are soft and loyal companions.”
🔍 Criterion：Relevance（相关性）
           ↓  ⚖ LLM-as-a-Judge
🎯 Score：PASS
💭 Rationale：Direct explanation, is on topic
```

Zheng et al. 的研究发现，强 LLM（如 GPT-4）作为 judge 时，与人类偏好的一致率超过 80%，**与人类标注者之间的一致率相当**。

**Judge prompt 示例**：

```
Evaluate how relevant the model's answer is to the user's prompt.
Prompt: {prompt}
Model Response: {model_response}
Return:
- Rationale (1–2 sentences)
- Score: 1 if mostly relevant, 0 if mostly irrelevant.
```

### 4.2 在实践中强制输出格式

回顾 Lecture 3 的 guided decoding：通过限制每一步允许的 token，保证输出符合指定格式。各家 API（Gemini、OpenAI、Anthropic）都提供 **structured outputs** 功能：

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

这样 judge 的输出**一定能被程序解析**，可以在成千上万个样本上自动统计。

### 4.3 相比之前方法的好处

- **不需要参考答案 / 标签**：judge 直接根据评分标准判断
- **可解释**：rationale 说明了为什么给这个分。例：“The response does not contain any grammatical mistakes. It also is clear, easy to understand and is helpful with respect to the task.”
- 快、便宜、可以大规模运行

### 4.4 两种主要形式

| | Pointwise | Pairwise |
|---|---|---|
| 问法 | “评价这个回答的质量” | “回答 A 和回答 B 哪个更好？” |
| 输出 | 一个分数或等级，如 “Very good” | 一个选择，如 “Response A”（也可以允许平局） |
| 适合 | 监控绝对质量（如线上回答的合格率） | 比较两个模型或两个版本（比较比打绝对分更稳定，人和 LLM 都更擅长比较） |

还有一种常用变体：**reference-guided**，给 judge 提供一个参考答案（例如数学题的正确答案），让它判断回答是否与参考一致，可以显著提升 judge 在有标准答案的任务上的准确度。

### 4.5 三种常见 bias（偏差）

**1）Position bias（位置偏差）**
- **症状**：pairwise 比较时，judge 偏好某个位置（例如总是倾向于选第一个）。把 A、B 交换顺序后，判断也跟着变 —— 第一次说 A 好，交换后又说 B（其实是同一个回答）好
- **补救**：**两种顺序各评一次，取平均**（或只在两次判断一致时才算数，不一致记为平局）；研究上也有调整 position embedding 的方法

**2）Verbosity bias（冗长偏差）**
- **症状**：偏好更长的回答。“This response is short and correct.” 可能输给 “This response goes into details, a lot of which are not needed and is overall not very helpful.”
- **补救**：在评分准则中明确写出“简洁不扣分，冗余内容扣分”；提供 few-shot 示例；对输出长度加惩罚

**3）Self-enhancement bias（自我偏好）**
- **症状**：judge 偏好**自己生成的**回答，甚至胜过人工精心撰写、完美回答了问题的答案
- **补救**：**不要用同一个模型既当生成者又当 judge**（🤖 ≠ ⚖）

### 4.6 最佳实践（逐条说明原因）

| 实践 | 原因 |
|---|---|
| **清晰明确的评分准则**（crisp guidelines） | 模糊的标准对 judge 和人一样会导致不一致 |
| **二元尺度优于细粒度尺度** | “合格 / 不合格”比“1–10 分”更容易判断、更稳定；10 分制中 6 分和 7 分的区别连人都说不清 |
| **先写 rationale，再给分** | 相当于让 judge 先做 CoT 推理，分数更有依据；反过来先给分再找理由，理由会变成“事后合理化” |
| **缓解各种 bias** | 交换位置、控制长度、换模型当 judge |
| **用人工判断校准** | 在一小批人工标注的样本上，测量 judge 与人的一致率（准确率、Cohen's κ），不断修改 judge prompt 直到一致率足够高 |
| **低 temperature** | 保证同样的输入得到同样的评分，结果可复现 |

### 4.7 修订后的评估工作流

```
🐢 人工标注一小批样本（慢，但是金标准）
    ↓
用人工标注校准 LLM-as-a-Judge（调整 judge prompt，直到与人的一致性足够高）
    ↓
🐇 用 LaaJ 大规模、快速地评估每一个新版本
    ↓
定期抽样人工复核，防止 judge 悄悄“漂移”
```

### 4.8 通常要评估的维度

| 🧠 Task performance（任务表现） | 🗣 Alignment（对齐） |
|---|---|
| 有用性、相关性、正确性、事实性、完整性等 | Tone（语气）、Style（风格）、Safety（安全） |

---

## 5. 扩展：事实性（Factuality）评估

**目标**：量化一段输出的事实性。

例：“Teddy bears, first created in the 1920s, were named after President Theodore Roosevelt after he proudly wanted to shoot a captured bear on a hunting trip.”

这段话**部分正确、部分错误**：teddy bear 实际诞生于 1902–1903 年；Roosevelt 在打猎时是**拒绝**射杀那只被拴住的熊，这件事被画成漫画，玩具熊因此以他的昵称 “Teddy” 命名。简单地给整段打“对”或“错”都不合适。**开放问题：如何量化这种细微差别？**

**做法：Fact decomposition（事实分解）**（参考 Wei et al., 2024，“Long-form factuality in large language models”。该论文提出的 SAFE 方法用 LLM 把长回答拆成原子事实，再对每个事实调用搜索引擎核查。）

1. 把原文拆成**独立的原子事实**（每条只陈述一件事）
2. 逐条核查真假（查搜索引擎、知识库或让 judge 判断）
3. 按重要性加权汇总：

$$
\text{score} = \frac{\sum_i w_i \cdot \mathbb{1}[\text{事实 } i \text{ 正确}]}{\sum_i w_i}
$$

| 原子事实 | 重要性 $w_i$ | 是否正确 |
|---|---|---|
| Teddy bears were first created in the 1920s. | 0.3 | ❌ |
| Teddy bears were named after President Theodore Roosevelt. | 0.4 | ✅ |
| Theodore Roosevelt was on a hunting trip where a bear was captured. | 0.2 | ✅ |
| Theodore Roosevelt proudly wanted to shoot the captured bear. | 0.1 | ❌ |

score = (0.4 + 0.2) / 1.0 = **0.60**

---

## 6. 扩展：Agent 的评估与失败模式

### 6.1 Agentic 视角

回顾 ReAct：Observe → Plan → Act 循环。一次典型的 agentic 调用会经历**多轮循环**，每轮都包含 Lecture 7 中 tool calling 的三步：

1. **LLM 为相关函数推断参数**：“Find a bear near me!” + `<function APIs>` → `find_teddy_bear(location=(37.42, -122.17))`
2. **执行函数调用**：后端返回 `{"name": "Teddy", …}`
3. **LLM 根据结果推出结论**，生成回复

每一步都可能出错。评估 agent 时，不能只看最终答案对不对，还要**定位是哪一步出了错**，才知道该怎么修。

### 6.2 失败类型 1：Tool prediction error（工具预测错误）

| 症状 | 可能原因 | 补救 |
|---|---|---|
| **不使用工具**：直接回复 “Sorry, I don't know where I can find one.” | Tool router 出错（没把该工具选进候选） | 重新训练 tool router |
| | 模型不知道怎么用这个工具 | 对模型做 SFT，或改进目标 API 对应的 prompt 说明 |
| **幻想出不存在的工具**：调用 `find_bear()` | 模型太弱 | 升级模型 |
| | API 命名不合逻辑，让模型“猜”了一个更自然的名字 | 重新设计 API |
| | 指令不清楚 | 迭代顶层指令 |
| **用错工具**：调用 `send_message(recipient=LocalBusiness, message="may I buy a teddy bear?")` | 模型选择了错误的工具 | 同上（升级模型、改进 API 命名和指令） |
| **参数推断错误**：`location = (0, -0)` | 参数**无法从上下文推断**（模型根本不知道用户在哪，只好编一个） | 引入辅助工具（如 `get_user_location()`），以及/或确保上下文中携带所需信息 |

### 6.3 失败类型 2：Tool call error（工具调用错误）

| 症状 | 原因 | 补救 |
|---|---|---|
| **返回错误结果**：工具返回了不正确的值，或者抛出异常（如 `ValueError: could not...`） | 工具实现有 bug（注意：有时报错是合理的，例如参数非法，但不应是常态） | **修复工具实现！** |
| **没有返回**：工具什么也不返回，常导致最终回复是**幻觉编造**的（模型拿不到结果，就自己编一个） | bug，或工具实现的“特性”（如成功时不返回任何内容） | **总要返回点东西**，哪怕是空 JSON；一般原则：输出有意义的工具结果 |

### 6.4 失败类型 3：Response generation error（回复生成错误）

| 症状 | 原因 | 补救 |
|---|---|---|
| 最终回复没有正确传达工具的结果：工具明明找到了熊，回复却说 “Didn't find any bear!” | 模型缺乏 **grounding** 能力（不能忠实地依据给定信息回答） | 升级负责综合回复的 LLM |
| | 工具返回的内容太多，塞满了 context window，关键信息被淹没 | 精简后端返回的信息 |
| | 工具输出的格式不能有意义地传达信息（如一堆没有字段名的数字） | 让工具输出格式更具描述性 |

### 6.5 总结

| 阶段 | 失败 |
|---|---|
| Tool prediction | 不用工具 / 幻想工具 / 用错工具 / 参数错误 |
| Tool call | 结果错误 / 没有结果 |
| Response generation | 回复错误 |

**问题来源归纳**：
- **建模方面**：推理和 grounding 能力弱；context window 中东西太多；工具的建模（命名、描述、粒度）不对
- **工具方面**：工具本身有问题；工具输出不易理解

**实践建议**：记录每一步的完整轨迹（模型看到了什么、调用了什么、返回了什么），按上面的分类给失败打标签、统计各类失败的比例，再优先修最常见的那类。调试和修复需要格外的细心和耐心。

---

## 7. 常用 Benchmarks

### 7.1 Knowledge（知识）

- 准确陈述关于世界的事实；更重**广度**而非深度；主要反映**预训练质量**

**MMLU**（Massive Multitask Language Understanding，Hendrycks et al., 2020）
- 57 个任务，覆盖初等数学、美国历史、计算机科学、法律等，每题 4 个选项
- 评估：从 A / B / C / D 中选对，**硬匹配**（模型输出的选项字母与答案比较）

### 7.2 Reasoning（推理）

- 解决多步问题；包括**数学**和**常识**两个子领域；考察模型的内部推理

**AIME**（American Invitational Mathematics Examination）
- 每年约 30 道数学竞赛题，涵盖几何、代数、分析等，需要深入的推理
- 评估：给出正确的 **3 位数**答案（000–999），容易自动判分
- 每年都有新题（如 AIME 2025），可以减少数据污染

**PIQA**（Physical Interaction: Question Answering，Bisk et al., 2019）
- 日常场景中的**物理常识**，约 20,000 个样例，每题两个解法（Sol1 / Sol2）
- 例：目标 “用矿泉水瓶分离蛋清和蛋黄”。解法 1：挤压瓶子后贴住蛋黄，松开，产生的吸力把蛋黄吸起来。解法 2：把瓶子贴住蛋黄，持续挤压，产生吸力把蛋黄吸起来。（正确的是解法 1：吸力来自松开，而不是挤压）
- 评估：选出正确的解法。对人很简单，对只从文本中学习的模型却不容易

### 7.3 Coding（代码）

- 生成语法正确的代码；考察编程能力；也是**工具使用能力的代理指标**

**SWE-bench**（Jimenez et al., 2023）
- 2,294 个来自真实 GitHub issue 的软件工程问题，涉及 12 个流行的 Python 仓库
- 每个问题包括：一个 base commit（问题出现时的代码版本）+ 一个已合并的、带测试的修复 PR
- 评估：模型生成的 patch 应用后，**通过所有测试用例**（包括 PR 中新增的、针对这个 issue 的测试）
- 常用子集 SWE-bench Verified（500 题，经过人工确认题目描述清楚、测试合理）

### 7.4 Safety（安全）

- 防止有害、有毒、不当的行为；在部署前暴露漏洞；与自定义的偏好对齐

**HarmBench**（Mazeika et al., 2024）
- 510 种独特的有害行为（400 种文本、110 种多模态），分为 standard（标准）、copyright（版权）、contextual（带上下文）、multimodal（多模态）四类
- 标准：违反法律或普遍接受的社会规范
- 评估：**Attack Success Rate（ASR，攻击成功率）** = 成功诱导出有害输出的攻击次数 / 总攻击次数，由一个分类器判断输出是否有害。ASR 越低越安全

### 7.5 Agents

**τ-bench**（Tool-Agent-User Interaction Benchmark，Yao et al., 2024）
- 给定数据库 schema、API 和业务规则（policy），覆盖两个领域：
  - **Airline agent**：500 个用户、300 个航班、2,000 个预订；约 10 个工具、50 个任务
  - **Retail agent**：500 个用户、50 个产品、1,000 个订单；约 10 个工具、115 个任务
- Agent 要与一个 LLM 模拟的用户多轮对话，遵守业务规则，并通过工具修改数据库
- 评估：最大化 reward（最终数据库状态是否正确），以及 **pass^k**

**pass^k**：k 次尝试**全部成功**的概率，衡量**一致性 / 可靠性**（对比 pass@k：至少一次成功）。给定 $n$ 次尝试中 $c$ 次成功，无偏估计为 $\binom{c}{k} / \binom{n}{k}$。

为什么重要：真实部署的客服 agent，用户期望**每次**都处理对，而不是“试几次总有一次对”。直觉上，如果每次成功率是 60% 且相互独立，pass^4 只有 $0.6^4 \approx 13\%$。τ-bench 论文中，当时最强的 function-calling agent 在 retail 领域的 pass^8 低于 25% —— 单次表现看着不错，一致性却很差。

### 7.6 Benchmark 的作用：能力画像

- 新模型发布时（如 2025 年 11 月的 Gemini 3），通常会报告 reasoning、coding、tool use、knowledge 等维度的 benchmark 成绩
- **Benchmark 是在某一个轴上对模型表现的投影**
- 不同的模型擅长不同的事情；排行榜上的总排名掩盖了这些差异
- 每个人也有自己的经验“事实”，例如“Claude 的 Sonnet 模型擅长写代码”“Gemini 的 Flash 模型便宜”

### 7.7 Pareto Frontier

**定义**：Pareto 曲线 = 在某种权衡下“最优”的解的集合 —— 在这条曲线上，想改进一个维度，就必须牺牲另一个维度；不在曲线上的模型，总能找到一个各方面都不差、且至少一方面更好的替代者。

常见的权衡：
- 质量 vs 成本 / 延迟
- 质量 vs 安全
- 质量 vs context 长度

选模型时，应该在**自己关心的**权衡维度上，从 Pareto 前沿中挑选，而不是只看谁的分数最高。

### 7.8 小心数据污染（Data Contamination）

**问题**：benchmark 的题目或答案线索可能出现在训练数据中（网上到处都有题目和解答），导致分数虚高 —— 模型是“背过答案”，而不是“会做题”。

**预防措施**：
- **使用唯一标识符**：如 BIG-bench 在所有数据文件中嵌入一个 canary 字符串（一段特定的 GUID）。训练数据中一旦出现这个字符串，就说明被污染了，应该把相关数据过滤掉
- **对工具使用 blocklist**：评估带搜索工具的模型时，禁止它访问 benchmark 答案所在的网站（例如 Gemini 3 的评估方法说明中提到了这一点）
- **在更新的测试版本上评估**：例如用今年新出的 AIME 题目

### 7.9 Goodhart's Law

> “When a measure becomes a target, it ceases to be a good measure.”
> （当一个指标变成目标时，它就不再是一个好指标。）

当所有人都在刷同一个 benchmark，模型会被（有意或无意地）优化到“擅长这个 benchmark”，而不一定是“真正变强”。

**教训**：
- 不要过度依赖 benchmark
- 需要更“自然”的视角来补全图景：如 Chatbot Arena（真实用户的盲测投票）
- ……**自己亲手试几个模型！** 用你自己的真实任务测一测

---

## 8. 要点回顾

1. 评估分输出质量（遵循指令、连贯、事实）和系统性能（延迟、成本、可靠性）；难点在于自由文本没有唯一正确答案
2. 人工评分是金标准，但主观、慢、贵；用 Cohen's κ 等指标衡量标注者间扣除运气后的一致性
3. BLEU（偏 precision）、ROUGE（偏 recall）、METEOR（词干、同义词、词序）便宜，但不懂换说法，与人类判断相关性差
4. LLM-as-a-Judge：按准则打分并给出理由，可用 structured outputs 保证格式；分 pointwise 和 pairwise
5. Judge 的三大偏差：位置（交换顺序取平均）、冗长（准则 + 长度惩罚）、自我偏好（生成者与 judge 用不同模型）
6. 最佳实践：清晰准则、二元尺度、先理由后分数、用人工校准、低 temperature
7. 事实性评估：拆成原子事实 → 逐条核查 → 按重要性加权
8. Agent 失败按 tool 调用三阶段定位：预测（不用 / 幻想 / 用错 / 参数错）、调用（结果错 / 无结果）、生成（回复与结果不符）
9. Benchmark 只是投影；关注 Pareto 前沿、提防数据污染和 Goodhart's Law，最终要自己试

---

## 9. 自测题

<details>
<summary>1. 两个标注者一致率 90%，但 κ 只有 0.2。这可能是什么情况？</summary>

类别极不平衡。例如两人都几乎把所有样本标成“合格”，随机情况下的预期一致率 p_e 就很高（接近 0.9），所以 κ = (0.9 − p_e)/(1 − p_e) 很低。说明两人在少数关键样本上的判断并不一致。
</details>

<details>
<summary>2. 为什么同一意思的不同说法，BLEU 分数会很低？</summary>

BLEU 只统计 n-gram 的字面重合。换了说法后用词几乎不重叠，n-gram precision 很低，尽管意思相同。这是规则指标与人类判断相关性差的主要原因。
</details>

<details>
<summary>3. 设计一个检测和缓解 position bias 的方法。</summary>

对每对回答评两次：一次 A 在前，一次 B 在前。如果两次结论矛盾，说明存在位置偏差，记为平局（或取两次的平均概率）；统计结论翻转的比例可以量化 position bias 的严重程度。
</details>

<details>
<summary>4. 为什么建议 judge “先写 rationale，再给分”？</summary>

先写理由相当于让 judge 先做一段 CoT 推理，分数建立在分析之上，更准确也更一致；如果先给分，后面的理由往往只是为已经给出的分数找借口。
</details>

<details>
<summary>5. Agent 回复 “Didn't find any bear!”，但日志显示工具返回了一只熊。这属于哪类失败？可能怎么修？</summary>

Response generation error。可能原因：模型 grounding 能力弱（升级模型）；工具返回内容太长淹没了关键信息（精简返回）；返回格式难以理解（让输出更具描述性）。
</details>

<details>
<summary>6. 一个 agent 单次成功率 70%，各次独立。pass^3 和 pass@3 分别是多少？</summary>

pass^3 = 0.7³ ≈ 0.343（三次全对）；pass@3 = 1 − 0.3³ = 0.973（至少一次对）。两者差距很大，说明“偶尔能做对”和“每次都做对”是完全不同的要求。
</details>

<details>
<summary>7. 什么是 Goodhart's Law？它对 LLM 评估有什么启示？</summary>

当一个指标变成目标时，它就不再是一个好指标。大家都优化同一个 benchmark，模型可能只是变得擅长这个 benchmark（甚至因数据污染而“背答案”），并不代表真实能力提升。所以要结合多种评估、真实用户反馈，以及在自己的任务上亲自测试。
</details>

---

## 参考论文

- Cohen, “A coefficient of agreement for nominal scales”, 1960；Fleiss, “Measuring nominal scale agreement among many raters”, 1971
- Papineni et al., “BLEU: A Method for Automatic Evaluation of Machine Translation”, 2002
- Lin, “ROUGE: A Package for Automatic Evaluation of Summaries”, 2004
- Banerjee et al., “METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments”, 2005
- Zheng et al., “Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena”, 2023
- Wei et al., “Long-form factuality in large language models”, 2024
- Yao et al., “ReAct: Synergizing Reasoning and Acting in Language Models”, 2022
- Hendrycks et al., “Measuring Massive Multitask Language Understanding”, 2020
- Bisk et al., “PIQA: Reasoning about Physical Commonsense in Natural Language”, 2019
- Jimenez et al., “SWE-bench: Can Language Models Resolve Real-World GitHub Issues?”, 2023
- Mazeika et al., “HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal”, 2024
- Yao et al., “τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains”, 2024
- Strathern, “‘Improving ratings’: audit in the British University system”, 1997（Goodhart's Law 的常见表述）
