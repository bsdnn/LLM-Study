# Lecture 6 · Reasoning Models：用强化学习扩展推理能力

> Stanford CME 295：Transformers & Large Language Models（Afshine Amidi & Shervine Amidi，Fall 2025）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture6.pdf`](../slides/fall25-cme295-lecture6.pdf)
> 预计阅读时间：18–22 分钟。预备知识：Lecture 3（CoT、self-consistency）、Lecture 5（RL、PPO、GRPO 的基本形式）。

## 本讲脉络

```
普通 LLM 的长处与短处 → 什么是 reasoning → 从 CoT 到 reasoning model（test-time scaling）
→ 怎么评估：代码与数学 benchmark，pass@k、pass@1、cons@k
→ 怎么训练：为什么用 RL、verifiable rewards（格式 + 正确性）、推理时怎么控制思考长度
→ GRPO 详解（与 PPO 对比）→ “回答越来越长”现象的成因与修正（DAPO、Dr. GRPO）
→ DeepSeek R1-Zero 与 R1 的完整训练流程 → Distillation：把推理能力蒸馏给小模型
```

**一句话总结**：**Reasoning model = 规模大得多的 Chain-of-Thought，而且推理过程是用强化学习“练”出来的，不是人写的。** 数学和代码题有天然的“对 / 错”信号，可以直接当作 reward；用 GRPO 优化后，模型自发学会长时间思考、自我检查、推翻重来。这开辟了一个新的扩展维度 —— **test-time scaling**：推理时多花 token，就能换来更好的答案。

---

## 0. 回顾与本讲定位

**到目前为止的 LLM 训练流程**：Pretraining → Finetuning（SFT）→ Preference tuning（RLHF / DPO）。Lecture 5 把 LLM 写成了 RL 问题（agent ↔ LLM，environment ↔ tokens，action = 下一个 token，policy = 下一个 token 的概率分布），用 PPO-clip 或 PPO-KL penalty 最大化 advantage、同时不偏离 base model 太远。

**普通 LLM 的长处与短处**：

| 长处 | 短处 |
|---|---|
| 擅长模仿、产生想法 | **推理能力有限**（本讲） |
| 生成和调试代码很强 | 知识是静态的（Lecture 7：RAG） |
| | 不能执行动作（Lecture 7：tools、agents） |
| | 难以评估（Lecture 8） |

---

## 1. 什么是 Reasoning

**暂定定义**：Reasoning = 解决问题的能力。

| 不是 reasoning（知识检索） | 是 reasoning（需要多步推导） |
|---|---|
| “What is the course code of Stanford's Transformers & LLMs class?” —— 记得就答得出，不记得想多久也没用 | “The bear was born in 2020. How old is this bear now?” —— 要先确定今年是哪年，再做减法 |

一个常用的类比是心理学中的两种思考方式：**System 1**（快速、直觉、自动，如认出一张脸）和 **System 2**（缓慢、刻意、逐步，如做一道多位数乘法）。普通 LLM 一个 token 接一个 token 地“脱口而出”，更像 System 1；reasoning model 试图让模型具备 System 2 式的思考。

---

## 2. 从 Chain-of-Thought 到 Reasoning Model

### 2.1 核心思路：先解释推理，再回答

回顾 CoT（Wei et al., 2022）：让模型在回答之前先写出推理过程，效果显著提升。

```
不用 CoT：
Q: How old is this bear?            A: 4.
Q: How old will the bear be next year?   A: 5.          ← 直接给答案，复杂题很容易错

用 CoT：
Q: How old is this bear?
A: The bear was born in 2020. It is therefore 4.
Q: How old will the bear be next year?
A: It will be one year older than its age this year, which was 4. Hence, it will be 5.
```

为什么有效：每个 token 的计算量固定，难题需要更多计算步骤；把中间结果写出来，后面的推理可以建立在它之上（相当于草稿纸）。

### 2.2 Reasoning model 的想法：把 CoT 做到大得多的规模

| 以前 | 新范式 |
|---|---|
| Question → Answer（可能带一小段 CoT） | Question → **很长的 reasoning chain** → Answer |
| 推理过程靠 prompt 引导（“Let's think step by step”） | 推理过程是**训练出来的**行为 |
| 推理通常几十到几百个 token | 推理可以有几千到几万个 token |

一个好的推理链里会出现人类解题时的典型行为：
- **规划**：“先求出 x，再代入第二个方程”
- **验证**：“检查一下：把 x = 3 代回去……成立”
- **回溯**：“等等，这里算错了，重新来”
- **尝试其他方法**：“换一种思路，用反证法试试”

### 2.3 Test-time scaling

传统的扩展方式是“训练时花更多算力”（更大的模型、更多数据）。Reasoning model 开辟了第二个维度：**推理（test time）时花更多算力 —— 让模型想得更久** —— 效果也会持续提升。

例：OpenAI 报告，在 AIME 2024 数学竞赛上，GPT-4o 平均只能答对约 12% 的题；o1 单次采样约 74%，64 次采样多数投票约 83%。

### 2.4 发展时间线

| 日期 | 发布 |
|---|---|
| 2024-09-12 | OpenAI o1-preview |
| 2024-12-19 | Gemini 2.0 Flash Thinking |
| 2025-01-20 | DeepSeek R1 |
| 2025-02-19 | Grok 3 Beta |
| 2025-02-24 | Claude 3.7 Sonnet |
| 2025-06-10 | Magistral（Mistral） |

（时间线仅作示意，展示的是各家 AI 实验室的首个推理模型发布。）

### 2.5 如何识别一个 reasoning model

- 聊天界面中通常会显示一段 **“thought summary”（思考摘要）**，**完整的 chain of thought 通常被隐藏**
- API 中有控制思考的参数：例如 reasoning effort（low / medium / high）或 thinking budget（思考 token 的上限）
- **思考 token 也要计费**（通常按 output token 的价格），所以 reasoning model 的单次调用往往更贵、更慢

---

## 3. 推理能力的 Benchmark 与指标

### 3.1 代码：解编程题、修 bug

**例题**：“有 n 只 teddy bear 排成一行，每只有一个大小。找出比最大那只小的、最大的那只。”

```python
def second_biggest_bear(bears):
    largest = max(bears)
    return max(b for b in bears if b < largest)
```

**验证**：用 t 个测试用例运行，**全部通过**才算对（Test case #1 ✅、#2 ✅ …… #t ✅）。

代表 benchmark：
- **HumanEval**：164 道手写 Python 函数题
- **CodeForces**：竞赛编程平台，可以用 Elo 等级分衡量模型水平
- **SWE-bench**：修复真实 GitHub 仓库中的 issue

### 3.2 数学：解难题（如奥赛）

**例题**：“The bear was born in 2020. How old is the bear now?”
推理：“It is 2025 now. Subtract the birth year from the current year: 2025 − 2020 = 5.” → Answer: 5
**验证**：提取最终答案，与 ground truth 比较（通常要求答案放在固定格式里，便于程序提取和规范化）。

代表 benchmark：
- **AIME**（美国数学邀请赛）：每年 30 道题，答案都是 0–999 的整数，便于自动判分
- **GSM8K**：小学数学应用题

**共同点：答案可以被程序自动验证** —— 这一点对后面的训练至关重要。

### 3.3 指标：pass@k

**Pass@k**（Chen et al., 2021）：**k 次尝试中至少有一次成功**的概率。

直接“采样 k 次看有没有对的”方差很大。标准做法：对每道题采样 $n \ge k$ 次，其中 $c$ 次正确，用无偏估计

$$
\text{pass@}k = 1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}
$$

分子是“从 $n - c$ 个错误回答中选出 $k$ 个”的方法数，即 $k$ 次全错的情况。

**算例**：$n = 10$ 次采样中 $c = 3$ 次正确。
- pass@1 = $1 - \binom{7}{1} / \binom{10}{1} = 1 - 7/10 = 0.3$（就是正确率）
- pass@2 = $1 - \binom{7}{2} / \binom{10}{2} = 1 - 21/45 \approx 0.53$
- pass@5 = $1 - \binom{7}{5} / \binom{10}{5} = 1 - 21/252 \approx 0.92$

**怎么选指标**：
- **Pass@k**：适合**容易检查答案**、或能接受更高延迟的场景（多生成几次，用测试挑出对的）
- **Pass@1**：只关心**一次生成**的场景，是最常报告的指标。实践中常对每题采样多次（如 16 次）后取平均正确率，以减小随机性
- **Cons@k**（consensus at k，DeepSeek-R1 论文）：生成 k 个答案，**多数投票**得到一个答案，再与 ground truth 比较（即 Lecture 3 中 self-consistency 的准确率）

---

## 4. 训练 Reasoning Model：发展 “Test-Time Scaling”

### 4.1 为什么用 RL，而不是 SFT

**想法**：激励模型在回答前先推理。

**考虑**：
- **推理链很难从零写出来**：让人类为几十万道难题手写详细的推理过程，既慢又贵，**手写 SFT 数据不现实**
- **不想把模型限制在人类写的推理方式上**：模型也许能找到人类没想到的、更有效的思路
- 这类问题有**天然可验证的 reward**：“解出来了吗？” → 是 / 否

→ **用 RL！** 不需要告诉模型“怎么想”，只告诉它“结果对不对”，让它自己摸索出好的推理方式。

### 4.2 两种 verifiable reward（可验证的奖励）

**Reward 1：格式 —— 检查 CoT 是否存在**

要求输出符合模板：

```
<think>
……推理过程……
</think>
ANSWER
```

**Reward 2：正确性 —— 检查答案是否正确**
- 代码：运行测试用例（Test case #1 … #c 全部 ✅）
- 数学：提取最终答案，与 ground truth 比较 ✅

```
Reward = r_formatting（有 think 分隔符吗？） + r_accuracy（答案正确吗？）
```

这种直接用规则判断对错的方式叫 **RLVR（RL with Verifiable Rewards）**。

**为什么规则式 reward 好**：
- **很难被 hack**：Lecture 5 的 reward model 是一个神经网络，有漏洞可钻；“答案是否等于 42”没有漏洞可钻。DeepSeek 在训练推理能力时特意**不用神经网络 reward model**，就是为了避免 reward hacking
- 便宜：不需要额外的大模型来打分
- 精确

**局限**：
- 只适用于**有可验证答案**的领域（数学、代码、逻辑题）；写作、对话等开放任务仍需要 reward model
- 测试用例也可能被“钻空子”（例如针对测试用例写特判，而不是真正解决问题）
- 只有最终对错，没有对中间步骤的评价（中间步骤错了但答案碰巧对，也会被奖励）

### 4.3 推理时控制“思考”

**问题**：不是所有 prompt 都一样难。“2 + 3 等于几”不需要思考 1,000 个 token（overthinking，浪费成本和时间）；一道奥赛题想得太少又会答错。

**控制思考长度的几种思路**：

| 方法 | 思路 |
|---|---|
| **Dynamic budget（动态预算）** | 根据问题难度分配不同的 token 预算 |
| **Context awareness（让模型知道预算）**（TALE，Han et al., 2024） | 先估计这道题需要多少 token，把预算写进 prompt（如“请在 50 个 token 内完成推理”），让模型据此调整推理详略 |
| **Budget forcing（强制预算）**（s1，Muennighoff et al., 2025） | 想让模型少想：达到上限时直接插入“思考结束”标记，强制它给出答案。想让模型多想：在它准备结束时把结束标记换成 **“Wait”**，迫使它继续检查自己的推理 |
| **“Continuous” thoughts（连续思考）**（Coconut，Hao et al., 2024） | 不把思考写成离散的 token，而是把模型最后一层的隐藏向量直接当作下一步的输入，在**连续的 latent space** 中推理。一个向量可以同时“保留多种可能性”，更适合需要搜索的问题 |

**s1 的例子**：只用 1,000 道精选题目（附带推理过程）对 Qwen2.5-32B 做 SFT，再配合 budget forcing，在竞赛数学题上超过了 o1-preview（MATH 和 AIME24 上最多高出 27%）。说明“让模型多想一会儿”这个简单技巧就很有效。

API 中的 reasoning effort 参数，本质上也是在控制思考预算。

---

## 5. GRPO（Group Relative Policy Optimization）

来自 DeepSeekMath（Shao et al., 2024），是训练 reasoning model 最常用的 RL 算法。

### 5.1 做法

对每个问题 $q$，用旧 policy 采样一组 $G$ 个回答 $o_1, \dots, o_G$，用 verifiable reward 打分得到 $r_1, \dots, r_G$。

**Advantage ≈ Reward − 组内平均 reward**（再除以组内标准差）：

$$
\hat A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}
$$

这个 advantage 赋给回答 $o_i$ 中的**每一个 token**。

**目标函数**（要最大化）：

$$
J(\theta) = \mathbb{E}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\min\Big(\rho_{i,t}\, \hat A_i,\; \text{clip}(\rho_{i,t},\, 1-\epsilon,\, 1+\epsilon)\, \hat A_i\Big)
\Bigg] - \beta\, \mathrm{KL}\big(\pi_\theta \,\|\, \pi_{\text{ref}}\big)
$$

其中 $\rho_{i,t} = \frac{\pi_\theta(o_{i,t} \mid q, o_{i,<t})}{\pi_{\theta_{\text{old}}}(o_{i,t} \mid q, o_{i,<t})}$ 是新旧 policy 的概率比值，$\beta$ 控制不偏离 base model 太远。

### 5.2 一个例子

问题：“The bear was born in 2020. How old is the bear in 2025?” 采样 4 个回答：

| 回答 | 最终答案 | Reward | Advantage |
|---|---|---|---|
| $o_1$：…2025 − 2020 = 5… | 5 | 1 | +1 |
| $o_2$：…2024 − 2020 = 4… | 4 | 0 | −1 |
| $o_3$：…the bear is 6… | 6 | 0 | −1 |
| $o_4$：…subtract to get 5… | 5 | 1 | +1 |

均值 0.5、标准差 0.5。训练会提高 $o_1$、$o_4$ 中每个 token 的概率，降低 $o_2$、$o_3$ 中每个 token 的概率。久而久之，模型学会了导向正确答案的推理方式。

**如果一组回答全对或全错**：advantage 全为 0，这道题提供不了学习信号（太简单或太难）。所以训练题目的难度要合适。

### 5.3 与 PPO 的比较

| | PPO | GRPO |
|---|---|---|
| 新旧 policy 的 ratio | ✅ | ✅ 相同 |
| Clipping | ✅ | ✅ 相同 |
| KL penalty | 加在**每个 token 的 reward** 里 | **直接加在 loss** 里 |
| Advantage 估计 | 训练一个 **value model**，用 GAE 计算 | 用**同组其他回答的 reward** 作为 baseline，**不需要 value model** |

**最大的区别：省掉了 value model**。这很重要：
- Value model 通常和 policy 一样大，省掉它节省大量显存和计算
- 对于长达上万 token、只在结尾才有 reward 的推理链，**训练一个准确的 token 级 value model 非常困难**；用同组回答的平均表现做 baseline 简单又有效

---

## 6. “回答越来越长”现象

### 6.1 观察

随着 RL 训练进行，**回答长度持续增加**。DeepSeek R1-Zero 的平均回答长度从几百个 token 一路涨到上万个 token。

这里面有好的成分：模型学会了更多验证和反思（著名的 **“aha moment”**：模型在推理中途写下 “Wait, wait. Wait. That's an aha moment I can flag here.”，然后推翻之前的思路重新求解）。但也有一部分来自**损失函数中的偏差** —— 一个坏的激励。

### 6.2 偏差从哪来

GRPO 的 loss 中，每个回答先在**内部按长度 $|o_i|$ 平均**，再在组内平均。所以回答 $o_i$ 中**每个 token 的梯度权重**是 $\hat A_i / |o_i|$：

- 对**答对**的回答（$\hat A > 0$）：越短，每个 token 分到的奖励越大 → 偏好**简短的正确回答**
- 对**答错**的回答（$\hat A < 0$）：越长，每个 token 受到的惩罚越小

**算例**：两个答错的回答，$\hat A = -1$：
- 长度 100：每个 token 的权重 $-1/100 = -0.01$
- 长度 1,000：每个 token 的权重 $-1/1000 = -0.001$

长的那个回答，每个 token 受到的惩罚只有短回答的 1/10。模型因此被激励：**答不出来的时候，就拼命写长**。

### 6.3 修正：让每个 token 的贡献相等

**DAPO**（Yu et al., 2025，“An Open-Source LLM Reinforcement Learning System at Scale”）的几项改进：
- **Token-level loss**：把一组中所有回答的所有 token 放在一起，除以**总 token 数**（而不是每个回答各自除以自己的长度），每个 token 权重相同
- **Clip-higher**：把 clip 的上限放宽（如下限 $\epsilon_{\text{low}} = 0.2$、上限 $\epsilon_{\text{high}} = 0.28$），给低概率 token 更多上升空间，防止 entropy（多样性）塌缩
- **Dynamic sampling**：过滤掉全对或全错的题目组（它们的 advantage 全为 0），保证每个 batch 都有有效的学习信号
- **Overlong reward shaping**：对超长、被截断的回答做特殊处理，避免误伤
- 结果：用 Qwen2.5-32B 基础模型在 AIME 2024 上达到 50 分，超过 DeepSeek-R1-Zero-Qwen-32B（47 分），且只用了一半的训练步数

**Dr. GRPO**（Liu et al., 2025，“Understanding R1-Zero-Like Training: A Critical Perspective”）：
- **去掉按回答长度的归一化**（$1/|o_i|$），改用常数
- **去掉按组内标准差的归一化**：标准差很小的题（几乎全对或全错的“太简单 / 太难”的题）会被除以一个小数而被过度放大，造成与难度相关的偏差
- 结果：在效果相当的情况下，回答长度明显更短（token 效率更高）

**其他调整**：与题目难度相关的偏差、鼓励多样性（保持 entropy）、按难度安排课程（curriculum）……还有很多。

---

## 7. 把一切串起来：DeepSeek R1

```
V3-Base（传统预训练）─┬─► V3（传统 SFT + RL 的聊天模型）
                      ├─► R1-Zero（概念验证：纯 RL，不做 SFT）
                      └─► R1（完整流程）
```

### 7.1 V3-Base

- 用“传统”方法预训练（约 14.8T tokens）
- MoE：总参数约 **671B**，每个 token 激活约 **37B**；使用 MLA 和 MTP（Lecture 3）

### 7.2 R1-Zero：纯 RL 的概念验证

**直接在 V3-Base 上用 GRPO + 推理数据训练，完全不做 SFT**。Prompt 模板：

```
A conversation between User and Assistant. The user asks a question, and the Assistant
solves it. The assistant first thinks about the reasoning process in the mind and then
provides the user with the answer. The reasoning process and answer are enclosed within
<think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think> <answer> answer here </answer>.
User: <reasoning query>
Assistant:
```

Reward = 格式 + 正确性（都是规则判断）。

| 好处 | 问题 |
|---|---|
| **不需要任何 SFT 数据就获得了推理能力**：AIME 2024 的 pass@1 从 15.6% 涨到 71.0%，cons@64 达到 86.7% | 推理链有**格式和可读性问题** |
| 推理长度随训练自然增长，出现自我反思、验证等行为 | **中英文混杂**，难以阅读 |

R1-Zero 的意义：证明了**推理能力可以纯粹通过 RL 激发出来**，不需要人类示范怎么推理。

### 7.3 R1：完整的四阶段训练

| 阶段 | 做法 | 数据 / Reward | 目的 |
|---|---|---|---|
| **1. 小规模 SFT（cold start）** | 用少量（数千条）长 CoT 数据做 SFT | 由 R1-Zero 等生成长推理链，再**人工改写**成可读的格式 | 让模型一开始就学会可读的推理格式；RL 收敛更快 |
| **2. GRPO（推理数据）** | 与 R1-Zero 几乎相同的 RL 流程 | Reward = 格式 + 正确性 + **语言一致性**（推理过程中目标语言词语的比例） | 提升推理能力，同时避免语言混杂（语言一致性奖励会略微降低分数，但可读性更好） |
| **3. 大规模 SFT（推理 + 非推理数据）** | 用阶段 2 的模型做 **rejection sampling**：每道题生成多个回答，只保留正确的（可验证的题用规则判，其他用 V3 当 judge），并过滤掉语言混杂、过长、格式差的推理 | 约 **60 万**条推理样本（数学、代码、逻辑）+ 约 **20 万**条通用样本（写作、问答等，主要复用 V3 的 SFT 数据） | 在保持推理能力的同时，恢复通用能力 |
| **4. GRPO（推理 + 非推理数据）** | 面向所有场景的 RL | 推理数据：格式 + 正确性；通用数据：**helpfulness + harmlessness** 的 reward model（主要复用 V3 的 RL 数据） | 让模型既会推理，又是一个好用、安全的助手 |

**结果**：R1 在 AIME 2024 上 pass@1 约 79.8%，MATH-500 约 97.3%，CodeForces 等级分约 2029（超过约 96% 的人类参赛者），与 OpenAI o1 相当。

---

## 8. Distillation（蒸馏）

| | Lecture 2 中的 distillation | 这里的 distillation（R1-Distill） |
|---|---|---|
| 方法 | 学生匹配老师的**下一个 token 分布**（soft targets） | 老师**生成完整回答**（含推理链），学生用 **SFT** 学习这些推理轨迹 |
| 目标 | 逐 token 模仿概率分布 | 学会推理**行为** |
| 需要老师的 | 每个位置的概率分布（logits） | 只需要老师生成的文本 |

**R1-Distill**：用 R1 生成的约 80 万条样本，对 Qwen2.5（1.5B、7B、14B、32B）和 LLaMA（8B、70B）做 SFT，**只做 SFT，不做 RL**。

**结果**：
- R1-Distill-Qwen-7B：AIME 2024 约 55.5%
- R1-Distill-Qwen-32B：AIME 2024 约 72.6%，MATH-500 约 94.3%，在多项 benchmark 上超过 o1-mini
- **对比实验**：在同一个 Qwen-32B 基础模型上直接做大规模 RL，AIME 只有约 47%；而蒸馏得到 72.6%

**结论**：
- 小模型的推理能力可以通过蒸馏获得，而且很有竞争力
- 是一种“好的”算力用法：**对小模型来说，从强模型蒸馏比自己做大规模 RL 更有效、更便宜**
- 但要**超越**老师，仍然需要在更强的基础模型上做 RL

---

## 9. 要点回顾

1. Reasoning model = 大规模的 CoT，推理链由 RL 训练出来，表现出规划、验证、回溯等行为
2. Test-time scaling：推理时想得越久（更多 token、更多采样），效果越好；思考 token 也要计费
3. 推理 benchmark 的共同点是答案可自动验证（测试用例、标准答案）；pass@k 用无偏估计 $1 - \binom{n-c}{k}/\binom{n}{k}$，cons@k 是多数投票
4. RLVR：reward = 格式 + 正确性，用规则判断，难以被 hack，但只适用于可验证的领域
5. 控制思考长度：动态预算、把预算告诉模型、budget forcing（插入 “Wait”）、在连续空间中思考
6. GRPO 用组内平均 reward 做 baseline，省掉 value model；全对或全错的题没有学习信号
7. 按回答长度归一化会让答错的长回答受罚更少，导致回答越来越长；DAPO（token 级 loss 等）和 Dr. GRPO（去掉长度与标准差归一化）修正了它
8. DeepSeek R1：R1-Zero 证明纯 RL 可行；R1 用“冷启动 SFT → RL → 拒绝采样 SFT → 全场景 RL”四阶段兼顾推理与可读性；蒸馏让小模型高效获得推理能力

---

## 10. 自测题

<details>
<summary>1. 对一道题采样 n = 8 次，c = 2 次正确。pass@1 和 pass@2 分别是多少？</summary>

pass@1 = 1 − C(6,1)/C(8,1) = 1 − 6/8 = 0.25。pass@2 = 1 − C(6,2)/C(8,2) = 1 − 15/28 ≈ 0.46。
</details>

<details>
<summary>2. 为什么训练推理能力时，DeepSeek 选择规则式 reward 而不是神经网络 reward model？</summary>

神经网络 reward model 有漏洞，长时间 RL 会让 policy 找到并利用这些漏洞（reward hacking）。规则式 reward（答案是否正确、格式是否符合）几乎没有可钻的空子，而且便宜、精确。
</details>

<details>
<summary>3. GRPO 相比 PPO 最大的区别是什么？为什么这对长推理链特别有用？</summary>

GRPO 不需要 value model，而是用同一题多个回答的平均 reward 作为 baseline。长推理链只有结尾有 reward，很难训练出准确的 token 级 value model；省掉它还节省了与 policy 同样大的显存和计算。
</details>

<details>
<summary>4. 解释为什么“按回答长度归一化”会让答错的回答越来越长。</summary>

每个 token 的梯度权重是 Â/|o|。答错时 Â < 0，回答越长，每个 token 分到的惩罚越小，所以模型被激励在答不出时写得更长。DAPO 改用 token 级平均，Dr. GRPO 去掉长度归一化，使每个 token 的惩罚相同。
</details>

<details>
<summary>5. Budget forcing 如何让模型“多想一会儿”？</summary>

当模型准备输出思考结束标记时，把它替换成 “Wait”，模型会接着写下去，通常会重新检查之前的推理，从而纠正错误。
</details>

<details>
<summary>6. R1-Zero 有什么问题？R1 是怎么解决的？</summary>

R1-Zero 的推理链可读性差、中英文混杂。R1 先用少量人工改写过的可读长 CoT 做冷启动 SFT，在 RL 中加入语言一致性奖励，再通过拒绝采样过滤掉语言混杂、格式差的样本做大规模 SFT。
</details>

<details>
<summary>7. 这里的 distillation 与 BERT 时代的 distillation 有什么不同？</summary>

BERT 时代（DistilBERT）是让学生逐 token 匹配老师的概率分布；R1-Distill 是让老师生成完整的推理回答，学生用普通 SFT 学习这些文本，只需要老师的输出文本，不需要它的概率分布。
</details>

---

## 参考论文

- Wei et al., “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models”, 2022
- Chen et al., “Evaluating Large Language Models Trained on Code” (HumanEval, pass@k), 2021
- OpenAI, “Learning to Reason with LLMs” (o1), 2024
- DeepSeek-AI, “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning”, 2025；“DeepSeek-V3 Technical Report”, 2024
- Han et al., “Token-Budget-Aware LLM Reasoning”, 2024
- Muennighoff et al., “s1: Simple test-time scaling”, 2025
- Hao et al., “Training Large Language Models to Reason in a Continuous Latent Space” (Coconut), 2024
- Shao et al., “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models” (GRPO), 2024
- Yu et al., “DAPO: An Open-Source LLM Reinforcement Learning System at Scale”, 2025
- Liu et al., “Understanding R1-Zero-Like Training: A Critical Perspective” (Dr. GRPO), 2025
