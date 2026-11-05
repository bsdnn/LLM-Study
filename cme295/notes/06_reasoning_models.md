# Lecture 6 · Reasoning Models：用 RL 扩展推理能力

> Stanford CME 295：Transformers & Large Language Models
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture6.pdf`](../slides/fall25-cme295-lecture6.pdf)

## 本讲脉络

```
普通 LLM 的长处与短处 → 什么是 reasoning → 从 CoT 到 reasoning model
→ Benchmark 与指标（pass@k、cons@k）→ 用 RL + verifiable rewards 训练 → 控制思考长度
→ GRPO（与 PPO 对比）→ “回答越来越长”现象与修正（DAPO、Dr. GRPO）
→ DeepSeek R1-Zero 与 R1 的完整训练流程 → Distillation
```

一句话总结：**Reasoning model = 大规模的 Chain-of-Thought，而且推理过程是用 RL 学出来的，不是人写的。** 数学和代码题有天然的“对/错”可验证 reward，用 GRPO 优化后，模型自发学会长时间思考、自我检查。这开辟了 **test-time scaling**：推理时多花 token，换更好的答案。

---

## 1. 普通 LLM 的长处与短处

| 长处 | 短处 |
|---|---|
| 擅长模仿和产生想法 | **推理能力有限**（本讲） |
| 生成和调试代码很强 | 知识是静态的（Lecture 7：RAG） |
| | 不能执行动作（Lecture 7：tools / agents） |
| | 难以评估（Lecture 8） |

---

## 2. 什么是 Reasoning

**暂定定义**：Reasoning = 解决问题的能力。

| 不是 reasoning（知识检索） | 是 reasoning（需要多步推导） |
|---|---|
| “What is the course code of Stanford's Transformers & LLMs class?” | “The bear was born in 2020. How old is this bear now?” |

### 核心思路：Chain-of-Thought

回顾 CoT（Wei et al., 2022）：让模型在回答前先解释推理过程。
- 不用 CoT：Q: How old will the bear be next year? → A: 5.（直接猜）
- 用 CoT：A: It will be one year older than its age this year, which was 4. Hence, it will be 5.

**Reasoning model 的想法：把 CoT 做到大得多的规模。**

| 以前 | 新范式 |
|---|---|
| Question → Answer | Question → **Reasoning chain** → Answer |
| | 输出 = Reasoning + Answer |

### 发展时间线

| 日期 | 发布 |
|---|---|
| 2024-09-12 | OpenAI o1-preview |
| 2024-12-19 | Gemini 2.0 Flash Thinking |
| 2025-01-20 | DeepSeek R1 |
| 2025-02-19 | Grok 3 Beta |
| 2025-02-24 | Claude 3.7 Sonnet |
| 2025-06-10 | Magistral（Mistral） |

### 如何识别 reasoning model

- 界面上通常显示 **“thought summary”**（思考摘要），完整的 chain of thought 往往被隐藏
- API 中有控制思考的参数（如 reasoning effort、thinking budget），**思考 token 也要计费**（通常按 output token 价格）

---

## 3. Reasoning 相关的 Benchmark

**代码**：解编程题或修 bug
- 例：“有 n 只 teddy bear 排成一行，各有大小。找出比最大那只小的最大的那只。”模型写出 `second_biggest_bear(bears)` 函数
- **验证**：运行 t 个测试用例，全部通过才算对
- 例：HumanEval、CodeForces、SWE-bench

**数学**：解难题（如奥赛）
- 例：“The bear was born in 2020. How old is the bear now?” → 推理 “It is 2025 now. 2025 − 2020 = 5.” → Answer: 5
- **验证**：和 ground truth 比较
- 例：AIME、GSM8K

关键共同点：**答案可以自动验证**。

### 指标

**Pass@k**（Chen et al., 2021）：k 次尝试中**至少一次成功**的概率。

无偏估计：生成 n ≥ k 个样本，其中 c 个正确，

```
pass@k = 1 − C(n − c, k) / C(n, k)
```

（C 为组合数。分子是“抽 k 个全部错误”的方法数。）

- **Pass@k**：适合容易检查答案、或能接受更高延迟的场景（多生成几次，挑对的）
- **Pass@1**：只关心一次生成的场景，是最常报告的特例
- **Cons@k**（consensus at k，DeepSeek-R1 论文）：生成 k 个答案，**多数投票**得到的答案与 ground truth 比较（即 self-consistency 的准确率）

---

## 4. 发展 “Test-Time Scaling”

**想法**：激励模型在回答前先推理。

**考虑**：
- 推理链很难从零写出来 —— **手写 SFT 数据不现实**
- 不想把模型限制在人类写的推理方式上（模型可能找到更好的思路）
- 这类问题有**天然可验证的 reward**：“解出来了吗？” → 是 / 否

→ **用 RL！**

### 两种 verifiable reward

1. **Formatting reward**：检查 CoT 是否存在，即输出是否符合模板 `<think> … </think> ANSWER`
2. **Accuracy reward**：检查答案是否正确
   - 代码：跑测试用例
   - 数学：与 ground truth 比较

```
Reward = r_formatting（有 think 分隔符吗？）+ r_accuracy（答案正确吗？）
```

这种用规则直接判断对错的方式叫 **RLVR（RL with Verifiable Rewards）**。好处是不需要训练 reward model，也很难被 hack。

### 推理时控制“思考”

**问题**：不是所有 prompt 都一样难，简单问题想太久浪费，难题想太少答错。

| 方法 | 思路 |
|---|---|
| **Dynamic budget** | 根据问题难度给不同的 token 预算 |
| **Context awareness**（Token-Budget-Aware LLM Reasoning，Han et al., 2024） | 把预算写进 prompt，让模型知道自己有多少 token 可用 |
| **Budget forcing**（s1，Muennighoff et al., 2025） | 想停止思考时强制插入结束标记；想让它多想时，在它要结束时追加 “Wait”，迫使模型继续检查 |
| **“Continuous” thoughts**（Coconut，Hao et al., 2024） | 不把思考写成离散 token，而是在连续的 latent space 中推理，把隐藏状态直接作为下一步输入 |

---

## 5. GRPO（Group Relative Policy Optimization）

来自 DeepSeekMath（Shao et al., 2024），是训练 reasoning model 最常用的 RL 算法。

### 做法

对每个问题 q，用旧 policy 采样一组 G 个回答 o₁, …, o_G，得到 reward r₁, …, r_G。

**Advantage ≈ Reward − 组内平均 reward**（再除以组内标准差）：

```
Â_i = ( r_i − mean(r₁…r_G) ) / std(r₁…r_G)
```

这个 advantage 赋给回答 o_i 中的**每一个 token**。

**目标函数**：

```
J(θ) = E [ (1/G) Σ_i (1/|o_i|) Σ_t min( ratio_{i,t} · Â_i,  clip(ratio_{i,t}, 1−ε, 1+ε) · Â_i ) ]
       − β · KL( π_θ ‖ π_ref )
```

### 与 PPO 的比较

| | PPO | GRPO |
|---|---|---|
| Ratio（新/旧 policy） | ✅ | ✅ 相同 |
| Clipping | ✅ | ✅ 相同 |
| KL penalty | 加在 reward 里（每个 token） | **直接加在 loss 里** |
| Advantage 估计 | 需要训练 **value model**，用 GAE 算 | **用组内其他回答的 reward 作为 baseline**，不需要 value model |

**最大区别**：省掉了 value model。value model 通常和 policy 一样大，省掉它能大幅节省显存和计算，也去掉了一个难训练的组件。

直觉：同一题的多个回答互为参照。答对的回答比组平均好 → advantage 为正，所有 token 的概率被提高；答错的则被降低。如果一组回答全对或全错，advantage 全为 0，这题提供不了学习信号。

---

## 6. “回答越来越长”现象

**观察**：随着 RL 训练进行，回答长度持续增加。部分原因是好的（模型学会了更多检查和反思，如出现 “aha moment”），部分原因是**损失函数中的偏差**。

**问题根源**：GRPO 的 loss 先在每个回答内部按长度 |o_i| 平均，再在组内平均。
- 对**答对**的回答（advantage > 0）：短回答中每个 token 分到的梯度更大 → 鼓励简短
- 对**答错**的回答（advantage < 0）：长回答中每个 token 受到的惩罚被稀释 → **错的时候写长一点，惩罚更小**
- 结果：模型被激励在答不出来时拼命写长 —— 坏的激励

**修正：让每个 token 的贡献相等**
- **DAPO**（Yu et al., 2025）：token-level loss，把一个 batch 内所有 token 放在一起平均（除以总 token 数），而不是先在每个回答内平均
- **Dr. GRPO**（Liu et al., 2025，“Understanding R1-Zero-Like Training”）：去掉按回答长度的归一化，也去掉按组内标准差的归一化

**其他调整**：
- 与题目难度相关的偏差：除以组内 std 会让太简单或太难的题（std 很小）权重过大
- 鼓励多样性（如 DAPO 的 clip-higher：放宽上限的 ε，给低概率 token 更多上升空间，防止 entropy 塌缩）
- ……还有很多

---

## 7. 把一切串起来：DeepSeek R1

```
V3-Base（传统预训练）─┬─► V3（传统 SFT + RL 的聊天模型）
                     ├─► R1-Zero（概念验证：纯 RL）
                     └─► R1（完整流程）
```

### 7.1 V3-Base

- 用“传统”方法预训练
- MoE：总参数约 **671B**，每 token 激活约 **37B**

### 7.2 R1-Zero：纯 RL 的概念验证

直接在 V3-Base 上用 GRPO + 推理数据训练，**完全不做 SFT**。Prompt 模板：

```
A conversation between User and Assistant. The user asks a question, and the Assistant
solves it. The assistant first thinks about the reasoning process in the mind and then
provides the user with the answer. The reasoning process and answer are enclosed within
<think> </think> and <answer> </answer> tags, respectively, i.e.,
<think> reasoning process here </think> <answer> answer here </answer>.
User: <reasoning query>
Assistant:
```

| 好处 | 问题 |
|---|---|
| **不需要任何 SFT 就获得了推理能力**；推理长度随训练自然增长，出现自我反思 | 推理链存在**格式和可读性问题**（中英文混杂、难以阅读） |

### 7.3 R1：完整的四阶段训练

| 阶段 | 做法 | 数据 / Reward |
|---|---|---|
| **1. 小规模 SFT（cold start）** | 用推理数据做 SFT，让模型先学会可读的推理格式 | 由 R1-Zero 生成长 CoT，再由人工改写 |
| **2. GRPO（推理数据）** | 与 R1-Zero 几乎相同的 RL 流程 | Reward = formatting + accuracy + **language consistency**（推理语言要与问题一致） |
| **3. 大规模 SFT（推理 + 非推理数据）** | 用阶段 2 的模型做 **rejection sampling**：生成多个回答，用规则和 V3 作为 judge 筛选，只保留好的 | 约 **600k** 推理样本（数学、代码、逻辑）+ 约 **200k** 通用样本（主要复用 V3 的 SFT 数据） |
| **4. GRPO（推理 + 非推理数据）** | 全场景 RL | 推理数据：formatting + accuracy；通用数据：helpfulness + harmlessness（主要复用 V3 的 RL 数据） |

**结果**：R1 在数学和代码推理 benchmark 上与 OpenAI o1 相当。

---

## 8. Distillation（蒸馏）

| | Lecture 2 中的 distillation | 这里的 distillation（R1-Distill） |
|---|---|---|
| 方法 | 学生匹配老师的**下一个 token 分布**（soft targets） | 老师**生成完整回答**（含推理链），学生用 **SFT** 学习这些推理轨迹 |
| 目标 | 逐 token 模仿概率 | 学会推理行为 |

**R1-Distill**：用 R1 生成约 800k 条数据，对 Qwen 和 LLaMA 系列的小模型做 SFT。

**结果**：
- 小模型的推理能力很有竞争力
- 是一种“好的”算力用法：对小模型来说，**从强模型蒸馏比自己直接做大规模 RL 更有效、更便宜**

---

## 参考论文

- Wei et al., “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models”, 2022
- Chen et al., “Evaluating Large Language Models Trained on Code” (pass@k, HumanEval), 2021
- DeepSeek-AI, “DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning”, 2025；“DeepSeek-V3 Technical Report”, 2024
- Han et al., “Token-Budget-Aware LLM Reasoning”, 2024
- Muennighoff et al., “s1: Simple test-time scaling”, 2025
- Hao et al., “Training Large Language Models to Reason in a Continuous Latent Space”, 2024
- Shao et al., “DeepSeekMath” (GRPO), 2024
- Yu et al., “DAPO: An Open-Source LLM Reinforcement Learning System at Scale”, 2025
- Liu et al., “Understanding R1-Zero-Like Training: A Critical Perspective” (Dr. GRPO), 2025
