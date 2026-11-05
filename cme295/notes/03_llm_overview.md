# Lecture 3 · LLM 概览：MoE、解码、Prompting 与推理优化

> Stanford CME 295：Transformers & Large Language Models
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture3.pdf`](../slides/fall25-cme295-lecture3.pdf)

## 本讲脉络

```
LLM 是什么 → MoE（Mixture of Experts）→ 如何生成回复（greedy / beam / sampling / temperature / guided decoding）
→ Prompting 策略（结构、ICL、CoT、self-consistency）
→ 推理优化（KV cache、GQA、PagedAttention、MLA、speculative decoding、MTP）
```

一句话总结：现代 LLM 是**超大的 decoder-only Transformer**，每次预测一个 token。本讲回答三个问题：怎么在不增加计算的前提下把模型做大（MoE）、怎么从概率分布中挑出 token（decoding）、怎么让推理更快更省（KV cache 及一系列优化）。

---

## 1. 什么是 LLM

**Language model**：一个给 token 序列分配概率的统计或机器学习模型，即估计 p(x₁, x₂, …, x_n)，通常分解为逐个预测下一个 token：

```
p(x₁, …, x_n) = Π p(x_t | x₁, …, x_{t−1})
```

**Large** 体现在：
- 模型规模：数十亿（billions）参数以上
- 训练数据：数千亿（hundreds of billions）tokens 以上
- 算力：大量 GPU

**结构**：几乎都是 **decoder-only Transformer**。例：GPT 系列、LLaMA、Gemma、DeepSeek、Mistral、Qwen……

---

## 2. MoE（Mixture of Experts）

### 2.1 动机

模型越大效果越好，但每个 token 都要经过全部权重，计算量随参数线性增长。
**观察**：并不是所有权重对每个输入都有用。能不能让每个 token 只用模型的一个子集？

### 2.2 结构

- 有 n 个 **expert**（E₁, …, E_n），每个是一个子网络
- 一个 **gating network / router（G）** 根据输入决定每个 expert 的权重

两种形式：
- **Dense MoE**：输出是**所有** expert 输出的加权平均（权重如 0.1、0.8、0.05…）—— 计算并没有省
- **Sparse MoE**（Shazeer et al., 2017）：只选 **top-k** 个 expert，输出是被选中 expert 的加权平均 —— 其他 expert 根本不计算

```
G(x) = softmax(TopK(x · W_g))
y = Σ_{i ∈ TopK} G(x)_i · E_i(x)
```

### 2.3 在 Transformer 中的位置

- 把每个 Transformer block 中的 **FFN 换成 n 个 FFN expert（FFNN₁ … FFNN_n）+ router**
- Attention 层保持不变
- **路由是按 token 进行的**：同一句话中的不同 token 可能去不同的 expert

结果：总参数量很大（存储需要），但每个 token 实际激活的参数量（计算需要）只是一小部分。

### 2.4 训练难点：Routing collapse

**症状**：router 总是选中同一个（或少数几个）expert。被选中的 expert 训练得更好 → 更容易被选中 → 恶性循环；其他 expert 得不到训练，等于浪费了参数。

**解决**：加一个 **auxiliary load-balancing loss**（Switch Transformers，Fedus et al., 2021），强迫其他 expert 也“参与游戏”：

```
L_aux = α · n · Σᵢ fᵢ · Pᵢ
fᵢ = 被路由到 expert i 的 token 比例
Pᵢ = 所有 token 对 expert i 的平均路由概率
```

当路由完全均匀时（每个 fᵢ = Pᵢ = 1/n），这个 loss 最小。

### 2.5 Expert 学到了什么？

Mixtral 论文（Jiang et al., 2024）把每个 token 按其被分配的 expert 上色。发现 expert **并不按“主题”分工**（不存在“生物 expert”“数学 expert”），而更多是按**语法/形式**特征分工，例如某些 expert 偏好处理缩进、标点或特定 token 类型。

---

## 3. 如何生成回复（Decoding）

LLM 的输出是下一个 token 在整个词表上的概率分布。给定 “[BOS] A teddy bear is”，模型预测下一个 token 的分布，然后需要一个**解码策略**从中挑一个 token，接到输入后面，再预测下一个，直到生成 `[EOS]`。

### 3.1 Greedy decoding

- 每一步选概率最高的 token
- 局限：每步最优 ≠ 整句最优；输出不自然、缺乏多样性，容易陷入重复

### 3.2 Beam search

- 每一步保留 **k 条（beam width）最可能的路径**，扩展后再保留总概率最高的 k 条，直到所有路径以 `[EOS]` 结束
- 例：“a” → {cute 0.7, fluffy 0.2…}，“the” → {sleepy 0.6…}，最后在 “dog / kitten / sloth” 等结尾中选整体概率最高的
- 局限：计算更多；仍然缺乏多样性和创造力（倾向于平淡、高频的句子）。适合翻译这类“有标准答案”的任务

### 3.3 Sampling

- 按概率分布**随机采样**下一个 token：x_t ~ p(· | x<t)
- 多样、更像人类写作，但可能采到低概率的离谱 token

**两种限制采样范围的策略**：
- **Top-k**：只在概率最高的 k 个 token 中采样（如 k = 4），先重新归一化
- **Top-p（nucleus sampling）**：只在**累积概率 ≥ p 的最小 token 集合**中采样（如 p = 90%）。模型很确定时集合很小，不确定时集合变大，比固定 k 更自适应

### 3.4 概率从哪来：Softmax 与 Temperature

模型最后一层输出 logits zᵢ，用带温度 T 的 softmax 转成概率：

```
pᵢ = exp(zᵢ / T) / Σⱼ exp(zⱼ / T)
```

- **T 小（→ 0）**：分布变尖，几乎总选最高分 token，接近 greedy，输出稳定、保守
- **T = 1**：原始分布
- **T 大**：分布变平，低概率 token 也有机会，输出更多样、更有创意，也更容易出错

推荐阅读：He et al., “Defeating Nondeterminism in LLM Inference”, 2025 —— 即使 T = 0，由于 GPU 浮点运算顺序和 batch 组成的不同，输出也可能不完全确定。

### 3.5 Guided decoding（约束解码）

**动机**：需要输出特定格式，例如要求“用 JSON 描述我 33 岁、喜欢阅读的 teddy bear”：

```json
{"first_name": "teddy", "last_name": "bear", "age": 33, "hobby": "reading"}
```

仅靠 prompt 要求，模型仍可能输出不合法的 JSON。

**思路**：在每一步解码时**只允许“合法”的下一个 token**。根据语法（如 JSON schema 或正则表达式）维护一个状态机，把不合法 token 的概率设为 0（mask 掉）。例如在 `{` 之后只能是 `"first_name"` 这样的 key，而不能是 “road” 或 “sun”。这样输出**保证**符合格式。

---

## 4. Prompting 策略

### 4.1 Context length

- 一次请求中，输入 prompt + 输出 共同占用 **context window**
- 数量级因输入类型而异（一句话约几十 tokens，一本书约十几万 tokens，代码库更大）
- 小心 **“context rot”**（Hong et al., 2025）：输入 token 越多，模型表现越差，即使信息都在 context 里

### 4.2 Prompt 的结构

一个好的 prompt 通常包括：

| 部分 | 例子 |
|---|---|
| **Context** | My teddy bear had a long day and needs a bedtime story. |
| **Instructions** | Generate a bedtime story that takes place in a specific location. |
| **Input** | Location: Country of teddy bears |
| **Constraints** | The story needs to be suitable for teddy bears that are tired. |

### 4.3 In-Context Learning（ICL）

GPT-3 论文（Brown et al., 2020，“Language Models are Few-Shot Learners”）发现：**不更新任何权重**，只在 prompt 中给例子，模型就能学会新任务。

| | Zero-shot | Few-shot |
|---|---|---|
| 做法 | 直接问，不给例子 | prompt 中给几组输入/输出例子 |
| 效果 | 很依赖模型本身的能力 | 通常更好 |

给例子通常更好，但：需要人工准备；增加计算和成本；增加延迟。

### 4.4 Chain-of-Thought（CoT，Wei et al., 2022）

**思路**：让模型先写出推理过程再给答案，能提升效果。

例（few-shot CoT）：
- 示例：Q: How old is this bear? A: The bear was born in 2020. It is therefore 4.
- 问题：Q: How old will the bear be next year?
- 模型：A: It will be one year older than its age this year, which was 4. Hence, it will be **5**.

为什么有效：每个 token 的计算量是固定的，难题需要更多“计算步骤”。让模型把中间步骤写出来，相当于给了它更多计算，而且后面的 token 可以基于前面写下的中间结果。

讨论：✅ 可解释、有解释过程；❌ 更多 tokens → 更高成本和延迟。

### 4.5 Self-Consistency（Wang et al., 2022）

**思路**：对同一问题采样多条 CoT 推理路径，再对最终答案**多数投票**。

例：三条路径分别得出 5、5、4 → 取 5。

讨论：在效果和额外成本（多次调用）之间权衡。

---

## 5. 推理优化

### 5.1 挑战与分类

计算很昂贵，有很多可以优化的维度：

| “精确”优化（结果不变） | 近似优化（可能影响结果） |
|---|---|
| 避免冗余计算（KV cache） | 架构改动（GQA、MLA） |
| 内存管理（PagedAttention） | Embedding 表示（如量化） |
| 改写数学形式（如 FlashAttention） | Token 预测方式（speculative decoding、MTP） |
| | 基于 attention 的技巧 |

### 5.2 KV Caching

**动机**：自回归生成时，新 token 要和**所有之前的 token** 做 attention。如果每一步都重新计算所有之前 token 的 K 和 V，会有大量重复计算。

**思路**：之前 token 的 K、V 不会变，把它们**存在 cache 里**。生成新 token 时，只计算新 token 的 q、k、v，然后把新的 k、v 追加到 cache，用新 q 和 cache 中所有 K、V 做 attention。

代价：cache 占用大量显存，大小 ∝ 层数 × KV head 数 × head 维度 × 序列长度 × batch。后面几种方法都在想办法缩小它。

### 5.3 共享 attention heads：GQA

- 标准 MHA：query head 数 = key head 数 = value head 数 = h
- **GQA**：把 h 个 query head 分成 G 组（G < h），每组共享一个 K/V head，cache 缩小 h/G 倍
- **MQA**：G = 1 的极端情况

### 5.4 PagedAttention（Kwon et al., 2023，vLLM）

**观察**：传统做法为每个请求预留一整块**连续**的显存来放 KV cache（按最大长度预留），实际生成往往更短，造成大量浪费（内部碎片、外部碎片、预留但未使用）。

**思路**：借鉴操作系统的**虚拟内存分页**，把 KV cache 切成固定大小的 block，存放在**不连续**的物理空间中，用一张 block table 记录逻辑位置到物理位置的映射。按需分配，几乎没有浪费，同样显存能服务更多并发请求；相同前缀的请求还能共享 block。

### 5.5 Latent attention（MLA，DeepSeek-V2）

**目标**：缩小存在显存中的 K、V 的维度。

- 原本每层每个 token 要存 h 个 head 的 K 和 V（太大）
- **MLA（Multi-head Latent Attention）**：把 token 的 hidden state 压缩成一个**低维 latent 向量 c**，只缓存 c；计算 attention 时再用上投影矩阵从 c 恢复出各 head 的 K 和 V
- 这些上投影矩阵可以在推理时与 query / output 投影合并，几乎不增加计算
- 效果：KV cache 大幅缩小，同时质量与 MHA 相当甚至更好

### 5.6 Speculative Decoding（Chen et al., 2023）

**问题**：大模型每次只能生成一个 token，而一次 forward 主要耗时在读取权重上（memory-bound），GPU 算力大量闲置。

**思路**：
1. 用一个小而快的 **draft model** 连续猜 k 个 token（如 “and smart …”），并记下它的概率 P₁…P_k
2. 把这 k 个 token 一次性喂给大的 **target model**，**一次 forward 并行**得到每个位置的目标概率 Q₁…Q_{k+1}
3. 从左到右逐个验证每个猜测的 token x：
   - 若 Qᵢ(x) ≥ Pᵢ(x)：✅ 接受
   - 否则：以概率 Qᵢ(x) / Pᵢ(x) 接受，以概率 1 − Qᵢ(x)/Pᵢ(x) ❌ 拒绝
4. 一旦拒绝，就从修正分布 **[Qᵢ − Pᵢ]₊**（归一化后）重新采样这个位置的 token，然后结束本轮；若全部接受，还能额外从 Q_{k+1} 采样一个 token

**关键性质**：这个接受/拒绝规则保证最终输出的分布**与只用 target model 采样完全相同**，是无损加速。猜得越准，每次大模型 forward 产出的 token 越多。

### 5.7 Multi-Token Prediction（MTP，Gloeckle et al., 2024）

**思路**：训练模型时增加 k 个预测头，在位置 t 同时预测 t+1, t+2, …, t+k。推理时：
- 额外的头可以充当 draft，然后由主头验证 —— **draft model 和 target model 是同一个模型**，不需要单独维护小模型
- 训练时多 token 目标本身也能提升模型质量

### 5.8 小结

| 挑战 | 解决方法 |
|---|---|
| KV cache 太大 | GQA、Latent attention（MLA）、PagedAttention |
| 逐 token 解码太慢 | Speculative decoding、Multi-token prediction |

---

## 参考论文

- Shazeer et al., “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”, 2017
- Fedus et al., “Switch Transformers”, 2021；Jiang et al., “Mixtral of Experts”, 2024
- He et al., “Defeating Nondeterminism in LLM Inference”, 2025
- Hong et al., “Context Rot”, 2025
- Brown et al., “Language Models are Few-Shot Learners”, 2020
- Wei et al., “Chain-of-Thought Prompting”, 2022；Wang et al., “Self-Consistency”, 2022
- Kwon et al., “PagedAttention”, 2023；DeepSeek-V2, 2024
- Chen et al., “Accelerating LLM Decoding with Speculative Sampling”, 2023
- Gloeckle et al., “Better & Faster LLMs via Multi-token Prediction”, 2024
