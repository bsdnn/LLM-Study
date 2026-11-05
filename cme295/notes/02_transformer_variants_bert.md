# Lecture 2 · Transformer 改进技巧与 BERT

> Stanford CME 295：Transformers & Large Language Models
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture2.pdf`](../slides/fall25-cme295-lecture2.pdf)

## 本讲脉络

```
Position embeddings（learned → sinusoidal → relative bias → RoPE）
→ Layer normalization（Post-LN → Pre-LN → RMSNorm）
→ Attention approximation（sparse / sliding window / MQA / GQA）
→ Transformer 模型三大类（encoder-decoder / encoder-only / decoder-only）
→ BERT 深入（预训练任务、架构、fine-tuning、DistilBERT、RoBERTa）
```

一句话总结：原始 Transformer（2017）之后，社区在**位置编码、归一化、attention 计算效率**三个方向上做了大量改进，今天的 LLM 基本标配 **RoPE + Pre-Norm RMSNorm + GQA**。而 BERT 代表了 2018–2022 年流行的 encoder-only 路线：先用代理任务预训练，再针对下游任务 fine-tune。

---

## 0. 回顾：Attention map

原论文中可视化了 attention 权重：第 5 层（共 6 层）的两个 head 似乎在做 **anaphora resolution（指代消解）** —— 对于单词 “its”，这两个 head 把注意力集中在它所指代的名词上。这说明不同 head 确实学到了不同的语言学功能。

---

## 1. Position Embeddings

### 1.1 为什么需要位置信息

Self-attention 让每个 token 与所有 token **直接相连**，这些“直接连接”本身不带顺序信息。如果不额外提供位置，模型分不出 “dog bites man” 和 “man bites dog”。

### 1.2 Learned position embedding

- 做法：为每个位置 0, 1, …, max_len−1 学一个向量，加到 token 向量上
- 局限：**只能处理训练时见过的最大长度**，想支持更长序列就得重新训练

### 1.3 Hardcoded（sinusoidal）position embedding

原 Transformer 的方案：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

- 每两个维度一组，用一个频率；低维频率高（变化快），高维频率低（变化慢）
- 把两个位置编码做点积，得到的“相似度”随距离增大而下降 —— 相近的位置更相似
- 好处：**是一个公式，可以扩展到任意长度**

### 1.4 从绝对位置到相对位置

**观察**：attention 计算的是 query 与 key 的关系，真正重要的往往是“两个 token 相距多远”（**relative position**），而不是“它在第 37 个位置”（absolute position）。

**思路**：不在输入上加位置，而是**直接改 attention 层**。

### 1.5 在 attention score 上加 bias

在 `q_i · k_j` 上加一个只依赖于距离 (i − j) 的偏置：

```
score(i, j) = q_i · k_j / √d + b(i − j)
```

- **T5 bias**（Raffel et al.）：bias 是**按 head 学习**的，把距离分桶（bucket），每个桶一个可学习标量
- **ALiBi**（Attention with Linear Biases，Press et al., 2021）：bias 是**线性、确定性、无上界**的：`b(i − j) = −m · |i − j|`，每个 head 用不同的斜率 m。距离越远惩罚越大
  - 论文标题 “Train Short, Test Long”：在短序列上训练，可以外推到更长序列

### 1.6 RoPE（Rotary Position Embeddings，Su et al., 2021）——当前默认选择

**思路**：用旋转矩阵**旋转 query 和 key 向量**，旋转角度与位置成正比。

二维情况下，位置 m 的旋转矩阵：

```
R(m) = [ cos(mθ)  −sin(mθ) ]
       [ sin(mθ)   cos(mθ) ]

q'_m = R(m) · q_m,    k'_n = R(n) · k_n
```

**关键性质**：旋转后的点积

```
q'_m · k'_n = (R(m) q_m)ᵀ (R(n) k_n) = q_mᵀ · R(n − m) · k_n
```

只依赖于**相对距离 n − m**。也就是说，虽然每个向量是按绝对位置旋转的，但 attention score 自然地只感知相对位置。

**推广到 d > 2 维**：把向量切成 d/2 个二维块，每块用不同频率 θ_i = 10000^(−2i/d) 旋转（与 sinusoidal 的频率选择相同）。

**好处**：
- 相对距离被干净地编码进 attention
- **Long-term decay**：attention 权重的上界随相对距离增大而衰减 —— 远处的 token 天然影响更小，符合语言直觉
- 不增加参数；可以通过调整频率（如 NTK scaling、YaRN 等方法）扩展 context 长度

| 方法 | 位置信息加在哪 | 绝对/相对 | 能否外推 |
|---|---|---|---|
| Learned PE | 输入 embedding | 绝对 | 否 |
| Sinusoidal PE | 输入 embedding | 绝对（点积隐含相对） | 理论上可以 |
| T5 bias | attention score | 相对，按 head 学习 | 有限 |
| ALiBi | attention score | 相对，线性固定 | 好 |
| **RoPE** | 旋转 Q、K | 相对 | 较好，可通过缩放扩展 |

---

## 2. Layer Normalization

### 2.1 LayerNorm（Ba et al., 2016）

对**单个 token 的向量**在特征维度上做归一化：

```
LN(x) = γ ⊙ (x − μ) / √(σ² + ε) + β
其中 μ = (1/d) Σ xᵢ,   σ² = (1/d) Σ (xᵢ − μ)²
```

- γ、β 是可学习的缩放和平移参数
- 与 BatchNorm 不同：LayerNorm 不依赖 batch，对变长序列和小 batch 都适用
- 好处：**提升训练稳定性和收敛速度**（防止各层激活值的尺度漂移）

### 2.2 Post-Norm vs Pre-Norm（Xiong et al., 2020）

| | Post-LN（原始 Transformer） | Pre-LN（现在常用） |
|---|---|---|
| LayerNorm 位置 | 残差连接**之后**：`LN(x + Sublayer(x))` | 子层**之前**：`x + Sublayer(LN(x))` |
| 梯度 | 靠近输出层的梯度很大，训练不稳定，需要 learning rate warm-up | 残差路径上有一条“干净”的恒等通道，梯度更平稳 |
| 训练 | 更难训，尤其是深层 | 更稳定，可以去掉或缩短 warm-up |

### 2.3 RMSNorm（Zhang et al., 2019）

```
RMSNorm(x) = γ ⊙ x / √( (1/d) Σ xᵢ² + ε )
```

- 去掉了减均值（re-centering）和 β，只按 root mean square 缩放
- 计算更便宜，效果基本不变

**如今的默认组合：Pre-Norm + RMSNorm**（LLaMA 等模型都如此）。

---

## 3. Attention Approximation（让 attention 更便宜）

完整 self-attention 的计算和内存是 **O(n²)**（n = 序列长度），长文档时代价很高。

### 3.1 Sparse attention：Longformer（Beltagy et al., 2020）

不让每个 token 看所有 token，而是组合：
- **Local attention**：每个 token 只看窗口内的邻居
- **Global attention**：少数特殊 token（如 `[CLS]`）看全部、也被全部看到

复杂度从 O(n²) 降到 O(n · w)。

### 3.2 Sliding Window Attention（SWA）

- 每个 token 只看前 w 个 token
- 层数叠加后，**感受野**会扩大：第 k 层的 token 间接看到了 k × w 范围的信息 —— 类似 CNN 里的 **receptive field**
- 例：**Mistral 7B** 使用 SWA
- 变体：local 层与 global 层交替（interleave）

### 3.3 共享 KV heads：MQA 与 GQA

推理时要缓存每个 token 的 K、V（KV cache），head 越多缓存越大。思路：**让多个 query head 共享 key/value head**。

| | MHA（Multi-Head） | GQA（Grouped-Query） | MQA（Multi-Query） |
|---|---|---|---|
| KV heads 数 | = query heads 数 | 每组 query 共享一个 KV head | 所有 query 共享 1 个 KV head |
| KV cache | 最大 | 中等 | 最小 |
| 质量 | 最好 | 接近 MHA | 有一定下降 |

GQA 是质量与效率的折中，现代 LLM 广泛使用。

---

## 4. 三类 Transformer 模型

| 类别 | 结构 | 典型用途 | 代表模型 | 流行时间 |
|---|---|---|---|---|
| **Encoder-decoder** | 完整 Transformer | Text-to-text（翻译、摘要） | T5、mT5、ByT5 | — |
| **Encoder-only** | 只保留 encoder，把 embedding 投影到类别上 | 分类（情感分析、NER…） | BERT、DistilBERT、RoBERTa | 约 2018–2022 |
| **Decoder-only** | 只保留（masked）decoder，自回归生成 | 文本生成，几乎一切任务 | GPT 系列 | **现在最流行** |

---

## 5. BERT 深入

### 5.1 名字的由来

**BERT = Bidirectional Encoder Representations from Transformers**（Devlin et al., 2018）

- **Bidirectional**：encoder 的 self-attention 没有 mask，每个 token 同时看到左边和右边的上下文
- 对比：GPT 只看左边；ELMo（Peters et al., 2018 年 2 月）用两个方向的 LSTM 分别训练再拼接，并非真正的联合双向
- 彩蛋：ELMo 和 BERT 都是《芝麻街》（Sesame Street）的角色

### 5.2 思路与策略

**只保留 encoder，用它输出的 encoded embeddings 做预测。**

两步训练：
1. **Pretraining**：在大量无标注文本上用代理任务（MLM 和 NSP）训练
2. **Fine-tuning**：针对具体任务（如情感分析）在少量标注数据上微调

| 优点 | 缺点 |
|---|---|
| Fine-tuning 不需要很多数据 | 不适合文本生成等任务 |
| 效果好 | Fine-tuning 是必需步骤，每个任务都要一份模型 |

### 5.3 输入处理

- **Tokenizer**：WordPiece，事先在训练集上训练，词表约 **30,000**；擅长识别常见的词缀
- **特殊 token**：
  - `[CLS]`：放在输入最前面，它的最终向量用于整句级别的分类
  - `[SEP]`：分隔两个句段，并放在末尾
  - `[MASK]`：遮住要预测的 token
  - `[PAD]`：把短序列补齐到相同长度

### 5.4 Input embedding（三种相加）

1. **Token embedding**：巨大的查找表，每个词表项一个向量
2. **Position embedding**：帮助模型把 token 和位置关联（BERT 用的是 learned）
3. **Segment embedding**（BERT 新增）：标记 token 属于句子 A 还是句子 B，同一段共享一个向量

最终输入 = token + position + segment embedding。

### 5.5 预训练任务一：Masked Language Modeling（MLM）

随机选 **15%** 的输入 token 作为预测目标，其中：
- **80%** 替换为 `[MASK]`
- **10%** 替换为随机词
- **10%** 保持不变

为什么不全部换成 `[MASK]`？因为 fine-tuning 和实际使用时输入中**没有** `[MASK]`。如果预训练只在 `[MASK]` 位置学习，就会产生训练/使用不一致。随机替换和保持不变迫使模型对每个位置都保持警惕。

好处：
- 模型基于**双向上下文**学习语言建模
- 这种随机性起到 regularization 作用，也反映了语言本身的概率性

### 5.6 预训练任务二：Next Sentence Prediction（NSP）

从语料中取两个句子：
- 50% 的情况下它们是连续的（标签 IsNext）
- 50% 的情况下第二句是随机抽的（标签 NotNext）

任务：预测它们是否连续（用 `[CLS]` 的输出做二分类）。
- 让网络隐式学习句子间的上下文关系
- 是一个**不需要人工标注**的简单分类任务

（后来 RoBERTa 发现 NSP 基本没用。）

### 5.7 超参数与规模

- **L**：层数；**H**：hidden size（embedding 维度）；**A**：attention head 数
- 数据维度：语言特定 / 多语言；cased（区分大小写）/ uncased（全部转小写）

| 模型 | L | H | 参数量 |
|---|---|---|---|
| BERT-Tiny | 2 | 128 | 4M |
| BERT-Mini | 4 | 256 | 11M |
| BERT-Small | 4 | 512 | 30M |
| BERT-Medium | 8 | 512 | 42M |
| **BERT-Base** | 12 | 768（A = 12） | **110M** |
| **BERT-Large** | 24 | 1024（A = 16） | **340M** |

### 5.8 Fine-tuning

**目标**：利用 BERT 学到的 embedding 来做一个“姊妹”任务。

技巧：
- 从大规模预训练好的权重开始
- **冻结前面几层**有时能在复杂度和效果之间取得更好平衡
- 只需少量标注数据就能取得好效果（取决于任务与预训练数据/目标的接近程度）

两类用法：
- **Sequence classification**：如情感分析，用 `[CLS]` 的向量
- **Token classification**：如 NER、extractive question answering（为每个 token 预测是否为答案的开始/结束）

**情感分析例子**：
1. “This teddy bear is SO CUTE!” → 小写化（uncased）→ “this teddy bear is so cute!”
2. Tokenize，加 `[CLS]` 作为情感的占位符，末尾加 `[SEP]`，用 `[PAD]` 补齐
3. 加上 position embedding（0, 1, 2, …）和 segment embedding（全部属于同一段）
4. 送入预训练 BERT
5. 取 `[CLS]` 位置的输出向量，接一个 FFN 分类头，预测情感
6. 用标注数据训练（通常整网一起微调，学习率很小）

### 5.9 BERT 的收获与不足

**优点**：当时的 SOTA；真正的上下文相关词表示（“bank” 在不同句子中有不同向量）；适配多种分类任务；在工业界广泛用于各种“编码”场景（搜索、推荐、分类）。

**局限**：
- Context window 有限（512 tokens）
- 计算昂贵，对低延迟/成本敏感的应用难以推广
- 训练范式复杂：MLM/NSP 预训练 + fine-tuning

### 5.10 效率变体：DistilBERT（Sanh et al., 2019）

**Knowledge distillation（知识蒸馏）**（Hinton et al., “Dark knowledge”）：
- 用大模型（teacher）的输出概率分布（**soft targets**）来训练小模型（student），而不仅仅用 one-hot 标签
- “The soft targets contain almost all the knowledge.” —— teacher 认为 “cat” 有 0.7 概率、“dog” 有 0.2、“car” 有 0.001，这些相对大小本身包含了大量知识（cat 和 dog 很像，和 car 不像）

DistilBERT：BERT-Base（12 层）→ 6 层，**约 1.6 倍速度，保留约 97% 的效果**。同时也去掉了 NSP 和 segment embedding。

### 5.11 性能变体：RoBERTa（Liu et al., 2019）

**目标**：系统地分析 BERT 还能在哪些方向上优化（Robustly optimized BERT approach）。

**建模方面**：
- 去掉 NSP 和 segment embedding：**几乎没有影响**
- 静态 masking → **dynamic masking**：每个 epoch 重新随机选择 mask 位置，而不是预处理时一次性固定

**数据方面**：
- 更丰富：预训练语料 16 GB → **160 GB**
- 更久：1M 步 × batch 256 → 500K 步 × batch **8K**（大 batch）

**结果**：架构完全相同，各项 benchmark 提升约 **4%** —— 说明 BERT 原本是“训练不足”的。

---

## 参考论文

- Vaswani et al., “Attention Is All You Need”, 2017
- Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer” (T5)
- Press et al., “Train Short, Test Long: Attention with Linear Biases” (ALiBi), 2021
- Su et al., “RoFormer: Enhanced Transformer with Rotary Position Embeddings” (RoPE), 2021
- Ba et al., “Layer Normalization”, 2016；Xiong et al., “On Layer Normalization in the Transformer Architecture”, 2020；Zhang et al., “Root Mean Square Layer Normalization”, 2019
- Beltagy et al., “Longformer”, 2020
- Devlin et al., “BERT”, 2018；Peters et al., “Deep contextualized word representations” (ELMo), 2018
- Sanh et al., “DistilBERT”, 2019；Liu et al., “RoBERTa”, 2019
