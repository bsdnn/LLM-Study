# Lecture 2 · Transformer 改进技巧与 BERT

> Stanford CME 295：Transformers & Large Language Models（Afshine Amidi & Shervine Amidi，Fall 2025）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture2.pdf`](../slides/fall25-cme295-lecture2.pdf)
> 预计阅读时间：15–20 分钟。预备知识：Lecture 1（self-attention、Transformer 结构）。

## 本讲脉络

```
回顾：attention map
→ Position embeddings：learned → sinusoidal → 相对位置偏置（T5、ALiBi）→ RoPE（当今默认）
→ Layer normalization：LayerNorm → Post-Norm vs Pre-Norm → RMSNorm
→ Attention 近似：Longformer、sliding window、MQA / GQA（以及 KV cache 的账怎么算）
→ Transformer 模型三大类：encoder-decoder、encoder-only、decoder-only
→ BERT 深入：输入、MLM、NSP、规模、fine-tuning → DistilBERT（蒸馏）→ RoBERTa（训练配方）
```

**一句话总结**：2017 年之后，社区在三个方向上改进了 Transformer —— **位置怎么编码、归一化放在哪、attention 怎么算得更省**。今天的 LLM 基本标配 **RoPE + Pre-Norm RMSNorm + GQA**。另一条主线是 BERT 代表的 encoder-only 路线：先用代理任务在海量无标注文本上预训练，再针对下游任务 fine-tune。它在 2018–2022 年统治了 NLP，后来让位于 decoder-only 的 GPT 路线。

---

## 0. 回顾：attention map 能看出什么

原始 Transformer 论文把 attention 权重可视化后发现：第 5 层（共 6 层）中有两个 head 似乎在做 **anaphora resolution（指代消解）** —— 对单词 “its”，这两个 head 把注意力集中在 “its” 所指代的名词上。也就是说，**不同的 head 自发学会了不同的语言学功能**，这是 multi-head 设计有效的直接证据。

推荐阅读：原始 Transformer 论文 “Attention Is All You Need”。

---

## 1. Position Embeddings（位置编码）

### 1.1 为什么需要位置信息

Self-attention 让每个 token 与所有 token **直接相连**，这些“直接连接”本身不携带顺序信息。可以这样验证：把输入序列打乱，每个 token 的输出向量不变（只是跟着换了位置）—— 数学上叫 **permutation equivariance（置换等变）**。不额外提供位置，模型分不出 “dog bites man” 和 “man bites dog”。

### 1.2 方案一：Learned（可学习的）绝对位置编码

- 做法：为位置 0, 1, …, max_len−1 各学一个向量，加到 token 向量上。BERT、GPT-2 都这样做
- 优点：简单，让模型自己学最合适的表示
- 局限：**只能处理训练时见过的最大长度**（如 BERT 是 512）。第 513 个位置根本没有对应的向量，想支持更长序列就得扩展并重新训练

### 1.3 方案二：Hardcoded（固定的）sinusoidal 编码

原始 Transformer 的方案：

$$
PE_{(pos,\,2i)} = \sin\!\Big(\frac{pos}{10000^{2i/d}}\Big),\qquad
PE_{(pos,\,2i+1)} = \cos\!\Big(\frac{pos}{10000^{2i/d}}\Big)
$$

- 每两个维度一组，第 $i$ 组的频率是 $\omega_i = 10000^{-2i/d}$。低维频率高（变化快），高维频率低（变化慢）
- 把不同位置的编码两两做点积，得到的“相似度”随距离增大而下降 —— 相近的位置更相似
- 好处：**是一个公式，可以扩展到任意长度**
- 局限：虽然理论上可以外推，但实践中模型在比训练更长的序列上表现明显变差

### 1.4 思路转变：从绝对位置到相对位置

**观察**：attention 计算的是 query 和 key 之间的关系。对语言来说，真正重要的往往是 **“两个 token 相距多远”（relative position）**，而不是“它在第 37 个位置”（absolute position）。一句话放在文章开头还是中间，词与词之间的关系不变。

**思路**：不再把位置加到输入上，而是**直接修改 attention 层**，让 attention 分数直接感知相对距离。

### 1.5 方案三：在 attention 分数上加相对位置偏置

在 query-key 分数上加一个**只依赖于距离** $i - j$ 的偏置：

$$
\text{score}(i, j) = \frac{q_i \cdot k_j}{\sqrt{d}} + b(i - j)
$$

**T5 bias**（Raffel et al.）：
- 偏置是**按 head 学习的标量**：每个 head 有自己的一组偏置值
- 把距离分进 32 个**桶（bucket）**：近距离每个距离一个桶，远距离按对数间隔合并，超过 128 的距离都归入同一个桶（远处的精确距离不重要）
- 所有层共享这组偏置参数

**ALiBi**（Attention with Linear Biases，Press et al., 2021）：
- 偏置是**线性的、确定的（不用学）、无上界的**：

$$
b(i - j) = -m \cdot |i - j|
$$

- 每个 head 用不同的斜率 $m$。8 个 head 时，斜率为几何序列 $\frac{1}{2}, \frac{1}{4}, \dots, \frac{1}{256}$。斜率大的 head 只关注附近的 token，斜率小的 head 能看得更远
- 距离越远，惩罚越大 —— 天然地偏好近处的 token
- 论文标题 “Train Short, Test Long”：一个 13 亿参数的模型在长度 1024 上训练，可以直接用于长度 2048，困惑度与在 2048 上训练的 sinusoidal 模型相同，同时训练快 11%、省 11% 内存

### 1.6 方案四：RoPE（Rotary Position Embeddings）—— 当今的默认选择

论文：Su et al., 2021，“RoFormer: Enhanced Transformer with Rotary Position Embeddings”。LLaMA、Qwen、Mistral 等主流 LLM 都用 RoPE。

**思路**：用一个**旋转矩阵**旋转 query 和 key，**旋转角度与位置成正比**。

**二维情况**：位置 $m$ 的旋转矩阵为

$$
R(m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix},\qquad
q'_m = R(m)\, q_m,\quad k'_n = R(n)\, k_n
$$

**关键性质**：旋转后的点积只依赖于**相对距离** $n - m$：

$$
q'_m \cdot k'_n = (R(m) q_m)^\top (R(n) k_n) = q_m^\top R(m)^\top R(n)\, k_n = q_m^\top R(n - m)\, k_n
$$

（用到了旋转矩阵的性质：$R(m)^\top = R(-m)$，$R(-m)R(n) = R(n-m)$。）

**几何直觉**：把 query 和 key 想象成平面上的两根指针。位置 $m$ 的指针转 $m\theta$ 角，位置 $n$ 的指针转 $n\theta$ 角。两根指针的夹角只取决于它们**多转了多少**，即 $(n-m)\theta$。所以虽然每个向量按绝对位置旋转，attention 分数却只“看到”相对位置。

**推广到 $d > 2$ 维**：把 $d$ 维向量切成 $d/2$ 个二维块，第 $i$ 块用频率 $\theta_i = 10000^{-2i/d}$ 旋转（频率选择与 sinusoidal 编码相同）。低维块转得快，捕捉近距离的精细位置；高维块转得慢，捕捉远距离关系。

**实现要点**：
- 只旋转 **Q 和 K**，不旋转 V（位置只需要影响“谁关注谁”，不需要改变被传递的内容）
- 每一层的 attention 里都做，而不是只在输入时加一次
- **不增加任何参数**

**好处**：
- 相对距离被干净地编码进 attention 分数
- **Long-term decay（远程衰减）**：可以证明 attention 分数的上界随相对距离增大而衰减 —— 远处的 token 天然影响更小，符合语言直觉
- 可以通过调整旋转频率扩展 context 长度（扩展阅读：Position Interpolation、NTK-aware scaling、YaRN 等方法，都是在“把位置压缩或改频率”上做文章）

### 1.7 位置编码对比

| 方法 | 位置信息加在哪 | 绝对 / 相对 | 是否有参数 | 外推能力 | 代表模型 |
|---|---|---|---|---|---|
| Learned PE | 输入 embedding | 绝对 | 有 | 不能超过最大长度 | BERT、GPT-2 |
| Sinusoidal PE | 输入 embedding | 绝对（点积隐含相对） | 无 | 理论可以，实践一般 | 原始 Transformer |
| T5 bias | attention 分数 | 相对，按 head 学习 | 有（少量标量） | 有限 | T5 |
| ALiBi | attention 分数 | 相对，线性固定 | 无 | 好 | BLOOM、MPT |
| **RoPE** | 旋转 Q、K | 相对 | 无 | 较好，可通过缩放扩展 | LLaMA、Qwen、Mistral |

---

## 2. Layer Normalization

### 2.1 为什么要归一化

深层网络中，每一层输入的分布会随着前面层参数的更新而不断变化，数值可能越来越大或越来越小。归一化把每层的输入拉回一个稳定的尺度，带来：**训练更稳定、收敛更快、对学习率更不敏感**。

### 2.2 LayerNorm（Ba et al., 2016）

对**单个 token 的向量**在特征维度上做归一化：

$$
\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,\qquad
\mu = \frac{1}{d}\sum_{i=1}^{d} x_i,\quad \sigma^2 = \frac{1}{d}\sum_{i=1}^{d} (x_i - \mu)^2
$$

- $\gamma$、$\beta$ 是可学习的缩放和平移参数（让网络在需要时可以“撤销”归一化）
- $\epsilon$ 是防止除零的小常数

**LayerNorm vs BatchNorm**：

| | BatchNorm | LayerNorm |
|---|---|---|
| 在哪个维度上统计 | 同一个特征，跨 batch 中的所有样本 | 同一个样本，跨所有特征 |
| 依赖 batch 吗 | 是，batch 小时统计不准；推理时要用训练时的滑动平均 | 否，每个 token 独立计算 |
| 变长序列 | 不同长度、padding 会干扰统计 | 没问题 |
| 常用于 | CNN（视觉） | Transformer（NLP） |

### 2.3 Post-Norm vs Pre-Norm（Xiong et al., 2020）

LayerNorm 放在哪里很重要：

$$
\text{Post-LN（原始 Transformer）:}\quad x_{l+1} = \text{LN}\big(x_l + \text{Sublayer}(x_l)\big)
$$

$$
\text{Pre-LN（现在常用）:}\quad x_{l+1} = x_l + \text{Sublayer}\big(\text{LN}(x_l)\big)
$$

| | Post-LN | Pre-LN |
|---|---|---|
| LayerNorm 位置 | 残差相加**之后** | 子层**之前**，残差路径上不做归一化 |
| 梯度 | 初始化时，靠近输出层的参数梯度很大，直接用大学习率会不稳定 | 残差路径上有一条“干净”的恒等通道，各层梯度更均衡 |
| Warm-up | **必须**，学习率要从很小慢慢升上去 | 可以去掉或大大缩短 |
| 注意 | — | 最后一层之后要再加一个 LayerNorm |

直觉：Pre-LN 中，$x_l$ 可以不经任何变换直接加到最后（$x_L = x_0 + \sum_l \text{Sublayer}_l$），梯度沿这条“高速公路”无阻碍地回传，所以深层模型也好训练。

### 2.4 RMSNorm（Zhang et al., 2019）

$$
\text{RMSNorm}(x) = \gamma \odot \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2 + \epsilon}}
$$

- 去掉了“减均值”（re-centering）和偏移 $\beta$，只按**均方根（root mean square）**缩放
- 论文的发现：LayerNorm 起作用的主要是“缩放”而不是“平移”
- 计算更便宜（论文报告在不同模型上提速 7%–64%），效果基本不变

**如今的默认组合：Pre-Norm + RMSNorm**（LLaMA 等主流模型都如此）。

---

## 3. Attention 近似：让 attention 更省

完整 self-attention 的计算和内存都是 **$O(n^2)$**（$n$ = 序列长度）。$n = 4{,}000$ 时分数矩阵有 1600 万个元素，$n = 100{,}000$ 时有 100 亿个 —— 长文档的代价非常高。思路有两类：**让每个 token 少看一些 token**（稀疏 attention），以及**让 KV 更小**（共享 head）。

### 3.1 Sparse attention：Longformer（Beltagy et al., 2020）

不让每个 token 看所有 token，而是组合两种模式：
- **Local attention（局部）**：每个 token 只看两侧各 $w/2$ 个邻居（滑动窗口）。还可以用 dilated window（窗口中隔几个取一个）扩大覆盖范围
- **Global attention（全局）**：少数特殊 token（如分类任务的 `[CLS]`、问答任务的问题 token）可以看全部 token，也被全部 token 看到

复杂度从 $O(n^2)$ 降到 $O(n \cdot w)$，与序列长度成线性关系。Longformer 能处理 4,096 个 token 的文档（BERT 只有 512）。

### 3.2 Sliding Window Attention（SWA）

- 每个 token 只看前面 $W$ 个 token
- **层数叠加后，感受野会扩大**：第 1 层每个 token 看到 $W$ 个 token；第 2 层看到的每个 token 又各自看过前 $W$ 个，所以间接覆盖 $2W$；第 $k$ 层间接覆盖 $k \times W$ —— 与 CNN 中的 **receptive field（感受野）** 完全类似
- 例：**Mistral 7B** 用 $W = 4096$、32 层，理论上的 attention 跨度约 $4096 \times 32 \approx 131$K tokens
- 额外好处：KV cache 只需保留最近 $W$ 个 token（rolling buffer cache）。32K 长度的序列，cache 内存减少到原来的 1/8
- 变体：**局部层与全局层交替（interleave）**，例如几层 sliding window 之后放一层 full attention，兼顾效率与长程信息

### 3.3 共享 KV heads：MQA 与 GQA

**先理解 KV cache 的代价**。生成文本时，每个已处理 token 的 K、V 都要缓存下来（Lecture 3 详述），大小为：

$$
\text{KV cache / token} = 2 \times L \times H_{kv} \times d_{\text{head}} \times \text{bytes}
$$

（2 表示 K 和 V 各一份；$L$ = 层数；$H_{kv}$ = KV head 数。）

例：Llama 2 7B（MHA，32 层、32 个 head、$d_{\text{head}} = 128$、FP16）：
$2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$ 字节 ≈ **0.5 MB / token**。4,096 个 token 的一条序列就要约 2 GB，batch 为 16 时要 32 GB，比模型权重（约 13.5 GB）还大。

**思路**：让多个 query head **共享**同一组 key/value head，KV cache 直接按比例缩小。

| | MHA（Multi-Head Attention） | GQA（Grouped-Query Attention） | MQA（Multi-Query Attention） |
|---|---|---|---|
| KV head 数 | = query head 数 $h$ | $G$ 组，每组 query 共享一个 KV head（$1 < G < h$） | 只有 1 个 |
| KV cache | 1× | $G/h$ 倍 | $1/h$ 倍 |
| 质量 | 最好 | 接近 MHA | 有一定下降 |
| 提出 | Vaswani et al., 2017 | Ainslie et al., 2023 | Shazeer, 2019 |

- **Llama 2 70B** 用 GQA：64 个 query head 共享 8 个 KV head，KV cache 是 MHA 的 1/8（每 token 320 KB，而不是 2.5 MB）
- 小技巧：已经训练好的 MHA 模型，可以把 KV head 分组求平均变成 GQA，再用约 5% 的原始预训练算力继续训练（uptraining），就能恢复质量
- GQA 是质量与效率的折中，现代 LLM 广泛使用

---

## 4. 三类 Transformer 模型

| 类别 | 结构 | attention mask | 典型用途 | 代表模型 | 流行时间 |
|---|---|---|---|---|---|
| **Encoder-decoder** | 完整 Transformer | encoder 双向，decoder 因果 + cross-attention | Text-to-text（翻译、摘要） | T5、mT5、ByT5 | — |
| **Encoder-only** | 只保留 encoder，把 embedding 投影到类别 | 双向（每个 token 看左右两边） | 分类、抽取（情感分析、NER、检索） | BERT、DistilBERT、RoBERTa | 约 2018–2022 |
| **Decoder-only** | 只保留（masked）decoder，自回归生成 | 因果（只看左边） | 文本生成，几乎一切任务 | GPT 系列、LLaMA | **现在最流行** |

**Encoder-decoder：T5**（Text-To-Text Transfer Transformer）
- 把所有任务都统一成“输入文本 → 输出文本”：“translate English to German: …”、“summarize: …”、“cola sentence: …” → “acceptable”
- 预训练目标是 **span corruption**：随机遮住连续的片段（平均长度 3，共约 15% 的 token），用哨兵 token（`<X>`、`<Y>`）替代，让模型输出被遮住的内容
- 训练数据是 C4（约 750 GB 清洗过的网页文本）；mT5 是多语言版（101 种语言），ByT5 直接在字节上工作，不需要 tokenizer

**为什么后来 decoder-only 胜出**：结构最简单；预训练目标（预测下一个 token）适用于任何文本，数据无限；天然能生成；规模足够大后，通过 in-context learning 就能完成各种任务，不需要为每个任务 fine-tune。

---

## 5. BERT 深入

### 5.1 名字与背景

**BERT = Bidirectional Encoder Representations from Transformers**（Devlin et al., 2018）

- **Bidirectional**：encoder 的 self-attention 没有 mask，每个 token 同时看到左边和右边的上下文
- 对比：GPT 只能看左边；ELMo（Peters et al., 2018 年 2 月）用一个从左到右的 LSTM 和一个从右到左的 LSTM 分别训练再拼接，两个方向在训练时并不互通，所以**不是真正的联合双向**
- 彩蛋：ELMo 和 BERT 都是《芝麻街》（Sesame Street）中的角色

### 5.2 思路与两步训练

**只保留 encoder，用它输出的 encoded embeddings 做预测**（例如把 embedding 投影到类别上）。

1. **Pretraining**：在大量无标注文本上用两个代理任务（MLM 和 NSP）训练。数据：BooksCorpus（8 亿词）+ 英文 Wikipedia（25 亿词）。batch 为 256 条序列 × 512 token，训练 100 万步，约等于在 33 亿词的语料上跑 40 个 epoch
2. **Fine-tuning**：针对具体任务，在少量标注数据上微调整个网络

| 优点 | 缺点 |
|---|---|
| Fine-tuning 不需要很多数据 | 不适合文本生成等任务 |
| 效果好（发布时刷新 11 项 NLP 任务纪录） | 每个任务都必须 fine-tune 一份模型 |

**架构**：就是原始 Transformer 的 encoder 部分（BERT-base 12 层，BERT-large 24 层），去掉了 decoder。

### 5.3 输入处理

**Tokenizer**：WordPiece，事先在训练集上训练好，词表约 **30,000**，擅长识别常见的词缀（如 `##ing`、`##ly`）。

**特殊 token**：

| Token | 用途 |
|---|---|
| `[CLS]` | 放在输入最前面。它的最终向量汇总整句信息，用于句子级分类 |
| `[SEP]` | 分隔两个句段，并放在末尾 |
| `[MASK]` | 遮住要预测的 token（MLM 用） |
| `[PAD]` | 把短序列补齐到相同长度（配合 attention mask，让其他 token 不去关注 padding） |

**Input embedding = 三种 embedding 相加**：
1. **Token embedding**：一张巨大的查找表，每个词表项一个向量
2. **Position embedding**：帮助模型把 token 和位置关联起来（BERT 用 learned，最长 512）
3. **Segment embedding**（BERT 新增）：标记 token 属于句子 A 还是句子 B，同一段的所有 token 共享一个向量

### 5.4 预训练任务一：Masked Language Modeling（MLM）

**做法**：随机选出 **15%** 的输入 token 作为预测目标。对于每个被选中的 token：
- **80%** 的情况替换为 `[MASK]`：`my dog is hairy` → `my dog is [MASK]`
- **10%** 的情况替换为随机词：`my dog is hairy` → `my dog is apple`
- **10%** 的情况保持不变：`my dog is hairy` → `my dog is hairy`

模型要在这些位置上预测原来的词（loss 只在这 15% 的位置上计算）。

**为什么不全部换成 `[MASK]`**：fine-tuning 和实际使用时，输入中**永远不会出现** `[MASK]`。如果预训练只在 `[MASK]` 位置学习，模型可能学会“只有看到 `[MASK]` 才认真预测”，造成预训练与使用之间的不一致（mismatch）。随机替换和保持不变迫使模型**对每个位置都保持警惕**：任何一个词都可能是被换过的，必须结合上下文判断它是否合理。

**好处**：
- 模型基于**双向上下文**学习语言建模（GPT 式的“预测下一个词”只能用左边的上下文）
- 这种随机性起到 regularization 的作用，也反映了语言本身的概率性

**代价**：每条序列只有 15% 的位置提供训练信号，数据利用率比“每个位置都预测下一个词”低。

### 5.5 预训练任务二：Next Sentence Prediction（NSP）

**做法**：从语料中取两个句段 A 和 B：
- 50% 的情况下 B 确实是 A 的下一句（标签 IsNext）
- 50% 的情况下 B 是从语料中随机抽的（标签 NotNext）

```
[CLS] the man went to [MASK] store [SEP] he bought a gallon [MASK] milk [SEP]      → IsNext
[CLS] the man [MASK] to the store [SEP] penguin [MASK] are flight ##less birds [SEP] → NotNext
```

用 `[CLS]` 的输出向量做二分类。
- 动机：问答、自然语言推理等任务需要理解**两个句子之间的关系**，而 MLM 只关注单个句子内部
- 是一个**不需要人工标注**的简单分类任务

（后来 RoBERTa 发现 NSP 基本没用，见 5.10。）

### 5.6 超参数与规模

- **模型**：$L$ = 层数；$H$ = hidden size（embedding 维度）；$A$ = attention head 数（通常 $H / 64$）
- **数据**：语言特定 / 多语言（multilingual BERT 覆盖 104 种语言）；cased（区分大小写）/ uncased（全部转小写）

| 模型 | L | H | A | 参数量 |
|---|---|---|---|---|
| BERT-Tiny | 2 | 128 | 2 | 4M |
| BERT-Mini | 4 | 256 | 4 | 11M |
| BERT-Small | 4 | 512 | 8 | 30M |
| BERT-Medium | 8 | 512 | 8 | 42M |
| **BERT-Base** | 12 | 768 | 12 | **110M** |
| **BERT-Large** | 24 | 1024 | 16 | **340M** |

发布时的成绩：GLUE 综合分 80.5（绝对提升 7.7 分）、SQuAD v1.1 F1 93.2、SQuAD v2.0 F1 83.1。

### 5.7 Fine-tuning

**目标**：利用 BERT 学到的 embedding，完成一个“姊妹”任务。

技巧：
- 从大规模预训练好的权重开始
- **冻结前面几层**，有时能在复杂度和效果之间取得更好平衡（底层学的是通用特征，顶层更任务相关）
- 只需少量标注数据就能取得好效果（取决于任务与预训练数据分布、目标的接近程度）
- 常用超参数：学习率 2e-5 到 5e-5，batch size 16 或 32，训练 2–4 个 epoch。学习率太大会“冲掉”预训练学到的知识（catastrophic forgetting）

**四类常见用法**：

| 任务类型 | 输入格式 | 用哪个输出 | 例子 |
|---|---|---|---|
| 单句分类 | `[CLS] 句子 [SEP]` | `[CLS]` 向量 → 线性层 → softmax | 情感分析 |
| 句对分类 | `[CLS] 句子 A [SEP] 句子 B [SEP]` | `[CLS]` 向量 | 自然语言推理（MNLI：B 是否能从 A 推出） |
| Token 分类 | `[CLS] 句子 [SEP]` | 每个 token 的向量 → 各自分类 | NER |
| 抽取式问答 | `[CLS] 问题 [SEP] 段落 [SEP]` | 每个 token 打“答案开始”和“答案结束”两个分数 | SQuAD |

抽取式问答的细节：学两个向量 $S$（start）和 $E$（end）。段落中第 $i$ 个 token 的输出为 $T_i$，答案从 $i$ 到 $j$ 的分数是 $S \cdot T_i + E \cdot T_j$（要求 $j \ge i$），取分数最高的区间作为答案。

**情感分析的完整例子**：
1. 原文 “This teddy bear is SO CUTE!” → 用 uncased 模型，先转小写：“this teddy bear is so cute!”
2. WordPiece 切分：`this` `teddy` `bear` `is` `so` `cute` `!`
3. 开头加 `[CLS]`（作为情感的占位符），结尾加 `[SEP]`，再用 `[PAD]` 补齐到固定长度（如 16）
4. 加上 position embedding（0, 1, 2, …, 15）和 segment embedding（都属于段 A）→ position- and segment-aware embedding
5. 送入预训练好的 BERT
6. 取 `[CLS]` 位置的输出向量，接一个 FFN 分类头，输出情感类别
7. 用标注数据训练（通常整个网络一起微调，学习率很小）

### 5.8 BERT 的收获与局限

**收获**：
- 当时的 state-of-the-art
- 近乎真正的上下文相关词表示：同一个 “bank” 在“河岸”和“银行”两个句子中得到不同的向量
- 适配多种分类任务
- 工业界广泛用于各种“编码”场景：搜索排序、推荐、文本分类、语义相似度

**局限**：
- Context window 有限（512 tokens）
- 计算昂贵：对低延迟、成本敏感的应用很难推广
- 训练范式复杂：MLM / NSP 预训练 + 每个任务单独 fine-tuning

### 5.9 效率变体：知识蒸馏与 DistilBERT

**Knowledge distillation（知识蒸馏）**（Hinton et al., “Dark knowledge”, 2014）：用一个大模型（**teacher**）的输出来训练一个小模型（**student**）。

关键在于用 teacher 输出的**完整概率分布（soft targets）**，而不只是 one-hot 标签：

> “The soft targets contain almost all the knowledge.”

例：给一张猫的图片，teacher 输出 cat 0.70、dog 0.25、car 0.001。one-hot 标签只说“这是猫”，而 soft targets 还告诉 student：“猫和狗很像，和车完全不像”。这些类别之间的相对关系就是所谓的 **dark knowledge（暗知识）**。

为了让这些小概率更明显，通常用带 **temperature** $T$ 的 softmax 把分布“软化”：

$$
p_i^{(T)} = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)},\qquad
\mathcal{L} = \alpha \cdot \text{CE}(y, p_s) + (1 - \alpha)\, T^2 \cdot \text{KL}\big(p_t^{(T)} \,\|\, p_s^{(T)}\big)
$$

（$T^2$ 用来补偿软化后梯度变小的影响。）

**DistilBERT**（Sanh et al., 2019）：
- BERT-Base（12 层）→ **6 层**（用 teacher 每隔一层取一层来初始化 student）
- 训练 loss = 蒸馏 loss（匹配 teacher 的 soft targets）+ MLM loss + cosine embedding loss（让 student 的隐藏状态方向与 teacher 对齐）
- 去掉了 NSP 和 segment embedding
- 结果：参数少 40%（66M），速度快约 60%（约 1.6 倍），保留约 **97%** 的效果

### 5.10 性能变体：RoBERTa（Liu et al., 2019）

**目标**：系统地分析 BERT 在哪些方向上还能优化（**Ro**bustly optimized **BERT** **a**pproach）。

**建模方面**：
- **去掉 NSP 和 segment embedding：几乎没有影响**（甚至略好）—— 说明 NSP 这个任务太简单，没教给模型什么有用的东西
- **静态 masking → 动态 masking（dynamic masking）**：BERT 在数据预处理时就固定了每条数据的 mask 位置，每个 epoch 都看到同样的 mask；RoBERTa 每次喂数据时重新随机选 mask 位置，相当于数据增强

**数据与训练方面**：
- **更多数据**：预训练语料从 16 GB 增加到 **160 GB**（加入 CC-News、OpenWebText、Stories 等）
- **训练更久、batch 更大**：从“100 万步 × batch 256”改为“50 万步 × batch **8K**”
- 改用 byte-level BPE（词表 50K）

**结果**：架构完全不变，各项 benchmark 平均提升约 **4%** —— 说明 BERT 原本是“训练不足（undertrained）”的。**训练配方（数据、步数、batch）和架构同样重要。**

**另一个变体：ALBERT**：把 embedding 矩阵分解成两个小矩阵（$V \times E$ 和 $E \times H$），并让所有层共享参数，大幅减少参数量；用 SOP（Sentence Order Prediction，判断两句顺序是否被交换）代替 NSP。

---

## 6. 要点回顾

1. Self-attention 对顺序不敏感，必须注入位置信息；趋势是从“绝对位置加在输入上”走向“相对位置作用在 attention 里”
2. RoPE 按位置旋转 Q、K，使得 $q'_m \cdot k'_n$ 只依赖 $n - m$；无参数、有远程衰减、可扩展长度，是当今默认
3. Pre-Norm 让残差路径保持干净，训练更稳定；RMSNorm 只做缩放，更便宜。现代组合是 Pre-Norm + RMSNorm
4. Full attention 是 $O(n^2)$；sliding window 把复杂度降为线性，多层叠加扩大感受野
5. KV cache 大小 = 2 × 层数 × KV head 数 × head 维度 × 字节数 × token 数；GQA / MQA 通过共享 KV head 成比例缩小它
6. 三类模型：encoder-decoder（T5）、encoder-only（BERT）、decoder-only（GPT，当今主流）
7. BERT = 双向 encoder + MLM（15%，80/10/10）+ NSP 预训练 + 下游 fine-tuning
8. DistilBERT 用知识蒸馏换来更小更快的模型；RoBERTa 证明更好的训练配方能在不改架构的情况下显著提升效果

---

## 7. 自测题

<details>
<summary>1. 用一句话说明 RoPE 为什么能编码相对位置。</summary>

位置 m 的 query 旋转 mθ、位置 n 的 key 旋转 nθ，两者点积等于原始 q 与“旋转了 (n−m)θ 的 k”的点积，只依赖相对距离 n−m。
</details>

<details>
<summary>2. ALiBi 中斜率 m 大的 head 和斜率小的 head 有什么区别？</summary>

偏置是 −m·|i−j|。斜率大的 head 对远处 token 惩罚很重，几乎只关注附近的 token；斜率小的 head 惩罚轻，可以关注更远的 token。不同 head 由此覆盖不同的距离尺度。
</details>

<details>
<summary>3. 为什么 Post-LN 需要 learning rate warm-up，而 Pre-LN 常常不需要？</summary>

Post-LN 在初始化时靠近输出层的参数梯度很大，一开始就用大学习率容易发散，所以要从小学习率慢慢升上来。Pre-LN 的残差路径上没有归一化，形成一条恒等通道，各层梯度更均衡，训练一开始就比较稳定。
</details>

<details>
<summary>4. 一个模型有 80 层、64 个 query head、8 个 KV head、head 维度 128，用 FP16。每个 token 的 KV cache 多大？如果改成 MHA 呢？</summary>

GQA：2 × 80 × 8 × 128 × 2 字节 = 327,680 字节 ≈ 320 KB。MHA（64 个 KV head）是它的 8 倍，约 2.5 MB。
</details>

<details>
<summary>5. MLM 中为什么要有 10% 随机替换和 10% 保持不变？</summary>

因为 fine-tuning 和使用时输入里没有 [MASK]。如果只在 [MASK] 位置学习，模型会和实际使用场景不一致。随机替换和保持不变让模型不能只依赖 [MASK] 这个信号，必须对每个位置都根据上下文判断，减小预训练与使用之间的差异。
</details>

<details>
<summary>6. RoBERTa 的实验对“NSP 任务”得出了什么结论？这说明了什么？</summary>

去掉 NSP 和 segment embedding 后效果几乎不变甚至略好，说明 NSP 太简单（随机句子往往主题都不同，靠话题就能判断），没有教给模型有用的东西；BERT 的提升主要来自 MLM 以及更多数据、更长训练。
</details>

<details>
<summary>7. 知识蒸馏中的 soft targets 比 one-hot 标签多提供了什么信息？</summary>

Soft targets 给出 teacher 对所有类别的概率，包含了类别之间的相似关系（例如猫更像狗而不像车），也就是“dark knowledge”。Student 学习这些相对关系，比只学“正确答案是哪个”能学到更多。
</details>

---

## 参考论文

- Vaswani et al., “Attention Is All You Need”, 2017
- Raffel et al., “Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer” (T5), 2019
- Press et al., “Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation” (ALiBi), 2021
- Su et al., “RoFormer: Enhanced Transformer with Rotary Position Embeddings” (RoPE), 2021
- Ba et al., “Layer Normalization”, 2016；Xiong et al., “On Layer Normalization in the Transformer Architecture”, 2020；Zhang et al., “Root Mean Square Layer Normalization”, 2019
- Beltagy et al., “Longformer: The Long-document Transformer”, 2020；Mistral AI, “Mistral 7B”, 2023
- Shazeer, “Fast Transformer Decoding: One Write-Head is All You Need” (MQA), 2019；Ainslie et al., “GQA”, 2023
- Devlin et al., “BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding”, 2018；Peters et al., “Deep contextualized word representations” (ELMo), 2018
- Hinton et al., “Distilling the Knowledge in a Neural Network”, 2015；Sanh et al., “DistilBERT”, 2019
- Liu et al., “RoBERTa: A Robustly Optimized BERT Pretraining Approach”, 2019；Lan et al., “ALBERT”, 2019
