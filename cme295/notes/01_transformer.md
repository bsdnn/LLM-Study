# Lecture 1 · NLP 基础与 Transformer 架构

> Stanford CME 295：Transformers & Large Language Models（Afshine Amidi & Shervine Amidi，Fall 2025）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture1.pdf`](../slides/fall25-cme295-lecture1.pdf)

## 本讲脉络

```
NLP 任务概览 → Tokenization → Word representation（Word2vec）→ RNN / LSTM
           → Attention 的由来 → Self-attention → Transformer 架构 → 端到端翻译例子
```

一句话总结：**语言模型要解决两件事——把文本变成向量（tokenization + embedding），以及让每个向量“看到”上下文。** RNN 按顺序逐个传递信息，又慢又容易遗忘；Transformer 用 self-attention 让任意两个位置直接交互，既能并行又能捕捉长距离依赖。

---

## 1. NLP 任务概览

按“输入 → 输出”的形状，NLP 任务大致分三类：

| 类型 | 输入 → 输出 | 典型任务 |
|---|---|---|
| **Classification** | 一段文本 → 一个标签 | Sentiment extraction（情感分析）、Intent detection（意图识别）、Language detection、Topic modeling |
| **“Multi”-classification** | 一段文本 → 每个 token 一个标签 | Part-of-speech (PoS) tagging（词性标注）、Named Entity Recognition (NER，命名实体识别)、Dependency parsing、Constituency parsing |
| **Generation** | 一段文本 → 另一段文本 | Machine Translation (MT)、Question Answering (QA)、Summarization、Text generation |

### 三个代表任务

**1）Sentiment extraction（情感分析）**
- 例：“This teddy bear is SO CUTE!” → 正面
- 数据集：Amazon reviews、IMDB 影评、Twitter
- 评估指标（二分类，设“正面”为 positive）：
  - **Accuracy** = 预测正确的样本占比
  - **Precision** = TP / (TP + FP)：预测为正的样本里，有多少真的是正
  - **Recall** = TP / (TP + FN)：真正为正的样本里，有多少被找出来
  - **F1** = 2 · P · R / (P + R)：precision 与 recall 的调和平均，两者都高时 F1 才高
  - 当类别不平衡（例如 95% 都是正面）时，accuracy 会误导，所以更常看 precision / recall / F1

**2）Named Entity Recognition（NER）**
- 例：“A cute **teddy bear** is reading...” → 把 “teddy bear” 标为一个实体
- 数据集：CoNLL-2003、CoNLL++（标注过的 Reuters 新闻）
- 指标：同样用 accuracy / precision / recall / F1，但在 **token 级别、按实体类型** 分别计算

**3）Machine Translation（MT）**
- 例：“A cute teddy bear is reading” → “Un ours en peluche mignon lit”
- 数据集：WMT'14 English-French、English-German
- 指标：
  - **BLEU**：生成译文中的 n-gram 有多少出现在参考译文中，类似 **precision**（并对太短的译文加 brevity penalty）
  - **ROUGE**：参考文本中的 n-gram 有多少被生成文本覆盖，类似 **recall**，常用于 summarization
  - **Perplexity (PPL)**：模型看到这些词组合时有多“惊讶”。定义为 PPL = exp(−(1/N) Σ log p(wᵢ | w<ᵢ))，越低越好；直观上，PPL = k 相当于模型每一步在 k 个词之间犹豫

### 历史时间线

| 年代 | 里程碑 |
|---|---|
| 1980s | **RNN**（Recurrent Neural Networks）理论基础 |
| 1997 | **LSTM**（Long Short-Term Memory） |
| 2013 | **Word2vec** |
| 2017 | **Transformer**（“Attention Is All You Need”） |
| 2020s | **Large Language Models** |

推动力：更多数据、不断增长的算力、对想法的快速迭代。

---

## 2. Tokenization

模型不能直接读字符串，第一步是把文本切成 **token**，再映射成整数 ID。以 “A cute teddy bear is reading.” 为例：

| 粒度 | 切分结果 |
|---|---|
| Arbitrary（人为） | `A` `cute` `teddy bear` `is` `reading` `.` |
| **Word-level** | `A` `cute` `teddy` `bear` `is` `reading` `.` |
| **Subword-level** | `A` `cute` `ted` `##dy` `bear` `is` `read` `##ing` `.` |
| **Character-level** | `A` `_` `c` `u` `t` `e` `_` `t` `e` `d` `d` `y` … |

（`##` 表示“接在前一个 token 后面，不是新词开头”，这是 WordPiece 的写法。）

### 三种方法比较

| 方法 | 优点 | 缺点 |
|---|---|---|
| **Word-level** | 简单、可解释 | 容易遇到 **OOV**（Out-Of-Vocabulary，词表外的词）；无法利用词根（read / reading / reader 被当成毫无关系的三个词） |
| **Subword-level**（WordPiece、BPE） | 利用常见前缀后缀；从数据中学习；OOV 风险小 | 仍有少量 OOV 风险（byte-level BPE 可以完全消除）；序列比 word-level 长 |
| **Character-level** | 几乎没有 OOV；对大小写和拼写错误鲁棒（“RoBUsT tO CASinG”） | 序列很长，计算慢；单个字符的 embedding 没有语义，不可解释 |

**为什么现代 LLM 都用 subword**：它在“词表大小”和“序列长度”之间取得平衡。常见词保留为整词（一个 token），罕见词拆成有意义的片段。

**BPE（Byte-Pair Encoding）的思路**：从字符（或字节）开始，统计语料中最常相邻出现的一对符号，把它合并成一个新符号；重复直到词表达到目标大小。合并顺序就是“学到的”tokenizer。

---

## 3. Token Representation（词向量）

### One-hot 编码的问题

最朴素的方式是 one-hot：词表大小为 V，每个词是一个长度 V 的向量，只有自己的位置是 1。
问题：
- 维度极高（V 可能是几万到几十万）
- **任意两个词的向量都正交**，点积为 0，无法表达 “cat” 和 “dog” 比 “cat” 和 “car” 更相似

### Learned embedding

把每个词映射到一个低维稠密向量（维度 d ≪ V），让语义相近的词在向量空间里距离也近。这个映射就是一个 V × d 的 **embedding matrix**，每一行是一个词的向量。

### Word2vec（Mikolov et al., 2013）

核心思想：**设计一个“代理任务”（proxy task），在几十亿词的文本上训练一个浅层神经网络；任务本身不重要，训练完后取出中间的 embedding 层就是词向量。**

两种 proxy task：
- **CBOW**（Continuous Bag of Words）：用上下文词预测中间词。“A cute ___ bear is” → teddy
- **Skip-gram**：用中间词预测上下文词。teddy → A、cute、bear、is

**网络结构**：输入层（大小 V，one-hot）→ 隐藏层（大小 d，就是 embedding）→ 输出层（大小 V，softmax 得到每个词的概率）

**训练过程举例**（预测下一个词，V = 6，d = 2）：
1. 输入 “A” 的 one-hot `[1,0,0,0,0,0]`
2. 乘以输入权重得到隐藏向量，例如 `[0.2, 0.9]` —— 这就是 “A” 的 embedding
3. 乘以输出权重再做 softmax，得到对 6 个词的预测分布，例如 `[0.2, 0.4, 0.1, 0.1, 0.1, 0.1]`
4. 正确答案是 “cute”，用 cross-entropy loss 反向传播，更新权重
5. 滑动窗口，对下一个词对重复（“cute” → “teddy”…）

训练结束后，输入层到隐藏层的权重矩阵（V × d）就是 embedding table。

**Word2vec 的局限**：
- 不考虑词序（窗口内的词是“一袋子”）
- **Embedding 与上下文无关**：“bank”（河岸）和 “bank”（银行）只有一个向量 —— 这是后来 contextual embedding（ELMo、BERT、Transformer）要解决的问题

---

## 4. RNN 与 LSTM

### RNN（Recurrent Neural Network）

- 1980 年代提出，连接构成**时间序列**的一类网络
- 每个时间步 t 维护一个 hidden state，用当前输入和上一步状态更新：

```
a⟨t⟩ = g₁(W_aa · a⟨t−1⟩ + W_ax · x⟨t⟩ + b_a)     # 更新隐藏状态
y⟨t⟩ = g₂(W_ya · a⟨t⟩ + b_y)                     # 输出
```

- 同一组权重在每个时间步共享
- 处理 “A cute teddy bear is reading” 时，依次读入 A → cute → teddy bear → is，每读一个词就更新一次状态并可以输出一次，hidden state 像一份不断被改写的“摘要”

**RNN 的几种用法**（对应前面的任务类型）：
- many-to-one：情感分类（读完整句后输出一个标签）
- many-to-many（等长）：NER / PoS tagging（每个 token 一个标签）
- one-to-many：文本生成
- many-to-many（不等长，encoder-decoder）：机器翻译

### LSTM（Long Short-Term Memory，1997）

RNN 的问题是 **vanishing gradient**：反向传播要经过很多步连乘，梯度会指数衰减，导致模型记不住很久以前的信息。

LSTM 在隐藏状态中加入结构化的 **cell state c⟨t⟩** 和几个 **gate**（门，取值在 0–1 之间的 sigmoid 输出）：
- **Forget gate**：旧记忆保留多少
- **Input / update gate**：新信息写入多少
- **Output gate**：记忆中输出多少到 hidden state

cell state 的更新主要是加法而不是连乘，梯度更容易长距离传播。GRU 是一个更简单的变体（两个门）。

### 小结

| 方法 | 优点 | 缺点 |
|---|---|---|
| **Word2vec**（CBOW、Skip-gram） | 非常简单但很强；embedding 直观 | 不考虑词序；embedding 不感知上下文 |
| **RNN**（传统 RNN、LSTM） | 考虑词序；当时的 state-of-the-art | Vanishing gradient；**必须串行计算，慢**（第 t 步要等第 t−1 步算完） |

---

## 5. Attention 的由来（Bahdanau et al., 2014）

**问题**：早期的 seq2seq 翻译模型用一个 RNN encoder 把整句源语言压成**一个固定长度的向量**，再交给 decoder 逐词生成。句子一长，这个向量就装不下所有信息——decoder 生成到后面时已经“记不清”源句在说什么。

例：翻译 “A cute teddy bear is reading” → “Un ours en peluche mignon ___”，生成最后一个词时，模型需要的是源句中的 “reading”，但它只能依赖那一个压缩向量。

**Attention 的解决方案**：decoder 每生成一个词时，不再只看一个压缩向量，而是**回头看 encoder 所有位置的 hidden states，并根据当前需要给它们分配权重**。生成 “lit” 时，模型会把大部分注意力放在 “reading” 上。这就是“对齐”（align）。

---

## 6. Self-Attention 机制

Transformer（Vaswani et al., 2017，“Attention Is All You Need”）把 attention 从 “decoder 看 encoder” 推广为 **“序列中每个位置都去看序列中所有位置”**，并且完全去掉了 RNN。

### Query、Key、Value 的直觉

类比查字典 / 搜索引擎：
- **Query (Q)**：我在找什么（当前 token 提出的“问题”）
- **Key (K)**：我有什么可以被找的（每个 token 贴的“标签”）
- **Value (V)**：如果被选中，我提供什么内容

对某个 token 来说：
1. 用它的 query 与每个 token 的 key 做点积，得到相关性分数
2. 对分数做 softmax，得到一组和为 1 的权重
3. 用这些权重对所有 token 的 value 做加权平均，得到该 token 新的、**融合了上下文的表示**

例：句子 “A cute teddy bear is reading” 中，“reading” 的 query 可能和 “teddy bear” 的 key 匹配度最高（谁在读？），于是 “reading” 的新表示里会混入很多 “teddy bear” 的信息。

### 矩阵形式

把所有 token 的向量堆成矩阵 X（n × d_model），用三个可学习矩阵投影：

```
Q = X · W_Q,   K = X · W_K,   V = X · W_V

Attention(Q, K, V) = softmax( Q · Kᵀ / √d_k ) · V
```

- Q · Kᵀ 是 n × n 矩阵，第 i 行第 j 列 = token i 对 token j 的相关性分数
- softmax 按行做，每行变成和为 1 的权重
- 再乘 V，每个 token 得到 value 的加权平均
- **为什么除以 √d_k**：q·k 是 d_k 项乘积之和，维度越大数值越大，softmax 会变得极端（接近 one-hot），梯度变小；除以 √d_k 让方差保持在 1 左右
- 所有 token 一次矩阵乘法就算完，**可以完全并行**，这是相对 RNN 的巨大优势

---

## 7. Transformer 架构

原始 Transformer 是一个 **encoder-decoder** 结构，用于机器翻译：

```
源句 ──► [Input Embedding + Positional Encoding] ──► Encoder × N ──┐
                                                                  │ (K, V)
目标句(右移) ──► [Output Embedding + PE] ──► Decoder × N ◄─────────┘
                                              │
                                     Linear ─► Softmax ─► 下一个词的概率
```

### 7.1 Input

- 文本先 tokenize，再查 **learned embedding** 表
- 参数：**V**（vocabulary size）、**d_model**（embedding 维度，原论文为 512）

### 7.2 Positional Encoding（PE）

Self-attention 本身对顺序不敏感：把句子里的词打乱，每个 token 得到的加权平均结果完全一样（只是换了位置）。“dog bites man” 和 “man bites dog” 对它来说没区别。所以要**把位置信息加到输入上**。

- 做法：position-aware embedding = token embedding + position embedding
- 可以是 **learned**（每个位置学一个向量），也可以是 **hardcoded**
- 原论文用 sinusoidal（正弦）编码：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

- 不同维度使用不同频率的正弦波，低维变化快，高维变化慢，像二进制计数的连续版本
- 目标：让模型理解**相对位置**。正弦编码的一个好性质是，两个位置编码的点积只依赖于它们的距离，距离越近点积越大

### 7.3 Encoder

每个 encoder 层包含：
1. **Self-attention**（encoder-encoder attention）：源句中每个 token 看源句中所有 token
2. **Feed Forward Neural Network (FFNN)**：对每个 token 独立做两层全连接 `FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂`
3. **Normalization layer**（LayerNorm）+ 残差连接（residual connection）：`x + Sublayer(x)` 后再归一化，帮助深层网络训练

参数：**N**（层数，原论文 6）、**h**（attention head 数，8）、**d_FF**（FFN 隐藏层宽度，2048）、**d_key / d_value**（每个 head 的维度，64）、**d_model**（512）

Encoder 输出的是 **context-aware encoded embeddings**：每个 token 的向量已经融合了整句的上下文。

### 7.4 Output “shifted right”

- 训练时，decoder 的输入是目标句**右移一位**：开头插入 `[BOS]`（beginning of sentence）
- 这样在位置 t，decoder 输入的是前 t−1 个正确的词，要预测第 t 个词
- 推理（翻译）时从 `[BOS]` 开始逐个生成

### 7.5 Decoder

每个 decoder 层包含：
1. **Masked self-attention**（decoder-decoder）：目标句中每个 token 只能看它**之前**的 token（用 mask 遮住未来，否则训练时就“偷看”了答案）
2. **Encoder-decoder attention**（cross-attention）：**Query 来自 decoder，Key 和 Value 来自 encoder 的输出** —— 这正是 Bahdanau attention 的位置：生成每个目标词时去源句中找相关部分
3. FFNN
4. Normalization + 残差

### 7.6 Output

- **Linear projection** 把 d_model 维向量映射到 V 维
- **Softmax** 得到词表上每个词的概率
- 本质上是一个**分类问题，类别 = 词表中的词**

### 7.7 计算技巧

**Multi-Head Attention（MHA）**
- 并行运行 h 个 self-attention，每个 head 有自己的 W_Q、W_K、W_V（维度 d_model / h）
- 各 head 输出拼接后乘 W_O 混合
- 好处：不同 head 可以捕捉不同类型的关系（语法依赖、指代、相邻词…）
- 类比：CNN 中一层有多个 filter，每个 filter 检测不同的特征

**Label Smoothing**
- 来自 2015 年一篇视觉论文的观察：**模型过度自信是有害的**
- 做法：训练时不用纯 one-hot 标签（正确词 = 1，其他 = 0），而是给正确词 1 − ε，把 ε 平均分给其他词（原论文 ε = 0.1）
- 好处：防止过拟合的通用技巧；perplexity 会略差，但 accuracy 和 BLEU 提升

---

## 8. 端到端例子：英译法

把 “A cute teddy bear is reading.” 翻译成 “Un ours en peluche mignon lit.”：

**Encoder 侧**
1. Tokenize：`[BOS] A cute teddy bear is reading . [EOS]`
2. 每个 token 查 embedding，加上 position embedding → **position-aware embeddings**，堆成矩阵
3. 乘 W_Q、W_K、W_V 得到 Q、K、V
4. 计算 Q·Kᵀ：每对 token 之间的相关性分数矩阵；除以 √d_k 后按行 softmax
5. 用权重对 V 做加权平均 —— 每个 token 的输出是 **values 的加权平均，权重是 query 与 key 相似度的函数**
6. 对 h 个 head 各做一遍，拼接后乘 W_O
7. 经过 FFN（及 norm、残差），得到 **context-aware encoded embeddings**
8. 重复 N 层，得到最终 encoder 输出

**Decoder 侧（逐词生成）**
1. 输入 `[BOS]`
2. Masked self-attention（此时只有 `[BOS]` 自己）
3. Encoder-decoder attention：用 decoder 的 query 去查 encoder 输出的 key / value
4. FFN → Linear → Softmax，得到词表上的概率分布，如 `[0.001, 0.0003, …, 0.4, …, 0.002]`，取最高概率的 “Un”
5. 把 “Un” 接到输入后面：`[BOS] Un`，再跑一遍 decoder，得到 “ours en peluche”
6. 重复，直到生成 `[EOS]`：`Un ours en peluche mignon lit [EOS]`

**训练 vs 推理的区别**：训练时目标句已知，decoder 可以一次并行处理整个右移后的目标句（靠 mask 保证不偷看）；推理时只能一个词一个词地生成（autoregressive）。

---

## 9. 本课程的常见缩写

| 类别 | 缩写 |
|---|---|
| Transformer 模型 | BERT、T5、GPT、LLaMA |
| 训练策略 | SFT（Supervised Fine-Tuning）、PEFT（Parameter-Efficient Fine-Tuning）、FLAN、RL、RM（Reward Model）、RLHF、PPO、DPO |
| 数据集 | MNLI、WNLI、C4、SQuAD、GLUE、MRPC |
| 其他架构与技术 | LSTM、GRU、GloVe、BPE、CoT（Chain-of-Thought）、ToT（Tree-of-Thought）、SC（Self-Consistency）、RAG |
| 任务 | NER、PoS、MLM（Masked Language Modeling）、NSP（Next Sentence Prediction）、MT、QA、NLG |
| 指标 | F1、PPL、ROUGE、BLEU、METEOR、LaaJ（LLM-as-a-Judge）、WER（Word Error Rate） |

## 参考资料

- 课程网站：cme295.stanford.edu
- 教材：*Super Study Guide: Transformers & Large Language Models*（https://superstudy.guide）
- VIP Cheatsheet：https://github.com/afshinea/stanford-cme-295-transformers-large-language-models
- Vaswani et al., “Attention Is All You Need”, 2017
- Bahdanau et al., “Neural Machine Translation by Jointly Learning to Align and Translate”, 2014
- Mikolov et al., “Efficient Estimation of Word Representations in Vector Space”, 2013
