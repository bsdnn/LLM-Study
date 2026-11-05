# Lecture 1 · NLP 基础与 Transformer 架构

> Stanford CME 295：Transformers & Large Language Models（Afshine Amidi & Shervine Amidi，Fall 2025）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture1.pdf`](../slides/fall25-cme295-lecture1.pdf)
> 预计阅读时间：15–20 分钟。预备知识：基础机器学习（分类、loss、梯度下降）、线性代数（矩阵乘法、向量点积）。

## 本讲脉络

```
NLP 任务与评估指标 → Tokenization（文本 → token）→ Word representation（token → 向量）
→ RNN / LSTM（按顺序处理序列）→ seq2seq 的瓶颈与 attention 的诞生
→ Self-attention（Q / K / V）→ Multi-head attention → Transformer 逐层拆解 → 端到端翻译例子
```

**一句话总结**：任何语言模型都要解决两件事 —— **把文本变成向量**（tokenization + embedding），以及**让每个向量融合上下文信息**。RNN 按顺序一个一个传递信息，既慢又容易遗忘；Transformer 用 self-attention 让任意两个位置直接交互，既能并行计算，又能捕捉长距离依赖。今天所有的 LLM 都建立在这个架构之上。

---

## 1. NLP 任务全景

### 1.1 按“输出的形状”分三类

| 类型 | 输入 → 输出 | 模型最后一层在做什么 | 典型任务 |
|---|---|---|---|
| **Classification** | 一段文本 → **一个**标签 | 对整段文本做一次 softmax 分类 | Sentiment extraction（情感分析）、Intent detection（意图识别）、Language detection（语种识别）、Topic modeling（主题分类） |
| **“Multi”-classification** | 一段文本 → **每个 token 一个**标签 | 对每个位置各做一次 softmax 分类 | Part-of-speech (PoS) tagging（词性标注）、Named Entity Recognition (NER，命名实体识别)、Dependency parsing（依存句法分析）、Constituency parsing（成分句法分析） |
| **Generation** | 一段文本 → **另一段文本** | 一次生成一个 token，每一步都是在整个词表上做 softmax | Machine Translation (MT)、Question Answering (QA)、Summarization（摘要）、Text generation |

这三类的区别在于“要做多少次分类、分类的类别是什么”。后面会看到，**生成本质上也是分类** —— 只不过类别是词表中的每一个词，并且要一个接一个地分类很多次。

### 1.2 分类任务与指标：以情感分析为例

**任务**：输入 “This teddy bear is SO CUTE!”，输出“正面”。
**常用数据集**：Amazon 商品评论、IMDB 影评、Twitter 推文。

**Confusion matrix（混淆矩阵）**：把我们关心的那一类叫 positive（正类）。每个预测结果落入四格之一：

| | 实际为正 | 实际为负 |
|---|---|---|
| **预测为正** | TP（True Positive，真阳性） | FP（False Positive，误报） |
| **预测为负** | FN（False Negative，漏报） | TN（True Negative，真阴性） |

四个指标：

$$
\text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN},\qquad
\text{Precision} = \frac{TP}{TP + FP},\qquad
\text{Recall} = \frac{TP}{TP + FN}
$$

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

- **Precision（精确率）**：所有被我预测为正的样本里，有多少真的是正 —— 衡量“报警准不准”
- **Recall（召回率）**：所有真正为正的样本里，我找出了多少 —— 衡量“有没有漏”
- **F1**：两者的**调和平均**。调和平均会被较小的那个拉低，所以只有 precision 和 recall **都高**时 F1 才高

**算例**：100 条评论中有 10 条是差评，我们要找出差评（差评 = 正类）。模型标出了 8 条差评，其中 6 条判断正确。
- TP = 6，FP = 2，FN = 10 − 6 = 4，TN = 100 − 6 − 2 − 4 = 88
- Accuracy = (6 + 88) / 100 = **94%**
- Precision = 6 / 8 = **75%**
- Recall = 6 / 10 = **60%**
- F1 = 2 × 0.75 × 0.6 / (0.75 + 0.6) = 0.9 / 1.35 ≈ **0.667**

**为什么不能只看 accuracy**：一个“永远说不是差评”的傻模型，accuracy 也有 90%，但 recall 是 0，一条差评都没找到。**类别不平衡时，accuracy 会严重误导**，这时要看 precision / recall / F1。

**多分类时的 F1**：
- **Macro-F1**：每个类别单独算 F1，再简单平均 —— 小类和大类同等重要
- **Micro-F1**：把所有类别的 TP、FP、FN 加总后再算 —— 被大类主导

### 1.3 序列标注：以 NER 为例

**任务**：标出文本中的实体，例如 “A cute **teddy bear** is reading...” 中的 “teddy bear”。

**BIO 标注方案**：实体可能跨多个 token，所以每个 token 的标签不只是“是不是实体”，而是：
- **B-X**：实体 X 的开头（Begin）
- **I-X**：实体 X 的内部（Inside）
- **O**：不属于任何实体（Outside）

```
A     cute   teddy   bear   is   reading
O     O      B-ENT   I-ENT  O    O
```

这样就把“找实体”变成了“给每个 token 分类”。

**数据集**：CoNLL-2003、CoNLL++（标注过的 Reuters 新闻），实体类型有 PER（人名）、LOC（地名）、ORG（组织）、MISC（其他）。

**评估**：同样用 accuracy / precision / recall / F1，但是**在 token 级别、按实体类型分别计算**（也常用更严格的实体级别评估：整个实体的边界和类型都对才算对）。

### 1.4 生成任务：以机器翻译为例

**任务**：“A cute teddy bear is reading” → “Un ours en peluche mignon lit”
**数据集**：WMT'14 English-French、WMT'14 English-German。

生成任务的难点在于：**正确答案不唯一**。同一句话有很多种正确译法，所以只能和“参考译文”比较相似程度。

**BLEU**（BiLingual Evaluation Understudy）—— 类似 **precision**：生成的译文中，有多少 n-gram 出现在参考译文里。

$$
\text{BLEU} = BP \cdot \exp\Big(\sum_{n=1}^{N} w_n \log p_n\Big),\qquad
BP = \begin{cases} 1 & c > r \\ e^{\,1 - r/c} & c \le r \end{cases}
$$

- $p_n$：**modified n-gram precision**。每个 n-gram 的匹配次数会被“截断”到它在参考译文中出现的次数，防止模型靠重复同一个词刷分
- $BP$（brevity penalty，过短惩罚）：$c$ 是生成长度，$r$ 是参考长度。没有这一项，模型只输出一两个最有把握的词，precision 就能很高
- 通常 $N = 4$，$w_n = 1/4$

**BLEU 算例**（为了简单只算到 bigram，即 BLEU-2）：
- 参考译文：`the teddy bear is reading a book`（7 个词）
- 生成译文：`the bear is reading the book`（6 个词）
- Unigram：生成中有 `the`×2、`bear`、`is`、`reading`、`book`。参考里 `the` 只出现 1 次，所以 `the` 只能算 1 次匹配 → $p_1 = 5/6$
- Bigram：生成的 5 个 bigram 是 `the bear`、`bear is`、`is reading`、`reading the`、`the book`，其中只有 `bear is` 和 `is reading` 在参考中 → $p_2 = 2/5$
- $BP = e^{1 - 7/6} \approx 0.846$
- BLEU-2 $= 0.846 \times \sqrt{5/6 \times 2/5} = 0.846 \times 0.577 \approx 0.49$

**ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）—— 类似 **recall**：参考文本中的 n-gram 有多少被生成文本覆盖。常用于摘要任务（摘要最怕漏掉关键信息）。ROUGE-L 用最长公共子序列（LCS）衡量。

**Perplexity（PPL，困惑度）**—— 衡量模型看到这串词时有多“惊讶”：

$$
\text{PPL} = \exp\Big(-\frac{1}{N}\sum_{t=1}^{N} \log p(w_t \mid w_{<t})\Big)
$$

**算例**：模型给一句 4 个词的句子中每个词的概率分别是 0.5、0.25、0.5、0.125。
- 负对数：0.693、1.386、0.693、2.079，平均 1.213
- PPL = $e^{1.213} \approx 3.36$

**直觉**：PPL ≈ 3.36 意味着模型平均每一步好像在大约 3.4 个词之间均匀犹豫。PPL 越低越好；如果模型每次都给正确词概率 1，PPL = 1。

（各指标的详细公式可参考 http://www.statmt.org/book/slides/08-evaluation.pdf）

### 1.5 历史时间线

| 年代 | 里程碑 | 意义 |
|---|---|---|
| 1980s | **RNN**（Recurrent Neural Networks） | 奠定了用神经网络处理序列的理论基础 |
| 1997 | **LSTM**（Long Short-Term Memory） | 用门控机制缓解梯度消失，能记住更长的依赖 |
| 2013 | **Word2vec** | 高效学习词向量，开启“预训练表示”的思路 |
| 2014 | **Seq2seq + Attention** | 机器翻译的突破，attention 首次出现 |
| 2017 | **Transformer**（“Attention Is All You Need”） | 完全基于 attention，可并行、可扩展 |
| 2020s | **Large Language Models** | 在海量数据上训练超大 Transformer |

推动发展的三股力量：**大量数据、不断增长的算力、对想法的快速迭代**。

---

## 2. Tokenization：把文本切成 token

### 2.1 为什么需要 tokenization

神经网络只能处理数字。第一步要把文本切成一个个单位（token），再把每个 token 映射到一个整数 ID（它在词表 vocabulary 中的位置）。词表大小记为 $V$。

以 “A cute teddy bear is reading.” 为例：

| 粒度 | 切分结果 |
|---|---|
| Arbitrary（人为） | `A` `cute` `teddy bear` `is` `reading` `.` |
| **Word-level** | `A` `cute` `teddy` `bear` `is` `reading` `.` |
| **Subword-level** | `A` `cute` `ted` `##dy` `bear` `is` `read` `##ing` `.` |
| **Character-level** | `A` `_` `c` `u` `t` `e` `_` `t` `e` `d` `d` `y` … |

（`##` 表示“接在前一个 token 后面，不是新词的开头”，这是 WordPiece 的写法。）

### 2.2 三种粒度的详细比较

**Word-level（按词切）**
- 优点：简单；每个 token 有明确含义，可解释
- 缺点 1：**OOV（Out-Of-Vocabulary，词表外的词）**。训练时没见过的词（新词、拼写错误、罕见人名）只能统一映射成 `[UNK]`，信息完全丢失
- 缺点 2：**无法利用词根**。read、reads、reading、reader 被当作四个毫无关系的词，各学各的向量；英语词形变化多，词表会膨胀到几十万
- 缺点 3：词表太大，embedding 矩阵（$V \times d$）也太大

**Character-level（按字符切）**
- 优点：词表极小（几百个字符），几乎不可能 OOV；对大小写和拼写错误鲁棒（“RoBUsT tO CASinG anD MIspeliNGs”）
- 缺点 1：**序列变得很长**（一个词变成 5–10 个 token）。后面会看到 attention 的计算量与序列长度的平方成正比，所以计算会慢很多
- 缺点 2：单个字符的 embedding **没有语义**（字母 “t” 本身没有意思），模型要花更多层去“拼出”词义

**Subword-level（按子词切，如 WordPiece、BPE）**
- 优点：常见词保留为整词（一个 token），罕见词拆成有意义的片段（read + ##ing）；能利用常见的前缀和后缀；切分规则**从数据中学出来**
- 缺点：仍有很小的 OOV 风险（byte-level BPE 可以完全消除）；序列比 word-level 稍长

**为什么现代 LLM 都用 subword**：它在“词表大小”和“序列长度”之间取得了平衡。典型词表大小在 3 万到 25 万之间。

### 2.3 BPE（Byte-Pair Encoding）算法逐步演示

BPE 的思想：**从字符开始，反复把语料中最常一起出现的相邻符号对合并成一个新符号**，直到词表达到目标大小。

**训练语料**（词 : 出现次数），每个词末尾加一个词尾标记 `</w>`：

```
l o w </w>          ×5
l o w e r </w>      ×2
n e w e s t </w>    ×6
w i d e s t </w>    ×3
```

**第 1 步：统计所有相邻符号对的频次**

| 符号对 | 频次 | 来源 |
|---|---|---|
| `e s` | 9 | newest (6) + widest (3) |
| `s t` | 9 | newest (6) + widest (3) |
| `t </w>` | 9 | newest (6) + widest (3) |
| `w e` | 8 | lower (2) + newest (6) |
| `l o` | 7 | low (5) + lower (2) |
| `o w` | 7 | low (5) + lower (2) |
| `n e`、`e w` | 6 | newest |
| … | … | … |

**第 2 步：合并频次最高的对**。`e s` 最高（与另外两个并列，按顺序取第一个），合并为新符号 `es`：

```
n e w es t </w>  ×6
w i d es t </w>  ×3
```

**第 3 步：重复**。现在 `es t` 频次 9，合并为 `est`；接着 `est </w>` 合并为 `est</w>`；然后 `l o` → `lo`，`lo w` → `low`……

每次合并都往词表里加一个新 token，合并顺序就是学到的 tokenizer。

**用学到的规则切分新词**：训练时从没见过的 “lowest”，按合并规则切成 `low` + `est</w>` —— 两个都是有意义的片段，不需要 `[UNK]`。

### 2.4 其他 subword 算法

| 算法 | 合并 / 选择准则 | 使用者 |
|---|---|---|
| **BPE** | 合并**频次最高**的相邻对 | GPT 系列（byte-level BPE）、LLaMA |
| **WordPiece** | 合并使语料似然提升最大的对，等价于最大化 $\frac{\text{count}(ab)}{\text{count}(a)\cdot\text{count}(b)}$ —— 偏好“两部分单独出现很少、但经常一起出现”的组合 | BERT |
| **Unigram LM**（SentencePiece） | 从一个大词表出发，逐步删除对语料似然影响最小的 token | T5、很多多语言模型 |

**Byte-level BPE**：不从“字符”开始，而是从 **256 个字节值**开始。任何文本（包括 emoji、中文、乱码）都能表示成字节序列，所以**永远不会 OOV**。

### 2.5 Token 数的实际意义

- 英文中 1 个 token 平均约 3–4 个字符（约 0.75 个英文单词）
- 中文、日文等语言在以英文为主的词表下通常更“费 token”
- **token 数决定成本（API 按 token 计费）、速度和能放进 context 的内容多少**

---

## 3. 词的向量表示（Token Representation）

### 3.1 One-hot 编码的问题

最朴素的表示：词表大小为 $V$，每个词是一个长度为 $V$ 的向量，只有自己的位置是 1，其余是 0。

```
cat = [0, 0, 1, 0, 0, …, 0]
dog = [0, 0, 0, 1, 0, …, 0]
car = [0, 1, 0, 0, 0, …, 0]
```

问题：
1. **维度极高**：$V$ 可能是几万到几十万
2. **任意两个词都正交**：cat·dog = 0，cat·car = 0。向量之间的相似度全是 0，无法表达 “cat 和 dog 比 cat 和 car 更像”

### 3.2 分布式假设与 Embedding 矩阵

**分布式假设（distributional hypothesis）**：“You shall know a word by the company it keeps.” —— 出现在相似上下文中的词，意思也相似。“我养了一只 ___”中能填 cat 也能填 dog，所以它们应该有相似的表示。

**Learned embedding**：把每个词映射成一个**低维稠密向量**（维度 $d$，通常几百到几千，远小于 $V$）。所有词向量组成一个 **embedding matrix** $E \in \mathbb{R}^{V \times d}$，第 $i$ 行就是第 $i$ 个词的向量。

查找一个词的向量，等价于用它的 one-hot 向量乘以 $E$：one-hot 中唯一的 1 会把 $E$ 的那一行“挑”出来。实践中直接按索引取行（lookup），不做真正的矩阵乘法。

### 3.3 Word2vec（Mikolov et al., 2013）

**核心思想**：设计一个**代理任务（proxy task）**，在几十亿词的文本上训练一个浅层神经网络。代理任务本身不重要，**训练完后，网络中间的 embedding 层就是我们要的词向量**。

**两种代理任务**：

| 任务 | 做法 | 例子（窗口内） |
|---|---|---|
| **CBOW**（Continuous Bag of Words） | 用**上下文词**预测**中间词** | 已知 A、cute、bear、is，预测 teddy |
| **Skip-gram** | 用**中间词**预测**上下文词** | 已知 teddy，预测 A、cute、bear、is |

**网络结构**：输入层（大小 $V$，one-hot）→ 隐藏层（大小 $d$，就是 embedding）→ 输出层（大小 $V$，softmax 得到每个词的概率）。

**前向计算**（以“用当前词预测下一个词”为例）：

$$
h = W_{\text{in}}^\top x \in \mathbb{R}^d \;(\text{即取出 } W_{\text{in}} \text{ 的一行}),\qquad
u = W_{\text{out}}\, h \in \mathbb{R}^V,\qquad
p = \text{softmax}(u)
$$

$$
\mathcal{L} = -\log p_{\text{target}}
$$

**训练过程算例**（词表 $V = 6$：A、cute、teddy、bear、is、reading；$d = 2$）：

| 步骤 | 输入 | 隐藏向量（embedding） | 输出概率分布 | 正确答案 | Loss |
|---|---|---|---|---|---|
| 1 | “A” = [1,0,0,0,0,0] | [0.2, 0.9] | [0.2, **0.4**, 0.1, 0.1, 0.1, 0.1] | cute（第 2 个） | −ln 0.4 ≈ 0.92 |
| 2 | “cute” = [0,1,0,0,0,0] | [0.8, 0.4] | [0.2, 0.2, **0.2**, 0.1, 0.2, 0.1] | teddy（第 3 个） | −ln 0.2 ≈ 1.61 |

每一步用 cross-entropy loss 反向传播、更新两个权重矩阵，然后把窗口往后滑一个词，重复几十亿次。训练结束后，$W_{\text{in}}$（$V \times d$）就是 embedding table。

**效率技巧：Negative sampling**。输出层对 $V$ 个词做 softmax 很贵（$V$ 可能是几十万）。Negative sampling 把它换成二分类：让真实的（中心词，上下文词）对得高分，再随机抽 $K$ 个“负样本”词让它们得低分：

$$
\mathcal{L} = -\log \sigma(u_o^\top v_c) - \sum_{k=1}^{K} \log \sigma(-u_k^\top v_c)
$$

这样每步只需计算 $K+1$ 个词（$K$ 通常 5–20），而不是整个词表。

### 3.4 Word2vec 学到了什么，有什么局限

**学到的性质**：
- 语义相近的词向量相近（用 **cosine similarity** 衡量：$\cos(a, b) = \frac{a \cdot b}{\|a\|\,\|b\|}$）
- 向量运算能表达关系：$\text{king} - \text{man} + \text{woman} \approx \text{queen}$

**局限**：
- **不考虑词序**：窗口内的词被当作“一袋子词”
- **Embedding 与上下文无关（static）**：“bank”（河岸）和 “bank”（银行）只有**一个**向量。这正是后来 contextual embedding（ELMo、BERT、Transformer）要解决的问题

---

## 4. RNN 与 LSTM

### 4.1 RNN 的计算

RNN（Recurrent Neural Network）在 1980 年代提出，是一类**连接构成时间序列**的网络。它维护一个 **hidden state**（隐藏状态）$a^{\langle t \rangle}$，每读一个 token 就更新一次：

$$
a^{\langle t \rangle} = g_1\big(W_{aa}\, a^{\langle t-1 \rangle} + W_{ax}\, x^{\langle t \rangle} + b_a\big),\qquad
y^{\langle t \rangle} = g_2\big(W_{ya}\, a^{\langle t \rangle} + b_y\big)
$$

- $g_1$ 通常是 tanh，$g_2$ 通常是 softmax
- **同一组权重在每个时间步共享**，所以能处理任意长度的序列

处理 “A cute teddy bear is reading” 时：读入 A → 更新状态 → 读入 cute → 更新状态 → …… hidden state 就像一份**不断被改写的摘要**，每一步都把新词“压缩”进去。

### 4.2 RNN 的几种结构

| 结构 | 输入 → 输出 | 对应任务 |
|---|---|---|
| Many-to-one | 读完整句，输出一个标签 | 情感分类 |
| Many-to-many（等长） | 每个 token 输出一个标签 | NER、PoS tagging |
| One-to-many | 一个输入，生成一串 | 文本生成（如给定主题写一段话） |
| Many-to-many（不等长，encoder-decoder） | 读完源句，再生成目标句 | 机器翻译 |

### 4.3 训练：BPTT 与梯度消失 / 爆炸

RNN 用 **BPTT（Backpropagation Through Time）** 训练：把网络沿时间展开成一个很深的网络，再做反向传播。

问题在于，loss 对很早之前的状态的梯度，要经过一长串连乘：

$$
\frac{\partial \mathcal{L}}{\partial a^{\langle 1 \rangle}} = \frac{\partial \mathcal{L}}{\partial a^{\langle T \rangle}} \prod_{t=2}^{T} \frac{\partial a^{\langle t \rangle}}{\partial a^{\langle t-1 \rangle}}
$$

每个因子大致是 $W_{aa}^\top$ 乘以激活函数的导数（tanh 的导数 ≤ 1）。
- 如果这些因子“小于 1”，连乘 50 次后梯度**指数级衰减**到几乎为 0 → **vanishing gradient（梯度消失）**：模型学不到长距离依赖，句首的信息影响不了句尾的预测
- 如果“大于 1”，梯度**指数级增长** → **exploding gradient（梯度爆炸）**：训练发散。通常用 **gradient clipping**（梯度裁剪：梯度范数超过阈值就按比例缩小）解决

梯度爆炸容易修，梯度消失才是 RNN 的根本难题。

### 4.4 LSTM：用门控保护记忆（1997）

LSTM 在 hidden state 之外增加一条 **cell state** $c^{\langle t \rangle}$（可以理解为“长期记忆传送带”），并用三个 **gate（门）** 控制信息的流动。每个门都是一个 sigmoid 输出，取值在 0 到 1 之间，相当于“开关的开度”：

$$
\begin{aligned}
\Gamma_f &= \sigma(W_f[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_f) && \text{forget gate：旧记忆保留多少} \\
\Gamma_u &= \sigma(W_u[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_u) && \text{update / input gate：新信息写入多少} \\
\Gamma_o &= \sigma(W_o[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_o) && \text{output gate：记忆输出多少} \\
\tilde{c}^{\langle t \rangle} &= \tanh(W_c[a^{\langle t-1 \rangle}, x^{\langle t \rangle}] + b_c) && \text{候选的新记忆} \\
c^{\langle t \rangle} &= \Gamma_u \odot \tilde{c}^{\langle t \rangle} + \Gamma_f \odot c^{\langle t-1 \rangle} && \text{更新记忆} \\
a^{\langle t \rangle} &= \Gamma_o \odot \tanh(c^{\langle t \rangle}) && \text{输出 hidden state}
\end{aligned}
$$

**为什么能缓解梯度消失**：cell state 的更新主要是**逐元素的加法**（$\Gamma_f \odot c^{\langle t-1 \rangle}$ 加上新内容），而不是反复乘一个矩阵再过 tanh。当 forget gate 接近 1 时，梯度可以几乎不衰减地沿着 cell state 传回很远的过去。

**GRU**（Gated Recurrent Unit）是一个更简单的变体：只有两个门（update、reset），没有单独的 cell state，参数更少、效果相近。

### 4.5 RNN 家族的根本局限

| 方法 | 优点 | 缺点 |
|---|---|---|
| **Word2vec**（CBOW、Skip-gram） | 非常简单但很强大；embedding 直观 | 不考虑词序；embedding 不感知上下文 |
| **RNN**（传统 RNN、LSTM） | 考虑词序；当时的 state-of-the-art | 梯度消失（LSTM 缓解但没根治）；**计算必须串行，很慢** |

“串行”是关键：第 $t$ 步必须等第 $t-1$ 步算完才能开始，GPU 擅长的大规模并行用不上。序列越长，训练越慢。

---

## 5. Attention 的诞生：seq2seq 的瓶颈

### 5.1 Seq2seq（encoder-decoder）

2014 年的机器翻译模型：
- **Encoder**：一个 RNN 读完整个源句，把最后的 hidden state 当作整句的表示（一个**固定长度的向量**）
- **Decoder**：另一个 RNN 以这个向量为起点，一个词一个词地生成译文

### 5.2 问题：一个固定长度向量装不下整句话

无论源句是 5 个词还是 50 个词，都要被压缩进同一个大小的向量。句子一长，信息就被挤掉了 —— decoder 生成到后面时，已经“记不清”源句在说什么。实验上，句子越长，翻译质量（BLEU）下降越明显。

例：翻译 “A cute teddy bear is reading” → “Un ours en peluche mignon ___”。生成最后一个词时，模型真正需要的是源句中的 “reading”，但它只能依赖那个已经被反复改写的压缩向量。

### 5.3 Bahdanau attention（Bahdanau et al., 2014）

**解决方案**：decoder 每生成一个词时，不再只看一个压缩向量，而是**回头看 encoder 所有位置的 hidden states**，根据当前需要给它们分配权重。

设 encoder 在每个源位置 $j$ 的状态为 $h_j$，decoder 上一步的状态为 $s_{i-1}$：

$$
e_{ij} = \text{score}(s_{i-1}, h_j),\qquad
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k} \exp(e_{ik})},\qquad
c_i = \sum_{j} \alpha_{ij}\, h_j
$$

1. 给每个源位置打一个“相关性分数” $e_{ij}$
2. 用 softmax 变成和为 1 的权重 $\alpha_{ij}$
3. 用权重对所有 $h_j$ 加权平均，得到这一步专属的 **context vector** $c_i$，交给 decoder 使用

生成 “lit” 时，模型会把大部分权重放在 “reading” 上。把所有 $\alpha_{ij}$ 画成热力图，就能看到源词和目标词之间的**对齐（alignment）**关系 —— 这就是论文标题中 “Jointly Learning to Align and Translate” 的含义。

---

## 6. Self-Attention

### 6.1 从“看 encoder”到“看自己”

Bahdanau attention 是 decoder 去看 encoder（两个不同序列之间）。Transformer 的关键一步是：**让序列中的每个位置去看同一序列中的所有位置**，即 **self-attention**。并且完全去掉 RNN —— 不再需要按顺序传递信息。

### 6.2 Query、Key、Value 的直觉

把 attention 想象成一次**“软”的字典查询**：
- **Query (Q)**：我在找什么（当前 token 提出的问题）
- **Key (K)**：我能被怎样找到（每个 token 贴的标签）
- **Value (V)**：如果被选中，我提供什么内容

普通字典是“精确匹配一个 key，取出一个 value”；attention 是“query 和**每个** key 都算一个相似度，然后按相似度对**所有** value 加权平均”。

例：在 “A cute teddy bear is reading” 中，“reading” 的 query 相当于在问“谁在读？”，它和 “teddy bear” 的 key 匹配度最高。于是 “reading” 的新表示中会混入大量 “teddy bear” 的信息 —— 它变成了一个**知道上下文**的表示。

### 6.3 一个 token 的计算过程（数值例子）

设 $d_k = 2$，三个 token 的 key 和 value 如下（为简单起见直接给出，实际中由投影矩阵计算）：

| Token | Key | Value |
|---|---|---|
| teddy | [1, 0] | [1, 0] |
| bear | [1, 1] | [0, 1] |
| reading | [0, 1] | [1, 1] |

计算 “reading” 的输出，它的 query 是 $q = [1, 1]$：

1. **打分**：$q \cdot k$ = [1, 2, 1]
2. **缩放**：除以 $\sqrt{d_k} = \sqrt{2} \approx 1.414$ → [0.707, 1.414, 0.707]
3. **Softmax**：$e^{0.707} = 2.03$，$e^{1.414} = 4.11$，总和 8.17 → 权重 [0.248, 0.503, 0.248]
4. **加权平均 value**：$0.248 \times [1,0] + 0.503 \times [0,1] + 0.248 \times [1,1] = [0.496, 0.751]$

“reading” 的新表示中约一半来自 “bear”，这正是 query 与 bear 的 key 最匹配的结果。

### 6.4 矩阵形式

实际中所有 token 一起算。把 $n$ 个 token 的向量堆成矩阵 $X \in \mathbb{R}^{n \times d_{\text{model}}}$，用三个可学习的矩阵投影：

$$
Q = X W_Q,\quad K = X W_K,\quad V = X W_V,\qquad W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_k},\; W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\Big(\frac{Q K^\top}{\sqrt{d_k}}\Big)\, V
$$

- $QK^\top$ 是 $n \times n$ 矩阵：第 $i$ 行第 $j$ 列 = token $i$ 对 token $j$ 的相关性分数
- softmax **按行**做，每行变成和为 1 的权重
- 再乘 $V$：每个 token 得到所有 value 的加权平均
- **几次矩阵乘法就算完所有 token，完全可以并行**，这是相对 RNN 的巨大优势

### 6.5 为什么要除以 $\sqrt{d_k}$

假设 $q$ 和 $k$ 的每个分量独立、均值 0、方差 1。那么

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i,\qquad \text{Var}(q \cdot k) = d_k
$$

$d_k = 64$ 时，点积的标准差是 8，分数很容易相差几十。softmax 遇到这么大的差距会变得极端（接近 one-hot：一个位置约等于 1，其余约等于 0），处在饱和区，**梯度几乎为 0**，训练很难进行。除以 $\sqrt{d_k}$ 后方差回到 1，softmax 保持在梯度正常的区域。

### 6.6 Masked self-attention

生成文本时，第 $t$ 个位置不能“偷看”后面的词（否则训练时就直接看到答案了）。做法是在 softmax 之前，把第 $i$ 行中所有 $j > i$ 的分数设成 $-\infty$：

$$
\text{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}} + M\Big),\qquad M_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

$e^{-\infty} = 0$，所以未来位置的权重恰好为 0。这个上三角的 mask 叫 **causal mask**。

### 6.7 与 RNN 的比较

| 层类型 | 每层计算量 | 必须串行的步数 | 任意两位置间的最长路径 |
|---|---|---|---|
| **Self-attention** | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent（RNN） | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |

- Self-attention 中**任意两个 token 直接相连**（路径长度 1），长距离依赖不再需要穿过几十个时间步
- 代价是计算量和内存与 $n^2$ 成正比 —— 这是后来各种长上下文优化（Lecture 2、3、4）要解决的问题

---

## 7. Multi-Head Attention（MHA）

**思路**：与其做一次大的 attention，不如**并行做 $h$ 次小的 attention**，每个 head 用自己的投影矩阵，关注不同类型的关系。

$$
\text{head}_i = \text{Attention}(X W_Q^{(i)},\, X W_K^{(i)},\, X W_V^{(i)}),\qquad
\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O
$$

- 原论文：$d_{\text{model}} = 512$，$h = 8$，每个 head 的 $d_k = d_v = 512 / 8 = 64$
- $h$ 个 head 的输出拼接回 512 维，再乘 $W_O \in \mathbb{R}^{512 \times 512}$ 混合
- **总计算量与一个 512 维的单 head 相当**：拆分改变的是“能学到什么”，而不是成本
- 好处：不同 head 能同时捕捉不同类型的关系（语法依赖、指代关系、相邻词……）。原论文可视化中，有的 head 专门处理指代：对单词 “its”，它把注意力集中在 “its” 所指的名词上
- 类比：CNN 的一层有多个 filter，每个 filter 检测一种不同的特征

---

## 8. Transformer 架构逐层详解

原始 Transformer（Vaswani et al., 2017）是一个用于机器翻译的 **encoder-decoder** 结构：

```
源句 ──► [Input Embedding + Positional Encoding] ──► Encoder × N ──────────┐
                                                                         │ （K、V）
目标句（右移一位）──► [Output Embedding + Positional Encoding] ──► Decoder × N ◄┘
                                                                  │
                                                       Linear → Softmax → 下一个词的概率
```

### 8.1 Input embedding

- 文本先 tokenize（原论文用 BPE，英德共享约 37,000 个 token 的词表），再查 **learned embedding** 表
- 参数：$V$（vocabulary size）、$d_{\text{model}}$（embedding 维度，512）
- 小细节：原论文把 embedding 乘以 $\sqrt{d_{\text{model}}}$ 后再加位置编码，让两者数值量级相当

### 8.2 Positional Encoding（位置编码）

**为什么需要**：self-attention 对顺序不敏感。把输入的词打乱顺序，每个 token 得到的加权平均结果完全一样（只是跟着换了位置）。对它来说 “dog bites man” 和 “man bites dog” 没有区别。所以必须**把位置信息加到输入上**：

$$
\text{position-aware embedding} = \text{token embedding} + \text{position embedding}
$$

位置向量可以是 **learned**（每个位置学一个向量），也可以是 **hardcoded**。原论文用 sinusoidal（正弦）编码：

$$
PE_{(pos,\,2i)} = \sin\!\Big(\frac{pos}{10000^{2i/d_{\text{model}}}}\Big),\qquad
PE_{(pos,\,2i+1)} = \cos\!\Big(\frac{pos}{10000^{2i/d_{\text{model}}}}\Big)
$$

**怎么理解**：
- 每两个维度一组，用同一个频率；不同组的频率不同，波长从 $2\pi$ 到 $10000 \cdot 2\pi$
- 低维的波变化快，高维的波变化慢 —— 像二进制计数（最低位变得最快）的连续版本，每个位置得到一个独一无二的“指纹”
- **相对位置性质**：对任意固定偏移 $k$，$PE_{pos+k}$ 可以由 $PE_{pos}$ 经过一个**只依赖于 $k$ 的旋转**得到（因为 $\sin(a+b)$、$\cos(a+b)$ 可以用 $\sin a$、$\cos a$ 线性表示）。所以模型更容易学会“相隔 $k$ 个位置”这种相对关系。两个位置编码做点积，结果随距离增大而下降 —— 相近的位置更相似
- 好处：是一个公式，理论上能扩展到训练时没见过的长度（原论文报告 learned 与 sinusoidal 效果相近）

（Lecture 2 会讲更现代的位置编码 RoPE，它把这个“旋转”的想法直接用在 attention 内部。）

### 8.3 Encoder layer

每个 encoder 层有两个 sublayer，每个 sublayer 外面都包着**残差连接 + LayerNorm**（原论文的 Post-Norm 写法）：

$$
x' = \text{LayerNorm}\big(x + \text{MHA}(x)\big),\qquad
x'' = \text{LayerNorm}\big(x' + \text{FFN}(x')\big)
$$

1. **Multi-head self-attention**（encoder-encoder attention）：源句中每个 token 看源句中所有 token（**不加 mask**，可以看左右两边）
2. **FFN（Feed Forward Network）**：对**每个位置独立**做同样的两层全连接

$$
\text{FFN}(x) = \max(0,\, x W_1 + b_1)\, W_2 + b_2,\qquad W_1 \in \mathbb{R}^{512 \times 2048},\; W_2 \in \mathbb{R}^{2048 \times 512}
$$

**分工**：attention 负责在 token 之间**交换信息**；FFN 负责对每个 token **单独加工**信息（可以理解为存储和调用知识的地方）。

**残差连接（residual connection）**：$x + \text{Sublayer}(x)$。每个 sublayer 只需要学“在原来基础上改多少”，梯度也可以沿着恒等路径直接传到底层，深层网络更容易训练。

**LayerNorm**：对每个 token 的向量在特征维度上归一化（减均值、除标准差，再乘可学习的缩放 $\gamma$、加偏移 $\beta$），稳定数值、加速收敛。

参数：$N$（层数，原论文 6）、$h$（head 数，8）、$d_{FF}$（FFN 隐藏宽度，2048）、$d_{\text{key}}$ / $d_{\text{value}}$（每个 head 的维度，64）、$d_{\text{model}}$（512）。

Encoder 的输出是 **context-aware encoded embeddings**：每个 token 的向量已经融合了整句话的上下文。

### 8.4 Output “shifted right”

- 训练时，decoder 的输入是**目标句右移一位**：开头插入 `[BOS]`（beginning of sentence）
- 于是在位置 $t$，decoder 的输入是前 $t-1$ 个正确的词，要预测第 $t$ 个词
- 推理（真正翻译）时从 `[BOS]` 开始，一个词一个词地生成，直到生成 `[EOS]`

### 8.5 Decoder layer

每个 decoder 层有三个 sublayer（同样都有残差 + LayerNorm）：

1. **Masked self-attention**（decoder-decoder）：目标句中每个 token 只能看它**之前**的 token（causal mask）
2. **Encoder-decoder attention**（cross-attention）：**Query 来自 decoder，Key 和 Value 来自 encoder 的最终输出**。这正是 Bahdanau attention 的位置：生成每个目标词时，去源句中找相关的部分
3. **FFN**

### 8.6 Output layer

- **Linear**：把 $d_{\text{model}}$ 维向量映射到 $V$ 维（每个词一个分数，叫 logits）
- **Softmax**：把分数变成词表上的概率分布
- 本质上是一个**分类问题，类别 = 词表中的每个词**
- 原论文让输入 embedding、输出 embedding 和这个 linear 层**共享同一个权重矩阵**（weight tying），减少参数

### 8.7 训练细节

**Teacher forcing**：训练时 decoder 的输入是**真实的**目标句（右移后），而不是模型自己上一步的预测。配合 causal mask，整句可以**一次并行**地算出每个位置的预测和 loss。

**Loss**：每个位置的 cross-entropy，再取平均：

$$
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta(y_t \mid y_{<t}, x)
$$

**Label smoothing**（$\epsilon = 0.1$）：
- 来自 2015 年一篇视觉论文的观察：**模型过度自信是有害的**
- 做法：训练目标不用纯 one-hot（正确词 = 1，其余 = 0），而是正确词 $1 - \epsilon$，再把 $\epsilon$ 平均分给所有词：

$$
y'_k = (1 - \epsilon)\cdot \mathbb{1}[k = \text{正确词}] + \frac{\epsilon}{V}
$$

- 效果：防止过拟合的通用技巧。perplexity 会略变差（模型不再把概率全押在一个词上），但 accuracy 和 BLEU 提高

**学习率调度**：Adam 优化器 + **warmup**，先线性增大、再按步数的平方根倒数衰减：

$$
lr = d_{\text{model}}^{-0.5} \cdot \min\big(\text{step}^{-0.5},\; \text{step} \cdot \text{warmup}^{-1.5}\big),\qquad \text{warmup} = 4000
$$

**规模与结果**：
- Base 模型约 6500 万参数（$N=6$，$d_{\text{model}}=512$，$d_{FF}=2048$，$h=8$）；Big 模型约 2.13 亿参数（$d_{\text{model}}=1024$，$d_{FF}=4096$，$h=16$）
- 参数量粗算（base）：每个 encoder 层 MHA 约 $4 d^2 \approx 1.05$M、FFN 约 $2 d \cdot d_{FF} \approx 2.1$M；每个 decoder 层多一个 MHA；6+6 层合计约 44M，加上约 19M 的 embedding，约 63M，与 65M 吻合
- WMT 2014 英德翻译 BLEU 28.4，英法翻译 BLEU 41.8，刷新当时的纪录；big 模型在 8 张 P100 GPU 上训练 3.5 天 —— 训练成本远低于之前的模型

---

## 9. 端到端例子：把一句英文翻译成法文

目标：“A cute teddy bear is reading.” → “Un ours en peluche mignon lit.”（以下维度按原论文 base 模型）

### 9.1 Encoder 侧

1. **Tokenize**，加特殊 token：`[BOS] A cute teddy bear is reading . [EOS]`，共 $n = 9$ 个 token
2. **Embedding**：每个 token 查表得到 512 维向量
3. **加位置编码** → position-aware embeddings，堆成矩阵 $X \in \mathbb{R}^{9 \times 512}$
4. **投影**：$Q = XW_Q$、$K = XW_K$、$V = XW_V$，每个 head 中都是 $9 \times 64$
5. **打分**：$QK^\top / \sqrt{64}$ 得到 $9 \times 9$ 的分数矩阵 —— 每对 token 之间一个分数
6. **Softmax**（按行）得到权重，再乘 $V$：每个 token 的输出是 **values 的加权平均，权重是 query 与 key 相似度的函数**，形状 $9 \times 64$
7. **Multi-head**：8 个 head 各做一遍，拼接成 $9 \times 512$，再乘 $W_O$
8. **残差 + LayerNorm → FFN → 残差 + LayerNorm**，得到这一层的 context-aware embeddings（仍是 $9 \times 512$）
9. 重复 $N = 6$ 层，得到最终的 **encoder 输出**（$9 \times 512$），作为 decoder 的“记忆”

### 9.2 Decoder 侧（逐词生成）

1. 输入 `[BOS]`
2. **Masked self-attention**：此时序列中只有 `[BOS]` 自己
3. **Encoder-decoder attention**：用 decoder 的 query 去查 encoder 输出的 key / value，找到源句中相关的部分
4. **FFN → Linear → Softmax**：得到词表上的概率分布，例如 `[0.001, 0.0003, …, 0.4, …, 0.002]`，取概率最高的 “Un”
5. 把 “Un” 接到输入后面：`[BOS] Un`，再跑一遍 decoder，得到 “ours en peluche”
6. 重复，直到生成 `[EOS]`：`Un ours en peluche mignon lit [EOS]`

这种“每次选概率最高的词”的做法叫 **greedy decoding**；Lecture 3 会介绍 beam search、sampling 等更好的解码策略。

### 9.3 训练 vs 推理

| | 训练 | 推理 |
|---|---|---|
| Decoder 的输入 | 真实目标句（右移一位，teacher forcing） | 模型自己之前生成的词 |
| 是否并行 | 整句一次并行处理（causal mask 保证不偷看） | **只能一个词一个词地生成**（autoregressive） |
| 每步计算 | 一次 forward 得到所有位置的 loss | 每生成一个词都要一次 forward |

推理时“只能逐个生成”是后来 LLM 推理优化（KV cache、speculative decoding 等，见 Lecture 3）的出发点。

---

## 10. 本课程的常见缩写

| 类别 | 缩写 |
|---|---|
| 基于 Transformer 的模型 | BERT、T5、GPT、LLaMA |
| 训练策略 | SFT（Supervised Fine-Tuning）、PEFT（Parameter-Efficient Fine-Tuning）、FLAN、RL、RM（Reward Model）、RLHF、PPO、DPO |
| 数据集 | MNLI、WNLI、C4、SQuAD、GLUE、MRPC |
| 其他架构与技术 | LSTM、GRU、GloVe、BPE、CoT（Chain-of-Thought）、ToT（Tree-of-Thought）、SC（Self-Consistency）、RAG |
| 任务 | NER、PoS、MLM（Masked Language Modeling）、NSP（Next Sentence Prediction）、MT、QA、NLG（Natural Language Generation） |
| 指标 | F1、PPL、ROUGE、BLEU、METEOR、LaaJ（LLM-as-a-Judge）、WER（Word Error Rate） |

---

## 11. 要点回顾

1. NLP 任务可按输出形状分为分类、逐 token 分类、生成三类；生成本质上是“在词表上反复做分类”
2. 类别不平衡时不能只看 accuracy；BLEU 偏 precision，ROUGE 偏 recall，perplexity 衡量模型的“惊讶程度”
3. Subword tokenization（BPE、WordPiece）在词表大小和序列长度之间取得平衡，并基本解决 OOV
4. Word2vec 通过代理任务学到稠密词向量，但向量是静态的，不随上下文变化
5. RNN 串行计算、存在梯度消失；LSTM 用门控和加法更新的 cell state 缓解梯度消失，但仍是串行的
6. Attention 让 decoder 每一步都能回看源句的所有位置，解决了固定长度向量的瓶颈
7. Self-attention = softmax($QK^\top/\sqrt{d_k}$)$V$：任意两个位置直接交互、可并行；代价是 $O(n^2)$
8. Transformer = embedding + 位置编码 + N ×（attention + FFN，各带残差和 LayerNorm）+ linear + softmax

---

## 12. 自测题

<details>
<summary>1. 为什么说 self-attention 本身“不知道”词的顺序？位置编码是怎么解决的？</summary>

Self-attention 的输出是对所有 value 的加权平均，权重只取决于 query 和 key 的内容。把输入打乱顺序，每个 token 算出的结果不变，只是跟着换了位置，所以它对顺序不敏感。位置编码把一个与位置相关的向量加到每个 token 的 embedding 上，使得同一个词在不同位置有不同的输入向量，模型就能区分顺序。
</details>

<details>
<summary>2. $d_k = 64$ 时，如果不除以 $\sqrt{d_k}$ 会发生什么？</summary>

点积的方差约为 64（标准差 8），不同 token 之间的分数差距很大，softmax 会接近 one-hot，落入饱和区，梯度接近 0，训练困难。除以 $\sqrt{64} = 8$ 后方差回到 1。
</details>

<details>
<summary>3. 一个模型给某句话中 3 个词的概率都是 0.1，perplexity 是多少？</summary>

PPL = exp(−(1/3)·3·ln 0.1) = exp(ln 10) = 10。相当于模型每一步在 10 个词之间均匀犹豫。
</details>

<details>
<summary>4. Encoder 的 self-attention、decoder 的 masked self-attention、encoder-decoder attention 三者的 Q、K、V 分别来自哪里？</summary>

Encoder self-attention：Q、K、V 都来自 encoder 当前层的输入（源句），无 mask。Decoder masked self-attention：Q、K、V 都来自 decoder 当前层的输入（目标句），加 causal mask。Encoder-decoder attention：Q 来自 decoder，K 和 V 来自 encoder 的最终输出。
</details>

<details>
<summary>5. LSTM 为什么比普通 RNN 更能记住长距离信息？</summary>

普通 RNN 的梯度要经过一长串“矩阵乘法 + tanh 导数”的连乘，容易指数衰减。LSTM 的 cell state 主要通过逐元素加法更新，forget gate 接近 1 时，梯度可以几乎不衰减地沿 cell state 传回很远的时间步。
</details>

<details>
<summary>6. 训练时 Transformer decoder 可以并行，推理时为什么不行？</summary>

训练时整句目标已知，用 teacher forcing 把右移后的目标句一次性输入，causal mask 保证每个位置看不到未来，所以所有位置可以一次算完。推理时下一个词依赖于模型刚生成的词，必须生成一个、再输入、再生成下一个，只能串行。
</details>

<details>
<summary>7. 用 BPE 时，训练语料中从没出现过的词会怎样？</summary>

它会被拆成训练中学到的子词片段（例如 lowest → low + est），只要片段在词表中就不需要 [UNK]。Byte-level BPE 从 256 个字节开始，任何文本都能表示，彻底没有 OOV。
</details>

---

## 参考资料

- 课程网站：cme295.stanford.edu
- 教材：*Super Study Guide: Transformers & Large Language Models*（https://superstudy.guide）
- VIP Cheatsheet：https://github.com/afshinea/stanford-cme-295-transformers-large-language-models
- Vaswani et al., “Attention Is All You Need”, 2017
- Bahdanau et al., “Neural Machine Translation by Jointly Learning to Align and Translate”, 2014
- Mikolov et al., “Efficient Estimation of Word Representations in Vector Space”, 2013
- Hochreiter & Schmidhuber, “Long Short-Term Memory”, 1997
- Sennrich et al., “Neural Machine Translation of Rare Words with Subword Units” (BPE), 2016
- Papineni et al., “BLEU”, 2002；Lin, “ROUGE”, 2004
- Kazemnejad, “Transformer Architecture: The Positional Encoding”, 2019
