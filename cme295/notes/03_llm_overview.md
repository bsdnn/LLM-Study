# Lecture 3 · LLM 概览：MoE、解码策略、Prompting 与推理优化

> Stanford CME 295：Transformers & Large Language Models（Afshine Amidi & Shervine Amidi，Fall 2025）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture3.pdf`](../slides/fall25-cme295-lecture3.pdf)
> 预计阅读时间：15–20 分钟。预备知识：Lecture 1–2（self-attention、decoder-only、GQA）。

## 本讲脉络

```
什么是 LLM（语言模型的概率定义）
→ MoE：只激活一部分参数就能把模型做大；路由、负载均衡、expert 学到了什么
→ 生成回复：greedy → beam search → sampling（top-k、top-p、temperature）→ guided decoding
→ Prompting：context 长度、prompt 结构、in-context learning、chain-of-thought、self-consistency
→ 推理优化：prefill 与 decode → KV cache → GQA → PagedAttention → MLA → speculative decoding → MTP
```

**一句话总结**：现代 LLM 是**超大的 decoder-only Transformer**，每次只预测下一个 token。本讲回答三个问题：
1. 怎么在**不增加每个 token 计算量**的前提下把模型做大？→ **MoE**
2. 怎么从概率分布中**挑出 token**？→ **解码策略**
3. 怎么让推理**更快、更省显存**？→ **KV cache 及一系列优化**

---

## 1. 什么是 LLM

**Language model（语言模型）**：一个给 token 序列分配概率的统计或机器学习模型。它估计 $p(x_1, \dots, x_n)$，通常用链式法则分解成“逐个预测下一个 token”：

$$
p(x_1, \dots, x_n) = \prod_{t=1}^{n} p(x_t \mid x_1, \dots, x_{t-1})
$$

例：$p(\text{A teddy bear is cute}) = p(\text{A}) \cdot p(\text{teddy} \mid \text{A}) \cdot p(\text{bear} \mid \text{A teddy}) \cdots$

**Large（大）** 体现在三个方面：
- **模型规模**：数十亿（billions）参数以上
- **训练数据**：数千亿（hundreds of billions）tokens 以上
- **算力**：大量 GPU

**结构**：几乎都是 **decoder-only Transformer**（causal mask，只看左边）。例：GPT 系列、LLaMA、Gemma、DeepSeek、Mistral、Qwen……

---

## 2. MoE（Mixture of Experts）

### 2.1 动机

模型越大，效果越好。但普通（dense）模型中，**每个 token 都要经过全部权重**，计算量与参数量成正比 —— 参数翻倍，每个 token 的计算也翻倍。

**观察**：并不是所有权重对每个输入都有用。处理 Python 代码时，模型中“擅长法语”的那部分参数可能用不上。

**想法**：把模型的一部分拆成很多个子网络（**expert**），每个 token 只用其中一小部分。这样**总参数量可以很大（存储更多知识），而每个 token 的计算量保持不变**。

### 2.2 结构：experts + router

- $n$ 个 **expert**（$E_1, \dots, E_n$），每个是一个独立的子网络
- 一个 **gating network / router**（$G$），根据输入决定每个 expert 的权重

**Dense MoE**：输出是**所有** expert 输出的加权平均（权重如 0.1、0.8、0.05……）—— 每个 expert 都要计算，并没有省计算。

**Sparse MoE**（Shazeer et al., 2017）：只选权重最大的 **top-$k$** 个 expert，其他 expert 根本不计算：

$$
G(x) = \text{softmax}\big(\text{TopK}(x W_g,\, k)\big),\qquad
y = \sum_{i \in \text{TopK}} G(x)_i \cdot E_i(x)
$$

（TopK 把不在前 $k$ 名的分数设成 $-\infty$，softmax 后它们的权重恰好为 0。）

Shazeer 等人的原论文还在 router 分数上加了可学习强度的高斯噪声（noisy top-k gating），帮助负载均衡和探索。

### 2.3 MoE 在 Transformer 中的位置

- 把每个 Transformer block 中的 **FFN 换成 $n$ 个 FFN expert + 一个 router**
- **Attention 层保持不变**（所有 token 共享）
- **路由按 token 进行**：同一句话里的不同 token，可能被送到不同的 expert

**参数怎么算**：以 **Mixtral 8x7B**（Jiang et al., 2024）为例，每层 8 个 expert，每个 token 选 top-2：
- 总参数约 **46.7B**（不是 8 × 7 = 56B，因为 attention、embedding 等是共享的，只有 FFN 被复制了 8 份）
- 每个 token 激活的参数约 **12.9B**
- 结果：用约 13B 模型的计算量，得到接近 70B dense 模型（Llama 2 70B）的效果

**存储 vs 计算**：
- **显存**由**总参数**决定（所有 expert 都要放在显存里，因为不知道下一个 token 会去哪个）
- **计算**由**激活参数**决定

### 2.4 训练难点一：Routing collapse（路由塌缩）

**症状**：router 几乎总是选中同一个（或少数几个）expert。

**为什么会发生**：一开始某个 expert 碰巧表现好一点 → router 更多地选它 → 它得到更多训练、变得更好 → router 更偏爱它 → 恶性循环。其他 expert 几乎得不到训练，等于浪费了参数。

**解决**：加一个 **auxiliary load-balancing loss（辅助负载均衡损失）**（Switch Transformers，Fedus et al., 2021），强迫其他 expert 也“参与游戏”：

$$
\mathcal{L}_{\text{aux}} = \alpha \cdot n \cdot \sum_{i=1}^{n} f_i \cdot P_i
$$

- $f_i$ = 实际被路由到 expert $i$ 的 token 比例（不可导，是计数）
- $P_i$ = 所有 token 对 expert $i$ 的平均路由概率（可导，梯度通过它传回 router）
- $\alpha$ 是一个小系数（如 0.01）

**算例**（$n = 4$ 个 expert）：
- 完全均衡：$f_i = P_i = 0.25$，$\sum f_i P_i = 4 \times 0.0625 = 0.25$，乘以 $n = 4$ 得 **1.0**（这是最小值）
- 塌缩：$f = [1, 0, 0, 0]$，$P = [0.97, 0.01, 0.01, 0.01]$，$\sum f_i P_i = 0.97$，乘以 4 得 **3.88**

塌缩时 loss 大得多，梯度会推动 router 降低对过热 expert 的偏好。

**其他工程细节**：
- **Expert capacity（容量）**：每个 expert 每个 batch 最多处理多少 token。超出容量的 token 会被丢弃（直接走残差连接），避免某个 expert 过载拖慢整个 batch
- **DeepSeek 的改进**：DeepSeekMoE 用“很多小 expert + 少量共享 expert”（细粒度划分）；DeepSeek-V3 用“给每个 expert 的路由分数加一个可调偏置”实现负载均衡，不再需要辅助 loss（auxiliary-loss-free）

### 2.5 Expert 学到了什么？

Mixtral 论文把每个 token 按它被分配的 expert 上色，发现：
- Expert **并不按“主题”分工**：不存在“生物 expert”“数学 expert”
- 更多是按**语法 / 形式**特征分工：例如 Python 代码中的 `self`、缩进 token，英文中的某些功能词，往往被路由到同一个 expert
- 相邻 token 常常被送到同一个 expert

所以“expert”这个名字有点误导 —— 它们不是人类意义上的“领域专家”。

---

## 3. 如何生成回复（Decoding）

LLM 的输出是下一个 token 在整个词表上的概率分布。给定 “[BOS] A teddy bear is”，模型给出 “cute” 0.3、“reading” 0.2、“soft” 0.1……然后需要一个**解码策略（decoding strategy）**从中挑一个 token，接到输入后面，再预测下一个，直到生成 `[EOS]`。

### 3.1 Greedy decoding（贪心解码）

- **做法**：每一步选概率最高的 token
- **问题**：
  - **每一步最优 ≠ 整句最优**。例如第一步 “the”（0.5）比 “a”（0.4）高，但 “a” 后面可能跟着一个概率很高的续写，使整句概率更大
  - 输出不自然、缺乏多样性，同一个 prompt 永远得到同一个回答
  - 容易陷入**重复循环**（“I think that I think that I think that…”）

### 3.2 Beam search（束搜索）

- **做法**：每一步保留 $k$ 条（**beam width**）**累积概率最高**的路径。每条路径扩展所有可能的下一个 token，然后在所有候选中再保留最好的 $k$ 条，直到所有路径都以 `[EOS]` 结束
- 例（$k = 2$）：第一步保留 “a” 和 “the”；“a” 可以接 “cute”（0.7）、“fluffy”（0.2），“the” 可以接 “sleepy”（0.6）……计算每条路径的累积概率（各步概率相乘），保留最好的 2 条；最后在 “… dog”“… kitten”“… sloth” 等完整句子中选整体概率最高的
- 实践细节：用对数概率相加代替概率相乘（避免数值下溢）；做**长度归一化**（除以长度），否则短句子总是占优（每多一个词，概率就多乘一个小于 1 的数）
- **问题**：
  - 计算量是 greedy 的约 $k$ 倍
  - 仍然**缺乏多样性和创造力**，倾向于平淡、高频的句子，也容易重复（Holtzman et al., 2019 系统地记录了这种“文本退化”现象）
- **适用**：机器翻译、摘要这类“有比较标准答案”的任务

### 3.3 Sampling（采样）

- **做法**：按概率分布**随机采样**下一个 token：$x_t \sim p(\cdot \mid x_{<t})$
- **优点**：多样、更像人类写作
- **问题**：词表有几万个 token，长尾中大量低概率 token 加起来概率并不小，偶尔会采到离谱的 token，一旦采错，后面的文本就会跑偏

### 3.4 限制采样范围：Top-k 与 Top-p

**Top-k sampling**：只在概率最高的 $k$ 个 token 中采样（如 $k = 4$），先把这 $k$ 个的概率重新归一化。
- 问题：$k$ 是固定的。模型很确定时（只有一两个合理选项），$k = 50$ 太宽；模型不确定时（有几百个合理选项），$k = 50$ 又太窄

**Top-p sampling（nucleus sampling，核采样）**（Holtzman et al., 2019）：只在**累积概率 ≥ $p$ 的最小 token 集合**中采样。

**算例**：概率从高到低为 [0.5, 0.2, 0.15, 0.1, 0.05]，$p = 0.9$：
- 前 3 个累积 0.85 < 0.9，不够
- 前 4 个累积 0.95 ≥ 0.9 → 保留前 4 个
- 重新归一化：[0.5, 0.2, 0.15, 0.1] / 0.95 = [0.526, 0.211, 0.158, 0.105]，从中采样

**自适应性**：模型很确定时（如第一个 token 概率 0.95），集合只有 1 个 token；不确定时集合会自动变大。所以 top-p 通常比 top-k 更好用。

### 3.5 概率从哪里来：Softmax 与 Temperature

模型最后一层输出每个 token 的分数（**logits**）$z_i$，用带温度 $T$ 的 softmax 转成概率：

$$
p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}
$$

**算例**：三个 token 的 logits 为 [2, 1, 0]：

| Temperature | 计算 | 概率 | 效果 |
|---|---|---|---|
| $T = 0.5$ | logits 变成 [4, 2, 0] | [0.867, 0.117, 0.016] | 更尖锐，几乎总选第一个 |
| $T = 1$ | 原始 | [0.665, 0.245, 0.090] | 原始分布 |
| $T = 2$ | logits 变成 [1, 0.5, 0] | [0.506, 0.307, 0.186] | 更平坦，各 token 更均匀 |

- **$T \to 0$**：接近 greedy，输出稳定、保守，适合事实问答、代码
- **$T$ 大**：输出更多样、更有创意，也更容易出错，适合头脑风暴、写故事
- 实践中通常先用 temperature 调整分布，再做 top-k / top-p 截断

**为什么 $T = 0$ 也可能不确定？**（推荐阅读：He et al., 2025，“Defeating Nondeterminism in LLM Inference”）
- 浮点加法不满足结合律：$(a + b) + c$ 和 $a + (b + c)$ 的结果可能有微小差别
- GPU kernel 做求和时的顺序，会随 **batch 大小**（同时在服务器上处理的请求数）而变化
- 服务器负载不同 → batch 大小不同 → 微小的数值差异 → 某一步两个 token 的分数恰好接近时，argmax 结果翻转 → 后续文本完全不同
- 解决方法：使用“batch-invariant”的 kernel（无论 batch 多大都用同样的计算顺序）

### 3.6 Guided decoding（约束解码）

**动机**：需要输出特定格式。例如：

> Generate a description of my 33-year old teddy bear who likes reading. Do this in JSON format.

期望输出：

```json
{"first_name": "teddy", "last_name": "bear", "age": 33, "hobby": "reading"}
```

仅靠 prompt 要求，模型仍可能输出不合法的 JSON（少一个引号、多一句解释）。

**思路**：在每一步解码时**只允许“合法”的下一个 token**。
- 根据格式（JSON schema、正则表达式、语法）维护一个**有限状态机**，知道当前位置哪些 token 是合法的
- 把所有不合法 token 的 logit 设为 $-\infty$（mask 掉），再做 softmax 和采样
- 例：刚输出 `{` 之后，只允许 `"first_name"` 这样的 key，而不能是 “road”、“sun”；在 `"age": ` 之后只允许数字
- 效果：输出**保证**符合格式。各家 API 的 “structured outputs” 功能就是这样实现的
- 注意：约束太死、与模型自然的表达方式冲突时，可能略微影响内容质量

---

## 4. Prompting 策略

### 4.1 Context length

一次请求中，**输入 prompt + 输出**共同占用 **context window**：

```
Is a teddy bear cute?   →   Yes.
└──── context size ────────────┘
└──────────── window size（模型能处理的上限）─────────────┘
```

数量级参考（1 个 token 约 0.75 个英文单词）：一句话约几十 tokens；一篇论文约 1 万 tokens；一本小说约 10–15 万 tokens；一个中型代码库可达数百万 tokens。

**小心 “context rot”**（Hong et al., 2025）：即使信息都在 context 里，输入 token 越多，模型的表现越差。所以“能塞进去”不等于“能用好”。

### 4.2 Prompt 的结构

一个好的 prompt 通常包括四部分：

| 部分 | 作用 | 例子 |
|---|---|---|
| **Context（背景）** | 交代情境 | My teddy bear had a long day and needs a bedtime story. |
| **Instructions（指令）** | 要做什么 | Generate a bedtime story that takes place in a specific location. |
| **Input（输入）** | 具体数据 | Location: Country of teddy bears |
| **Constraints（约束）** | 限制条件 | The story needs to be suitable for teddy bears that are tired. |

### 4.3 In-Context Learning（ICL，上下文学习）

GPT-3 论文（Brown et al., 2020，“Language Models are Few-Shot Learners”）的关键发现：**不更新任何权重**，只在 prompt 中给几个例子，足够大的模型就能学会新任务。

| | Zero-shot | Few-shot |
|---|---|---|
| 做法 | 直接问，不给例子 | prompt 中给几组输入 / 输出例子 |
| 效果 | 很依赖模型本身的能力 | 通常更好 |

```
Few-shot 例子：
Translate English to French:
sea otter => loutre de mer
cheese => fromage
teddy bear =>
```

给例子通常更好，但：
- 需要人工准备好的例子
- 增加计算量和成本（每次请求都要带上例子）
- 增加延迟
- 效果对例子的选择、顺序、格式比较敏感

### 4.4 Chain-of-Thought（CoT，思维链）（Wei et al., 2022）

**思路**：让模型在给出答案前先写出推理过程，能显著提升效果。

**Few-shot CoT 例子**：
```
Q: How old is this bear?
A: The bear was born in 2020. It is therefore 4.        ← 示例中包含推理过程
Q: How old will the bear be next year?
A: It will be one year older than its age this year, which was 4. Hence, it will be 5.
```

对比：不用 CoT 时模型直接答 “5”，遇到更复杂的题时很容易算错。

**效果**：在数学应用题 benchmark GSM8K 上，540B 参数的 PaLM 用普通 few-shot prompting 准确率约 18%，用 CoT prompting 提升到约 57%。

**Zero-shot CoT**（Kojima et al., 2022）：连例子都不给，只在问题后加一句 “Let's think step by step.”，也能大幅提升推理题的准确率。

**为什么有效**：
- 每生成一个 token，模型做的计算量是固定的。难题需要更多“计算步骤”，让模型把中间步骤写出来，相当于给了它**更多计算**
- 后面的 token 可以**基于前面写下的中间结果**继续推理，相当于有了一张草稿纸

**讨论**：
- ✅ 可解释，能看到模型是怎么想的
- ❌ 更多 tokens → 更高成本和延迟

### 4.5 Self-Consistency（自洽性）（Wang et al., 2022）

**思路**：对同一个问题，用 temperature > 0 采样**多条** CoT 推理路径，然后对最终答案**多数投票**。

例：三条推理路径
- 路径 1：“It will be one year older than its age this year, which was 4. Hence, it will be 5.” → 5
- 路径 2：“The bear was born in 2020. It will therefore be 5.” → 5
- 路径 3：“Next year is 2024. The bear will then be 4.” → 4
- 多数投票 → **5**

**直觉**：一道题的正确推理方式可能有很多种，但它们会殊途同归到同一个答案；错误的推理则各错各的，答案分散。所以出现次数最多的答案更可能是对的。

**效果**：在 GSM8K 上比普通 CoT 再提升约 18 个百分点（PaLM 540B）。

**讨论**：在效果和额外成本（采样 $N$ 次 = $N$ 倍成本）之间权衡。

---

## 5. 推理优化

### 5.1 挑战与分类

LLM 推理很昂贵，有很多可以优化的维度：

| “精确”优化（结果不变） | 近似优化（可能影响结果） |
|---|---|
| 避免冗余计算（KV cache） | 架构改动（GQA、MLA） |
| 内存管理（PagedAttention） | 数值表示（如量化） |
| 改写计算方式（如 FlashAttention，Lecture 4） | Token 预测方式（speculative decoding、MTP） |
| | 基于 attention 的技巧（sliding window 等） |

### 5.2 先理解两个阶段：Prefill 与 Decode

- **Prefill（预填充）**：一次 forward 处理整个 prompt，所有 prompt token 并行计算，并输出第一个生成的 token。计算密集（compute-bound），决定**首个 token 的延迟**
- **Decode（解码）**：之后每次只生成一个新 token，每个 token 都要一次 forward。每次 forward 都要把**全部模型权重从显存读一遍**，却只做很少的计算，所以是**内存带宽受限（memory-bound）**的，决定**每秒生成多少 token**

后面的很多优化，都是围绕“decode 阶段太慢、显存太紧”展开的。

### 5.3 KV Caching

**动机**：生成第 $t$ 个 token 时，新 token 要和**所有之前的 token** 做 attention，需要它们的 K 和 V。如果每一步都重新计算之前所有 token 的 K、V，会有大量重复计算（生成 $n$ 个 token 共需重复计算约 $n^2/2$ 次）。

**观察**：之前 token 的 K、V **不会变**（causal mask 下，前面的 token 看不到后面的 token，所以它们的表示与后面生成什么无关）。

**做法**：把每层每个 token 的 K、V **存进 cache**。生成新 token 时：
1. 只为这一个新 token 计算 $q$、$k$、$v$
2. 把新的 $k$、$v$ 追加进 cache
3. 用新的 $q$ 和 cache 里所有的 K、V 做 attention

**代价**：cache 占用大量显存。

$$
\text{KV cache 大小} = 2 \times L \times H_{kv} \times d_{\text{head}} \times \text{bytes} \times \text{序列长度} \times \text{batch}
$$

**算例**：Llama 2 7B（32 层、32 个 KV head、head 维度 128、FP16）：
- 每个 token：$2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$ 字节 ≈ **0.5 MB**
- 一条 4,096 token 的序列：约 **2 GB**
- batch 16：约 **32 GB** —— 比模型权重本身（约 13.5 GB）还大

所以 **KV cache 往往是限制并发量和 context 长度的瓶颈**。后面几种方法都在想办法缩小它或用好它。

### 5.4 共享 attention heads：GQA

（Lecture 2 已介绍。）标准 MHA 中 query head 数 = key head 数 = value head 数 = $h$。**GQA** 把 $h$ 个 query head 分成 $G$ 组（$G < h$），每组共享一个 K/V head，KV cache 缩小 $h / G$ 倍。例：Llama 2 70B 用 64 个 query head、8 个 KV head，缩小 8 倍。**MQA** 是 $G = 1$ 的极端情况。

### 5.5 PagedAttention（Kwon et al., 2023，vLLM）

**观察**：传统做法为每个请求**预留一整块连续的显存**来放 KV cache，大小按最大可能长度（如 2,048 token）分配。但实际上：
- 生成往往比最大长度短得多 → 预留了却没用上（**reservation / internal fragmentation，内部碎片**）
- 不同请求的连续块之间留下用不了的空隙（**external fragmentation，外部碎片**）
- vLLM 论文测得，已有系统中 **60%–80% 的 KV cache 显存被浪费**

**思路**：借鉴操作系统的**虚拟内存分页**：
- 把 KV cache 切成固定大小的 **block**（如每块存 16 个 token 的 K、V）
- 每个请求的 block 可以存放在**不连续**的物理显存中
- 用一张 **block table** 记录“逻辑上第几块”对应“物理上哪一块”
- **按需分配**：生成满一块才分配下一块，浪费最多不到一个 block

**效果**：
- 显存浪费降到 4% 以下 → 同样的显存能同时服务更多请求 → 吞吐量提高 2–4 倍
- **共享**：多个请求有相同前缀（如同一个 system prompt）或同一个 prompt 采样多个回答时，可以共享相同的物理 block（写入时才复制，copy-on-write）

### 5.6 Latent attention（MLA，DeepSeek-V2）

**目标**：缩小存在显存中的 K、V 的维度。

- 标准 MHA：每层每个 token 要存 $h$ 个 head 的 K 和 V，共 $2 \times h \times d_{\text{head}}$ 个数（DeepSeek-V2 中是 $2 \times 128 \times 128 = 32{,}768$ 个数）—— 太大
- **MLA（Multi-head Latent Attention）**：先把 token 的 hidden state **压缩成一个低维的 latent 向量** $c$（如 512 维），**只缓存 $c$**；计算 attention 时，再用上投影矩阵从 $c$ 恢复出各个 head 的 K 和 V

$$
c = W_{\text{down}}\, h,\qquad K = W_{UK}\, c,\qquad V = W_{UV}\, c
$$

- 关键技巧：上投影矩阵 $W_{UK}$ 可以和 query 的投影矩阵**合并**（矩阵乘法满足结合律），$W_{UV}$ 可以和输出投影合并。所以推理时并不需要真的把 K、V 展开，几乎不增加计算
- 小细节：RoPE 和这种合并不兼容，所以 MLA 额外用一小段单独的维度（decoupled RoPE）来携带位置信息
- **效果**：与 DeepSeek 67B 相比，KV cache 减少约 **93%**，最大生成吞吐提高约 5.8 倍，质量与 MHA 相当甚至更好

| 方法 | 每 token 每层缓存的内容 | 相对 MHA |
|---|---|---|
| MHA | $h$ 个 K + $h$ 个 V | 1× |
| GQA | $G$ 个 K + $G$ 个 V | $G/h$ |
| MQA | 1 个 K + 1 个 V | $1/h$ |
| MLA | 1 个低维 latent $c$（+ 一小段 RoPE key） | 很小，且质量不降 |

### 5.7 Speculative Decoding（投机解码）（Chen et al., 2023）

**问题**：大模型每次 forward 只产出一个 token。而 decode 阶段是 memory-bound 的 —— 读一遍权重的时间里，GPU 的算力大量闲置。**用大模型一次验证 $k$ 个 token，花的时间和生成 1 个 token 差不多。**

**思路**：用一个小而快的 **draft model**（草稿模型）先猜，再让大的 **target model**（目标模型）一次性验证。

**步骤**：
1. Draft model 连续生成 $k$ 个 token（如 “and smart …”），记下它对每个 token 给出的概率 $P_1, \dots, P_k$
2. 把这 $k$ 个 token 一次性喂给 target model，**一次 forward 并行**得到每个位置的目标概率 $Q_1, \dots, Q_{k+1}$
3. 从左到右逐个验证每个被猜的 token $x$：
   - 若 $Q_i(x) \ge P_i(x)$：✅ **接受**
   - 否则：以概率 $Q_i(x) / P_i(x)$ 接受，以概率 $1 - Q_i(x)/P_i(x)$ ❌ **拒绝**
4. 一旦某个位置被拒绝，就从修正分布 $[Q_i - P_i]_+$（把负的部分截为 0 后归一化）中重新采样这个位置的 token，然后结束本轮；如果 $k$ 个全被接受，还能额外从 $Q_{k+1}$ 中采样一个 token
5. 从新的位置开始下一轮

**关键性质：无损**。这个接受 / 拒绝规则保证最终输出的分布**与只用 target model 采样完全相同**。直觉：对每个 token $x$，被接受的概率是 $\min(P(x), Q(x))$；被拒绝后从 $[Q - P]_+$ 中补上的部分，恰好补齐了 $Q(x)$ 比 $P(x)$ 多出来的那部分。两者相加正好等于 $Q(x)$。

**加速多少**：取决于**接受率** $\alpha$（draft 猜得有多准）。每次大模型 forward 平均产出的 token 数为

$$
\frac{1 - \alpha^{k+1}}{1 - \alpha}
$$

例：$\alpha = 0.8$、$k = 4$ 时约为 3.4 个 token。Chen et al. 在 Chinchilla 70B 上实测加速 2–2.5 倍。

**draft model 的要求**：足够快（通常小 10 倍以上）、与 target model 用同一个 tokenizer、预测分布尽量接近 target。

### 5.8 Multi-Token Prediction（MTP）（Gloeckle et al., 2024）

**思路**：训练时在模型主干之上加 $k$ 个预测头，在位置 $t$ **同时预测** $t+1, t+2, \dots, t+k$ 处的 token。

**推理时的用法**：
- 额外的预测头可以充当 draft，再由主预测头验证 —— **draft model 和 target model 是同一个模型**，不需要单独维护一个小模型（self-speculative decoding）
- 论文中 4-token 预测的模型推理速度最高提升约 3 倍

**训练时的好处**：多 token 目标迫使模型提前“规划”，本身也能提升模型质量，尤其在代码任务上（13B 模型在 HumanEval 上多解出 12% 的题）。DeepSeek-V3 也使用了 MTP。

### 5.9 小结

| 挑战 | 解决方法 | 原理 |
|---|---|---|
| 重复计算之前 token 的 K、V | **KV cache** | 存下来复用 |
| KV cache 太大 | **GQA**、**MLA** | 共享 KV head；压缩成低维 latent |
| KV cache 显存碎片 | **PagedAttention** | 分页、按需分配、可共享 |
| 逐 token 解码太慢 | **Speculative decoding**、**MTP** | 先便宜地猜多个，再一次验证 |

---

## 6. 要点回顾

1. LLM 是用链式法则建模 $p(x_t \mid x_{<t})$ 的超大 decoder-only Transformer
2. Sparse MoE 把 FFN 换成多个 expert，每个 token 只走 top-k 个：总参数大（显存），激活参数小（计算）；需要负载均衡防止 routing collapse
3. Greedy 和 beam search 确定但单调；sampling 多样但可能离谱；top-p 自适应地截断长尾；temperature 控制分布的尖锐程度
4. Guided decoding 通过 mask 掉不合法 token 保证输出格式
5. Few-shot、CoT、self-consistency 都是不改权重、只改 prompt 或推理方式来提升效果的手段，代价是更多 token
6. Decode 阶段 memory-bound；KV cache 避免重复计算但占大量显存，GQA / MLA 缩小它，PagedAttention 管理它
7. Speculative decoding 用小模型猜、大模型并行验证，输出分布与只用大模型完全一致

---

## 7. 自测题

<details>
<summary>1. Mixtral 8x7B 为什么总参数是 46.7B 而不是 56B？推理时显存需要放下多少参数？</summary>

只有 FFN 被复制了 8 份，attention、embedding 等部分是共享的，所以总量小于 8 × 7B。推理时显存要放下全部 46.7B 参数（因为不知道 token 会被路由到哪个 expert），但每个 token 只计算约 12.9B 激活参数。
</details>

<details>
<summary>2. 概率为 [0.6, 0.25, 0.1, 0.05]，p = 0.8 时 top-p 会保留哪些 token？</summary>

前 1 个累积 0.6 < 0.8，前 2 个累积 0.85 ≥ 0.8，所以保留前两个，重新归一化为 [0.706, 0.294] 后采样。
</details>

<details>
<summary>3. Temperature 从 1 降到 0.5，分布会怎样变化？为什么？</summary>

分布变得更尖锐。logits 除以 0.5 等于乘以 2，分数之间的差距被放大，经过指数函数后最高分 token 的概率占比更大。
</details>

<details>
<summary>4. 为什么 KV cache 可以复用之前 token 的 K、V？</summary>

在 causal mask 下，之前的 token 看不到之后的 token，它们在每一层的表示（以及 K、V）与后面生成什么无关，一旦算出就不会改变，所以可以存起来直接复用。
</details>

<details>
<summary>5. 一个 32 层、8 个 KV head、head 维度 128 的模型，用 FP16，一条 8,192 token 的序列 KV cache 多大？</summary>

每 token：2 × 32 × 8 × 128 × 2 字节 = 131,072 字节 = 128 KB。8,192 个 token：128 KB × 8,192 = 1 GB。
</details>

<details>
<summary>6. Speculative decoding 为什么能加速，又为什么不影响输出质量？</summary>

加速：decode 是 memory-bound 的，大模型一次 forward 验证 k 个 token 和生成 1 个 token 耗时差不多，只要 draft 猜得准，每次大模型 forward 就能产出多个 token。不影响质量：接受 / 拒绝规则加上从 [Q − P]+ 重新采样，可以证明最终每个 token 的分布严格等于 target model 的分布 Q。
</details>

<details>
<summary>7. Self-consistency 为什么比单次 CoT 更准？代价是什么？</summary>

正确的推理路径虽然多样，但往往得到相同答案；错误的路径答案分散。多次采样后多数投票，能过滤掉偶然的推理错误。代价是需要采样 N 次，成本和延迟大约是 N 倍。
</details>

---

## 参考论文

- Shazeer et al., “Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer”, 2017
- Fedus et al., “Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity”, 2021
- Jiang et al., “Mixtral of Experts”, 2024；DeepSeek-AI, “DeepSeek-V3 Technical Report”, 2024
- Holtzman et al., “The Curious Case of Neural Text Degeneration” (nucleus sampling), 2019
- He et al., “Defeating Nondeterminism in LLM Inference”, 2025
- Hong et al., “Context Rot: How Increasing Input Tokens Impacts LLM Performance”, 2025
- Brown et al., “Language Models are Few-Shot Learners”, 2020
- Wei et al., “Chain-of-Thought Prompting Elicits Reasoning in Large Language Models”, 2022；Kojima et al., “Large Language Models are Zero-Shot Reasoners”, 2022
- Wang et al., “Self-Consistency Improves Chain of Thought Reasoning in Language Models”, 2022
- Kwon et al., “Efficient Memory Management for Large Language Model Serving with PagedAttention”, 2023
- DeepSeek-AI, “DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model”, 2024
- Chen et al., “Accelerating Large Language Model Decoding with Speculative Sampling”, 2023；Leviathan et al., “Fast Inference from Transformers via Speculative Decoding”, 2022
- Gloeckle et al., “Better & Faster Large Language Models via Multi-token Prediction”, 2024
