# Lecture 9 · LLM Serving Basics I：Introduction to LLM

> Harvard CS 2680：Modern AI Systems: Agents and System Optimizations（Juncheng Yang，Fall 2026）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/9_llm_serving_basics_1.pdf`](../slides/9_llm_serving_basics_1.pdf)
> 预计阅读时间：18–22 分钟。预备知识：线性代数（矩阵乘法）。本讲假设传统的 self-attention Transformer 架构。

## 本讲脉络

```
一个 LLM 请求的生命周期：Tokenize → Schedule → Prefill → Decode（+ Sampling）→ Detokenize → Stream
→ 服务端关心的指标：TTFT、每 token 延迟、吞吐量
→ 模型内部：符号约定 → 一叠相同的 Transformer block → RMSNorm 与残差 → self-attention（Q/K/V 的形状、O(n²)）
  → multi-head → GQA / MQA 与 KV cache 大小 → 以 Llama 3 70B 为例算参数、计算、显存
→ Dense vs MoE：显存看总参数，计算看激活参数
→（进阶）其他架构：sliding window、MLA、linear attention、state-space model、sparse attention、looped transformer、diffusion LM
```

**一句话总结**：一个请求在服务端经历六个阶段，其中 **prefill 一次处理整个 prompt，是计算密集的，决定首 token 延迟（TTFT）**；**decode 每生成一个 token 都要把全部权重读一遍，是内存带宽密集的，决定每秒生成多少 token**。模型内部是几十层相同的 block，每层的 attention 让 token 互相交流、FFN 单独加工每个 token。服务系统要算清三本账：**权重多大、每个 token 多少计算、KV cache 每 token 多大** —— 后面所有的 serving 优化都围绕这三本账展开。

（本讲的目的是建立直觉，不深入所有细节。）

---

## 1. 一个 LLM 请求的生命周期

```
Tokenize → Schedule → Prefill → Decode ⟲（每个输出 token 重复一次）→ Detokenize → Stream
                          │            │
                  首 token 延迟（TTFT）   每秒生成的 token 数
             整个 prompt 一次 forward    每个输出 token 一次 forward
```

### 1.1 Stage 1：Tokenization（分词）

- 文本被切成 **token**，而不是单词。例：“Serving is hard” → `Serv` `ing` `is` `hard` —— **4 个 token，不是 3 个词**
- 英文中平均约 **3–4 个字符一个 token**
- **token 数决定成本和内存**（API 按 token 计费，KV cache 按 token 增长）
- 不同模型的 tokenizer 不同，同一段文本的 token 数也不同

**Byte-level BPE**：从 256 个字节值开始，**反复把最常出现的相邻对合并成一个新 token**，直到词表达到目标大小。因为从字节开始，任何文本都能表示，**不存在 unknown token**。

**词表大小的权衡**：
- 词表越大 → 同样的文本切出的 token 越少 → 序列更短、更快
- 但 embedding 表（和输出层）越大：Llama 3 的词表有 128,256 个 token，embedding 和输出层合计约 21 亿参数

### 1.2 Stage 2：Scheduling and batching（调度与批处理）

- 请求**各自独立地**到达，而 GPU 需要**成批**处理才能高效
- **Scheduler（调度器）**决定每一步由哪些请求参与计算

**静态批处理的问题**：把一批请求凑齐一起跑，必须等**最长的那个**生成完，整批才结束；短请求早早结束后，它的位置就空着（或者填充无用的 padding）。

**Continuous batching（连续批处理）**：**每一步**都可以让新请求加入、让完成的请求离开，不需要等整批结束。

```
step t      step t+1    step t+2
R1          R1          R1
R2          free        R5      ← R2 完成后离开，R5 立刻补上
R3          R3          R3
free        R4          R4      ← R4 在中途加入
```

- **准入受 KV cache 显存的限制**：每个正在运行的请求都要占用 KV cache，显存满了就不能再接新请求

### 1.3 Stage 3：Prefill（预填充）

```
prompt tokens：t1 t2 t3 t4 t5 t6  →  一次 forward pass 处理所有 token  →  第一个输出 token
```

- 整个 prompt 在**一次 forward pass** 中处理，所有 prompt token **并行**计算 attention
- **Compute-bound（计算受限）**：成本随 prompt 长度增长
- 生成 **KV cache**（所有 prompt token 的 K、V），并输出第一个 token
- 决定 **TTFT（Time To First Token，首 token 延迟）**

### 1.4 Stage 4：Decode（解码）与 Stage 5：Sampling（采样）

**Decode**：
- **每个新 token 一次 forward pass**
- KV cache 每一步增长一个 token
- **Bandwidth-bound（带宽受限）**：每一步都要把全部模型权重从显存读一遍，却只为一个 token 做计算 → 所以要**把很多请求凑成 batch**，让一次读取的权重同时服务多个请求
- KV cache 满了 → 抢占（preempt）某些请求，或者让新请求排队

**Sampling（采样）**：decode 得到的是词表上每个 token 的分数（logits），采样从中挑一个：

| 方法 | 设置示例 | 含义 |
|---|---|---|
| **Temperature** | temperature = 0 | 0：总是选最高分的 token；> 1：分布更平，更多 token 有机会 |
| **Top-k** | top_k = 50 | 只在概率最高的 k 个 token 中采样；k 固定 |
| **Top-p（nucleus）** | top_p = 0.9 | 在累积概率达到 p 的最少 token 中采样；随模型的确定程度自适应 |

**同一个模型、同一个 prompt，这些设置会改变答案，也会改变运行时长**（生成多少 token）。

### 1.5 Stage 6：Detokenize and stream（反分词与流式输出）

```
t1：The
t2：The model
t3：The model streams
t4：The model streams tokens        ← 用户看到的文字逐个出现
```

- Token ID 一生成就转回文本
- **不完整的字符要先缓冲**：byte-level BPE 中，一个汉字或 emoji 可能被拆成多个 token，只有凑齐所有字节后才能显示，否则会出现乱码
- Streaming 让交互感更好（用户不必等全部生成完）
- **提前停止能释放 cache 内存**
- 结束条件：生成 stop token、达到长度上限，或客户端断开连接

### 1.6 服务端关心的指标

| 指标 | 含义 | 主要由什么决定 |
|---|---|---|
| **TTFT**（Time To First Token） | 从请求到达到第一个 token 输出 | 排队 + prefill |
| **每 token 延迟**（TPOT / inter-token latency） | 相邻两个输出 token 之间的时间 | decode 的一步 |
| **Throughput（吞吐量）** | 整个系统每秒生成的 token 总数 | batch 大小、显存 |

延迟和吞吐量之间有权衡：batch 越大，吞吐量越高，但每个请求的每 token 延迟也会变长。

### 1.7 小结

1. **Tokenize**：把文本变成 token ID；词表越大 token 越少，但 embedding 表越大
2. **Schedule**：continuous batching 让请求每一步都可以加入和离开；KV cache 显存限制准入
3. **Prefill**：整个 prompt 一次 compute-bound 的 forward，生成 KV cache 和第一个 token
4. **Decode and sample**：每个 token 一次 memory-bound 的 forward；采样从 logits 中挑出每个 token
5. **Detokenize and stream**：把每个 token ID 转成文本并立即发送

---

## 2. 窥探模型内部

### 2.1 符号约定（以 Llama 3 70B 为例）

| 符号 | 含义 | Llama 3 70B |
|---|---|---|
| $d$ | 模型宽度（每个 token 向量的维度） | 8,192 |
| $L$ | 层数 | 80 |
| $H$ | query head 数 | 64 |
| $H_{kv}$ | key / value head 数 | 8 |
| $d_k$ | 每个 head 的维度 $= d / H$ | 128 |
| $d_{ff}$ | MLP 隐藏层宽度 | 28,672 |
| $V$ | 词表大小 | 128,256 |
| $n$ | 序列中的 token 数 | 如 20,000 |
| $B$ | 并发请求数 | 如 16 |
| $N$ | 参数量 | 70.6B |
| $W$ | 权重字节数（BF16） | 141 GB |
| $KV$ | 每个 token 的 KV cache 字节数 | 320 KB |

KV cache 的公式（2 表示 K 和 V 各一份，最后的 2 B 是 BF16 每个数 2 字节）：

$$
KV = 2 \times L \times H_{kv} \times d_k \times 2\,\text{B}
$$

（$K^\top$ 表示 $K$ 的转置。）

### 2.2 LLM 是一叠相同的 Transformer block

```
token IDs           输入：文本以整数 token ID 的形式进入
   ↓
Embedding           每个 ID 变成一个 8,192 维的向量
   ↓
Transformer block × L   Llama 3 70B 有 L = 80 个 block，每个都有 norm + 残差
   ├─ Self-attention     在 token 之间混合信息
   └─ Feed-forward (FFN) 对每个 token 单独变换
   ↓
LM head             给词表中的每个 token 一个分数
   ↓
Softmax             分数变成下一个 token 的概率
```

### 2.3 每个 block：RMSNorm + attention + MLP，都加到残差流上

$$
h = x + \text{Attn}\big(\text{RMSNorm}(x)\big),\qquad
y = h + \text{FFN}\big(\text{RMSNorm}(h)\big)
$$

$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_i x_i^2 + \varepsilon}} \odot g
$$

| 概念 | 含义 |
|---|---|
| **RMSNorm** | 按均方根（root mean square）缩放；每个维度有一个可学习的增益 $g$ |
| **Pre-norm** | 对每个 sublayer 的**输入**做归一化，让很深的堆叠也能稳定训练 |
| **Residual stream（残差流）** | $x$ 贯穿全部 80 个 block，每个 sublayer 只在上面**加一个更新量** |

可以把残差流想象成一条传送带：每一层从传送带上读取信息，处理后把“修改意见”加回传送带。

### 2.4 Self-attention：每个 token 读取它之前的所有 token

- 每个 token 生成一个 **query**、一个 **key** 和一个 **value**
- query 与 key 的点积分数，决定从每个 token 读取多少信息
- **Causal mask（因果掩码）** 隐藏未来的 token（生成时不能偷看后面）
- $n$ 个 token → $n^2$ 个分数：**成本随 context 增长**

$$
\text{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}} + M\Big) V
$$

例：句子 “The cat sat on the” 中，“on” 这一行的权重可能是 The 0.1、cat 0.2、sat 0.5、on 0.2（和为 1），而它之后的位置被 mask 掉。

**为什么除以 $\sqrt{d_k}$**：$q \cdot k$ 是 $d_k$ 个乘积之和，数值随 $d_k$ 增大；缩放后 softmax 不会饱和（饱和时梯度几乎为 0）。

### 2.5 从 X 到 Q、K、V：看清楚矩阵的形状

$$
Q = X W_Q\ (\text{每个 token 在找什么}),\quad
K = X W_K\ (\text{每个 token 能提供什么}),\quad
V = X W_V\ (\text{每个 token 传递什么})
$$

| 矩阵 | 形状 | 说明 |
|---|---|---|
| $X$ | $n \times d$ | 每一行是一个 token（如 The、cat、sat、on、the） |
| $W_Q, W_K, W_V$ | $d \times d_k$（每个 head） | 学到的权重 |
| $Q, K, V$ | $n \times d_k$ | 一个 head 中的 query、key、value |

### 2.6 $QK^\top$ 给每一对 token 打分，softmax 把每一行变成权重

| 步骤 | 计算 | 形状 |
|---|---|---|
| 输入 | $X$ | $n \times d$ |
| Queries | $Q = X W_Q$ | $n \times d_k$ |
| Scores | $S = QK^\top / \sqrt{d_k}$ | $n \times n$：第 $i$ 行第 $j$ 列 = query $i$ · key $j$ |
| Weights | $A = \text{softmax}(S + M)$ | $n \times n$：$M$ 在对角线以上为 $-\infty$；每行和为 1 |
| Output | $O = A V$ | $n \times d_v$：每一行是 V 各行的加权组合 |

例：$A$ 的第 4 行表示 token 4 从 token 1–4 各取多少信息，这一行的和为 1。

### 2.7 Attention 的 $O(n^2)$ 复杂度

**每层的 FLOPs**：$QK^\top$ 约 $2n^2 d$，$AV$ 约 $2n^2 d$，合计约 $4 n^2 d$。

**如果把分数矩阵 $S$ 完整存下来**（$n = 8{,}192$、64 个 head、80 层、BF16）：

$$
n^2 \times \text{heads} \times \text{layers} \times 2\,\text{B} = 8192^2 \times 64 \times 80 \times 2\,\text{B} \approx 690\ \text{GB}
$$

根本存不下。**FlashAttention** 分块（tile by tile）计算 softmax，**从不存储完整的 $S$**，所以可以处理长序列。

### 2.8 Multi-head attention：64 个小 attention 并行，再用 $W_O$ 混合

$$
\text{head}_i = \text{softmax}\Big(\frac{Q_i K_i^\top}{\sqrt{d_k}} + M\Big) V_i,\qquad
\text{MHA}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_H)\, W_O
$$

| 步骤 | 说明 |
|---|---|
| **Split（拆分）** | Q、K、V 拆成 $H = 64$ 个 head，每个 $d_k = 8192 / 64 = 128$ 维 |
| **Attend（各自注意）** | 每个 head 在自己的 128 维上独立做带 mask 的 softmax attention |
| **Mix（混合）** | $W_O$ 把各 head 的输出融合成一个向量 |

**FLOPs 与一个宽 head 相同**（$64 \times 128 = 8192$）：拆分改变的是**学到什么**，而不是成本。

**每个 head 用自己的 128 维，学会不同的模式**：例如 head 1 关注前一个 token，head 2 关注第一个 token，head 3 关注附近的 token，head 64 关注整个 context。

```
X（n × 8,192）→ 64 个 head 各自投影成 Q、K、V（各 128 维）→ 各自 attention → 64 个输出（n × 128）
→ Concat（n × 8,192）→ × W_O → n × 8,192
```

### 2.9 共享 key/value head 可以缩小 KV cache

| | Multi-head（MHA） | Grouped-query（GQA） | Multi-query（MQA） |
|---|---|---|---|
| 8 个 query head 对应 | 8 个 KV head | 2 个 KV head | 1 个 KV head |
| KV cache 大小 | 1× | ¼× | ⅛× |

**Llama 3 70B**：64 个 query head 共享 8 个 KV head，所以 KV cache 是 MHA 的 1/8：

$$
2 \times 80 \times 8 \times 128 \times 2\,\text{B} = 327{,}680\,\text{B} \approx 320\ \text{KB / token}\quad(\text{MHA 则约 } 2.5\ \text{MB / token})
$$

### 2.10 Llama 3 70B 的结构一览

| 参数 | 值 | 说明 |
|---|---|---|
| $d$ | 8,192 | 一个 token = 8,192 个数 |
| $h$、$d_h$ | 64、128 | 64 个 Q head × 128 = 8,192 |
| $h_{kv}$ | 8 | 共享 8 个 KV head |
| $d_{ff}$ | 28,672 | FFN 隐藏宽度，3.5 × $d$ |
| $L$ | 80 | 80 个 transformer block 依次排列 |
| $V$ | 128,256 | 词表中的 token 数，最后输出 128,256 个分数 |

### 2.11 参数都在哪里

| 组件 | 公式 | 参数量 |
|---|---|---|
| Attention / 层 | $2d \cdot H \cdot d_k + 2d \cdot H_{kv} \cdot d_k$（$W_Q$、$W_O$ 各 $d \times d$；$W_K$、$W_V$ 各 $d \times H_{kv} d_k$） | 151M |
| FFN / 层 | $3 \cdot d \cdot d_{ff}$（SwiGLU 有 gate、up、down 三个矩阵） | 705M |
| 所有层 | $80 \times 856\text{M}$ | 68.5B |
| Embedding + LM head | $2 \cdot V \cdot d$ | 2.1B |
| **总计** | | **70.6B** |

**观察**：FFN 占了每层参数的 **80% 以上**；GQA 让 attention 的参数也变少了（$W_K$、$W_V$ 只有 $W_Q$ 的 1/8）。

### 2.12 系统直觉：三本账

| 账 | 公式 | Llama 3 70B（BF16） | 决定什么 |
|---|---|---|---|
| **Weights（权重）** | 2 B × 参数量 | **141 GB** | 至少需要几张卡；decode 每步要读多少字节 |
| **Compute（计算）** | ≈ 2 × 参数量 FLOPs / token | **141 GFLOPs / token** | prefill 要多久 |
| **KV cache** | $2 \times L \times H_{kv} \times d_k \times 2$ B | **320 KB / token** | 能同时服务多少请求、多长 context |

**算例**：用 2 张 H200（合计 1,978 TFLOP/s、9.6 TB/s、282 GB 显存），处理 $n = 80{,}000$ 个 token 的 prompt，$B$ 为请求数：

| 量 | 公式 | 1 个请求 | 64 个请求 |
|---|---|---|---|
| Prefill（80K tokens） | $B \cdot (2N n + 2 L d n^2) \div \text{FLOP/s}$ | **9.95 s** | **637 s** |
| 需要的显存 | $W + B \cdot n \cdot KV$ | **167 GB** | **1,779 GB** |

拆开看 prefill：$2Nn = 2 \times 70.6\text{B} \times 80\text{K} \approx 1.13 \times 10^{16}$（矩阵乘法）；$2Ld n^2 = 2 \times 80 \times 8192 \times 80\text{K}^2 \approx 8.4 \times 10^{15}$（attention 的 $n^2$ 项，长 context 时已经占了相当比例）；合计约 $1.97 \times 10^{16}$ FLOPs，除以 $1.978 \times 10^{15}$ FLOP/s 约 10 秒。

显存：一个 80K 的请求 KV cache 约 26 GB，单个请求还放得下；64 个请求需要约 1.8 TB，远超 282 GB。

**问题**：如何支持更长的序列、更多的并发？—— 这就是后续 serving 优化（PagedAttention、量化、KV cache 压缩、更高效的 attention 结构……）要回答的问题。

---

## 3. Dense vs MoE

### 3.1 MoE 把每个 block 的 FFN 换成一个 router 和一组小的 expert FFN

```
Dense block：                MoE block：
Attention                    Attention
   ↓                            ↓
FFN（一个大 MLP，             Router 选出 top-2
每个 token 全部都要跑）          ↓
                             E0 E1 E2 E3 E4 E5 E6 E7（8 个小 FFN，每个 token 只跑 2 个）
                                ↓
                             加权求和
```

### 3.2 显存随总参数增长，计算随激活参数增长

| 模型 | Experts | 总参数 | 每 token 激活 | 权重（BF16） | GFLOPs / token |
|---|---|---|---|---|---|
| Llama 2 70B | 全部（dense） | 69B | 69B | 138 GB | 138 |
| Mixtral 8x7B | 8 选 2 | 46.7B | 12.9B | 93 GB | 26 |
| Llama 3.1 405B | 全部（dense） | 405B | 405B | 810 GB | 810 |
| DeepSeek-V3 | 257 选 9 | 671B | 37B | 1,342 GB | 74 |

（GFLOPs ≈ 2 × 激活参数。）

**同样的质量，更少的计算**：
- Mixtral 与 Llama 2 70B 效果相当，计算量只有约 **1/5**
- DeepSeek-V3 可以和 Llama 3.1 405B 比肩，计算量只有约 **1/11**，但需要约 **1.7 倍**的显存

对 serving 来说，MoE 意味着：**显存压力大（所有 expert 都要放下），但每个 token 的计算少**。

---

## 4. 进阶：其他 LLM 架构

### 4.1 Sliding window：局部层限制 KV cache，少数全局层保持长程能力

| 方式 | 说明 |
|---|---|
| Full causal | 最后一个 token 读所有之前的 token，KV 随 $n$ 增长 |
| Sliding window（$w = 3$） | 最后一个 token 只读最近 3 个，KV 上限为 $w$ 个 token |

| 模型 | 层的配置 |
|---|---|
| Mistral 7B | 每层都是局部层，$w = 4{,}096$ |
| Gemma 2 | 1 个局部层（$w = 4{,}096$）: 1 个全局层，交替 |
| Gemma 3 | 5 个局部层（$w = 1{,}024$）: 1 个全局层，重复 |

**Serving 效果**：128K context 时，Gemma 3 的 KV cache 约为全部用全局层时的 1/6。

### 4.2 MLA（Multi-head Latent Attention）：每个 token 只存一个压缩的 latent

```
token h → 压缩 → latent c（缓存它）→ 需要时展开成 K、V
```

- **存摘要，重建细节**：attention 需要时，从 $c$ 重新计算出 K 和 V
- 普通 MHA 要为每个 head 缓存 K 和 V；MLA 每个 token 只缓存一个 latent $c$
- **额外的计算，更少的内存**：decode 时，上投影矩阵可以合并进 query 和输出的权重中，几乎不增加计算
- **Serving 效果**：与 DeepSeek 67B 相比，DeepSeek-V2 的 KV cache 减少 93%，生成吞吐提高 5.8 倍

### 4.3 Linear attention：用固定大小的状态代替不断增长的 KV cache

| Softmax attention | Linear attention |
|---|---|
| KV cache：每个 token 永远多一条 | 状态 $S_t$：$d_k \times d_v$ 的矩阵，**每一步大小不变**，原地更新 |

**Plain linear attention**：每个 token 把 “value × key” 加到状态上；query 从状态中读取：

$$
S_t = S_{t-1} + v_t k_t^\top,\qquad o_t = S_t\, q_t
$$

**Gated DeltaNet**：先让旧状态**衰减**，再**覆写**新 key 所指向的内容：

$$
S_t = \alpha_t\, S_{t-1}\big(I - \beta_t\, k_t k_t^\top\big) + \beta_t\, v_t k_t^\top
$$

- $\alpha_t$：**decay gate（衰减门）**，忘掉多少
- $\beta_t$：**write gate（写入门）**，覆写得多用力

**Qwen3-Next**：48 层中 36 层是 Gated DeltaNet、12 层是普通 attention（3:1），只有那 12 层需要 KV cache。Kimi Linear、Qwen3.5 也采用类似的 3:1 配方。

**Serving 效果**：四分之三的层在任何 context 长度下都只需要固定大小的状态。

### 4.4 State-space models（如 Mamba）：同样保持固定大小的状态

| | Gated linear attention | State-space（Mamba） |
|---|---|---|
| 更新 | $S \leftarrow \alpha S + \beta\, v k^\top$ | $h \leftarrow \bar A h + \bar B x$ |
| 状态 | 一个 $d_k \times d_v$ 的 key-value 矩阵 | 每个通道一个小向量 $h$ |
| 每个 token | 把 value × key 加到 $S$ 上，按 $\beta$ 缩放 | 先按 $\bar A$ 衰减 $h$，再加上 $\bar B \times$ token $x$ |
| 遗忘 | 门 $\alpha$：由 token 决定 $S$ 衰减多少 | $\bar A$：由 token 决定 $h$ 衰减多少 |
| 起源 | 去掉 softmax 的 attention | 信号处理与 RNN |

- **Selective（选择性）**：$\bar A$ 和 $\bar B$ 取决于每个 token，所以这一层可以选择保留什么、忘掉什么
- **Fixed state（固定状态）**：decode 时内存不随 $n$ 增长
- **Hardware-aware scan（硬件感知的扫描）**：用并行扫描（parallel scan）让状态留在 SRAM 中；Mamba-2 把它改写成矩阵乘法
- **Serving 效果**：没有不断增长的 KV cache；Mamba 报告生成吞吐最高可达 Transformer 的 5 倍

### 4.5 Sparse attention：每个 query 只读最相关的 2,048 个历史 token

（DeepSeek-V3.2）

1. **打分**：一个轻量的 **indexer** 为每个历史 token $s$ 打分：$I_{t,s} = \sum_j w_{t,j} \cdot \text{ReLU}(q_{t,j} \cdot k_s)$
2. **选择**：$S_t$ = 分数最高的 top-$k$ 个 token（$k = 2{,}048$）
3. **注意**：只对选中的 token 做 attention：$o_t = \text{Attn}(q_t, \{c_s : s \in S_t\})$

- Indexer：只有很少的 head，用 FP8，每个 token 一个 128 维的 key —— 很便宜
- 主 attention：只读 2,048 个 token，而不是全部 $n$ 个

**Serving 效果**：每个 token 的工作量上限是 2,048 次读取，**但所有 token 的 KV 仍然要存着**（因为不知道哪些会被选中）。

### 4.6 Looped transformers：一个 block 重复使用很多次

```
Input → Prelude P → Block R × K（同一组权重循环 K 次）→ Coda C → Output
```

- **共享权重**：同一个 block $R$ 被应用 $K$ 次，增加深度**不增加参数**
- **测试时调深度**：一个 3.5B 的 recurrent-depth 模型，推理任务上 $K$ 越大效果越好
- **系统上的权衡**：权重很小，但每次循环都是**串行**的，延迟随 $K$ 增长

**不增加参数的计算**：$K$ 次循环花 $K$ 倍的 block 计算量，但不增加任何权重内存。

### 4.7 Diffusion LMs：在几个并行的步骤中逐步填出被 mask 的 token

```
Step 0：[M]  [M]  [M]    [M]   [M]   [M]
Step 1：[M]  read [M]    weight [M]  [M]
Step 2：GPUs read [M]    weight [M]  step
Step 3：GPUs read every  weight each step
```

- **LLaDA（2025）**：8B 的 masked diffusion 模型，从零在 2.3T tokens 上训练，与 Llama 3 8B 有竞争力
- **Dream 7B（2025）**：从 Qwen2.5 7B 的权重出发；网络本身仍是 Transformer
- **Serving 效果**：一次 forward 能填出多个 token，但步数越少，质量越差

（其他方向还有 multi-token prediction 等。）

---

## 5. 要点回顾

1. 请求生命周期：tokenize → schedule → prefill → decode + sample → detokenize → stream
2. Prefill 一次并行处理整个 prompt，compute-bound，决定 TTFT；decode 每 token 一次 forward，memory-bound，决定 tokens/s
3. Continuous batching 让请求每步进出；KV cache 显存限制并发
4. 模型 = embedding + L × (RMSNorm → attention → 残差，RMSNorm → FFN → 残差) + LM head + softmax
5. Attention 的分数矩阵是 $n \times n$，每层约 $4n^2 d$ FLOPs；FlashAttention 不存完整的分数矩阵
6. Multi-head 不增加 FLOPs；GQA / MQA 共享 KV head，KV cache 成比例缩小
7. 三本账（Llama 3 70B）：权重 141 GB、计算 141 GFLOPs/token、KV 320 KB/token；长 context 和高并发时 KV cache 成为瓶颈
8. MoE：显存看总参数、计算看激活参数
9. Sliding window、MLA、linear attention、SSM、sparse attention 都在缩小或限制 KV；looped transformer 用计算换参数；diffusion LM 用并行步骤换 forward 次数

---

## 6. 自测题

<details>
<summary>1. 为什么 prefill 是 compute-bound，而 decode 是 memory-bound？</summary>

Prefill 一次处理整个 prompt，读一遍权重就能为成千上万个 token 做计算，计算量大、每字节权重被复用很多次。Decode 每步只为一个 token（或一个 batch 中每个请求一个 token）计算，却要把全部权重读一遍，读取的字节数远多于计算量，瓶颈在显存带宽。
</details>

<details>
<summary>2. Continuous batching 比静态批处理好在哪里？</summary>

静态批处理必须等最长的请求完成，短请求结束后的位置浪费；continuous batching 每一步都可以让完成的请求离开、新请求加入，GPU 一直满载，吞吐更高、新请求的等待时间更短。
</details>

<details>
<summary>3. 一个 32 层、32 个 query head、8 个 KV head、head 维度 128 的模型，BF16 下每个 token 的 KV cache 多大？</summary>

2 × 32 × 8 × 128 × 2 B = 131,072 B = 128 KB。
</details>

<details>
<summary>4. 为什么 byte-level BPE 在流式输出时需要缓冲？</summary>

一个汉字或 emoji 的 UTF-8 编码有多个字节，可能被拆到多个 token 中。只输出了部分字节时无法显示成完整字符，需要缓冲到字节凑齐后再输出，否则会出现乱码。
</details>

<details>
<summary>5. Llama 3 70B 中 FFN 与 attention 每层各有多少参数？为什么 FFN 多这么多？</summary>

FFN 约 705M，attention 约 151M。FFN 有三个 d × d_ff 的矩阵（d_ff = 3.5d）；attention 的 W_Q、W_O 是 d × d，而由于 GQA，W_K、W_V 只有 d × (8 × 128)，所以 attention 参数少得多。
</details>

<details>
<summary>6. DeepSeek-V3 的计算量远小于 Llama 3.1 405B，为什么显存需求反而更大？</summary>

MoE 每个 token 只激活约 37B 参数，所以计算少；但所有 671B 参数（全部 expert）都必须放在显存中随时待用，所以权重显存（约 1.3 TB）比 405B 的 dense 模型（约 810 GB）更大。
</details>

<details>
<summary>7. Sparse attention（如 DeepSeek-V3.2）为什么减少了计算，却没有减少 KV cache？</summary>

每个 query 只对 indexer 选出的 2,048 个 token 做 attention，计算有上限；但哪些 token 会被选中事先不知道，而且每一步都可能不同，所以所有历史 token 的 KV 都必须保留。
</details>
