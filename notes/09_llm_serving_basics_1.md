# Lecture 9 · LLM Serving Basics I：Introduction to LLM

> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/9_llm_serving_basics_1.pdf`](../slides/9_llm_serving_basics_1.pdf)

## 本讲要点

- **LLM request 的生命周期**：Tokenize → Schedule → Prefill → Decode（+ Sampling）→ Detokenize → Stream
- **Prefill 是 compute-bound**（决定 TTFT），**Decode 是 memory-bandwidth-bound**（决定 tokens/s）
- **Transformer 架构**：Embedding → L × (Self-attention + FFN) → LM head；RMSNorm、residual stream
- Attention 的 O(n²) 复杂度；**KV cache** 大小公式；MHA / GQA / MQA
- 以 Llama 3 70B 为例估算参数、计算量、显存
- **Dense vs MoE**；进阶架构：Sliding window、MLA、Linear attention、SSM、Sparse attention、Looped transformer、Diffusion LM

> 本讲假设传统的 self-attention transformer 架构。

---

## 1. LLM Request 的生命周期

```
Tokenize → Schedule → Prefill → Decode ⟲（每个输出 token 一次）→ Detokenize → Stream
                       │           │
              time to first token   tokens per second
              整个 prompt 一次 forward   每个输出 token 一次 forward
```

### Stage 1：Tokenization

- 文本被切成 tokens 而不是单词（“Serving is hard” → `Serv` `ing` `is` `hard`，4 个 tokens）
- 约 3–4 个字符 / token；token 数决定成本和内存；不同模型的 tokenizer 不同
- **Byte-level BPE**：从 256 个字节值开始，反复合并最频繁的相邻对作为新 token；没有 unknown token
- 词表越大 → token 越少，但 embedding table 越大

### Stage 2：Scheduling 与 Batching

- 请求独立到达，而 GPU 需要 batch
- Scheduler 决定每一步谁运行
- **Continuous batching**：请求可以在中途加入或离开，无需等待整个 batch 完成
- 准入数量受 **KV cache 显存**限制

### Stage 3：Prefill

- 整个 prompt 在一次 forward pass 中处理，所有 prompt tokens 并行 attend
- **Compute-bound**，成本随长度增长
- 生成 KV cache，并输出第一个 token → 决定 **TTFT（time to first token）**

### Stage 4：Decode + Sampling

- 每个新 token 一次 forward pass；KV cache 每步增长一个 token
- **Bandwidth-bound**，所以要 batch；cache 满了就 preempt 或排队
- Sampling：
  - **Temperature**：0 总是选最高概率 token；>1 让更多 token 有机会
  - **Top-k**：从概率最高的 k 个中采样（k 固定，如 50）
  - **Top-p（nucleus）**：概率和达到 p 的最少 token 集合（如 0.9），随置信度自适应
- 同一模型、同一 prompt，这些设置会改变答案以及运行时长

### Stage 5：Detokenize 与 Stream

- Token ID 产生即转为文本；不完整的字符会被缓冲
- Streaming 让交互感更好；提前停止可释放 cache 内存
- 结束条件：stop token、长度上限或断开连接

---

## 2. Transformer 架构

### 符号（以 Llama 3 70B 为例）

| 符号 | 含义 | 值 |
|---|---|---|
| d | model width | 8,192 |
| L | layers | 80 |
| H | query heads | 64 |
| H_kv | key/value heads | 8 |
| d_k | head dimension = d ÷ H | 128 |
| d_ff | MLP hidden width | 28,672 |
| V | vocabulary size | 128,256 |
| N | parameters | 70.6B |
| W | weight bytes (BF16) | 141 GB |
| KV | 每 token 的 KV 字节 | 320 KB |

### 整体结构

token IDs → **Embedding**（每个 ID 变成 8,192 维向量）→ **Transformer block × L**（Self-attention 在 tokens 之间混合信息；FFN 对每个 token 独立变换）→ **LM head**（每个词表 token 一个分数）→ **Softmax** → 下一个 token 的概率

### Block 内部：Pre-norm + Residual

```
h = x + Attn(RMSNorm(x))
y = h + FFN(RMSNorm(h))
RMSNorm(x) = x / sqrt(mean(x²) + ε) ⊙ g
```

- **RMSNorm**：按 root-mean-square 缩放，每个维度一个可学习 gain g
- **Pre-norm**：对每个 sublayer 的输入做归一化，使深层堆叠训练稳定
- **Residual stream**：x 贯穿所有 80 个 blocks，每个 sublayer 只加一个更新

### Self-Attention

- 每个 token 产生 query、key、value：`Q = X·W_Q`（在找什么）、`K = X·W_K`（能提供什么）、`V = X·W_V`（传递什么）
- `Attention = softmax(Q·Kᵀ / √d_k + M) · V`
- **Causal mask M**：对角线以上为 −∞，隐藏未来 tokens
- softmax 后每行和为 1，表示该 token 从之前各 token 读取多少
- **为什么除以 √d_k**：q·k 是 d_k 个乘积之和，会随 d_k 增大，缩放防止 softmax 饱和

### O(n²) 复杂度

| 步骤 | 形状 |
|---|---|
| X | n × d |
| Q = X W_Q | n × d_k |
| S = Q Kᵀ / √d_k | **n × n** |
| A = softmax(S + M) | n × n |
| O = A V | n × d_v |

- 每层 FLOPs：Q·Kᵀ 约 2n²d，A·V 约 2n²d，总计 ≈ **4n²d**
- 如果存储 S：n² × heads × layers × 2B = 8,192² × 64 × 80 × 2B ≈ **690 GB**
- **FlashAttention** 分块计算 softmax，从不存储 S

### Multi-Head Attention

```
head_i = softmax(Q_i K_iᵀ / √d_k + M) V_i
MHA(X) = Concat(head_1, …, head_H) W_O
```

- Q、K、V 拆成 H = 64 个 head，每个 d_k = 128
- 每个 head 学不同的模式（前一个 token、第一个 token、附近 tokens、整个 context…）
- W_O 把 heads 混合回一个向量
- FLOPs 与一个宽 head 相同（64 × 128 = 8,192）：**拆分改变学到的东西，不改变成本**

### 共享 KV heads：MHA / GQA / MQA

| | MHA | GQA | MQA |
|---|---|---|---|
| 8 个 query heads 对应 | 8 个 KV heads | 2 个 KV heads | 1 个 KV head |
| KV cache 大小 | 1× | ¼× | ⅛× |

**KV cache per token = 2 × L × H_kv × d_k × bytes**

Llama 3 70B：64 个 query heads 共享 8 个 KV heads → 2 × 80 × 8 × 128 × 2B = **320 KB / token**（MHA 则是 2.5 MB），小 8 倍。

### Llama 3 70B 参数分布

| 组件 | 公式 | 参数量 |
|---|---|---|
| Attention / layer | 2d·H·d_k + 2d·H_kv·d_k | 151M |
| FFN / layer | 3·d·d_ff（SwiGLU FFN 有 gate、up、down 三个矩阵） | 705M |
| 所有层 | 80 × 856M | 68.5B |
| Embedding + LM head | 2·V·d | 2.1B |
| **总计** | | **70.6B** |

### 系统直觉：Weights、Compute、KV cache

| | 公式 | Llama 3 70B (BF16) |
|---|---|---|
| Weights | 2B × params | 141 GB |
| Compute | ≈ 2 × params FLOPs/token | 141 GFLOPs/token |
| KV cache | 2 × L × H_kv × d_k × 2B | 320 KB/token |

在 2× H200 上（1,978 TFLOP/s、9.6 TB/s、282 GB），n = 80,000 tokens：

| 量 | 公式 | 1 个请求 | 64 个请求 |
|---|---|---|---|
| Prefill（80K tokens） | B·(2N·n + 2L·d·n²) ÷ FLOP/s | 9.95 s | 637 s |
| 显存需求 | W + B·n·KV | 167 GB | 1,779 GB |

结论：单个 80K 请求还能放下，但 64 个并发请求需要约 1.8 TB 显存，远超两张 H200 —— **KV cache 才是长 context 和高并发的主要瓶颈**，这引出了后面的 GQA、MLA、sliding window、paging 等优化。

---

## 3. Dense vs MoE

- **MoE（Mixture of Experts）**：把每个 block 的 FFN 换成一个 **router** + 多个小的 expert FFN；例如 8 个 experts 中每个 token 只运行 top-2，结果加权求和
- **显存与总参数成正比，计算与激活参数成正比**

| 模型 | Experts | 总参数 | 每 token 激活 | Weights (BF16) |
|---|---|---|---|---|
| Llama 2 70B | dense | 69B | 69B | 138 GB |
| Mixtral 8x7B | 2 of 8 | 46.7B | 12.9B | 93 GB |
| Llama 3.1 405B | dense | 405B | 405B | 810 GB |
| DeepSeek-V3 | 9 of 257 | 671B | 37B | 1,342 GB |

- Mixtral 以约 1/5 FLOPs 达到 Llama 2 70B 的水平；DeepSeek-V3 以约 1/11 FLOPs 比肩 Llama 3.1 405B，但需要 1.7 倍显存
- GFLOPs ≈ 2 × active params

---

## 4. Advanced：进阶 LLM 架构（选读）

| 架构 | 思路 | Serving 影响 |
|---|---|---|
| **Sliding window** | 局部层只看最近 w 个 tokens，少数全局层保持长程 recall。Mistral 7B 全部局部（w=4,096）；Gemma 2 局部:全局 = 1:1；Gemma 3 为 5:1（w=1,024） | KV cache 上限为 w；128K context 下 Gemma 3 的 KV 约为全局版本的 1/6 |
| **MLA（Multi-head Latent Attention）** | 每个 token 只缓存一个压缩后的 latent c，需要时从 c 重建 K、V；up-projection 在 decode 时折叠进 query/output 权重 | DeepSeek-V2 相比 DeepSeek 67B KV cache 减少 93%，生成吞吐提高 5.8 倍 |
| **Linear attention** | 用固定大小的 state 代替不断增长的 KV cache；**Gated DeltaNet**：先按 α 衰减旧 state，再按 β 覆写新 key 指向的内容。Qwen3-Next：36 层 GDN + 12 层 attention（3:1，Kimi Linear、Qwen3.5 也用此配方） | 3/4 的层在任意 context 长度下 state 大小恒定 |
| **State-space models（Mamba）** | `h ← Ā h + B̄ x`，Ā、B̄ 依赖 token（selective）；每个 channel 一个小 state；hardware-aware parallel scan（Mamba-2 转成 matmul） | 没有增长的 KV cache；报告生成吞吐可达 Transformer 的 5 倍 |
| **Sparse attention**（DeepSeek-V3.2） | 轻量 indexer（少量 heads、FP8）为每个过去 token 打分，选 top-k（k = 2,048）做 attention | 每 token 工作量上限为 2,048 次读取，但所有 tokens 的 KV 仍要存储 |
| **Looped transformers** | 一个 block 重复使用 K 次（Prelude → Block × K → Coda） | 深度增加计算和延迟而不增加参数；循环是串行的，latency 随 K 增长 |
| **Multi-token prediction** | 一次预测多个 token | （列于目录，slides 未展开） |
| **Diffusion LMs** | 在几步并行 refinement 中填充被 mask 的 tokens（LLaDA 8B 从头训练；Dream 7B 从 Qwen2.5 7B 初始化） | 一次 pass 填多个 token，但步数越少质量越差 |
