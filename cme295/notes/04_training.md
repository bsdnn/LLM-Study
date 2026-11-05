# Lecture 4 · LLM 训练：Pretraining、训练优化、SFT 与 LoRA

> Stanford CME 295：Transformers & Large Language Models
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture4.pdf`](../slides/fall25-cme295-lecture4.pdf)

## 本讲脉络

```
范式转变（从零训练 → transfer learning → pretrain + tune）
→ Pretraining（数据、目标、规模、scaling laws、Chinchilla）
→ 训练优化（显存瓶颈、Data parallelism + ZeRO、Model parallelism、FlashAttention、混合精度）
→ SFT 与 instruction tuning、评估（benchmark 与 Chatbot Arena）
→ Parameter-efficient finetuning（LoRA、QLoRA）
```

一句话总结：LLM 训练分两步——先在万亿级 tokens 上**预训练**学会语言和知识，再用高质量的指令数据**微调**成助手。训练的核心瓶颈是**显存**，所以有了 ZeRO、各种并行、FlashAttention、混合精度；而 LoRA / QLoRA 让普通人也能用小 GPU 微调大模型。

---

## 1. 范式转变

| 阶段 | 做法 |
|---|---|
| **Traditional ML** | 每个任务（垃圾邮件检测、情感分析、翻译）都从零训练一个模型 |
| **Transfer learning** | 复用一个训练好的模型的部分能力 |
| **LLM 训练** | ① **Pretraining**：让模型理解语言 → “pretrained” model；② **Tuning**：针对最终任务调整 |

---

## 2. Pretraining

### 2.1 目标与数据

- **目标**：学习语言和代码的模式（英文、多语言、Python、Go……）
- **Objective function**：**预测下一个 token**。给定 “[BOS] A teddy bear is”，预测 “?” 处的 token，用 cross-entropy loss
- **数据混合（data mixture）**：
  - 网页爬取：Common Crawl、Wikipedia
  - 代码：GitHub、StackOverflow
- **规模**：约万亿（trillions）tokens

| 模型 | 预训练 tokens |
|---|---|
| GPT-3 | 3000 亿（300B） |
| LLaMA 3 | 15 万亿（15T） |

### 2.2 符号：FLOPs vs FLOPS

- **FLOPs**（FLoating-point OPerations）：浮点运算**总次数**，衡量计算量。训练一个模型的计算量 ≈ **6 × N × D**（N = 参数量，D = 训练 tokens 数；forward 约 2ND，backward 约 4ND）
- **FLOPS / FLOP/s**（per Second）：每秒浮点运算次数，衡量硬件**性能**

### 2.3 Scaling Laws（Kaplan et al., 2020）

- 模型的 test loss 与**参数量 N、数据量 D、计算量 C** 都呈**幂律关系**：任何一项增加，loss 都按幂律平滑下降（在其他两项不成为瓶颈时）
- **Sample efficiency**：更大的模型达到同样 loss 所需的数据更少

### 2.4 Chinchilla Law（Hoffmann et al., 2022）

- 问题：给定固定的计算预算，应该把模型做多大、训练多少 tokens？
- 结论：**参数量和训练 tokens 应该同比例增长**，compute-optimal 的经验比例约为 **每个参数 20 个 tokens**
- 当时很多模型（如 GPT-3：175B 参数只训练了 300B tokens）是“训练不足”的；Chinchilla（70B，1.4T tokens）用同样算力打败了更大的 Gopher（280B）
- 实践补充：如果要考虑**推理成本**，常常会把较小的模型训练远超 20 倍 tokens（如 LLaMA 3 8B 训练了 15T tokens），因为小模型推理更便宜

### 2.5 Pretraining 的挑战

**成本**：
- 至少数百万美元
- 耗时（数周到数月）
- 环境与电力消耗

**学到的知识**：
- **Knowledge cutoff**：模型只知道训练数据截止日期之前的信息
- **难以编辑知识**：想修正一条错误事实，没有简单办法
- **“Plagiarism”**：可能逐字复现训练数据

---

## 3. 训练优化

### 3.1 回顾一次训练迭代需要的显存

| 步骤 | 需要存储 | 大小 |
|---|---|---|
| 初始化 | **Model parameters** | O(数十亿) – O(数千亿) |
| Forward pass：计算 loss | **Activations**（反向传播要用） | 取决于模型大小、batch size、context length |
| Backward pass：计算梯度 | **Gradients** | 与参数量相同 |
| 更新权重 | **Optimizer state** | Adam 需要每个参数的一阶矩 m 和二阶矩 v，即 2 倍参数量 |

**Adam 优化器**（Kingma et al., 2014；AdamW 见 Loshchilov et al., 2017）：

```
m_t = β₁ m_{t−1} + (1 − β₁) g_t           # 梯度的指数移动平均（动量）
v_t = β₂ v_{t−1} + (1 − β₂) g_t²          # 梯度平方的指数移动平均
m̂_t = m_t / (1 − β₁ᵗ),  v̂_t = v_t / (1 − β₂ᵗ)   # 偏差修正
θ_t = θ_{t−1} − η · m̂_t / (√v̂_t + ε)
```

粗略估算：混合精度 + Adam 下，每个参数约需 **16 字节**（2B 半精度权重 + 2B 梯度 + 4B fp32 主权重 + 4B m + 4B v），还不含 activations。一个 70B 模型仅这部分就要约 1.1 TB。

### 3.2 瓶颈：显存

单张 GPU（如 NVIDIA H100）显存只有 **几十 GB**（80 GB），远不够放下大模型的训练状态。所以需要把东西拆到多张 GPU 上。

### 3.3 Data Parallelism（数据并行）

- 把一个 batch 的数据**分到多个设备**上
- 每个设备上都有**完整的模型副本**（parameters、gradients、optimizer state）
- 各自算梯度后通过 all-reduce 求平均，再各自更新

问题：每张卡都存一份完整的 parameters + gradients + optimizer state，**大量冗余**。

### 3.4 ZeRO（Zero Redundancy Optimizer，Rajbhandari et al., 2019）

**思路**：既然各设备上的这些信息是冗余的，就把它们**切片分散**存储，需要时再通信取回。

| 变体 | 分片的内容 | 每卡显存 |
|---|---|---|
| **ZeRO-1** | Optimizer state | 显著下降（optimizer state 占大头） |
| **ZeRO-2** | Optimizer state + gradients | 进一步下降 |
| **ZeRO-3** | Optimizer state + gradients + parameters | 最小（≈ 总量 / GPU 数），但通信最多 |

（PyTorch 的 FSDP 本质上就是 ZeRO-3。）

### 3.5 Model Parallelism（模型并行）

**思路**：把模型的**计算本身**拆到多个设备上。

| 变体 | 怎么拆 |
|---|---|
| **Tensor Parallelism (TP)** | 把单个矩阵乘法按行/列切开，多卡各算一部分（层内切分），通信频繁，适合同一节点内高速互联 |
| **Pipeline Parallelism (PP)** | 按层切分，不同 GPU 负责不同的层，像流水线；用 micro-batch 减少等待（“气泡”） |
| **Sequence Parallelism (SP)** | 沿序列维度切分 LayerNorm、dropout 等操作的 activations |
| **Context Parallelism (CP)** | 沿序列维度切分整个 attention 计算，用于超长 context |
| **Expert Parallelism (EP)** | MoE 模型中把不同 expert 放到不同 GPU |

推荐阅读：HuggingFace, “The Ultra-Scale Playbook: Training LLMs on GPU Clusters”, 2025。

### 3.6 FlashAttention（Dao et al., 2022）

**GPU 内存层级**：
- **SRAM**（片上，靠近计算单元）：**小而快**
- **HBM**（显存）：**大而慢**

**标准 attention 的问题**：计算本身不慢，慢在**反复读写 HBM**：

```
从 HBM 加载 Q、K → 计算 S = QKᵀ → 把 S 写回 HBM
从 HBM 读 S → 计算 P = softmax(S) → 把 P 写回 HBM
从 HBM 加载 P、V → 计算 O = PV → 把 O 写回 HBM
```

S 和 P 是 n × n 的大矩阵，读写它们成为瓶颈（attention 是 memory-bound 的）。

**想法 1：Tiling（分块）+ 减少 HBM 读写**
- 把 Q、K、V 切成小块载入 SRAM，在 SRAM 里直接算出 O 的一个块再写回 HBM，**从不把完整的 S、P 写到 HBM**
- 关键技巧：**online softmax** —— 不需要先算完整的一行 S 才能做 softmax。对每个分块维护当前的最大值 m 和归一化和 ℓ，新块进来时用 exp(m_old − m_new) 修正之前的结果。这样可以分块、增量地得到精确结果

**想法 2：Backward 中重算而不是存储**
- 之前：forward 时把 S、P 存到 HBM，backward 时读出来
- 之后：backward 时用 tiling 在 SRAM 中**重新计算**它们
- **FLOPs 更多了，但运行时间更短** —— 因为瓶颈在内存读写而不是计算

**结果**：显著加速，且是**精确计算**（不是近似），显存从 O(n²) 降到 O(n)。

### 3.7 数值精度与混合精度训练

**浮点数表示** = 符号位（sign）+ 指数位（exponent，决定范围）+ 尾数位（mantissa，决定精度）

| 格式 | Exponent | Mantissa | 特点 |
|---|---|---|---|
| FP64 | 11 | 52 | 科学计算 |
| FP32 | 8 | 23 | 传统训练默认 |
| FP16 | 5 | 10 | 范围小，容易溢出/下溢，需要 loss scaling |
| **BF16**（Brain Float 16） | **8** | 7 | 与 FP32 **范围相同**、精度更低，深度学习中更稳定，现在训练常用 |

**精度越低 → 计算越快、显存越省**（GPU 的 Tensor Core 对低精度有更高吞吐）。

**Mixed Precision Training（Micikevicius et al., 2017）**：
- **Forward**：activations 用低精度
- **Backward**：梯度计算用低精度
- **Weights update**：**权重的主副本保持高精度（FP32）** —— 因为每步的更新量很小，用低精度累加会被舍入掉

目标：加速训练、降低显存，同时保持精度。

---

## 4. Supervised Fine-Tuning（SFT）

### 4.1 为什么需要微调

预训练模型只会“续写”。问它 “Can I put my teddy bear in the washer?”，它可能续写一段关于 teddy bear 材质的百科文字（“Teddy bears are often made of materials like polyester and cotton…”），而不是回答问题。

### 4.2 SFT 是什么

- **思路**：通过调整权重来改变模型行为
- **做法**：收集体现期望行为的输入/输出对（**SFT data**），用“给定输入，预测下一个 token”的目标训练（loss 通常只算在输出部分）
- **特例**：在指令跟随数据上做 SFT，叫 **instruction tuning**

### 4.3 Instruction Tuning（Wei et al., 2022，FLAN）

**目标**：让模型“毕业”成为一个有用的助手。

示例数据：
- 写故事：“Write a short story about a teddy bear who likes to read poetry.”
- 生成列表：“List three fun activities a teddy bear might do on a rainy day.” → “Sure! 1. Read poetry with friends. 2. Be cute. 3. Hug its owner tightly.”
- 写诗、解释概念……

**数据混合**：人写的和合成的都有
- 助手对话（assistant dialogs）
- 合成指令（synthetic instructions）
- 数学、推理、代码
- 安全对齐（safety alignment）：学会拒绝有害请求

**规模**：数千到数百万条样本（例如 1.3 万条到 1000 万条不等）。

**效果**：同样问 “Can I put my teddy bear in the washer?”，微调后回答 “No, it might get damaged. Try hand washing instead.”

### 4.4 SFT 的挑战

- 需要**非常高质量**的数据
- 对 prompt 的分布敏感
- 泛化能力难保证
- 难以评估
- 计算昂贵

### 4.5 评估

**Benchmarks**：

| 维度 | Benchmark |
|---|---|
| 通用知识 | MMLU |
| 基础推理 | ARC-Challenge |
| 数学推理 | GSM8K |
| 代码生成 | HumanEval |

**有效性问题**：不同模型训练数据差异大，直接比较不公平。有研究建议**在比较前让各模型都在测试任务上训练**（Dominguez-Olmedo et al., 2024，“Training on the Test Task Confounds Evaluation and Emergence”）。

**“真实感受”：Chatbot Arena（LMArena）**
- 对用户的 prompt 做 A/B 测试：两个匿名模型回答，用户投票
- 好处：给“vibes”（主观感受）打了一个数字（Elo 排名）
- 挑战：
  - 模型曝光不均 / 新模型 cold start
  - 容易被操纵（“rig”）
  - 用户无法准确评估事实性等重要方面
  - 个人偏好偏差（分布不具代表性）
  - 对安全拒答的惩罚（拒绝回答的模型容易输）

结论：**评估本身就是一个难题**。

### 4.6 LLM 生命周期

```
初始化模型 ──Pretraining──► 有“基础知识”的模型（语言、代码…）
           ──SFT──────────► 针对特定任务调整的模型
           ──Preference tuning（Lecture 5）──► 不太会“乱来”的模型（alignment）
```

---

## 5. Parameter-Efficient Fine-Tuning（PEFT）

### 5.1 LoRA（Low-Rank Adaptation，Hu et al., 2021）

**背景**：SFT 资源消耗大，不是每个人都有大 GPU。

**思路**：微调时权重的变化量 ΔW 是“低秩”的，可以用两个小矩阵的乘积近似：

```
W = W₀ + ΔW ≈ W₀ + B · A
W₀ ∈ ℝ^{d×k}（冻结）,  B ∈ ℝ^{d×r},  A ∈ ℝ^{r×k},  r ≪ min(d, k)
```

- 只训练 A 和 B，W₀ 冻结
- 例：d = k = 4096，r = 8 时，全量要训练 1677 万参数，LoRA 只需 6.5 万（约 0.4%）
- 初始化：A 随机，B = 0，所以一开始 ΔW = 0，模型行为不变
- 推理时可以把 BA 合并回 W₀，**不增加推理延迟**

**讨论**：
- 只训练一小部分参数，效果接近全量微调
- 其他 PEFT 方法：prefix tuning、adapters

**好处：换矩阵 = 换任务**。同一个基础模型，配不同的 (B, A) 对就变成垃圾邮件检测、情感分析、翻译等不同任务的模型，存储和切换都很便宜。

### 5.2 LoRA 加在哪里

- **原论文**：只加在 attention 的 W_Q、W_V 上
- **现在的建议**（Schulman et al., 2025，“LoRA Without Regret”）：加在**所有**线性层上，尤其是 **MLP / FFN 层，这是最重要的位置**

### 5.3 训练动态的注意点

- LoRA 需要比全量微调**更高的 learning rate**（经验上约 10 倍）
- LoRA 在**大 batch size** 下比全量微调表现差

### 5.4 QLoRA（Dettmers et al., 2023）

**思路**：把所有**冻结的权重量化**到 4-bit 存储，缓解显存瓶颈；LoRA 的 A、B 保持全精度；计算时把权重反量化回高精度再算。

**高效量化技巧**：
- **NF4（4-bit NormalFloat）**：神经网络权重近似服从正态分布。普通 INT 量化按**均匀间隔**切分取值范围；NF4 按**正态分布的分位数**切分，让每个量化区间里的权重数量相同，更好地利用 16 个取值
- **Double quantization**：量化需要每个 block 存一个缩放常数（quantization constant），这些常数本身也占空间；QLoRA **把这些常数再量化一次**

**收益**：
- 节省显存，可以在更小的 GPU 上微调，而且更快
- 在显存与质量之间取得更好的平衡
- 数量级（LLaMA 65B）：微调时显存节省约 **16 倍**；double quantization 额外节省约 6%

---

## 参考论文

- Brown et al., “Language Models are Few-Shot Learners” (GPT-3), 2020；LLaMA team, “The Llama 3 Herd of Models”, 2024
- Kaplan et al., “Scaling Laws for Neural Language Models”, 2020
- Hoffmann et al., “Training Compute-Optimal Large Language Models” (Chinchilla), 2022
- Kingma et al., “Adam”, 2014；Loshchilov et al., “Decoupled Weight Decay Regularization”, 2017
- Rajbhandari et al., “ZeRO”, 2019；HuggingFace, “The Ultra-Scale Playbook”, 2025
- Dao et al., “FlashAttention”, 2022
- Micikevicius et al., “Mixed Precision Training”, 2017
- Wei et al., “Finetuned Language Models are Zero-Shot Learners” (FLAN), 2022
- Dominguez-Olmedo et al., “Training on the Test Task Confounds Evaluation and Emergence”, 2024；Huang et al., “Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards”, 2025
- Hu et al., “LoRA”, 2021；Schulman et al., “LoRA Without Regret”, 2025；Dettmers et al., “QLoRA”, 2023
