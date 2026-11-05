# Lecture 4 · LLM 训练：Pretraining、训练优化、SFT 与 LoRA

> Stanford CME 295：Transformers & Large Language Models（Afshine Amidi & Shervine Amidi，Fall 2025）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture4.pdf`](../slides/fall25-cme295-lecture4.pdf)
> 预计阅读时间：18–22 分钟。预备知识：Lecture 1–3（Transformer、decoder-only、KV cache）。

## 本讲脉络

```
范式转变：从“每个任务从零训练”到“先预训练、再调优”
→ Pretraining：目标、数据、规模、FLOPs 怎么算、scaling laws、Chinchilla、挑战
→ 训练优化：显存都花在哪 → 数据并行 + ZeRO → 模型并行（TP / PP / SP / CP / EP）→ FlashAttention → 混合精度
→ SFT 与 instruction tuning：为什么需要、数据、挑战、怎么评估（benchmark 与 Chatbot Arena）
→ Parameter-efficient fine-tuning：LoRA（原理、参数量、放在哪、训练注意事项）→ QLoRA（NF4、double quantization）
```

**一句话总结**：LLM 训练分两大步 —— 先在万亿级 tokens 上**预训练（pretraining）**学会语言和知识，再用高质量的指令数据**微调（SFT）**成为助手。训练的核心瓶颈是**显存**，所以才有 ZeRO、各种并行、FlashAttention、混合精度这些技术；而 LoRA / QLoRA 让普通人也能在一张消费级或单张专业 GPU 上微调大模型。

---

## 1. 范式转变

| 阶段 | 做法 | 问题 |
|---|---|---|
| **Traditional ML** | 每个任务（垃圾邮件检测、情感分析、翻译）都单独收集数据、**从零训练**一个模型 | 每个任务都需要大量标注数据；知识不能复用 |
| **Transfer learning** | 复用一个已经训练好的模型的部分能力（如用 ImageNet 预训练的 CNN 做医学图像） | 需要选择复用哪部分 |
| **LLM 训练** | ① **Pretraining**：用海量无标注文本让模型理解语言 → “pretrained” model；② **Tuning**：针对最终用途调整 | 预训练极其昂贵，但只需做一次 |

为什么“先预训练”有效：语言知识（语法、常识、事实、推理模式）在各种任务之间是**共享**的。在海量文本上学到这些通用能力后，下游任务只需要少量数据就能适配。

---

## 2. Pretraining（预训练）

### 2.1 目标：预测下一个 token

**目标**：学习语言和代码的模式 —— 英文故事、多语言文本（法语、波斯语……）、Python 和 Go 代码，统统放在一起学。

**Objective function**：给定 “[BOS] A teddy bear is”，预测下一个 token。对整个序列，loss 是每个位置的 cross-entropy 之和：

$$
\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t})
$$

训练时用 causal mask，一条长度为 $T$ 的序列一次 forward 就能得到 $T$ 个位置的预测，**每个 token 都提供训练信号**（比 BERT 的 MLM 只用 15% 的位置效率高）。

**为什么“预测下一个词”能学到这么多**：要把下一个词预测准，模型必须掌握语法（“The bears ___” 后面接 are 而不是 is）、事实（“The capital of France is ___”）、常识、甚至推理（“3 + 4 = ___”）。这个看似简单的目标，逼迫模型压缩并理解训练数据中的大量规律。

### 2.2 数据

**来源（data mixture，数据配比）**：
- 网页爬取：Common Crawl（数十亿网页的公开快照）、Wikipedia
- 代码：GitHub、StackOverflow
- 以及书籍、论文、数学数据等

**处理流程**（原始网页质量参差不齐，必须清洗）：
1. 从 HTML 中抽取正文
2. 语种识别
3. **去重**：精确去重 + 模糊去重（如 MinHash 找出近似重复的网页）。重复数据会让模型死记硬背，浪费算力
4. **质量过滤**：启发式规则（如去掉太短、符号太多的页面）+ 质量分类器
5. 安全过滤：去掉有害内容、个人隐私信息（PII）
6. **去污染（decontamination）**：去掉与评测集重叠的内容，防止“考试题出现在课本里”

**配比很重要**：例如 LLaMA 3 的最终数据配比约为 50% 通用知识、25% 数学与推理、17% 代码、8% 多语言。

**规模**：约万亿（trillions）tokens。

| 模型 | 预训练 tokens |
|---|---|
| GPT-3（2020） | 3000 亿（300B） |
| LLaMA 3（2024） | 15 万亿（15T） |

### 2.3 FLOPs 与 FLOPS：训练要多少计算

- **FLOPs**（FLoating-point OPerations）：浮点运算的**总次数**，衡量计算量
- **FLOPS 或 FLOP/s**（per Second）：每秒浮点运算次数，衡量硬件性能

**经验公式**：训练一个 $N$ 参数的模型、用 $D$ 个 token，总计算量约为

$$
C \approx 6 N D
$$

推导：forward 时每个参数对每个 token 做约 1 次乘法 + 1 次加法 = **2 FLOPs**；backward 要分别对激活值和权重求梯度，约是 forward 的 2 倍 = **4 FLOPs**；合计约 6。

**算例 1：GPT-3**（$N = 175$B，$D = 300$B）：

$$
C \approx 6 \times 1.75\times10^{11} \times 3\times10^{11} \approx 3.15\times10^{23}\ \text{FLOPs}
$$

假设用 1,000 张 A100（BF16 峰值 312 TFLOP/s），实际利用率（MFU）40%，即每张卡有效 125 TFLOP/s：

$$
\frac{3.15\times10^{23}}{1000 \times 1.25\times10^{14}} \approx 2.5\times10^{6}\ \text{秒} \approx 29\ \text{天}
$$

**算例 2：LLaMA 3 405B**，训练 15.6T tokens：$6 \times 4.05\times10^{11} \times 1.56\times10^{13} \approx 3.8\times10^{25}$ FLOPs，比 GPT-3 多约 100 倍，用了上万张 H100。

### 2.4 Scaling Laws（Kaplan et al., 2020）

**发现**：模型的 test loss 与 **参数量 $N$、数据量 $D$、计算量 $C$** 都呈**幂律关系**。在另外两项不成为瓶颈时：

$$
L(N) \approx \Big(\frac{N_c}{N}\Big)^{\alpha_N},\quad
L(D) \approx \Big(\frac{D_c}{D}\Big)^{\alpha_D},\quad
L(C) \approx \Big(\frac{C_c}{C}\Big)^{\alpha_C}
$$

（论文中 $\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$，$\alpha_C \approx 0.05$。）

**含义**：
- 在双对数坐标上是**直线**：只要持续投入，loss 会平滑、可预测地下降，没有明显的“天花板”
- 可以用小模型的实验**外推**大模型的表现，提前规划训练
- **Sample efficiency（样本效率）**：更大的模型达到同样 loss 所需的数据更少

### 2.5 Chinchilla Law（Hoffmann et al., 2022）

**问题**：给定固定的计算预算 $C$，应该把模型做多大（$N$）、训练多少 tokens（$D$）？

**结论**：**参数量和训练 tokens 应该同比例增长**（$N_{\text{opt}} \propto C^{0.5}$，$D_{\text{opt}} \propto C^{0.5}$）。经验比例约为 **每个参数 20 个 tokens**。

**例子**：
- Gopher：280B 参数、300B tokens（约 1 token/参数），计算量约 $5\times10^{23}$
- Chinchilla：70B 参数、1.4T tokens（20 tokens/参数），计算量约 $5.9\times10^{23}$，与 Gopher 相当
- 结果：**Chinchilla 小 4 倍，却在几乎所有任务上打败了 Gopher**
- 同理，GPT-3（175B 参数只训练 300B tokens）是“训练不足”的

**实践补充**：Chinchilla 只考虑**训练**成本最优。如果模型要被大量使用，**推理成本**更重要，而小模型推理便宜。所以业界常常把较小的模型训练远超 20 倍 tokens（over-training）。例如 LLaMA 3 8B 训练了 15T tokens，约 1,900 tokens/参数。

### 2.6 Pretraining 的挑战

**成本**：
- 至少数百万美元，前沿模型更高
- 耗时：数周到数月
- 环境与电力消耗

**学到的知识**：
- **Knowledge cutoff（知识截止）**：模型只知道训练数据截止日期之前的信息（模型卡上会写明截止日期）
- **知识难以编辑**：想修正一条错误事实，没有简单的办法（不能直接找到“存这条知识的那个参数”）
- **“Plagiarism”（剽窃）**：模型可能逐字复现训练数据中的段落，带来版权和隐私问题

---

## 3. 训练优化

### 3.1 一次训练迭代需要存什么

| 步骤 | 需要存储 | 大小 |
|---|---|---|
| 初始化 | **Model parameters（参数）** | O(数十亿) – O(数千亿) |
| Forward pass：计算 loss | **Activations（激活值）**：每层的中间结果，backward 时要用 | 取决于模型大小、batch size、context length |
| Backward pass：计算梯度 | **Gradients（梯度）** | 与参数量相同 |
| Weights update：更新参数 | **Optimizer state（优化器状态）** | Adam 是参数量的 2 倍 |

**Adam 优化器**（Kingma et al., 2014）为每个参数维护两个状态：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\, g_t,\qquad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t^2
$$

$$
\hat m_t = \frac{m_t}{1-\beta_1^t},\quad \hat v_t = \frac{v_t}{1-\beta_2^t},\qquad
\theta_t = \theta_{t-1} - \eta\, \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}
$$

- $m$：梯度的指数移动平均（**动量**），让更新方向更平滑
- $v$：梯度平方的指数移动平均，用来**按参数自适应地调整步长**（梯度一直很大的参数步长变小）
- $\hat m$、$\hat v$：偏差修正（初始时 $m$、$v$ 都是 0，前几步会偏小）
- **AdamW**（Loshchilov et al., 2017）：把 weight decay 从梯度中“解耦”出来，直接加在更新上：$\theta_t = \theta_{t-1} - \eta\big(\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon} + \lambda\,\theta_{t-1}\big)$，是训练 LLM 的标准选择

**显存账本**（混合精度 + Adam，每个参数）：

| 内容 | 精度 | 字节 |
|---|---|---|
| 参数（用于计算） | BF16 | 2 |
| 梯度 | BF16 | 2 |
| 参数的主副本（master weights） | FP32 | 4 |
| Adam 的 $m$ | FP32 | 4 |
| Adam 的 $v$ | FP32 | 4 |
| **合计** | | **16** |

**算例**：一个 7B 模型，仅这些状态就需要 $7\times10^9 \times 16 \approx 112$ GB，还没算 activations。

**Activations 的处理**：
- **Activation checkpointing（重计算）**：forward 时只保存一部分 activations（例如每层的输入），backward 时需要的其他中间结果**重新算一遍**。用约 30% 的额外计算换取大量显存
- **Gradient accumulation（梯度累积）**：把一个大 batch 拆成几个小 micro-batch，依次 forward + backward，梯度累加，最后再更新一次参数。用时间换显存，实现更大的有效 batch

### 3.2 瓶颈：显存

单张 GPU 的显存只有 **几十 GB**（如 NVIDIA H100 为 80 GB）。一个 7B 模型的训练状态就放不下一张卡，70B、400B 模型更不用说。所以必须把东西**拆到多张 GPU 上**。

### 3.3 Data Parallelism（数据并行，DP）

- **思路**：把一个 batch 的数据**分到多个设备**上，每个设备上都有**完整的模型副本**
- **流程**：每张卡用自己那份数据算 forward 和 backward → 所有卡之间做 **all-reduce**（梯度求平均）→ 每张卡用相同的平均梯度更新参数，保持副本一致
- 全局 batch size = 每卡 batch × 卡数
- **问题**：每张卡都存一份完整的 parameters、gradients、optimizer state —— **大量冗余**。模型本身放不进一张卡时，DP 无能为力

### 3.4 ZeRO（Zero Redundancy Optimizer，Rajbhandari et al., 2019）

**思路**：既然各设备上的这些状态是**冗余**的，就把它们**切片分散**存储（每张卡只负责 $1/N_d$），需要时再通过通信取回。

设参数量为 $\Psi$，卡数为 $N_d$，优化器状态每参数 12 字节（FP32 主副本 + $m$ + $v$）：

| 变体 | 分片的内容 | 每卡显存 | 7.5B 模型、64 卡 |
|---|---|---|---|
| 普通 DP | 不分片 | $(2 + 2 + 12)\Psi = 16\Psi$ | 120 GB |
| **ZeRO-1** | Optimizer state | $4\Psi + 12\Psi / N_d$ | 31.4 GB |
| **ZeRO-2** | Optimizer state + gradients | $2\Psi + 14\Psi / N_d$ | 16.6 GB |
| **ZeRO-3** | Optimizer state + gradients + parameters | $16\Psi / N_d$ | 1.9 GB |

- ZeRO-1 收益最大（optimizer state 占了 16 字节中的 12 字节）
- ZeRO-3 每卡显存最小，但每一层计算前都要从其他卡收集该层参数（all-gather），**通信量约为普通 DP 的 1.5 倍**
- PyTorch 的 **FSDP**（Fully Sharded Data Parallel）本质上就是 ZeRO-3

### 3.5 Model Parallelism（模型并行）

**思路**：把模型的**计算本身**拆到多个设备上。

**Tensor Parallelism（TP，张量并行）**：把**单个矩阵乘法**切开，多张卡各算一部分。
- 以 FFN 为例（Megatron-LM）：第一个线性层 $W_1$ **按列切**，每张卡得到一部分隐藏单元，各自做激活函数（不需要通信）；第二个线性层 $W_2$ **按行切**，每张卡算出一个部分和，最后做一次 all-reduce 相加
- Attention 按 head 切分，不同卡负责不同的 head
- 每层都要通信，所以通常放在**同一台机器内**（GPU 之间有高速 NVLink）

**Pipeline Parallelism（PP，流水线并行）**：**按层切分**，不同 GPU 负责不同的层，像工厂流水线。
- 问题：GPU 1 算完第 1–10 层才能交给 GPU 2，其他 GPU 在等待（**bubble，气泡**）
- 解决：把 batch 切成 $m$ 个 micro-batch 依次流入，让各阶段同时处理不同的 micro-batch
- 气泡占比约为 $\frac{p - 1}{m + p - 1}$（$p$ = 阶段数）。例：$p = 4$、$m = 16$ 时约 16%

**Sequence Parallelism（SP，序列并行）**：LayerNorm、dropout 等操作在 TP 下会被重复计算，SP 沿**序列维度**把这些操作的 activations 切开，节省显存。

**Context Parallelism（CP，上下文并行）**：沿序列维度切分**整个 attention 计算**（如 ring attention：每张卡持有一段序列的 Q，依次轮转接收其他段的 K、V），用于超长 context。

**Expert Parallelism（EP，专家并行）**：MoE 模型中把不同 expert 放到不同 GPU，token 通过 all-to-all 通信被送到对应 expert 所在的卡。

**实际训练会组合多种并行**：例如机器内用 TP（8 卡、NVLink 高速），机器之间用 PP，再在多个副本之间用 DP / ZeRO。推荐阅读：HuggingFace, “The Ultra-Scale Playbook: Training LLMs on GPU Clusters”, 2025。

### 3.6 FlashAttention（Dao et al., 2022）

**GPU 内存层级**：

| 层级 | 容量（以 A100 为例） | 带宽 | 特点 |
|---|---|---|---|
| **SRAM**（片上，靠近计算单元） | 每个 SM 192 KB，共约 20 MB | 约 19 TB/s | **小而快** |
| **HBM**（显存） | 40–80 GB | 约 1.5–2 TB/s | **大而慢** |

**标准 attention 的问题**：计算本身不慢，慢在**反复读写 HBM**：

```
从 HBM 加载 Q、K → 计算 S = QKᵀ → 把 S 写回 HBM
从 HBM 读 S → 计算 P = softmax(S) → 把 P 写回 HBM
从 HBM 加载 P、V → 计算 O = PV → 把 O 写回 HBM
```

$S$ 和 $P$ 都是 $n \times n$ 的大矩阵。$n = 8{,}192$ 时每个 head 的 $S$ 就有 6700 万个元素。读写它们的时间远超计算时间 —— attention 是 **memory-bound（内存受限）** 的。

**想法 1：Tiling（分块）+ 减少 HBM 读写**
- 把 Q、K、V 切成小块载入 SRAM，在 SRAM 里直接算出输出 $O$ 的一个块，再写回 HBM
- **从不把完整的 $S$、$P$ 写到 HBM**

**难点**：softmax 需要整行的最大值和总和，怎么分块算？答案是 **online softmax**：对每一行维护“目前为止的最大值 $m$”和“目前为止的归一化和 $\ell$”，每来一个新块就更新它们，并用 $e^{m_{\text{old}} - m_{\text{new}}}$ 修正之前累积的结果。

**Online softmax 算例**：某一行的分数分两块到达：块 1 = [1, 2]，块 2 = [3, 0]。
- 处理块 1：$m = 2$，$\ell = e^{1-2} + e^{2-2} = 0.368 + 1 = 1.368$，部分输出 $o = 0.368\,v_1 + 1\cdot v_2$
- 处理块 2：新最大值 $m' = 3$；修正旧值：$\ell' = 1.368 \times e^{2-3} + (e^{3-3} + e^{0-3}) = 0.503 + 1.050 = 1.553$；$o' = o \times e^{2-3} + (1\cdot v_3 + 0.050\,v_4)$
- 最终输出 $o' / \ell'$
- 验证：一次性计算时，最大值 3，$e^{-2} + e^{-1} + e^{0} + e^{-3} = 0.135 + 0.368 + 1 + 0.050 = 1.553$ ✓ 完全一致

**想法 2：Backward 中“重算”而不是“存储”**
- 之前：forward 时把 $S$、$P$ 存到 HBM，backward 时读出来
- 之后：forward 只存每行的 $m$ 和 $\ell$（很小），backward 时用 tiling 在 SRAM 中**重新计算** $S$、$P$
- **FLOPs 更多了，但运行时间更短** —— 因为瓶颈在内存读写，而不是计算

**结果**：
- 显著加速：GPT-2（长度 1K）训练快约 3 倍，BERT-large 端到端训练快约 15%
- 显存从 $O(n^2)$ 降到 $O(n)$，使更长的 context 成为可能
- **是精确计算，不是近似** —— 结果与标准 attention 完全相同（仅有浮点舍入差异）

### 3.7 数值精度与混合精度训练

**浮点数的表示** = 符号位（sign）+ 指数位（exponent，决定**数值范围**）+ 尾数位（mantissa，决定**精度**）

| 格式 | Exponent | Mantissa | 最大值 | 特点 |
|---|---|---|---|---|
| FP64 | 11 | 52 | ~$1.8\times10^{308}$ | 科学计算 |
| FP32 | 8 | 23 | ~$3.4\times10^{38}$ | 传统训练默认 |
| FP16 | 5 | 10 | 65,504 | 范围小：大数会溢出，小于约 $6\times10^{-8}$ 的数变成 0 |
| **BF16**（Brain Float 16） | **8** | 7 | ~$3.4\times10^{38}$ | 与 FP32 **范围相同**、精度更低（约 2–3 位有效数字），训练更稳定 |

**精度越低 → 计算越快、显存越省**：GPU 的 Tensor Core 对 16 位（甚至 8 位）运算有数倍于 FP32 的吞吐。

**Mixed Precision Training（Micikevicius et al., 2017）**：

| 环节 | 精度 | 原因 |
|---|---|---|
| Forward：activations | 低精度（FP16 / BF16） | 快、省显存 |
| Backward：梯度 | 低精度 | 快、省显存 |
| **Weights update：参数主副本** | **高精度（FP32）** | 每次更新量很小，低精度下会被舍入掉 |

**为什么主副本必须用 FP32**：BF16 只有 7 位尾数，在 1.0 附近能表示的相邻两个数相差 $2^{-7} \approx 0.0078$。如果某个参数是 1.0，这一步的更新量是 0.001，那么 1.0 + 0.001 在 BF16 中仍然是 1.0 —— **更新直接消失**。成千上万步的小更新累积起来才是训练的效果，所以必须在 FP32 中累加。

**FP16 的额外问题与 loss scaling**：FP16 范围小，很多梯度（如 $10^{-8}$）会下溢成 0。做法是把 loss 先乘以一个大数（如 1024）再 backward，梯度也随之放大到可表示的范围，更新前再除回去。BF16 的范围与 FP32 相同，通常不需要 loss scaling，这是它更受欢迎的原因。

**更进一步**：H100 等新 GPU 支持 **FP8**，前沿模型训练开始在部分计算中使用 FP8。

---

## 4. Supervised Fine-Tuning（SFT）

### 4.1 为什么需要微调

预训练模型只会“续写”。问它 “Can I put my teddy bear in the washer?”，它可能续写一段关于 teddy bear 材质的百科文字：

> Teddy bears are often made of materials like polyester and cotton, with plastic eyes and sometimes small accessories.

它并没有理解“这是一个需要回答的问题”。**补救：fine-tuning**，把模型调成针对特定用途的模型。

### 4.2 SFT 是什么

- **思路**：通过调整权重来改变模型的行为
- **做法**：
  1. 收集体现期望行为的“输入 → 输出”对（**SFT data**）
  2. 用“给定输入，预测下一个 token”的目标训练
- **特例**：在“遵循指令”的数据上做 SFT，叫 **instruction tuning**

**对话格式与 loss mask**：数据通常组织成带角色的对话，用特殊 token 标记：

```
<|system|> You are a helpful assistant.
<|user|> Can I put my teddy bear in the washer?
<|assistant|> No, it might get damaged. Try hand washing instead.<|end|>
```

**Loss 只在 assistant 的回答部分计算**（prompt 部分的 token 不计 loss）。我们要模型学的是“怎么回答”，而不是“怎么提问”：

$$
\mathcal{L}_{\text{SFT}} = -\sum_{t \,\in\, \text{response}} \log p_\theta(y_t \mid x, y_{<t})
$$

### 4.3 Instruction Tuning（Wei et al., 2022，FLAN）

**目标**：让模型“毕业”成为一个有用的助手。

示例数据（slides 中的四种类型）：

| 类型 | 输入 | 输出 |
|---|---|---|
| 写故事 | Write a short story about a teddy bear who likes to read poetry. | Once upon a time, a bear, Teddy, stumbled upon verses from Attar… |
| 生成列表 | List three fun activities a teddy bear might do on a rainy day. | Sure! 1. Read poetry with friends. 2. Be cute. 3. Hug its owner tightly. |
| 写诗 | Create a poem about my cute teddy bear. | Soft and cuddly, full of charm, … |
| 解释 | Explain why a teddy bear is a great friend. | A teddy bear is a great friend because it provides comfort and companionship… |

**FLAN 的发现**：把 60 多个 NLP 数据集改写成自然语言指令的形式来微调，模型在**从没见过的任务**上的 zero-shot 能力大幅提升。137B 的 FLAN 在 25 个数据集中的 20 个上超过了 175B GPT-3 的 zero-shot 成绩。**学会“遵循指令”这件事本身是可以泛化的。**

**数据配比**：人工撰写的和合成的数据都有
- 助手对话（assistant dialogs）
- 合成指令（synthetic instructions，例如让强模型生成指令和回答，Self-Instruct 方法）
- 数学、推理、代码
- 安全对齐（safety alignment）：学会拒绝有害请求

**规模**：数千到数百万条。例如 InstructGPT 的 SFT 数据约 1.3 万条；LLaMA 3 的后训练使用了超过 1000 万条人工标注样本。

**质量胜过数量**：LIMA（Zhou et al., 2023）只用 1,000 条精心挑选的样本微调 65B 模型，就得到了很不错的助手 —— 说明模型的知识主要来自预训练，SFT 更多是在教它“以什么格式和风格回答”。

**效果**：同样问 “Can I put my teddy bear in the washer?”，微调后回答 “No, it might get damaged. Try hand washing instead.”

### 4.4 SFT 的挑战

- 需要**非常高质量**的数据（模型会忠实地模仿数据中的错误和坏习惯）
- 对 prompt 的分布敏感（训练数据没覆盖到的提问方式，表现可能变差）
- 泛化能力难以保证
- 难以评估
- 计算昂贵

### 4.5 评估

**Benchmarks**：

| 维度 | Benchmark | 内容 |
|---|---|---|
| 通用知识 | **MMLU** | 57 个学科（数学、历史、法律、医学……）的四选一题，约 1.4 万道测试题 |
| 基础推理 | **ARC-Challenge** | 小学科学选择题中，简单检索方法答不对的“难题”子集 |
| 数学推理 | **GSM8K** | 约 8,500 道小学数学应用题，需要 2–8 步计算 |
| 代码生成 | **HumanEval** | 164 道手写的 Python 编程题，用单元测试判分（pass@k） |

**有效性问题**：不同模型的训练数据差异很大，有的“碰巧”见过类似题型，直接比较不公平。有研究建议在比较前，**让每个模型都先在测试任务的训练集上训练**，以消除这种差异（Dominguez-Olmedo et al., 2024，“Training on the Test Task Confounds Evaluation and Emergence”）。

**“真实感受”：Chatbot Arena（LMArena）**
- 做法：用户输入任意 prompt，两个**匿名**模型分别回答，用户投票选更好的；再用 Bradley-Terry / Elo 模型把大量两两比较结果转成排行榜
- 好处：**给“vibes”（主观感受）打了一个数字**，更接近真实使用体验
- 挑战：
  - 模型曝光不均 / 新模型的 cold start 问题
  - 容易被操纵（“rig”）：例如针对性地投票（Huang et al., 2025）
  - 用户无法准确评估事实性等重要方面（回答看起来自信就容易被选）
  - 个人偏好偏差：投票用户的分布不具代表性
  - 对安全拒答的惩罚：拒绝回答的模型容易输

结论：**评估本身就是一个难题**（Lecture 8 详述）。

### 4.6 LLM 的生命周期

```
初始化模型
  ──Pretraining──────────► 有“基础知识”的模型（语言、代码……）
  ──SFT──────────────────► 针对特定任务调整的模型
  ──Preference tuning────► 不太会“乱来”的模型（alignment，Lecture 5）
```

---

## 5. Parameter-Efficient Fine-Tuning（PEFT）

### 5.1 动机

全量微调（full fine-tuning）每个参数都要存梯度和优化器状态，显存需求和预训练一样（约 16 字节 / 参数）：7B 模型需要 100+ GB。而且**每个任务都要存一整份模型副本**。不是每个人都有大 GPU。

### 5.2 LoRA（Low-Rank Adaptation，Hu et al., 2021）

**核心假设**：微调时权重的**变化量** $\Delta W$ 是“低秩”的 —— 虽然 $\Delta W$ 是一个很大的矩阵，但它真正需要的自由度很少，可以用两个小矩阵的乘积近似。

$$
W = W_0 + \Delta W \approx W_0 + B A,\qquad
W_0 \in \mathbb{R}^{d \times k}\ (\text{冻结}),\quad B \in \mathbb{R}^{d \times r},\quad A \in \mathbb{R}^{r \times k},\quad r \ll \min(d, k)
$$

前向计算：$h = W_0 x + \frac{\alpha}{r} B A x$（$\alpha$ 是缩放系数，$\alpha / r$ 让调整 $r$ 时不必重新调学习率）。

**训练细节**：
- 只训练 $A$ 和 $B$，$W_0$ 完全冻结（不需要它的梯度和优化器状态）
- 初始化：$A$ 用随机高斯，**$B = 0$**。所以一开始 $BA = 0$，模型行为与原模型完全相同，训练从原模型平滑地出发

**参数量算例 1**（单个矩阵）：$d = k = 4096$，$r = 8$
- 全量微调：$4096 \times 4096 \approx 1678$ 万参数
- LoRA：$r(d + k) = 8 \times 8192 = 65{,}536$ 参数，约 **0.4%**

**参数量算例 2**（整个模型）：LLaMA 2 7B，$r = 16$，加在每层所有线性层上（q、k、v、o 各 4096×4096；gate、up 为 4096×11008；down 为 11008×4096）：
- 每层：$4 \times 16 \times 8192 + 3 \times 16 \times 15104 \approx 125$ 万
- 32 层：约 **4000 万**参数，约占 7B 的 **0.6%**
- 冻结的权重用 BF16 存约 13.5 GB，可训练部分的梯度和优化器状态不到 1 GB —— 一张 24 GB 的显卡就能微调

**为什么低秩就够**：研究发现预训练模型有很低的“内在维度（intrinsic dimension）”—— 在一个很小的子空间里调整参数，就足以适配新任务。

**原论文结果**（GPT-3 175B）：可训练参数减少约 10,000 倍，GPU 显存减少约 3 倍，效果与全量微调相当甚至更好。

**推理**：
- 可以把 $BA$ **合并回** $W_0$（$W = W_0 + BA$），**不增加任何推理延迟**
- 也可以保持分离：同一个基础模型 + 多组小的 $(B, A)$

**好处：换矩阵 = 换任务**。同一个基础模型，配上 $(B_{\text{spam}}, A_{\text{spam}})$ 就是垃圾邮件检测模型，配上 $(B_{\text{sentiment}}, A_{\text{sentiment}})$ 就是情感分析模型，配上 $(B_{\text{translation}}, A_{\text{translation}})$ 就是翻译模型。每组只有几十 MB，存储和切换都很便宜，一台服务器可以同时为很多任务服务。

**其他 PEFT 方法**：prefix tuning（在每层前面加一段可学习的“虚拟 token”）、adapters（在层之间插入小的瓶颈网络）。

### 5.3 LoRA 应该加在哪里

- **原论文**：只在 attention 的 $W_Q$、$W_V$ 上实验
- **现在的建议**（Schulman et al., 2025，“LoRA Without Regret”）：加在**所有**线性层上，尤其是 **MLP / FFN 层，这是最重要的位置**（模型的大部分参数和“知识”在 FFN 中）

### 5.4 训练动态的注意点

“LoRA Without Regret” 的经验观察：
- LoRA 需要比全量微调**更高的学习率**（约 10 倍）
- LoRA 在**很大的 batch size** 下表现比全量微调差
- 在中小规模的数据集上，配置得当的 LoRA 可以达到与全量微调相同的效果

### 5.5 QLoRA（Dettmers et al., 2023）

**思路**：LoRA 中冻结的 $W_0$ 仍然占大量显存（7B 模型 BF16 约 13.5 GB，65B 约 130 GB）。QLoRA 把所有**冻结的权重量化**到 4-bit 存储：
- $W_0$：以 4-bit 存储
- LoRA 的 $A$、$B$：全精度（BF16）存储和训练
- 计算时：把 $W_0$ 的一小块**临时反量化**回 BF16，再与输入相乘（计算仍在高精度下进行）

**技巧 1：4-bit NormalFloat（NF4）**
- 4 bit 只能表示 16 个值，怎么选这 16 个值最好？
- 普通 INT4 量化**均匀**切分取值范围
- 观察：神经网络的权重近似服从**均值为 0 的正态分布** —— 大部分值集中在 0 附近，很少有很大的值。均匀切分会浪费很多“格子”在几乎没有权重的尾部
- NF4 按**正态分布的分位数**切分：让每个区间中的权重数量（概率质量）相同。0 附近格子密，尾部格子稀，信息损失最小
- 量化时每 64 个权重为一组（block），用组内绝对值最大值把它们缩放到 [−1, 1]，再映射到最近的 NF4 取值

**技巧 2：Double quantization（双重量化）**
- 每个 block 需要存一个 FP32 的缩放常数（quantization constant），平均每个参数多占 $32 / 64 = 0.5$ bit
- QLoRA **把这些常数本身再量化一次**（量化到 8-bit，每 256 个常数一组）
- 每参数的额外开销从 0.5 bit 降到约 0.127 bit，节省约 0.37 bit / 参数（对 65B 模型约省 3 GB）

**技巧 3：Paged optimizers**：显存不够时，把优化器状态临时换出到 CPU 内存，避免显存峰值导致的 OOM。

**收益**：
- 显存大幅节省，可以在更小的 GPU 上微调，而且更快
- 在显存与质量之间取得更好的平衡（4-bit QLoRA 的效果接近 16-bit 全量微调）
- 数量级（LLaMA 65B）：微调所需显存从 780 GB 以上降到 48 GB 以下（约 **16 倍**）；65B 参数 × 0.5 字节 ≈ 33 GB，可以放进一张 48 GB 的 GPU
- double quantization 额外节省约 6% 的显存

---

## 6. 要点回顾

1. LLM 训练 = 预训练（next-token prediction，海量数据）+ 后训练（SFT、preference tuning）
2. 训练计算量 $C \approx 6ND$；loss 随 $N$、$D$、$C$ 按幂律下降；Chinchilla：算力最优约 20 tokens / 参数，考虑推理成本时会 over-train 小模型
3. 混合精度 + Adam 每参数约 16 字节；显存是训练的核心瓶颈
4. 数据并行复制模型；ZeRO 把 optimizer state、梯度、参数分片消除冗余；TP / PP / SP / CP / EP 把计算本身拆开
5. FlashAttention 通过 tiling + online softmax + 重计算减少 HBM 读写，是精确算法，显存从 $O(n^2)$ 降到 $O(n)$
6. BF16 范围与 FP32 相同，训练更稳定；参数主副本必须用 FP32，否则小更新会被舍入掉
7. SFT 用“输入 → 期望输出”对微调，loss 只算回答部分；数据质量比数量更重要；评估本身很难
8. LoRA 冻结原权重、只训练低秩矩阵 $BA$，可训练参数通常 < 1%，可合并、可热插拔；QLoRA 再把冻结权重量化为 NF4

---

## 7. 自测题

<details>
<summary>1. 用 6ND 估算：一个 8B 模型训练 15T tokens 需要多少 FLOPs？</summary>

6 × 8×10⁹ × 1.5×10¹³ = 7.2×10²³ FLOPs。
</details>

<details>
<summary>2. 按 Chinchilla 法则，一个 10B 参数的模型算力最优时应训练多少 tokens？为什么实际中常训练更多？</summary>

约 20 × 10B = 200B tokens。Chinchilla 只考虑训练成本最优；如果模型要被大量调用，小模型推理更便宜，所以宁可多花训练算力把小模型训练得更好（over-training）。
</details>

<details>
<summary>3. 一个 13B 模型用混合精度 + Adam 训练，不算 activations，需要多少显存？用 ZeRO-3 分到 32 张卡上，每卡多少？</summary>

13×10⁹ × 16 字节 = 208 GB。ZeRO-3 每卡约 208 / 32 = 6.5 GB（另加通信缓冲和 activations）。
</details>

<details>
<summary>4. FlashAttention 计算的 FLOPs 比标准 attention 更多，为什么反而更快？</summary>

Attention 的瓶颈是 HBM 读写，而不是计算。FlashAttention 分块在 SRAM 中计算，从不把 n×n 的 S、P 矩阵写回 HBM，backward 时宁可重算也不读取，大幅减少了内存流量。
</details>

<details>
<summary>5. 为什么混合精度训练要保留一份 FP32 的参数主副本？</summary>

每步的参数更新量通常很小。BF16 / FP16 在 1.0 附近能表示的最小间隔约为 0.008 / 0.001，小于这个间隔的更新会被直接舍入掉。在 FP32 中累加更新才能保证小更新不丢失。
</details>

<details>
<summary>6. LoRA 中为什么把 B 初始化为 0？</summary>

这样训练开始时 BA = 0，模型的输出与原预训练模型完全相同，微调从一个已知良好的起点平滑出发，不会一开始就被随机扰动破坏。
</details>

<details>
<summary>7. NF4 和普通 INT4 量化的区别是什么？为什么 NF4 更适合神经网络权重？</summary>

INT4 在取值范围内均匀地放 16 个格子；NF4 按正态分布的分位数放格子，让每个格子里的权重数量相同。权重近似正态分布，大部分集中在 0 附近，NF4 在 0 附近格子更密，量化误差更小。
</details>

<details>
<summary>8. SFT 时为什么 loss 只算在回答部分？</summary>

目标是让模型学会如何回答，而不是如何生成用户的提问。如果对 prompt 也计算 loss，模型会花能力去模仿用户输入，还可能学会续写用户的话。
</details>

---

## 参考论文

- Brown et al., “Language Models are Few-Shot Learners” (GPT-3), 2020；LLaMA team, “The Llama 3 Herd of Models”, 2024
- Kaplan et al., “Scaling Laws for Neural Language Models”, 2020
- Hoffmann et al., “Training Compute-Optimal Large Language Models” (Chinchilla), 2022
- Kingma et al., “Adam: A Method for Stochastic Optimization”, 2014；Loshchilov et al., “Decoupled Weight Decay Regularization”, 2017
- Rajbhandari et al., “ZeRO: Memory Optimizations Toward Training Trillion Parameter Models”, 2019
- Shoeybi et al., “Megatron-LM”, 2019；Huang et al., “GPipe”, 2019；HuggingFace, “The Ultra-Scale Playbook”, 2025
- Dao et al., “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness”, 2022
- Micikevicius et al., “Mixed Precision Training”, 2017
- Wei et al., “Finetuned Language Models are Zero-Shot Learners” (FLAN), 2022；Zhou et al., “LIMA: Less Is More for Alignment”, 2023
- Dominguez-Olmedo et al., “Training on the Test Task Confounds Evaluation and Emergence”, 2024；Huang et al., “Exploring and Mitigating Adversarial Manipulation of Voting-Based Leaderboards”, 2025
- Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models”, 2021；Schulman et al., “LoRA Without Regret”, 2025
- Dettmers et al., “QLoRA: Efficient Finetuning of Quantized LLMs”, 2023
