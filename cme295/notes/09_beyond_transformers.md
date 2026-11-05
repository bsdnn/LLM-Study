# Lecture 9 · 超越文本 Transformer：ViT、VLM、Diffusion LLM 与展望

> Stanford CME 295：Transformers & Large Language Models（Afshine Amidi & Shervine Amidi，Fall 2025）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture9.pdf`](../slides/fall25-cme295-lecture9.pdf)
> 预计阅读时间：18–22 分钟。预备知识：Lecture 1–3（self-attention、BERT 的 MLM、自回归解码）。

## 本讲脉络

```
整学期回顾（Lecture 1–8）
→ Transformer 用于图像：为什么可以、inductive bias 的取舍 → ViT（把图像切成 patch 当作词）
→ 图文结合：VLM 的两种架构（拼接 token / cross-attention）→ Transformer 无处不在
→ 自回归生成的局限 → Diffusion：图像上的直觉与 DDPM → 文本上的 masked diffusion（训练、生成、LLaDA）→ 优缺点
→ 模态之间相互借鉴：架构、输入表示（DeepSeek-OCR）、技巧（RoPE 变体）
→ 基础研究仍在进行：设计选择、数据（model collapse）、架构、硬件
→ LLM 与日常生活、尚未解决的挑战、如何跟进前沿
```

**一句话总结**：Transformer 不只用于文本 —— 把图像切成小块当作“词”，就得到了 **ViT**；把图像变成 token 接进 LLM，就得到了能看图说话的 **VLM**。另一方面，生成方式也不一定是“一个接一个”的自回归：借鉴图像扩散模型，**diffusion LLM** 从一串全是 `[MASK]` 的序列出发，**并行地、分几步把 token“揭开”**，生成速度可达自回归的约 10 倍。最后，这门课的结论是：基础研究远未结束，从优化器到数据到硬件都还在快速变化。

---

## 1. 整学期回顾

| 讲次 | 主题 | 核心内容 |
|---|---|---|
| 1 | Transformers | Tokenization、embedding、self-attention（Q / K / V）、encoder-decoder 架构、端到端翻译例子 |
| 2 | 基于 Transformer 的模型与技巧 | RoPE、GQA、Pre-Norm / RMSNorm、encoder-only 的 BERT |
| 3 | LLM | MoE（router + experts）、解码策略、prompting、KV cache 等推理优化 |
| 4 | LLM 训练 | Scaling laws、Chinchilla、FlashAttention（SRAM 小而快，HBM 大而慢）、并行与混合精度、SFT、LoRA |
| 5 | LLM 调优 | Preference tuning：Bradley-Terry reward model、RLHF（PPO：最大化 reward、不偏离 base model 太远）、DPO |
| 6 | LLM 推理 | CoT → reasoning model（Question → Reasoning chain → Answer）、GRPO vs PPO、DeepSeek-R1 |
| 7 | Agentic LLMs | RAG、tool calling（模型生成 API 参数）、ReAct agents |
| 8 | LLM 评估 | LLM-as-a-Judge（prompt + 回答 + 准则 → 理由 + 分数）、benchmarks（MMLU、AIME、PIQA、SWE-bench、HarmBench） |

---

## 2. Transformer 能用于别的东西吗？

### 2.1 回顾与思路

- Transformer 在 2017 年为机器翻译提出，依赖 self-attention，核心概念是 query、key、value
- **Attention 本身并不关心输入是什么**：只要能把输入表示成一组向量（token），就能用 attention 让它们互相交流

**Transformer 的好处**：
- **Inductive bias（归纳偏置）更弱**
- **泛化性更强**：一种架构适用于多种数据类型

**什么是 inductive bias**：模型结构中预先“写死”的假设。CNN 有很强的 inductive bias：
- **局部性**：卷积核只看邻近像素
- **平移等变性**：同一个卷积核在图像各处共享，猫在左上角和右下角被同样识别

这些假设对图像很合理，让 CNN 用较少的数据就能学好。Transformer 没有这些假设：任意两个位置都能直接交互，位置关系要从数据中学。**代价是需要更多数据；回报是数据足够多时，上限更高、更通用**。

### 2.2 能不能直接把像素当 token？

不行。一张 224 × 224 的图有 50,176 个像素，self-attention 要计算 $50{,}176^2 \approx 25$ 亿对关系，太贵了。解决方法：**把图像切成小块（patch），每块当作一个 token**。

### 2.3 Vision Transformer（ViT）

论文：Dosovitskiy et al., 2020，“An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”（一张图值 16×16 个词）

**问题**：如何把 Transformer 用于图像理解任务（如图像分类）？

**端到端流程**（识别一张 teddy bear 的图片）：

1. **切 patch**：把 $H \times W \times C$ 的图像（$C$ = 颜色通道数，RGB 为 3）切成 $N = \frac{HW}{P^2}$ 个 $P \times P$ 的小块。例：224 × 224 的图、$P = 16$ → $14 \times 14 = $ **196 个 patch**
2. **展平 + 线性投影**：每个 patch 展平成长度 $P^2 C$ 的向量（$16 \times 16 \times 3 = 768$），再用一个线性层投影到 $D$ 维 → patch embedding（等价于一个卷积核大小和步长都等于 $P$ 的卷积）
3. **加 `[CLS]` token**：在序列最前面加一个可学习的 `[CLS]` embedding（与 BERT 相同的思路），用来汇总整张图的信息
4. **加 position embedding**：得到 position-aware embedding（否则模型不知道每个 patch 在图像中的位置）
5. **送入 Transformer encoder**：得到 encoded embeddings，每个 patch 的表示都融合了整张图的信息（第一层就能看到全图，而 CNN 要堆很多层才能扩大感受野）
6. **分类头**：取 `[CLS]` 的输出 → FFN → 投影到类别数 → softmax → “teddy bear”

**模型规模**：ViT-Base 约 8600 万参数（12 层、768 维），ViT-Large 约 3.07 亿，ViT-Huge 约 6.32 亿。

**关键发现**：只在 ImageNet（约 130 万张图）上训练时，ViT 比同等规模的 CNN（ResNet）差几个百分点 —— inductive bias 不足；但在更大的数据集（如 3 亿张图的 JFT-300M）上预训练后，ViT 反超 CNN。**数据足够多时，“学出来的”结构胜过“写死的”结构。**

### 2.4 VLM（Vision Language Model，视觉语言模型）

**任务**：输入图片 + 文字问题，输出文字回答。例：给一张 teddy bear 的照片，问 “How cute is this teddy bear?” → “Very cute!”

**视觉编码器**：通常用一个预训练好的 ViT，最常见的是 **CLIP** 的图像编码器。CLIP 用约 4 亿对网络图文数据做**对比学习**：让配对的图像和文字 embedding 靠近，不配对的远离。所以它的图像特征天然与语言的语义对齐。

**方法 1：复用 decoder-only 架构**（如 LLaVA，Liu et al., 2023，“Visual Instruction Tuning”）

```
图像 → 视觉编码器（CLIP ViT）→ 投影层（线性层或小 MLP）→ 一串“图像 token”
                                                          ↓
                         [图像 token] + [文字 token] 拼成一个序列 → “普通”的 LLM → 回答
```

- 投影层把视觉特征映射到 LLM 的 embedding 空间，让 LLM 把它们当作“外语单词”来读
- 例：336 × 336 的图、patch 大小 14 → $24 \times 24 = 576$ 个图像 token
- 两阶段训练：
  1. **特征对齐**：冻结视觉编码器和 LLM，只训练投影层，用约 60 万对图像-描述数据
  2. **视觉指令微调**：用约 15.8 万条由 GPT-4 生成的图文指令数据（关于图像的对话、描述、推理问答），训练投影层和 LLM
- 优点：简单，LLM 的结构完全不用改
- 缺点：图像 token 占用大量 context（一张图几百个 token，多张图更多）

**方法 2：利用 cross-attention 层**（如 LLaMA 3 的视觉版本，“The Llama 3 Herd of Models”, 2024）

- 在 LLM 的 decoder 层之间插入 **cross-attention 层**：**query 来自文本，key / value 来自图像特征**
- 图像信息通过 cross-attention “注入”文本的表示中，**不占用文本序列的长度**
- 与原始 Transformer 中 decoder 关注 encoder 输出的方式一样（Lecture 1）
- 训练时可以冻结原 LLM 的参数，只训练新加的 cross-attention 层，从而**完整保留纯文本能力**

| | 拼接 token（LLaVA 式） | Cross-attention（LLaMA 3 视觉式） |
|---|---|---|
| 是否修改 LLM 结构 | 否 | 是，插入新层 |
| 图像是否占用 context | 是 | 否 |
| 对纯文本能力的影响 | 训练时可能被影响 | 可以冻结原 LLM，不影响 |
| 实现难度 | 简单 | 较复杂 |

### 2.5 Transformer 无处不在

- **文本生成**：本课程（CME 295）
- **视觉理解**：Vision Transformer（ViT）
- **图像生成**：Diffusion Transformer（DiT）、MultiModal-DiT（MM-DiT）等
- 还有很多其他领域：推荐系统、语音等

---

## 3. Diffusion LLM

### 3.1 当前方法及其局限

**范式**：AutoRegressive Modeling（ARM，自回归建模）。“[BOS] A teddy bear is” → 预测下一个 token → 接上 → 再预测……

**问题**：**推理时的生成无法并行**（尽管训练可以并行，因为训练时整个目标序列已知，配合 causal mask 一次算完）。生成 $N$ 个 token 就必须做 $N$ 次 forward pass。而每次 forward 都要把全部权重从显存读一遍（Lecture 3：decode 是 memory-bound 的），**延迟与输出长度成正比**。

### 3.2 新闻中的替代方案

- Google 在 2025 年 5 月 20 日的 I/O 大会上展示了基于 diffusion 的 LLM（Gemini Diffusion）
- 创业公司 Inception 推出了基于 diffusion 的 LLM（Mercury 系列），2025 年 11 月有相关新闻报道
- ByteDance 于 2025 年 7 月 31 日发布了 diffusion 语言模型（Seed Diffusion）
- 这些模型宣称能达到**每秒上千 token** 的生成速度

### 3.3 背景：图像生成模型的直觉

为什么从**噪声**开始生成？
- **噪声很容易采样**（例如标准正态分布，随手就能生成）
- 我们学习一个从噪声到数据的**变换**
- **数学上定义清晰**（有严格的概率框架）

**类比：雕塑**

> “The sculpture is already complete within the marble block, before I start my work. It is already there, I just have to chisel away the superfluous material.” —— Michelangelo
> （雕像在我动工之前就已经完整地在大理石里了，我只需要凿去多余的部分。）

**高层目标**：学习某种变换，把**噪声分布**变成**期望的数据分布**（例如“真实照片”的分布）。

### 3.4 图像上的 Diffusion（DDPM，Ho et al., 2020）

**1. Forward process（加噪过程）**：逐步向图像加入少量高斯噪声，经过 $T$ 步（如 1,000 步）后变成纯噪声。

$$
q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1 - \beta_t}\; x_{t-1},\; \beta_t I\big)
$$

$\beta_t$ 是每步加入的噪声量（noise schedule）。好的性质：可以**一步直接**得到任意第 $t$ 步的结果：

$$
x_t = \sqrt{\bar\alpha_t}\; x_0 + \sqrt{1 - \bar\alpha_t}\; \epsilon,\qquad \epsilon \sim \mathcal{N}(0, I),\quad \bar\alpha_t = \prod_{s=1}^{t}(1 - \beta_s)
$$

**2. Reverse process（去噪过程）**：训练一个神经网络，学习每一步怎样去掉一点噪声。最简单有效的做法是**让网络预测被加进去的噪声**：

$$
\mathcal{L} = \mathbb{E}_{x_0,\, t,\, \epsilon}\Big[\big\|\epsilon - \epsilon_\theta(x_t, t)\big\|^2\Big]
$$

训练：随机取一张图 $x_0$、一个步数 $t$、一个噪声 $\epsilon$，按上式得到 $x_t$，让网络从 $x_t$ 和 $t$ 中猜出 $\epsilon$。

**生成**：从纯噪声 $x_T$ 出发，用训练好的网络反复“去掉一点噪声”，一步步得到 $x_{T-1}, x_{T-2}, \dots, x_0$ —— 一张全新的图像。

**直觉**：“从零画出一张照片”很难；但“给一张有点模糊的照片去掉一点噪声”容易得多。把难题拆成很多步简单的去噪，每一步只做小修正。

### 3.5 从图像到文本

- 图像是**连续**的：每个像素值可以加上一点点高斯噪声
- 文本是**离散**的：token ID 是整数，给 “teddy” 的 ID 加 0.3 没有意义
- 解决办法：**用 `[MASK]` 充当“噪声”**。“加噪” = 把 token 替换成 `[MASK]`；“纯噪声” = 全部是 `[MASK]`

### 3.6 文本上的 Diffusion：MDM（Masked Diffusion Model）

**1. Forward process（加噪）**：取一个比例 $t \in [0, 1]$，把每个 token **独立地**以概率 $t$ 替换为 `[MASK]`。

| $t$ | 序列 |
|---|---|
| 0 | A cute teddy bear is reading |
| 0.5 | A [MASK] teddy [MASK] is [MASK] |
| 1 | [MASK] [MASK] [MASK] [MASK] [MASK] [MASK] |

**2. Reverse process（去噪）**：学习把 `[MASK]` 还原成 token。

**训练**（以 LLaDA 为例，Nie et al., 2025，“Large Language Diffusion Models”）：
- 网络是一个**不带 causal mask 的 Transformer**：每个位置都能看到整个序列（像 BERT 一样是双向的）
- 每条训练数据：随机采样 $t$，按概率 $t$ 遮盖 token，让模型**同时预测所有被遮住的 token**
- Loss 只在被 mask 的位置上计算 cross-entropy，并按 $1/t$ 加权：

$$
\mathcal{L}(\theta) = -\,\mathbb{E}_{t,\, x_0,\, x_t}\Big[\frac{1}{t}\sum_{i=1}^{L} \mathbb{1}\big[x_t^i = \texttt{[MASK]}\big]\, \log p_\theta\big(x_0^i \mid x_t\big)\Big]
$$

**与 BERT 的关系**：可以看作 BERT 的 MLM 的推广。BERT 固定只遮 15%，所以只能“填空”；MDM 的遮盖比例从 0% 到 100% 随机变化，模型学会了“在任何遮盖程度下填空”，包括从**全部遮住**开始 —— 这就是生成。

**推理（生成）**：
1. 在 prompt 后面接一段固定长度、**全部是 `[MASK]`** 的序列
2. 一次 forward pass，**同时预测所有被 mask 的位置**
3. 只保留其中一部分预测（通常是模型**最有把握**的那些），把其余位置重新 mask 掉
4. 重复若干步，直到没有 `[MASK]`

**算例**：生成 6 个 token，分 3 步，每步揭开 2 个：

| 步骤 | 序列 | 这一步揭开了 |
|---|---|---|
| 0 | [M] [M] [M] [M] [M] [M] | — |
| 1 | [M] [M] teddy [M] [M] reading | 最有把握的 “teddy”、“reading” |
| 2 | A [M] teddy bear [M] reading | “A”、“bear” |
| 3 | A cute teddy bear is reading | “cute”、“is” |

**3 次 forward 生成了 6 个 token**，而自回归需要 6 次。步数可以远少于 token 数 —— **用更少的 forward pass 完成解码**。

实践中常用 **semi-autoregressive（半自回归）** 的方式：把输出分成若干块，块与块之间从左到右生成，块内用 diffusion 并行生成，兼顾质量与速度。

**LLaDA 的结果**：8B 参数、在 2.3T tokens 上从零训练，在多项 benchmark 上与 LLaMA 3 8B 相当。因为是双向建模，它在“**reversal curse（逆转诅咒）**”问题上表现更好：自回归模型学了 “A 是 B” 却常常答不出 “B 是 A”；LLaDA 在一个“根据下一句补全上一句诗”的任务上甚至超过了 GPT-4o。

### 3.7 讨论

**优点**：
- ⚡ **速度**：输出速度约为自回归模型的 **10 倍**（tokens / 秒）
- 💻 **更适合某些任务**：因为是双向的，可以同时考虑前后文，适合**填空（infilling）**、**代码编辑**、需要**全局规划**的生成
- **可调节的速度-质量权衡**：步数越多质量越好，步数越少速度越快

**挑战与当前工作**：
- **效果**：整体质量仍在追赶同规模的顶尖自回归模型
- **移植 ARM 上的技术**：很多成熟的技术都是为自回归设计的：
  - **KV cache**：双向 attention 下，每一步任何位置都可能变化，之前算的 K、V 不能直接复用（需要近似的缓存方法）
  - **变长输出**：要预先定好生成长度
  - **RL、preference tuning、CoT 推理**：都需要重新适配
- 整个推理服务的软件栈（Lecture 3 中的各种优化）也都是围绕自回归构建的

---

## 4. 模态之间的相互借鉴（Cross-Pollination）

**趋势**：不同模态互相借用行之有效的想法。

| 方面 | 例子 | 说明 |
|---|---|---|
| **架构** | 文本 ← 图像：**diffusion 训练用于文本输出**（LLaDA） | 见第 3 节 |
| | 图像 ← 文本：**用 Transformer 处理图像**（DiT，Peebles et al., 2022） | 扩散模型的去噪网络原本是 U-Net（一种 CNN），DiT 换成了作用在图像 patch 上的 Transformer，并发现计算量越大、生成质量越好（与 LLM 的 scaling law 类似） |
| **输入表示** | **DeepSeek-OCR**（Wei et al., 2025，“Contexts Optical Compression”） | 把长文本**渲染成图像**，再用视觉 token 表示。在约 10 倍压缩率下（视觉 token 数是文本 token 数的 1/10），解码回文字的准确率约 97%；20 倍压缩时仍有约 60%。设想：可以用这种“光学压缩”存储很长的历史 context，越久远的内容压缩得越狠，类似人类记忆的逐渐模糊 |
| **技巧** | **RoPE 的变体用于图像**（如 Qwen-Image 的 Multimodal Scalable RoPE，MSRoPE） | 图像的位置是二维的（行、列），可以把 RoPE 的维度分成两半，分别编码横向和纵向位置；MSRoPE 进一步把图像和文本的位置放进同一个坐标体系，让模型同时处理图文 |

---

## 5. 回到基础：基础研究仍在进行

### 5.1 微观层面：设计选择仍然五花八门

不同论文和模型在很多选择上并没有共识：

| 设计选择 | 选项 |
|---|---|
| **Optimizer（优化器）** | AdamW vs **MuonClip** vs 其他。Muon 对矩阵参数的更新做“正交化”，让各个方向的更新幅度更均衡；Kimi K2 使用 MuonClip（Muon + 限制 attention 分数过大的 QK-clip），在 15.5T tokens 上训练了一个 1T 总参数（32B 激活）的 MoE 模型，全程没有出现 loss 突刺 |
| **Normalization** | 用哪种（LayerNorm / RMSNorm）、放在哪里（Pre / Post），是否对 Q、K 也做归一化 |
| **Attention head 设计** | MHA / MQA / GQA / MLA |
| **Activation function（激活函数）** | ReLU、GELU、SwiGLU 等 |
| **MoE 还是 dense** | 以及 expert 的数量、大小、是否有共享 expert |
| **层数与宽度** | 更深还是更宽 |

### 5.2 “燃料”：未来的高质量数据从哪来？

- 互联网上的高质量人类文本是**有限**的，前沿模型已经用掉了其中的大部分
- 越来越多的网络内容由 AI 生成
- **Model collapse（模型崩溃）**（Shumailov et al., 2023，“The Curse of Recursion: Training on Generated Data Makes Models Forget”）：如果一代代模型都在前一代模型生成的数据上训练，模型会逐渐**“遗忘”数据分布的尾部**（罕见但重要的内容），分布越来越窄，最终质量退化
- 直观例子：反复“用一组样本拟合一个正态分布 → 从拟合的分布中重新采样 → 再拟合”，方差会一代代缩小，最终塌缩成一个点。每一代采样都更难采到尾部的值，尾部就这样一点点消失了
- 应对：保留人类原创数据、追踪数据来源、对合成数据做严格验证和筛选（Lecture 6 中 reasoning 数据的拒绝采样就是一种“带验证的合成数据”）

### 5.3 退一步看：Transformer 是最好的架构吗？

- 人们一直在探索新的前沿（如 state-space model、linear attention、以及它们与 attention 的混合架构）
- **到目前为止**：关注**最好的效果**
- **越来越多地**：关注**最佳的质量 / 成本权衡**（Pareto frontier：给定成本下质量最高，或给定质量下成本最低）

### 5.4 硬件优化

**观察**：当前 GPU 针对**矩阵-向量 / 矩阵-矩阵运算**做了优化。

**挑战**：Transformer block 有特殊需求：
- Attention 需要**频繁读写 KV cache**
- 在 GPU 上，**数据搬运（memory movement）的成本占主导**，而不是计算本身（Lecture 3、4 反复出现的主题）

**想法**：在硬件中“原生”支持 attention 运算（Leroux et al., 2025，analog in-memory computing）：
- 把 KV cache 存在**专用的存储单元**中，计算直接在数据所在的地方完成（in-memory computing），不再来回搬运数据
- 用**模拟信号**完成乘加运算（利用电流、电压的物理规律直接“算”出点积），而不是数字电路

**结果**：在 attention 计算上，相比 NVIDIA H100，延迟最多降低约 **100 倍**，能耗最多降低约 **70,000 倍**（研究原型阶段）。

---

## 6. LLM 与日常生活

### 6.1 现在

LLM 已经极大地改变了日常生活，主要应用：
- **写代码**（真正的编程，以及 text-to-query，如把自然语言转成 SQL）
- **通用对话助手**（常识问答、网页搜索）
- **创意**（写作、头脑风暴）
- **学习**
- ……

### 6.2 未来可以期待

| 时间 | 可能的发展 |
|---|---|
| **明天** | 现有 agent 的**普及**：例如 2025 年 12 月 Google 推出 Workspace Studio，让普通用户也能用 AI agent 自动化日常工作 |
| **近期** | **浏览器**：例如 2025 年 10 月 OpenAI 推出 ChatGPT Atlas 浏览器；更进一步，**操作系统级别的 LLM**？ |
| **长期** | “真正”自主、承担大量职责的 agents？终于**真正有用的客服**？ |

### 6.3 仍待解决的挑战

| 挑战 | 说明 |
|---|---|
| **权重固定，不会更新**（≠ continual learning，持续学习） | 模型部署后不能从新经验中学习；直接在新数据上继续训练，又容易“灾难性遗忘”以前学会的东西 |
| **Hallucination（幻觉）** | 自信地编造不存在的事实 |
| **Personalization（个性化）** | 如何让模型适应每个用户的偏好和背景，而不需要为每个人单独训练 |
| **Interpretability（可解释性）** | 我们仍不太清楚模型内部是怎么工作的、某个回答为什么会产生 |

---

## 7. 如何跟进 NLP 前沿

**论文**：
- arXiv > Computer Science > Computation and Language（cs.CL）
- 通用机器学习会议：NeurIPS、ICML、ICLR；NLP 会议：ACL、EMNLP

**代码**：
- 论文中链接的作者 GitHub 仓库
- Hugging Face 的 “trending papers”：浏览最新的数据集和方法

**其他**：
- Twitter / X（学术界和业界的领军人物）
- YouTube：理论向（Two Minute Papers、Yannic Kilcher）和实践向（Google Developers、HuggingFace、Andrej Karpathy）
- 公司和学术机构的技术报告与博客：Amazon Science、Anthropic、Apple ML、Google DeepMind、Meta AI、Microsoft AI、OpenAI、Stanford NLP……

**课程资源**：
- VIP Cheatsheet（已有中文等多种语言版本）：https://github.com/afshinea/stanford-cme-295-transformers-large-language-models
- 教材 *Super Study Guide: Transformers & Large Language Models*：https://superstudy.guide

---

## 8. 要点回顾

1. Attention 不关心输入类型；Transformer 的 inductive bias 弱，需要更多数据，但上限更高、更通用
2. ViT：图像切成 $P \times P$ 的 patch → 线性投影 → 加 `[CLS]` 和位置编码 → encoder → 用 `[CLS]` 分类；大数据上超过 CNN
3. VLM 两种做法：把投影后的图像 token 拼进 LLM 的输入（LLaVA，简单但占 context），或在 LLM 中插入 cross-attention 层（LLaMA 3 视觉，不占 context、可保留文本能力）
4. 自回归生成 N 个 token 需要 N 次 forward，无法并行
5. 图像扩散：forward 逐步加高斯噪声，网络学习预测噪声，生成时从纯噪声反复去噪
6. 文本扩散用 `[MASK]` 当噪声：随机比例遮盖、双向 Transformer 同时预测所有被遮位置；生成时从全 mask 出发，每步揭开最有把握的一部分，步数远少于 token 数
7. Diffusion LLM 快、适合填空和编辑，但质量仍在追赶，KV cache、RL 等技术需要重新适配
8. 模态间相互借鉴（DiT、DeepSeek-OCR、多模态 RoPE）；优化器、数据（model collapse）、架构、硬件都仍在变化；持续学习、幻觉、个性化、可解释性仍未解决

---

## 9. 自测题

<details>
<summary>1. 一张 384 × 384 的 RGB 图，patch 大小 16，ViT 的输入序列有多少个 token？每个 patch 展平后多少维？</summary>

(384/16)² = 24² = 576 个 patch，再加 1 个 [CLS]，共 577 个 token。每个 patch 展平后 16 × 16 × 3 = 768 维。
</details>

<details>
<summary>2. 为什么 ViT 只在 ImageNet 上训练时不如 CNN，在更大的数据上却更好？</summary>

CNN 内置了局部性和平移等变性等 inductive bias，数据少时这些先验很有帮助；ViT 没有这些先验，要从数据中学出空间结构，数据少时学不好。数据足够多时，ViT 学到的结构更灵活，不受固定假设限制，上限更高。
</details>

<details>
<summary>3. LLaVA 式 VLM 和 cross-attention 式 VLM 各有什么优缺点？</summary>

LLaVA 式把图像 token 拼进输入序列，不用改 LLM 结构、实现简单，但图像占用大量 context。Cross-attention 式在 LLM 中插入新层，图像不占 context，并且可以冻结原 LLM 以保留纯文本能力，但需要修改架构、实现更复杂。
</details>

<details>
<summary>4. 文本为什么不能像图像那样加高斯噪声？文本扩散用什么代替？</summary>

文本 token 是离散的整数 ID，加一个小的连续噪声没有意义。文本扩散用 [MASK] 作为“噪声”：加噪就是把 token 替换成 [MASK]，完全加噪就是全部为 [MASK]。
</details>

<details>
<summary>5. Masked diffusion LM 与 BERT 的 MLM 有什么关系和区别？为什么前者能生成文本而后者不能？</summary>

两者都是双向 Transformer 预测被遮住的 token。BERT 固定只遮 15%，只学会了在大部分上下文已知时填空；MDM 的遮盖比例在 0–100% 间随机，学会了在任意遮盖程度下预测，包括全部遮住的情况，因此可以从全 [MASK] 开始逐步生成整段文本。
</details>

<details>
<summary>6. Diffusion LLM 为什么很难直接使用 KV cache？</summary>

KV cache 依赖于“之前 token 的表示不会再改变”，这在 causal mask 的自回归模型中成立。Diffusion LLM 是双向 attention，每一步任何位置（包括刚揭开的 token）都会影响所有其他位置的表示，之前计算的 K、V 在下一步就可能失效，不能直接复用。
</details>

<details>
<summary>7. 什么是 model collapse？用一句话解释它为什么会发生。</summary>

模型一代代在前代模型生成的数据上训练，逐渐丢失数据分布的尾部，输出越来越单一、质量退化。原因是每一代采样都更难采到罕见内容，罕见内容在下一代的训练数据中就更少，最终消失。
</details>

---

## 参考论文

- Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale” (ViT), 2020
- Radford et al., “Learning Transferable Visual Models From Natural Language Supervision” (CLIP), 2021
- Liu et al., “Visual Instruction Tuning” (LLaVA), 2023；Llama Team, “The Llama 3 Herd of Models”, 2024
- Ho et al., “Denoising Diffusion Probabilistic Models”, 2020
- Lou et al., “Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution”, 2023
- Sahoo et al., “Simple and Effective Masked Diffusion Language Models”, 2024
- Nie et al., “Large Language Diffusion Models” (LLaDA), 2025
- Peebles et al., “Scalable Diffusion Models with Transformers” (DiT), 2022
- Wei et al., “DeepSeek-OCR: Contexts Optical Compression”, 2025；Wu et al., “Qwen-Image Technical Report”, 2025
- Kimi Team, “Kimi K2: Open Agentic Intelligence”, 2025
- Shumailov et al., “The Curse of Recursion: Training on Generated Data Makes Models Forget”, 2023
- Leroux et al., “Analog in-memory computing attention mechanism for fast and energy-efficient large language models”, 2025
