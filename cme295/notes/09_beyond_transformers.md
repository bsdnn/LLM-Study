# Lecture 9 · 超越文本 Transformer：ViT、VLM、Diffusion LLM 与展望

> Stanford CME 295：Transformers & Large Language Models
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture9.pdf`](../slides/fall25-cme295-lecture9.pdf)

## 本讲脉络

```
整学期回顾（Lecture 1–8）
→ Transformer 用于图像：ViT → 图文结合：VLM（两种做法）→ Transformer 无处不在
→ 自回归的局限 → Diffusion：图像上的直觉（DDPM）→ 文本上的 Masked Diffusion（MDM / LLaDA）
→ 模态之间的相互借鉴 → 基础研究仍在进行（设计选择、数据、架构、硬件）
→ LLM 对日常生活的影响与未解决的挑战 → 如何跟进前沿
```

一句话总结：**Transformer 不只用于文本**——把图像切成 patch 当作“词”，就得到 ViT；把图像 token 接进 LLM，就得到 VLM。另一方面，**生成方式也不一定是自回归**：借鉴图像扩散模型，diffusion LLM 从全是 `[MASK]` 的序列出发，并行地逐步“揭开”token，生成速度可达自回归的约 10 倍。

---

## 1. 整学期回顾

| 讲次 | 主题 | 核心内容 |
|---|---|---|
| 1 | Transformers | Tokenization、embedding、self-attention（Q/K/V）、encoder-decoder 架构 |
| 2 | Transformer 模型与技巧 | RoPE、GQA、Pre-Norm/RMSNorm、BERT |
| 3 | LLM | MoE、解码策略、prompting、KV cache 等推理优化 |
| 4 | LLM 训练 | Scaling laws、Chinchilla、FlashAttention、并行与混合精度、SFT、LoRA |
| 5 | LLM 调优 | Preference tuning：reward model（Bradley-Terry）、RLHF/PPO、DPO |
| 6 | LLM 推理 | CoT → reasoning model、GRPO、DeepSeek-R1 |
| 7 | Agentic LLMs | RAG、tool calling、ReAct agents |
| 8 | LLM 评估 | LLM-as-a-Judge、benchmarks（MMLU、AIME、PIQA、SWE-bench、HarmBench） |

---

## 2. Transformer 能用于别的东西吗？

### 2.1 回顾

- 2017 年为机器翻译提出，依赖 self-attention，核心概念是 query、key、value
- Attention 本身**并不关心输入是什么**：只要能把输入表示成一组向量，就能用 attention 让它们互相交流

**好处**：
- **Inductive bias（归纳偏置）更弱**：CNN 内置了“局部性”（只看邻近像素）和“平移等变性”的假设；Transformer 没有这些假设，任意两个位置都能直接交互
- **泛化性更强**：一种架构适用于多种数据类型
- 代价：没有先验假设，需要**更多数据**才能学好

### 2.2 Vision Transformer（ViT）

论文：Dosovitskiy et al., 2020，“An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale”

**问题**：如何把 Transformer 用于图像理解（如图像分类）？

**做法**（端到端例子：识别一张 teddy bear 图片）：

1. **切 patch**：把 H × W × C 的图像切成 N 个 P × P 的小块。例如 224 × 224 的图、P = 16 → 14 × 14 = **196 个 patch** —— 每个 patch 就像一个“词”
2. **展平 + 线性投影**：每个 patch 展平为长度 P²·C 的向量（16 × 16 × 3 = 768），再用一个线性层投影到 D 维 → patch embedding
3. **加 `[CLS]` token**：在序列最前面加一个可学习的 `[CLS]` embedding（与 BERT 相同的思路）
4. **加 position embedding**：得到 position-aware embedding（否则模型不知道 patch 的空间位置）
5. **送入 Transformer encoder**：得到 encoded embeddings，每个 patch 的表示都融合了整张图的信息
6. **分类头**：取 `[CLS]` 的输出，经过 FFN → 投影 → 类别概率 → “teddy bear”

ViT 在足够大的数据上预训练后，效果可以超过 CNN。

### 2.3 VLM（Vision Language Model）

**任务**：输入图片 + 文字问题，输出文字回答。例：给一张 teddy bear 的图，问 “How cute is this teddy bear?” → “Very cute!”

**方法 1：复用 decoder-only 架构**（如 LLaVA，Liu et al., 2023，“Visual Instruction Tuning”）
- 用视觉编码器（如 ViT）把图像编码成一串向量
- 通过一个投影层（projection）把它们映射到 LLM 的 embedding 空间，变成“图像 token”
- 把图像 token 和文本 token **拼接**成一个序列，交给一个“普通”的 LLM
- 优点：简单，LLM 结构不用改；缺点：图像 token 占用大量 context

**方法 2：利用 cross-attention 层**（如 Llama 3 的视觉版本，“The Llama 3 Herd of Models”, 2024）
- LLM 的 decoder 中插入 **cross-attention 层**：query 来自文本，key/value 来自图像特征
- 图像信息通过 cross-attention “注入”，不占用文本序列的长度
- 类似原始 Transformer 中 decoder 关注 encoder 输出的方式

### 2.4 Transformer 无处不在

- **文本生成**：本课程（CME 295）
- **视觉理解**：Vision Transformer（ViT）
- **图像生成**：Diffusion Transformer（DiT）、MultiModal-DiT（MM-DiT）等
- 还有推荐系统、语音等很多领域

---

## 3. Diffusion LLM

### 3.1 当前方法及其局限

**范式**：AutoRegressive Modeling（ARM，自回归建模）。“[BOS] A teddy bear is” → 预测下一个 token → 接上 → 再预测……

**问题**：**推理时的生成无法并行**（尽管训练可以并行，因为训练时目标序列已知）。生成 N 个 token 就必须做 N 次 forward pass。

### 3.2 新闻中的替代方案

- Google 在 2025 年 5 月 20 日的 I/O 大会上展示了 diffusion-based LLM（Gemini Diffusion）
- 创业公司 Inception 推出基于 diffusion 的 LLM（Mercury）
- ByteDance 于 2025 年 7 月 31 日发布了 diffusion 语言模型（Seed Diffusion）

### 3.3 背景：图像生成模型的直觉

- **噪声很容易采样**（例如标准正态分布）
- 学习一个从噪声到数据的**变换**
- 数学上定义清晰

**类比：雕塑**

> “The sculpture is already complete within the marble block, before I start my work. It is already there, I just have to chisel away the superfluous material.” —— Michelangelo
> （雕像在我动工之前就已经在大理石中完整存在了，我只需凿去多余的部分。）

**高层目标**：学习某种变换，把噪声分布变成期望的数据分布。

### 3.4 图像上的 Diffusion（DDPM，Ho et al., 2020）

**1. Forward process（加噪）**：逐步向图像加高斯噪声，T 步后变成纯噪声

```
q(x_t | x_{t−1}) = N( √(1 − β_t) · x_{t−1},  β_t · I )
可以一步直接采样：x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε,   ε ~ N(0, I),  ᾱ_t = Π_{s≤t} (1 − β_s)
```

**2. Reverse process（去噪）**：训练一个神经网络，学习每一步怎样去掉一点噪声

```
训练目标：让网络预测被加进去的噪声
L = E_{x_0, t, ε} [ ‖ ε − ε_θ(x_t, t) ‖² ]
```

**生成**：从纯噪声 x_T 出发，用训练好的网络反复去噪，最终得到一张新图像。

### 3.5 从图像到文本

- 图像是**连续**的：可以加高斯噪声
- 文本是**离散**的：给 token ID 加高斯噪声没有意义
- 解决办法：用 **`[MASK]` 充当“噪声”**。“加噪” = 把 token 替换成 `[MASK]`；“纯噪声” = 全是 `[MASK]`

### 3.6 文本上的 Diffusion：MDM（Masked Diffusion Model）

**1. Forward process（加噪）**：以某个概率 t 把每个 token 独立地替换为 `[MASK]`
- t = 0：原句 “A cute teddy bear is reading”
- t = 0.5：“A [MASK] teddy [MASK] is [MASK]”
- t = 1：全是 `[MASK]`

**2. Reverse process（去噪）**：学习把 `[MASK]` 还原成 token

**训练**（以 LLaDA 为例，Nie et al., 2025）：
- 网络是一个**不带 causal mask 的 Transformer**（每个位置都能看到整个序列，和 BERT 一样是双向的）
- 随机采样 t ∈ (0, 1]，按概率 t 遮盖 token，让模型预测所有被遮住的 token
- Loss 只在被 mask 的位置上计算 cross-entropy，并按 1/t 加权：

```
L(θ) = − E_{t, x_0, x_t} [ (1/t) · Σ_i 1[x_t^i = MASK] · log p_θ(x_0^i | x_t) ]
```

可以看作 BERT 的 MLM 的推广：BERT 固定遮 15%，MDM 的遮盖比例从 0 到 100% 随机变化，因此能从全 mask 开始生成。

**推理（生成）**：
1. 从一个固定长度、**全是 `[MASK]`** 的序列开始（拼在 prompt 后面）
2. 一次 forward pass **同时预测所有被 mask 的位置**
3. 保留一部分预测（例如最有把握的那些），把其余的重新 mask 掉
4. 重复若干步，直到没有 `[MASK]`

关键：步数可以**远少于**要生成的 token 数 —— **用更少的 forward pass 完成解码**。

### 3.7 讨论

**优点**：
- ⚡ 输出速度约为自回归模型的 **10 倍**（tokens / 秒）
- 💻 更适合某些任务：例如填空（infilling）、代码编辑、需要全局规划的生成。因为是双向的，可以同时考虑前后文

**挑战与当前工作**：
- **效果**：整体质量仍在追赶同规模的自回归模型
- **移植 ARM 上的技术**：KV cache、RL / preference tuning、CoT 推理、变长输出等，都是为自回归设计的，需要重新适配

推荐阅读：
- Lou et al., “Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution”, 2023
- Sahoo et al., “Simple and Effective Masked Diffusion Language Models”, 2024
- Nie et al., “Large Language Diffusion Models” (LLaDA), 2025

---

## 4. 模态之间的相互借鉴（Cross-Pollination）

**趋势**：不同模态互相借用行之有效的想法。

| 方面 | 例子 |
|---|---|
| **架构** | 文本借用图像的 **diffusion 训练**（LLaDA）；图像借用文本的 **Transformer**（DiT，Peebles et al., 2022：用 Transformer 替代 U-Net 作为扩散模型的去噪网络） |
| **输入表示** | **DeepSeek-OCR**（Wei et al., 2025，“Contexts Optical Compression”）：把长文本**渲染成图像**，用少量视觉 token 表示大量文字，实现 context 压缩 |
| **技巧** | **RoPE 的变体用于图像**：如 Qwen-Image 的 Multimodal Scalable RoPE（MSRoPE），让位置编码同时处理二维图像位置和文本位置 |

---

## 5. 回到基础：基础研究仍在进行

### 5.1 微观层面：设计选择仍然五花八门

不同论文和模型在很多选择上并没有共识：
- **Optimizer**：AdamW vs MuonClip（Kimi K2 使用）vs 其他
- **Normalization**：用哪种、放在哪里
- **MHA / MQA / GQA** 的选择
- **Activation function**（如 ReLU、GELU、SwiGLU）
- **是否用 MoE**
- **层数**等

### 5.2 “燃料”：未来的高质量数据从哪来？

- 互联网上的高质量人类文本是有限的
- 越来越多的网络内容由 AI 生成
- **Model collapse**（Shumailov et al., 2023，“The Curse of Recursion: Training on Generated Data Makes Models Forget”）：如果一代代模型都在前一代生成的数据上训练，模型会逐渐“遗忘”数据分布的尾部（罕见但重要的内容），分布越来越窄，质量退化

### 5.3 退一步看：Transformer 是最好的架构吗？

- 人们一直有兴趣探索新的前沿（如 state-space models、linear attention 等）
- **到目前为止**：关注**最好的效果**
- **越来越多地**：关注**最佳的质量 / 成本权衡**（Pareto frontier：在给定成本下质量最高，或在给定质量下成本最低）

### 5.4 硬件优化

**观察**：当前 GPU 针对**矩阵-向量 / 矩阵-矩阵运算**优化。

**挑战**：Transformer block 有特殊需求：
- Attention 需要**频繁读写 KV cache**
- 在 GPU 上，**数据搬运（memory movement）占主导成本**，而不是计算本身

**想法**：在硬件中“原生”支持 attention 运算（Leroux et al., 2025，analog in-memory computing）：
- 把 KV cache 存在**专用存储单元**中，计算直接在存储的地方进行（in-memory computing），不用来回搬数据
- 用**模拟信号**完成计算

**结果**：相比 NVIDIA H100，延迟最多降低约 **100 倍**，能耗最多降低约 **70,000 倍**。

---

## 6. LLM 与日常生活

### 6.1 现在

LLM 已经极大地改变了日常生活，主要应用：
- **写代码**（真正的编程，以及 text-to-query，如自然语言转 SQL）
- **通用对话助手**（常识问答、网页搜索）
- **创意**
- **学习**
- ……

### 6.2 未来可以期待

- **明天**：现有 agent 的普及（如 2025 年 12 月 Google 推出 Workspace Studio，让普通用户用 AI agent 自动化日常工作）
- **近期**：浏览器（如 2025 年 10 月 OpenAI 推出 ChatGPT Atlas 浏览器）；更进一步，**操作系统级别的 LLM**？
- **长期**：“真正”自主、承担大量职责的 agents？终于“真正有用”的客服？

### 6.3 仍待解决的挑战

- **权重固定，不会更新**（≠ continual learning，持续学习）：模型部署后不能从新经验中学习
- **Hallucination（幻觉）**
- **Personalization（个性化）**
- **Interpretability（可解释性）**：我们仍不太清楚模型内部是怎么工作的

---

## 7. 如何跟进 NLP 前沿

**论文**：
- arXiv > Computer Science > Computation and Language（cs.CL）
- 通用 ML 会议：NeurIPS、ICML、ICLR；NLP 会议：ACL、EMNLP

**代码**：
- 论文中链接的作者 GitHub 仓库
- Hugging Face 的 “trending papers”：浏览最新的数据集和方法

**其他**：
- Twitter / X（学者和业界领袖）
- YouTube：理论向（Two Minute Papers、Yannic Kilcher）和实践向（Google Developers、HuggingFace、Andrej Karpathy）
- 公司 / 学术机构的技术报告和博客：Amazon Science、Anthropic、Apple ML、Google DeepMind、Meta AI、Microsoft AI、OpenAI、Stanford NLP……

**课程资源**：
- VIP Cheatsheet（有中文等多种语言版本）：https://github.com/afshinea/stanford-cme-295-transformers-large-language-models
- 教材 *Super Study Guide: Transformers & Large Language Models*：https://superstudy.guide

---

## 参考论文

- Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale” (ViT), 2020
- Liu et al., “Visual Instruction Tuning” (LLaVA), 2023；Llama Team, “The Llama 3 Herd of Models”, 2024
- Ho et al., “Denoising Diffusion Probabilistic Models”, 2020
- Lou et al., 2023；Sahoo et al., 2024；Nie et al., “Large Language Diffusion Models” (LLaDA), 2025
- Peebles et al., “Scalable Diffusion Models with Transformers” (DiT), 2022
- Wei et al., “DeepSeek-OCR: Contexts Optical Compression”, 2025；Wu et al., “Qwen-Image Technical Report”, 2025
- Kimi Team, “Kimi K2: Open Agentic Intelligence”, 2025
- Shumailov et al., “The Curse of Recursion: Training on Generated Data Makes Models Forget”, 2023
- Leroux et al., “Analog in-memory computing attention mechanism for fast and energy-efficient large language models”, 2025
