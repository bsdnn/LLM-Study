# Lecture 5 · Preference Tuning：RLHF、PPO 与 DPO

> Stanford CME 295：Transformers & Large Language Models（Afshine Amidi & Shervine Amidi，Fall 2025）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture5.pdf`](../slides/fall25-cme295-lecture5.pdf)
> 预计阅读时间：18–22 分钟。预备知识：Lecture 4（pretraining、SFT），基础概率（sigmoid、期望、KL 散度）。

## 本讲脉络

```
为什么 SFT 之后还需要 preference tuning
→ 偏好数据：pointwise / pairwise / listwise，怎么收集
→ 把 LLM 写成强化学习问题：agent、state、action、policy、reward；policy gradient 的直觉
→ RLHF 第 1 步：reward model（Bradley-Terry）
→ RLHF 第 2 步：RL（KL 约束、reward hacking）→ PPO（advantage、value model、GAE、clip、KL penalty）
→ PPO 的替代（REINFORCE、RLOO、GRPO）与 RL 的难点 → Best-of-N
→ DPO：完整推导、梯度直觉、数值例子、变体 → PPO vs DPO
```

**一句话总结**：SFT 只告诉模型“该怎么说”，没有告诉它“哪些不该说”。**Preference tuning 用“回答 A 比回答 B 好”这种比较数据，把负面信号注入模型。** RLHF 先训练一个 reward model 给回答打分，再用 PPO 优化模型去拿高分，同时用 KL 约束防止它跑偏；DPO 则证明同一个目标可以改写成一个普通的监督学习 loss，直接在偏好对上训练，不需要 reward model 和强化学习。

---

## 1. 在 LLM 生命周期中的位置

```
初始化模型
  ──Pretraining──────────► 有“基础知识”的模型（语言、代码……）
  ──Finetuning（SFT）────► 针对特定任务调整的模型
  ──Preference tuning────► 与（人类）偏好对齐的模型      ← 本讲
```

这一步常被称为 **alignment（对齐）**：让模型的行为符合人类期望 —— 有帮助（helpful）、诚实（honest）、无害（harmless）。

---

## 2. 为什么需要 Preference Tuning

### 2.1 SFT 后的模型仍可能“乱来”

- Prompt：“Suggest a new activity I could do with my teddy bear.”
- SFT 模型可能的坏回答：“I'd suggest you do not spend much time with your teddy bear at all.”
- 期望的好回答：“Of course! Teddy bears not only make awesome companions for a delightful sleep, but can also be great buddies for fun activities. How about you both watch a movie together?”

SFT 的训练目标是**提高好回答的概率**（模仿示范），但它从来没有明确地**降低坏回答的概率**。坏回答在 SFT 之后可能仍然有不小的概率。我们需要**注入负面信号**。

**想法**：收集偏好对（preference pairs），在上面训练模型。

### 2.2 为什么用“偏好比较”而不是继续加 SFT 数据

- **比较比生成容易**：判断“A 比 B 好”远比从零写出一个完美的回答 A 容易。标注更快、更便宜，不同标注者之间也更一致。很多时候人自己写不出最好的答案，但能认出哪个更好
- **SFT 对数据分布非常敏感**，很容易“搞砸”：示范数据里的小问题（啰嗦、错误、风格不一致）会被模型忠实模仿
- **SFT 不易扩展**：高质量示范数据很重要，也很难大量获得

不过，模型“乱来”也可能是一个提醒：**应该回头检查 SFT 数据的质量**。

---

## 3. 偏好数据（Preference Data）

### 3.1 三种形式

一条观测 = (prompt $x$，response $y$)。

| 形式 | 做法 | 例子 | 特点 |
|---|---|---|---|
| **Pointwise** | 给每个回答打一个绝对分 | Obs 1 → 0.4，Obs 2 → 0.9，Obs 3 → 0.1 | 不同标注者的打分尺度难以统一 |
| **Pairwise** | 比较两个回答哪个更好 | $y_A \succ y_B$ | **最常用**，判断最容易、最一致 |
| **Listwise** | 对多个回答排序 | 1 > 2 > 3 > 4 | 一次标注能得到很多对比较 |

Listwise 可以拆成 pairwise：$K$ 个回答的排序包含 $\binom{K}{2}$ 对比较。例如 InstructGPT 让标注者对 4–9 个回答排序，9 个回答的排序就能产生 36 对。

### 3.2 收集 pairwise 偏好数据的流程

**第 1 步：为同一个 prompt 生成一对回答**
- **Prompt 来源**：线上真实用户日志、或按目标分布构造的参考 prompt
- **回答来源**：用 SFT 模型以不同的 temperature 采样多次；用不同的模型生成；合成数据；人工改写（例如把一个好回答改坏，或把坏回答改好）

**第 2 步：标注哪个更好**，得到 $y_w$（winner，被选中的）和 $y_l$（loser，被拒绝的）
- **人工评分**：按照详细的标注指南（有用性、真实性、无害性等）
- **代理方法**：LLM-as-a-judge（让强模型来判断）、BLEU / ROUGE（与参考答案比较）等
- 变体：**二元**尺度（更好 / 更差）或**更细致**的尺度（“明显更好 / 略好 / 差不多”）

最终的一条数据通常长这样：

```json
{"prompt": "Suggest a new activity I could do with my teddy bear.",
 "chosen": "Of course! ... How about you both watch a movie together?",
 "rejected": "I'd suggest you do not spend much time with your teddy bear at all."}
```

---

## 4. 把 LLM 写成强化学习问题

### 4.1 强化学习的基本框架

| 概念 | 含义 |
|---|---|
| **Agent** | 做决策的主体 |
| **State** $s$ | 当前所处的情况 |
| **Action** $a$ | agent 采取的动作 |
| **Policy** $\pi(a \mid s)$ | 在某个 state 下选择各个 action 的概率 |
| **Environment** | agent 所处的环境，接收 action、返回新的 state 和 reward |
| **Reward** $r$ | 环境给出的奖励信号 |

目标：找到一个 policy，使期望总奖励最大：$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$（$\tau$ 是一条轨迹，即一连串 state 和 action）。

### 4.2 对应到 LLM

| RL 概念 | LLM 中的对应 |
|---|---|
| Agent | LLM |
| Environment | tokens（文本） |
| State | 到目前为止的输入（prompt + 已生成的 tokens） |
| Action | 下一个 token |
| Policy $\pi_\theta$ | 下一个 token 的概率分布 $\pi_\theta(y_t \mid x, y_{<t})$ —— 就是 LLM 本身 |
| Reward | 人类偏好（通常只在整个回答生成完之后给出一次） |

**目标**：学习 $\pi_\theta$，使生成的回答符合人类偏好。

### 4.3 Policy gradient 的直觉

我们没法对“人类偏好”求导，那怎么更新模型？**Policy gradient** 的核心公式：

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\Big[\sum_{t} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot R(\tau)\Big]
$$

直觉：
- 让模型自己生成回答（采样）
- 如果这个回答得到了高 reward，就**提高**生成它的每个 token 的对数概率
- 如果得到了低 reward，就**降低**这些 token 的概率
- 就像 SFT 的梯度（提高某个回答的概率），只是每个回答被它的 reward **加权**了，而且回答来自模型自己

问题：只用 $R$ 加权，梯度的**方差很大**（同一个 prompt 的不同回答 reward 波动大，“得了 3 分”到底算好还是坏，取决于平时能得几分）。后面的 baseline、advantage 都是为了降低方差。

---

## 5. RLHF（Reinforcement Learning from Human Feedback）

来自 InstructGPT 论文（Ouyang et al., 2022）。完整流程是三步：先 SFT，然后：

- ⚖ **Step 1 – Reward modeling**：学会区分好坏。输入 (prompt, response)，输出一个数值分数
- ⚙ **Step 2 – Reinforcement learning**：对齐模型。输入 prompt，模型生成 response，用 reward model 的分数作为奖励来优化模型

InstructGPT 的规模参考：SFT 约 1.3 万条 prompt，reward model 约 3.3 万条 prompt 的比较数据，PPO 约 3.1 万条 prompt。结果：**1.3B 参数的 InstructGPT 的回答，比 175B 的 GPT-3 更受标注者青睐** —— 对齐带来的提升超过了 100 倍的参数量。

### 5.1 Step 1：Reward Model（RM）

**目标**：训练一个模型 $r_\phi(x, y)$，知道哪些回答好、哪些回答坏：“watch a movie together” 那个回答得高分，“do not spend much time with your teddy bear” 得低分。

**Bradley-Terry 模型**（Bradley et al., 1952）：给定两个回答的 reward $r_i$、$r_j$，“$y_i$ 比 $y_j$ 好”的概率定义为

$$
P(y_i \succ y_j \mid x) = \frac{e^{r_i}}{e^{r_i} + e^{r_j}} = \sigma(r_i - r_j)
$$

其中 $\sigma(z) = 1/(1 + e^{-z})$ 是 sigmoid 函数。

**直觉与算例**：
- 只有 reward 的**差值**有意义（两个 reward 同时加 100，概率不变）
- $r_A = 2$、$r_B = 0.5$：$P(A \succ B) = \sigma(1.5) \approx 0.82$
- $r_A = r_B$：$P = 0.5$，无法区分
- 差值越大，越确定谁更好

**训练**：在 pairwise 偏好数据上做最大似然，等价于最小化

$$
\mathcal{L}_{\text{RM}}(\phi) = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\big(r_\phi(x, y_w) - r_\phi(x, y_l)\big)\Big]
$$

这个 loss 推动 reward model 给 winner 的分数比 loser 高，而且差得越多越好。

**数据**：约 **O(10,000)** 条观测，标签 = 人类评分（**这就是 RLHF 中 “HF” 的来源**）。

**模型**：
- 通常用一个预训练 LLM（常从 SFT 模型初始化），把“预测下一个 token”的输出头换成一个**输出单个标量的头**：读入 prompt + response，取最后一个 token 的隐藏向量，经过一个线性层得到 reward
- 也可以用 encoder-only 模型（如 BERT），用 `[CLS]` 的向量投影成分数
- 评估 reward model 好坏的 benchmark：RewardBench（Lambert et al., 2024）

### 5.2 Step 2：Reinforcement Learning

**目标**：用 RL 改变 LLM 的权重，**惩罚坏回答、鼓励好回答**。
- 🔥 被训练：policy（LLM，从 SFT 模型初始化）
- ❄ 冻结：reward model

**数据**：约 **O(100,000)** 条 prompt；标签 = reward model 给出的分数（这一步不再需要人工标注）。

**目标函数**：

$$
\max_\theta\;
\underbrace{\mathbb{E}_{x \sim D,\; y \sim \pi_\theta(\cdot \mid x)}\big[r_\phi(x, y)\big]}_{\text{最大化 reward}}
\;-\;
\underbrace{\beta \cdot \mathrm{KL}\big(\pi_\theta(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big)}_{\text{不要偏离 base model 太远}}
$$

$\pi_{\text{ref}}$ 是参考模型（通常就是 SFT 模型，训练中保持不变），$\beta$ 控制约束的强弱。

**为什么一定要有 KL 约束**：
- **防止 reward hacking**：reward model 只是人类偏好的**近似**，它有漏洞。如果只追求高分，policy 会找到能骗过 reward model、实际上很差的回答。典型例子：reward model 碰巧偏好长回答 → policy 学会写得越来越长、不停重复；偏好讨好的语气 → policy 学会无条件附和用户（sycophancy）
- reward model 只在“正常回答”的分布上训练过，policy 跑得太远，reward model 的打分就不可信了
- 保持语言流畅性和多样性，防止训练不稳定

**实践中的写法**：KL 惩罚通常按 token 计算，每个 token 都有一个小惩罚 $-\beta \log \frac{\pi_\theta(y_t \mid \cdot)}{\pi_{\text{ref}}(y_t \mid \cdot)}$，最后一个 token 再加上 reward model 的分数。

### 5.3 PPO（Proximal Policy Optimization，Schulman et al., 2017）

PPO 是 RLHF 中最常用的 RL 算法。

**（1）Advantage 而不只是 reward**

PPO 最大化的是 **advantage（优势）**：

$$
A \approx \text{Reward} - \text{Baseline}
$$

直觉：只看 reward 方差很大。“这个回答得了 3 分”好不好，要看平时能得几分。advantage 衡量的是“这个动作**比平均水平好多少**”：好于预期则为正，差于预期则为负。

**（2）Baseline 用 value function 估计**
- **Value function** $V(s)$：从当前状态出发，按当前 policy 继续生成下去，**预期能得到多少 reward**
- 在 LLM 中是 **token 级别**的：每生成一个 token 都有一个 value 估计
- 由一个单独的 **value model（critic）** 给出，与 policy 一起训练，训练目标是拟合实际得到的 reward
- 常用 **GAE（Generalized Advantage Estimation，Schulman et al., 2015）** 计算每个 token 的 advantage：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t),\qquad
\hat A_t = \sum_{l=0}^{\infty} (\gamma\lambda)^l\, \delta_{t+l}
$$

$\delta_t$ 是“这一步实际发生的”与“预期的”之差；$\lambda$ 在“只看一步（偏差大、方差小）”和“看到底（偏差小、方差大）”之间折中。

**（3）为什么要用新旧 policy 的比值**

为了提高样本利用率，PPO 会用同一批生成的回答做**多次**梯度更新。但更新几次之后，当前 policy $\pi_\theta$ 已经和生成数据时的 policy $\pi_{\theta_{\text{old}}}$ 不一样了。用重要性采样的比值来修正：

$$
\text{ratio}_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
$$

**（4）变体 1：PPO-Clip**

限制比值的范围，防止一次更新太大：

$$
L^{\text{CLIP}}(\theta) = \mathbb{E}_t\Big[\min\big(\text{ratio}_t \cdot \hat A_t,\; \text{clip}(\text{ratio}_t,\, 1-\epsilon,\, 1+\epsilon) \cdot \hat A_t\big)\Big]
$$

**算例**（$\epsilon = 0.2$）：
- **好动作**（$\hat A = +2$），新 policy 已把它的概率提高到原来的 1.5 倍（ratio = 1.5）：$\min(1.5 \times 2,\; 1.2 \times 2) = 2.4$。目标值被“封顶”在 ratio = 1.2 处，ratio 再增大也不会让目标变大 → **不再鼓励继续提高**
- **坏动作**（$\hat A = -1$），新 policy 已把它的概率降到原来的 0.7 倍（ratio = 0.7）：$\min(0.7 \times (-1),\; 0.8 \times (-1)) = -0.8$。目标被固定在 −0.8，ratio 再降低也没有收益 → **不再继续压低**

所以每次更新都是“近端的（proximal）”：在旧 policy 附近的一个信任区域内改进，不会一步走太远。

术语提醒：
- 这是一个要**最大化**的 objective，不是要最小化的 loss（代码里通常取负号当 loss）
- 这里的 ratio 在很多论文中也写成 $r_t(\theta)$，**不要和 reward 混淆**

**（5）变体 2：PPO-KL Penalty**

不裁剪，而是直接惩罚新旧 policy 分布的差异：

$$
L^{\text{KL}}(\theta) = \mathbb{E}_t\Big[\text{ratio}_t \cdot \hat A_t - \beta \cdot \mathrm{KL}\big(\pi_{\theta_{\text{old}}} \,\|\, \pi_\theta\big)\Big]
$$

- old = 上一轮 RL 迭代的模型；ref = base model
- 现在的常见做法是对 **ref（base model）** 计算 KL（即 5.2 节目标函数中的那一项）

**（6）一轮 PPO 训练的完整流程**

1. 从 prompt 集中采样一批 prompt
2. 用当前 policy 生成回答
3. 用 reward model 打分，加上每个 token 的 KL 惩罚，得到每个 token 的 reward
4. 用 value model 估计每个 token 的 value，用 GAE 算出 advantage
5. 用 PPO-Clip 目标，对这批数据做几个 epoch 的小批量梯度更新（同时更新 value model）
6. 回到第 1 步

### 5.4 PPO 的局限与替代方案

**局限**：需要同时维护 **4 个模型**：

| 模型 | 状态 | 作用 |
|---|---|---|
| Policy | 🔥 训练 | 生成回答 |
| Value model | 🔥 训练 | 估计 baseline |
| Reward model | ❄ 冻结 | 打分 |
| Reference（base）model | ❄ 冻结 | 计算 KL |

每个都和 LLM 差不多大，显存占用约是单个模型的 4 倍，工程复杂度很高。值得吗？

**替代方案**：
- **REINFORCE**：最基础的 policy gradient，不需要 value model，用一个简单的 baseline（例如一批回答的平均 reward）来降低方差
- **RLOO**（REINFORCE Leave-One-Out）：对同一个 prompt 采样 $k$ 个回答，第 $i$ 个回答的 baseline 用**其余** $k-1$ 个回答的平均 reward
- **GRPO**（Group Relative Policy Optimization，DeepSeekMath，Shao et al., 2024）：对同一个 prompt 采样一组 $G$ 个回答，用组内 reward 的均值和标准差做归一化，作为每个回答的 advantage：

$$
\hat A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}
$$

**GRPO 算例**：
- 4 个回答的 reward 为 [1, 0, 0, 1]（答对 / 答错）：均值 0.5，标准差 0.5 → advantage = [+1, −1, −1, +1]
- reward 为 [0.9, 0.5, 0.1, 0.5]：均值 0.5，标准差约 0.28 → advantage ≈ [+1.41, 0, −1.41, 0]

GRPO **省掉了 value model**，同组的其他回答充当了 baseline。它是训练推理模型最常用的算法（Lecture 6 详述）。

### 5.5 RL 方法的挑战

- 需要先训练 reward model（两阶段流程），reward model 的缺陷会被 RL 放大
- **超参数多**（$\beta$、clip 范围、学习率、GAE 参数、每批更新几轮……），需要大量调参
- **训练不稳定**：可能突然崩溃（输出变成乱码或重复）
- **很难找到监控训练进展的好指标**：reward 上升可能只是 reward hacking
- **需要回答的多样性**：如果模型对每个 prompt 总是生成几乎一样的回答，同组回答的 reward 都一样，就学不到什么
- 并不清楚 preference tuning 是否**必须**用 RL

---

## 6. 不想用 RL？Best-of-N（BoN）

**思路**：跳过 RL 步骤，直接利用 reward model 的分数。

**做法**：
1. 给定 prompt，用 SFT 模型生成 $N$ 个回答
2. 用 reward model 给每个回答打分
3. 选分数最高的那个返回

例：
- “Of course! Teddy bears not only make awesome companions … How about you both watch a movie together?” → 0.8 ✅
- “I'd suggest you do not spend much time with your teddy bear at all.” → −2

**优点**：简单、稳定，常常是很强的 baseline。
**缺点**：**每次推理**都要生成 $N$ 个回答，推理成本是 $N$ 倍；$N$ 很大时同样会出现 reward hacking（挑出的是“最能骗过 RM”的回答）。

**延伸用法：rejection sampling fine-tuning**。把 BoN 选出的最佳回答收集起来，作为新的 SFT 数据再训练模型。这样把 BoN 的好处“蒸馏”进模型，推理时不再需要采样 $N$ 次（LLaMA 2 就同时用了这种方法和 PPO）。

---

## 7. DPO（Direct Preference Optimization，Rafailov et al., 2023）

### 7.1 动机

- RL 方法有上述种种麻烦
- Best-of-N 推理成本高
- **能不能用监督学习的方式直接在偏好数据上训练？**

### 7.2 DPO 的 loss

$$
\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)}\Big[\log \sigma\Big(
\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
- \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\Big)\Big]
$$

- **不需要单独训练 reward model** —— 式中**没有 $r(x, y)$**
- **直接在偏好数据上操作**，像普通的监督学习一样训练
- 形式与 reward model 的 Bradley-Terry loss **完全一样**，只是 reward 换成了一种特殊的“**隐式 reward**”：

$$
\hat r_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}
$$

- 直观理解：**提高好回答相对参考模型的概率，降低坏回答相对参考模型的概率**
- 论文副标题：“Your Language Model is Secretly a Reward Model”（你的语言模型其实就是一个 reward model）

### 7.3 DPO 公式从哪来：完整推导

**第 1 步：从 RLHF 的目标出发**

$$
\max_\pi\; \mathbb{E}_{x,\, y \sim \pi}\big[r(x, y)\big] - \beta\, \mathrm{KL}\big(\pi(\cdot \mid x) \,\|\, \pi_{\text{ref}}(\cdot \mid x)\big)
$$

**第 2 步：求出最优 policy 的闭式解**

把 KL 展开，对每个 $x$，目标可以改写为

$$
\sum_y \pi(y \mid x)\Big[r(x, y) - \beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)}\Big]
= -\beta\, \mathrm{KL}\Big(\pi \,\Big\|\, \tfrac{1}{Z(x)}\pi_{\text{ref}}\, e^{r/\beta}\Big) + \beta \log Z(x)
$$

KL 散度最小为 0，当且仅当两个分布相等。所以最优 policy 是

$$
\pi^*(y \mid x) = \frac{1}{Z(x)}\, \pi_{\text{ref}}(y \mid x)\, \exp\Big(\frac{r(x, y)}{\beta}\Big),\qquad
Z(x) = \sum_y \pi_{\text{ref}}(y \mid x)\, \exp\Big(\frac{r(x, y)}{\beta}\Big)
$$

直觉：最优 policy 就是**在参考模型的基础上，按 reward 的指数重新加权**。$Z(x)$ 要对所有可能的回答求和，根本算不出来 —— 这就是为什么不能直接用这个公式。

**第 3 步：反解出 reward**

对上式取对数并整理：

$$
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
$$

reward 可以用“最优 policy 和参考 policy 的对数比值”来表示（再加一个只依赖于 $x$ 的项）。

**第 4 步：代入 Bradley-Terry**

$$
P(y_w \succ y_l) = \sigma\big(r(x, y_w) - r(x, y_l)\big)
= \sigma\Big(\beta \log \frac{\pi^*(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi^*(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\Big)
$$

两个 reward 相减时，**只依赖于 $x$ 的 $\beta \log Z(x)$ 正好抵消** —— 最难算的部分消失了！

**第 5 步：推出 DPO loss**

用我们要训练的 $\pi_\theta$ 代替 $\pi^*$，在偏好数据上做最大似然（取负对数），就得到 $\mathcal{L}_{\text{DPO}}$。

**总结这个推导的妙处**：RLHF 是“先学 reward，再用 RL 找最优 policy”；DPO 发现 reward 和最优 policy 之间有一一对应的关系，于是**直接学 policy**，reward model 和 RL 都省了。

### 7.4 梯度直觉

DPO loss 的梯度为

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta\, \mathbb{E}\Big[\underbrace{\sigma\big(\hat r_\theta(x, y_l) - \hat r_\theta(x, y_w)\big)}_{\text{模型排错时权重大}}\Big(\nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x)\Big)\Big]
$$

- 括号里：提高 $y_w$ 的概率、降低 $y_l$ 的概率
- 权重 $\sigma(\hat r_l - \hat r_w)$：当前隐式 reward **把坏回答排得比好回答还高**（排错了）时，权重接近 1，大力纠正；已经排对且差距很大时，权重接近 0，几乎不再更新。**训练会自动聚焦在模型还做错的样本上**

### 7.5 数值例子

设 $\beta = 0.1$。对某条偏好数据：
- $\log \pi_\theta(y_w) - \log \pi_{\text{ref}}(y_w) = +2$（好回答的概率比参考模型提高了）
- $\log \pi_\theta(y_l) - \log \pi_{\text{ref}}(y_l) = -1$（坏回答的概率比参考模型降低了）
- 隐式 reward 差：$0.1 \times (2 - (-1)) = 0.3$
- Loss：$-\log \sigma(0.3) = -\log 0.574 \approx 0.55$

对比训练开始时（$\pi_\theta = \pi_{\text{ref}}$，两个对数比值都是 0）：loss $= -\log \sigma(0) = \log 2 \approx 0.69$。训练让 loss 下降，方向正确。

**$\beta$ 的含义**：和 RLHF 中 KL 约束的强度对应。$\beta$ 小 → 允许模型离参考模型更远；$\beta$ 大 → 更保守。常用值 0.1 左右。

### 7.6 实践要点与变体

- 训练时需要**两个模型**：要训练的 policy 和冻结的 reference（通常是 SFT 模型）。reference 的对数概率可以预先算好
- 数据通常是**离线（offline）**的：偏好对是事先收集的，可能来自别的模型。模型自己的“新错误”不会出现在数据里。改进方法：**iterative / online DPO**（每轮用当前模型生成新的回答对、重新标注、再训练）
- 已知现象：训练中 $y_w$ 和 $y_l$ 的概率可能**同时下降**（只要 $y_l$ 降得更多，loss 就会下降），需要注意过拟合
- **常见变体**：IPO（对过拟合更稳健）、KTO（不需要成对数据，只需要每个回答的“好 / 坏”标签）、SimPO（不需要 reference 模型，并按长度归一化）、ORPO（把 SFT 和偏好优化合成一步）

---

## 8. PPO-based RLHF 还是 DPO？

| | PPO-based RLHF | DPO |
|---|---|---|
| 实现难度 | 多阶段训练（RM → RL） | 普通的监督学习 |
| 需要的额外模型 | reward model、value model、base model | 只需要 base（reference）model |
| 数据 | **在线（on-policy）**：模型不断生成新回答，由 RM 打分 | **离线**：固定的偏好对数据集 |
| 稳定性 | 训练不稳定、调参困难 | 稳定、简单 |
| 效果 | **没有绝对共识**，随任务而异，且对实现细节很敏感 | 同左 |

研究（Xu et al., 2024，“Is DPO Superior to PPO for LLM Alignment?”）发现，精心调好的 PPO 在某些任务上（如竞赛编程）可以明显超过 DPO；但 DPO 的简单性使它在开源社区非常流行。

---

## 9. 效果对比

问 “Can I put my teddy bear in the washer?”：

| 模型 | 回答 |
|---|---|
| Pretrained | 续写 teddy bear 材质的百科内容（“Teddy bears are often made of materials like polyester and cotton…”） |
| Pretrained + instruction tuned | “No, it might get damaged. Try hand washing instead.” |
| + Preference tuned | “It's better not to. Your teddy could get hurt! A gentle hand wash is safer.”（更体贴，更符合人类偏好的语气） |

---

## 10. 要点回顾

1. SFT 只模仿好示范，缺少负面信号；preference tuning 用“A 比 B 好”的比较数据补上这一点。比较比生成更容易标注
2. LLM 可以看作 RL 中的 policy：state = 已有文本，action = 下一个 token，reward = 人类偏好
3. Reward model 用 Bradley-Terry：$P(y_w \succ y_l) = \sigma(r_w - r_l)$，在约 1 万条人工偏好上训练
4. RL 目标 = 最大化 reward − β·KL(π‖π_ref)；KL 防止 reward hacking
5. PPO 用 value model 估计 baseline、用 GAE 算 advantage、用 clip 限制每次更新的幅度；需要 4 个模型
6. GRPO 用同组回答的平均 reward 当 baseline，省掉 value model
7. Best-of-N 不训练、只在推理时挑最好的，成本是 N 倍；挑出的好回答也可以拿去做 SFT
8. DPO 由 RLHF 目标的闭式最优解推出，配分函数在 Bradley-Terry 中抵消，得到直接在偏好对上训练的监督 loss

---

## 11. 自测题

<details>
<summary>1. Reward model 给回答 A 打 1.0 分、给 B 打 −1.0 分。Bradley-Terry 认为 A 比 B 好的概率是多少？</summary>

σ(1.0 − (−1.0)) = σ(2) = 1 / (1 + e⁻²) ≈ 0.88。
</details>

<details>
<summary>2. 如果去掉 RLHF 目标中的 KL 项，最可能发生什么？</summary>

Reward hacking：policy 会找到 reward model 的漏洞来刷高分，例如写得越来越长、不停重复、过度讨好，实际质量下降，甚至输出退化成乱码。模型也会远离 reward model 熟悉的分布，使打分失去意义。
</details>

<details>
<summary>3. PPO-Clip 中 ε = 0.2，某个动作 advantage 为 +1，ratio 为 1.3。clip 后的目标值是多少？继续提高这个动作的概率还有收益吗？</summary>

min(1.3 × 1, 1.2 × 1) = 1.2。目标被封顶在 ratio = 1.2，再提高概率不会增加目标值（梯度为 0），所以没有收益。
</details>

<details>
<summary>4. GRPO 中一组 4 个回答的 reward 全是 1，advantage 是多少？这说明什么？</summary>

均值 1，所有回答减去均值都是 0（标准差为 0，实现中会加一个小 ε 防止除零），advantage 全为 0。这组回答提供不了学习信号 —— 题目太简单（或全错时太难）。
</details>

<details>
<summary>5. DPO 推导中，配分函数 Z(x) 为什么会消失？</summary>

reward 可以写成 β·log(π*/π_ref) + β·log Z(x)。Bradley-Terry 只用两个回答 reward 的差值，而两个回答对应同一个 prompt x，β·log Z(x) 相同，相减时正好抵消。
</details>

<details>
<summary>6. DPO 的梯度权重 σ(r̂_l − r̂_w) 起什么作用？</summary>

当隐式 reward 把坏回答排得比好回答高（排错了）时，权重接近 1，更新幅度大；已经排对且差距大时，权重接近 0，几乎不再更新。训练自动聚焦在模型还做错的样本上。
</details>

<details>
<summary>7. PPO 需要哪四个模型？哪些在训练、哪些冻结？</summary>

Policy（训练）、value model（训练）、reward model（冻结）、reference model（冻结）。
</details>

---

## 参考论文

- Ouyang et al., “Training language models to follow instructions with human feedback” (InstructGPT), 2022
- Bradley et al., “The rank analysis of incomplete block designs: I. the method of paired comparisons”, 1952
- Lambert et al., “RewardBench: Evaluating Reward Models for Language Modeling”, 2024
- Schulman et al., “Proximal Policy Optimization Algorithms”, 2017；Schulman et al., “High-Dimensional Continuous Control Using Generalized Advantage Estimation”, 2015
- Shao et al., “DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models” (GRPO), 2024
- Touvron et al., “Llama 2: Open Foundation and Fine-Tuned Chat Models” (rejection sampling + PPO), 2023
- Rafailov et al., “Direct Preference Optimization: Your Language Model is Secretly a Reward Model”, 2023
- Xu et al., “Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study”, 2024
- Azar et al., IPO, 2023；Ethayarajh et al., KTO, 2024；Meng et al., SimPO, 2024；Hong et al., ORPO, 2024
