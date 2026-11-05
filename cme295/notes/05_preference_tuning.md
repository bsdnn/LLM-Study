# Lecture 5 · Preference Tuning：RLHF、PPO 与 DPO

> Stanford CME 295：Transformers & Large Language Models
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture5.pdf`](../slides/fall25-cme295-lecture5.pdf)

## 本讲脉络

```
为什么需要 preference tuning → 偏好数据的形式与收集
→ 把 LLM 写成 RL 问题 → RLHF：① Reward model（Bradley-Terry）② RL（PPO：advantage、clip、KL penalty）
→ PPO 的替代（REINFORCE、GRPO）与 RL 的难点 → Best-of-N
→ DPO：不用 RL 的监督式偏好优化（推导思路）→ PPO vs DPO
```

一句话总结：SFT 只告诉模型“该怎么说”，没告诉它“哪些不该说”。**Preference tuning 用“A 比 B 好”这种比较数据，把负面信号注入模型。** RLHF 先训练一个 reward model 再用 PPO 优化；DPO 则证明可以把同一目标改写成一个普通的监督学习 loss，直接在偏好数据上训练。

---

## 1. 在 LLM 生命周期中的位置

```
初始化 ──Pretraining──► 有基础知识的模型
       ──Finetuning（SFT）──► 针对任务调整的模型
       ──Preference tuning──► 与（人类）偏好对齐的模型   ← 本讲
```

## 2. 为什么需要 Preference Tuning

**背景**：SFT 后的模型仍可能“乱来”。例：
- Prompt：“Suggest a new activity I could do with my teddy bear.”
- 坏回答：“I'd suggest you do not spend much time with your teddy bear at all.”
- 好回答：“Of course! Teddy bears not only make awesome companions for a delightful sleep, but can also be great buddies for fun activities. How about you both watch a movie together?”

SFT 只学“正确示范”，没有机制告诉模型“这种回答不好”。我们需要**注入负面信号**。

**思路**：收集偏好对（preference pairs），在上面训练模型。

**为什么用偏好，而不是继续加 SFT 数据？**
- **比较比生成容易**：判断“A 比 B 好”远比从零写出一个完美的 A 容易，标注更便宜、更一致
- SFT 对数据分布非常敏感，很容易“搞砸”
- SFT 不易扩展：数据质量很重要且难以获得
- 不过，模型“乱来”也是一个提醒：应该回头检查 SFT 数据的质量

---

## 3. 偏好数据

一条观测 = (prompt x, response y)。三种标注形式：

| 形式 | 做法 | 例子 |
|---|---|---|
| **Pointwise** | 给每个回答打一个绝对分 | Obs 1 → 0.4，Obs 2 → 0.9 |
| **Pairwise** | 比较两个回答哪个更好 | y_A > y_B（最常用） |
| **Listwise** | 对多个回答排序 | 1 > 2 > 3 > 4 |

### 收集 pairwise 偏好数据的流程

1. **为同一个 prompt 生成一对回答**
   - Prompt 来源：线上日志、参考分布
   - 回答来源：用 SFT 模型以不同 temperature 采样 / 合成 / 人工改写
2. **标注哪个更好**（得到 y_w = winner，y_l = loser）
   - 人工评分
   - 代理方法：LLM-as-a-judge、BLEU、ROUGE 等
   - 变体：二元（更好/更差）或更细粒度的打分尺度

---

## 4. 把 LLM 写成强化学习问题

**通用 RL 框架**：Agent 处于某个 State，根据 Policy 采取 Action，Environment 返回 Reward 和新的 State。

**对应到 LLM**：

| RL 概念 | LLM 中的对应 |
|---|---|
| Agent | LLM |
| State | 到目前为止的输入（prompt + 已生成的 tokens） |
| Action | 下一个 token |
| Policy π_θ | 下一个 token 的概率分布 π_θ(y_t \| x, y<t) |
| Reward | 人类偏好（通常只在整个回答结束后给出） |

**目标**：学习 π_θ，使生成的回答符合人类偏好。

---

## 5. RLHF（Reinforcement Learning from Human Feedback）

来自 InstructGPT 论文（Ouyang et al., 2022），两步：

- ⚖ **Step 1 – Reward modeling**：学会区分好坏。输入 (prompt, response)，输出一个数值分数
- ⚙ **Step 2 – Reinforcement learning**：对齐模型。输入 prompt，输出 response，用 reward model 的分数作为奖励优化 LLM

### 5.1 Step 1：Reward Model（RM）

**目标**：训练一个模型 r_φ(x, y)，好回答得高分（如 “watch a movie together” 那个回答），坏回答得低分。

**Bradley-Terry 模型**（Bradley et al., 1952）：给定两个回答的 reward，“y_i 比 y_j 好”的概率定义为

```
P(y_i ≻ y_j | x) = exp(r_i) / (exp(r_i) + exp(r_j)) = σ(r_i − r_j)
```

σ 是 sigmoid。直觉：reward 差距越大，越确定谁更好；只有**差值**有意义。

**训练**：在 pairwise 偏好数据上最大化似然，等价于最小化

```
L_RM(φ) = −E_{(x, y_w, y_l)} [ log σ( r_φ(x, y_w) − r_φ(x, y_l) ) ]
```

**数据**：约 **O(10,000)** 条观测，标签 = 人类评分（这就是 RLHF 中 “HF” 的来源）。

**模型**：
- 预训练 LLM + **分类/回归头**（输出一个标量，而不是预测下一个 token）
- 或 encoder-only 模型（如 BERT），用 `[CLS]` 的投影
- 评估 reward model 的 benchmark：RewardBench（Lambert et al., 2024）

### 5.2 Step 2：Reinforcement Learning

**目标**：用 RL 改变 LLM 的权重，惩罚坏回答、鼓励好回答。
- 🔥 被训练：policy（LLM，初始化为 SFT 模型）
- ❄ 冻结：reward model

**数据**：约 **O(100,000)** 条 prompt，标签 = reward model 给的分数（不再需要人工）。

**目标函数**：

```
max_θ  E_{x ~ D, y ~ π_θ(·|x)} [ r_φ(x, y) ]  −  β · KL( π_θ(·|x) ‖ π_ref(·|x) )
         └──────── 最大化 reward ────────┘     └─ 不要偏离 base model 太远 ─┘
```

**为什么要 KL 约束**：
- 防止 **reward hacking**：reward model 只是人类偏好的近似，有漏洞。policy 会找到能骗过 RM 得高分、但实际上很差的回答（比如无意义的重复、过度讨好）
- 防止训练不稳定、保留语言能力
- π_ref 是参考模型（通常就是 SFT 模型），β 控制约束强度

### 5.3 常用 RL 算法：PPO（Proximal Policy Optimization，Schulman et al., 2017）

**Advantage 而不只是 reward**：
- PPO 实际最大化的是 **advantage**：A ≈ Reward − Baseline
- 直觉：只看 reward 方差很大。“这个回答得了 3 分”好不好要看平时能得几分。advantage 衡量的是“这个动作比平均水平好多少”
- **Baseline 用 value function V(s) 估计**：
  - token 级别：从当前状态出发、按当前 policy 继续生成，**预期能得多少 reward**
  - 与 policy 一起训练，标签是 reward
  - 常用 **GAE（Generalized Advantage Estimation，Schulman et al., 2015）** 计算每个 token 的 advantage

**变体 1：PPO-Clip**

限制新旧 policy 之间的比值，防止一次更新过大：

```
ratio_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)

L_CLIP(θ) = E_t [ min( ratio_t · A_t,  clip(ratio_t, 1 − ε, 1 + ε) · A_t ) ]
```

- 若 advantage > 0（好动作），想提高它的概率，但 ratio 最多涨到 1+ε，再多就不给奖励
- 若 advantage < 0（坏动作），想降低它的概率，但最多降到 1−ε
- 每次更新都“近端”（proximal），不会一步走太远
- 术语提醒：这是一个要**最大化**的 objective，不是 loss；这里的 ratio 也常写成 r，不要与 reward 混淆

**变体 2：PPO-KL Penalty**

不裁剪，而是直接惩罚新旧 policy 分布的差异：

```
L_KL(θ) = E_t [ ratio_t · A_t − β · KL( π_θ_old ‖ π_θ ) ]
```

- 术语：old = 上一轮 RL 迭代的模型；ref = base model
- 现在的常见做法是对 **ref（base model）** 计算 KL

### 5.4 PPO 的局限与替代

- 需要同时维护 **4 个模型**：policy、value model、reward model、base（reference）model —— 显存和工程复杂度很高
- 值得吗？

**变体**：
- **REINFORCE**：最基础的 policy gradient，不需要 value model，用简单 baseline（如平均 reward）
- **GRPO**（Group Relative Policy Optimization，DeepSeekMath，Shao et al., 2024）：对同一 prompt 采样**一组**回答，用组内 reward 的均值和标准差做归一化作为 advantage，`A_i = (r_i − mean(r)) / std(r)`，**省掉了 value model**
- 还有很多其他变体

### 5.5 RL 方法的挑战

- 需要训练 reward model（两阶段流程）
- 超参数多，需要大量调参
- 训练不稳定
- 很难找到监控训练进展的好指标
- 需要回答的多样性（如果模型对每个 prompt 总是生成几乎一样的回答，就学不到什么）
- 并不清楚 preference tuning 是否**必须**用 RL

---

## 6. 不想用 RL 的变通：Best-of-N（BoN）

**思路**：跳过 RL 步骤，直接利用 reward model 的分数。

**做法**：
1. 给定 prompt，用 SFT 模型生成 N 个回答
2. 用 reward model 给每个回答打分
3. 选分数最高的那个

例：“Take your teddy bear on a picnic in your backyard.” 得分 0.8；“I'd suggest you do not spend much time…” 得分 −2 → 选前者。

**局限**：每次推理都要生成 N 个回答，**推理成本高**。

---

## 7. DPO（Direct Preference Optimization，Rafailov et al., 2023）

### 7.1 动机

- RL 方法有上述种种麻烦
- Best-of-N 推理成本高
- 能不能用**监督学习**的方式训练？

### 7.2 DPO 的 loss

把 RLHF 的目标改写成一个直接在偏好数据上的监督 loss：

```
L_DPO(θ) = −E_{(x, y_w, y_l)} [ log σ( β · log( π_θ(y_w|x) / π_ref(y_w|x) )
                                      − β · log( π_θ(y_l|x) / π_ref(y_l|x) ) ) ]
```

- 不需要单独训练 reward model —— 式中**没有 r(x, y)**
- 直接在偏好数据上操作
- 形式与 Bradley-Terry 一样，只是 reward 换成了一种特殊的“隐式 reward”：`r̂(x, y) = β · log( π_θ(y|x) / π_ref(y|x) )`
- 直观理解：**提高好回答相对参考模型的概率，降低坏回答相对参考模型的概率**
- 论文副标题：“Your Language Model is Secretly a Reward Model”

### 7.3 DPO 公式从哪来（推导思路）

1. **从 PPO / RLHF 的目标出发**：max E[r(x, y)] − β · KL(π ‖ π_ref)
2. **求出最优 policy 的闭式解**：
   ```
   π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp( r(x, y) / β )
   ```
   Z(x) 是归一化常数
3. **反解出 reward 项**：
   ```
   r(x, y) = β · log( π*(y|x) / π_ref(y|x) ) + β · log Z(x)
   ```
4. **代入 Bradley-Terry**：P(y_w ≻ y_l) = σ(r(x, y_w) − r(x, y_l))。两个 reward 相减时，**只依赖于 x 的 β log Z(x) 正好抵消**，而 Z(x) 恰恰是最难算的部分
5. **推出 DPO loss**：用 π_θ 代替 π*，对偏好数据最大化似然，就得到上面的 L_DPO

### 7.4 PPO-based RLHF 还是 DPO？

| | PPO-based RLHF | DPO |
|---|---|---|
| 实现难度 | 多阶段训练；需要额外的 reward model、value model、base model | 监督学习；只需要额外一个 base（reference）model |
| 效果 | 没有绝对共识，随任务而异，且对实现细节敏感（Xu et al., 2024，“Is DPO Superior to PPO for LLM Alignment?”） | 同左 |

---

## 8. 效果对比

问 “Can I put my teddy bear in the washer?”：

| 模型 | 回答 |
|---|---|
| Pretrained | 续写 teddy bear 材质的百科内容 |
| Pretrained + instruction tuned | “No, it might get damaged. Try hand washing instead.” |
| + Preference tuned | “It's better not to. Your teddy could get hurt! A gentle hand wash is safer.”（更体贴、更符合人类偏好的语气） |

---

## 参考论文

- Ouyang et al., “Training language models to follow instructions with human feedback” (InstructGPT), 2022
- Bradley et al., “The rank analysis of incomplete block designs: I. the method of paired comparisons”, 1952
- Lambert et al., “RewardBench”, 2024
- Schulman et al., “Proximal Policy Optimization Algorithms”, 2017；Schulman et al., “High-Dimensional Continuous Control Using Generalized Advantage Estimation”, 2015
- Shao et al., “DeepSeekMath” (GRPO), 2024
- Rafailov et al., “Direct Preference Optimization: Your Language Model is Secretly a Reward Model”, 2023
- Xu et al., “Is DPO Superior to PPO for LLM Alignment? A Comprehensive Study”, 2024
