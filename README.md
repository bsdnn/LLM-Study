# LLM-Study

LLM 相关课程的中文学习笔记，专业名词保留英文。**每篇笔记都可以独立阅读，不需要对照 PDF。**

| 课程 | 文件夹 | 内容方向 |
|---|---|---|
| Harvard **CS 2680**：Modern AI Systems: Agents and System Optimizations（Juncheng Yang，Fall 2026） | [`cs2680/`](cs2680/) | Agent 的使用与设计、LLM serving、GPU |
| Stanford **CME 295**：Transformers & Large Language Models（Afshine & Shervine Amidi，Fall 2025） | [`cme295/`](cme295/) | Transformer 原理、LLM 训练与调优、推理模型、RAG 与 agents、评估 |

每个课程文件夹下：
- `notes/`：中文知识笔记
- `slides/`：原始课程 slides（PDF），仅供参考

## CME 295：Transformers & Large Language Models

建议先读这门课打基础（Transformer 原理 → 训练 → 应用），再读 CS 2680 的 agent 与系统部分。

| # | 主题 | 笔记 | 核心内容 |
|---|---|---|---|
| 1 | NLP 基础与 Transformer 架构 | [笔记](cme295/notes/01_transformer.md) · [PDF](cme295/slides/fall25-cme295-lecture1.pdf) | NLP 任务与指标、tokenization、Word2vec、RNN/LSTM、self-attention（Q/K/V）、encoder-decoder、端到端翻译例子 |
| 2 | Transformer 改进技巧与 BERT | [笔记](cme295/notes/02_transformer_variants_bert.md) · [PDF](cme295/slides/fall25-cme295-lecture2.pdf) | Sinusoidal / ALiBi / RoPE、Pre-Norm 与 RMSNorm、sliding window、MQA/GQA、三类 Transformer 模型、BERT（MLM、NSP、fine-tuning）、DistilBERT、RoBERTa |
| 3 | LLM 概览：MoE、解码、Prompting 与推理优化 | [笔记](cme295/notes/03_llm_overview.md) · [PDF](cme295/slides/fall25-cme295-lecture3.pdf) | MoE 与 routing collapse、greedy / beam / top-k / top-p / temperature、guided decoding、ICL、CoT、self-consistency、KV cache、PagedAttention、MLA、speculative decoding、MTP |
| 4 | LLM 训练：Pretraining、训练优化、SFT 与 LoRA | [笔记](cme295/notes/04_training.md) · [PDF](cme295/slides/fall25-cme295-lecture4.pdf) | Scaling laws、Chinchilla、Adam 与显存、ZeRO、TP/PP/SP/CP/EP、FlashAttention、混合精度、instruction tuning、Chatbot Arena、LoRA、QLoRA |
| 5 | Preference Tuning：RLHF、PPO 与 DPO | [笔记](cme295/notes/05_preference_tuning.md) · [PDF](cme295/slides/fall25-cme295-lecture5.pdf) | 偏好数据、Bradley-Terry reward model、PPO（advantage、clip、KL）、GRPO、Best-of-N、DPO 及其推导 |
| 6 | Reasoning Models：用 RL 扩展推理能力 | [笔记](cme295/notes/06_reasoning_models.md) · [PDF](cme295/slides/fall25-cme295-lecture6.pdf) | pass@k / cons@k、verifiable rewards、test-time scaling、GRPO vs PPO、长度偏差（DAPO、Dr. GRPO）、DeepSeek R1-Zero 与 R1、distillation |
| 7 | RAG、Tool Calling 与 Agents | [笔记](cme295/notes/07_rag_tools_agents.md) · [PDF](cme295/slides/fall25-cme295-lecture7.pdf) | Chunking、bi-encoder / BM25 / hybrid、HyDE、contextual retrieval、cross-encoder 重排序、NDCG / MRR、tool calling、MCP、ReAct、A2A、agent 安全 |
| 8 | LLM 评估 | [笔记](cme295/notes/08_evaluation.md) · [PDF](cme295/slides/fall25-cme295-lecture8.pdf) | Cohen's Kappa、BLEU / ROUGE / METEOR、LLM-as-a-Judge 与三种 bias、factuality、agent 失败模式、MMLU / AIME / PIQA / SWE-bench / HarmBench / τ-bench、pass^k、数据污染 |
| 9 | 超越文本 Transformer：ViT、VLM、Diffusion LLM 与展望 | [笔记](cme295/notes/09_beyond_transformers.md) · [PDF](cme295/slides/fall25-cme295-lecture9.pdf) | ViT、VLM 两种架构、DDPM、masked diffusion LLM（LLaDA）、模态间借鉴、model collapse、注意力硬件、未解决的挑战 |

## CS 2680：Modern AI Systems: Agents and System Optimizations

| # | 主题 | 笔记 | 核心内容 |
|---|---|---|---|
| 2 | Agent From a User's Perspective I：How Claude Code Works | [笔记](cs2680/notes/02_agent_use_1.md) · [PDF](cs2680/slides/2_agent_use_1.pdf) | Agentic loop、context 与 memory（CLAUDE.md、auto memory）、Skill / MCP / Hook / Plugin |
| 3 | Agent From a User's Perspective II：Multi-agent and Best Practice | [笔记](cs2680/notes/03_agent_use_2.md) · [PDF](cs2680/slides/3_agent_use_2.pdf) | Superpowers 与 Caveman 案例、parallel agents、subagents、agent teams、`/goal` `/loop`、Software 3.0 |
| 4 | Agent Design I：Context Management and Tool Design | [笔记](cs2680/notes/04_agent_design_1.md) · [PDF](cs2680/slides/4_agent_design_1.pdf) | ReAct、search/edit/execution tools 设计、prompt engineering、context engineering、Agent = Model × Harness |
| 5 | Agent Design II：Multi-Agent Systems | [笔记](cs2680/notes/05_agent_design_2.md) · [PDF](cs2680/slides/5_agent_design_2.pdf) | 八种 multi-agent 架构、task decomposition、agent communication、A2A vs MCP |
| 6 | Agent Design III：Memory and Reliability | [笔记](cs2680/notes/06_agent_design_3.md) · [PDF](cs2680/slides/6_agent_design_3.pdf) | Memory 的写入/组织/存储/检索/遗忘、memory 系统、五类 agent failure 及对策 |
| 7 | Agent Design IV：Evaluation | [笔记](cs2680/notes/07_agent_design_4.md) · [PDF](cs2680/slides/7_agent_design_4.pdf) | pass@k / pass^k、GAIA、WebArena、OSWorld、SWE-bench、τ-bench、Terminal-Bench、data leakage |
| 8 | Agent Design V：Cost-Effective Agents | [笔记](cs2680/notes/08_agent_design_5.md) · [PDF](cs2680/slides/8_agent_design_5.pdf) | 成本分解公式、caching / routing、speculative actions、Monolithic / Modular / Microkernel harness |
| 9 | LLM Serving Basics I：Introduction to LLM | [笔记](cs2680/notes/09_llm_serving_basics_1.md) · [PDF](cs2680/slides/9_llm_serving_basics_1.pdf) | Request 生命周期、prefill vs decode、Transformer、KV cache、GQA、MoE、进阶架构 |
| 10 | LLM Serving Basics II：Intro to GPUs | [笔记](cs2680/notes/10_llm_serving_basics_2a.md) · [PDF](cs2680/slides/10_llm_serving_basics_2a.pdf) | GPU vs CPU、SIMT、roofline model、memory hierarchy、Little's law、Tensor Core、其他加速器 |

## 核心概念速查

| 术语 | 一句话解释 |
|---|---|
| **Agent** | LLM + 一个循环（loop）+ tools + state（context 与 memory）。LLM 提出动作，harness 执行，结果再喂回 LLM |
| **Harness** | 包在模型外面的运行框架：组装 context、定义 tools、检查 permission、执行、管理 context、决定何时停止 |
| **Agentic loop / ReAct** | Thought → Action → Observation 的循环，直到任务完成或触发停止规则 |
| **Context window** | 模型一次能看到的全部 tokens，是 agent 的 working memory，随 session 结束而消失 |
| **Persistent memory** | 跨 session 保留的信息（如 CLAUDE.md、memory 文件、memory 数据库） |
| **Compaction** | context 快满时，用 LLM 把旧历史总结成摘要，是有损的最后手段 |
| **Context engineering** | 为下一步精确挑选放进 context 的信息（admission、offloading、transformation、compaction） |
| **Skill** | 按需加载的知识与流程（一个带 `SKILL.md` 的文件夹） |
| **MCP**（Model Context Protocol） | agent ↔ 外部能力（tools、resources、prompts）的标准协议 |
| **A2A**（Agent-to-Agent） | agent ↔ agent 的标准协议，核心抽象是 message、task、artifact |
| **Hook** | 在 lifecycle event（如 SessionStart、PreToolUse）自动执行的逻辑，用于强制约束 |
| **Plugin** | 把 skills、hooks、MCP 配置、agents 打包分发的单元 |
| **Subagent** | 拥有独立 context 的子 agent，只把 summary 返回给调用者 |
| **pass@k / pass^k** | k 次中至少一次成功 / k 次全部成功（后者衡量一致性） |
| **Prefill** | 一次 forward pass 处理整个 prompt，compute-bound，决定 TTFT（time to first token） |
| **Decode** | 每生成一个 token 做一次 forward pass，memory-bandwidth-bound，决定 tokens/s |
| **KV cache** | 缓存每个已处理 token 的 key/value，避免重复计算；大小 = 2 × L × H_kv × d_k × bytes/token |
| **GQA / MQA** | 多个 query heads 共享 KV heads，以缩小 KV cache |
| **MoE**（Mixture of Experts） | 用 router 为每个 token 选少数 expert FFN；显存看总参数，计算看激活参数 |
| **Arithmetic intensity** | FLOPs ÷ 搬运的字节数；低于 ridge point 为 memory-bound，高于为 compute-bound |
| **Roofline model** | 用 arithmetic intensity 判断 kernel 受限于带宽还是算力的模型 |
| **Warp / SIMT** | 32 个线程组成一个 warp，同一条指令驱动 32 个线程执行 |
| **Tensor Core** | GPU 上专门做小矩阵乘加的单元，承担 LLM 99% 以上的 FLOPs |
| **Tokenization / BPE** | 把文本切成 token；BPE 从字符开始反复合并最常见的相邻对，得到 subword 词表 |
| **Self-attention** | softmax(QKᵀ/√d_k)·V：每个 token 用 query 与所有 key 比较，按权重汇总 values |
| **RoPE** | 按位置旋转 Q、K 向量，使 attention 分数只依赖相对距离；当前位置编码的默认选择 |
| **Pre-Norm / RMSNorm** | 在子层之前做归一化（训练更稳），RMSNorm 只按均方根缩放（更便宜） |
| **BERT** | Encoder-only 模型，用 MLM 和 NSP 预训练，再针对下游分类任务 fine-tune |
| **Scaling laws / Chinchilla** | Loss 随参数、数据、算力按幂律下降；算力最优时约每个参数 20 个训练 tokens |
| **FlashAttention** | 分块在 SRAM 中计算 attention、不把 n×n 矩阵写回显存的精确算法 |
| **SFT / Instruction tuning** | 用期望的输入/输出对微调模型，使其成为遵循指令的助手 |
| **LoRA / QLoRA** | 冻结原权重，只训练低秩矩阵 B·A；QLoRA 再把冻结权重量化到 4-bit |
| **RLHF** | 先用人类偏好训练 reward model，再用 RL（如 PPO）优化模型，同时用 KL 限制偏离 |
| **DPO** | 把 RLHF 目标改写成直接在偏好对上训练的监督 loss，不需要 reward model 和 RL |
| **GRPO** | 用同一问题的一组回答的平均 reward 作为 baseline，省掉 value model 的 RL 算法 |
| **Reasoning model** | 先生成长推理链再回答的模型，通常用 verifiable rewards 的 RL 训练 |
| **RAG** | 先从知识库检索相关片段，加入 prompt，再让 LLM 生成回答 |
| **LLM-as-a-Judge** | 用 LLM 按评分准则给回答打分并给出理由；要注意位置、长度、自我偏好等 bias |
| **ViT** | 把图像切成 patch 当作 token，用 Transformer encoder 做图像理解 |
| **Diffusion LLM** | 从全是 `[MASK]` 的序列出发，并行地多步“揭开” token 的非自回归生成方式 |
