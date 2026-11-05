# LLM-Study

Harvard **CS 2680：Modern AI Systems: Agents and System Optimizations**（Juncheng Yang，Fall 2026）的学习笔记。

- `notes/`：中文知识笔记，专业名词保留英文。**每篇都可以独立阅读，不需要对照 PDF。**
- `slides/`：原始课程 slides（PDF），仅供参考。

## 目录

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
