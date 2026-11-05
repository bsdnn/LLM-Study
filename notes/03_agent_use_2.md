# Lecture 3 · Agent From a User's Perspective II：Multi-agent and Best Practice

> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/3_agent_use_2.pdf`](../slides/3_agent_use_2.pdf)

## 本讲要点

- 两个 plugin case study：**Superpowers**（workflow 型）和 **Caveman**（token 压缩型）
- 用多个 agent 扩展：Parallel agents、Subagents、Agent teams、Dynamic workflow、Long-running agents（`/goal`、`/loop`）
- **Programming / Software 3.0**：从写代码到表达目标，核心主题是 uncertainty
- 来自 OpenAI Codex 的实际使用数据

---

## 1. Recap

- Agent loop
- State management：context = working memory；memory = 跨 session 持久化（CLAUDE.md / AGENTS.md、auto memory）
- Tools：built-in tools + 扩展（skill、MCP、hook、plugin）

---

## 2. Case Study：Superpowers

一个软件开发 plugin（`/plugin install superpowers@claude-plugins-official`）。

**同样的 model、同样的 tools，只是 instructions 不同**：

- 默认：Feature request → 直接写代码 → 靠迭代修复
- Superpowers：Clarify problem → Approve design → Write detailed plan → Test-driven build → Review & verify → Finish/integrate branch

### 9 个 Skills

| Skill | 职责 |
|---|---|
| using-superpowers | 发现并调用 skill 的规则 |
| brainstorming | 把请求变成被批准的 design |
| writing-plans | 把 design 拆成具体 tasks |
| using-git-worktrees | 创建隔离的 workspace |
| test-driven-development | 强制 red → green → refactor |
| systematic-debugging | 先找 root cause 再提修复 |
| requesting-code-review | 对照 plan review 实现 |
| verification-before-completion | 报告成功前要求证据 |
| finishing-a-development-branch | merge、PR、保留或清理 |

### SKILL.md 的结构

- Front matter（`name`、`description`）是 metadata；body 是 agent 遵循的 procedure
- 组成：name、description、instructions、checkpoints、references、scripts
- 一个 process skill 可以定义：entry conditions、required actions、prohibited actions、human approval gates、verification requirements、**下一个适用的 skill**

### 为什么只有 Skills 不够

安装 skill 只是让它“可用”，不保证 agent 会去检查它。Superpowers 用 **SessionStart hook**（matcher：`startup|clear|compact`）在每个 session 开始、clear 和 compaction 后，把 bootstrap skill（using-superpowers/SKILL.md）作为 `additionalContext` 注入 context。

### 设计经验

| 设计问题 | Superpowers 的回答 |
|---|---|
| 复杂行为如何表示？ | 小而可组合的 skills |
| agent 如何记得用？ | session hook 注入 bootstrap skill |
| context 如何控制？ | 只在相关时加载 task skill |
| 各阶段如何协调？ | 每个 skill 写明 gates 和 successor |
| 用户如何安装？ | 打包为 plugin |
| 每个 plugin 都需要 commands 或 MCP 吗？ | 不需要，按需添加 |

---

## 3. Case Study：Caveman

五种扩展机制 + 一个 local proxy，目标是**压缩 agent 的 tokens**。

为什么要压缩：
- Context 被 logs、diffs、tool output 填满
- 每个 turn 都要付费、都要等待
- Compaction 太晚会丢掉重要细节

例：786 行 test output → “3 tests fail: ETIMEDOUT at client.ts:88”。原则：**双向压缩，同时保留原文可检索**。

### 双向压缩

- **Agent output**：response skill 改变模型的写作方式（报告：output tokens 减少 65%，但 skill 本身增加 input tokens，整体节省更少）
- **Agent input**：local proxy + engine 压缩模型读到的 logs、tool results、diffs、files

### 各组件

| 组件 | 作用 |
|---|---|
| Skills | 简洁风格（保留术语、命令、路径、代码、错误、数字、否定；删除客套和 hedging） |
| Commands | `/caveman [lite\|full\|ultra\|off]`、`/caveman-commit`、`/caveman-review`、`/caveman-compress`、`/caveman-stats` |
| Hooks | SessionStart（`caveman-activate.js`：决定 mode、注入规则、compaction 后恢复）；UserPromptSubmit（`caveman-mode-tracker.js`：检测 mode 切换） |
| MCP | `caveman_compress`、`caveman_retrieve`（byte-exact 原文）、`caveman_stats`、`caveman_toon_encode/decode`（JSON ↔ TOON） |
| Local proxy | 位于 agent 和 model provider 之间，**type-aware compression**：logs 保留 errors/stack traces/head+tail；code 保留 imports/signatures/types；diffs 保留 file+hunk headers；JSON 保留 keys/结构 |
| Plugin | 打包 Claude Code 集成 |

**Proxy vs MCP**：proxy 拦截每个请求，MCP 只在被调用时运行；proxy 能改变 context，MCP 不能。**Hook 不能透明地改写发给 LLM 的 request，所以需要 proxy**（除非你自己写 harness）。

---

## 4. 用多个 Agent 成为 100x Engineer

### 三个扩展维度

| 维度 | 机制 | 带来什么 |
|---|---|---|
| Breadth | Parallel agents（每个在独立 worktree） | throughput（wall-clock 时间），不提升答案质量 |
| Breadth + depth | Subagents（独立 context，返回 summary） | throughput + focus，parent 看不到中间过程 |
| Duration | Long-running agent | scope：更少干预完成更多工作 |

Agent team 和 dynamic workflow 是它们的组合。

### Parallel Agents by Hand

- `claude -w feature-payments`、`claude -w bugfix-auth`：每个 session 一个 **git worktree**（独立工作目录和 branch，共享同一个 `.git` history）
- 无需 stash、无需第二个 clone
- 局限：plan 在你脑子里；session 之间互相看不到；你要自己 reconcile branches；不可重复。3 个 session 你能管，20 个就需要“克隆自己”

### Subagents

带独立 context window、tools 和 permissions 的专门助手。好处：
- **Preserve context**：logs、搜索结果不进入主对话
- **Enforce constraints**：限制 tools、指定 model
- **Reuse**：跨项目共享
- **Parallelization**：总时间由最慢的那个决定
- **Control costs**：路由到 Haiku 等更便宜的模型

**Built-in subagents**：Explore（只读搜索）、Plan（plan mode 中的 research）、General-purpose。Explore 和 Plan 跳过 CLAUDE.md 和 git status 以保持快和便宜。

**定义**：`.claude/agents/code-reviewer.md`（project scope）或 `~/.claude/agents/`（user scope），YAML frontmatter（`name`、`description`、`tools`、`model`），body 是 system prompt。

**Nested subagents**：`tools` 中必须包含 `Agent` 才能再派生；最多 5 层，**2–3 层最合适**。每层都从 fresh context 开始，只返回 summary。

**调用方式**（控制力递增）：自然语言提及 → `@agent-code-reviewer`（保证运行）→ `claude --agent code-reviewer`（整个 session 采用）。

**什么时候用 subagent**：
- 留在主对话：反复来回、阶段间共享 context、快速小改、latency 敏感
- 交给 subagent：冗长且不复用的输出、需要 tool/permission 限制、自包含且返回 summary、并行独立工作、可交给小模型的简单工作
- 想在主 context 中复用 workflow？用 Skill

### Agent Teams（实验功能）

一个 lead + 多个 teammates（独立的 Claude Code 实例），组件：Team lead、Teammates、**共享 Task list**、**Mailbox**（每个 agent 一个 JSON inbox）。Teammates 之间可以直接通信，但不继承你的对话。

适合：多角度 review、独立模块、竞争假设 debugging、跨层（前端/后端/测试）。不适合：顺序任务、同文件编辑、强依赖。

| | Subagents | Agent teams |
|---|---|---|
| Context | 独立，结果回给 caller | 独立 |
| 通信 | 只回 caller | teammate 互相通信 |
| 协调 | lead 管理 | 共享 task list |
| 适合 | 只要结果的任务 | 需要讨论的工作 |
| Token 成本 | 较低 | 较高 |

### Dynamic Workflow

一个用 JavaScript 脚本在后台编排大量 subagents 的机制，适合 codebase 级 bug sweep、500 文件迁移、交叉验证的研究等。启动：prompt 中写 `ultracode`，或 `/effort ultracode`。

### Long-running Agents

- **`/goal "npm run lint exits with zero errors"`**：给出**可验证**的终止状态；evaluator 每次尝试后检查，满足或 `--max-tries` 用尽时停止。“all integration tests pass” 可以，“refactor the code” 不行
- **`/loop 5m "check CI, fix whatever broke"`**：按间隔重复运行同一 prompt，直到你停止；默认 prompt 可放 `.claude/loop.md`

### 案例：16 个 Agent 构建 C 编译器

- Rust 写的 C compiler，能编译 Linux 6.9（x86、ARM、RISC-V），约 10 万行代码
- 16 agents、约 2,000 sessions、两周、2B input / 140M output tokens、约 $20,000
- 能编译 QEMU、FFmpeg、Postgres、Redis、Doom；主要测试集约 99% 通过
- **协调方式很朴素**：每个 agent 独立 container + git checkout；用 lock file（`current_tasks/`）认领任务；**Git 作为协调和集成层**；新 session 读 progress docs 恢复 context；专门的 docs/performance/quality/dedup agents；自动化测试作为进度信号
- 局限：merge conflict 频繁；所有 agent 都碰到同一个 kernel failure 时并行失效；16-bit x86 部分仍依赖 GCC；生成代码效率低于 GCC

---

## 5. Programming / Software 3.0

**讲者定义的 Programming 3.0**：
- 1.0：人 → 机器指令/汇编 → 硬件
- 2.0：人 → programming language → compiler → 硬件
- 3.0：人 → natural language → **agent** → programming language → … → 硬件
- 趋势：生产力更高，**uncertainty 更多**

**Karpathy 的 Software 3.0**：1.0 = source code；2.0 = neural network weights（数据 + 优化）；3.0 = LLM prompts（natural language）。知道 1.0/2.0/3.0 的边界画在哪里是重要的工程能力。

### Uncertainty

Uncertainty 不是新问题（硬件架构、并发、compiler 版本），2.0 都有解决办法；3.0 新增：nondeterministic kernels、floating-point variance、agent 一边写一边改动环境、sampling。

不确定性来源：
- Natural language：semantic uncertainty（单个 prompt 的输入有歧义）
- Retrieval / context：retrieval drift（index 更新、chunking、truncation）
- Model generation：sampling + 软硬件 nondeterminism
- Agent/tool loop：tool failure、环境变化
- 生成的程序：一旦生成通常是确定的（除非运行时调用模型或 tools）

### Best Practices

1. **DEFINE**：先说明 assumptions（有歧义让模型确认）；指定 outcomes 而不是 prompts（requirements、invariants、constraints、acceptance tests）
2. **CONTROL**：proposal 与 execution 分离；显式管理 context（memory、provenance、过期信息）
3. **VERIFY & GOVERN**：收紧 generate-verify loop（频繁便宜的检查胜过最后一次昂贵 review）；measure one level deeper（测试、多指标、可推理的 synthetic workloads）

> 不要期望 agent 是确定性的；要让周围的系统可靠。Autonomy 越高，uncertainty 和 risk 越高 —— 根据风险匹配 autonomy。

---

## 6. 来自 OpenAI Codex 的数据

- 从 **Conversational AI**（人执行）到 **Agentic AI**（人 review 和集成）
- 过去 28 天使用 Codex：个人 < 1%，组织 17.3%，OpenAI 员工几乎全部；2026 上半年活跃用户增长 5 倍以上
- 委派任务变大：≥ 1 human-hour 的任务从 35.4% 到 70.2%；≥ 8 human-hours 的任务从 2.1% 到 25.6%（12 倍）
- **一人多 agent**：28.6% 的 OpenAI 用户一周内同时运行 5+ agents；中位数约 2.5 agent-hours/天，P99 约 71 agent-hours/天
- 人的角色转向：planner、delegator、monitor、reviewer、integrator
- 2026 年 9 月：agent 的工作量已达人类研究员的 3.14 倍；工程师交付速度接近 8 倍；但仍需要人工干预

**Delegation Ladder**：Chatbot → Assistant → Agent → Multi-agent worker → Systematized workflows（OpenAI 员工还处在较低层级）。

---

## 7. Optional：CLI Coding Agents 概况（2026 年中）

- 类别：Model labs（Claude Code、Codex CLI、Antigravity CLI…）、Platform CLIs（Copilot CLI、Cursor CLI…）、Open source（OpenCode、Aider、Cline、OpenHands…）、Minimal cores（Pi、mini-SWE-agent…）
- 五个区分维度：open vs closed；vendor-locked vs BYOK；OS sandbox vs permission prompts；local vs cloud；maximal vs minimal
