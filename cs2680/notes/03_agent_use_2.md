# Lecture 3 · Agent From a User's Perspective II：Multi-agent and Best Practice

> Harvard CS 2680：Modern AI Systems: Agents and System Optimizations
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/3_agent_use_2.pdf`](../slides/3_agent_use_2.pdf)

## 本讲脉络

```
回顾（agent loop、context 与 memory、扩展机制）
→ Case study 1：Superpowers（用 skills + hook 改变 agent 的工作流程）
→ Case study 2：Caveman（用 skill + hook + command + MCP + 本地 proxy 压缩 tokens）
→ 用多个 agent 放大生产力：parallel agents（git worktree）、subagents、agent teams、dynamic workflow、/goal、/loop
→ 案例：16 个 agent 写出能编译 Linux 的 C 编译器
→ Programming / Software 3.0 与不确定性、最佳实践
→ 来自 OpenAI Codex 的使用数据
```

一句话总结：**同一个模型、同一套工具，只要换一套指令和扩展（skills、hooks、proxy），agent 的行为就会大不相同；而真正的生产力放大来自“多个 agent 同时工作”和“一个 agent 长时间工作”。** 随之而来的核心问题是**不确定性**：你不能指望 agent 是确定性的，但可以让它周围的系统变得可靠。

---

## 1. 回顾

- **Agent loop**：模型提议 → harness 执行 → 结果回到 context
- **State management**：
  - Context 是 working memory：每次用户 prompt、LLM 回复、tool 结果都会追加进去
  - Agent memory 跨 session 持久化：CLAUDE.md（或 AGENTS.md）、auto memory
- **Tools**：内置工具 + 扩展能力（skill、MCP、hook、plugin）

---

## 2. Case Study 1：Superpowers

一个软件开发 plugin（https://github.com/obra/superpowers），安装：`/plugin install superpowers@claude-plugins-official`

### 2.1 同一个模型，不同的工作流程

| 默认行为 | Superpowers |
|---|---|
| 收到 feature request → **直接开始写代码** → 靠反复修改来修 bug | 0. 人输入 prompt |
| | 1. **Clarify**：先澄清问题 |
| | 2. **Approve a design**：人批准设计 |
| | 3. **Write a detailed plan**：写详细计划 |
| | 4. **Test-driven build**：测试驱动开发 |
| | 5. **Review and verify**：审查并验证 |
| | 6. **Finish or integrate**：完成或合并分支 |

两个 agent **共享同样的模型和工具，只有指令不同**。这说明 agent 的行为很大程度上由 harness 中的指令决定。

### 2.2 九个 Skills

| Skill | 职责 |
|---|---|
| using-superpowers | 规定如何发现和调用其他 skills 的规则 |
| brainstorming | 把一个请求变成一个被批准的设计 |
| writing-plans | 把设计分解成具体任务 |
| using-git-worktrees | 创建隔离的工作空间 |
| test-driven-development | 强制 red → green → refactor（先写失败的测试，再让它通过，再重构） |
| systematic-debugging | 先找到 root cause，再提出修复 |
| requesting-code-review | 对照计划 review 实现 |
| verification-before-completion | 报告成功之前必须拿出证据 |
| finishing-a-development-branch | merge、PR、保留或清理分支 |

### 2.3 架构

```
superpowers/
├── .claude-plugin/plugin.json     # plugin 元数据
├── hooks/
│   ├── hooks.json
│   └── session-start              # session 启动脚本
├── skills/
│   ├── using-superpowers/
│   ├── brainstorming/
│   ├── writing-plans/
│   ├── test-driven-development/
│   └── …
└── tests/
```

**Plugin 是分发容器，skills 和 hooks 提供实际行为。**

### 2.4 一个 Skill 的解剖

```
skills/brainstorming/
├── SKILL.md
├── visual-companion.md
└── scripts/
```

```markdown
---
name: brainstorming
description: Use before creative work.
---
## Workflow
1. Inspect the project
2. Classify the scope
3. Clarify requirements
4. Compare approaches
5. Present a design
6. Wait for user approval
7. Invoke writing-plans
```

Front matter 是元数据，正文是 agent 要遵循的流程。

**SKILL.md 的六个组成部分**：

| 部分 | 作用 |
|---|---|
| name | 稳定的标识符 |
| description | 什么时候适用 |
| Instructions | 要遵循的流程 |
| Checkpoints | 继续之前需要的批准 |
| References | 只在需要时才看的细节 |
| Scripts | 确定性的操作 |

**一个流程型 skill 可以定义**：进入条件、必须做的动作、**禁止做的动作**、人工审批关卡、验证要求，以及**下一个适用的 skill**（例如 brainstorming 结束后调用 writing-plans）。每个 skill 引导 agent 的行为，并可以指定它的“后继者”，于是多个 skill 串成一条完整流程。

### 2.5 为什么只有 Skills 不够

安装 skill 只是让它们**可用**，并不保证 agent 会去检查它们。模型可能直接开始写代码，根本没想到要先 brainstorm。

Superpowers 的解决办法：**SessionStart hook**。

```json
{
  "hooks": {
    "SessionStart": [{
      "matcher": "startup|clear|compact",
      "hooks": [{ "type": "command", "command": "run-hook.cmd", "async": false }]
    }]
  }
}
```

Hook 保证：
- **确定性的时机**：每个 session 都执行
- 在 agent 行动**之前**运行
- 在 clear 和 compaction **之后重新运行**（`matcher: "startup|clear|compact"`），所以规则不会因为压缩而丢失

### 2.6 Session-start 脚本做了什么

```bash
#!/usr/bin/env bash
set -euo pipefail
ROOT="$(plugin_root)"
skill="$ROOT/skills/using-superpowers/SKILL.md"
content="$(cat "$skill")"
printf '{"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": "%s" }}' \
  "$(escape_json "$content")"
```

1. 找到已安装的 plugin 目录
2. 读取 bootstrap skill（using-superpowers/SKILL.md）
3. 把 Markdown 转义成 JSON 字符串
4. 作为 `additionalContext` 返回
5. Claude Code 把它加入 context

于是每个 session 一开始，模型就“知道”要按 Superpowers 的规则选择 skill。

### 2.7 端到端流程：“Add authentication”

```
Claude Code 启动 session → Hook 读取 bootstrap skill → Agent 收到 skill 选择规则
→ 用户：“Add authentication”
→ Agent 加载 brainstorming，澄清需求 → 用户批准设计
→ Agent 加载 writing-plans → 实现计划 → 隔离的 worktree → 测试驱动开发
→ Code review → Verification → 分支合并
```

### 2.8 设计经验

| 设计问题 | Superpowers 的回答 |
|---|---|
| 复杂行为应该如何表示？ | 小而可组合的 skills |
| Agent 如何记得使用它们？ | 通过 session hook 注入一个 bootstrap skill |
| 如何控制 context 大小？ | 只在相关时加载任务 skill |
| 各阶段如何协调？ | 每个 skill 写明自己的关卡（gates）和后继者 |
| 用户如何安装整个系统？ | 打包成 plugin |
| 每个 plugin 都需要 commands 或 MCP 吗？ | 不需要，只在需要时添加 |

---

## 3. Case Study 2：Caveman

五种扩展机制 + 一个本地 proxy。安装：

```
claude plugin marketplace add JuliusBrussee/caveman
claude plugin install caveman@caveman
```

### 3.1 为什么要压缩 agent 的 tokens

- **Context 被填满**：日志、diff、tool 输出挤占了真正的工作空间
- **每个 turn 都有成本**：tokens 双向都要付钱、都要等待
- **Compaction 会丢细节**：等到 context 满了再压缩，往往会丢掉重要内容

例子：
- 压缩前：`npm test` 输出 786 行，其中包括 `FAIL src/api/client.test.ts`、`connect ETIMEDOUT 10.0.3.8:443 at client.ts:88`，以及 776 行其他内容，最后是 “3 failed, 214 passed”
- 压缩后，模型读到的是：“3 tests fail: ETIMEDOUT at client.ts:88”

原则：**双向压缩，并保留原文以便需要时检索。**

### 3.2 两个方向的压缩

```
AGENT INPUT（日志、tool 结果、diff、文件）→ 本地 proxy + 压缩引擎 → Model
Model → AGENT OUTPUT（你读到的内容）← response skill（早先注入 context）
```

- **Skill 改变模型怎么写**（输出更简洁）
- **Proxy 改变模型读到什么**（输入被压缩）

报告结果：output tokens 减少 65%。但 skill 本身会增加 input tokens，所以整个 session 的节省比这个数字低。

### 3.3 一个完整例子

用户输入 `/caveman ultra`，然后：“Run the tests and explain the failure.”

1. Hook 记录下 ultra 模式
2. Hook 注入“回答要简洁”的提醒
3. Agent（LLM 决定）执行测试命令
4. Caveman proxy 压缩嘈杂的测试输出
5. 模型收到精简后的 context
6. 模型返回简洁的诊断
7. 如需原文，通过 MCP 检索

### 3.4 各组件

**Plugin 结构**：

```
caveman/
├─ .claude-plugin/plugin.json
├─ skills/      # caveman, caveman-commit, caveman-review, caveman-compress, caveman-stats
├─ commands/
├─ src/hooks/
├─ mcp/
├─ proxy/
├─ engine/
└─ agents/
```

```json
{
  "name": "caveman",
  "hooks": {
    "SessionStart": ["caveman-activate.js"],
    "UserPromptSubmit": ["caveman-mode-tracker.js"]
  }
}
```

**1）Response skill：让输出不啰嗦**

```markdown
---
name: caveman
description: Compressed mode
---
# Caveman response mode
Preserve: technical terms, commands, paths, code, errors, numbers, units, negation
Remove: filler, repetition, pleasantries, hedging
Short sentences. Active voice.
```

注意它**保留**的东西：技术术语、命令、路径、代码、错误、数字、单位，尤其是**否定词**（丢掉 “not” 会让意思完全相反）。它改变的是 agent 的沟通方式。

**2）Commands：给用户显式的控制**

| 命令 | 作用 |
|---|---|
| `/caveman` | 启用默认的 full 模式 |
| `/caveman ultra` | 最大压缩 |
| `/caveman off` | 关闭 |
| `/caveman-commit` | 生成简洁的 commit message |
| `/caveman-review` | 生成简洁的 review 意见 |
| `/caveman-compress` | 压缩一个 memory 文件 |
| `/caveman-stats` | 本 session 的 token 统计 |
| `/caveman-help` | 可用操作 |

命令文件示例（`commands/caveman.md`）：带 `argument-hint: [lite|full|ultra|off]`，正文是 “Activate Caveman mode: $ARGUMENTS. If no level is supplied, use full. If off, deactivate.”

**3）Hooks：让 prompt 中的选择持续有效**

| Hook | 脚本 | 做什么 |
|---|---|---|
| SessionStart | `caveman-activate.js` | 决定本 session 的模式；安全地保存模式状态；把规则作为隐藏 context 注入；compaction 后恢复规则；清理旧状态；提供 status line |
| UserPromptSubmit | `caveman-mode-tracker.js` | 检查每个用户 prompt；检测 `/caveman ultra`；检测 “normal mode” 之类的短语；更新 session 的模式；在专门操作后恢复模式 |

SessionStart 初始化行为，UserPromptSubmit 检测对话中途的变化。

**4）MCP：把压缩暴露为可调用的工具**

| 工具 | 作用 |
|---|---|
| `caveman_compress` | 压缩并返回一个恢复句柄 |
| `caveman_retrieve` | 取回逐字节一致的原文 |
| `caveman_stats` | 压缩统计 |
| `caveman_toon_encode` / `decode` | JSON ↔ TOON（一种更省 token 的数据格式） |

返回示例：`{"compressed": "3 tests fail. Root error: timeout.", "ratio": 0.72, "recovery_handle": "ccr_abc123"}`

**5）本地 Proxy：压缩模型读到的内容**

Proxy 位于 coding agent 与模型提供方之间，做**按类型的压缩（type-aware compression）**：

| 内容类型 | 保留什么 |
|---|---|
| 日志 | 错误、stack trace、开头和结尾 |
| 代码 | imports、函数签名、类型、函数体 |
| Diff | 文件和 hunk 头、改动的行 |
| JSON | key、结构、错误子树 |
| 搜索结果 | 高价值的匹配、诊断信息 |
| HTML / 文本 | 标题、相关段落 |

**Proxy vs MCP**：
- Proxy **拦截每一个请求**；MCP 只在被调用时运行
- Proxy 位于 agent 和 provider **之间**；MCP 位于 agent **旁边**
- **Proxy 能改变 context；MCP 不能**

为什么需要 proxy？**Hook 可以拦截请求，但不能透明地改写发给 LLM 的请求内容**，所以需要 proxy。如果你自己写 harness，就可以直接在 harness 里做。

### 3.5 Caveman 如何使用各种能力

| 层 | 组件 | 作用 |
|---|---|---|
| 打包 | Plugin | 一个安装和更新单元 |
| 机制 | Skills | 简洁风格、专注的工作流 |
| | Hooks | 激活、跟踪、强化模式 |
| | Commands | 选择模式、请求操作 |
| | MCP | 压缩、检索、统计、编码 |
| 状态与引擎 | Session mode state、compression engine | 记录模式、执行压缩 |
| 拦截 | 本地 proxy | 拦截发往 provider 的请求 |

---

## 4. 用多个长时间运行的 agent 成为 100x 工程师

### 4.1 三个扩展维度

| 维度 | 方式 | 带来什么 |
|---|---|---|
| **Breadth（广度）** | **Parallel agents**：同时跑多个独立任务，每个在隔离的 working tree 中 | Throughput：节省 wall-clock 时间，**但不提高答案质量** |
| **Breadth + depth** | **Subagents**：把一个有边界的任务交给一个新的 context，拿回 summary | Throughput + focus：主 agent 永远看不到中间的混乱过程 |
| **Duration（时长）** | **Long-running agent**：让一个 agent 跨很多步完成更大的工作 | Scope：更少干预、完成更多工作 |

Agent team 和 dynamic workflow 是这些方式的组合。注意：agent 并行“扇出”的同时，**账单也在扇出**。

### 4.2 手动运行并行 agents：Git Worktree

**每个任务一个 session，每个 session 一个 git worktree**：

```bash
claude -w feature-payments    # 终端 1：在自己的分支上开发新功能
claude -w bugfix-auth         # 终端 2：修 bug，不受终端 1 影响
```

**Git worktree 是什么**：同一个 repository 的多个工作目录，每个目录检出不同的分支，但**共享同一份 `.git`**（历史、分支、对象、远程）。

```
一个 repository（.git：history · branches · objects · remotes）
├── ../repo-main   — branch: main            → Agent 1：review
├── ../feat-auth   — branch: feature/auth    → Agent 2：开发
└── ../fix-flaky   — branch: fix/flaky-test  → Agent 3：debug
```

好处：一个 session 的编辑永远不会碰到另一个 session 的文件；不需要 stash，也不需要第二次 clone。在桌面 app 中，每个新 session 自动获得一个 worktree。

**手动并行的极限**：
- **你掌握计划**：每次交接和下一步都在你脑子里
- **Session 之间互相看不到**：发现只能靠你手动传递
- **你来协调分支**：合并顺序和冲突都要你处理
- **不可重复**：运行过程没有记录下来，无法重跑

3 个 session 你还能 hold 住；20 个 session 就需要一个“你的分身”了。

### 4.3 Subagents

**是什么**：一个专门的助手，有**自己的 context window、工具和权限**。

**为什么用**：

| 好处 | 说明 |
|---|---|
| **Preserve context** | 日志、搜索结果、文件 dump 留在 subagent 里，不污染主对话 |
| **Enforce constraints** | 限制 subagent 能用哪些工具、用哪个模型 |
| **Reuse anywhere** | 针对某个领域的 prompt 可以跨项目、跨团队共享 |
| **Parallelization** | 独立的子任务并发运行，总时间由最慢的那个决定（例如 style-checker、security-scanner、test-coverage 同时跑） |
| **Control costs** | 把任务交给更快、更便宜的模型（在 frontmatter 中写 `model: haiku`） |

**内置 subagents**（Claude 会自动委派）：

| Subagent | 用途 |
|---|---|
| **Explore** | 只读地搜索代码库；禁止 Write 和 Edit；每次调用可设定彻底程度 |
| **Plan** | 在 plan mode 中做调研，让探索过程不进入主窗口 |
| **General-purpose** | 既需要探索又需要行动的复杂多步任务 |

Explore 和 Plan 会跳过 CLAUDE.md 和 git status，以保持快速、便宜。

**定义一个 subagent**：一个带 YAML frontmatter 的 Markdown 文件，正文就是它的 system prompt。

```markdown
---
name: code-reviewer
description: Reviews code for quality
tools: Read, Glob, Grep
model: sonnet
---
You are a code reviewer. Analyze the code and give actionable feedback.
```

- `.claude/agents/`：project scope，提交到 version control 与团队共享
- `~/.claude/agents/`：user scope，你所有的项目都能用

**嵌套 subagent**：
- 要让 subagent 能再派生子 agent，它的 `tools` 中必须包含 `Agent`
- 最多允许 **5 层**嵌套，**2–3 层是最佳点**
- 每一层都从**全新的 context** 开始，只返回 summary

常见的多层模式：

```
Level 0：主 agent（掌握计划）
Level 1：Implementer subagent
Level 2：Test-writer、Debugging、Code-review、Verification subagents
```

**使用 subagent 的三种方式**（控制力递增）：

| 方式 | 例子 | 范围 |
|---|---|---|
| 自然语言 | “Use the test-runner subagent to fix tests”（Claude 决定是否委派） | 一次性 |
| @-mention | `@agent-code-reviewer look at auth changes`（保证运行） | 一个任务 |
| `--agent` 参数 | `claude --agent code-reviewer`（整个 session 采用它的 prompt、工具和模型） | 整个 session |

**什么时候用 subagent，什么时候留在主对话**：subagent 从零开始，看不到历史，需要付出收集 context 的延迟。

| 留在主对话 | 交给 subagent |
|---|---|
| 需要反复来回的交互 | 冗长且不会再用的输出 |
| 多个阶段共享 context | 需要工具或权限限制 |
| 快速、有针对性的修改 | 自包含、只需返回 summary 的工作 |
| 延迟很重要 | 并行、独立的工作 |
| | 可以交给小模型的简单工作 |

想在主 context 中复用一套工作流程？用 **Skill**。

### 4.4 Agent Teams（实验功能，默认关闭）

**几个 Claude Code session 协同工作：一个 lead，若干 teammates。**

适合场景：
- **Review**：teammates 从不同角度审查，然后互相质疑对方的发现
- **独立模块**：每个 teammate 负责一个模块，互不干扰
- **Debugging**：并行测试相互竞争的假设，更快收敛
- **跨层**：前端、后端、测试各由一个 teammate 负责

不是免费的：协调开销和 token 成本随团队规模增长。顺序性任务、编辑同一文件、依赖很重的工作，应该用单个 session 或 subagents。

**团队的四个部分**：

| 部分 | 作用 |
|---|---|
| **Team lead** | 主 session：创建 teammates、协调工作、综合结果 |
| **Teammates** | 独立的 Claude Code 实例，各有自己的 context window |
| **Task list** | 共享的工作项，teammates 认领并标记完成 |
| **Mailbox** | 每个 agent 一个 JSON 收件箱，teammates 可以**直接互相发消息** |

Teammates **不继承你的对话**；它们会自动加载 CLAUDE.md 和 skills，任务细节要写在 prompt 里。

**Subagents vs Agent teams**：

| | Subagents | Agent teams |
|---|---|---|
| Context | 独立窗口，结果返回给调用者 | 独立窗口，完全独立 |
| 通信 | 只能回给调用者 | teammate 之间直接通信 |
| 协调 | lead 管理所有工作 | 共享 task list |
| 最适合 | 只需要结果的任务 | 需要讨论的工作 |
| Token 成本 | 较低 | 较高 |

团队更贵，只在工作确实需要讨论时使用。

### 4.5 Dynamic Workflow

**一个在后台编排大量 subagents 的 JavaScript 脚本。** 当任务需要的 agent 多到一个对话协调不过来时使用：
- 整个代码库范围的 bug 扫描
- 500 个文件的迁移
- 多方交叉验证的调研
- 从多个角度制定计划

**两种启动方式**：在 prompt 中输入 `ultracode`；或用 `/effort ultracode` 让 Claude 自行决定。两种方式都会派生大量 subagents，负责计划、设计、实现、审查。

关键区别：**计划保存在脚本里，而不是聊天记录里**，因此可以重复、可以扩展到很大规模。

### 4.6 Long-running Agents

**`/goal`：给出一个可验证的终态，Claude 一直工作直到它成立**

```
/goal "npm run lint exits with zero errors"
```

- **定义完成**：一个 Claude 可以检验的条件（测试通过、零 lint 错误）
- **自我检查**：每次尝试后，evaluator 重新检查条件，不满足就把 Claude 送回去继续
- **自己停止**：目标达成，或 `--max-tries` 用完
- 关键是**可验证**：“all integration tests pass” 可以；“refactor the code” 不行（没法判断什么时候算完成）

**`/loop`：按时间间隔重复运行一个 prompt，直到你停止它**

```
/loop 5m "check CI, fix whatever broke"
```

- **是触发器，不是测试**：你指定间隔，同一个 prompt 每次重新运行
- **你来结束**：在两次运行之间的等待期按 Esc
- 可以在 `.claude/loop.md` 中放默认 prompt

**区别**：`/goal` 在条件成立时停止；`/loop` 在你要求时停止。

### 4.7 小结

| 更多 agent 同时工作 | 一个 agent 工作更久 |
|---|---|
| Parallel agents（手动）—— 你自己协调 | `/goal`：运行到一个可验证的终态 |
| Subagents —— Claude 委派，结果返回给它 | `/loop`：按计划重复运行直到你停止 |
| Agent teams —— lead 通过共享列表管理同伴 | |
| Workflows —— 计划在脚本里，不在聊天里 | |

---

## 5. 选读：16 个 Agent 写出一个 C 编译器

https://www.anthropic.com/engineering/building-c-compiler

### 5.1 规模与结果

| 规模 | 结果 |
|---|---|
| 16 个 agent 并行工作 | 用 Rust 写的 C 编译器，能编译 Linux 6.9（x86、ARM、RISC-V） |
| 约 2,000 个 session，历时两周 | 约 100,000 行代码 |
| 20 亿 input / 1.4 亿 output tokens | 能编译 QEMU、FFmpeg、Postgres、Redis、Doom |
| API 成本略低于 $20,000 | 主要测试集通过率约 99% |

### 5.2 如何协调：朴素的基础设施

没有聊天，没有 manager agent —— **agents 通过共享的 Git 状态协调**：

- 每个 agent 在**隔离的容器**中，有自己的 Git checkout
- 通过简单的 **lock 文件**认领任务
- **Git 作为协调和集成层**
- 新 session 通过读**进度文档**重新获得 context
- 有专门负责文档、性能、质量、去重的 agents
- **自动化测试**给 agents 提供进度信号

**每个 agent 的循环**：

```
读 repo（clone、扫描文档和日志）
→ 认领任务（在 current_tasks/ 中创建 lock 文件）
→ 在自己的容器中实现并测试
→ 先拉取别人的改动，再 merge、push、解锁
→ 重复：agent 自己挑选下一个明显的任务
```

- **冲突自行解决**：同一个任务被认领两次？一个 agent 会转去做别的；Git 解决重复认领
- **Artifacts 携带 context**：README、进度文档、CI 失败信息为新 session 提供简报

### 5.3 成功，但并不完美

- Merge conflict 很频繁
- 当所有 agent 都卡在同一个 kernel 失败上时，并行就不再有帮助
- 16 位 x86 的部分功能仍依赖 GCC
- 生成的代码效率不如 GCC

Agent 还不完美，你还有很多工作要做。

---

## 6. Programming / Software 3.0

### 6.1 讲者定义的 Programming 3.0

| 阶段 | 流程 |
|---|---|
| **Programming 1.0**（编程硬件） | 人 → 显式的机器指令或汇编 → 硬件 |
| **Programming 2.0**（编程语言） | 人 → 编程语言 → compiler → 机器指令 → 硬件 |
| **Programming 3.0**（表达目标） | 人 → **自然语言** → **agent** → 编程语言 → 机器指令 → 硬件 |

趋势：**生产力越来越高，不确定性也越来越多。**

### 6.2 Karpathy 的 Software 3.0

| 范式 | 程序是什么 | 如何创造 |
|---|---|---|
| 1.0 | 源代码 | 写 Python / C++ |
| 2.0 | 神经网络权重 | 收集数据 + 优化 |
| 3.0 | LLM prompts | 自然语言 |

- Software 1.0：人 → 代码 → 计算机
- Software 2.0：人 → 数据 → 优化器 → 神经网络
- Software 3.0：人 → 语言 / context → LLM → 行动

**知道在哪里划分 1.0、2.0、3.0 的边界，是一项重要的工程技能。** 例如：能用规则精确表达的逻辑用 1.0；有大量数据的感知任务用 2.0；需要灵活理解和推理的用 3.0。

### 6.3 不确定性

**不确定性并不是新问题。** “同样的源代码 ≠ 同样的输出”早就存在：

| 不确定性来源 | Programming 2.0 如何解决 | Programming 3.0 新增 |
|---|---|---|
| **硬件**：32 位 vs 64 位、字节序 | 固定宽度类型、交叉编译、虚拟化 | GPU 计算：非确定性的 kernel、浮点误差 |
| **环境**：并发、不可靠的网络 | memory consistency model、幂等 API | agent 一边写代码一边改动环境 |
| **软件**：compiler 版本、依赖漂移 | 升级前全面测试 | compiler 变成了模型：不同 kernel、采样 |

**从意图到结果，3.0 多了一层不确定性**：
- 1.0 / 2.0：人（PM）→ 程序员 → 程序 → 结果。人的需求本来就模糊，但这层不确定性常被忽略（或靠形式化验证解决）
- 3.0：人 → LLM agent → 程序 → 结果。单个 prompt 的输入本身就有歧义，这是需要更多研究的地方

### 6.4 学会与不确定性共存

| 环节 | 不确定性类型 | 具体来源 |
|---|---|---|
| 自然语言 | 语义不确定性 | prompt 有歧义 |
| Retrieval / context | 检索漂移 | 索引更新、chunking、context 截断 |
| 模型生成 | 概率变化 | 采样；软硬件的非确定性 |
| Agent / tool 循环 | 轨迹 / 环境的非确定性 | 工具失败、环境变化 |
| 生成的程序 | 一旦生成通常是确定的 | 除非运行时还会调用模型或工具 |

### 6.5 Programming / Software 3.0 的最佳实践

**1 · DEFINE（定义）**
- **先说明假设**：有歧义的地方让模型在执行前先确认
- **指定结果，而不是指定 prompt**：给出需求、不变量、约束和验收测试

**2 · CONTROL（控制）**
- **把提议与执行分开**：agent 提出计划；在任何东西被提交之前先验证
- **显式管理 context**：自己掌握 memory、来源（provenance）和检索；让过时信息过期

**3 · VERIFY & GOVERN（验证与治理）**
- **收紧“生成—验证”循环**：频繁、便宜的检查胜过最后一次昂贵的 review
- **多测一层**：写测试、测多个指标、用你能推理的合成负载

> **不要指望 agent 是确定性的；让它周围的系统变得可靠。**
> **按风险匹配自主程度：自主程度越高，不确定性和风险越大。**

---

## 7. 来自真实世界的证据：OpenAI Codex

### 7.1 从对话到委派（2026 年 6 月报告）

| Conversational AI | Agentic AI（Codex） |
|---|---|
| 人问一个问题 → 模型回答 → **人去执行工作** | 人委派一个任务 → agent 规划、调用工具、编辑文件、运行命令 → 交付 artifact 或动作 |
| 人是执行者 | **人负责 review 和集成** |

过去 28 天用过 Codex 的比例：个人用户 < 1%；组织 17.3%；OpenAI 员工几乎全部。2026 年上半年活跃用户增长超过 5 倍。

### 7.2 委派的任务越来越大

| 任务规模 | 2025 年 12 月 | 2026 年 5 月 |
|---|---|---|
| 任务 ≥ 1 个人类工时 | 35.4% | 70.2% |
| 任务 ≥ 8 个人类工时（一整天的工作） | 2.1% | 25.6%（**12 倍**） |

### 7.3 一个人，多个 agent

- 28.6% 的 OpenAI 用户在一周内同时运行过 5 个以上 agent
- OpenAI 员工的中位数：每天约 **2.5 agent-hours**
- 第 99 百分位：每天约 **71 agent-hours**（比 4 月增长 88%）

人的角色转变为：**planner（规划者）、delegator（委派者）、monitor（监督者）、reviewer（审查者）、integrator（集成者）**。

### 7.4 OpenAI 研究员的情况（2026 年 9 月报告）

- 2026 年 6 月之前，agent 的运行时长还落后于人类劳动时长；**现在 agent 完成的工作量是人类的 3.14 倍**
- 大多数研究员同时有 4 个以上的工作流
- 工程师的交付速度快了将近 **8 倍**
- Agent 能解决越来越复杂的任务，**但仍然需要人工干预**

### 7.5 Delegation Ladder（委派阶梯）

1. **Chatbot**：AI 回答，人做事
2. **Assistant**：人和 AI 一起完成任务
3. **Agent**：人把一整个工作单元委派出去
4. **Multi-agent worker**：人协调多个 agent
5. **Systematized workflows**：skills 和 agents 运行整套流程

即使是 OpenAI 的员工，整体也还处在阶梯的较低位置。

---

## 8. 选读：CLI Coding Agents 概况（2026 年中）

### 8.1 四类

| 类别 | 特点 | 代表 |
|---|---|---|
| Model labs | 模型和 harness 一体 | Claude Code、Codex CLI、Antigravity CLI、Grok Build、Mistral Vibe、Kimi Code、Qwen Code |
| Platform CLIs | 嵌入某个开发平台 | Copilot CLI、Cursor CLI、Amp、Auggie、Droid、Junie CLI、Qoder、CodeBuddy |
| Open source | 自带 API key | OpenCode、Crush、Goose、Aider、Cline、Kilo、Continue、OpenHands、Reasonix、Codebuff、DeepSeek Harness |
| Minimal cores | 小而可审计 | Pi、oh-my-pi（Omp）、mini-SWE-agent、gptme、Open Interpreter |

### 8.2 五个区分维度

| 维度 | 一端 | 另一端 |
|---|---|---|
| Open vs closed | 开源：Codex CLI、OpenCode、Goose、Cline、Pi | 闭源：Claude Code、Copilot CLI、Cursor、Droid、Junie |
| Locked vs BYOK | 单一厂商：Claude Code、Antigravity CLI、Grok Build | 任意端点：OpenCode（75+ provider）、Kilo（500+ 模型） |
| Sandbox vs prompt | OS 级 sandbox：Codex CLI、Copilot CLI、OpenHands 容器 | 权限提示：Claude Code 和大多数其他工具 |
| Local vs cloud | 本地优先、无遥测：Goose、Nanocoder、gptme | 交给云端：Copilot CLI、Codex、Mistral Vibe 的远程 agent |
| Maximal vs minimal | 功能齐全：Omp（功能最多）、Claude Code | 只有四个工具：Pi、mini-SWE-agent（约 100 行 Python） |
