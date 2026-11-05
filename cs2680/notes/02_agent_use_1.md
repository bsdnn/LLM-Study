# Lecture 2 · Agent From a User's Perspective I：How Claude Code Works

> Harvard CS 2680：Modern AI Systems: Agents and System Optimizations（Juncheng Yang，Fall 2026）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/2_agent_use_1.pdf`](../slides/2_agent_use_1.pdf)
> 预计阅读时间：15–18 分钟。不需要预备知识。

## 本讲脉络

```
你输入一个 prompt 之后发生了什么？
→ Agentic loop：model 提议、harness 执行、结果回到 context，循环多次
→ Context：开始前已有的内容、按需拉取代码、permissions、增长与 compaction
→ Memory：为什么需要、CLAUDE.md 与 auto memory、放在哪、写什么、代价
→ Tools：核心工具集
→ 扩展 agent 的四种机制：Skill / MCP / Hook / Plugin
→（选读）MCP 深入：架构、三个 primitives、安全、gateway、新版 spec
```

一句话总结：**LLM 本身是无状态的、只会“说”不会“做”；Claude Code 这样的 agent 是在 LLM 外面包一层 harness：它负责把合适的信息放进 context、把模型提出的 tool 调用真正执行、把结果喂回去，循环直到任务完成。** Memory 让知识跨 session 保留；Skill / MCP / Hook / Plugin 分别解决“怎么做、能访问什么、何时自动执行、如何分发”这四个问题。

> 本课程的约定：暂时把 LLM 当作一个“会推理的魔法盒子”，不需要了解内部原理也能学 agent；LLM 内部原理放在课程后半段（Lecture 9 起）。

---

## 1. The Agentic Loop：一个 prompt 背后是多次模型调用

### 1.1 三步循环

当你在 Claude Code 中输入 “Fix the failing auth test”，并不是发一次请求、拿一次回答，而是进入一个循环：

| 步骤 | 谁来做 | 做什么 |
|---|---|---|
| 1. **Gather context** | harness | 收集 prompt、相关文件、上一步 tool 的结果 |
| 2. **Generate action** | model | 决定下一步做什么，输出一个 tool 调用请求，例如 `Bash("pytest")`、`Read(...)`、`Edit(...)` |
| 3. **Run tool** | harness（在你的机器上） | 真正执行：跑测试、读输出、重新读取文件状态 |

然后回到第 1 步。每一步都**以上一步的结果为条件**：模型看到测试失败的输出，才知道下一步该读哪个文件。

三个角色要分清：
- **Model**：只负责“想”，输出文字或 tool 调用请求
- **Harness**（Claude Code 程序本身）：负责组装 context、检查权限、执行 tool、管理 context
- **Your machine**：tool 真正运行的地方（文件系统、shell）

### 1.2 一个具体的执行轨迹

```
U  → M₁ → T₁ → M₂ → T₂ → … → Mₙ → A
用户   推理   工具   推理   工具        推理   回答
```

以 “Fix the failing auth test” 为例，实际可能是这样：

1. M₁ 推理：“先跑测试看看哪里失败” → T₁：`pytest`，输出显示 `test_create_session` 失败
2. M₂ 推理：“看看 session 是怎么创建的” → T₂：`grep create_session`
3. M₃ → T₃：`read session.py`
4. M₄ → T₄：`edit session.py`（修复 bug）
5. M₅ → T₅：`pytest` → all green
6. M₆ → 最终回答：“已修复，原因是……”

**一个用户 turn 展开成了推理与工具执行交替的长链。** 这也意味着：一次用户请求的成本和延迟，是很多次 LLM 调用的总和。

### 1.3 你还没打字时，context 里已经有 10K+ tokens

模型每次被调用时看到的内容（context）大致包括：

| 部分 | 内容 | 说明 |
|---|---|---|
| SYSTEM | Claude Code 的 system instructions | 告诉模型它是谁、该怎么工作 |
| PROJECT | 环境信息 | 工作目录、操作系统、git 状态等 |
| MEMORY | CLAUDE.md、用户/项目笔记 | 跨 session 保留的知识（见第 2 节） |
| TOOLS | Read、Edit、Bash、Glob、Grep、Web、Agent… 的定义 | 每个 tool 的名字、描述、参数 schema |
| SKILLS / MCP | 只有 name 和 description | **完整内容等真正要用时才加载**，节省 context |
| USER | 你的 prompt | 例如 “Fix the failing auth test” |

### 1.4 它不会把整个 repo 发给模型

一个代码库可能有几千个文件、几百万 tokens，远超 context 容量，也极其昂贵。所以 repo **留在磁盘上**，由模型按需“拉取切片”：

```
Claude 想：“我需要什么？” → Grep("authentication") → 只返回几行匹配结果 → 这几行进入 context
```

这和人类工程师的工作方式一样：不会通读整个代码库，而是搜索、跳转、只读相关部分。

### 1.5 Model 提议，Harness 执行

- 模型**从不直接执行任何东西**，它只是输出一个结构化请求，例如 `Bash("pytest")`
- Harness 解析这个请求、检查权限、在你的机器上执行
- 执行的 stdout / stderr 作为 `tool_result` 放回模型的 context

这种分离很重要：**安全控制点在 harness**，而不是寄希望于模型“自觉”。

### 1.6 Permissions：意图与执行之间的关卡

模型发出的 tool 请求只是一个**提议（proposal）**，需要你或另一个模型批准。例如模型请求 `Bash("rm -rf build/")`：

| 模式 | 行为 |
|---|---|
| Manual | 每个有风险的操作都要你确认 |
| Accept Edits | 自动接受文件编辑，其他操作仍需确认 |
| Bypass permission | 全部自动执行（风险最高） |
| Auto | 用一个小模型检查请求是否安全，安全则放行 |

结果有三种：**allowed**（命令执行）、**ask**（先问你）、**denied**（什么都不发生）。

### 1.7 Context 会增长，然后被 Compaction

- 典型 context window：**1M tokens**（Sonnet / Opus / Fable、GPT 5.6）；今年早些时候 256K 还很常见
- 每一步都在增加 tokens：模型回复、文件内容、命令输出……

**Context 不是免费的**。为什么？每次调用模型，**整个 context 都要重新发送**一次。第 1 步发 10K，第 2 步发 12K，第 50 步可能发 200K。服务端会把已经算过的前缀缓存起来（**prefix cache**），命中缓存的部分按较低价格计费（通常约为正常输入价格的 1/10）。但因为每一步都要重读前面所有内容，**prefix-cache reads 仍然占了大部分成本**。

**Compaction（压缩）**：context 快满时，harness 让模型把旧的 trajectory **总结成摘要**，用摘要替换原始历史，然后**重新注入 CLAUDE.md 和 memory**（防止关键规则丢失）。代价是细节会丢失。

Context management 是 harness 设计的核心组成部分（Lecture 4 详细讨论）。

### 1.8 小结

```
Prompt + context → Model 提议 → Harness 执行 → 结果并入 context → 循环直到任务完成
```

---

## 2. Memory

### 2.1 Agent 的组成

```
Agent = LLM（无状态，负责 reasoning & planning）
      + State（context + memory）
      + Tools（行动、观察 environment）
```

- LLM 本身是 **stateless（无状态）** 的：每次调用之间不记得任何东西
- Agent 在不同 session 之间也是无状态的：**每个新 session 都从零开始**
- 实践中，所谓 “memory” 就是 **session 启动时加载进 context 的文件**

两种 memory：

| | Context = Working memory | Persistent memory |
|---|---|---|
| 是什么 | 模型此刻能看到的内容 | 跨 session 保留、下次重新加载的内容 |
| 生命周期 | 随 session 结束而消失 | 持久存在 |

> 每创建一条 memory，就是做了一个决定：**以后每个 session 都要重读它**。

### 2.2 为什么 agent 需要 memory

| 问题 | 例子 |
|---|---|
| **Repeated instructions**（重复的指令） | 每个 session 都要重新解释：“测试用 `pnpm test`，不要用 npm” |
| **Lost decisions / assumptions**（决策丢失） | 上次决定“为了兼容旧客户端保留 v1 API”，compaction 后这个“为什么”消失了 |
| **Repeated discovery**（重复探索） | agent 每次都重新读一遍相同的文件来了解项目结构 |
| **Inconsistent behavior**（行为不一致） | 不同 session 对同一问题做出不同选择 |

**Memory 让 instruction 变成 convention（惯例）。**

### 2.3 两套系统：CLAUDE.md 与 Auto memory

两者都在每个 session 开始时加载进 context。

| | CLAUDE.md 文件 | Auto memory |
|---|---|---|
| 谁写 | **你** | **Claude 自己** |
| 内容 | Instructions、rules（规则和指令） | Learnings、patterns（学到的东西和模式） |
| Scope | Project、user 或 organization | 每个 repository，多个 worktree 共享 |
| 加载 | 每个 session | 每个 session（只加载前 200 行 / 25KB） |
| 用途 | 规范、工作流程、架构说明 | 你的偏好、你的纠正、项目背景 |

### 2.4 CLAUDE.md 放在哪里

按 scope 从宽到窄依次加载，**离当前目录最近的文件最后读**（因此在冲突时通常更有分量）：

| 层级 | 位置 | 用途 |
|---|---|---|
| Managed policy | 系统级目录（如 `…/ClaudeCode/CLAUDE.md`） | 组织统一下发，**不能被排除** |
| User instructions | `~/.claude/CLAUDE.md` | 你个人的偏好，所有项目通用 |
| Project instructions | `./CLAUDE.md` 或 `./.claude/CLAUDE.md` | 团队共享，放进 source control |
| Local instructions | `./CLAUDE.local.md` | 个人的项目设置，gitignored，不提交 |

### 2.5 CLAUDE.md 里该写什么

**应该写**：
- **命令**：怎么跑测试、build、linter（例：`pnpm test -- --watch=false`）
- **硬约束**：永远不要直接 commit 到 main；永远不要编辑生成的文件
- **agent 不能便宜地推断出的事实**：目录布局、谁负责哪个模块、已知的坑（gotchas）

**不该写**：
- 模型已经知道的语言或框架知识（“Python 用缩进表示代码块”）
- linter 或 formatter 已经强制执行的风格规则
- 长篇大论、changelog、任何 secret

**写法要求**：

| 维度 | 要求 |
|---|---|
| Size | 每个文件 200 行以内 |
| Structure | 用 Markdown 标题和列表 |
| Specificity | 具体规则，而不是模糊的（“函数不超过 50 行”而不是“保持代码简洁”） |
| Consistency | 没有互相矛盾或过时的规则 |

> **Context is rented, memory is owned.**（context 是租来的，memory 是你拥有的 —— 要谨慎选择拥有什么。）

### 2.6 太长了？拆到 `.claude/rules/`

```
.claude/rules/
├── testing.md
├── api-style.md
└── db-migrations.md
```

- **一个文件一个主题**：每个规则文件只关心一件事，保持简短具体
- **加 `paths:` frontmatter**：只有当 Claude 操作匹配的文件时才加载（没有 `paths:` 的规则每个 session 都加载）。例如 `db-migrations.md` 只在编辑 `migrations/**` 时加载
- 好处：只加载与当前工作相关的规则，context 保持精简；CLAUDE.md 保持简短，细节放在 rules 里

### 2.7 Auto memory：Claude 自己的笔记

```
~/.claude/projects/<project>/memory/
├── MEMORY.md              # 索引
├── user_role.md
├── feedback_testing.md
└── other-topic-files.md
```

四种类型（每个文件用 type 字段标记）：

| 类型 | 记什么 | 例子 |
|---|---|---|
| `user` | 你的角色、专长、工作偏好 | “用户是后端工程师，熟悉 Go，不熟悉前端” |
| `feedback` | 你给出的纠正、你确认过的做法 | “用户要求测试必须用真实数据库，不要 mock” |
| `project` | 进行中的工作、代码里看不出来的决策 | “正在迁移到 v2 API，v1 保留到 12 月” |
| `reference` | 项目之外的信息 | “内部 API 文档在 wiki 某页” |

行为：
- **MEMORY.md 是索引**，每个 session 加载它的前 200 行（25KB）
- 存在本机上；同一 repo 的多个 worktree 共享这个目录
- 默认开启，可以用 `/memory` 关闭

### 2.8 检索靠索引

```
Session 开始 → 加载 MEMORY.md 索引 → 判断哪个 topic 相关 → 按需读取该 topic 文件 → 应用到任务
```

索引始终在 context 中，但 topic 文件只在相关时才读 —— 这样 context window 保持较小。

### 2.9 查看和控制 memory 的命令

| 命令 | 作用 |
|---|---|
| `/memory` | 浏览、编辑、删除或关闭 memory |
| `/context` | 查看加载了哪些指令文件 |
| `/init` | 生成一个初始 CLAUDE.md |
| `/doctor` | 找出可能不该放在 CLAUDE.md 里的内容 |

### 2.10 记忆的代价

Context 有限，而且你为发给模型的每个 token 付费：
- **每个永久 token 都是“租金”**，每个 session 开始都要付一次
- Memory 与任务**争夺同一个 context window**
- **过时的 memory 比没有更糟**：它会主动误导模型（例如记着“用 npm”，项目早就换成 pnpm 了）
- 像对待代码一样对待 memory：review、修剪、放进 version control

---

## 3. Tools：让 agent 能行动

Agent 的完整图景：LLM 负责推理和规划，通过 **Tool / Action** 作用于 **Environment**，通过 **Observation** 看到结果，并借助 **Memory** 保持状态。

### 核心工具

| Tool | 作用 | 说明 |
|---|---|---|
| **Read** | 打开文件、检查状态 | 最常用的工具 |
| **Search** | grep 和 glob | 不必读完所有文件就能找到目标 |
| **Fetch** | 拉取网页或文档 | 注意：会把 **untrusted text** 带进 context（可能有 prompt injection） |
| **Edit / Write** | 修改文件 | 第一个**有后果**的工具 |
| **Shell** | build、test、git | 如果不加 sandbox，等于拥有整台机器的权限 |
| **Delegate** | 启动 subagent | 一个返回 summary 的工具（见 Lecture 3） |

用哪个工具由模型自己决定，使用者通常不需要干预；如果你自己写 harness，这些工具就是最先要实现的部分。

---

## 4. 扩展 Agent：Skill、MCP、Hook、Plugin

### 4.1 四种机制解决四个不同问题，不能互相替代

| 机制 | 解决的问题 | 一句话 |
|---|---|---|
| **Skill** | agent **应该如何**执行某个任务 | 知识和流程（decides how） |
| **MCP** | agent **能访问哪些**外部数据和动作 | 外部能力（reaches outside） |
| **Hook** | 在某个 lifecycle event **自动发生什么** | 强制执行（enforces） |
| **Plugin** | 这些扩展**如何打包和分发** | 打包分发（distributes） |

### 4.2 它们挂在 agent loop 的哪里

```
User request
   ↓
Model reasoning      ← Skill 提供知识和流程
   ↓
Tool selection       ← MCP 提供外部工具
   ↓
Tool execution       ← Hook 可以在任何阶段检查、拦截或响应
（Plugin 把 skills、MCP servers、hooks 打包成一个可安装单元）
```

### 4.3 例子：发布流程

任务：准备一个 release，验证它，发布到 GitHub，通知团队。

| 需求 | 机制 |
|---|---|
| 按照 release 流程一步步做 | **Skill** |
| 在 GitHub 上创建 release | **MCP**（GitHub MCP server） |
| 测试失败时阻止发布 | **Hook**（确定性地拦截，不依赖模型自觉） |
| 把整套配置分享给团队 | **Plugin** |

### 4.4 Skill：可复用的流程

**Skill 携带什么**：领域知识、参考资料、可重复的工作流；可以由模型调用，也可以由用户调用；在运行时由模型推理如何应用。本质上是一种 **context engineering**：在需要的时候把正确的知识放进 context。

**目录结构**：

```
.claude/skills/release/
├── SKILL.md          # 必需：frontmatter + 流程正文
├── references/       # 可选：参考资料
└── scripts/          # 可选：确定性的脚本
```

```markdown
---
name: release
description: Prepare a release
---
1. Read the release notes
2. Confirm the version
3. Run the tests
4. Prepare the commit
5. Ask before publishing
```

- **description 控制发现**：模型靠 description 判断什么时候该用这个 skill
- **正文是流程**：模型加载后照着做
- 附加文件提供参考资料和脚本
- 用户也可以直接运行：`/release 2.4.0`

**两种调用方式与 context 成本**：

| Model-invoked | User-invoked |
|---|---|
| 当 description 与任务匹配时，Claude 自动加载 | 用户显式运行 `/<skill-name>` |

- session 开始时只加载所有 skill 的 **description**，正文只在 skill 运行时才加载
- **有副作用的 workflow（如发布、部署）通常应要求显式调用**，避免模型“自作主张”
- description 写得清晰、互不重叠，可以减少选错 skill

### 4.5 MCP：外部能力层

**Model Context Protocol** 通过一个标准协议把 agent 连接到外部服务，例如：读取和更新 GitHub issues、查询数据库、获取监控数据、访问 Figma 设计、在 Slack 发消息、控制浏览器。

MCP 给了 Claude **既不在模型里、也不在本地文件里**的能力。

**一次 MCP tool 调用的六步**：
1. Claude 识别出需要某个动作
2. MCP client 调用工具
3. MCP server 处理连接
4. 外部服务执行操作
5. 结构化结果返回给 Claude
6. Claude 评估结果

对模型来说，MCP tool **与本地 tool 几乎一样**：同样有名字、描述、typed schema，同样是“调用—返回”的契约，只是运行在另一个进程里。**协议负责传递调用，是否调用仍由 agent 决定。**

### 4.6 Hook：生命周期自动化

一个 session 从左到右推进，hook 可以在以下事件触发：

| 事件 | 典型用途 |
|---|---|
| **SessionStart** | 加载 context、记录日志 |
| **UserPromptSubmit** | 验证或补充用户的 prompt |
| （Model reasoning） | **没有 hook**：推理本身不是一个事件 |
| **PreToolUse**（工具调用前） | 检查或**拦截**（例如禁止写生产环境） |
| **PostToolUse**（工具调用后） | 格式化、跑测试、记录日志（例如每次编辑后自动运行 formatter） |
| **Stop**（停止前） | 验证测试是否通过，没通过就不让停 |

Hook 可以运行命令、HTTP 请求、MCP tool、prompt 或 subagent。

**Hook vs Skill 的关键区别**：skill 是给模型的建议，模型可能忽略；hook 是 harness 确定性执行的，**一定会发生**。需要“保证”的事情用 hook。

### 4.7 Plugin：打包层

```
plugin/
├── skills/
├── hooks/
├── agents/
├── .mcp.json
├── .lsp.json
└── .claude-plugin/plugin.json
```

打包带来的好处：
- **Installation**：一条命令安装
- **Versioning**：固定版本
- **Namespacing**：不同 plugin 的同名组件不冲突
- **Team reuse**：所有人用同一套配置
- **Distribution**：通过 marketplace 分享

Plugin 让“一个在你机器上能用的配置”变成“别人也能安装的东西”。

### 4.8 四者如何配合

```
Skill（知识与流程）→ Claude（推理与工具使用）→ MCP（GitHub · Slack · 数据库）
Hook 在生命周期事件时触发 —— 每次工具调用的前后
Plugin 把 skill、MCP 配置和 hooks 打包，供别处安装
```

### 4.9 `.claude/` 目录总览

```
.claude/
├── CLAUDE.md
├── settings.json          # permissions、hooks、env
├── settings.local.json    # 个人设置，不提交
├── commands/
├── skills/
├── agents/
├── hooks/
└── plugins/
```

| 类别 | 内容 |
|---|---|
| Memory | CLAUDE.md、rules/ |
| Configuration | permissions、hooks、env（在 settings.json 中） |
| Workflows | skills/、commands/ |
| Delegation & guardrails | agents/、hooks/ |

`~/.claude/` 有同样的布局（user scope）。**Project 设置优先于 user 设置。**

### 4.10 设计练习：每个需求该用哪种机制

| 需求 | 答案 | 理由 |
|---|---|---|
| 应用公司的 API 规范 | **Skill** | 是“怎么做”的知识 |
| 从 Jira 读取 issue | **MCP** | 需要访问外部系统 |
| 每次编辑后自动格式化 | **Hook** | 要在固定事件确定性地执行 |
| 指导一个多步骤部署 | **Skill** | 是流程知识 |
| 发送完成通知 | **Hook + MCP** | Hook 决定“何时”（Stop 事件），MCP 负责“发到哪”（Slack） |
| 阻止写入生产环境 | **Hook 或 policy** | 必须强制执行，不能依赖模型 |
| 在多个 repo 间共享一切 | **Plugin** | 打包和分发 |

---

## 5. 选读：MCP 深入

### 5.1 为什么需要 MCP

- 问题：agent 要访问 Gmail、GitHub 等外部服务
- 没有 MCP 之前：N 个 AI 客户端 × M 个服务 = **N×M 种集成**，每对都要单独开发
- 有了 MCP：每个客户端实现一次协议，每个服务实现一次 server，变成 **N+M**
- 定义：一个开放的（stateless）协议，让 AI 应用发现并使用外部 server 暴露的能力（https://modelcontextprotocol.io）

### 5.2 架构

```
Host（你使用的应用，如 Claude Code）
  └── MCP client（每个 server 一个连接，由 host 运行）
          ⇅  MCP 协议
      MCP server（把一个系统的能力暴露为 tools）
          ⇅
      External system（GitHub、Slack、PostgreSQL、本地服务…）
```

### 5.3 三个核心 primitives

| Primitive | 是什么 | 例子 | 谁控制 |
|---|---|---|---|
| **Tools** | Claude 可以调用的动作 | `create_issue(…)` | 模型决定调用 |
| **Resources** | 可读的数据 | `github://issues/123`（一个 issue、一个 schema） | 应用决定附加 |
| **Prompts** | server 提供的可复用 prompt 模板 | `triage_issue(…)` | **用户**控制，应用把它们显示为命令或菜单 |

在 Claude Code 中，MCP prompts 以带命名空间的 slash command 出现：`/mcp__github__triage_issue 123 high`。它会展开成类似：“Review issue #123. Determine whether it is reproducible. Assess its severity as high. Recommend an owner and next action.”

**三者如何协同**：用户说 “Triage issue #123 and assign it if it is a critical authentication bug.”
- **Prompt** `triage_issue`：提供组织的 triage 流程
- **Resource** `github://issues/123`：提供 issue 内容
- **Tool** `get_related_errors`：获取当前的诊断信息

### 5.4 MCP 安全要点

一个已连接的 server 能看到你发给它的内容，并以**你的凭证**行动。
- **最小权限**：token 的权限只限于你真正需要的 repo、表或频道
- **读写分离**：把写操作放在审批之后
- **不信任 tool 输出**：永远不要把返回的内容当作指令执行（prompt injection）
- **安装前审查**：第三方 server 可能携带 prompt injection

连接前先问：**这个 server 能读什么？能改什么？**

### 5.5 MCP Gateway

**是什么**：位于 AI 客户端和一个或多个 MCP server 之间的控制层。类似 API gateway，但它“说”MCP 协议（tools/list、tools/call、resources、prompts、schemas、协议版本）。对客户端来说，它看起来就是另一个 MCP server。

```
Claude Code / agent → MCP gateway（auth · policy · logging · routing · rate limits）→ GitHub MCP / Jira MCP / Database MCP
```

**功能**：

| 功能 | 说明 |
|---|---|
| Aggregate | 一个连接、一个带命名空间的目录：`github.*`、`jira.*`、`postgres.*` |
| Route | 把每个调用转发给对应的 server；`Mcp-Method` header 让路由无需解析 body |
| Enforce policy | 谁可以连接，哪些工具可见、可调用、需要人工审批 |
| Broker identity | 客户端只认证一次，上游凭证留在 gateway 后面 |
| Observe | 集中审计：身份、工具、决策、延迟、结果、成本 |
| Scale | 负载均衡、重试、熔断、目录缓存、配额 |

**何时值得用**：有很多用户、agents、servers、凭证或合规规则需要统一管理时。

**但它是一个信任边界**：凭证权限过宽、丢失终端用户身份、名字冲突、日志中泄露 secret、单点故障。

### 5.6 新版 Spec（2026-07-28）

MCP 发布以来最大的修订：
- **Stateless core**：不再需要保持 session，任何请求可以落到任何实例上，放在普通负载均衡器后面即可
- **Routing**：`Mcp-Method` header 让 gateway 无需读取 body 就能路由；list 结果可以带 TTL 缓存
- **Extensions**：长时间运行的 **Tasks** 和由 server 渲染 UI 的 **MCP Apps** 以扩展形式提供

---

## 6. 要点回顾

1. 一个用户 prompt 会展开成“推理 → 工具执行”交替的长链；**模型只提议，harness 执行**，安全控制点在 harness
2. 你打字之前 context 里已有 10K+ tokens（system、环境、memory、工具定义、skill / MCP 的简介）；代码库留在磁盘上，按需拉取切片
3. Permission 位于意图与执行之间：allowed、ask、denied 三种结果，有 Manual、Accept Edits、Bypass、Auto 等模式
4. 每一步都要重发整个 context：prefix cache 让重复的前缀便宜，但 cache reads 仍是大部分成本；context 快满时 compaction 用摘要替换历史，并重新注入 CLAUDE.md 和 memory
5. LLM 是无状态的；memory 就是 session 启动时加载的文件。CLAUDE.md 是你写的规则（managed → user → project → local 分层），auto memory 是 Claude 写的笔记（MEMORY.md 索引 + 按需读取的 topic 文件）
6. CLAUDE.md 只写命令、硬约束和不易推断的事实，每个文件不超过 200 行；过时的 memory 比没有更糟，要像代码一样 review 和修剪
7. 四种扩展机制各管一件事：Skill 决定怎么做，MCP 提供外部能力，Hook 在生命周期事件上确定性地执行，Plugin 负责打包分发
8. MCP 的三个 primitive：tools（模型调用）、resources（应用附加）、prompts（用户选择）；连接前问清楚它能读什么、能改什么，永远不要把工具输出当作指令

---

## 7. 自测题

<details>
<summary>1. 为什么说“一个用户 prompt 不是一次请求”？</summary>

Harness 会反复调用模型：模型提出一个工具调用，harness 执行后把结果放回 context，再调用模型决定下一步。修一个失败的测试可能经历“跑测试 → 搜索 → 读文件 → 编辑 → 再跑测试”等多次推理和工具执行，成本和延迟是所有这些调用的总和。
</details>

<details>
<summary>2. 模型请求执行 Bash("rm -rf build/") 时，谁决定是否执行？可能有哪些结果？</summary>

由 harness 的 permission 层决定（根据你选择的模式，可能由你确认，或由一个小模型检查）。结果有三种：allowed（直接执行）、ask（先问你）、denied（什么都不发生）。
</details>

<details>
<summary>3. 有了 prefix cache，为什么 context 越长还是越贵？</summary>

每一步都要把整个 context 重新发给模型。命中 cache 的部分虽然按较低的价格计费，但每一步都要再付一次；context 越长、步数越多，累计的 cache read 费用越大，所以 cache reads 往往占了大部分成本。
</details>

<details>
<summary>4. CLAUDE.md 与 auto memory 有什么区别？</summary>

CLAUDE.md 由你编写，内容是规则和指令，按 managed / user / project / local 分层，每个 session 全部加载。Auto memory 由 Claude 自己编写，记录你的偏好、纠正和项目背景，每个 session 只加载 MEMORY.md 索引（前 200 行），具体的 topic 文件按需读取。
</details>

<details>
<summary>5. “每次编辑后自动运行格式化工具”应该用 skill 还是 hook？为什么？</summary>

用 hook（PostToolUse）。Skill 只是给模型的建议，模型可能忘记或忽略；hook 由 harness 在每次工具调用后确定性地执行，保证一定会发生。
</details>

<details>
<summary>6. 为什么 skill 的正文不在 session 开始时全部加载？</summary>

为了节省 context：如果有很多 skill，全部加载会占用大量 token，而大多数 skill 在当前任务中用不到。所以开始时只加载 name 和 description 用于发现，真正用到时才加载正文。
</details>

<details>
<summary>7. MCP 的 tools、resources、prompts 分别由谁控制？</summary>

Tools 由模型决定是否调用；resources 由应用（host）决定是否附加到 context；prompts 由用户选择使用（在 Claude Code 中以 slash command 的形式出现）。
</details>
