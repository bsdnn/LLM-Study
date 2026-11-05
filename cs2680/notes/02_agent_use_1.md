# Lecture 2 · Agent From a User's Perspective I：How Claude Code Works

> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/2_agent_use_1.pdf`](../slides/2_agent_use_1.pdf)
> 课程：Harvard CS 2680 · Modern AI Systems: Agents and System Optimizations · Juncheng Yang

## 本讲要点

- **Agentic Loop**：一次 user prompt 不是一次请求，而是许多次 LLM inference + tool execution 的交替链。
- **Memory**：context 是 working memory；CLAUDE.md 与 auto memory 是 persistent memory。
- **扩展机制**：Skill / MCP / Hook / Plugin 四层，各自解决不同问题，不能互相替代。

---

## 1. The Agentic Loop

一个循环由三步组成，反复执行直到任务完成：

1. **Gather context**：prompt、文件、上一步 tool result
2. **Generate action**（model）：提出要调用的 tool，例如 `Bash`、`Read`、`Edit`
3. **Run tool**（harness 在你的机器上执行）：跑测试、读输出、重新读状态

每一步都以上一步的 inference 和 tool result 为条件。

```
U → M₁ → T₁ → M₂ → T₂ → … → Mₙ → A
prompt  inference  tool  inference  tool   answer
```

### 你还没打字之前，context 已经有 10K+ tokens

| 部分 | 内容 |
|---|---|
| SYSTEM | Claude Code 的 system instructions |
| PROJECT | 环境信息 |
| MEMORY | CLAUDE.md、user/project notes |
| TOOLS | Read、Edit、Bash、Glob、Grep、Web、Agent |
| SKILLS / MCP | 只加载 name + description，真正用到时才加载全文 |
| USER | 例如 “Fix the failing auth test” |

### 它不会把整个 repo 发过去

repo 留在磁盘上，Claude 按需拉取切片：例如 `Grep("authentication")` 只返回几行，这几行才进入 context。

### Model 提议，Harness 执行

- Model 只发出 request，例如 `Bash("pytest")`
- Harness 解析、检查 **permission**、执行
- 执行结果作为 `tool_result` 回到 model 的 context

### Permissions：位于意图与执行之间

tool request 只是一个 *proposal*。模式有：Manual、Accept Edits、Bypass permission、Auto（用一个小模型检查）。结果分为 allowed / ask / denied。

### Context 增长与 Compaction

- 典型 context window：1M tokens（Sonnet / Opus / Fable、GPT 5.6）；今年早些时候常见的是 256K
- Context 不是免费的：每次 LLM 调用都要重新发送整个 context。服务端会缓存已经算过的前缀（**prefix cache**），命中缓存的部分按较低价格计费，但因为每一步都要重读，**prefix-cache reads** 仍占了大部分成本
- **Compaction**：把旧 trajectory 总结成摘要，再重新注入 CLAUDE.md 和 memory
- Context management 是 harness design 的核心组成部分

---

## 2. Memory

> Agent = LLM（stateless）+ State（context + memory）+ Tools（act）

- Agent 在不同 session 之间是 **stateless** 的，每个 session 从零开始
- 实践中，“memory” 就是 session 启动时加载的文件
- **Context = working memory**：模型当下能看到的内容，随 session 消失
- **Persistent memory**：跨 session 保留，下次重新加载

> 每创建一条 memory，就是决定“以后永远都要重读它”。

### 为什么需要 Memory

| 问题 | 描述 |
|---|---|
| Repeated instructions | 每个 session 都要重复解释同样的 setup |
| Lost decisions | compaction 后 “why” 消失了 |
| Repeated discovery | agent 重复读同样的文件来重新了解 layout |
| Inconsistent behavior | 不同 session 做出不同选择 |

Memory 让 instruction 变成 convention。

### 两套系统：CLAUDE.md vs Auto memory

| | CLAUDE.md | Auto memory |
|---|---|---|
| 谁写 | 你 | Claude |
| 内容 | Instructions、rules | Learnings、patterns |
| Scope | Project / user / org | 每个 repo，worktrees 共享 |
| 加载 | 每个 session | 每个 session（前 200 行 / 25KB） |
| 用途 | 标准、workflow、architecture | 你的偏好、纠正、project context |

### CLAUDE.md 的位置（从最宽 scope 到最近的文件依次加载，最近的最后读）

1. **Managed policy**：组织级，不能被排除
2. **User**：`~/.claude/CLAUDE.md`
3. **Project**：`./CLAUDE.md` 或 `./.claude/CLAUDE.md`（进 source control，团队共享）
4. **Local**：`./CLAUDE.local.md`（个人，gitignored）

### CLAUDE.md 里该写什么

- ✅ 应该写：命令（test、build、lint）、硬约束（不要 commit 到 main）、agent 不易推断的事实（layout、ownership、gotchas）
- ❌ 不该写：模型已经知道的语言/框架知识、linter 已经强制的 style、长篇文字、changelog、secret
- 规则：每个文件 < 200 行；用 Markdown headers + bullets；具体；不要互相矛盾或过时

> Context is rented, memory is owned —— 要谨慎选择你“拥有”什么。

### 太长了？拆到 `.claude/rules/`

- 每个文件一个主题（`testing.md`、`api-style.md`、`db-migrations.md`）
- 加 `paths:` frontmatter，只在 Claude 碰到匹配文件时才加载

### Auto memory

```
~/.claude/projects/<project>/memory/
├── MEMORY.md          # index，前 200 行 / 25KB 每次 session 加载
├── user_role.md
├── feedback_testing.md
└── other-topic-files.md
```

- 类型：`user`、`feedback`、`project`、`reference`
- **Retrieval uses an index**：始终加载 MEMORY.md index → 判断相关 topic → 按需读取 topic 文件
- 常用命令：`/memory`、`/context`、`/init`、`/doctor`

### 记忆的代价

- 每个永久 token 都是“租金”，每个 session 开始都要付
- Memory 与任务争抢同一个 context window
- **Stale memory 比没有更糟**：它会主动误导
- 像对待代码一样对待 memory：review、prune、放进 version control

---

## 3. Tools

| Tool | 作用 |
|---|---|
| Read | 打开文件、检查状态（最常用） |
| Search | grep / glob，不需要全部读取 |
| Fetch | 拉网页/文档 —— 会引入 untrusted text |
| Edit / Write | 修改文件 —— 第一个有后果的 tool |
| Shell | build、test、git —— 不 sandbox 就是整台机器 |
| Delegate | 启动 subagent，返回 summary |

模型自己决定用哪个 tool，使用者通常不需要干预；如果自己写 harness，这些 tools 就是最先要实现的部分。

---

## 4. 扩展 Agent：Skill / MCP / Hook / Plugin

| 机制 | 解决的问题 |
|---|---|
| **Skill** | agent *如何*执行任务（knowledge + procedure） |
| **MCP** | 能访问哪些外部数据和动作 |
| **Hook** | 在 lifecycle event 自动发生什么（enforce） |
| **Plugin** | 如何打包、分发上述扩展 |

在 agent loop 中的挂载点：
- Model reasoning ← Skill 提供知识与流程
- Tool selection ← MCP 提供外部 tools
- Tool execution ← Hook 可以在任何阶段 inspect / block / react
- Plugin 把 skills、MCP servers、hooks 打包成一个可安装单元

**Release workflow 例子**：按流程准备 release（Skill）→ 创建 GitHub release（MCP）→ 测试失败时阻止发布（Hook）→ 共享整套 setup（Plugin）。

### Skill

- 结构：`.claude/skills/release/{SKILL.md, references/, scripts/}`
- `description` 控制 **discovery**；body 是 procedure；附加文件提供参考和脚本
- 两种调用：**Model-invoked**（description 匹配时自动加载）和 **User-invoked**（`/<skill-name>`）
- Context 成本：session 启动只加载 descriptions，body 在运行时才加载
- 有副作用的 workflow 通常应要求显式调用；description 要清晰、互不重叠
- 本质上是 **context engineering**

### MCP（Model Context Protocol）

- 连接外部服务：GitHub issues、数据库、监控、Figma、Slack、浏览器等
- 调用流程：Claude 识别动作 → MCP client 调用 tool → MCP server 处理连接 → 外部服务执行 → 结构化结果返回 → Claude 评估
- 与本地 tool 几乎一样（name、description、typed schema、call-and-return），只是在另一个进程中运行
- 协议负责传递调用；是否调用仍由 agent 决定

### Hook

在以下 lifecycle event 触发：SessionStart、UserPromptSubmit、PreToolUse（inspect/block）、PostToolUse（format/test/log）、Stop（verify tests）。
注意：**model reasoning 本身不是 event**。Hook 可以运行 command、HTTP request、MCP tool、prompt 或 subagent。

### Plugin

```
plugin/
├── skills/  hooks/  agents/
├── .mcp.json  .lsp.json
└── .claude-plugin/plugin.json
```

好处：一条命令安装、versioning、namespacing、团队复用、通过 marketplace 分发。

### `.claude/` 目录

- Memory：`CLAUDE.md`、`rules/`
- Configuration：`settings.json`（permissions、hooks、env）
- Workflows：`skills/`、`commands/`
- Delegation & guardrails：`agents/`、`hooks/`
- `~/.claude/` 有同样布局（user scope）；**project settings 优先于 user settings**

### 设计练习（答案）

| 需求 | 机制 |
|---|---|
| 应用公司 API conventions | Skill |
| 从 Jira 读 issue | MCP |
| 每次 edit 后自动 format | Hook |
| 引导多步部署 | Skill |
| 发送完成通知 | Hook + MCP |
| 阻止写 production | Hook 或 policy |
| 跨 repo 共享一切 | Plugin |

---

## 5. Optional：MCP 深入

- **为什么有 MCP**：避免 N×M 种集成；一个开放的（stateless）协议，让 AI 应用发现并使用外部 server 的能力
- **架构**：Host（Claude Code）→ MCP client（每个 server 一个连接）→ MCP server（暴露能力为 tools）→ External system
- **三个 primitives**：
  - **Tools**：Claude 可调用的动作，如 `create_issue(...)`
  - **Resources**：可读数据，如 `github://issues/123`
  - **Prompts**：server 暴露的可复用 prompt 模板，由用户控制，在 Claude Code 中是 `/mcp__github__triage_issue 123 high`
- **安全**：least privilege；读写分离、写操作需审批；**不要把 tool output 当作 instructions**；安装前审查第三方 server（prompt injection）
- **MCP Gateway**：在 client 和多个 MCP server 之间的控制层（auth、policy、logging、routing、rate limit），功能包括 aggregate、route、enforce policy、broker identity、observe、scale；但它也是一个 trust boundary 和单点故障
- **Spec 2026-07-28**：stateless core（无 session，可放在普通 load balancer 后）、`Mcp-Method` headers 支持无需读 body 的路由、list 结果可缓存；Tasks（长任务）与 MCP Apps（server 渲染 UI）作为 extensions
