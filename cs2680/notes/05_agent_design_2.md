# Lecture 5 · Agent Design II：Multi-Agent Systems

> Harvard CS 2680：Modern AI Systems: Agents and System Optimizations（Juncheng Yang，Fall 2026）
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/5_agent_design_2.pdf`](../slides/5_agent_design_2.pdf)
> 预计阅读时间：18–22 分钟。预备知识：Lecture 3（subagents、agent teams）、Lecture 4（agent loop、context 管理）。

## 本讲脉络

```
什么时候、为什么要把工作拆给多个 agent
→ 分析任何架构的三张图：control graph、task graph、communication graph
→ 八种 multi-agent 架构（supervisor-worker、hierarchical、pipeline、blackboard、committee、market、swarm、dynamic）及混合架构
→ Task decomposition：按什么维度拆、子任务要写清什么、粒度权衡公式、静态与动态、六种常见失败
→ Agent communication：topology、channel、medium、format；消息要带语义、控制与内容分离
→ 通信是分布式系统难题：投递语义、幂等、补偿；multi-agent 特有的挑战
→ 两个协议：A2A（agent ↔ agent）与 MCP（agent ↔ 工具）
→ 设计原则
```

**一句话总结**：多 agent 系统的价值来自**专业分工、权限隔离、并行和 context 隔离**，代价是**协调开销、更多的 token 和分布式系统的全部难题**。设计时用三张图（谁协调、怎么拆、怎么通信）来分析架构；拆任务时要写清每个子任务的目标、输出格式和完成条件；通信时用“自然语言写内容、结构化协议做协调”。**第一条原则永远是：先问单个 agent 够不够。**

（Multi-agent 并不是新话题，它有几十年的研究历史，本课不展开。）

---

## 1. 什么时候、为什么要把工作拆给多个 agent

| 理由 | 含义 | 例子 |
|---|---|---|
| **Specialization（专业分工）** | 不同的 agent 用不同的模型、prompt 和工具 | 搜索用便宜快速的小模型，写代码用最强的模型；“安全审查 agent”有专门的审查清单 |
| **Isolation（隔离）** | 给不同的 agent 不同的权限 | 负责浏览网页的 agent 会读到不可信内容（可能有 prompt injection），所以它**没有写文件、发邮件的权限**；真正执行修改的 agent 不直接接触外部网页 |
| **Parallelism（并行）** | 独立的子任务同时执行 | 同时审查 10 个文件，总时间由最慢的那个决定，而不是 10 个相加 |
| **Context management（context 管理）** | 不必让每个 agent 都看到完整的 trajectory | 一个 agent 读了 40 个文件找线索，只把 5 行结论交给主 agent；主 agent 的 context 保持干净 |

但多 agent 不是免费的：token 成倍增加、需要协调、信息在传递中丢失、失败更难排查（第 6 节）。所以本讲最后的第一条设计原则是：**真的需要 multi-agent 吗？**

---

## 2. 分析任何架构的三张图

任何 multi-agent 架构都可以用三个问题（三张图）来描述：

| 图 | 问题 | 可能的答案 |
|---|---|---|
| **Control graph（控制图）** | **谁来协调？** | 一个 supervisor、一个层级结构，还是根本没有人协调？运行过程中控制权会不会转移（handoff）？ |
| **Task graph（任务图）** | **工作如何分解？** | 由 planner 预先分好、通过竞标（bidding）认领，还是从局部规则中自然涌现？ |
| **Communication graph（通信图）** | **它们如何通信？** | 直接发消息、广播，还是通过一个共享的工作区？ |

后面每种架构，都会用这三张图来总结。

---

## 3. 八种 Multi-agent 架构

```
1 Supervisor-worker   2 Hierarchical   3 Pipeline / DAG   4 Blackboard
5 Committee           6 Market         7 Swarm            8 Dynamic topology
```

### 3.1 Supervisor-worker（主管-工人）

**结构**：一个 agent（supervisor）负责**规划、委派和整合**；workers（如 Researcher、Coder、Reviewer）**只看到自己的子任务**。

- ✅ **最适合**：子任务彼此独立、可以并行的情况
- ❌ **弱点：supervisor 成为瓶颈**。它要重读每个 worker 的结果，**它的 context 增长得最快**：10 个 worker 各返回 2,000 token，supervisor 的 context 就多了 20,000 token，还要在这些结果之上推理和整合
- 三张图：Task —— supervisor 预先拆分；Control —— 一个 supervisor；Comms —— 星形，所有通信经过 supervisor

（Lecture 3 中 Claude Code 的 subagent 就是这种模式。）

### 3.2 Hierarchical（层级式）

**结构**：supervisor-worker **递归应用**：supervisor 管理下一级的 sub-supervisor。

例：开发一个功能

```
Feature manager
├── Backend lead
│   ├── API
│   └── Database
└── Frontend lead
    ├── UI
    └── E2E tests
```

- ✅ **适合本身就有层级结构的任务**：如存储系统、查询引擎、测试这样天然分模块的工程
- ❌ **挑战**：在哪一层做分解？每跨越一层边界，要传多少 context？
  - 每往上汇报一层，下面的细节就被总结掉一次，**信息逐层丢失**
  - 上层的一个错误判断会传给下面所有子节点，**错误被放大**
- 三张图：Task —— 每一层再拆一次；Control —— 嵌套的 supervisor；Comms —— 树形，只有父子之间通信

### 3.3 Pipeline / DAG（流水线 / 有向无环图）

**结构**：固定的阶段，**每个阶段的边界有一个有类型的契约（typed contract）**，即明确的输入输出格式。

例：发票处理

```
OCR → Extract fields（提取字段）→ Match to PO（匹配采购单）→ Approve（审批）→ Post to ledger（记账）
```

- ✅ **最容易 debug**：错误可以定位到具体的某个阶段（第 3 阶段的输出不符合契约，问题就在第 3 阶段）
- ❌ **弱点：没有回头路**。第 4 阶段（审批）发现 OCR 识别出的金额可疑，也没法让第 1 阶段重新识别
- 改进：**加一条重试边（retry edge）胜过多加一个 agent** —— 允许审批阶段在发现问题时把发票退回给 OCR 重做
- 三张图：Task —— 阶段在设计时就固定；Control —— 顺序本身就是控制；Comms —— 阶段到阶段

### 3.4 Blackboard / Shared memory（黑板 / 共享内存）

**结构**：agents **通过共享状态协调，而不是直接互相发消息**。共享工作区可以是数据库、向量库、文件系统、Git 仓库或任务图。

例：多个 agent 共享一个 Git 仓库（任务列表、commits、历史），Feature agent、Refactor agent、Test agent 各自从仓库读取状态、提交自己的工作。（Lecture 3 中 16 个 agent 写 C 编译器就是这种模式。）

- ✅ agent 可以随时**加入或离开**，不需要重新接线；共享状态是**持久的、可检查的**（随时可以看黑板上有什么）
- ❌ **挑战**：
  - **Consistency（一致性）**：agent A 读到的状态，可能已经被 agent B 改了
  - **Concurrency control（并发控制）**：两个 agent 同时修改同一个文件怎么办？（锁、乐观并发、Git 的 merge）
  - **Provenance（来源）**：黑板上这条信息是谁、什么时候、根据什么写的？
  - **Versioning（版本管理）**：需要回退到之前的状态时怎么办？
  - **Garbage collection（垃圾回收）**：过时的条目谁来清理？
- 三张图：Task —— 从共享状态中涌现（谁看到什么需要做就去做）；Control —— 没有中心；Comms —— 间接的，经由工作区

### 3.5 Committee / Debate（委员会 / 辩论）

**结构**：多个 agent 各自提出答案并讨论，然后由一个 **judge** 或**投票**解决分歧。

例：作文评分 —— Grader A 和 Grader B 各自打分，Judge 综合裁决。

- ✅ **提高判断类任务的准确性**
- ❌ **失败模式：没有独立性的一致**。如果所有成员用同一个模型、同样的 context，它们就有**同样的盲点**，会一起错 —— 这叫 groupthink

**为什么独立性至关重要**：假设 3 个评审员各自有 80% 的准确率，且**错误相互独立**。多数投票正确的概率是“3 人都对”加上“恰好 2 人对”：

$$
0.8^3 + 3 \times 0.8^2 \times 0.2 = 0.512 + 0.384 = 0.896
$$

从 80% 提升到约 90%。但如果三者的错误**完全相关**（同一个模型、同样的 prompt），它们要么一起对、要么一起错，多数投票仍然只有 80%。所以要用**不同的模型、不同的 prompt 或不同的信息**来保证多样性。

- 三张图：Task —— 同一个任务被并行重复做多次；Control —— judge 做决定；Comms —— 先在同伴之间讨论，再交给 judge

### 3.6 Market and auction（市场与拍卖）

**结构**：agents 根据自己的**能力、成本和延迟**，对公布出来的任务**竞标**。agents 是异构的：模型、工具、机器、成本、专长都不同。

例：GPU 训练任务路由器，规则是“5 秒内完成的最便宜出价获胜”。任务：“谁能跑这个训练任务？”

| 节点 | 报价 | 预计时间 |
|---|---|---|
| A100 node | $0.20 | 3s |
| T4 node | $0.05 | 10s（超过 5 秒，出局） |
| H100 node | $0.40 | 1s |

→ 任务交给 **A100 node**（满足时限的最便宜者）。

- ✅ 工作自动落到**最便宜的、能胜任的** agent 上，**不需要中心 planner** 了解每个 agent 的能力
- ❌ **弱点：出价是自报的**，一个 agent 可以靠**夸大自己的能力**赢得任务。需要验证机制或信誉记录（过去报价准不准）
- 三张图：Task —— 公布出来，靠竞标认领；Control —— 拍卖者（auctioneer）清算；Comms —— 广播任务，回收报价

### 3.7 Swarm（群体）

**结构**：**没有永久的协调者**，agents 之间直接协商、交换发现。需要一整套协议：发现彼此（discovery）、寻址（addressing）、同步（synchronization）、终止（termination）、信任（trust）。

例：仓库机器人互相协商谁去拣哪个订单（Picker 1–4 两两相连）。

- ✅ 没有协调者成为瓶颈或单点故障；有 agent 加入或掉线时，工作仍能继续
- ❌ **弱点**：
  - **通信复杂度 $O(n^2)$**：$n$ 个 agent 两两之间都可能通信，共 $\frac{n(n-1)}{2}$ 条通道。10 个 agent 就有 45 条
  - **冲突解决困难**：两个 agent 都认为该自己处理同一个订单时，谁说了算？
  - **产品中很少使用**
- 三张图：Task —— 局部协商；Control —— 没有永久协调者；Comms —— 所有人对所有人，$O(n^2)$

### 3.8 Dynamic topology（动态拓扑）

**结构**：任务**运行过程中**增加、移除或重新连接 agents。

例：客服机器人。普通工单只有一个 Support bot 处理；**工单升级**后，才加入 Billing agent（账单）和 Log agent（查日志）。

- ✅ **能力随难度扩展**，简单工单保持便宜
- ❌ **弱点：每次运行的形状都不一样，失败难以复现**。为了能调试，需要把每次运行的拓扑变化（谁在何时被加入、由谁触发）完整记录下来
- 三张图：Task —— 工作变难时重新拆分；Control —— 随 agent 加入而转移；Comms —— 运行时动态决定

### 3.9 真实系统是混合的

例：

```
Supervisor
  └─ Planner agent ◄──────────────────┐
       └─ Dynamic task DAG             │ 需要时重新规划（re-plan）
            ├─ Coder                   │
            ├─ Researcher              │
            └─ Tester                  │
                 └─ Shared Git         │
                      └─ Reviewer ─────┘
```

它结合了 **hierarchy（层级）**、**shared memory（共享 Git）** 和 **pipeline（规划 → 执行 → 审查）**。

### 3.10 八种架构总结

| 类别 | 架构 | 核心思想 | 最适合 | 弱点 |
|---|---|---|---|---|
| **Centralized（集中式）** | Supervisor-worker | 一个规划者，多个工人 | 可分离的子任务 | 规划者成为瓶颈 |
| | Hierarchical | 主管的主管 | 可递归拆分的任务 | 错误被放大 |
| | Pipeline / DAG | 每个阶段喂给下一个 | 清晰有序的阶段 | 错误向下游传播 |
| | Market / auction | agents 竞标任务 | 动态分配 | 虚假的出价 |
| **Decentralized（去中心化）** | Blackboard | 共享工作区 | 一个共同的产物 | 读到过时的状态 |
| | Committee / debate | 各自回答，再调和 | 判断类问题 | Groupthink |
| | Swarm | 直接对话，没有控制者 | 局部协商 | $O(n^2)$ 条消息 |
| **Hybrid（混合）** | Dynamic topology | 随工作推进重新接线 | 结构事先未知 | 难以调试 |

---

## 4. Task Decomposition（任务分解）

```
复杂任务 → 子任务 1 / 子任务 2 / 子任务 3 → 合并结果
```

### 4.1 按什么维度拆

| 维度 | 怎么拆 | 例子 |
|---|---|---|
| **Data（数据）** | 每个 agent 处理一部分文档或记录 | 1,000 份合同分给 10 个 agent，每个 100 份 |
| **Expertise（专长）** | 按专业领域 | 安全、性能、正确性、法律分析 |
| **Stage（阶段）** | 按工作流程 | 规划、实现、测试、审查 |
| **Hypothesis（假设）** | 每个 agent 调查一种可能的解释 | debug 时，一个查网络、一个查数据库、一个查配置 |
| **Tool（工具）** | 按使用的工具 | 网页搜索、数据库查询、代码执行、内部文档 |
| **Authority（权限域）** | 按不同的管理域 | 一个 agent 能访问财务系统，另一个能访问 HR 系统 |

维度可以组合。例：审查一个改动了 200 个文件的 PR —— **先按数据（文件）拆**，再对每组文件**按专长（安全、性能）拆**。

### 4.2 创建子任务时要写清楚什么

| 要素 | 说明 |
|---|---|
| **Objective（目标）** | 要完成什么 |
| **Expected output schema（输出格式）** | 返回什么字段、什么格式（便于合并） |
| **Inputs and relevant context（输入与相关背景）** | 子 agent 看不到你的完整对话，必须把需要的信息交给它 |
| **Completion test（完成条件）** | 怎样算完成 |
| **Boundaries and exclusions（边界与排除项）** | 什么不在范围内 |
| **Budget and deadline（预算与时限）** | 最多花多少 token、多少时间 |
| **Available tools and permissions（可用工具与权限）** | 能用什么、不能做什么 |
| **Dependencies（依赖）** | 依赖哪些其他子任务的结果 |

**对比**：
- ❌ 差的子任务：“Research storage systems”（研究一下存储系统）—— 范围不清、不知道何时算完、不知道返回什么
- ✅ 好的子任务：“找出 2026 年发表的与存储相关的论文；返回标题、日期、系统贡献、评估规模和原始来源 URL；排除数据库查询优化方向。”—— 目标、范围、输出格式、排除项都写清楚了

### 4.3 粒度权衡

| 任务太大 | 任务太小 |
|---|---|
| 并行度低，工作基本还是串行的 | 调度和通信的开销占主导 |

多 agent 完成任务的总时间大约是：

$$
T_{\text{multi}} \approx T_{\text{planning}} + T_{\text{critical path}} + T_{\text{communication}} + T_{\text{merge}}
$$

（critical path，关键路径 = 必须串行完成的那条最长依赖链。）**只有当节省下来的串行工作，超过规划、通信和整合的成本时，并行才有帮助。**

**算例**：20 个独立的单行查询，每个 10 秒。假设每启动一个子 agent 要 15 秒（加载 context），每合并一个结果要 5 秒，规划要 10 秒。
- **串行**：20 × 10 = 200 秒
- **20 个 agent，每个 1 个查询**：10（规划）+ 25（启动 15 + 查询 10）+ 20 × 5（合并）= **135 秒**
- **4 个 agent，每个 5 个查询**：10 + 65（启动 15 + 查询 50）+ 4 × 5 = **95 秒**

把 20 个查询**打包成 4 个子任务**最快 —— 开销被摊薄，并行收益仍在。

### 4.4 静态分解与动态分解

| 类型 | 说明 | 例子 |
|---|---|---|
| **Static（静态）** | 执行前就已知，可预测、易于评估 | 总结上传的 5 个 PDF（每个 PDF 一个子任务） |
| **Dynamic（动态）** | agent 根据执行中的发现创建新任务 | 发现一个失效的链接 → 派生一个查找替代来源的任务 |
| **Recursive（递归）** | worker 自己也可以继续委派 | 一个综述 agent 为每个子领域启动一个 subagent |
| **Opportunistic（机会式）** | 空闲的 agent 认领新出现的任务 | 空闲的 agent 去拿下一个还没标注的文档 |

**动态分解必须有限制**：spawning limits（最多派生几个）、budgets（预算）、depth limits（最大深度）和 termination conditions（终止条件）。否则委派会**递归爆炸**：每个 agent 派生 3 个子 agent，5 层之后就是 $3^5 = 243$ 个 agent。

### 4.5 六种常见的分解失败

| 失败 | 例子 | 应对 |
|---|---|---|
| **01 Overlap（重叠）** | 两个 agent 总结了同一篇论文 | 明确划分归属（每份数据只分给一个人） |
| **02 Gap（遗漏）** | 一个负责“2020–2023”，一个负责“2025–2026”，**2024 年没人管** | 分解后检查覆盖范围是否完整 |
| **03 Hidden dependency（隐藏依赖）** | “定价”需要“尺寸”的结果，但这一点运行时才发现 | 事先分析依赖；允许动态重新规划 |
| **04 Unmergeable outputs（无法合并的输出）** | 一个 agent 返回自由文本，另一个返回 JSON，合并失败 | 在子任务中规定统一的输出 schema |
| **05 Lost global context（丢失全局视角）** | 每一节单独看都没问题，但整份报告**自相矛盾** | 最后由一个整合者做全局一致性检查 |
| **06 Straggler（掉队者）** | 6 个 agent 20 秒完成，1 个要 4 分钟 | 设置超时；用部分结果；为慢任务启动备份副本 |

---

## 5. Agent Communication（agent 之间的通信）

例：用户 → Research Agent →（Web Agent、Coding Agent、Data Agent）→ 最终回答

### 5.1 四个维度

| 维度 | 问题 | 选项 |
|---|---|---|
| **Topology（拓扑）** | 谁能和谁说话？ | supervisor-worker、peer-to-peer（由第 3 节的架构决定） |
| **Channel（通道）** | 信息如何传递？ | 点对点、广播、发布-订阅、共享状态、请求-响应 |
| **Medium（介质）** | 信息通过什么交换？ | 消息、共享内存、文件、数据库 |
| **Format & Semantics（格式与语义）** | 消息如何组织？表达什么？ | delegate、report、ask、handoff、cancel…… |

### 5.2 通信介质

| 介质 | 方式 | 例子 |
|---|---|---|
| **Message passing（消息传递）** | agents 通过队列或总线发送显式的消息 | coder 把结果发到 orchestrator 的收件箱 |
| **Shared memory（共享内存）** | 一个进程内的状态对象，所有 agent 都读写它 | 一个共享的任务列表，完成一项就更新一项 |
| **Files and artifacts（文件与产物，含数据库）** | 工作成果通过文件存储交接 | researcher 写好 `notes.md`，writer 打开它 |

### 5.3 通信通道

| 通道 | 例子 |
|---|---|
| **Direct point-to-point（点对点）** | planner 把一份任务说明交给 coder agent |
| **Broadcast（广播）** | orchestrator 通知所有 agent：build 失败了 |
| **Publish-subscribe（发布-订阅）** | agents 订阅 “new ticket” 主题，有新工单时自动收到并响应 |
| **Shared blackboard（共享黑板）** | agents 读写同一个“发现记录本” |
| **Request-response（请求-响应）** | 一个 agent 向检索 agent 发起查询，并等待返回文档 |
| **Artifact passing（产物传递）** | 一个 agent 写出 `report.md`，下一个 agent 读它 |

### 5.4 消息格式

| 自然语言 | 结构化消息 |
|---|---|
| 灵活，但有歧义 | 容易校验，但表达力较弱 |

**实用的设计是两者结合**：

```json
{
  "task_id": "T17",
  "type": "result",
  "status": "complete",
  "claim": "The regression began after commit X",
  "confidence": 0.82,
  "evidence": ["log:2391", "commit:X"],
  "artifact": "analysis/T17.json",
  "requires_response": false
}
```

`claim`（结论）用自然语言；`task_id`、`type`、`status`、`confidence`、`evidence`、`artifact`、`requires_response` 是程序可以直接读取的结构化字段。

### 5.5 消息要带语义：把控制和内容分开

❌ 所有东西塞进一个字符串：`{"message": "some text"}` —— 系统必须先让 LLM 读懂这段话，才知道该做什么

✅ 控制信息放在字段里：

```json
{"id": "msg-381", "task_id": "t-17", "from": "planner", "to": "researcher",
 "type": "DELEGATE", "payload": { ... }}
```

**Runtime 认识的操作类型**：

| 类型 | 含义 |
|---|---|
| `DELEGATE` | 委派一个任务 |
| `QUERY` | 提问，需要回答 |
| `REPORT` | 汇报进展或结果 |
| `HANDOFF` | 移交控制权 |
| `UPDATE` | 更新信息或任务 |
| `CANCEL` | 取消任务 |
| `ACK` | 确认收到 |
| `ERROR` | 报告错误 |
| `COMPLETE` | 任务完成 |

好处：**runtime 可以直接路由、重试和停止，不需要问 LLM“这条消息是什么意思”** —— 更快、更便宜、更可靠。

> **自然语言用于内容，结构化协议用于协调。**

### 5.6 通信设计空间小结

| 维度 | 选项 | 经验法则 |
|---|---|---|
| Topology | supervisor-worker、peer-to-peer、mesh | **反映任务的依赖关系** |
| Medium | 消息、共享内存、文件 | **每条链路只用一种介质** |
| Channel | 点对点、广播、发布-订阅、请求-响应 | **只有所有人都需要知道时才广播** |
| Format | 自然语言 + 控制字段 | **按字段路由，而不是按文字内容** |

每个选择都在**延迟、context 和可靠性**之间权衡。

---

## 6. 通信是一个困难的分布式系统问题

### 6.1 一个例子

Agent A 给 Agent B 发消息：“purchase 8 GPU instances”（购买 8 台 GPU 实例）。然后**网络超时，没有回复**。

**B 到底执行了没有？**
- B 可能根本没收到；也可能收到了、买了，只是回复丢了；还可能正在执行中
- 重试，还是放弃？如果重试而 B 其实已经买了，就会**重复购买 16 台**
- 谁为重复购买买单？

**投递语义（delivery semantics）**：

| 语义 | 含义 | 风险 |
|---|---|---|
| **At-most-once（至多一次）** | 失败了不重试 | 可能丢失（GPU 没买成） |
| **At-least-once（至少一次）** | 没收到确认就重试 | 可能重复（买了两次） |
| **Exactly-once（恰好一次）** | 理想情况 | 在不可靠的网络上很难真正实现 |

**常见做法：at-least-once + 幂等性（idempotency）**：
- **Idempotency key（幂等键）**：用 `task_id` 作为请求的唯一标识。B 记住已经处理过的 task_id，收到重复的请求时直接返回上次的结果，不再执行一遍。这样 A 可以放心重试
- **Compensation / rollback（补偿 / 回滚）**：真的执行错了，要有撤销路径（例如释放多买的实例）

### 6.2 分布式系统本来就难

| 难题 | 说明 |
|---|---|
| **Concurrency and non-determinism（并发与非确定性）** | 很多机器同时修改状态，执行顺序每次都可能不同 |
| **Unreliable network（不可靠的网络）** | 数据包可能延迟、丢失、乱序、重复 |
| **No global clock（没有全局时钟）** | 不同机器上的时钟不一致，很难判断事件的先后 |
| **Partial failure（部分失败）** | **你分不清一个 agent 是“慢”还是“死了”**（只能靠超时和心跳来猜） |
| **Consistency among replicated states（副本之间的一致性）** | 在一致性、可用性、网络分区容忍和延迟之间必须权衡 |
| **Debugging and observability（调试与可观察性）** | 请求很多、日志分散在不同机器上、时钟还有偏差，很难还原到底发生了什么（需要为每个请求附加统一的追踪 ID） |

**Multi-agent 系统继承了所有这些常见的分布式系统问题。**

### 6.3 运行多个 agent 时的额外挑战

| 挑战 | 说明 |
|---|---|
| **Scheduling（调度）** | 谁下一个运行？运行多久？ |
| **Resource allocation（资源分配）** | 模型调用额度、CPU 等共享资源怎么分 |
| **State management（状态管理）** | checkpoint、memory、来源（provenance）、版本管理 |

而且**没有预先确定的请求流**：系统无法事先知道谁需要运行、运行多久、每个 agent 会消耗多少资源。

**LLM 是黑盒，让这一切更难**：

| 类别 | 问题 | 例子 |
|---|---|---|
| **Context** | Information loss（信息丢失） | worker 的 summary 漏掉了它知道的关键细节 |
| | Context pollution（context 污染） | 重要的信息被淹没在大量无关内容中 |
| | Stale information（过时信息） | X 已经从 3 改成了 5，B 仍然按 X = 3 行动 |
| | Conflicting beliefs（信念冲突） | A 说“安全”，B 说“有漏洞”，没人调和 |
| **Coordination and communication** | Lost ownership（无人负责） | 每个 agent 都以为另一个 agent 会处理这件事 |
| | Communication loops（通信死循环） | A 问 B，B 问 C，C 又问 A |
| **Observability and failure diagnosis** | 为什么失败了？ | 很难从分散的轨迹中找到根因 |

---

## 7. A2A：agent ↔ agent

### 7.1 交互流程

```
Agent A                                   Agent B
   │── 发现能力（discover capabilities）──►│
   │── 发送消息、创建任务 ────────────────►│
   │◄──────────────── 进度更新 ────────────│
   │◄──────────────── 返回产物（artifact）─│
```

**例子**：“分析公司 X 是否应该把推理服务迁移到 GPU Y 上。”Research Agent 通过各个 agent 的 **Agent Card** 找到同伴：
- Finance Agent：估算 GPU 成本
- Search Agent：查找 benchmark 数据
- Coding Agent：运行这些 benchmark

### 7.2 Agent Card

每个 A2A server 发布一张**机器可读的卡片**，地址是 `GET /.well-known/agent-card.json`，用于**发现**和**能力描述**：

```json
{
  "name": "Travel Agent",
  "description": "Plans business travel",
  "skills": [{ "id": "book-flight" }],
  "supportedInterfaces": [ ... ],
  "capabilities": { "streaming": true },
  "securitySchemes": { ... }
}
```

调用者能从中了解到：
- **Identity（身份）**：这个 agent 是谁
- **Skills（技能）**：它声称自己能做什么
- **Endpoints（端点）**：在哪里、怎么调用它
- **Capabilities（能力）**：是否支持流式输出、推送通知
- **Auth（认证）**：要满足哪种安全方案

### 7.3 A2A 的三个抽象

| 抽象 | 含义 | 例子 |
|---|---|---|
| **Message（消息）** | client 与 agent 之间的一轮交流 | “Book a flight to Boston” |
| **Task（任务）** | **有状态**的工作单元，有 ID 和生命周期 | task-42：working → completed |
| **Artifact（产物）** | 任务产出的结果 | 行程单 PDF 和预订数据 |

**一条消息启动一个任务，任务以 artifact 的形式返回结果。** 后续的追问会加入同一个任务：“Which date?” → “Oct 3”。

### 7.4 任务生命周期

A2A 是一个**有状态的任务协议**：工作可以暂停（等待输入或认证），然后继续。

```
Submitted → Working → Completed
                   ↘ Failed / Rejected / Canceled
            Working ⇄ Input required（需要用户补充信息）
            Working ⇄ Auth required（需要认证）
```

例：“Book a flight” → Input required：“Which date?” → 用户答 “Oct 3” → 回到 Working。

### 7.5 跟踪长时间运行的任务

任务可能运行几秒，也可能几小时，调用者可以选择跟踪方式：

| 方式 | 做法 | 适合 |
|---|---|---|
| **Polling（轮询）** | 客户端反复查询，直到完成 | 短任务 |
| **Streaming（流式）** | 服务端持续推送实时更新 | 需要交互式输出 |
| **Push notification（推送通知）** | 完成时触发 webhook 通知客户端 | 几小时的长任务 |

**注意**：A2A 标准化的是**通信**，不是**编排（orchestration）**。怎么委派、怎么拆分任务、什么时候停止，仍然由 harness 决定。

---

## 8. MCP：agent ↔ capability（能力）

**MCP 标准化了什么**：一个 agent 连接 filesystem、database、GitHub、browser 等多个 server，通过统一的协议使用：
- **Tools**：可调用的、有类型的能力
- **Resources**：host 可以附加到 context 中的可读数据
- **Prompts**：可复用的交互模板
- 一个 client 可以和多个 server 通信
- **由 host 而不是模型决定暴露哪些能力**

MCP 是 agent 的服务边界：一次工具调用，一个结果。

### MCP vs A2A

| | MCP | A2A |
|---|---|---|
| 主要抽象 | 能力 / 工具 | agent / 任务 |
| 关系 | agent → 服务 | agent → agent |
| Discovery（发现） | tools、resources、prompts | agent 的能力（Agent Card） |
| 工作单元 | 一次工具调用 | 长时间运行的任务 |
| 内部状态 | 暴露能力接口 | 理想情况下**不透明**（调用者不需要知道对方内部怎么做） |
| 类比 | 设备 / API 接口 | 服务之间的 RPC |

两者是互补的：**一个通过 A2A 被调用的远程 agent，内部可能自己在调用 20 个 MCP server。**

---

## 9. 设计原则

1. **Multi-agent 真的有必要吗？** 单个 agent 往往就够了。多 agent 常常只是增加了开销和复杂度
2. **决定拓扑**：反映任务的依赖关系，尽量减少通信
3. **识别信息丢失点**：把每个**汇总节点**都当作细节会丢失的地方（supervisor 读 summary、上级读下级汇报），在那里特别注意保留关键信息
4. **识别故障隔离的边界**：确保任何一个 agent 失败或卡住时，整个系统不会停滞（超时、部分结果、重试）

> 稠密的网状结构（dense mesh）很少是好的默认选择。**更多的通信会增加 context 压力，却不一定增加有用的信息。**

---

## 10. 要点回顾

1. 拆给多个 agent 的四个理由：专业分工、权限隔离、并行、context 管理；代价是 token、协调与分布式系统的难题
2. 用三张图分析架构：控制图（谁协调）、任务图（怎么拆）、通信图（怎么交流）
3. 集中式（supervisor-worker、层级、流水线、市场）易控制但有瓶颈和错误传播；去中心化（黑板、委员会、群体）灵活但有一致性、groupthink、$O(n^2)$ 通信问题；真实系统多是混合的
4. 委员会投票只有在成员错误相互独立时才有效
5. 子任务要写清目标、输出格式、背景、完成条件、边界、预算、权限和依赖；粒度要让并行收益超过规划、通信、整合的开销
6. 六种分解失败：重叠、遗漏、隐藏依赖、无法合并、丢失全局视角、掉队者
7. 通信：自然语言写内容、结构化字段做控制；消息类型（DELEGATE、REPORT、CANCEL……）让 runtime 无需 LLM 就能路由
8. 网络不可靠时，用 at-least-once + 幂等键 + 补偿；分不清慢和死，只能靠超时和心跳
9. A2A（Agent Card、Message / Task / Artifact、有状态生命周期）连接 agent 与 agent；MCP 连接 agent 与工具
10. 第一原则：先问单个 agent 够不够

---

## 11. 自测题

<details>
<summary>1. 用三张图描述 supervisor-worker 架构。</summary>

Control：一个 supervisor；Task：supervisor 预先拆分子任务；Communication：星形，所有通信都经过 supervisor。
</details>

<details>
<summary>2. 为什么“三个用同一模型、同一 prompt 的评审员投票”往往没有提升准确率？</summary>

它们的错误高度相关，会在同样的地方一起出错，多数投票无法纠正共同的盲点。只有错误相互独立时，投票才能提高准确率（如三个 80% 准确且独立的评审员，多数投票约 89.6%）。
</details>

<details>
<summary>3. 一个 pipeline 的最后一个阶段发现第一个阶段的输入有问题，但无法退回。该怎么改进？</summary>

加一条从后面阶段指回前面阶段的重试边（retry edge），允许把数据退回重做；这通常比再加一个 agent 更有效。
</details>

<details>
<summary>4. Agent A 让 B 执行一个付款操作后超时。A 如何安全地重试？</summary>

在请求中带一个唯一的幂等键（如 task_id）。B 记录已处理过的幂等键，收到重复请求时直接返回之前的结果而不再付款。这样 A 可以按 at-least-once 的方式重试，而不会重复付款；同时准备好补偿（退款）路径以防万一。
</details>

<details>
<summary>5. 为什么要把消息的控制信息和内容分开？</summary>

控制信息（类型、任务 ID、收发方、状态）放在结构化字段中，runtime 可以直接路由、重试、取消，不必调用 LLM 去理解消息含义，更快、更便宜、更可靠；内容仍用自然语言，保持表达的灵活性。
</details>

<details>
<summary>6. 动态分解为什么需要深度限制和预算？</summary>

允许子 agent 继续派生子 agent 时，数量会指数增长（每个派生 3 个，5 层就是 243 个），token 和成本失控，而且可能永不终止。深度限制、派生上限、预算和终止条件防止这种递归爆炸。
</details>

<details>
<summary>7. MCP 和 A2A 的工作单元有什么不同？</summary>

MCP 的工作单元是一次工具调用（调用—返回）；A2A 的工作单元是一个有 ID、有生命周期的长时间任务，可以暂停等待输入或认证，并以 artifact 返回结果。
</details>
