# Lecture 5 · Agent Design II：Multi-Agent Systems

> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/5_agent_design_2.pdf`](../slides/5_agent_design_2.pdf)

## 本讲要点

- 什么时候、为什么把工作拆给多个 agent
- 每个架构背后的三张图：**Control graph / Task graph / Communication graph**
- 八种 multi-agent architecture 及其优缺点
- **Task decomposition**：维度、粒度、static vs dynamic、常见失败
- **Agent communication**：topology、channel、medium、format；A2A 与 MCP
- Multi-agent 继承了 distributed systems 的所有难题

---

## 1. 为什么要拆给多个 Agent？

| 理由 | 说明 |
|---|---|
| **Specialization** | 不同的 models、prompts、tools |
| **Isolation** | 给不同 agent 不同 privileges |
| **Parallelism** | 独立 subtasks 并发执行 |
| **Context management** | 不必让每个 agent 都看到完整 trajectory |

（Multi-agent 不是新东西，有几十年研究历史，本课不展开。）

## 2. 每个架构背后的三个问题

- **Control graph — 谁来协调？** 一个 supervisor、一个层级，还是没人？运行时会不会变（handoff）？
- **Task graph — 工作如何分解？** 由 planner 预先分解、通过 bidding，还是从局部规则中涌现？
- **Communication graph — 如何通信？** Direct messages、broadcast，还是 shared workspace？

---

## 3. 八种 Multi-agent Architecture

### 1. Supervisor–worker
- 一个 agent 规划、委派、整合；workers 只看到自己的 subtask
- ✅ subtasks 独立且可并行时最好
- ❌ supervisor 成为瓶颈：要重读每个结果，context 增长最快
- Task：supervisor 预先拆分 · Control：一个 supervisor · Comms：星形，经过 supervisor

### 2. Hierarchical
- 递归的 supervisor–worker：supervisor 管理 sub-supervisor（例：Feature manager → Backend lead / Frontend lead → API、Database、UI、E2E tests）
- ✅ 适合天然层级分解的任务
- ❌ 挑战：在哪里分解？每个边界传递多少 context？错误会被放大
- Comms：树形，只有父子通信

### 3. Pipeline / DAG
- 固定阶段，每个边界有 typed contract（例：发票处理 OCR → Extract fields → Match to PO → Approve → Post to ledger）
- ✅ **最容易 debug**，错误可以定位到某个阶段
- ❌ 没有回头路：第四阶段不能要求第一阶段提供更好的输入 —— **加一条 retry edge 胜过多加一个 agent**
- Control：顺序本身就是控制

### 4. Blackboard / Shared memory
- agents 通过共享状态而非直接消息协调；workspace 可以是数据库、vector store、文件系统、Git repo、task graph（例：Feature / Refactor / Test agent 共享一个 Git repo）
- ✅ agent 可以随时加入或离开；共享状态持久且可检查
- ❌ consistency、concurrency control、provenance、versioning、garbage collection
- Task：从共享状态涌现 · Control：无中心 · Comms：间接，经 workspace

### 5. Committee / Debate
- agents 提出并讨论，然后由 judge 或投票解决分歧（例：essay grading，Grader A/B + Judge）
- ✅ 提高判断型任务的准确性
- ❌ Failure mode：**没有独立性的一致** —— 同一个模型和 context 有同样的盲点（groupthink）

### 6. Market / Auction
- agents 根据 capability、cost、latency 对公布的任务出价；agents 是异构的
- 例：GPU job router，“5s 内最便宜的出价获胜”：A100 $0.20·3s、T4 $0.05·10s、H100 $0.40·1s → 交给 A100
- ✅ 无需中心 planner，工作落到最便宜的胜任 agent
- ❌ 出价是自报的，agent 可能通过夸大能力赢得任务

### 7. Swarm
- 没有永久协调者，agents 直接协商并交换发现；需要 discovery、addressing、synchronization、termination、trust 等协议（例：仓库机器人协商谁拣哪个订单）
- ✅ 没有协调者成为瓶颈或单点故障
- ❌ 通信复杂度 O(n²)；冲突解决困难；**产品中很少使用**

### 8. Dynamic topology
- 任务运行中增删或重连 agents（例：support bot 在工单升级时才加入 billing agent 和 log agent）
- ✅ 能力随难度扩展，简单任务保持便宜
- ❌ 每次运行形状不同，失败难以复现

### 实际系统是混合的

例：Supervisor → Planner agent → Dynamic task DAG → Coder / Researcher / Tester → Shared Git → Reviewer →（需要时）re-plan。结合了 hierarchy、shared memory、pipeline。

### 总结表

| 类别 | 架构 | 核心思想 | 适合 | 弱点 |
|---|---|---|---|---|
| Centralized | Supervisor–worker | 一个 planner，多个 workers | 可分离的 subtasks | Planner 瓶颈 |
| | Hierarchical | supervisor 的 supervisor | 递归拆分 | 错误放大 |
| | Pipeline / DAG | 每个阶段喂给下一个 | 清晰有序的阶段 | 错误传播 |
| | Market / auction | agents 竞标任务 | 动态分配 | 坏的出价 |
| Decentralized | Blackboard | 共享 workspace | 一个共享 artifact | Stale state |
| | Committee / debate | 先回答，再调和 | 判断题 | Groupthink |
| | Swarm | 直接对话，无控制者 | 局部协商 | O(n²) 消息 |
| Hybrid | Dynamic topology | 随工作推进重连 | 结构未知 | 难 debug |

---

## 4. Task Decomposition

### 分解维度

| 维度 | 说明 |
|---|---|
| Data | 每个 agent 处理一部分文档/记录 |
| Expertise | security、performance、correctness、legal |
| Stage | plan、implement、test、review |
| Hypothesis | agents 调查相互竞争的解释 |
| Tool | web、database、code execution、内部文档 |
| Authority | agents 在不同管理域中行动 |

例：一个 200 文件的 PR，先按 data（文件）拆，再按 expertise（security、perf）拆。

### 创建 sub-task 要考虑什么

Objective、expected output schema、inputs 与相关 context、completion test、boundaries & exclusions、budget & deadline、可用 tools 与 permissions、dependencies。

- ❌ 差：“Research storage systems”
- ✅ 好：“找出 2026 年发表的 storage 相关论文；返回 title、date、system contribution、evaluation scale 和 primary-source URL；排除 database query optimization”

### 粒度权衡

- 任务太大：并行度低，工作仍然是串行的
- 任务太小：调度和通信开销占主导
- **T_multi ≈ T_planning + T_critical_path + T_communication + T_merge**
- 只有节省的串行工作超过 planning、通信和整合成本时，并行才有帮助。例：把 20 个单行查询打包成 4 个 subtasks

### Static vs Dynamic

| 类型 | 说明 | 例子 |
|---|---|---|
| Static | 执行前已知，可预测、易评估 | 总结上传的 5 个 PDF |
| Dynamic | agent 根据发现创建新任务 | 遇到死链就派生一个查找任务 |
| Recursive | worker 自己也可以再委派 | review agent 为每个子领域启动一个 subagent |
| Opportunistic | 空闲 agent 认领新任务 | 空闲 agent 拿下一个未标注文档 |

**分解需要 spawning limits、budgets、depth limits 和 termination conditions**，否则委派会递归爆炸。

### 常见分解失败

1. **Overlap**：两个 agent 总结同一篇论文
2. **Gap**：“2020–2023” + “2025–2026” 漏掉了 2024
3. **Hidden dependency**：pricing 需要 sizing，但只有运行时才发现
4. **Unmergeable outputs**：一个返回自由文本，一个返回 JSON
5. **Lost global context**：每一节都没问题，但整份报告自相矛盾
6. **Straggler**：六个 agent 20 秒完成，一个要 4 分钟

---

## 5. Agent Communication

### 四个维度

| 维度 | 问题 | 选项 |
|---|---|---|
| **Topology** | 谁能和谁说话？ | supervisor–worker、peer-to-peer（由架构决定） |
| **Channel** | 信息如何传递？ | p2p、broadcast、pub–sub、shared state、request–response |
| **Medium** | 信息通过什么交换？ | messages、shared memory、files、database |
| **Format & Semantics** | 消息结构与含义 | delegate、report、ask、handoff、cancel |

**Medium**：
- Message passing：通过 queue/bus 发显式消息
- Shared memory：一个进程内所有 agent 读写的状态对象
- Files & artifacts（含 DB）：通过文件存储交接成果（如 researcher 写 `notes.md`，writer 打开它）

**Channels**：Direct p2p、Broadcast（编排者告诉所有 agent build 失败）、Publish–subscribe（订阅 “new ticket” topic）、Shared blackboard、Request–response、Artifact passing。

### Message Format

- 自然语言：灵活但有歧义；结构化消息：易校验但表达力弱 → **两者结合**
- **把 control 和 content 分开**：不要 `{"message": "some text"}`，而要包含 `id`、`task_id`、`from`、`to`、`type`、`payload`
- Runtime 能理解的操作：`DELEGATE`、`QUERY`、`REPORT`、`HANDOFF`、`UPDATE`、`CANCEL`、`ACK`、`ERROR`、`COMPLETE` —— runtime 可以路由、重试、停止，**无需问 LLM 消息是什么意思**
- 原则：**自然语言用于内容，结构化协议用于协调**

### 设计空间小结

| 维度 | 经验法则 |
|---|---|
| Topology | 反映任务的依赖关系 |
| Medium | 每条链路一种 medium |
| Channel | 只有所有人都需要知道时才 broadcast |
| Format | 按字段路由，而不是按文字 |

每个选择都在 latency、context、reliability 之间权衡。

### 通信是分布式系统难题

例：Agent A 让 B “purchase 8 GPU instances”，网络超时无回复 —— B 到底执行了没有？
- retry 还是放弃？at-most-once 还是 at-least-once？有没有 exactly-once？
- 用 task_id 作为 **idempotency key**；准备 compensation / rollback 路径；重复购买谁买单？

Distributed systems 的经典难题：并发与非确定性、不可靠网络（延迟、丢包、乱序、重复）、没有全局时钟、partial failure（分不清慢 agent 和死 agent）、副本状态一致性（consistency/availability/partition/latency 权衡）、debugging 与 observability。

**Multi-agent 额外的挑战**：
- Scheduling、resource allocation、state management（checkpoint、memory、provenance、versioning）—— 没有预定的请求流
- 黑盒 LLM 让这更难：
  - Context：信息丢失（summary 丢掉 worker 知道的）、context pollution、stale information（X=5 之后 B 仍按 X=3 行动）、conflicting beliefs
  - 协调：lost ownership（每个 agent 都以为别人负责）、communication loops（A 问 B，B 问 C，C 问 A）、失败诊断困难

---

## 6. A2A：Agent ↔ Agent

- 流程：discover capabilities → send message / create task → progress updates → return artifact
- **Agent Card**：每个 A2A server 在 `GET /.well-known/agent-card.json` 发布机器可读的卡片，包括 identity（name、description）、skills、endpoints（supportedInterfaces）、capabilities（streaming、push）、auth（securitySchemes）
- **三个抽象**：
  - **Message**：client 与 agent 之间的一轮
  - **Task**：有 ID 和 lifecycle 的有状态工作单元
  - **Artifact**：task 产出的结果（如行程 PDF 和预订数据）
  - Follow-up 加入同一个 task（“Which date?” → “Oct 3”）
- **Task lifecycle**：Submitted → Working → Completed / Failed / Rejected / Canceled；可以暂停在 Input required 或 Auth required，然后恢复
- 跟踪长任务：**Polling**（短任务）、**Streaming**（交互输出）、**Push notification / webhook**（数小时的任务）
- A2A 标准化的是**通信**，不是编排；委派、拆分和停止仍由 harness 负责

## 7. MCP：Agent ↔ Capability

- 标准化：tools（可调用、typed）、resources（host 可附加的可读数据）、prompts（可复用模板）；一个 client 连接多个 servers；由 **host 而不是 model** 决定暴露什么
- MCP 是 agent 的 service boundary：一次 tool call，一个结果

### MCP vs A2A

| | MCP | A2A |
|---|---|---|
| 主要抽象 | capability / tool | agent / task |
| 关系 | agent → service | agent → agent |
| Discovery | tools、resources、prompts | agent capabilities |
| 工作单元 | 一次 tool invocation | 长时间运行的 task |
| 内部状态 | capability interface | 理想情况下不透明 |
| 类比 | 设备 / API 接口 | service-to-service RPC |

一个 A2A remote agent 自身可能调用 20 个 MCP servers。

---

## 8. 设计原则

1. **真的需要 multi-agent 吗？** 单 agent 往往就够了，多 agent 常常只增加开销和复杂度
2. **决定 topology**：反映任务依赖，最小化通信
3. **识别信息丢失点**：聚合节点是细节丢失的地方
4. **识别 failure-containment 边界**：确保系统不会卡死

> 稠密 mesh 很少是好的默认选择。更多通信会增加 context 压力，却不一定增加有用信息。
