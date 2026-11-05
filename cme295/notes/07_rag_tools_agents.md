# Lecture 7 · RAG、Tool Calling 与 Agents

> Stanford CME 295：Transformers & Large Language Models
> 本笔记可独立阅读，原始 slides 仅作参考：[`slides/fall25-cme295-lecture7.pdf`](../slides/fall25-cme295-lecture7.pdf)

## 本讲脉络

```
普通 LLM 的短处：知识静态、不能行动
→ RAG：动机 → 建知识库 → 检索（候选召回 + 重排序）→ 检索评估指标
→ Tool calling：原理 → 如何教模型用 tool → 工具选择 → MCP
→ Agents：定义 → ReAct → multi-agent 与 A2A → 安全 → 总结建议
```

一句话总结：**RAG 让 LLM 读到它没学过的知识，tool calling 让 LLM 能查询和操作外部世界，agent 则把推理和工具调用放进一个循环，自主完成多步任务。**

---

## 0. 回顾与本讲定位

| 普通 LLM 的短处 | 对应讲次 |
|---|---|
| 推理能力有限 | Lecture 6（reasoning models） |
| **知识是静态的** | 本讲：RAG |
| **不能执行动作** | 本讲：tool calling、agents |
| 难以评估 | Lecture 8 |

---

## 1. RAG（Retrieval-Augmented Generation）

### 1.1 动机

- **知识受限于预训练数据**：模型有 knowledge cutoff（例如 GPT-5 的模型卡写明了截止日期），也不知道你的私有文档
- **Context 大小有限**：不能把整个知识库塞进 prompt
- **无关信息会分散 LLM 的注意力**：“Needle in a Haystack” 测试显示，信息埋在长 context 中间时容易被忽略
- **按输入/输出 token 计费**：塞太多内容很贵

### 1.2 RAG 概览（Lewis et al., 2020）

**思路**：用相关的信息片段增强 prompt。三步：

1. **Retrieve**：在知识库中用相似度运算找到相关文档
2. **Augment**：把检索到的信息加到 prompt 里（用户问题 + 检索内容）
3. **Generate**：LLM 基于增强后的 prompt 生成回答

### 1.3 前提：建立知识库

```
收集文档（Collect）→ 切分成 chunk（Divide）→ 每个 chunk 转成向量（Embed）→ 存入向量数据库
```

超参数：
- **Embedding size**：向量维度
- **Chunk size**：太大则检索粒度粗、含无关信息；太小则缺上下文
- **Chunk 之间的 overlap**：防止关键信息恰好被切断

### 1.4 检索的两阶段

| 阶段 | 目标 | 方法 |
|---|---|---|
| 🎣 **Step 1：候选召回（Candidate retrieval）** | **最大化 recall**：别漏掉相关的 | 语义 embedding，可选加关键词方法；在整个知识库上快速搜索 |
| 💯 **Step 2：排序（Ranking）** | **最大化 precision**：把最相关的排前面 | 在小候选集上用更精细（更慢）的 re-ranker |

### 1.5 Step 1：候选召回

**方法 1：Semantic search（基于 embedding 的相似度）**
- 用同一个 encoder 分别把 query 和每个 chunk 编码成向量，计算相似度（通常是 cosine similarity）
- 这种结构叫 **bi-encoder**（Sentence-BERT，Reimers et al., 2019）：query 和文档**分开编码**，所以文档向量可以离线预先算好，查询时只需编码 query，再做最近邻搜索，很快
- 例：query “Where is Cuddly?”，候选 chunk 中 “Cuddly spends most days surrounded by books” 得分高；“he was in a cuddly mood” 只是词面相似

**方法 2：Keyword matching（BM25）**
- 经典的词频统计方法：一个词在 chunk 中出现越多、在整个语料中越罕见，得分越高（TF-IDF 的改进版，并对文档长度做了归一化）
- 擅长精确匹配：专有名词、编号、代码标识符

**方法 3：Hybrid**
- 把语义分数和 BM25 分数结合，取长补短

**有助于召回的扩展技巧**：

- **HyDE**（Hypothetical Document Embeddings，Gao et al., 2022）
  - 问题：问题（短、疑问句）和文档（长、陈述句）的 embedding“性质”不同，直接比较不理想
  - 做法：先让 LLM 根据 prompt 写一篇**假的回答文档**（“Cuddly is in…”），再用这篇假文档的 embedding 去检索真实 chunk。假文档内容可能不对，但“长得像”真文档

- **Contextual Retrieval**（Anthropic, 2024）
  - 问题：单独一个 chunk 常缺少上下文（例如 “该公司收入增长 3%”，不知道是哪家公司、哪个季度）
  - 做法：对每个 chunk，把**整篇文档 + 该 chunk** 交给 LLM，让它生成一小段上下文说明，前置到 chunk 上再做 embedding 和 BM25 索引。Prompt 大意：

    ```
    <document>{WHOLE_DOCUMENT}</document>
    Here is the chunk we want to situate within the whole document:
    {CHUNK_CONTENT}
    Please give a short succinct context to situate this chunk within the overall
    document for the purposes of improving search retrieval of the chunk.
    Answer only with the succinct context and nothing else.
    ```

  - 成本：每个 chunk 都要发送一次整篇文档。用 **prompt caching**（缓存的输入 token 价格便宜很多）可以大幅降低成本，因为同一文档的所有 chunk 共享相同的长前缀

### 1.6 Step 2：Re-ranking

- 用更复杂的 **cross-encoder** 给每个候选打最终相关性分数
- **Cross-encoder**：把 query 和 chunk **拼在一起**输入一个模型（如 BERT），直接输出相关性分数。两者在每一层都能互相 attend，所以比 bi-encoder 准确得多
- 代价：不能预先计算，每对 (query, chunk) 都要跑一次模型 —— 所以只用在第一阶段筛出的少量候选上
- 输出：重新排序的 chunk 列表（如 d、b、a、c）

| | Bi-encoder | Cross-encoder |
|---|---|---|
| 输入 | query 和文档分别编码 | query + 文档拼接后一起编码 |
| 速度 | 快，文档向量可预计算 | 慢，每对都要计算 |
| 准确度 | 较低 | 较高 |
| 用途 | 候选召回 | 重排序 |

### 1.7 检索效果评估

设定：检索返回一个排序列表，取 Top-k，判断每个 chunk 是否相关。

| 指标 | 定义 | 关注点 |
|---|---|---|
| **Precision@k** | Top-k 中相关 chunk 的比例 | 返回的有多少是有用的 |
| **Recall@k** | 所有相关 chunk 中有多少出现在 Top-k | 有没有漏 |
| **RR@k**（Reciprocal Rank） | 1 / （第一个相关 chunk 的排名）；Top-k 中没有则为 0。多个 query 平均后为 **MRR** | 第一个有用结果出现得多早 |
| **NDCG@k**（Normalized Discounted Cumulative Gain） | DCG@k = Σ_{i=1..k} rel_i / log₂(i + 1)；NDCG@k = DCG@k / IDCG@k，其中 IDCG 是完美排序时的 DCG | 相关的是否排在前面，排得越靠后贡献越小；取值 0–1 |

---

## 2. Tool Calling

### 2.1 思路

除了非结构化文本，很多信息存在**结构化**数据中（如数据库表：ID、字段、值）。与其把数据塞进 prompt，不如给 LLM 一个**函数**，让它在需要时调用：

```python
def get_data(id, field, ...):
    # Logic.
    return result
```

**定义**（IBM）：Tool calling 让自主系统能够动态访问外部资源并对其采取行动，从而完成复杂任务。

### 2.2 真实例子

- 普通 LLM：“Find a bear near me!” → “Sorry, I don't know which bears are near you.”
- 有 tools 的 LLM：可以调用 `find_teddy_bear` 函数

一个好的 tool 函数应该：
- 有**描述清楚、文档完善的 API**（函数名、参数类型、docstring）
- （可选）调用某个后端服务
- 返回一些信息

```python
@dataclass
class TeddyBearInfo:
    name: str
    distance_meters: float
    mood: str
    message: str

def find_teddy_bear(location: tuple[float, float]) -> TeddyBearInfo:
    """
    Finds the nearest teddy bear to the given GPS coordinates.
    Parameters:
        location: A (latitude, longitude) pair representing the user's current location.
    Returns:
        TeddyBearInfo: Information about the nearest teddy bear found.
    """
    # 调用后端 API 获取最近的 teddy bear，再用 geopy 计算距离并返回
    ...
```

### 2.3 工作流程

1. **LLM 决定调用哪个函数并生成参数**：`find_teddy_bear(location=(37.42, -122.17))`（LLM 只输出调用请求，不自己执行）
2. **系统执行函数调用**：后端返回 `{"name": "Teddy", …}`
3. **LLM 根据结果推出结论**：生成面向用户的回答

### 2.4 怎么教模型使用 tool

**方法 1：训练**
- 构造训练数据，教模型两件事：
  - **Tool prediction**：给定对话历史（如 “Find a bear in Paris!” + function API），期望输出是正确的调用 `location = (48.86, 2.35)`
  - **Response generation**：给定 tool 返回结果（`{"name": "Ours", …}`），期望输出是合适的回答

**方法 2：Prompting**
- 在 prompt 中放入 function API + **如何使用它的详细说明**
- 如何写好这段说明？一种方法：用 SFT 数据对作为评估集，让一个强大的 reasoning model 帮你写、改、测

### 2.5 常见用途

| 类型 | 例子 |
|---|---|
| **信息** | 网页 / 数据库搜索；天气、股票等实时数据；代码库 |
| **计算** | 计算器；代码执行（通常是 Python） |
| **动作** | 发邮件 / 消息；电脑上的其他操作；助手领域内的一切 |

### 2.6 实际中：很多 tools

一个 teddy bear 助手可能有：`find_teddy_bear`、`hug_teddy_bear`、`check_teddy_mood`、`send_teddy_gift`、`schedule_playdate`、`send_message`……全部放在 prompt 的函数列表中。

### 2.7 小结

| 好处 | 挑战 |
|---|---|
| LLM 一下子有用得多 | **Tools 越多，表现越差**（模型更容易选错） |
| 可以与真实世界交互 | Context 长度有限：不可扩展 |
| 克服 knowledge cutoff | 要定义很多 tools，工作量大 |

### 2.8 Tool selection（Robert et al., 2024）

**目标**：同时降低延迟和提升效果。
**做法**：加一个 **router**，根据当前请求先从所有 tools 中挑出相关的少数几个，只把被选中的 function API 放进 prompt。

### 2.9 MCP（Model Context Protocol，Anthropic, 2024）

**动机**：避免重复实现。没有标准时，每个 LLM 应用都要为每个 tool 写一份集成，是 N × M 的工作量。

**思路**：用一个**标准方式**把 tools 和数据连接到 LLM。

**架构**：
- **MCP host**：用户使用的应用（如 Claude Desktop）
- **MCP client**：host 内部，为每个 server 维护一个连接
- **MCP server**：暴露能力的服务，提供三种东西：
  - **Tools**：可调用的函数
  - **Prompts**：可复用的 prompt 模板
  - **Resources**：可读取的数据

**例子**：用户说 “Recommend a new poetry book to my teddy bear”。Claude Desktop（host）通过 MCP 连接一个 book provider server，它提供 `find_title`（查个人藏书）和 `recommend_taste`（根据口味推荐）等工具，以及个人藏书、畅销书等资源。

---

## 3. Agents

### 3.1 定义

“An agent is a system that autonomously pursues goals and completes tasks on a user's behalf.”
（Agent 是一个代表用户、自主追求目标并完成任务的系统。）

| 传统方式 | Agent |
|---|---|
| 输入 → LLM（可能调用一次工具）→ 输出 | 输入 → 循环：**Observe → Plan → Act** → … → 输出 |

### 3.2 ReAct（Reason + Act，Yao et al., 2022）

把推理和行动交替进行。每轮包含：

| 步骤 | 内容 |
|---|---|
| **Observe** | 综合之前的行动结果，明确目前已知的信息（包括模型自己的知识） |
| **Plan** | 推理密集的步骤：弄清楚还需要什么，详细说明要完成哪些任务、调用哪些 tools |
| **Act** | 通过 API 执行动作，或在文档数据库中查找信息 |

**例子：“My teddy bear is cold. Please do something.”**

触发输入可以是用户手动输入，也可以是外部事件（如某个指标超过阈值）。

1. **Observe**：用户的 teddy bear 冷了，可能是因为房间温度，而当前温度未知
2. **Plan**：确定房间温度
3. **Act**：`get_current_room_temperature()`
4. **Observe**：房间当前 65°F，比平均温度低约 5°F，需要升温
5. **Plan**：把温度提高 5°F
6. **Act**：`increase_temperature(value=5)`
7. **Observe**：恒温器现在设为 70°F，应该足够暖和了
8. **输出**：“The thermostat is now set to 70F. Your teddy bear will soon feel warmer.”

### 3.3 “Agentic” 视角与 multi-agent

上面的系统可以看作一个 **thermostat agent**。智能家居中还可能有很多 agents：occupancy agent（检测是否有人）、energy management agent、air quality agent……它们之间如何通信？

### 3.4 A2A（Agent2Agent Protocol，Google, 2025）

一个让 agents 互相发现和协作的标准协议，核心概念：

- **AgentSkill**：描述 agent 的一项能力
  - `id='maintain_comfort'`：设定目标温度；例：“Set to focus mode while I'm working”
  - `id='prepare_home'`：在用户到家前准备；例：“Warm up the house to 70F when I am back”
  - `id='optimize_energy'`：降低能耗；例：“Save energy overnight”
- **AgentCard**：agent 的“名片”，包括 `name='thermostat agent'`、`url`、`version='1.0.0'`、`skills=[...]`，供其他 agent 发现它能做什么
- **AgentExecutor**：实际执行请求的接口，包含 `execute(...)` 和 `cancel(...)`

MCP 连接的是 agent 与工具/数据，A2A 连接的是 agent 与 agent。

### 3.5 安全

**风险**：agent 能在真实世界中行动，因此可能造成真实的伤害。例：**数据外泄（data exfiltration）**—— 恶意内容诱导 agent 把用户的私有数据发送出去（ToolSword，Ye et al., 2024）。

**补救措施**：
- 训练阶段：做 tool use 的对齐（Chen et al., 2024，“Towards Tool Use Alignment of LLMs”）
- 推理阶段：加安全防护（safeguards）
- Benchmark：如 Agent-SafetyBench

现实案例：Anthropic 在 2025 年报告了首个被披露的、由 AI 编排的网络间谍活动 —— 这是非常重要的话题。

---

## 4. 总结与建议

- **Hallucination 是（大）问题**
- **推理能力是瓶颈**：fine-tuning 有帮助，但很难
- 新能力（tools、agents）非常受欢迎
- **评估很有挑战**
- **先从简单开始，再迭代、逐步扩展**
- **先用能力强的模型，之后再优化模型大小**
- **透明度 / 可观察性**有助于用户信任和调试

额外：AI agent 在日常生活中最好的用途之一是**写代码**。

---

## 参考论文

- Lewis et al., “Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks”, 2020
- Kamradt, “Needle in a Haystack - Pressure Testing LLMs”, 2023
- Reimers et al., “Sentence-BERT”, 2019
- Gao et al., “Precise Zero-Shot Dense Retrieval without Relevance Labels” (HyDE), 2022
- Anthropic, “Introducing Contextual Retrieval”, 2024
- Robert et al., “Automatic Tool Selection to Reduce Large Language Model Latency”, 2024
- Anthropic, “Introducing the Model Context Protocol”, 2024
- Yao et al., “ReAct: Synergizing Reasoning and Acting in Language Models”, 2022
- Google, “Announcing the Agent2Agent Protocol (A2A)”, 2025
- Ye et al., “ToolSword”, 2024；Chen et al., “Towards Tool Use Alignment of Large Language Models”, 2024
