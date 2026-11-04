# Chapter 4 智能体经典范式构建｜学习笔记

## 0. 本章在解决什么问题

第三章解决的是：

> LLM 本身是怎么工作的？

第四章开始解决：

> 如何把 LLM、工具、上下文和控制逻辑组装成一个真正能够完成任务的 Agent？

本章重点学习三种经典 Agent 范式：

```text
ReAct
→ 边思考、边行动、边根据结果调整

Plan-and-Solve
→ 先做完整计划，再逐步执行

Reflection
→ 先执行，再反思，再优化
```

它们的区别，本质上在于：

> **Agent 如何组织自己的“思考—行动—反馈—改进”过程。**

---

# 4.1 环境准备与基础组件

## 4.1.1 为什么先封装 LLM Client

我们实际运行的是：

```text
【本地】
Agent 控制逻辑
↓
LLM Client
↓
API

──────── 网络边界 ────────

【云端】
Tokenizer
↓
LLM
↓
生成 Token
↓
Decode
↓
返回文本
```

所以：

> Agent 可以运行在本地，但 LLM 不一定部署在本地。

我们通过 API 调用在线模型。

---

## 4.1.2 `.env` 文件

`.env` 用来保存模型调用相关配置，例如：

```text
LLM_API_KEY
LLM_MODEL_ID
LLM_BASE_URL
```

可以理解为：

```text
API Key
→ 身份凭证 / 调用权限

Base URL
→ 去哪个模型服务商

Model ID
→ 调用哪个具体模型
```

这些配置与程序代码分离，可以避免把 API Key 直接写进代码。

---

# 4.1.3 HelloAgentsLLM

教材封装：

```python
class HelloAgentsLLM:
```

目的不是构建 Agent，而是建立一个：

> **本地 Agent 与在线 LLM 服务之间的统一通信接口。**

核心结构：

```text
HelloAgentsLLM
├── __init__()
│   → 初始化模型配置
│   → 创建 API Client
│
└── think()
    → 把 messages + temperature 发给模型
    → 获取模型返回内容
```

---

## `__init__()` 在做什么

例如：

```python
self.model = model
self.client = OpenAI(...)
```

就是给当前这个 LLM Client 保存：

- 模型
- API 客户端
- 相关配置

---

## `think()` 在做什么

```text
messages
+
temperature
↓
API Client
↓
在线 LLM
↓
模型生成结果
↓
返回字符串
```

`messages` 可以包含：

- System Prompt
- 用户当前输入
- 历史对话
- Observation
- 其他 Context

需要注意：

> LLM 通常不会因为“之前调用过”就自动知道之前的内容。

如果下一次请求希望模型知道历史信息，就需要再次把相关历史放进 `messages / prompt` 中。

---

## 本地和 LLM 之间传输的是什么

通常 API 传输的是：

```json
{
  "model": "...",
  "messages": [...],
  "temperature": 0
}
```

也就是文本 + JSON 结构。

不是本地先把文字转换成 Token ID 再发送。

通常流程是：

```text
本地文字
↓
API
↓
云端 Tokenizer
↓
Token IDs
↓
LLM
↓
生成 Token IDs
↓
Decode
↓
文字
↓
返回本地
```

---

## Stream 流式输出

教材使用：

```python
stream=True
```

模型不是等整段回答完成后一次返回，而是：

```text
生成一点
↓
返回一个 chunk
↓
再生成一点
↓
再返回一个 chunk
```

代码最后再把所有 chunk 拼起来：

```text
chunk1
+
chunk2
+
chunk3
↓
完整 response
```

---

# 4.2 ReAct

ReAct：

**Reasoning + Acting**

即：

> 推理 + 行动

核心思想：

```text
Thought
↓
Action
↓
Observation
↓
新的 Thought
↓
新的 Action
↓
...
↓
Finish
```

它不是一开始就把整个解决路径设计好，而是：

> **走一步，看一步，根据 Observation 动态决定下一步。**

---

# 4.2.1 ReAct 的三个核心元素

## Thought

思考当前应该做什么。

例如：

```text
用户问的是实时信息
↓
我的内部知识可能已经过时
↓
应该调用 Search
```

---

## Action

真正决定采取什么行动。

例如：

```text
Search[华为最新手机]
```

---

## Observation

工具执行以后返回的信息。

例如：

```text
Search
↓
网页搜索
↓
搜索结果摘要
↓
Observation
```

新的 Observation 会重新进入 Context：

```text
Question
+
之前的 Action
+
Observation
↓
LLM
↓
新的 Thought / Action
```

因此形成循环。

---

# 4.2.2 Tool：Agent 如何与外部世界连接

如果 LLM 是大脑，那么 Tool 就相当于 Agent 可以调用的外部能力。

例如：

```text
Search
Calculator
Database
Maps
Weather API
Code Interpreter
```

一个 Tool 通常包含三个核心信息：

```text
Name
→ 工具叫什么

Description
→ 这个工具什么时候应该使用

Execution Logic
→ 真正执行任务的函数
```

其中 Description 很重要，因为：

> LLM 根据工具描述判断自己应该调用哪个工具。

---

# Search Tool

教材实现了：

```python
def search(query):
```

它做的是：

```text
query
↓
SerpApi
↓
Google Search
↓
返回 results JSON
↓
提取有价值的信息
↓
返回文字 Observation
```

---

## SerpApi 返回结果为什么需要解析

API 可能返回一个很大的 JSON：

```text
results
├── answer_box
├── knowledge_graph
├── organic_results
├── metadata
├── pagination
└── ...
```

Agent 不需要其中所有字段。

因此教材按照优先级：

```text
answer_box_list
↓
answer_box.answer
↓
knowledge_graph.description
↓
organic_results 前三条
```

找到最有用的信息就直接 `return`。

这里所谓“智能解析”，严格来说其实是：

> **基于规则的优先级解析。**

并不是另一个 AI 在判断。

---

## 为什么不直接把完整 JSON 给 LLM

不是因为 LLM 看不懂 JSON。

LLM 一般可以很好地处理 JSON。

但完整 API 返回结果可能：

- Token 很多
- 信息冗余
- 包含大量无关字段

所以：

```text
原始 JSON
↓
代码筛选
↓
保留真正有用的信息
↓
形成精简 Observation
↓
交给 LLM
```

可以降低 Token 成本和信息噪音。

---

# 4.2.2 ToolExecutor

当 Agent 拥有很多 Tool 时，需要统一管理。

教材定义：

```python
class ToolExecutor:
```

本质上是：

> **工具注册表 + 工具查找器。**

核心：

```text
ToolExecutor
├── tools
├── registerTool()
├── getTool()
└── getAvailableTools()
```

---

## `self.tools`

开始时：

```python
self.tools = {}
```

这是一个工具仓库。

注册 Search 后可能变成：

```python
{
    "Search": {
        "description": "网页搜索工具",
        "func": search
    }
}
```

---

## `registerTool()`

作用：

> 把工具存进工具箱。

```text
Name
+
Description
+
Function
↓
registerTool()
↓
self.tools
```

---

## `getTool()`

作用：

> 根据名字找到真正的工具函数。

```text
"Search"
↓
self.tools
↓
找到 Search
↓
找到 func
↓
得到 search 函数
```

因此：

```python
tool_function = toolExecutor.getTool("Search")
```

就是：

> 去工具箱找到 Search 对应的真实函数。

然后：

```python
observation = tool_function(tool_input)
```

才是真正执行 Tool。

---

## `getAvailableTools()`

作用：

> 把目前所有工具的 Name + Description 整理成文字，告诉 LLM 它有哪些工具。

例如：

```text
- Search: 用于搜索互联网
- Calculator: 用于数学计算
```

然后放进 Prompt。

因此 Agent 使用 Tool 涉及两件不同的事：

```text
LLM 决策之前
↓
getAvailableTools()
↓
告诉 LLM 有哪些工具

LLM 选择工具之后
↓
getTool()
↓
真正找到工具函数
↓
执行
```

---

# Python 代码阅读补充

## class 是什么

```python
class ReActAgent:
```

可以理解成：

> 定义一种对象应该具有什么属性和能力的模板。

例如：

```text
class ReActAgent
= ReAct Agent 的模板

agent = ReActAgent(...)
= 根据模板创建出来的一个具体 Agent
```

---

## self 是什么

`self`：

> 当前正在操作的这个实例本身。

例如：

```python
agent = ReActAgent(...)
```

在 `agent` 内部：

```python
self
```

可以近似理解成：

```python
agent
```

因此：

```python
self.tool_executor
```

就是：

> 当前这个 Agent 自己的 tool_executor。

而：

```python
self.tool_executor.getTool(tool_name)
```

就是：

> 让当前这个 ReActAgent 自己的 ToolExecutor 根据名字寻找 Tool。

---

## 实例化

```python
agent = ReActAgent(...)
```

叫：

**Instantiation（实例化）**

即：

> 根据 `class ReActAgent` 创建一个真正可以运行的 Agent。

假设：

```python
def __init__(
    self,
    llm_client,
    tool_executor,
    max_steps=5
):
```

那么创建时需要提供：

```python
agent = ReActAgent(
    llm_client,
    tool_executor
)
```

其中：

```text
llm_client
tool_executor
```

没有默认值，所以必须提供。

而：

```text
max_steps = 5
```

已有默认值，可以不写。

`self` 不需要手动填写，Python 自动处理。

---

# 4.2.3 ReActAgent

前面的工作其实是在准备三个东西：

```text
LLM Client
+
Tools / ToolExecutor
+
Agent 控制逻辑
```

最终：

```python
class ReActAgent:
```

负责把它们组装起来。

---

## ReAct Prompt

Prompt 中包含：

```text
角色
+
可用 Tools
+
Thought / Action 输出格式
+
Question
+
History
```

因此：

```text
Question
+
History
+
Tool Description
↓
LLM
↓
Thought
+
Action
```

---

## ReActAgent 的核心循环

```text
用户 Question
↓
构建 Prompt
↓
LLM
↓
Thought + Action
↓
解析 Action

如果：
Finish[answer]
↓
返回最终答案

否则：
找到 Tool
↓
执行 Tool
↓
得到 Observation
↓
写入 History
↓
下一轮
```

也就是：

```text
Question
↓
Thought
↓
Action
↓
Tool
↓
Observation
↓
History
↓
Thought
↓
...
↓
Finish
```

---

## 为什么需要 max_steps

例如：

```python
max_steps = 5
```

属于 Safety Boundary（安全边界）。

防止：

```text
Agent 一直循环
↓
不断调用 LLM / Tool
↓
成本增加
↓
延迟增加
↓
甚至死循环
```

---

## `agent.run(question)`

创建 Agent：

```python
agent = ReActAgent(
    llm_client,
    tool_executor
)
```

只是完成组装。

真正开始执行任务：

```python
agent.run(question)
```

因此：

```text
ReActAgent(...)
→ 创建 Agent

agent.run(...)
→ 让 Agent 开始干活
```

如果做命令行交互，还可以：

```python
question = input("请输入问题：")
agent.run(question)
```

教材重点是 Agent 内部机制，因此没有专门实现聊天 UI。

---

# 4.2.4 ReAct 的特点

## 优势

### 动态调整能力强

```text
Action
↓
Observation
↓
发现情况与预期不同
↓
调整下一步 Action
```

因此特别适合：

- 实时搜索
- 故障排查
- 探索性任务
- Tool Calling
- 路径无法提前确定的任务

---

### 工具协同能力强

LLM 负责：

```text
判断
规划
选择工具
```

Tool 负责：

```text
搜索
计算
查询数据库
执行 API
```

---

## 局限

### 缺乏全局规划

ReAct 更像：

```text
走一步
↓
看结果
↓
再决定下一步
```

因此可能出现：

- 绕路
- 重复尝试
- 局部最优
- 原地打转

### 成本与延迟

每一轮可能都要：

```text
LLM Call
+
Tool Call
```

复杂任务循环很多次时成本会上升。

---

# 4.3 Plan-and-Solve

Plan-and-Solve 的核心：

```text
Plan
↓
Solve
```

即：

> **先整体规划，再严格执行。**

与 ReAct 最大区别：

```text
ReAct
→ 下一步根据刚刚的 Observation 决定

Plan-and-Solve
→ 一开始就生成完整计划
```

---

# 4.3.1 工作流程

```text
用户 Question
↓
Planner
↓
生成 Plan

[
  Step 1,
  Step 2,
  Step 3
]

↓
Executor
↓
Step 1
↓
Step 2
↓
Step 3
↓
Final Answer
```

适合：

- 多步数学问题
- 报告撰写
- 代码设计
- 结构清楚的复杂任务

---

# ReAct vs Plan-and-Solve 不是非此即彼

现实任务中很多场景两种范式都能使用。

例如：

> 明天去哪玩？

可以先：

```text
Plan：
1. 查天气
2. 找候选地点
3. 查交通
4. 查人流
5. 综合比较
```

但执行后：

```text
查天气
↓
发现下雨
↓
原计划不再合理
↓
改查室内地点
```

这里又出现 ReAct。

因此真实 Agent 更常见：

```text
先做粗 Plan
↓
执行
↓
获得 Observation
↓
判断计划是否仍合理
↓
继续 / 局部调整 / Replan
```

所以实际选择时最重要的问题不是：

> “这个任务属于 ReAct 还是 Plan-and-Solve？”

而是：

> **任务开始前，下一步该做什么能不能可靠确定？**

如果：

```text
路径高度依赖实时反馈
→ 更偏 ReAct
```

如果：

```text
任务结构一开始就比较清楚
→ 更偏 Plan-and-Solve
```

---

# 4.3.2 Planner

Planner：

> 专门负责把复杂任务拆成多个可执行步骤。

Prompt 要求 LLM 输出：

```python
["步骤1", "步骤2", "步骤3"]
```

这样做的原因：

> 自然语言对人容易理解，但程序很难稳定解析；固定格式更容易被普通代码处理。

---

## `PLANNER_PROMPT_TEMPLATE`

它就是提前定义好的 Prompt 模板。

例如：

```python
prompt = PLANNER_PROMPT_TEMPLATE.format(
    question=question
)
```

意思是：

```text
Prompt 模板
+
当前 Question
↓
形成真正发送给 LLM 的 Prompt
```

---

## 为什么还要解析 LLM 输出

LLM 返回的是：

```text
字符串 String
```

例如：

```text
'["步骤1", "步骤2", "步骤3"]'
```

虽然它看起来像 Python List，但本质仍是文字。

程序真正希望得到：

```python
["步骤1", "步骤2", "步骤3"]
```

一个真实的 List。

因此：

```text
response_text
↓
split()
↓
从 ```python ... ``` 中提取计划字符串
↓
ast.literal_eval()
↓
转换成真正的 Python List
```

---

## `ast.literal_eval()`

作用：

```text
字符串：
'["步骤1", "步骤2"]'

↓ ast.literal_eval()

Python List：
["步骤1", "步骤2"]
```

这一步体现一个很重要的 Agent 工程思想：

> **LLM 负责生成；普通程序负责把输出转换成可靠、可执行的结构。**

---

# 4.3.3 Executor

教材这里的：

```python
class Executor:
```

和 ReAct 的：

```python
ToolExecutor
```

不是一回事。

### ToolExecutor

```text
管理和执行外部 Tool
```

### Plan-and-Solve Executor

```text
逐步执行 Planner 制定出来的 Plan
```

教材这个示例没有使用外部 Tool。

Executor 的每一步其实是：

```text
当前 Step
+
原始 Question
+
完整 Plan
+
之前执行结果
↓
LLM
↓
完成这个 Step
```

然后把结果记录进 History：

```text
Step 1 → Result 1
↓
History

Step 2 + History
↓
Result 2
```

所以教材里的 Executor 本质是：

> **调用 LLM 一个步骤一个步骤地完成任务。**

不是调用 Search / Calculator。

---

# 4.3.4 PlanAndSolveAgent

最后把：

```text
Planner
+
Executor
```

组合起来：

```python
class PlanAndSolveAgent:
```

完整结构：

```text
Question
↓
Planner
↓
Plan
↓
Executor
↓
逐步 Solve
↓
Final Answer
```

---

# Planner 和 Executor 未来可以怎么扩展

现实系统里：

```text
Planner
↓
生成 Plan
↓
Executor
```

Executor 不一定只能调用 LLM。

它完全可以：

```text
Executor
├── LLM
├── Search Tool
├── Database
├── Weather API
└── 甚至内部再跑 ReAct
```

因此：

> Planner 负责“想清楚整件事怎么做”，Executor 负责“真正把每一步做掉”。

---

# 4.4 Reflection

Reflection 的核心不是第一次就做对，而是：

> **先做一版，再通过反思不断改善。**

核心循环：

```text
Execution
↓
Reflection
↓
Refinement
↓
Reflection
↓
Refinement
↓
...
```

也就是：

```text
执行
↓
评估
↓
发现问题
↓
修改
↓
再评估
```

---

# 4.4.1 Reflection 的三个阶段

## Execution

先生成一个初稿。

例如：

```text
用户：写一段代码
↓
LLM
↓
初版代码
```

---

## Reflection

让模型扮演 Evaluator / Reviewer：

```text
初稿
↓
检查：
事实错误？
逻辑错误？
遗漏？
效率问题？
↓
生成 Feedback
```

---

## Refinement

把：

```text
原始任务
+
上一版输出
+
Feedback
```

重新交给模型：

```text
→ 生成改进版本
```

---

# Reflection 与 Memory

这一句话非常重要：

> **Reflection 的核心在于迭代，而迭代的前提是能够记住之前的尝试和获得的反馈。**

因此可以理解成：

> **Reflection = 有记忆的迭代。**

如果没有 Memory：

```text
第一次执行
↓
反馈
↓
忘掉
↓
第二次重新开始
```

这不是有效迭代。

有 Memory：

```text
第一次执行
↓
反馈
↓
记录
↓
下一轮基于之前的问题继续改进
```

---

# 短期记忆与长期经验

教材的 Reflection 示例主要实现的是：

> Short-term Memory（短期记忆）

即当前任务中的执行轨迹：

```text
Execution 1
↓
Reflection 1
↓
Execution 2
↓
Reflection 2
```

这可以称为：

**Trajectory（轨迹）**

例如：

```text
方法 A
↓
失败
↓
Reflection：A 效率太低
↓
方法 B
↓
成功
```

整个过程属于当前任务的短期轨迹。

---

更严格地区分：

```text
Trajectory / Short-term Memory
= 这次任务里具体发生了什么

Experience / Long-term Memory
= 从多个轨迹中沉淀出的、未来还能复用的经验
```

例如：

```text
Trajectory：
A → 报错 → B → 成功

↓

Experience：
“遇到这种输入时，应该先做格式检查再使用 B。”
```

教材这一章重点实现的是短期 Memory。

真正系统的长期 Memory 会在后续章节继续学习。

---

# 4.4.2 Memory 类

Memory 内部主要有：

```text
records
add_record()
get_trajectory()
get_last_execution()
```

---

## records

```python
self.records = []
```

用于：

> 按顺序保存 Execution 和 Reflection。

例如：

```text
[
  execution 1,
  reflection 1,
  execution 2,
  reflection 2
]
```

---

## add_record()

作用：

> **写入 Memory。**

```text
新的 Execution / Reflection
↓
add_record()
↓
records
```

一句话：

> `add_record = 存`

---

## get_trajectory()

作用：

> 把已经存进去的历史轨迹读取出来，并整理成适合放进 Prompt 的文字。

```text
records
↓
get_trajectory()
↓
整理成文本
↓
放进 Context
↓
LLM 看见之前发生过什么
```

一句话：

> `get_trajectory = 取 + 整理`

因此：

```text
add_record
≠
get_trajectory

add_record
→ 写入

get_trajectory
→ 读取并整理
```

---

## get_last_execution()

作用：

> 找到最近一次执行的结果。

因为 Reflection 需要知道：

```text
上一版输出是什么？
```

才能对它进行评审。

---

# 4.4.3 ReflectionAgent

Reflection 需要三套 Prompt。

### Initial Prompt

```text
完成第一次任务
↓
得到初稿
```

### Reflection Prompt

```text
初稿
↓
评审
↓
得到 Feedback
```

### Refinement Prompt

```text
初稿
+
Feedback
↓
生成优化版本
```

完整循环：

```text
Initial Execution
↓
Memory.add_record(execution)

↓
Reflection
↓
Memory.add_record(reflection)

↓
是否“无需改进”？

是
→ Finish

否
↓
Refinement
↓
Memory.add_record(execution)
↓
下一轮 Reflection
```

---

# Reflection 的优势与成本

## 优势

适合需要：

- 更高准确性
- 更强可靠性
- 更高质量
- 复杂推理
- 代码优化
- 报告审查

的任务。

Reflection 不只是修正“错误”，也可以：

> 把一个合格方案不断优化成更好的方案。

---

## 成本

每一轮通常至少需要：

```text
一次 Reflection LLM Call
+
一次 Refinement LLM Call
```

因此：

```text
质量 ↑
同时
成本 ↑
延迟 ↑
Prompt 复杂度 ↑
```

所以 Reflection 本质是一种：

> **以成本换质量的策略。**

---

# 三种 Agent 范式对比

| 范式 | 核心逻辑 | 适合什么任务 | 主要问题 |
|---|---|---|---|
| ReAct | 边做边看反馈 | 动态、实时、探索性任务 | 缺少全局规划 |
| Plan-and-Solve | 先规划再执行 | 结构清晰、多步骤任务 | 初始 Plan 可能过时 |
| Reflection | 执行后反思优化 | 高质量、高可靠任务 | 成本和延迟高 |

---

# 三种范式的核心流程

## ReAct

```text
Thought
↓
Action
↓
Observation
↓
Thought
↓
Action
↓
...
```

核心问题：

> 下一步应该做什么？

---

## Plan-and-Solve

```text
Question
↓
Plan
↓
Step 1
↓
Step 2
↓
Step 3
↓
Final Answer
```

核心问题：

> 整个任务应该怎么拆？

---

## Reflection

```text
Execution
↓
Reflection
↓
Refinement
↓
Reflection
↓
Refinement
```

核心问题：

> 已经做出来的结果还能怎么变好？

---

# 实际 Agent 往往是混合架构

现实产品通常不会机械地：

```text
只能 ReAct
或
只能 Plan-and-Solve
或
只能 Reflection
```

而可能是：

```text
Plan-and-Solve
↓
先做整体 Plan

↓

执行某一步时
↓
ReAct
根据实时 Observation 调整

↓

得到最终方案以后
↓
Reflection
检查并优化
```

于是形成：

```text
Plan
↓
Act
↓
Observe
↓
Replan / Continue
↓
完成初稿
↓
Reflect
↓
Refine
↓
Final
```

所以 AI 产品设计真正需要判断的是：

```text
哪些内容应该提前规划？

哪些节点需要根据实时反馈动态决定？

哪些结果值得花额外成本进行反思和优化？
```

而不是机械判断：

> “这个场景属于哪个范式？”

---

# 第四章代码结构总览

这一章从代码角度其实一直在重复同一种系统设计思想：

```text
① 封装基础能力
↓
LLM Client

② 准备外部能力
↓
Tools / Memory / Planner 等组件

③ 定义控制逻辑
↓
Agent Class

④ 实例化
↓
agent = XxxAgent(...)

⑤ 输入任务
↓
agent.run(question)

⑥ Agent Loop
↓
持续调用 LLM / Tool / Memory
直到 Finish
```

---

# 第四章最重要的 Agent 工程思想

## 1. LLM 不等于 Agent

LLM：

```text
输入
↓
生成
```

Agent：

```text
LLM
+
Prompt
+
Context / Memory
+
Tools
+
控制逻辑
+
循环
```

---

## 2. LLM 负责“判断”，普通代码负责“控制”

例如：

```text
LLM：
我应该 Search

↓
普通代码：
解析 Action

↓
ToolExecutor：
找到 search()

↓
普通代码：
真正执行 search()

↓
得到 Observation

↓
重新交给 LLM
```

因此 Agent 并不是“全部交给 AI”。

而是：

> **概率性的 LLM + 确定性的程序逻辑共同工作。**

---

## 3. Structured Output 很重要

LLM 自然语言非常灵活，但程序不容易解析。

因此需要：

```text
Thought:
Action:

或者：

["Step 1", "Step 2"]

或者：

JSON
```

通过固定输出格式，把：

```text
自然语言模型
↓
结构化结果
↓
普通代码
```

连接起来。

---

## 4. History / Memory 是 Agent 能迭代的基础

如果 Observation、执行结果、Reflection 不重新进入 Context：

```text
Agent
↓
每一轮都像第一次开始
```

因此：

```text
Action
↓
Observation
↓
History / Memory
↓
下一轮 LLM
```

是 Agent Loop 的关键。

---

## 5. Agent 的自主性来自“让谁决定下一步”

普通 Workflow：

```text
开发者提前写死
A → B → C → D
```

Agent：

```text
Goal
↓
LLM 根据 Context
↓
决定下一步 Action
```

实际产品常用：

```text
固定 Workflow
+
部分 Agent 决策节点
+
必要时 Human-in-the-loop
```

---

# AI 产品视角：第四章真正需要掌握什么

学完这一章，不需要做到：

> 从零手写一个完整 Agent Framework。

更重要的是能够回答：

### 1. 为什么这个场景需要 Agent？

因为任务路径存在不确定性，需要模型根据 Context 动态决策。

### 2. 为什么需要 Tool？

因为 LLM 自身可能：

- 没有实时信息
- 无法访问企业数据
- 计算不可靠
- 无法执行真实操作

### 3. Tool 应该如何提供给 Agent？

```text
Name
+
Description
+
Input
+
Execution Logic
```

### 4. 什么时候用 ReAct？

路径依赖实时反馈：

```text
Observation
↓
决定下一步
```

### 5. 什么时候用 Plan-and-Solve？

任务结构较清楚：

```text
先整体拆解
↓
再逐步执行
```

### 6. 什么时候加入 Reflection？

当：

```text
结果质量的重要性
>
额外延迟和成本
```

### 7. 为什么需要 Memory？

因为：

> Reflection 的核心是迭代，而迭代的前提是记得之前做过什么、哪里出了问题。

---

# 第四章术语速查

| 英文 / 缩写 | 中文 | 核心含义 |
|---|---|---|
| Agent | 智能体 | 能感知、决策并行动的系统 |
| ReAct | Reasoning + Acting | 思考与行动交替进行 |
| Thought | 思考 | 判断当前情况和下一步 |
| Action | 行动 | Agent 决定执行的操作 |
| Observation | 观察 | Action 执行后的反馈 |
| Tool | 工具 | Agent 可调用的外部能力 |
| Tool Calling | 工具调用 | Agent 选择并调用 Tool |
| ToolExecutor | 工具执行器 | 管理和查找 Tool |
| API | Application Programming Interface，应用程序接口 | 程序调用外部服务的接口 |
| Client | 客户端 | 本地调用远程服务的接口组件 |
| JSON | JavaScript Object Notation | 常见结构化数据格式 |
| Stream | 流式响应 | 内容生成一点返回一点 |
| Prompt Template | 提示词模板 | 可动态插入变量的 Prompt |
| History | 历史记录 | 当前任务中此前发生的信息 |
| Plan | 计划 | 对任务进行整体拆解 |
| Planner | 规划器 | 负责生成 Plan |
| Executor | 执行器 | 按 Plan 完成任务步骤 |
| Plan-and-Solve | 先规划后执行 | 两阶段 Agent 范式 |
| Orchestrator | 编排器 / 协调器 | 调度多个组件工作的模块 |
| Reflection | 反思 | 对已有结果进行评估 |
| Feedback | 反馈 | 反思后得到的问题和建议 |
| Refinement | 优化 / 修订 | 根据 Feedback 修改结果 |
| Memory | 记忆 | 保存 Agent 的历史信息 |
| Trajectory | 轨迹 | 当前任务的执行过程 |
| Short-term Memory | 短期记忆 | 当前任务内的历史轨迹 |
| Long-term Memory | 长期记忆 | 跨任务可复用的经验 |
| Experience | 经验 | 从历史中沉淀的可复用信息 |
| Instantiation | 实例化 | 根据 class 创建具体对象 |
| Class | 类 | 定义一种对象的模板 |
| Instance | 实例 | 根据 class 创建的具体对象 |
| Context | 上下文 | 当前模型可以使用的信息 |
| Human-in-the-loop | 人在回路 | 关键节点由人工参与 |
