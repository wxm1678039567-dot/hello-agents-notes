# Chapter 3 大语言模型基础｜学习笔记

## 3.1 语言模型与 Transformer 架构

### 3.1.1 从 N-gram 到 RNN / LSTM

#### 1. 语言模型

语言模型（Language Model, **LM**）的核心任务是：

> 计算一个词序列出现的概率，并基于已有文本预测后续内容。

现代大语言模型生成文本时，本质上也是不断预测下一个 Token 的概率分布。

---

#### 2. N-gram

N-gram 基于马尔可夫假设（Markov Assumption）：

> 预测当前词时，不考虑全部历史，只考虑前面有限的 n-1 个词。

例如：

- Bigram：只看前 1 个词
- Trigram：只看前 2 个词

流程：

```text
前 n-1 个词
↓
统计语料库 Corpus 中的出现频率
↓
估计下一个词的条件概率
```

概率通常通过最大似然估计（Maximum Likelihood Estimation, **MLE**）计算。

主要缺点：

1. 数据稀疏性（Sparsity）：没出现过的组合概率可能直接为 0
2. 泛化能力差：无法理解相似词之间的语义关系
3. 上下文窗口固定

---

#### 3. 神经网络语言模型

前馈神经网络语言模型（Feedforward Neural Network Language Model, **NNLM**）引入了词嵌入（Word Embedding）。

```text
词
↓
Embedding
↓
高维连续向量
↓
神经网络
↓
预测下一个词的概率分布
```

Embedding 的核心意义：

> 把词从离散符号变成连续向量，使语义相近的词在向量空间中更加接近。

余弦相似度（Cosine Similarity）可以用来衡量两个词向量的相似程度。

注意：

> 模型并不是通过“找夹角最小的词”直接预测下一个词；余弦相似度主要用于描述向量之间的语义关系。

NNLM 虽然解决了一部分泛化问题，但上下文窗口仍然固定。

---

#### 4. RNN

循环神经网络（Recurrent Neural Network, **RNN**）引入隐藏状态（Hidden State），将历史信息持续向后传递。

```text
当前输入
+
上一时刻 Hidden State
↓
新的 Hidden State
↓
传递给下一时刻
```

因此 RNN 不再只能看到固定数量的前文。

但长序列会产生长期依赖问题（Long-term Dependency Problem），以及梯度消失 / 梯度爆炸。

---

#### 5. LSTM

长短期记忆网络（Long Short-Term Memory, **LSTM**）属于特殊的 RNN。

核心增加：

- 细胞状态（Cell State）
- 门控机制（Gating Mechanism）

三个主要 Gate：

- Forget Gate：遗忘门，决定丢掉哪些旧信息
- Input Gate：输入门，决定加入哪些新信息
- Output Gate：输出门，决定输出哪些信息

核心作用：

> 有选择地保存长期信息，从而缓解 RNN 的长期依赖问题。

---

#### 模型演进主线

```text
N-gram
有限上下文 + 统计频率
↓
NNLM
有限上下文 + Embedding + 神经网络
↓
RNN
Hidden State 累积历史信息
↓
LSTM
门控机制改善长期记忆
↓
Transformer
Attention + 并行计算
```

---

## 3.1.2 Transformer 架构解析

### 1. Transformer 为什么出现

RNN / LSTM 必须顺序处理：

```text
Token 1
↓
Token 2
↓
Token 3
↓
...
```

后一步必须等待前一步完成，因此难以大规模并行。

Transformer 抛弃循环结构，核心依赖注意力机制（Attention），从而允许训练阶段对序列进行并行计算。

---

### 2. 原始 Transformer：Encoder-Decoder

最初 Transformer 主要用于机器翻译。

整体结构：

```text
Source Input
↓
Encoder
完整理解输入
↓
形成每个 Token 的上下文表示
↓
────────────────────
Decoder
参考：
① 自己已经生成的 Target 前文
② Encoder 对 Source 的理解
↓
逐 Token 生成 Target
```

编码器（Encoder）：

> 负责理解完整输入。

解码器（Decoder）：

> 根据已经生成的输出 + Encoder 提供的信息，预测下一个 Token。

---

### 3. Token 进入 Transformer

完整顺序：

```text
Token
↓
Embedding
获得语义表示
↓
+ Positional Encoding
加入位置信息
↓
得到“语义 + 位置”的 Token 表示
↓
通过权重矩阵
↓
生成 Q / K / V
```

因此：

> Position Encoding 不是和 Q/K/V 并列的第四种特征。

而是先：

```text
Embedding + Position
```

再基于这个表示生成 Q/K/V。

---

## 4. Self-Attention 自注意力

自注意力（Self-Attention）让一个 Token 可以根据上下文中的其他 Token 更新自己的表示。

每个 Token 会生成：

- Query（**Q**，查询）：我现在需要寻找什么信息
- Key（**K**，键）：我这里有什么信息可供匹配
- Value（**V**，值）：如果你关注我，我真正提供什么内容

计算过程：

```text
当前 Token 的 Q
↓
与所有 Token 的 K 做点积
↓
得到 Attention Score
↓
Scaling：除以 √d_k
↓
Softmax
↓
得到 Attention Weight
↓
对所有 V 加权求和
↓
得到融合上下文信息后的新表示
```

其中：

- `d_k`：Key 向量维度
- Scaling 的目的：避免高维点积数值过大，导致 Softmax 过于尖锐、训练不稳定

核心理解：

> Q × K 决定“应该关注谁”。

> V 决定“真正从对方那里拿什么信息”。

例如：

```text
我 喜欢 北京
```

计算“喜欢”的新表示：

```text
Q(喜欢)
↓
分别与：
K(我)
K(喜欢)
K(北京)
计算相关性
↓
得到三个 Attention Weight
↓
对：
V(我)
V(喜欢)
V(北京)
加权求和
↓
得到“我喜欢北京”语境下的“喜欢”
```

---

## 5. Multi-Head Attention 多头注意力

多头注意力（Multi-Head Attention）让模型同时从多个表示子空间分析 Token 之间的关系。

更准确的理解：

```text
Token 表示
↓
可学习的 W_Q / W_K / W_V 投影
↓
生成 Q / K / V
↓
按多个 Head 划分成不同子空间
↓
Head 1 独立做 Attention
Head 2 独立做 Attention
Head 3 独立做 Attention
...
↓
Concatenate
↓
Linear Transformation
↓
最终表示
```

也可以等价理解为：

> 每个 Head 拥有自己对应的一部分 Q/K/V 投影参数，因此可以学习不同类型的关系。

不同 Head 可能分别关注：

- 指代关系
- 语法关系
- 主谓关系
- 时态关系
- 长距离依赖

---

## 6. FFN 前馈神经网络

前馈神经网络（Feed-Forward Network, **FFN**）位于 Attention 之后。

核心：

> Attention 负责 Token 之间的信息交流；FFN 负责对单个 Token 当前的表示进一步加工。

流程：

```text
Attention 后的 Token 新表示
↓
第一层线性变换：升维
↓
ReLU 等非线性激活
↓
第二层线性变换：降维
↓
得到进一步加工的新表示
```

通常：

```text
d_model
↓
d_ff
↓
d_model
```

例如 `d_ff` 可能约为 `4 × d_model`。

注意：

> FFN 会独立作用于每一个 Token，但所有 Token 共用同一套 FFN 参数。

---

## 7. Residual Connection + LayerNorm

### Residual Connection 残差连接

Residual Add：

```text
当前模块输入 x
+
Sublayer(x)
↓
新表示
```

核心：

> 不让新模块完全覆盖旧信息，而是保留旧表示并叠加本层学习到的新信息。

多层 Transformer 中：

```text
旧表示
↓
Attention 学到新信息
↓
Add：旧信息 + 新信息
↓
下一层继续加工
```

因此 Add 可以理解为：

> 帮助 Token 在多层网络中持续保留、累积上下文加工结果。

但 Add 本身不是 Context。

---

### Layer Normalization 层归一化

LayerNorm：

> 对单个 Token 当前整个特征向量做归一化，使数值分布稳定。

不是把整个句子的多个 Token 一起归一化。

---

### 一个基础 Transformer Block

```text
Token 表示
↓
Attention
↓
Add & Norm
↓
FFN
↓
Add & Norm
↓
进入下一层
```

真实 Transformer 会把这一套 Block 重复 N 层：

```text
Block 1
↓
Block 2
↓
Block 3
↓
...
↓
Block N
```

因此所谓“多层 Transformer”不是额外模块，而是：

> Attention + FFN + Add&Norm 这一套结构重复堆叠。

---

## 8. Positional Encoding 位置编码

Self-Attention 本身只计算 Token 之间的关系，不天然知道 Token 顺序。

例如：

```text
agent learns
```

和：

```text
learns agent
```

如果没有位置信息，Attention 很难区分顺序。

因此需要位置编码（Positional Encoding, **PE**）。

流程：

```text
Token Embedding
+
Positional Encoding
↓
包含“语义 + 位置”的表示
```

PE 也是一个 `d_model` 维的向量。

原始 Transformer 使用 sin / cos 函数生成不同位置、不同维度上的数字。

- `pos`：Token 在序列中的位置
- `i`：位置向量的维度索引
- `d_model`：Embedding / Model Representation 的维度

重点：

> 对于同一个位置，pos 固定；不同位置有不同位置编码。

---

# 9. Attention 的不同类型

几个术语不是互斥的，而是在描述不同维度。

### 信息来自哪里

```text
Self-Attention
Q / K / V 来自同一序列
```

```text
Cross-Attention
Q 来自一个序列
K / V 来自另一个序列
```

### 能看到哪些位置

```text
Full / Bidirectional Attention
可以看到整个输入序列
```

```text
Masked / Causal Attention
当前位置只能看到自己和左侧过去 Token
不能看到未来 Token
```

### 同时从几个角度看

```text
Single-Head
只有一个 Attention Head
```

```text
Multi-Head
多个 Head 并行计算 Attention
```

所以完全可以组合：

```text
Multi-Head Self-Attention
Masked Multi-Head Self-Attention
Multi-Head Cross-Attention
```

一句话：

> Self / Cross：看谁

> Masked / Causal：能看多少

> Multi-Head：从几个角度看

---

## 10. 原始 Transformer 的 Decoder

原始 Decoder 每层主要有三个模块：

```text
Target 已生成前文
↓
Masked / Causal Self-Attention
↓
Add & Norm
↓
Cross-Attention
↓
Add & Norm
↓
FFN
↓
Add & Norm
```

### 第一步：Masked Self-Attention

```text
Decoder 自己已有的 Target 前文
↓
Q / K / V 都来自 Decoder
↓
只能看当前位置及左侧历史
```

作用：

> 理解“我自己目前已经生成了什么”。

---

### 第二步：Cross-Attention

```text
Decoder 当前状态
↓
生成 Q

Encoder 对 Source 的完整理解
↓
提供 K / V

Decoder Q
×
Encoder K
↓
Attention
↓
读取对应 Encoder V
```

因此：

> Cross-Attention = Decoder 根据当前需要，去查询 Encoder 的 Source Representation。

重点：

```text
Q：来自 Decoder
K / V：来自 Encoder
```

---

### 第三步：预测 Token

经过多层 Decoder Block 后：

```text
最终 Decoder Representation
↓
Linear
↓
Softmax
↓
整个 Vocabulary 上的 Token 概率分布
↓
选择 / 采样下一个 Token
```

然后：

```text
新 Token 加入 Target 前文
↓
进入下一轮
```

---

# 3.1.3 Decoder-Only 架构

GPT（Generative Pre-trained Transformer）采用 Decoder-Only 架构。

Decoder-Only：

> 去掉独立 Encoder 和 Cross-Attention，只保留 Decoder 的核心结构。

---

## 1. Decoder-Only 完整结构

```text
完整 Prompt + 已生成内容
↓
Tokenizer
↓
Token IDs
↓
Embedding + Position
↓
重复 N 层 Decoder Block：

Causal / Masked Self-Attention
↓
Add & Norm
↓
FFN
↓
Add & Norm

↓
Linear
↓
Softmax
↓
下一个 Token 的概率分布
↓
Sampling
↓
生成 Token
↓
加入 Context
↓
继续下一轮
```

注意：

> Causal Self-Attention 是 Decoder-Only Transformer 内部的模块，不是 Transformer 之后另外再做的一步。

---

## 2. 为什么没有 Encoder 也能理解 Prompt

关键：

> Decoder-only 看不到“未来输出”，但它可以看到完整 Prompt。

例如：

```text
我 喜欢 北京
```

对于最后一个位置“北京”：

```text
北京可以看到：
我
喜欢
北京
```

因此经过多层 Causal Self-Attention 后：

```text
“北京”的当前 Hidden Representation
↓
已经融合此前 Context 的相关信息
↓
用于预测下一个 Token
```

生成：

```text
我 喜欢 北京 I
```

下一轮：

```text
I 可以看到：
我
喜欢
北京
I
↓
预测 like
```

继续：

```text
我 喜欢 北京 I like
↓
预测 Beijing
```

因此 Decoder-only 把：

```text
理解输入
+
生成输出
```

统一成：

> 基于所有已有 Context，预测下一个 Token。

---

## 3. Encoder-Decoder vs Decoder-Only

### Encoder-Decoder

```text
完整 Source
↓
Encoder
双向理解完整输入
↓
形成独立 Source Representation
↓
Decoder：
Masked Self-Attention
+
Cross-Attention
↓
预测 Target 下一个 Token
```

预测依据：

> 已生成的 Target 历史 + 独立的双向 Source 表示。

---

### Decoder-Only

```text
Prompt + 已生成输出
↓
全部放在同一条序列
↓
Causal Self-Attention
↓
最后位置融合所有已有 Context
↓
预测下一个 Token
```

预测依据：

> 同一序列里所有已经出现的 Context。

---

## 4. Autoregressive 自回归

Decoder-Only 的生成模式称为自回归（Autoregressive）。

```text
已有文本
↓
预测下一个 Token
↓
Token 加到原文本末尾
↓
新的已有文本
↓
继续预测
```

---

## 5. 训练阶段 vs 生成阶段

### 训练阶段

完整文本可以一次性输入：

```text
完整训练序列
↓
计算 Attention Score
↓
应用 Causal Mask
↓
未来位置被屏蔽
↓
多个位置可以并行训练
```

Causal Mask：

> 在 Softmax 前，将未来 Token 的 Attention Score 设为极大的负数，使 Softmax 后权重约等于 0。

---

### 生成阶段

未来 Token 本来就不存在：

```text
已有 Context
↓
预测一个 Token
↓
加入 Context
↓
再预测
```

因此训练和生成都遵循：

> 当前 Token 只能依赖自己及左侧历史信息。

---

# 3.2 与大语言模型交互

## 3.2.1 Prompt Engineering 提示工程

提示工程（Prompt Engineering）：

> 通过设计 Prompt，引导模型输出更符合需求的结果。

---

## 1. Temperature

Temperature 控制生成的随机性。

```text
Temperature ↓
↓
概率分布更陡峭
↓
高概率 Token 更占优势
↓
结果更稳定、确定
```

```text
Temperature ↑
↓
概率分布更平坦
↓
低概率 Token 获得更多机会
↓
结果更多样、发散
```

大致应用：

- 事实 / 数学 / 代码：低 Temperature
- 日常对话：中等 Temperature
- 创意写作 / Brainstorm：高 Temperature

---

## 2. Top-k

```text
全部 Token 概率
↓
从高到低排序
↓
只保留前 k 个
↓
重新归一化
↓
从 k 个候选中采样
```

当：

```text
k = 1
```

只剩概率最高的 Token，相当于 Greedy Sampling（贪心采样）。

---

## 3. Top-p

Top-p 又称 Nucleus Sampling（核采样）。

```text
Token 按概率从高到低排序
↓
逐个累计概率
↓
直到累计概率 ≥ p
↓
这些 Token 构成候选集合
↓
重新归一化
↓
采样
```

Top-p 的候选数量不是固定的，会根据概率分布动态变化。

---

## 4. Temperature + Top-k + Top-p

教材给出的组合逻辑：

```text
Temperature
调整整体概率分布
↓
Top-k
先限制候选数量
↓
重新归一化
↓
Top-p
在剩余候选中按累计概率继续过滤
↓
最终候选集
↓
Sampling
```

通常 Top-k 和 Top-p 二选一即可。

---

## 5. Zero-shot / One-shot / Few-shot

根据 Prompt 中提供示例（Exemplar）的数量：

### Zero-shot Prompting

零样本提示：

> 不提供任何示例，直接给任务指令。

### One-shot Prompting

单样本提示：

> 提供 1 个完整输入输出示例。

### Few-shot Prompting

少样本提示：

> 提供多个示例，让模型理解任务格式、边界与规律。

---

## 6. In-context Example

上下文示例（In-context Example）：

> 在当前 Prompt 中提供输入输出示例，帮助模型临时理解任务。

因此：

```text
In-context Example
= “给示例”这种方法

Zero / One / Few-shot
= “给几个示例”
```

---

## 7. Instruction Tuning

指令调优（Instruction Tuning）：

```text
预训练模型
↓
使用大量：
“Instruction → Response”
数据继续训练
↓
模型更会理解并遵循人类指令
```

因此现代 Chat Model 可以直接接受：

```text
请帮我……
请总结……
请翻译……
```

而早期纯文本补全模型往往需要更多示例才能理解任务。

---

## 8. Role-playing

角色扮演（Role-playing）：

> 在 Prompt 中设定模型的角色、语气、知识背景和任务边界。

---

## 9. Chain-of-Thought

思维链（Chain-of-Thought, **CoT**）：

> 引导模型把复杂任务拆成多个推理步骤。

适合：

- 多步骤计算
- 逻辑推理
- 复杂规划

---

# 3.2.2 Tokenization 文本分词

分词（Tokenization）：

```text
自然语言
↓
Tokenizer
↓
Token
↓
Token ID
↓
模型
```

分词器（Tokenizer）：

> 定义把文本切成 Token，并将 Token 映射为数字 ID 的规则。

---

## 1. 四个容易混的概念

### Corpus

语料库（Corpus）：

> 用来训练 Tokenizer / 模型的大量文本数据。

### Vocabulary

词表（Vocabulary）：

> Tokenizer 最终允许使用的 Token 集合。

### Prompt

> 用户本次提交给模型的输入。

### Model Parameters

模型参数：

> 模型经过训练后学到的权重，语言规律主要压缩在参数中。

因此：

```text
Corpus
↓
训练
↓
Model Parameters
```

Corpus 本身不等于模型的“记忆”。

---

## 2. 为什么不能直接按完整单词切

按词分词（Word-based）：

问题：

- Vocabulary 太大
- 容易出现未登录词
- 低频词难学习
- 词形变化之间不容易共享信息

未登录词：

Out-Of-Vocabulary（**OOV**）

---

## 3. 为什么不能全部按字符切

按字符分词（Character-based）：

优点：

- Vocabulary 小
- 几乎不存在 OOV

缺点：

- 单字符语义弱
- 序列会非常长
- 学习效率低

因此现代 LLM 普遍采用：

Subword Tokenization（子词分词）。

---

# 4. BPE

字节对编码（Byte-Pair Encoding, **BPE**）是一种常见的 Subword Tokenization 算法。

训练逻辑：

```text
Corpus
↓
初始化为基本字符
↓
统计所有相邻 Token Pair
↓
寻找最高频 Pair
↓
合并成新 Token
↓
加入 Vocabulary
↓
重新统计所有 Pair
↓
重复
↓
Vocabulary 达到预设大小
```

关键：

> 新生成的 Token 会继续参与下一轮相邻 Pair 统计。

教材案例中如果多个 Pair 并列最高频，并没有给出明确 Tie-breaking Rule；实际实现通常会使用固定规则保证结果可复现。

---

## 5. WordPiece

WordPiece 与 BPE 类似。

区别：

```text
BPE
→ 优先合并最高频 Pair
```

```text
WordPiece
→ 优先选择能更好提升整体语料语言模型概率的 Pair
```

---

## 6. SentencePiece

SentencePiece 是一种语言无关的分词工具。

特点：

> 把空格本身也视为普通字符并显式编码。

因此：

```text
原始文本
↓
空格信息不会丢失
↓
Token Sequence
↓
Decode
↓
可恢复原始文本
```

这使 Tokenization / Detokenization 更容易做到可逆，同时不依赖某种语言必须用空格分词。

---

# 3.2.3 本地运行开源模型

在线 API：

```text
Prompt
↓
API
↓
第三方服务器上的 LLM
↓
Response
```

本地开源模型：

```text
选择 Model
↓
下载 Model Weights
+
下载匹配的 Tokenizer
↓
Prompt
↓
Tokenizer
↓
Token IDs
↓
本地 Model.generate()
↓
New Token IDs
↓
Decode
↓
文本
```

教材中的主要类：

- `AutoTokenizer`
- `AutoModelForCausalLM`

Causal LM：

Causal Language Model，即因果语言模型 / 自回归语言模型。

本地运行优点：

- 数据隐私
- 可离线
- 自主控制程度高
- 可定制

代价：

- GPU / 内存要求
- 环境配置
- 运维成本

---

# 3.2.4 模型选择

模型选择不是：

> 越大越好。

而是 Trade-off（权衡）。

主要考虑：

```text
Performance / Capability
性能与能力
↓
Cost
成本
↓
Latency
延迟
↓
Context Window
上下文窗口
↓
Deployment
部署方式
↓
Ecosystem / Toolchain
生态与工具链
↓
Fine-tuning / Customization
可微调性与定制
↓
Safety / Ethics
安全性与伦理
```

---

## 闭源模型 vs 开源模型

### Closed-source Model

闭源模型：

```text
API 调用
↓
上手快
能力通常较强
无需自己维护服务器
↓
但：
按 Token 付费
数据发送第三方
自主控制有限
```

### Open-source Model

开源模型：

```text
本地部署
↓
隐私更强
控制权更高
可深度定制
↓
但：
需要硬件
需要部署与运维
```

---

# 3.3 Scaling Laws 与模型局限

## 3.3.1 Scaling Laws

缩放法则（Scaling Laws）描述：

> 模型性能与参数量、训练数据量、计算资源之间存在相对可预测的关系。

三个核心变量：

```text
Model Size
≈ Parameters
模型参数量
```

```text
Data Size
训练 Token 数量
```

```text
Compute
训练计算资源 / 计算预算
```

性能通常用 Loss（损失）衡量。

---

## Chinchilla Law

Chinchilla 定律指出：

> 在给定 Compute Budget 下，Parameters 和 Training Data 之间存在更优配比。

因此：

> 不是单纯把参数做得越大越好。

数据量不足时，更大的模型也未必更强。

---

## Emergent Abilities

能力涌现（Emergent Abilities）：

> 当模型规模达到一定水平后，一些在小模型中很弱或不存在的能力可能明显出现。

例如教材提到：

- Chain-of-Thought
- Instruction Following
- Multi-step Reasoning
- Code Generation

---

# 3.3.2 Hallucination

模型幻觉（Hallucination）：

> 模型生成看似合理，但与事实、输入或上下文不一致的信息。

常见类型：

### Factual Hallucination

事实性幻觉：

> 与现实世界事实不符。

### Faithfulness Hallucination

忠实性幻觉：

> 摘要、翻译等结果没有忠实反映 Source。

### Intrinsic Hallucination

内在幻觉：

> 输出与输入提供的信息直接矛盾。

---

## 幻觉为什么发生

主要原因：

```text
训练数据可能有错误 / 矛盾
+
LLM 本质是预测下一个 Token
而不是事实数据库
+
复杂推理链条可能出错
↓
生成“看似合理但实际错误”的信息
```

在 Agent 中风险更大：

```text
某一步产生错误
↓
错误进入 Observation / Context
↓
下一轮继续基于错误信息推理
↓
错误被多轮放大
```

此外还有：

- Knowledge Staleness：知识时效性不足
- Bias：训练数据偏见

---

## 如何缓解 Hallucination

### 数据层

- 高质量数据
- 数据清洗
- 人类反馈强化学习

RLHF：

Reinforcement Learning from Human Feedback  
基于人类反馈的强化学习

### 推理 / 产品层

```text
RAG
+
多步验证
+
Tool Calling
+
Search Engine
+
Calculator
+
Code Interpreter
+
必要时 Human-in-the-loop
```

RAG：

Retrieval-Augmented Generation  
检索增强生成

流程：

```text
用户问题
↓
外部知识库检索
↓
找到相关资料
↓
资料作为 Context 提供给 LLM
↓
基于资料生成回答
```

---

# 第三章完整主线

```text
自然语言 Prompt
↓
Tokenizer
↓
Tokens / Token IDs
↓
Embedding + Position
↓
Decoder-only Transformer

重复 N 层：
Causal Self-Attention
↓
Add & Norm
↓
FFN
↓
Add & Norm

↓
Linear + Softmax
↓
下一个 Token 的概率分布
↓
Sampling
Temperature / Top-k / Top-p
↓
生成 Token
↓
加入 Context
↓
继续生成
↓
Decode
↓
最终文本
```

---

# AI 产品视角：第三章真正要记住什么

1. LLM 本质上是概率生成模型，不是事实数据库。
2. Decoder-only 的核心是基于已有 Context 自回归预测下一个 Token。
3. Attention 负责 Token 之间的信息交互，FFN 负责单个 Token 内部的信息加工。
4. Prompt、Context 和 Sampling Parameters 都会影响输出。
5. Token 数会影响 Context Window、API 成本和模型可处理的信息量。
6. 模型选型必须在能力、成本、延迟、隐私、部署和生态之间做 Trade-off。
7. Hallucination 是模型固有风险之一，实际 AI 产品需要通过 RAG、Tool、Verification、Human-in-the-loop 等机制提升可靠性。
8. 构建 Agent 时不能只问“模型够不够强”，还要问：
   - 模型需要什么 Context？
   - 哪些事实应该通过 Tool 获取？
   - 哪些输出必须验证？
   - 哪些场景需要人工兜底？

---

# Chapter 3 术语速查表

| 缩写 / 英文 | 中文 | 核心含义 |
|---|---|---|
| LM | Language Model，语言模型 | 建模语言序列概率 |
| LLM | Large Language Model，大语言模型 | 大规模参数的语言模型 |
| NLP | Natural Language Processing，自然语言处理 | 计算机处理自然语言 |
| MLE | Maximum Likelihood Estimation，最大似然估计 | 用观察频率估计概率 |
| NNLM | Neural Network Language Model，神经网络语言模型 | 用神经网络预测语言概率 |
| Embedding | 词嵌入 / 向量表示 | 将 Token 变成连续向量 |
| RNN | Recurrent Neural Network，循环神经网络 | 用 Hidden State 保存历史 |
| LSTM | Long Short-Term Memory，长短期记忆网络 | 用门控机制增强长期记忆 |
| Hidden State | 隐藏状态 | RNN 中持续传递的历史表示 |
| Cell State | 细胞状态 | LSTM 中长期信息通路 |
| Attention | 注意力机制 | 判断哪些 Token 信息更重要 |
| Self-Attention | 自注意力 | Q/K/V 来自同一序列 |
| Cross-Attention | 交叉注意力 | Q 与 K/V 来自不同序列 |
| Masked Self-Attention | 掩码自注意力 | 屏蔽部分位置 |
| Causal Attention | 因果注意力 | 只能关注当前及历史 Token |
| Q | Query，查询 | 当前想寻找什么信息 |
| K | Key，键 | 可供匹配的信息标签 |
| V | Value，值 | 真正被提取的信息 |
| FFN | Feed-Forward Network，前馈网络 | 对 Token 表示进一步加工 |
| PE | Positional Encoding，位置编码 | 给 Token 加入顺序信息 |
| d_model | Model Dimension | Token / 模型表示维度 |
| d_k | Key Dimension | K 向量维度 |
| d_ff | Feed-Forward Dimension | FFN 中间层维度 |
| GPT | Generative Pre-trained Transformer，生成式预训练 Transformer | 典型 Decoder-only 模型 |
| Autoregressive | 自回归 | 用已有内容逐步生成未来内容 |
| Prompt | 提示 / 提示词 | 给模型的当前输入 |
| Prompt Engineering | 提示工程 | 设计 Prompt 引导输出 |
| Zero-shot | 零样本提示 | 不提供示例 |
| One-shot | 单样本提示 | 提供 1 个示例 |
| Few-shot | 少样本提示 | 提供多个示例 |
| Exemplar | 示例 | Prompt 中给模型参考的案例 |
| In-context Example | 上下文示例 | 在当前上下文中提供输入输出示例 |
| Instruction Tuning | 指令调优 | 用指令-回答数据继续训练 |
| CoT | Chain-of-Thought，思维链 | 多步骤推理方法 |
| Tokenization | 分词 | 将文本转换为 Token |
| Tokenizer | 分词器 | 执行分词规则的组件 |
| Token | 词元 | 模型处理文本的基本单位 |
| Token ID | 词元编号 | Token 对应的数字 ID |
| Vocabulary | 词表 | Tokenizer 可使用的 Token 集合 |
| Corpus | 语料库 | 用于训练的大量文本 |
| OOV | Out-Of-Vocabulary，未登录词 | 不存在于词表中的词 |
| BPE | Byte-Pair Encoding，字节对编码 | 高频相邻 Token 逐步合并 |
| WordPiece | WordPiece 子词算法 | 根据语言模型概率选择合并 |
| SentencePiece | SentencePiece 分词工具 | 语言无关、保留空格信息 |
| API | Application Programming Interface，应用程序接口 | 程序之间调用服务的接口 |
| GPU | Graphics Processing Unit，图形处理器 | 常用于加速模型训练 / 推理 |
| Context | 上下文 | 模型当前能够看到的信息 |
| Context Window | 上下文窗口 | 单次可处理的最大 Token 数 |
| Temperature | 温度 | 控制输出随机性 |
| Top-k | Top-k Sampling | 只保留概率最高的 k 个 Token |
| Top-p | Top-p / Nucleus Sampling，核采样 | 保留累计概率达到 p 的 Token |
| Sampling | 采样 | 从概率分布选择输出 Token |
| Parameters | 参数 | 模型训练后学到的权重 |
| Compute | 计算资源 / 计算量 | 模型训练所需计算预算 |
| Scaling Laws | 缩放法则 | 参数、数据、算力与性能的规律 |
| Loss | 损失 | 衡量模型预测误差 |
| Hallucination | 幻觉 | 看似合理但事实错误的生成 |
| RAG | Retrieval-Augmented Generation，检索增强生成 | 检索外部知识再生成 |
| RLHF | Reinforcement Learning from Human Feedback，人类反馈强化学习 | 用人类反馈优化模型 |
| MoE | Mixture of Experts，混合专家模型 | 每次只激活部分专家网络 |
| Human-in-the-loop | 人在回路 | 关键环节由人工审核 / 决策 |
