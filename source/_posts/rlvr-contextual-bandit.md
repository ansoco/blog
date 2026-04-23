---
title: "从 Scorer 到 Contextual Bandit：拆解 RLVR 的概念体系与本质"
date: 2026-04-17 19:00:00
mathjax: true
tags:
  - RL
  - RLHF
  - RLVR
  - LLM
  - Agentic Training
categories:
  - AI Research
description: "从 RL 训练中常见的 Scorer / Rubric / Reward / Checker / Verifier 等概念出发，逐步厘清它们之间的关系，最终回答一个核心问题：RLVR 的本质是什么？"
top_img: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%)
cover: false
---

## 一、基础概念：Scorer / Rubric / Reward / Checker / Verifier

在 LLM 评估与 RLHF 训练中，有一组高频出现但容易混淆的术语。它们的层级关系如下：

```
Scorer（泛称：给输出打分的东西）
├── 二值 Scorer
│   ├── Checker（规则/程序判对错，偏工程用语）
│   └── Verifier（≈Checker，偏学术用语，也可以是模型）
├── 连续 Scorer
│   ├── GRM（Generative Reward Model，生成式打分）
│   └── QRM（Quality Reward Model，质量打分）
└── Rubric 是 Scorer 依据的标准，不是 Scorer 本身

Reward = f(各种 Scorer 的输出加权合成)
```

| 概念 | 本质 | 输出 |
|:---:|:---|:---:|
| **Scorer** | 泛称，任何给输出打分的东西 | 连续分或二值 |
| **Checker / Verifier** | 二值判定器，语义上几乎等价 | 0 / 1 |
| **GRM** | 一种 Scorer（生成式，基于 Rubrics） | 连续分 |
| **QRM** | 一种 Scorer（质量维度） | 连续分 |
| **Reward** | 最终信号，由上述 Scorer 合成 | 标量 |
| **Rubric** | 评分标准本身，不是执行者 | — |

其中 **Rubric** 是纯描述性的评分量规，定义"好坏"的标准（类似考试评分细则）。Scorer 拿着 Rubric 去打分。Reward 是最终喂给 RL trainer 的标量信号，由多种 Scorer 加权合成。

Checker 和 Verifier 在多数文献和工程实践中**可互换**，都是"判对错"的二值判定器。如果硬要区分：Verifier 更偏学术用语（*Let's Verify Step by Step*, RLVR 等），Checker 更偏工程用语（unit test checker, format checker）。

---

## 二、GRM 与 QRM：为什么需要两个？

### 定义

- **GRM（Generative Reward Model）**：用生成式模型逐条判定 Rubrics 覆盖度，输出连续分数。
- **QRM（Quality Reward Model）**：综合评估回答的整体质量（准确性、完整性、可读性、安全性等），与 GT 对比后给出连续质量分。

### 在训练流程中的组合

```
Reward 信号来源
├── Verifier / Checker（规则验证）
│   └── 确定性答案：数学 / 代码测试用例 → 0 或 1
├── GRM → Rubrics 覆盖度 → 连续分
├── QRM → 整体质量 → 连续分
└── 加权合成 → final reward
```

最终 reward 的合成公式形如：

$$
reward = w_{rubrics} \times score_{rubrics} + w_{qrm} \times score_{qrm}
$$

### 为什么同时需要两者

核心动机来自实际训练中的观察：QRM 单独使用时，在部分任务上偏序一致率不足。特别是在"模型回答比 GT 更好"的 case 上，QRM 难以给出准确判断。

- **GRM + Rubrics 解决客观性问题**：结构化判据，逐条判定要点是否命中，减少风格偏置。
- **QRM 补充主观质量维度**：捕捉整体可读性、用户体验等难以用 checklist 穷举的维度。

{% note info %}
**GRM 是结构化锚点，QRM 是质量兜底。两者互补。**
{% endnote %}

---

## 三、Agentic 训练（Long Trajectory）中的体现

当从单轮 RLHF 走向 agentic / long-trajectory 训练时，核心矛盾是：**trajectory 很长、action 很多、reward 很稀疏**。每一个概念都面临新的挑战。

### ORM → PRM 的演进

| 类型 | 全称 | 粒度 | 角色 |
|:---:|:---|:---:|:---|
| **ORM** | Outcome Reward Model | trajectory 级 | Verifier/Checker 的模型化版本——只看最终结果 |
| **PRM** | Process Reward Model | step 级 | Scorer 的 agentic 版本——对每一步打分 |
| **AgentPRM** | Agent Process Reward Model | step 级 + action-aware | PRM 专为 agent action 序列设计的变体 |

在 agent trajectory $[s_0 \rightarrow a_1 \rightarrow s_1 \rightarrow a_2 \rightarrow s_2 \rightarrow ... \rightarrow a_n \rightarrow s_n]$ 中：

- **Verifier/Checker**：只看 $s_n$（最终结果对不对），给出 outcome reward（0/1）。
- **PRM / AgentPRM**（step-level Scorer）：看每个 $(s_i, a_{i+1})$，判断这步 action 好不好。
- **Rubric**：定义每步 action 的好坏标准——工具调用参数正确？工具选对了？检索 query 是否相关？

```
step_reward[i] = PRM(s_i, a_{i+1})   ← step-level
outcome_reward = Verifier(s_n)     ← trajectory-level
total_reward = f(step_rewards, outcome_reward)
```

{% note warning %}
引入 PRM 的核心意义是解决 **credit assignment** 问题：如果只有 trajectory 级的 outcome reward，50 步的 agent trajectory 中，模型无法分辨是哪一步导致了成功或失败。
{% endnote %}

---

## 四、RLVR 的本质是 Contextual Bandit

### "RLVR 本质上是多臂老虎机"——这个说法的来源

在标准 RLVR（如 DeepSeek-R1 的数学训练）中：

1. 模型一次性生成完整 response（一条 trajectory）。
2. Verifier 只看最终答案，给 0 或 1。
3. 没有中间 reward，没有状态转移的反馈。

用 GRPO 来看更直观：对同一个 prompt 采样 N 条 response → verifier 打分 → 好的 response 整体 upweight，差的 downweight。**每条完整 response 是一个 arm，reward 是 binary，策略是调整各 arm 的采样概率。**

### 严格来说是 Contextual Bandit

| 要素 | 经典 MAB | Contextual Bandit | RLVR |
|:---:|:---|:---|:---|
| Context | 无 | 每轮有不同的 x | prompt |
| Arm/Action | K 个固定 arm | 给定 context 选 action | 给定 prompt 采样 response |
| 策略 | 全局一个分布 $\pi(a)$ | 条件策略 $\pi(a \mid x)$ | $\pi(response \mid prompt)$ = LLM |
| 目标 | 最大化累计 reward | 最大化 $E[r(x,a)]$ | 最大化 $E[verifier(prompt, response)]$ |

三个条件确认 RLVR 是 contextual bandit：

1. **有 context**（prompt）→ 不是经典 MAB。
2. **Action 不影响下一个 context**（prompt 来自数据集，不因 response 改变）→ 不是 MDP。
3. **单步决策**，整条 response 作为一个不可分的 action，拿到一个 scalar reward 就结束 → bandit。

### "Token 级梯度存在"不等于"Token 级 Credit Assignment"

RLVR 中 REINFORCE/GRPO 的梯度公式展开到了每个 token：

$$
\nabla J \approx \sum_i \nabla \log \pi(a_i \mid s_{<i}) \cdot R(\tau)
$$

但 **$R(\tau)$ 对所有 token 完全相同**。token 37 做了关键推理，token 12 输出了 filler word，两者乘的 reward 信号一模一样。梯度在形式上是 token 级的，但**信息量是 trajectory 级的**。

{% note info %}
Token 级梯度是 autoregressive 参数化的副产品，不是 token 级的 credit assignment。只要 reward 信号对所有 token 是同一个标量广播，优化的信息结构就是 bandit。
{% endnote %}

### 泛化靠参数共享

Contextual bandit 的核心难题是**泛化**：模型不可能对每个 prompt 都采样足够多次，必须从已见 prompt 的 reward 信号泛化到未见 prompt。

泛化的物理载体是**参数共享**：所有 prompt 共享同一组 Transformer 参数 $\theta$，语义相似的 prompt 产生相似的 hidden states，对 prompt A 的策略更新通过 attention / FFN 权重传导到 prompt B。

这与经典 MAB 形成鲜明对比：经典 MAB 每个 arm 有独立的 reward 估计（Q-table），arm A 的经验不影响 arm B。**参数共享 + 表征学习是让 contextual bandit 在无限 context 空间上可行的唯一途径。**

---

## 五、RLVR 超越经典 Contextual Bandit 的部分

RLVR 的问题结构是 contextual bandit，但它的实现方式超越了经典 CB 框架的若干基础假设：

| 维度 | 经典 CB | RLVR 超越之处 |
|:---:|:---|:---|
| **Action 空间** | 有限离散 | 无穷、变长、结构化 token 序列 |
| **Action 内部** | 原子不可分 | autoregressive 序列，梯度可达 token 级 |
| **正则化** | L2 on 参数 | $KL(\pi_\theta \parallel \pi_{ref})$ on 分布 |
| **平稳性** | 假设 i.i.d. | on-policy 导致非平稳 |
| **采样策略** | 单次拉取 | N 路并行 + 组内对比（GRPO） |

经典 bandit 更新的是 arm 的权重，RLVR 更新的是**生成 arm 的"工厂"的参数**。这些超越使得 RL 工具箱在 bandit 问题上的"大材小用"有了合理性。

---

## 六、KL 正则化：参数空间 vs 分布空间

### RLVR 的优化目标

$$
\max_\theta \mathbb{E}\_{x \sim D, y \sim \pi\_\theta(\cdot \mid x)} [ R(x, y) ] - \beta \cdot KL( \pi\_\theta \parallel \pi_{ref} )
$$

KL 项在 autoregressive 模型中分解为逐 token 求和，每个 token 位置都有独立的 KL 惩罚：

- token 与 ref 一致且 response 正确 → **强化**。
- token 偏离 ref 但 response 正确 → **有条件强化**（偏离代价小于 reward 增益时才保留）。
- token 偏离 ref 且 response 错误 → **双重惩罚**，快速拉回。

{% note info %}
KL 实际上承担了部分 credit assignment 的功能——它告诉模型"哪些 token 偏离了正常轨道"，虽然这个信号不是 task-specific 的。
{% endnote %}

### 与经典 CB 正则化的对比

| 维度 | 经典 CB 正则化 | RLVR KL 正则化 |
|:---:|:---|:---|
| **正则化对象** | 参数 $\theta$ 本身（权重向量） | 策略分布 $\pi_\theta$（函数空间） |
| **度量** | L2 范数 $\|\theta\|^2$ | $KL(\pi_\theta \parallel \pi_{ref})$ |
| **参考点** | 原点（$\theta=0$） | SFT reference model |
| **语义** | "参数不要太大" → 防过拟合 | "行为不要偏离太远" → 防能力退化 |
| **动机** | 让学习过程在统计上可行 | 让学习结果在功能上可用 |

核心差异：**L2 约束参数空间到原点的距离，KL 约束策略分布到 reference policy 的距离。** 对 LLM 这种超参数化模型，参数空间的距离和行为变化几乎脱钩。

### 现代 CB 中的分布级正则化

一些现代 CB 工作也用了分布级正则化（IPS clipping、POEM variance penalty 等），但动机不同：

- **CB 的分布级正则化**：解决 off-policy 估计的统计偏差——"别离数据来源太远，否则 importance weight 不可信"。
- **RLVR 的 KL 正则化**：解决 on-policy 训练中的能力退化——"别离好老师太远，否则会忘本"。

RLVR 的 KL 约束还有一个优美的闭式解：

$$
\pi^*(y \mid x) \propto \pi_{ref}(y \mid x) \cdot \exp(R(x,y) / \beta)
$$

最优策略是在 reference policy 基础上按 reward 做指数重加权。这正是 DPO 的理论基础——DPO 直接拟合这个闭式解而跳过了 RL。

---

## 七、总结：从概念到本质

1. **Scorer / Rubric / Reward / Checker / Verifier** 是评估与打分的基础概念体系。Scorer 是泛称，Checker/Verifier 是二值特例，Rubric 是标准，Reward 是合成结果。

2. **GRM 和 QRM** 都是 Scorer 的具体实例，分别解决客观要点覆盖和主观质量评估问题，两者互补。

3. 进入 **agentic / long-trajectory** 训练后，所有概念从 trajectory 级下推到 step 级，核心新增物是 **PRM**（Process Reward Model），解决 credit assignment 问题。

4. 回到 RLVR 本身，它的问题结构是 **contextual bandit**：有 context（prompt），单步决策（整条 response 作为不可分的 action），action 不影响下一个 context。

5. RLVR 超越经典 CB 的部分——无穷结构化 action 空间、KL 正则化、on-policy 非平稳性——正是让 RL 工具箱在 bandit 问题上"大材小用"的合理性所在。

6. **从 bandit 到 MDP 的跨越**发生在引入 PRM 或真实环境交互的那一刻：action 开始影响环境 state，reward 开始因 step 而异，问题才真正成为 RL。

{% note primary %}
*当前 LLM 训练中被称为"RL"的大部分工作，本质上是在用 RL 的算法工具箱求解 bandit 问题；而真正的 RL 问题——agentic training——才刚刚开始。*
{% endnote %}
