---
title: Search-R1及agentic学习
date: 2026-06-21
mathjax: true
tags:
  - agent
  - rl
  - verl
categories:
  - 
description:
---
# Search-R1
## 数学原理
采用PPO：
$$L^{PPO}=L^{Clip}+c_1L^{V}-c_2H(\pi_\theta)$$
- 策略和价值函数异步更新，熵函数鼓励策略探索性

 ```text
                                                                          standard PPO:                                                                                                                                                                                                             
L_actor = E[ min(ratio·A, clip(ratio)·A) ] - β · H(π(·|s))                                                                                                                                          
					↑ pg_loss            ↑ entropy_loss · entropy_coeff                                                                                                                                                 
 ```                                    

- 奖励函数
$$R=R_{out\_ans}+\alpha R_{search\_valid}-\beta R_{search\_invlaid}$$
- 标准Rollout采样
$$\tau=(s_0,a_0,s_1,a_1,\dots,a_{T-1},s_T)$$
其中$$a_t\sim\pi_\theta(a_t\mid s_t),\qquad s_{t+1}=\mathop{env}(s_t,a_t)$$
在此处为$$\pi_\theta(a_t\mid s_t)=P(a_{generate}\mid s_t)+P(a_{search}\mid s_t)$$
$a_{generate}$为常规文本推理，$a_{search}$输出\<search>特殊token触发搜索

> VeRL在Rollout阶段会保存每一步$\pi_\theta(a_t\mid s_t)$，用于PPO更新的重要性采样ratio
---
## 工程细节

- 推理训练解耦独立运行
- 多卡分布式rollout【pytorch distributed】
- 统一数据结构（TITO）
- 采样策略【核采样、温度采样】
Rollout标准流程：
```
样本加载—>多卡批量分片—>模型前向推理—>单步采样—>状态拼接—>终止判断—>轨迹打包—>推送训练序列
```

### 原项目中的缺陷不足

把原生 Search-R1 放在 TITO（Token-In, Token-Out）尺度下衡量，缺陷集中在四个层面：训推一致性、工具协议、工程稳定性、奖励与算法侵入度。

其中**训推一致性**是根本：

训推一致性的破口在 `search_r1/llm_agent/generation.py:_postprocess_responses`。它走的是 decode → 字符串 split → 重新 encode 的闭环：

```python
responses_str = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
responses_str = [resp.split('</search>')[0] + '</search>' if '</search>' in resp ...]
responses = self._batch_tokenize(responses_str)   # ← 重新 encode，TITO 已断
```

BPE 合并不跨边界稳定，把一段 token decode 成字符串、再 split、再 encode 回去，结果与采样时不保证逐 token 一致；observation 拼接 `<information>...</information>` 同样要重新 tokenize，与采样时的 token 边界也可能错位。错位的代价是连锁的：loss mask、IS ratio、KL 都建立在错位的 token 上——训练阶段模型"看到/回传"的 token，与它"采样"时实际产生的 token 不再是同一批，分数偏低、loss 尖刺、function call 漂移这些症状由此产生，但很难追到根因。

**工具协议的僵化**是第二层缺陷。`<search>...</search>` 与 `<information>...</information>` 是写死在 stopping criteria 里的硬编码；单 turn 单 query，多视角检索只能靠多 turn 串联（轨迹被拉长，credit assignment 更难）；没有 `BaseTool` 抽象，换工具（rerank / Google / MCP）需要改 `generation.py`。更深一层的问题是它与 native tool-call 基座（Qwen2.5-Instruct、Llama3.1）的 ChatML 体系互斥：模型若已在 SFT 阶段学过 `<tool_call>`，在此处会被强行带回旧协议，同时学两套互斥规范，function call 大面积错乱几乎是必然结果。

**工程稳定性**体现在四个具体环节。
- retriever 调用是裸 `requests.post`，无重试无限流——retriever 一次抖动 → trajectory 整条报废 → 当 batch advantage 估计被污染。
- `run_llm_loop` 是同步 Python for 循环，batch 内最慢的样本会拖住其他样本进入下一轮，吞吐由最长尾决定。
- vLLM 锁在 `0.6.3` 加 `XFORMERS`——这是 Qwen2-7B 配 flash_attn 已知兼容性问题的产物，但代价是 Qwen3 / Llama3.2 / DeepSeek 等需要新 vLLM 的模型全部上不去。
- 最后是 `max_prompt_length` 的手算公式 `max_start + max_resp*(turns-1) + max_obs*turns`，配错就静默截断 observation，框架不给任何断言。

**奖励颗粒度**构成第四层。终点 EM-only 意味着每一步 tool call 的 step reward 都是 0，错答案与无答案在 `format_score=0` 设定下不可区分；KL 又是 KL-on-reward 形式直接灌进 reward 信号，advantage 容易爆炸。

>整套实现是对 vendored veRL（停在 `version=0.1`）的 invasive patch，跟主线脱钩——任何升级都意味着把全部 patch 重新 rebase 一次。

---

### VeRL官方的优化实现

>四层改造各自对应上面的痛点，主导思想可以一句话概括：把"搜索"从对训练框架的 invasive patch，变成训练框架的标准化插件。

第一层是**采样逻辑**。原生那个手写的同步 `run_llm_loop` 整个被替换为 SGLang multi-turn engine，循环本身、KV cache 跨轮复用、loss mask、turn 计数全部由 SGLang 在 token 边界上接管，配置上只暴露一个 `actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2` 单一开关，TITO 不变量由 backend 保证而不是用户责任。并发模型从"batch 同步 turn"升级为"per-trajectory async + continuous batching"——A 样本在 turn 3、B 样本在 turn 1 可以同时跑，吞吐红利在同硬件、相同 retriever 下能到 2–3×。

第二层是**状态更新**，抽象为 `BaseTool` 接口。工具实现 `verl/tools/search_tool.py` 只暴露四个 async 方法：`create(instance_id, **kwargs)` 在 trajectory 启动时拿 `tools_kwargs`，`execute(instance_id, parameters)` 返回 `(ToolResponse, step_reward, metrics)` 三元组，`calc_reward(instance_id)` 给工具维度的累计反馈，`release(instance_id)` 释放 trajectory 状态。每条 trajectory 用独立 `instance_id` 隔离，失败不会污染其他样本。`tools_kwargs` 在数据预处理阶段（`examples/data_preprocess/preprocess_search_r1_dataset.py`）就被塞进 parquet 的 `extra_info`：

```python
tools_kwargs = {"search": {"create_kwargs":
    {"ground_truth": ground_truth, "question": question, "data_source": data_source_tagged}}}
```

这让工具在每条 trajectory 启动时就拿到完整上下文，从而支持 per-trajectory reward shaping（例如检索结果命中 ground truth 实体就给 step reward）、按数据源分流、训练与验证用不同 topk——这些在原生方案里都得改 trainer 才能做到。

第三层是**特殊 token 处理**。训练阶段模型看到的不再是自定义 `<search>` 标签，而是标准 ChatML 工具调用：

```text
<think>...</think>
<tool_call>{"name": "search", "arguments": {"query_list": ["...", "..."]}}</tool_call>
<tool_response>{"result": "Doc 1 (Title: ...) ..."}</tool_response>
...
<answer>...</answer>
```

这一换有三个量化收益。其一，与 Qwen2.5-Instruct、Llama3.1、DeepSeek 等已 native tool-call 训练过的基座对齐 ChatML 前缀，不再"同时学两套互斥规范"。其二，训练后的 checkpoint 直接能跑在 vLLM、SGLang、TGI 的 OpenAI tools API 上，无需自写 parser。其三，`query_list: array<string>` 让单 turn 可以批量发出多条查询，缩短轨迹长度的同时把单 turn reward signal 也稠密化了——如果原本需要 N turn 才能覆盖的多视角检索，现在可以压在 1 turn 内完成。

第四层是外部**工具调用**的可靠性。`verl/tools/search_tool.py` 里的 `SearchExecutionWorker` 与 `TokenBucketWorker` 是 Ray 单例 token bucket，默认 `rate_limit=120`、`num_workers=120`，用来防止训练把 retriever 自己打崩；`call_search_api` 自带 10 次指数退避（`verl/tools/utils/search_r1_like_utils.py:MAX_RETRIES=10`），覆盖 5xx、`ConnectionError`、`Timeout`，失败时返回字符串 `{"result": "Search error: ..."}` 给模型当 observation——trajectory 不会因此报废，反而让模型有机会学习"工具失败时换查询"的行为。`tool_metrics` 输出 `query_count / status / total_results / api_request_error`，可以直接进 W&B，让你能定量区分"训练慢是因为 25% 检索 timeout"还是"真在算梯度"。算法层面也同步收敛到 GRPO 的现代稳定配置：`use_kl_in_reward=False` + `kl_loss_type=low_var_kl` + `kl_loss_coef=0.001`，把 KL-on-reward 换成 KL-on-loss，advantage 的方差明显更小。

需要诚实指出的是：v0.7.1 也没解决两件事。
其一，奖励颗粒度仍是 session-wise EM——`SearchTool.execute()` 当前实现把 `tool_reward_score` 写死成 `0.0`，`calc_reward` 返回的也只是检索文本而非 scalar；每个 turn 实际上不给奖励，`compute_score` 仅在终点跑一次 EM（`verl/utils/reward_score/search_r1_like_qa_em.py`），再由 GRPO 在 group 内的 5 条 trajectory 之间做归一化、均匀广播到每个 assistant token 的 logprob 上。
其二，灵活性下降——自定义 tag、自定义 loss-mask 在标准 schema 下不容易做。最后一个事实是定位变化：v0.7.1 之后 verl 主线把 `SearchTool` 类整个下掉了，文档要求用户自己继承 `BaseTool` 实现，"开箱即用"已不再成立，`search_r1_like` 被官方主动调整为"接入参考实现"。

---

### [基于qwen的实现改造](https://github.com/Xinyi-0724/Search-R1-Qwen3)

把原生 Search-R1 的算法层原封不动搬到一个更新版的 verl（v0.5.0.dev）上的兼容性升级。逐文件比对的结果是 `llm_agent/generation.py` 与原生逐字节相同（仅尾换行符差异），`<search>` / `<information>` 协议、`run_llm_loop`、loss mask、turn 限制全部继承。真正升级发生在 vendored verl 本身：从 `v0.1` 跳到 `v0.5.0.dev`，带来 vLLM 0.7.0+ 支持、Megatron-core、`verl/experimental` 与 `verl/interactions` 子树、Hydra 配置体系，以及一系列新模型 transformer 适配（`qwen2_vl.py`、`qwen2_5_vl.py`、`kimi_vl.py`、`npu_patch.py`）。

把三方差异并到一张定量表里看最清楚：

| 维度 | 原生 Search-R1 | Search-R1-Qwen3 | verl 官方 |
|---|---|---|---|
| `verl/version/version` | 0.1 | 0.5.0.dev | 0.7.1 |
| `llm_agent/generation.py` | 原版 | 与原生逐字节相同 | 不存在（用 `BaseTool`） |
| 推理后端 | vLLM 0.6.3 + XFORMERS | vLLM 0.7.0+，可选 SGLang | SGLang only |
| Tag 协议 | `<search>` / `<information>` | `<search>` / `<information>` | `<tool_call>` / `<tool_response>` |
| 单 turn 多 query | 不支持 | 不支持 | 支持（`query_list: array<string>`） |
| Loss mask | 手写 `state_masking=true` | 手写 `state_masking=true` | SGLang 引擎接管 |
| Reward 颗粒度 | session-wise EM | session-wise EM | session-wise EM |
| 限流 / 重试 | 无 | 无 | Ray TokenBucket + 10 次指数退避 |
| 配置框架 | 散装 args | Hydra + Ray runtime env | Hydra + Ray runtime env |
| Megatron-core | 无 | 有 | 有 |
| 独立批量评估 | 无 | `main_eval_search_r1.py` | 无（靠 trainer val loop） |

可以一句话概括 Qwen3 版的定位：**算法零改动，工程升级换骨。**
Qwen3-8B、Llama3.2、DeepSeek 这些"新模型 = 新 vLLM = 新 verl"的链路被这一版打通；批量 jsonl 评估则是它在三个版本里独家的能力。工程化补丁本身也很务实——`train_grpo.sh` 用 `N_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)` 自动检测 GPU 数，用 `RAY_TMPDIR=/tmp/ray-$(whoami)` 配 `RAY_ADDRESS=local` 隔离多用户 Ray 冲突，默认 `WANDB_MODE=offline` 保证无 API key 环境也能落本地日志；`merge.sh` 把 FSDP checkpoint 合并回 HuggingFace 格式（部署与评估的必要前置）；`eval_grpo.sh` 配合 `verl/trainer/main_eval_search_r1.py` 把 rollout loop 包成独立批量评估入口，输出含完整 search trace 的 `inference/nq_grpo_search_results.jsonl`；`verl/trainer/main_ppo.py` 改写为 Hydra + Ray runtime env，可读性显著优于原生散装 `argparse`。

但算法零改动的另一面，是原生的 TITO 隐患全部继承——而 Qwen3.5 / Qwen2.5 的 ChatML 体系恰好把这些隐患放大。第一处放大点是 `bos_token = null`。这是 ChatML 设计本身：它依赖 `<|im_start|>` 与 `<|im_end|>` 标界、不需要 BOS；但训练框架若假设 bos 非空，会把前缀错误拼成 `Noneassistant\n`，直接造成训推不一致和分数偏低。这一版没做相应断言，依赖 verl 主线行为，上线前需要自查"渲染出的前缀是 `<|im_start|>assistant\n` 而非 `Noneassistant\n`"。

第二处放大点是 tool_call 协议冲突：原生 `<search>` 与 Qwen 目标格式 `<tool_call>` 是两套规范，若基座已在 SFT 阶段学过 native tool-call，会被强行带回旧格式而同时学两套互斥规范——最稳妥的处理是上线前把数据通过一个中间表示做 N→1 映射统一到目标格式。第三处放大点是 `_postprocess_responses` 里的 decode-split-encode 闭环没有修复，mask 边界与采样 token 边界仍可能错位，IS ratio 与 KL 也跟着错位。第四处是 `</think>` 与 `</search>` 不闭合在原生 reward 里不会被惩罚——TITO 正解是用 reward shaping 给小惩罚，而不是事后塞 token，因为后者直接破坏 TITO 不变量。第五处是 observation 走"文本拼接 → tokenize → 手写 mask"的老路，任何截断或格式抖动都会再次违反"mask 边界 = token 边界"这条铁律。第六处是评估走 `main_eval_search_r1.py`、训练走 trainer 的 rollout loop，两条路径的截断和拼接逻辑没有做对齐断言；建议把 jsonl 评估输出与训练时的 rollout buffer 做一次 token 级 diff 验证。

---

# Agentic RL 通用要点

前面三节反复回到 TITO 这个尺子：原生缺陷的根、官方改造的红利、Qwen 场景下的隐患放大，全都在它上面。这一节把镜头拉开，从 Search-R1 这个具体项目延伸到"做 agentic RL 普遍要注意什么"——既覆盖通用层面的不变量，也针对 Qwen3.5 这套 ChatML 基座补一份专属注意事项。

## 一句话抓核心：TITO 不变量

梯度必须落在模型"真正采样出来"的那批 token 上——不多、不少、不改写。

agentic RL 里几乎所有诡异 bug——分数莫名偏低、loss 尖刺、function call 大面积混乱、熵塌缩——追到底都是同一个根因被破坏：**训练时模型看到/回传的 token，和它推理（采样）时实际产生的 token，不是同一批。**

为什么容易破坏？因为 tokenization 不可逆：把一段 token decode 成字符串、再 encode 回去，结果可能和原来不一样（BPE 合并不跨边界稳定，叠加 JSON 空格、字段顺序、special token 重渲染等自由度）。只要训练循环"每轮把消息列表重新渲染→重新分词"，漂移就会发生，而且静默出错。Search-R1 原生 `_postprocess_responses` 走的就是这条错路——这也是为什么前文反复把它作为反面教材。

正反两条路对比：

```
✅ 正确：采样 token ──► 唯一 buffer（source of truth）──► 直接算 loss
                         （永不 decode 再 encode）

❌ 错误：采样 token ──decode──► 字符串 ──重渲染/改写──► encode ──► 新 token ──► 算 loss
                                                          ↑ 和采样的对不上了
```

## 通用避坑清单

通用避坑清单可以分成静态层、动态层、history rewriting、不闭合标签、RL 稳定性五块。

**静态层是"还没开训就埋好的雷"，属于渲染（render）和分词（encode）阶段。** 第一类是 special token 非 null 的检查：bos/eos 为 null 时前缀会被渲染成 `Noneassistant\n` 而非 `<|im_start|>assistant\n`，正确姿势是训练前断言 bos/eos 等特殊 token 不为 null（null 也要显式处理）。第二类是 chat template 前缀保持：含 tool 消息与不含时前缀必须一致，方法是渲染"含 tool"与"不含 tool"两份对话、断言前者是后者的逐 token 前缀。第三类是 tool_call 格式统一：训练数据混入多种协议（XML/JSON/旧格式）会让模型学乱，正解是统一为单一目标格式、多来源用一个中间表示做 N→1 映射。第四类是 think 标签闭合：模型不爱输出 `</think>` 会导致 thinking 不闭合、重复超长，数据里要保证 think 结构完整、让模型学会输出终止符。

**动态层在采样—训练循环里发生，是 agentic（多轮）特有的。** off-policy 纠偏方面，异步 rollout 让轨迹相对当前策略 off-policy，需要用 importance sampling，且 IS 权重只在"真采样 token"上计算——前提是 prompt/response 的 token 切分与采样时完全一致。逐轮 loss mask 方面，mask 边界 = token 边界；坏动作负向、好动作正向、中性轮跳过（mask=0）；错位等于梯度作用在错误序列上。环境产物一律 mask=0：tool_response、注入的 hint、压缩 summary 都不是策略采样的，可进 prompt 但绝不回传梯度。截断规则训推统一：日志/历史过长要截断时，截断点对齐 token，且训练与采样用同一套 truncate 规则，否则两边看到的 prompt 不同。少碰模板：为兼容性临时改 chat template（再回滚）是反模式——每改一次都在动渲染规则，能在解析/路由层容错就别改模板。

**历史改写（history rewriting）是头号隐形杀手。** 部署期的 agent 常做上下文压缩、丢历史 thinking、注入 hint、子代理摘要，这些都在"事后改写历史"，使得"现在用来算概率的历史 ≠ 当初采样时的历史"。两条铁律必须遵守：训练必须忠实复现部署的改写行为（压缩/丢 think 都要进训练轨迹），否则部署一压缩，模型就进 OOD；改写注入的 token（summary/hint）一律 mask=0、只当 prompt，模型真采样的 token 一律 mask=1，改写只改"后续轮看到的 prompt"，不追溯已发生的梯度。一条经验法则：**"是不是推理过程"和"影不影响训推一致"是两个独立问题。** 环境反馈不是推理过程（mask=0），但只要它进上下文，截断/格式不统一就照样破坏训推一致。

**think 不闭合的 TITO 式正解（举例）。** 不要"补一个 `</think>` token 再重编码"（破坏 TITO），而是用容错解析器只返回下标、不造新 token：

```python
def parse_think_span(token_ids, tok):
    start = token_ids.index(tok.convert_tokens_to_ids("<think>")) + 1
    close = tok.convert_tokens_to_ids("</think>")
    if close in token_ids:
        end = token_ids.index(close)
    else:  # 缺闭合：用下一个结构边界兜底
        cands = [i for i,t in enumerate(token_ids)
                 if t in (tok.convert_tokens_to_ids("<tool_call>"), tok.eos_token_id) and i > start]
        end = min(cands) if cands else len(token_ids)
    return start, end, (close in token_ids)   # buffer 原封不动
```

buffer 是唯一真相，解析只吐索引 + closed 标志；mask 套在原始下标上；想让模型学会闭合 → 用 reward shaping（没闭合给小惩罚），而不是事后塞 token。

**RL 稳定性常见坑两条。** 熵塌缩到 0 / 空输出常由 advantage 计算 bug + off-policy 程度过大叠加放大，先查优势估计、再查 train/inference 配比。rollout KL 爆炸常由训练侧与推理侧的截断逻辑不一致造成（如一边 prompt/resp 分开左截断、另一边联合截断），导致 `old_log_probs` 对不上，解法是截断即丢弃该样本、保证一致性。

## Qwen3.5 的特殊考虑

Qwen3.5 用 ChatML 体系（`<|im_start|>{role}\n...<|im_end|>\n`），与旧自有格式差异大，踩坑集中在以下几处。

`bos_token = null` 是设计而非 bug。ChatML 用 `<|im_start|>` / `<|im_end|>` 标界，不依赖 BOS，所以 Qwen 的 `bos_token` 为 null。风险在于训练框架若假设 bos 非空，会把前缀错误拼成 `Noneassistant\n`，直接造成训推不一致、分数异常偏低。对策是模板层显式处理 null bos，断言渲染出的前缀是 `<|im_start|>assistant\n`。

tool_call 必须统一成 Qwen XML 格式。历史数据混入多种协议会让 Qwen3.5 同时学多套互斥的工具调用规范，导致 function call 大面积混乱。需要统一为 Qwen 目标格式：

```xml
<!-- 旧 XML65（要清掉） -->
<seed:tool_call>
<function name="read_file">
<parameter name="path" string="true">src/main.py</parameter>
</function>
</seed:tool_call>

<!-- 统一为 Qwen3.5 目标格式 -->
<tool_call>
<function=read_file>
<parameter=path>
src/main.py
</parameter>
</function>
</tool_call>
```

要点是：XML65 / Seed / JSON 多来源统一映射到 Qwen XML，以一个中间表示做"多对一"，避免逐对适配。

think 结构要保留并补齐空 think。已有 `<think>...</think>` 或 `<think_never_used_xxx>...</think_never_used_xxx>` 的样本原样保留，交给训练侧 chat template 处理；assistant 没有 think 的，补一个空的 `<think> </think>`——目的是让模型稳定学习 `</think>` 终止符，又不伪造思维链内容。这正是从源头缓解"think 不闭合"的手段。

tool result 保持原样、不强制规范化。tool result 只作为上下文输入，不是模型要学习输出的 target。保留原始返回形态可增强模型对真实工具返回多样性的泛化能力——对应通用清单里"环境产物 mask=0、只当 prompt"。

清理 toolcall 前后的桥接废话。删除 `<tool_call>` 前后的自然语言桥接句（如"我需要先调用工具查看文件。"），让 assistant target 更干净，避免模型学会在 toolcall 前输出多余自然语言（降低格式漂移）。

清理 system prompt 旧协议污染。清除 system prompt 里残留的 XML65 / Seed 指令（如 `string="true"`、`seed:tool_call`、"严格遵守 XML65 协议"）及品牌/IDE/环境泄露——否则会持续把模型往旧格式带。

脏样本直接丢弃、不强行修复。以下情况丢弃而非硬修：assistant 残留 `{"toolcall":...}` / `<seed:tool_call>` / `<action>` / `<mcfile>`；think 标签不闭合；`<tool_call>` 结构损坏；清洗后无有效 toolcall；toolcall 外仍有无法解释的残留文本。
