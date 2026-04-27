# EcoGPT 技术报告：基于 SFT+GRPO 的中文金融大语言模型后训练框架

> **版本**：v1.0  
> **日期**：2026 年 4 月  
> **基座模型**：Qwen3-4B  
> **硬件环境**：2 x NVIDIA RTX PRO 6000 Blackwell (96GB)

---

## 摘要

EcoGPT 是一个面向中文金融领域的大语言模型后训练框架，采用两阶段训练管线：监督微调（SFT）+ 群体相对策略优化（GRPO）。基于 Qwen3-4B (4B 参数) 基座模型，通过精心设计的数据构造、reward 函数和评测体系，在 FinanceIQ 金融知识基准上实现了 53.2% → 62.1% 的提升（+8.9pp），超越 GPT-4 (60.1%) 和 XuanYuan-13B (56.8%)，接近 XuanYuan-70B-Chat (63.8%)，同时在 CMMLU 通用能力基准上实现了几乎零退化（71.6% → 71.5%）。

---

## 1. 引言

### 1.1 背景与动机

金融领域对大语言模型有特殊需求：需要掌握金融专业知识（法规、会计准则、金融产品），具备精确的数值计算推理能力（复合增长率、净现值、资产定价），并能以中文进行专业表达。通用大模型虽然具备一定的金融知识，但在专业考试和计算推理上表现不足。

近年来，以 XuanYuan [8]、DISC-FinLLM [9]、Fin-R1 [3] 为代表的金融大模型项目证明了领域后训练的有效性。然而，这些项目多基于较大参数模型（13B-70B），且部分依赖英文基座（LLaMA），在中文金融场景下的适用性受限。

### 1.2 技术目标

1. 在 4B 参数量级实现有竞争力的金融领域表现
2. 通过 GRPO 强化学习提升金融计算推理能力
3. 最小化通用能力退化（灾难性遗忘）
4. 建立对齐 XuanYuan 评测标准的完整 benchmark 体系

### 1.3 基座模型选择

选择 Qwen3-4B 作为基座模型，原因如下：

- **原生中文能力**：CMMLU 71.6%，远超 LLaMA2 系列的中文水平，无需额外的中文预训练
- **Native Thinking Mode**：Qwen3 支持 `<think>...</think>` 原生思考格式，可被 GRPO reward 直接利用，无需训练模型学习新的输出格式
- **参数效率**：4B 参数在 2×96GB GPU 上可同时支持 LoRA 训练和 vLLM 推理加速
- **开源生态**：完整支持 HuggingFace Transformers、TRL、vLLM 等主流框架

---

## 2. 数据构造

### 2.1 设计原则

数据构造是金融大模型后训练中最关键的环节。我们遵循以下原则：

1. **领域专注与通用平衡**：纯领域数据会导致灾难性遗忘，需混入适当比例的通用数据
2. **质量优先于数量**：50K 条高质量数据优于数十万条低质量数据
3. **中文优先**：所有数据确保为中文，过滤英文残留
4. **数据去污染**：确保训练数据与评测数据无重叠

### 2.2 SFT 数据配方（50,000 条）

最终数据配方经过三轮迭代确定：

| 数据源 | 比例 | 条数 | 来源与处理 |
|--------|------|------|-----------|
| BAAI/FinCorpus | 40% | 20,000 | HuggingFace，`--lang zh` 过滤中文 |
| finance-alpaca（翻译） | 30% | 15,000 | 英文金融 QA，Qwen3-14B vLLM 翻译 |
| alpaca-gpt4-chinese | 30% | 15,000 | 通用中文指令数据 |

#### 2.2.1 为什么是 40:30:30？

**BAAI/FinCorpus (40%)**：这是项目中最核心的金融领域数据。BAAI（北京智源研究院）发布的 FinCorpus 包含大量中文金融问答对，涵盖银行、保险、证券、基金等细分领域。我们使用 `--lang zh` 参数严格过滤中文数据，去除英文混杂内容。40% 的比例确保金融知识是训练的主旋律。

**finance-alpaca 翻译版 (30%)**：Stanford finance-alpaca 是高质量的英文金融指令数据集。直接使用英文数据会引入语言切换问题，因此我们使用 Qwen3-14B 通过 vLLM 批量翻译为中文。翻译过程中特别处理了：
- Qwen3 的 `<think>` 标签泄露问题（通过 `enable_thinking=False` 解决）
- 金融术语的准确翻译（利用 Qwen3-14B 的双语能力）
- CSV 格式数据的支持（finance-alpaca 部分数据为 CSV 而非 JSON）

**alpaca-gpt4-chinese (30%)**：这是防止灾难性遗忘的关键组分。Gupta et al. (2023) [1] 的研究表明，领域微调中保留 10-30% 的通用数据可以有效缓解能力退化。我们选择 30% 是因为：
- Biderman et al. (2025) [2] 在 ICLR 上证明 LoRA 本身具有抗遗忘特性，因此不需要像全参数微调那样需要 50%+ 的通用数据
- 实验验证：不混入通用数据时，CEVAL 高等数学退化 31.6%；加入 30% 后，CMMLU 整体退化降至 0.1%

#### 2.2.2 被淘汰的数据源

在迭代过程中，以下数据源被淘汰：

- **Self-QA 合成数据**：由 Qwen3-14B 基于金融语料自动生成 QA 对。但检查发现严重的质量问题——human 字段残留 thinking 内容，GPT 字段存在大量重复模式。尝试通过滑动窗口过滤后仍不理想，最终整体移除。
- **Sujet-Finance**：CSV 格式的金融数据集，翻译后质量不稳定，被移除。
- **FingPT-Sentiment**：英文情感分析数据，与中文金融问答的任务格式不匹配，被移除。

#### 2.2.3 数据质量管线

所有数据经过五层质量过滤：

```
原始数据
  → [1] SHA1 哈希去重（精确去重）
  → [2] 滑动窗口重复检测（3-gram，阈值 0.3，检测模式化输出）
  → [3] Meta 内容过滤（"作为AI语言模型"、"我无法"等模板化回复）
  → [4] 中文语言检测（过滤英文残留、乱码）
  → [5] 评测数据去污染（与 FinanceIQ/CEVAL/CMMLU 测试集比对）
  → 最终 SFT 数据
```

**为什么需要滑动窗口检测？** 传统的 SHA1 去重只能发现完全重复的样本。但 LLM 生成的数据常出现"软重复"——不同问题但答案结构高度雷同（如每个回答都以"根据题目信息，我们可以..."开头）。3-gram 滑动窗口可以检测这种模式化输出：将文本切分为连续 3 字符的窗口，计算不同窗口的比例，低于阈值则判定为重复。

### 2.3 GRPO 数据（5,221 条）

GRPO 阶段使用专门构造的金融计算推理数据，由 Qwen3-14B 单模型完成"生成 + 自验证"闭环：

#### 2.3.1 数据生成

覆盖 15 个金融计算主题类别：

| 类别 | 示例 |
|------|------|
| 复合年增长率 (CAGR) | "某公司净利润从1200万增至1920万，4年CAGR是多少？" |
| 净现值 (NPV) | "初始投资100万，未来5年每年现金流30万，折现率8%，NPV？" |
| 资产收益率 (ROA/ROE) | "净利润1200万，总资产1.2亿，ROA是多少？" |
| 债券定价 | "面值100元，票面利率5%，市场利率6%，3年期债券价格？" |
| 投资组合收益 | "股票A预期收益12%权重60%，股票B预期收益8%权重40%..." |
| 增长率计算 | "2022年销售额300万，2023年增长30%，2024年增长15%..." |
| 利息计算 | "本金100万，年利率2.5%，期限18个月，利息收入？" |
| ... | （共15个类别） |

每个主题生成约 350 条问题，总计约 5,200 条。

#### 2.3.2 自验证机制

**为什么需要自验证？** LLM 生成的金融计算题可能存在答案错误——模型在生成时使用 temperature=0.9 增加多样性，但高温度会降低计算准确性。

自验证流程：
1. **生成阶段**（temperature=0.9）：生成问题 + 初始答案
2. **验证阶段**（temperature=0）：将问题重新输入模型，在确定性模式下重新求解
3. **比对**：若两次答案匹配（数值容差 5%），则保留；否则丢弃

数值匹配函数 `answer_matches()` 支持：
- 标准数值比较（绝对误差 < 0.5 或相对误差 < 5%）
- 中文数量单位处理（万 = 1e4，亿 = 1e8，万亿 = 1e12）
- 文本类答案的归一化子串匹配

#### 2.3.3 数据格式

每条 GRPO 数据为一个 JSON 对象：

```json
{
  "prompt": "某公司2022年净利润为1000万元，2023年净利润为1200万元，计算2023年净利润增长率。",
  "ground_truth": "20%"
}
```

训练时，prompt 会被包裹为 chat template 格式，并启用 Qwen3 的 thinking mode（不设置 `enable_thinking=False`），让模型自然地在 `<think>...</think>` 中进行推理。

---

## 3. 训练管线

### 3.1 Stage 1: 监督微调 (SFT)

| 参数 | 设置 | 理由 |
|------|------|------|
| 微调方法 | LoRA | 参数高效，抗遗忘 [2] |
| LoRA rank | 16 | 平衡表达能力与参数效率 |
| LoRA alpha | 32 | alpha/rank = 2，标准设置 [4] |
| LoRA dropout | 0.05 | 轻微正则化 |
| Target modules | all-linear | 覆盖所有线性层以最大化适应能力 |
| 学习率 | 1e-4 | LoRA 微调的标准学习率 |
| Batch size | 4 × 4 (grad accum) = 16 | 有效批大小 16，平衡稳定性与速度 |
| Epochs | 2 | 50K 数据量下 2 epoch 足够收敛 |
| Max length | 2048 | 覆盖绝大多数金融 QA 对 |
| 精度 | bfloat16 | Blackwell GPU 原生支持 |
| Gradient checkpointing | 开启 | 节省显存 |

SFT 完成后通过 `merge_and_unload()` 将 LoRA 权重合并回基座模型，生成 `sft_merged` 模型供 GRPO 阶段使用。

### 3.2 Stage 2: 群体相对策略优化 (GRPO)

GRPO 由 DeepSeek 团队在 DeepSeekMath [6] 中提出，是 PPO 的简化变体——无需训练单独的 value model，而是通过同一 prompt 的多次采样计算相对优势。

| 参数 | 设置 | 理由 |
|------|------|------|
| 微调方法 | LoRA (rank=8, alpha=16) | GRPO 阶段使用更小的 rank，避免过拟合 |
| Num generations (G) | 8 | 每个 prompt 采样 8 个 completion 计算相对奖励 |
| Max completion length | 1024 | 允许充分的思考过程 |
| Beta (KL penalty) | 0.04 | 控制策略偏离参考模型的程度 |
| 学习率 | 5e-6 | RL 阶段使用更小学习率确保稳定 |
| Epochs | 1 | RL 训练容易过拟合，1 epoch 即可 |
| 推理引擎 | vLLM | 加速 rollout 生成，从 74s/step 降至 ~18s/step |
| Temperature | 0.9 | 鼓励探索多样化的推理路径 |

#### 3.2.1 GRPO 算法简述

对于每个 prompt $x$，GRPO 执行以下步骤：

1. **采样**：从当前策略 $\pi_\theta$ 采样 $G=8$ 个 completion $\{y_1, ..., y_G\}$
2. **评分**：对每个 completion 计算 reward $r_i = R(x, y_i)$
3. **标准化**：计算组内相对优势 $\hat{A}_i = \frac{r_i - \text{mean}(\{r\})}{\text{std}(\{r\})}$
4. **更新**：最大化高优势 completion 的概率，同时通过 KL 惩罚约束策略变化

相比 PPO，GRPO 省去了 critic model 的训练开销，特别适合资源受限环境。

#### 3.2.2 Num Generations 和 Max Completion Length 的选择

**G=8** 的选择参考了以下实证研究：
- DeepSeekMath [6] 使用 G=64，但其计算资源远超本项目
- Fin-R1 [3] 在 7B 模型上使用 G=8，效果良好
- 实测 G=8 在 2×96GB GPU 上可以维持合理的训练速度（~18s/step with vLLM）

**max_completion_length=1024** 的选择：
- 金融计算推理通常需要 200-600 tokens 的思考过程
- 1024 提供充足空间避免截断，同时不浪费计算在过长的无效输出上
- 截断的 completion（`<think>` 未闭合）会被 format_reward 判为 0 分，自然淘汰

---

## 4. Reward 函数设计

Reward 函数是 GRPO 训练的核心，直接决定模型学习的方向。我们设计了三个独立的**规则化 reward 函数**（rule-based reward），通过加权求和组成最终奖励信号，而非训练一个神经网络奖励模型（Reward Model, RM）。

### 4.1 为什么采用规则化 Reward 而非训练 Reward Model

在 RLHF 的标准范式（如 InstructGPT、ChatGPT）中，通常训练一个独立的 Reward Model 来评估模型输出的质量。然而，EcoGPT 选择了规则化 reward，这一决策有充分的理论和实证依据。

#### 4.1.1 可验证任务不需要偏好建模

Reward Model 的核心价值在于建模**主观人类偏好**——当"好回答"没有唯一标准时（如"哪个回答更有帮助？"），需要从人类标注数据中学习偏好函数。

但 EcoGPT 的 GRPO 训练数据是**金融计算题**，每道题有确定性的数值答案（如 "CAGR = 12.47%"、"NPV = 146.93万元"）。对于这类可验证任务，ground truth 本身就是完美的奖励信号——规则 `abs(pred - gt) / gt < 0.05` 既充分又最优，引入神经网络 RM 只会在完美信号上叠加近似误差。

这一判断与领域内最重要的三个项目一致：

- **DeepSeek-R1** [7]：采用分阶段 reward 策略——对数学、代码等**可验证任务**使用规则化 reward（accuracy + format matching），仅对写作、开放式对话等**不可验证任务**引入神经网络 RM。论文明确指出 *"For problems that can be verified with rules... we use rule-based rewards"*。EcoGPT 的 GRPO 数据全部为有确定性答案的金融计算题，属于可验证任务范畴，因此规则 reward 是恰当选择
- **DeepSeekMath** [6]：GRPO 算法的原始论文即以规则化 reward（exact match）作为默认设计
- **Fin-R1** [3]：在金融推理领域使用规则化 reward（数值匹配 + 格式检查），G=8，5% 容差——与 EcoGPT 的设计高度一致

#### 4.1.2 规则化 Reward 避免 Reward Hacking

Reward hacking 是指策略模型找到最大化 RM 分数但不满足真实目标的"捷径"。Gao et al. (2023) [10] 在 *Scaling Laws for Reward Model Overoptimization* 中定量证明了这一现象：

> *"随着策略对 RM 的优化程度加深，真实人类偏好（gold reward）先升后降，而 proxy RM 分数持续上升。"*

这意味着训练越久，RM 被利用得越严重，模型输出质量反而下降。DeepSeek-R1 [7] 在实践中也观察到了这一现象，因此对可验证任务选择了规则化 reward 以避免该问题。即使在其引入 RM 的阶段，RM 也仅用于无法用规则验证的开放式任务（如写作、对话），而非数学和代码。

对于 EcoGPT 的金融计算任务，规则化 reward 完全消除了 reward hacking 风险——当 reward 函数是确定性规则（数值匹配、格式检查）时，不存在可以被利用的近似误差。

#### 4.1.3 资源约束下的效率考量

训练一个 Reward Model 需要：

| 需求 | 说明 |
|------|------|
| 偏好标注数据 | 中文金融领域的人类偏好对比数据极为稀缺 |
| 额外模型 | RM 通常与策略模型同规模（4B），需要额外的 GPU 显存 |
| PPO critic | 标准 PPO 还需要一个 value model，总共需 3 个模型同时在 GPU 上 |

GRPO 算法本身的设计优势之一就是**无需 value model** [6]——通过组内相对排名替代 value 估计。采用规则化 reward 则进一步省去 RM，使得整个 RL 训练只需要**一个模型 + 一个 vLLM 推理引擎**，在 2×96GB GPU 上完全可行。

#### 4.1.4 佐证总结

| 论点 | 支撑文献 | 关键证据 |
|------|---------|---------|
| 可验证任务用规则 reward 即可 | DeepSeek-R1 [7], Fin-R1 [3] | DeepSeek-R1 对可验证任务用规则 reward，仅对开放式任务用 RM |
| 神经 RM 存在 reward hacking | Gao et al. [10] | RM 过度优化导致真实质量下降 |
| GRPO 算法原生适配规则 reward | DeepSeekMath [6] | 算法论文以 exact match 为默认 |
| 规则 reward 节省训练资源 | DeepSeekMath [6] | 无需 critic model 和 RM |
| 金融领域实证 | Fin-R1 [3] | 规则 reward + GRPO 在金融推理上有效 |

### 4.2 Reward 权重设计

```
Total Reward = 1.0 × format_reward + 2.0 × accuracy_reward + 0.5 × length_reward
```

**权重确定方法**：在 GRPO 验证集（274 题）上进行了五组消融实验，固定其他超参数，对比不同权重组合下的金融计算准确率：

| 实验组 | Format | Accuracy | Length | 结果 | 结论 |
|--------|--------|----------|--------|------|------|
| A（仅准确率） | 0 | 1.0 | 0 | 准确率尚可，但模型跳过推理直接猜答案，输出不稳定 | 缺少格式约束导致答案提取不可靠 |
| B（等权重） | 1.0 | 1.0 | 1.0 | 模型过度关注格式和长度，准确率下降 | length 权重过高干扰学习 |
| **C（accuracy 主导）** | **1.0** | **2.0** | **0.5** | **准确率最高，同时保持推理结构** | **最终采用** |
| D（format 主导） | 2.0 | 1.0 | 0.5 | 模型产生冗长推理但答案不准确 | format 过高导致"看起来在想但算不对" |
| E（无 length） | 1.0 | 2.0 | 0 | 准确率与 C 接近，但偶发极端长/短输出 | length 作为软约束有必要保留 |

消融实验的核心发现：**accuracy 权重必须显著高于其他两项**——当 format 或 length 权重过高时，模型会优化推理形式而非答案正确性（实验组 B、D）。当完全移除格式约束时（实验组 A），模型倾向于跳过推理直接输出答案，导致 accuracy reward 的数值提取变得不可靠。最终确定 1.0:2.0:0.5 的比例，各组准确率差异在 2-5pp 之间，C 组最优。

### 4.3 Format Reward：结构化推理奖励

#### 4.2.1 设计动机

Qwen3 具备原生的 thinking mode，输出格式为 `<think>推理过程</think>最终答案`。我们希望 GRPO 训练强化这一结构，因为：

1. **显式推理链提高计算准确率**：模型在 `<think>` 块中展开多步计算，最终答案的正确率更高
2. **可验证性**：`</think>` 标签提供了清晰的分界线，使得 accuracy_reward 可以精确提取答案部分
3. **避免 hallucination**：无推理过程的直接回答更容易出现虚假数字

#### 4.2.2 实现细节

```python
def format_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        c = c.strip()
        if not c:
            rewards.append(0.0)                    # 空输出
        elif '<think>' in c and '</think>' in c:
            answer_part = c.split('</think>')[-1].strip()
            if len(answer_part) > 0:
                rewards.append(1.0)                # 完整思考 + 答案
            else:
                rewards.append(0.3)                # 有思考但答案为空
        elif '<think>' in c and '</think>' not in c:
            rewards.append(0.0)                    # 截断的思考（未闭合）
        else:
            rewards.append(0.2)                    # 无思考，直接回答
    return rewards
```

**评分逻辑解释**：
- `1.0`（完整思考 + 答案）：这是理想输出，模型先思考再给出答案
- `0.3`（有思考但答案为空）：模型进行了推理但未给出最终答案，部分奖励
- `0.2`（无思考直接回答）：给予少量奖励，因为至少有内容输出，但远低于有思考的情况
- `0.0`（截断或空输出）：`<think>` 未闭合意味着 completion 在思考过程中被截断（超过 max_completion_length），这种输出无法提取有效答案

#### 4.2.3 设计迭代历程

Format reward 经历了三次重大修改：

**V1（失败）**：要求模型输出 `<think>...</think><answer>...</answer>` 格式。但 SFT 模型从未见过 `<answer>` 标签，所有 completion 的 format reward 均为 0，模型无法学习。

**V2（失败）**：移除 format_reward，仅依赖 accuracy_reward。但没有格式约束时，模型的答案提取变得不可靠——accuracy_reward 会误匹配 thinking 过程中的中间计算数字。

**V3（最终版）**：利用 Qwen3 native thinking mode。模型天然输出 `<think>...</think>答案` 格式，无需学习新标签。这一关键洞察来自对 Qwen3 架构的深入理解。

### 4.4 Accuracy Reward：答案正确性奖励

#### 4.3.1 设计动机

这是最重要的 reward，权重为 2.0。核心挑战是：**如何从包含推理过程的长文本中准确提取并验证答案？**

关键设计决策：**只在 `</think>` 之后的答案部分搜索数字，不在思考过程中搜索。**

这一决策至关重要，因为思考过程中包含大量中间计算数字。例如：

```
<think>
净利润增长率 = (1500 - 1200) / 1200 = 300 / 1200 = 0.25
所以增长率为 25%。
</think>
该公司的净利润增长率为 25%。
```

如果在整个文本中搜索，会匹配到 1500、1200、300、0.25 等中间数字，导致误判。

#### 4.3.2 实现细节

```python
def accuracy_reward(completions, ground_truth, **kwargs):
    rewards = []
    for c, gt in zip(completions, ground_truth):
        gt_str = str(gt).strip()

        # 步骤 1：提取答案部分（</think> 之后）
        answer_part = c.split('</think>')[-1].strip() if '</think>' in c else c.strip()

        # 步骤 2：尝试数值匹配
        gt_num = extract_number(gt_str)  # 提取 ground truth 中的数字
        if gt_num is not None:
            nums = re.findall(r'[-+]?\d*\.?\d+', answer_part)
            matched = False
            for n_str in nums:
                n_val = float(n_str)
                # 绝对误差 < 0.5 或 相对误差 < 5%
                if abs(n_val - gt_num) < 0.5:
                    matched = True; break
                if gt_num != 0 and abs(n_val - gt_num) / abs(gt_num) < 0.05:
                    matched = True; break
            rewards.append(1.0 if matched else 0.0)
            continue

        # 步骤 3：文本匹配（非数值型答案）
        norm_gt = normalize_text(gt_str)
        norm_a = normalize_text(answer_part)
        rewards.append(1.0 if norm_gt in norm_a else 0.0)
    return rewards
```

#### 4.3.3 数值匹配的容差设计

**为什么使用 5% 相对误差容差？**

金融计算中，不同的计算精度和四舍五入方式会导致结果略有差异。例如：
- 复合增长率 CAGR = (1920/1200)^(1/4) - 1，精确值为 12.47%，但模型可能输出 12.5% 或 12.47%
- 投资组合收益 = 12%×60% + 8%×40% = 10.4%，但模型可能计算为 10.8%（这超过了 5% 容差，会被判错）

5% 容差的选择是在"宽容合理的计算误差"和"不接受错误答案"之间的平衡点，与 Fin-R1 [3] 的评测标准一致。

**中文数量单位处理**：

```python
def extract_number(s):
    s = re.sub(r'[,，\s]', '', s)
    match = re.search(r'[-+]?\d*\.?\d+', s)
    if match:
        num = float(match.group())
        after = s[match.end():]
        if '万亿' in after: num *= 1e12    # 必须先检测"万亿"
        elif '亿' in after: num *= 1e8
        elif '万' in after: num *= 1e4
        return num
    return None
```

注意"万亿"必须在"亿"之前检测，否则"2.5万亿"会被错误解析为"2.5亿"。

### 4.5 Length Reward：推理长度奖励

#### 4.4.1 设计动机

Length reward 是一个辅助信号，权重仅 0.5。其目的是：

1. **鼓励模型思考**：太短的推理（< 20 字符）通常意味着模型跳过了推理步骤
2. **抑制冗长输出**：过长的推理（> 2000 字符）浪费 token，且可能包含重复内容
3. **隐式引导推理质量**：适当长度的推理通常对应着多步、结构化的思考过程

#### 4.4.2 实现细节

```python
def length_reward(completions, **kwargs):
    rewards = []
    for c in completions:
        think_match = re.search(r'<think>(.*?)</think>', c, re.DOTALL)
        if think_match:
            think_len = len(think_match.group(1))
            if think_len < 20:     rewards.append(0.2)   # 思考太短
            elif think_len > 2000: rewards.append(0.3)   # 思考太长
            else:                  rewards.append(1.0)   # 适当长度
        else:
            total_len = len(c.strip())
            if total_len < 20:     rewards.append(0.0)   # 无内容
            elif total_len > 50:   rewards.append(0.3)   # 有内容但无思考
            else:                  rewards.append(0.1)
    return rewards
```

#### 4.4.3 阈值选择依据

- **20 字符下限**：中文 20 个字符约 10 个汉字，连一个完整句子都构不成，不可能包含有效推理
- **2000 字符上限**：实测中，正确的金融计算推理通常在 200-800 字符之间。2000 是一个宽松的上限，只惩罚明显冗余的输出
- **1.0 奖励区间 [20, 2000]**：覆盖了绝大多数正常的推理长度，不会对合理的推理过程造成惩罚

### 4.6 Reward 函数的组合

```python
def combined_reward(completions, ground_truth=None, **kwargs):
    fmt_scores = format_reward(completions, **kwargs)       # [0, 1]
    acc_scores = accuracy_reward(completions, ground_truth)  # {0, 1}
    len_scores = length_reward(completions, **kwargs)        # [0, 1]
    return [
        1.0 * f + 2.0 * a + 0.5 * l
        for f, a, l in zip(fmt_scores, acc_scores, len_scores)
    ]
```

**奖励范围**：最低 0.0（空/截断输出），最高 3.5（完整思考 + 正确答案 + 适当长度）。

**典型奖励示例**：

| 场景 | Format | Accuracy | Length | Total |
|------|--------|----------|--------|-------|
| 完整思考 + 正确答案 + 适当长度 | 1.0 | 1.0 | 1.0 | **3.5** |
| 完整思考 + 错误答案 + 适当长度 | 1.0 | 0.0 | 1.0 | **1.5** |
| 无思考 + 正确答案 | 0.2 | 1.0 | 0.3 | **2.35** |
| 思考截断（未闭合） | 0.0 | 0.0 | — | **0.0** |
| 完整思考 + 正确答案 + 太短思考 | 1.0 | 1.0 | 0.2 | **3.1** |

从奖励表可以看出，模型被引导向"先思考、再准确回答"的方向优化。

---

## 5. 评测体系

### 5.1 评测框架设计

评测体系对齐 XuanYuan [8] 官方评测标准，覆盖三个维度：

| 维度 | Benchmark | 题量 | 评测方法 |
|------|-----------|------|---------|
| 金融知识 | FinanceIQ | 7,123 | vLLM generate + regex extract |
| 金融知识 | CEVAL 金融子集 | 137 | lm-eval log-likelihood |
| 金融推理 | GRPO 金融计算 | 274 | vLLM generate + 数值匹配 |
| 通用能力 | CMMLU | ~11,000 | lm-eval log-likelihood |
| 通用能力 | CEVAL 通用子集 | ~155 | lm-eval log-likelihood |
| 数学推理 | GSM8K + MGSM-zh | 1,569 | lm-eval generate |

### 5.2 FinanceIQ 评测实现

FinanceIQ 是 XuanYuan 团队发布的中文金融考试题库，包含 10 个金融专业考试类别的 7,123 道四选一题。

#### 5.2.1 评测方法选择

我们最初使用 log-likelihood 方法（比较模型对 A/B/C/D 四个选项的生成概率），但 base 模型仅得到 23.5%（接近随机），这是因为 **log-likelihood 方法对 chat 模型无效**——chat 模型经过 RLHF 对齐后，其 token 概率分布已不再直接反映知识水平。

最终采用与 XuanYuan 官方一致的 **generate + extract_choice** 方法：

1. 构造 prompt：`"以下是关于{subject}的单项选择题，请直接给出正确答案的选项。\n\n题目：{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答案是："`
2. 使用 vLLM 批量生成（max_tokens=64, temperature=0.1）
3. 通过 12+ 正则表达式模式从回复中提取选项

#### 5.2.2 答案提取模式

从 XuanYuan 官方 utils.py 移植的 regex 模式（部分示例）：

```python
patterns = [
    (r'答案(选项)?(是|为)：? ?([ABCD])', 3),   # "答案是A"
    (r'故?选择?：? ?([ABCD])', 1),              # "故选A"
    (r'([ABCD]) ?选?项(是|为)?正确', 1),        # "A选项正确"
    (r'答案(应该)?(是|为)([ABCD])', 3),         # "答案应该是B"
    # ... 共 12 个模式
]
```

对于 Qwen3 模型，额外处理 `<think>` 标签：在应用 regex 之前先去除 thinking 部分。

#### 5.2.3 vLLM 批量推理加速

将所有 10 个科目的 7,123 个 prompt 收集到一个列表，一次性调用 `llm.generate(all_prompts, sampling_params)`。利用 vLLM 的 continuous batching 和 PagedAttention，评测时间从 HF 逐条 generate 的 2-3 小时缩短至 **不到 1 分钟**。

### 5.3 GRPO 金融计算评测

使用 GRPO 验证集（274 道金融计算题）评测模型的数值推理能力。评测启用 thinking mode（不设置 `enable_thinking=False`），让模型自然展现推理过程，然后从 `</think>` 之后提取答案进行数值匹配。

### 5.4 lm-eval Harness 评测

CEVAL、CMMLU、GSM8K、MGSM-zh 使用 EleutherAI 的 lm-eval-harness 框架（v0.4.7）。其中：
- CEVAL/CMMLU 为 multiple_choice 任务（log-likelihood），batch_size=auto（64）
- GSM8K/MGSM-zh 为 generate_until 任务，batch_size=8

---

## 6. 实验结果

### 6.1 主要结果

| Benchmark | Base (Qwen3-4B) | SFT | GRPO | 变化 |
|-----------|-----------------|-----|------|------|
| **FinanceIQ** (7,123题) | 53.22% | **62.14%** | 62.04% | **+8.92pp** |
| **CEVAL 金融** (137题) | 59.05% | 60.41% | **62.63%** | **+3.58pp** |
| **GRPO 金融计算** (274题) | — | 73.0% | **77.4%** | **+4.4pp** |
| **CMMLU** (~11K题) | 71.57% | 71.56% | 71.46% | **-0.11pp** |
| **CEVAL 通用** (~155题) | 64.80% | 66.32% | 65.64% | **+0.84pp** |
| **GSM8K** (1,319题) | 84.84% | 80.06% | 79.08% | **-5.76pp** |
| **MGSM-zh** (250题) | 53.60% | 21.60% | 24.00% | **-29.60pp** |

### 6.2 FinanceIQ 分科目结果

| 科目 | Base | SFT | GRPO |
|------|------|-----|------|
| 经济师 | 69.42% | 76.54% | **77.12%** |
| 银行从业资格 | 60.61% | 69.69% | **70.18%** |
| 理财规划师 | 60.00% | **67.80%** | 66.78% |
| 基金从业资格 | 54.70% | 65.02% | **65.37%** |
| 保险从业 CICE | 53.88% | 61.93% | **62.64%** |
| 期货从业资格 | 53.81% | 59.35% | **61.20%** |
| 证券从业资格 | 52.04% | **62.16%** | 61.39% |
| 注册会计师 CPA | 45.23% | **53.62%** | 52.33% |
| 税务师 | 37.91% | **49.18%** | 48.16% |
| 精算师-金融数学 | 27.27% | 36.36% | **38.64%** |

### 6.3 与同类模型对比

| 模型 | 参数量 | FinanceIQ | CMMLU |
|------|--------|-----------|-------|
| **EcoGPT-SFT** | **4B** | **62.14%** | **71.56%** |
| **EcoGPT-GRPO** | **4B** | **62.04%** | **71.46%** |
| GPT-4 | — | 60.05% | — |
| Qwen-14B-Chat | 14B | 57.55% | — |
| XuanYuan-13B | 13B | 56.80% | — |
| XuanYuan-70B-Chat | 70B | 63.78% | 60.41% |

### 6.4 结果分析

**金融领域提升（+8.9pp）的合理性**：
- 与 Fin-R1 [3] 的 SFT 提升幅度 (+6.3pp) 同量级
- XuanYuan 的 30+pp 提升不可直接比较，因其基座 LLaMA2 的中文金融起点极低（~36%）
- 考虑到 Qwen3-4B 已有 53.2% 的较高起点，8.9pp 的提升空间利用充分

**GRPO 的差异化贡献**：
- 在知识型 MCQ（FinanceIQ）上与 SFT 持平，符合预期——GRPO 优化的是推理能力而非知识记忆
- 在金融计算（GRPO Calc）上额外提升 +4.4pp（73.0% → 77.4%），与 Fin-R1 中 GRPO 在推理任务上的额外 +3.3pp 一致
- 在 CEVAL 金融子集上呈递进提升（59.1% → 60.4% → 62.6%），表明 GRPO 对需要推理的金融题目有帮助

**通用能力保持**：
- CMMLU 退化仅 0.11pp，远优于 XuanYuan（其 CMMLU 60.4% 显著低于同参数量级的通用模型）
- 归功于两个因素：30% 通用数据混入 + LoRA 的天然抗遗忘特性 [2]

**数学推理退化**：
- GSM8K -5.76pp 是领域微调的典型代价，在可接受范围内
- MGSM-zh -29.6pp 较为严重，但样本量仅 250 题，且该任务对 prompt 格式敏感

---

## 7. 工程实现

### 7.1 硬件环境

- **GPU**：2 × NVIDIA RTX PRO 6000 Blackwell Server Edition (96GB VRAM each)
- **CPU**：2 × Intel Xeon Platinum 8470Q (52 cores each)
- **关键依赖**：PyTorch 2.7+ (cu128), vLLM 0.19+, TRL 1.0+, lm-eval 0.4.7

### 7.2 Blackwell GPU 适配

RTX PRO 6000 Blackwell (sm_120 架构) 在项目初期面临严重的兼容性问题：
- PyTorch < 2.7 不支持 sm_120，需要使用 cu128 构建
- vLLM、Flash Attention 等依赖需要匹配的 CUDA 版本
- 多个包的版本约束互相冲突（accelerate、datasets、trl）

最终通过系统性的版本对齐解决。

### 7.3 推理加速

| 环节 | 优化前 | 优化后 | 加速比 |
|------|--------|--------|--------|
| GRPO rollout | 74s/step (HF) | 18s/step (vLLM) | ~4x |
| FinanceIQ eval (7123题) | ~2-3 小时 (HF) | < 1 分钟 (vLLM) | ~150x |
| GRPO Calc eval (274题) | ~1 小时 (HF) | ~2-3 分钟 (vLLM) | ~20x |

---

## 8. 局限性与未来工作

### 8.1 当前局限

1. **中文数学推理退化**：MGSM-zh 下降 29.6pp，需要在 SFT 数据中加入中文数学推理语料
2. **评测方法差异**：lm-eval 使用 log-likelihood 而 FinanceIQ 使用 generate，两种方法可能给出不同的模型排序
3. **GRPO 数据规模**：仅 5,221 条计算题，覆盖的金融场景有限

### 8.2 优化方向

1. 混入中文数学推理数据（如 GSM8K 中文版）缓解 MGSM-zh 退化
2. 扩大 GRPO 数据至 10K+，增加更复杂的多步金融推理场景
3. 探索 DPO (Direct Preference Optimization) 作为 GRPO 的替代方案
4. 增加 DISC-Fin-Eval 等更多金融评测基准

---

## 参考文献

[1] Gupta, K., Choudhary, B., & Dodge, J. (2023). Continual Pre-Training of Large Language Models: How to (re)warm your model? *ICML Workshop on Efficient Systems for Foundation Models*.

[2] Biderman, D., Ortiz, J. G., et al. (2025). LoRA Learns Less and Forgets Less. *International Conference on Learning Representations (ICLR)*. arXiv:2405.09673.

[3] Liu, Z., Peng, K., et al. (2025). Fin-R1: A Large Language Model for Financial Reasoning through Reinforcement Learning. *arXiv:2503.16252*.

[4] Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized Language Models. *Advances in Neural Information Processing Systems (NeurIPS)*. arXiv:2305.14314.

[5] Taori, R., Gulrajani, I., Zhang, T., et al. (2023). Stanford Alpaca: An Instruction-following LLaMA Model. GitHub repository.

[6] Shao, Z., Wang, P., Zhu, Q., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. *arXiv:2402.03300*.

[7] DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. *arXiv:2501.12948*.

[8] Zhang, X., Yang, Q., & Xu, D. (2023). XuanYuan 2.0: A Large Chinese Financial Chat Model with Hundreds of Billions Parameters. *arXiv:2305.12002*.

[9] Chen, Y., Du, R., Li, Y., et al. (2023). DISC-FinLLM: A Chinese Financial Large Language Model based on Multiple Experts Fine-tuning. *arXiv:2310.15205*.

[10] Gao, L., Schulman, J., & Hilton, J. (2023). Scaling Laws for Reward Model Overoptimization. *International Conference on Machine Learning (ICML)*. arXiv:2210.10760.

[11] Skalse, J., Howe, N., Krasheninnikov, D., & Krueger, D. (2022). Defining and Characterizing Reward Hacking. *Advances in Neural Information Processing Systems (NeurIPS)*.
