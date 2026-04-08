# EcoGPT - Financial LLM Post-Training Framework

基于 SFT + GRPO 的金融大模型后训练框架，融合 XuanYuan 和 DISC-FinLLM 的数据构造与训练方法。

## 项目结构

```
EcoGPT/
├── configs/                          # 训练配置模板
│   ├── sft_config_example.json
│   └── grpo_config_example.json
├── data/                             # 数据集存放
│   ├── sft/                          # SFT 训练数据
│   │   ├── raw/                      #   原始数据（DISC-FIN-SFT, CFData 等）
│   │   ├── processed/                #   处理后数据
│   │   ├── train/                    #   训练集
│   │   └── val/                      #   验证集
│   ├── grpo/                         # GRPO 训练数据
│   │   ├── raw/                      #   原始数据（Fin-R1-Data 等）
│   │   ├── processed/                #   处理后数据
│   │   ├── train/                    #   训练集（需含 ground_truth）
│   │   └── val/                      #   验证集
│   ├── eval/                         # 评测基准（不可用于训练）
│   │   ├── financeiq/                #   FinanceIQ (XuanYuan) [主指标]
│   │   ├── fineval/                  #   FinEval
│   │   ├── ceval_finance/            #   CEVAL 金融子集 [主指标]
│   │   └── disc_fin_eval/            #   DISC-FIN-Eval [参考指标, 同源风险]
│   └── pretrain/                     # 继续预训练数据（可选）
├── models/                           # 模型存放
│   ├── base/                         #   基座模型 (Qwen2.5-7B-Instruct)
│   ├── sft_lora/                     #   SFT LoRA 适配器
│   ├── sft_merged/                   #   SFT 合并后模型
│   ├── grpo_lora/                    #   GRPO LoRA 适配器
│   ├── grpo_merged/                  #   GRPO 合并后最终模型
│   └── reward_model/                 #   奖励模型（可选, PPO 路径）
├── scripts/
│   ├── training/                     # 训练脚本
│   │   ├── supervised_finetuning.py  #   SFT 训练
│   │   ├── grpo_training.py          #   GRPO 训练 [新建]
│   │   ├── dpo_training.py           #   DPO 训练（备选）
│   │   ├── ppo_training.py           #   PPO 训练（备选）
│   │   └── reward_modeling.py        #   奖励模型训练（备选）
│   ├── data_processing/              # 数据处理
│   │   ├── decontaminate.py          #   数据去污染 [新建]
│   │   ├── apaca2conversation.py     #   Alpaca → 对话格式
│   │   ├── data_filter.py            #   Token 长度过滤
│   │   ├── data_split.py             #   训练/验证划分
│   │   ├── check2.py                 #   SHA1 + MinHash 去重
│   │   └── analyze_dataset.py        #   数据质量分析
│   ├── evaluation/                   # 评估脚本
│   │   ├── sanity_check.py           #   SFT→GRPO 过渡检查 [新建]
│   │   ├── eval_ppl_sft_jsonl.py     #   PPL 评估
│   │   ├── pairwise_acc.py           #   奖励模型准确率
│   │   └── compare.py                #   模型对比
│   ├── utils/                        # 工具
│   │   ├── template.py               #   Prompt 模板（15+ 模型）
│   │   ├── merge_lora.py             #   LoRA 合并
│   │   └── chat_cli.py               #   交互式对话
│   ├── run_step0_data_prepare.sh     # Step 0: 数据准备
│   ├── run_step1_sft.sh              # Step 1: SFT 训练
│   ├── run_step1_5_sanity_check.sh   # Step 1.5: 过渡检查
│   ├── run_step2_grpo.sh             # Step 2: GRPO 训练
│   └── run_step3_eval.sh             # Step 3: 评估
├── logs/                             # TensorBoard 日志
├── outputs/                          # 输出结果
└── requirements.txt
```

## 训练流程

```
Step 0   数据准备 → 格式统一 → 去重 → 去污染 → 划分
Step 1   SFT 训练 → LoRA rank=16, lr=1e-4, 2 epochs
Step 1.5 Sanity Check → pass@8 验证 (10%~80% 为最佳)
Step 2   GRPO 训练 → LoRA rank=8, lr=5e-6, beta=0.04
Step 3   评估 → FinanceIQ + CEVAL + 通用回归测试
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 放置基座模型
# 将 Qwen2.5-7B-Instruct 下载到 models/base/

# 3. 放置训练数据
# SFT 数据 → data/sft/raw/
# GRPO 数据 → data/grpo/raw/
# 评测数据 → data/eval/

# 4. 按步骤执行
bash scripts/run_step0_data_prepare.sh
bash scripts/run_step1_sft.sh
python scripts/utils/merge_lora.py --base_model models/base/Qwen2.5-7B-Instruct --adapter models/sft_lora --output models/sft_merged
bash scripts/run_step1_5_sanity_check.sh
bash scripts/run_step2_grpo.sh
python scripts/utils/merge_lora.py --base_model models/sft_merged --adapter models/grpo_lora --output models/grpo_merged
bash scripts/run_step3_eval.sh
```

## 数据源

| 数据集 | 用途 | 来源 |
|--------|------|------|
| DISC-FIN-SFT | SFT 核心数据 (246K) | `eggbiscuit/DISC-FIN-SFT` |
| CFData-sft | SFT 补充 (采样50-100K) | `TongjiFinLab/CFGPT` |
| FinCorpus | Self-QA 生成源 | `Duxiaoman-DI/FinCorpus` |
| Fin-R1-Data | GRPO 训练数据 (60K) | arxiv:2503.16252 |
| FinanceIQ | 评测 [主指标] | `Duxiaoman-DI/XuanYuan` |
| CEVAL | 评测 [主指标] | 标准 benchmark |

## GRPO 数据格式

```jsonl
{"prompt": "某公司2023年营收500亿，净利率10%，2024年营收增长20%，净利率提升至12%，求2024年净利润。", "ground_truth": "72"}
```

## 关键超参数

| 阶段 | 微调方式 | LR | LoRA | beta |
|------|----------|------|------|------|
| SFT | LoRA r=16 | 1e-4 | alpha=32 | - |
| GRPO | LoRA r=8 | 5e-6 | alpha=16 | 0.04 |
