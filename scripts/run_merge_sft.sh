#!/bin/bash
# ============================================================
# EcoGPT - Merge SFT Datasets with Target Ratios
# ============================================================
# Combine multiple financial SFT datasets into one training file.
#
# 修正后配比 (基于实际可用数据):
#   baai_finance    40%  - BAAI IndustryInstruction (123K, 中英双语, 核心)
#   self_qa         25%  - Self-QA 从 FinCorpus 生成 (中文原创)
#   alpaca_zh       15%  - Finance-Alpaca 翻译后 (英→中)
#   sujet_zh        10%  - Sujet-Finance 翻译后 (英→中)
#   sentiment       10%  - FinGPT 情感分析 (任务多样性补充)
#
# 注意: BAAI 数据集已含中英双语，其中的中文部分兼顾"通用中文"防遗忘

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROCESSED="${PROJECT_ROOT}/data/sft/processed"
OUTPUT="${PROCESSED}/merged_sft.jsonl"

echo "============================================"
echo "  EcoGPT: Merge SFT Datasets"
echo "============================================"

python "${PROJECT_ROOT}/scripts/data_processing/merge_sft_data.py" \
    --sources \
        "baai_finance:${PROCESSED}/baai_finance.jsonl:0.40" \
        "self_qa:${PROCESSED}/self_qa.jsonl:0.25" \
        "alpaca_zh:${PROCESSED}/finance_alpaca_zh.jsonl:0.15" \
        "sujet_zh:${PROCESSED}/sujet_finance_zh.jsonl:0.10" \
        "sentiment:${PROCESSED}/fingpt_sentiment.jsonl:0.10" \
    --total 200000 \
    --output "${OUTPUT}" \
    --seed 42

echo ""
echo "Merged SFT data: ${OUTPUT}"
echo "Next steps:"
echo "  1. bash scripts/run_dedup_check.sh"
echo "  2. bash scripts/run_data_filter.sh"
echo "  3. bash scripts/run_step0_data_prepare.sh  (decontaminate + split)"
