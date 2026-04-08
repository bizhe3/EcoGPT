#!/bin/bash
# ============================================================
# EcoGPT - Download All Required Datasets
# ============================================================
# Downloads training data and evaluation benchmarks.
# Requires: pip install huggingface_hub
#
# 数据可用性核查 (2026-04):
#   DISC-FIN-SFT   → 仅样本 (1.72MB), 不可用于训练
#   CFData-sft     → 仅样本, 不可用
#   Fin-R1-Data    → 未公开发布
#   ICE-FIND       → 仅评测子集
#
# 实际可用数据源:
#   ✅ BAAI/IndustryInstruction_Finance-Economics  (123K, 中英双语)
#   ✅ Sujet-Finance-Instruct-177k                 (177K, 英文)
#   ✅ gbharti/finance-alpaca                      (69K, 英文)
#   ✅ FinGPT/fingpt-sentiment-train               (77K, 英文情感)
#   ✅ Duxiaoman-DI/FinCorpus                      (60GB, 中文原始语料)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ============================================================
# HuggingFace 中国镜像配置
# ============================================================
export HF_ENDPOINT="https://hf-mirror.com"
# export HF_HUB_ENABLE_HF_TRANSFER=1  # 可选：pip install hf_transfer 后启用

echo "============================================"
echo "  EcoGPT: Download Datasets"
echo "============================================"
echo "  HF_ENDPOINT: ${HF_ENDPOINT}"
echo "  如需切回官方源，注释掉 export HF_ENDPOINT 行"
echo ""

# ==========================================================
# 1. SFT Training Data (实际可下载的数据集)
# ==========================================================

echo "==============================="
echo " [1/5] BAAI/IndustryInstruction_Finance-Economics"
echo "       123K 条, 中英双语, SFT 核心数据"
echo "==============================="
huggingface-cli download BAAI/IndustryInstruction_Finance-Economics \
    --repo-type dataset \
    --local-dir "${PROJECT_ROOT}/data/sft/raw/baai_finance" \
    || echo "  [WARN] Download failed. Visit: ${HF_ENDPOINT}/datasets/BAAI/IndustryInstruction_Finance-Economics"

echo ""
echo "==============================="
echo " [2/5] Sujet-Finance-Instruct-177k"
echo "       177K 条, 英文金融指令 (需翻译)"
echo "==============================="
huggingface-cli download sujet-ai/Sujet-Finance-Instruct-177k \
    --repo-type dataset \
    --local-dir "${PROJECT_ROOT}/data/sft/raw/sujet_finance" \
    || echo "  [WARN] Download failed. Visit: ${HF_ENDPOINT}/datasets/sujet-ai/Sujet-Finance-Instruct-177k"

echo ""
echo "==============================="
echo " [3/5] gbharti/finance-alpaca"
echo "       69K 条, 英文金融指令 (需翻译)"
echo "==============================="
huggingface-cli download gbharti/finance-alpaca \
    --repo-type dataset \
    --local-dir "${PROJECT_ROOT}/data/sft/raw/finance_alpaca" \
    || echo "  [WARN] Download failed. Visit: ${HF_ENDPOINT}/datasets/gbharti/finance-alpaca"

echo ""
echo "==============================="
echo " [4/5] FinGPT/fingpt-sentiment-train"
echo "       77K 条, 金融情感分析 (任务型补充)"
echo "==============================="
huggingface-cli download FinGPT/fingpt-sentiment-train \
    --repo-type dataset \
    --local-dir "${PROJECT_ROOT}/data/sft/raw/fingpt_sentiment" \
    || echo "  [WARN] Download failed. Visit: ${HF_ENDPOINT}/datasets/FinGPT/fingpt-sentiment-train"

echo ""
echo "==============================="
echo " [5/5] Duxiaoman-DI/FinCorpus"
echo "       ~60GB 中文金融原始语料 (用于 Self-QA 生成)"
echo "       注意: 数据量大, 按需下载子集"
echo "==============================="
echo "  下载全量 (21GB 压缩):"
echo "    huggingface-cli download Duxiaoman-DI/FinCorpus --repo-type dataset --local-dir ${PROJECT_ROOT}/data/sft/raw/fincorpus"
echo ""
echo "  或仅下载金融考试子集 (96MB, 适合快速开始):"
echo "    huggingface-cli download Duxiaoman-DI/FinCorpus --repo-type dataset --include 'fin_exam.jsonl.gz' --local-dir ${PROJECT_ROOT}/data/sft/raw/fincorpus"
echo ""
echo "  跳过自动下载 (太大), 请手动按需执行上述命令"

# ==========================================================
# 2. Evaluation Benchmarks
# ==========================================================

echo ""
echo "============================================"
echo "  Evaluation Benchmarks"
echo "============================================"

echo "  [eval-1] FinanceIQ (XuanYuan benchmark)"
echo "           Visit: ${HF_ENDPOINT}/Duxiaoman-DI"
echo "           Place in: ${PROJECT_ROOT}/data/eval/financeiq/"
echo ""
echo "  [eval-2] CEVAL: auto-downloaded by lm-eval harness"
echo ""
echo "  [eval-3] DISC-FIN-Eval"
echo "           Visit: https://github.com/FudanDISC/DISC-FinLLM"
echo "           Place in: ${PROJECT_ROOT}/data/eval/disc_fin_eval/"

# ==========================================================
# 3. Base Model
# ==========================================================

echo ""
echo "============================================"
echo "  Base Model"
echo "============================================"
echo "  HuggingFace 镜像:"
echo "    huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ${PROJECT_ROOT}/models/base/Qwen2.5-7B-Instruct"
echo ""
echo "  ModelScope 备选:"
echo "    pip install modelscope"
echo "    modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ${PROJECT_ROOT}/models/base/Qwen2.5-7B-Instruct"

# ==========================================================
# Summary
# ==========================================================

echo ""
echo "============================================"
echo "  Download Summary"
echo "============================================"
echo ""
echo "  data/sft/raw/"
echo "    baai_finance/      <- BAAI IndustryInstruction (123K, 中英双语) [核心]"
echo "    sujet_finance/     <- Sujet-Finance-Instruct (177K, 英文, 需翻译)"
echo "    finance_alpaca/    <- Finance-Alpaca (69K, 英文, 需翻译)"
echo "    fingpt_sentiment/  <- FinGPT Sentiment (77K, 情感分类)"
echo "    fincorpus/         <- FinCorpus (中文原始语料, 用于Self-QA)"
echo ""
echo "  data/eval/"
echo "    financeiq/         <- FinanceIQ benchmark"
echo "    disc_fin_eval/     <- DISC-FIN-Eval"
echo ""
echo "  models/base/"
echo "    Qwen2.5-7B-Instruct/"
echo ""
echo "============================================"
echo "  下一步: 数据处理"
echo "============================================"
echo "  1. 英文数据翻译:  python scripts/data_processing/translate_to_zh.py"
echo "  2. Self-QA 生成:  bash scripts/run_self_qa.sh"
echo "  3. 合并+清洗:     bash scripts/run_merge_sft.sh"
