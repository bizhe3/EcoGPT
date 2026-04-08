#!/bin/bash
# ============================================================
# EcoGPT - Download All Required Datasets
# ============================================================
# Downloads training data and evaluation benchmarks.
# Requires: pip install huggingface_hub
#
# NOTE: Some datasets may need manual download or access approval.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ============================================================
# HuggingFace 中国镜像配置
# 如果在中国大陆，建议启用镜像加速（取消下面两行注释）
# 也可在 ~/.bashrc 中永久设置
# ============================================================
export HF_ENDPOINT="https://hf-mirror.com"
# export HF_HUB_ENABLE_HF_TRANSFER=1  # 可选：安装 hf_transfer 后启用多线程下载

echo "============================================"
echo "  EcoGPT: Download Datasets"
echo "============================================"
echo "  HF_ENDPOINT: ${HF_ENDPOINT}"
echo ""
echo "  如需切回官方源，注释掉脚本中的 export HF_ENDPOINT 行"
echo "  如需使用其他镜像："
echo "    https://hf-mirror.com     (推荐)"
echo "    https://hf-api.gitee.com  (Gitee 镜像)"
echo ""

# ---- SFT Training Data ----

echo ""
echo "[1/4] DISC-FIN-SFT (~246K financial instructions)"
echo "      Source: HuggingFace eggbiscuit/DISC-FIN-SFT"
huggingface-cli download eggbiscuit/DISC-FIN-SFT \
    --repo-type dataset \
    --local-dir "${PROJECT_ROOT}/data/sft/raw/disc_fin_sft" \
    || echo "  [WARN] Download failed. Try: pip install -U huggingface_hub, or visit ${HF_ENDPOINT}/datasets/eggbiscuit/DISC-FIN-SFT"

echo ""
echo "[2/4] FinCorpus (~60GB, for Self-QA generation)"
echo "      Source: HuggingFace Duxiaoman-DI/FinCorpus"
echo "      NOTE: Large dataset - downloading a sample only"
echo "      For full dataset, run:"
echo "        huggingface-cli download Duxiaoman-DI/FinCorpus --repo-type dataset --local-dir data/sft/raw/fincorpus"
echo "      Skipping auto-download (too large). Place manually if needed."

# ---- GRPO Training Data ----

echo ""
echo "[3/4] GRPO source data"
echo "      Fin-R1-Data: Check https://arxiv.org/abs/2503.16252 for download link"
echo "      Place files in: ${PROJECT_ROOT}/data/grpo/raw/"
echo "      Expected format: jsonl with 'prompt' and 'ground_truth' fields"

# ---- Evaluation Benchmarks ----

echo ""
echo "[4/4] Evaluation Benchmarks"

echo "  [4a] FinanceIQ (XuanYuan benchmark)"
echo "       Check: ${HF_ENDPOINT}/Duxiaoman-DI for FinanceIQ dataset"
echo "       Place in: ${PROJECT_ROOT}/data/eval/financeiq/"

echo "  [4b] CEVAL will be auto-downloaded by lm-eval harness"

echo "  [4c] DISC-FIN-Eval"
echo "       Check: https://github.com/FudanDISC/DISC-FinLLM for eval data"
echo "       Place in: ${PROJECT_ROOT}/data/eval/disc_fin_eval/"

# ---- Base Model ----

echo ""
echo "============================================"
echo "  Base Model"
echo "============================================"
echo "Download Qwen2.5-7B-Instruct (镜像已通过 HF_ENDPOINT 自动生效):"
echo "  huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir models/base/Qwen2.5-7B-Instruct"
echo ""
echo "Or for 14B variant:"
echo "  huggingface-cli download Qwen/Qwen2.5-14B-Instruct --local-dir models/base/Qwen2.5-14B-Instruct"
echo ""
echo "如果 huggingface-cli 下载慢，也可以用 modelscope 下载:"
echo "  pip install modelscope"
echo "  modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir models/base/Qwen2.5-7B-Instruct"

echo ""
echo "============================================"
echo "  Download Summary"
echo "============================================"
echo "After downloading, your data/ directory should look like:"
echo ""
echo "  data/sft/raw/"
echo "    disc_fin_sft/        <- DISC-FIN-SFT dataset"
echo "    fincorpus/           <- FinCorpus (optional, for Self-QA)"
echo ""
echo "  data/grpo/raw/"
echo "    fin_r1_data.jsonl    <- Fin-R1-Data"
echo ""
echo "  data/eval/"
echo "    financeiq/           <- FinanceIQ benchmark"
echo "    disc_fin_eval/       <- DISC-FIN-Eval"
echo ""
echo "  models/base/"
echo "    Qwen2.5-7B-Instruct/ <- Base model"
