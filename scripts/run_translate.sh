#!/bin/bash
# ============================================================
# EcoGPT - Translate English Financial Data to Chinese
# ============================================================
# Translates finance-alpaca and Sujet-Finance datasets.
# Requires a local LLM (Qwen2.5-72B recommended).
#
# 如果没有 72B 模型的 GPU 资源，可以：
#   1. 用 Qwen2.5-7B-Instruct (质量较低但可用)
#   2. 启动 vLLM server 后用 api 模式
#   3. 用第三方翻译 API

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- 配置 (EDIT THESE) ----
MODEL="Qwen/Qwen2.5-72B-Instruct"   # 或本地路径
MODE="local_vllm"                     # local_vllm / local_hf / api
API_BASE="http://localhost:8000/v1"   # api 模式用
BATCH_SIZE=64

echo "============================================"
echo "  EcoGPT: Translate EN → ZH"
echo "============================================"

# 1. Finance-Alpaca (69K)
echo ""
echo "[1/2] Translating finance-alpaca..."
python "${PROJECT_ROOT}/scripts/data_processing/translate_to_zh.py" \
    --input "${PROJECT_ROOT}/data/sft/raw/finance_alpaca" \
    --output "${PROJECT_ROOT}/data/sft/processed/finance_alpaca_zh.jsonl" \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --api_base "${API_BASE}" \
    --batch_size ${BATCH_SIZE} \
    --max_samples 30000

# 2. Sujet-Finance (177K → 取 30K)
echo ""
echo "[2/2] Translating Sujet-Finance (sampling 30K)..."
python "${PROJECT_ROOT}/scripts/data_processing/translate_to_zh.py" \
    --input "${PROJECT_ROOT}/data/sft/raw/sujet_finance" \
    --output "${PROJECT_ROOT}/data/sft/processed/sujet_finance_zh.jsonl" \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --api_base "${API_BASE}" \
    --batch_size ${BATCH_SIZE} \
    --max_samples 30000

echo ""
echo "Translation complete!"
echo "  ${PROJECT_ROOT}/data/sft/processed/finance_alpaca_zh.jsonl"
echo "  ${PROJECT_ROOT}/data/sft/processed/sujet_finance_zh.jsonl"
