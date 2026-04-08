#!/bin/bash
# ============================================================
# EcoGPT - Translate English Financial Data to Chinese
# ============================================================
# 14B 双实例并行翻译 (2×A100 80G, 各占 1 卡)
# 预计耗时: ~8.3 小时 (60K 条)
#
# 翻译用 14B 原因: 有英文原文锚定, 14B 质量够用
# Self-QA 用 72B 原因: 无原文, 完全靠模型金融知识

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROCESSED="${PROJECT_ROOT}/data/sft/processed"
mkdir -p "${PROCESSED}"

MODEL="Qwen/Qwen2.5-14B-Instruct"
MODE="local_vllm"
BATCH_SIZE=64
TP=1
GPU_UTIL=0.90

echo "============================================"
echo "  EcoGPT: Translate EN → ZH (14B × 2 实例)"
echo "============================================"
echo "  Model:    ${MODEL}"
echo "  Strategy: 2 parallel vLLM instances (GPU 0 & GPU 1)"
echo ""

# 实例 0: finance-alpaca → GPU 0
echo "[Instance 0] finance-alpaca (30K) → GPU 0"
CUDA_VISIBLE_DEVICES=0 python "${PROJECT_ROOT}/scripts/data_processing/translate_to_zh.py" \
    --input "${PROJECT_ROOT}/data/sft/raw/finance_alpaca" \
    --output "${PROCESSED}/finance_alpaca_zh.jsonl" \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --batch_size ${BATCH_SIZE} \
    --tensor_parallel ${TP} \
    --gpu_utilization ${GPU_UTIL} \
    --max_samples 30000 &
PID_0=$!

# 实例 1: Sujet-Finance → GPU 1
echo "[Instance 1] Sujet-Finance (30K) → GPU 1"
CUDA_VISIBLE_DEVICES=1 python "${PROJECT_ROOT}/scripts/data_processing/translate_to_zh.py" \
    --input "${PROJECT_ROOT}/data/sft/raw/sujet_finance" \
    --output "${PROCESSED}/sujet_finance_zh.jsonl" \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --batch_size ${BATCH_SIZE} \
    --tensor_parallel ${TP} \
    --gpu_utilization ${GPU_UTIL} \
    --max_samples 30000 &
PID_1=$!

echo ""
echo "Waiting for both instances..."
FAIL=0
wait ${PID_0} || { echo "[FAIL] Instance 0 failed"; FAIL=1; }
wait ${PID_1} || { echo "[FAIL] Instance 1 failed"; FAIL=1; }

if [ ${FAIL} -eq 0 ]; then
    echo ""
    echo "Translation complete!"
    echo "  ${PROCESSED}/finance_alpaca_zh.jsonl"
    echo "  ${PROCESSED}/sujet_finance_zh.jsonl"
    echo ""
    echo "Next: bash scripts/run_self_qa.sh"
else
    exit 1
fi
