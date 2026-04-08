#!/bin/bash
# ============================================================
# EcoGPT - Self-QA Data Generation (14B ×2 实例, 2×A100 80G)
# ============================================================
# 全程 14B 方案: 翻译 + Self-QA 都用 14B, 节省时间
# 预计耗时: ~2.4h (5K段落 → ~15K QA, 双实例并行)
#
# 注意: 14B Self-QA 幻觉率较高 (~15-20%)
# 脚本会在生成后自动运行 filter_self_qa.py 做一致性过滤
#
# Prerequisites:
#   1. FinCorpus fin_exam 已下载并解压
#   2. gunzip data/sft/raw/fincorpus/fin_exam.jsonl.gz

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROCESSED="${PROJECT_ROOT}/data/sft/processed"
mkdir -p "${PROCESSED}"

INPUT="${PROJECT_ROOT}/data/sft/raw/fincorpus/fin_exam.jsonl"
OUTPUT="${PROCESSED}/self_qa_raw.jsonl"
FILTERED="${PROCESSED}/self_qa.jsonl"

MODEL="Qwen/Qwen2.5-14B-Instruct"
MODE="local_vllm"
TP=1
GPU_UTIL=0.90

echo "============================================"
echo "  EcoGPT: Self-QA Generation (14B ×2)"
echo "============================================"
echo "  Input:  ${INPUT}"
echo "  Model:  ${MODEL}"
echo ""

# =============================================
# Phase 1: 生成 (双实例并行)
# =============================================
# 将语料分成两半, 各占一张卡
HALF=$(( 5000 / 2 ))

echo "[Phase 1] Generating QA pairs (2 instances parallel)..."

CUDA_VISIBLE_DEVICES=0 python "${PROJECT_ROOT}/scripts/data_processing/self_qa_generate.py" \
    --input "${INPUT}" \
    --output "${PROCESSED}/self_qa_raw_part0.jsonl" \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --text_field "text" \
    --num_questions 3 \
    --max_samples ${HALF} \
    --include_reasoning \
    --temperature 0.7 \
    --batch_size 32 \
    --tensor_parallel ${TP} \
    --gpu_utilization ${GPU_UTIL} \
    --seed 42 &
PID_0=$!

CUDA_VISIBLE_DEVICES=1 python "${PROJECT_ROOT}/scripts/data_processing/self_qa_generate.py" \
    --input "${INPUT}" \
    --output "${PROCESSED}/self_qa_raw_part1.jsonl" \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --text_field "text" \
    --num_questions 3 \
    --max_samples ${HALF} \
    --include_reasoning \
    --temperature 0.7 \
    --batch_size 32 \
    --tensor_parallel ${TP} \
    --gpu_utilization ${GPU_UTIL} \
    --seed 1337 &
PID_1=$!

echo "  Instance 0 (GPU 0): PID ${PID_0}"
echo "  Instance 1 (GPU 1): PID ${PID_1}"

FAIL=0
wait ${PID_0} || { echo "[FAIL] Instance 0 failed"; FAIL=1; }
wait ${PID_1} || { echo "[FAIL] Instance 1 failed"; FAIL=1; }

if [ ${FAIL} -ne 0 ]; then
    echo "[ERROR] Generation failed"
    exit 1
fi

# 合并两部分
cat "${PROCESSED}/self_qa_raw_part0.jsonl" "${PROCESSED}/self_qa_raw_part1.jsonl" > "${OUTPUT}"
rm -f "${PROCESSED}/self_qa_raw_part0.jsonl" "${PROCESSED}/self_qa_raw_part1.jsonl"

RAW_COUNT=$(wc -l < "${OUTPUT}")
echo ""
echo "  Raw QA pairs: ${RAW_COUNT}"

# =============================================
# Phase 2: 一致性过滤 (14B 幻觉补偿)
# =============================================
echo ""
echo "[Phase 2] Filtering by answer-source consistency..."
python "${PROJECT_ROOT}/scripts/data_processing/filter_self_qa.py" \
    --input "${OUTPUT}" \
    --output "${FILTERED}" \
    --min_overlap 0.3 \
    --min_answer_len 30 \
    --min_question_len 10

CLEAN_COUNT=$(wc -l < "${FILTERED}")
echo ""
echo "============================================"
echo "  Self-QA complete!"
echo "============================================"
echo "  Raw generated:  ${RAW_COUNT}"
echo "  After filter:   ${CLEAN_COUNT}"
echo "  Output:         ${FILTERED}"

# GRPO reasoning data
REASONING_0="${PROCESSED}/self_qa_raw_part0_reasoning.jsonl"
REASONING_1="${PROCESSED}/self_qa_raw_part1_reasoning.jsonl"
REASONING="${PROCESSED}/self_qa_reasoning.jsonl"
if [ -f "${REASONING_0}" ] || [ -f "${REASONING_1}" ]; then
    cat "${REASONING_0}" "${REASONING_1}" > "${REASONING}" 2>/dev/null
    rm -f "${REASONING_0}" "${REASONING_1}"
    GRPO_COUNT=$(wc -l < "${REASONING}" 2>/dev/null || echo "0")
    echo "  GRPO reasoning: ${GRPO_COUNT} pairs → ${REASONING}"
fi

echo ""
echo "Next: bash scripts/run_step0_data_prepare.sh"
