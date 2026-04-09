#!/bin/bash
# ============================================================
# EcoGPT - Build GRPO Training Data
# ============================================================
# Uses DeepSeek-R1-Distill-Qwen-14B for everything:
#   1. Extract short answers from Self-QA reasoning
#   2. Generate new financial calculation QA pairs
#   3. Self-verify: re-solve each problem, keep only consistent answers
#   4. Merge, dedup, split into train/val
#
# Single model = no GPU switching, faster, more consistent.
# Estimated time: ~15 min

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${PROJECT_ROOT}/models/base/DeepSeek-R1-Distill-Qwen-14B"
INPUT="${PROJECT_ROOT}/data/sft/processed/self_qa_reasoning.jsonl"
OUTPUT="${PROJECT_ROOT}/data/grpo/train/grpo_all.jsonl"
NUM_GENERATE=10000

echo "============================================"
echo "  EcoGPT: Build GRPO Data (R1 Distill)"
echo "============================================"
echo "  Model:    ${MODEL}"
echo "  Extract:  ${INPUT}"
echo "  Generate: ${NUM_GENERATE} new QA pairs"
echo "  Verify:   self-verification (same model re-solves)"
echo "  Output:   ${OUTPUT}"
echo ""

mkdir -p "${PROJECT_ROOT}/data/grpo/train" "${PROJECT_ROOT}/data/grpo/val"

CUDA_VISIBLE_DEVICES=0 python "${PROJECT_ROOT}/scripts/data_processing/build_grpo_data.py" \
    --mode both \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --model "${MODEL}" \
    --self_verify \
    --num_samples ${NUM_GENERATE} \
    --batch_size 256 \
    --tensor_parallel 1 \
    --gpu_utilization 0.9 \
    --seed 42

TOTAL=$(wc -l < "${OUTPUT}")
echo ""
echo "Total GRPO samples: ${TOTAL}"

# Split train/val
echo ""
echo "Splitting train/val (95/5)..."
python "${PROJECT_ROOT}/scripts/data_processing/data_split.py" \
    --in "${OUTPUT}" \
    --out_dir "${PROJECT_ROOT}/data/grpo" \
    --valid_ratio 0.05 \
    --seed 42

TRAIN_COUNT=$(wc -l < "${PROJECT_ROOT}/data/grpo/train.jsonl" 2>/dev/null || echo "?")
VAL_COUNT=$(wc -l < "${PROJECT_ROOT}/data/grpo/valid.jsonl" 2>/dev/null || echo "?")

# Move into subdirectories for grpo_training.py
mv "${PROJECT_ROOT}/data/grpo/train.jsonl" "${PROJECT_ROOT}/data/grpo/train/train.jsonl" 2>/dev/null || true
mv "${PROJECT_ROOT}/data/grpo/valid.jsonl" "${PROJECT_ROOT}/data/grpo/val/valid.jsonl" 2>/dev/null || true

echo ""
echo "============================================"
echo "  GRPO Data Ready!"
echo "============================================"
echo "  Train: ${TRAIN_COUNT} samples"
echo "  Val:   ${VAL_COUNT} samples"
echo ""
echo "  Next: bash scripts/run_step1_5_sanity_check.sh"
echo "  Then: bash scripts/run_step2_grpo.sh"
