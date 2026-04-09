#!/bin/bash
# ============================================================
# EcoGPT - Build GRPO Training Data
# ============================================================
# 1. Extract short answers from Self-QA reasoning (LLM-based)
# 2. Generate new financial calculation QA pairs (LLM-based)
# 3. Merge, dedup, split into train/val
#
# Uses Qwen3-14B for both extraction and generation.
# Dual GPU: extraction on GPU 0, then generation on GPU 0.
# Estimated time: ~30 min (1.2K extract + 5K generate)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="${PROJECT_ROOT}/models/base/Qwen3-14B"
INPUT="${PROJECT_ROOT}/data/sft/processed/self_qa_reasoning.jsonl"
OUTPUT="${PROJECT_ROOT}/data/grpo/train/grpo_all.jsonl"
NUM_GENERATE=5000

echo "============================================"
echo "  EcoGPT: Build GRPO Data"
echo "============================================"
echo "  Model:    ${MODEL}"
echo "  Extract:  ${INPUT}"
echo "  Generate: ${NUM_GENERATE} new QA pairs"
echo "  Output:   ${OUTPUT}"
echo ""

mkdir -p "${PROJECT_ROOT}/data/grpo/train" "${PROJECT_ROOT}/data/grpo/val"

CUDA_VISIBLE_DEVICES=0 python "${PROJECT_ROOT}/scripts/data_processing/build_grpo_data.py" \
    --mode both \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --model "${MODEL}" \
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
