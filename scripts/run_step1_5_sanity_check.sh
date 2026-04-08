#!/bin/bash
# ============================================================
# EcoGPT Step 1.5: Sanity Check (SFT → GRPO transition)
# ============================================================
# Validates that the SFT model has sufficient capability for GRPO.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Paths (EDIT THESE) ----
SFT_MODEL="${PROJECT_ROOT}/models/sft_merged"
GRPO_DATA="${PROJECT_ROOT}/data/grpo/train"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/sanity_check"

mkdir -p "${OUTPUT_DIR}"

echo "============================================"
echo "  EcoGPT Step 1.5: Sanity Check"
echo "============================================"
echo "  SFT Model: ${SFT_MODEL}"
echo "  GRPO Data: ${GRPO_DATA}"
echo ""

python "${PROJECT_ROOT}/scripts/evaluation/sanity_check.py" \
    --model_path "${SFT_MODEL}" \
    --data_path "${GRPO_DATA}" \
    --num_samples 200 \
    --num_generations 8 \
    --temperature 0.9 \
    --max_tokens 1024 \
    --output_dir "${OUTPUT_DIR}"

echo ""
echo "Check report: ${OUTPUT_DIR}/sanity_check_report.json"
echo ""
echo "Decision guide:"
echo "  pass@8 < 10%  → Need Reasoning SFT first"
echo "  10% ~ 80%     → Ready for GRPO (Step 2)"
echo "  > 80%         → GRPO data too easy, use harder data"
