#!/bin/bash
# ============================================================
# EcoGPT - Merge Reward Model LoRA
# ============================================================
# Merge LoRA-trained reward model adapter into base model.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Paths (EDIT THESE) ----
BASE_MODEL="${PROJECT_ROOT}/models/base/Qwen2.5-7B-Instruct"
ADAPTER="${PROJECT_ROOT}/models/reward_model/lora"
OUTPUT="${PROJECT_ROOT}/models/reward_model/merged"
DTYPE="bfloat16"

echo "============================================"
echo "  EcoGPT: Merge Reward Model LoRA"
echo "============================================"
echo "  Base:    ${BASE_MODEL}"
echo "  Adapter: ${ADAPTER}"
echo "  Output:  ${OUTPUT}"
echo ""

python "${PROJECT_ROOT}/scripts/utils/merge_rm.py" \
    --base_model "${BASE_MODEL}" \
    --adapter "${ADAPTER}" \
    --output "${OUTPUT}" \
    --dtype "${DTYPE}"
