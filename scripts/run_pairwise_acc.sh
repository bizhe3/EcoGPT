#!/bin/bash
# ============================================================
# EcoGPT - Reward Model Pairwise Accuracy
# ============================================================
# Evaluate reward model on chosen/rejected pairs.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Paths (EDIT THESE) ----
RM_PATH="${PROJECT_ROOT}/models/reward_model/merged"
VALID_DIR="${PROJECT_ROOT}/data/grpo/val"
MAX_LENGTH=1024

echo "============================================"
echo "  EcoGPT: Reward Model Pairwise Accuracy"
echo "============================================"
echo "  RM:       ${RM_PATH}"
echo "  Data:     ${VALID_DIR}"
echo "  Max Len:  ${MAX_LENGTH}"
echo ""

python "${PROJECT_ROOT}/scripts/evaluation/pairwise_acc.py" \
    --rm_path "${RM_PATH}" \
    --valid_dir "${VALID_DIR}" \
    --max_length ${MAX_LENGTH}
