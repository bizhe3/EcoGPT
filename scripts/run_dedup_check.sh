#!/bin/bash
# ============================================================
# EcoGPT - Deduplication Analysis
# ============================================================
# Run exact + approximate (MinHash LSH) deduplication analysis.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Paths (EDIT THESE) ----
DATA="${PROJECT_ROOT}/data/sft/processed/train.jsonl"
OUT_DIR="${PROJECT_ROOT}/outputs/dup_report"
THRESHOLD=0.9
NUM_PERM=128

echo "============================================"
echo "  EcoGPT: Deduplication Analysis"
echo "============================================"
echo "  Data:      ${DATA}"
echo "  Output:    ${OUT_DIR}"
echo "  Threshold: ${THRESHOLD}"
echo ""

python "${PROJECT_ROOT}/scripts/data_processing/check2.py" \
    --data "${DATA}" \
    --out_dir "${OUT_DIR}" \
    --threshold ${THRESHOLD} \
    --num_perm ${NUM_PERM}
