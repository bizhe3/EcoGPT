#!/bin/bash
# ============================================================
# EcoGPT - Merge SFT Datasets with Target Ratios
# ============================================================
# Combine multiple financial SFT datasets into one training file.
#
# Recommended ratio (v2 plan):
#   consulting  25%  - DISC-FIN-SFT consulting subset
#   task        35%  - DISC-FIN-SFT task + CFData subset
#   computing   20%  - DISC-FIN-SFT computing + Fin-R1
#   retrieval   10%  - DISC-FIN-SFT retrieval
#   general     10%  - General Chinese instruction (prevent forgetting)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROCESSED="${PROJECT_ROOT}/data/sft/processed"
OUTPUT="${PROCESSED}/merged_sft.jsonl"

echo "============================================"
echo "  EcoGPT: Merge SFT Datasets"
echo "============================================"

python "${PROJECT_ROOT}/scripts/data_processing/merge_sft_data.py" \
    --sources \
        "consulting:${PROCESSED}/disc_consulting.jsonl:0.25" \
        "task:${PROCESSED}/disc_task.jsonl:0.35" \
        "computing:${PROCESSED}/disc_computing.jsonl:0.20" \
        "retrieval:${PROCESSED}/disc_retrieval.jsonl:0.10" \
        "general:${PROCESSED}/general_zh.jsonl:0.10" \
    --total 250000 \
    --output "${OUTPUT}" \
    --seed 42

echo ""
echo "Merged SFT data: ${OUTPUT}"
echo "Next: run decontamination, then train/val split"
