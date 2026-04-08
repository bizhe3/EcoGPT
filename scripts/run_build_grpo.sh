#!/bin/bash
# ============================================================
# EcoGPT - Build GRPO Training Data
# ============================================================
# Convert financial QA sources into GRPO format:
#   {"prompt": "...", "ground_truth": "..."}
#
# Input sources:
#   - Fin-R1-Data (financial reasoning with answers)
#   - DISC-FIN-SFT computing subset (calculation with answers)
#   - Self-QA reasoning pairs (generated)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RAW="${PROJECT_ROOT}/data/grpo/raw"
OUTPUT="${PROJECT_ROOT}/data/grpo/processed/grpo_all.jsonl"

echo "============================================"
echo "  EcoGPT: Build GRPO Data"
echo "============================================"

python "${PROJECT_ROOT}/scripts/data_processing/build_grpo_data.py" \
    --inputs \
        "${RAW}/fin_r1_data.jsonl" \
        "${RAW}/disc_computing.jsonl" \
    --output "${OUTPUT}" \
    --format auto \
    --max_gt_length 200 \
    --min_gt_length 1 \
    --seed 42

echo ""
echo "GRPO data built: ${OUTPUT}"
echo "Next: split into train/val"

python "${PROJECT_ROOT}/scripts/data_processing/data_split.py" \
    --in "${OUTPUT}" \
    --out_dir "${PROJECT_ROOT}/data/grpo" \
    --valid_ratio 0.05 \
    --seed 42

echo "Split complete:"
echo "  Train: ${PROJECT_ROOT}/data/grpo/train.jsonl"
echo "  Val:   ${PROJECT_ROOT}/data/grpo/valid.jsonl"
