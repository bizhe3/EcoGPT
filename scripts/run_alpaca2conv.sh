#!/bin/bash
# ============================================================
# EcoGPT - Alpaca to Conversations Format Converter
# ============================================================
# Convert Alpaca-style json/jsonl to ShareGPT conversations format.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Paths (EDIT THESE) ----
INPUT="${PROJECT_ROOT}/data/sft/raw/finance_alpaca.json"
OUTPUT_DIR="${PROJECT_ROOT}/data/sft/processed"

echo "============================================"
echo "  EcoGPT: Alpaca -> Conversations"
echo "============================================"
echo "  Input:  ${INPUT}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

python "${PROJECT_ROOT}/scripts/data_processing/apaca2conversation.py" \
    --input "${INPUT}" \
    --output_dir "${OUTPUT_DIR}"
