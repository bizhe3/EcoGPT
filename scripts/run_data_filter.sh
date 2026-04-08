#!/bin/bash
# ============================================================
# EcoGPT - Token Length Filter
# ============================================================
# Filter SFT data by token count (prompt + completion).

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Paths (EDIT THESE) ----
INPUT="${PROJECT_ROOT}/data/sft/processed/train.jsonl"
OUTPUT="${PROJECT_ROOT}/data/sft/processed/train_filtered.jsonl"
TOKENIZER="${PROJECT_ROOT}/models/base/Qwen2.5-7B-Instruct"
MAX_TOKENS=2048
MIN_TOKENS=1

echo "============================================"
echo "  EcoGPT: Token Length Filter"
echo "============================================"
echo "  Input:     ${INPUT}"
echo "  Output:    ${OUTPUT}"
echo "  Tokenizer: ${TOKENIZER}"
echo "  Range:     [${MIN_TOKENS}, ${MAX_TOKENS}]"
echo ""

python "${PROJECT_ROOT}/scripts/data_processing/data_filter.py" \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --tokenizer "${TOKENIZER}" \
    --max_total_tokens ${MAX_TOKENS} \
    --min_total_tokens ${MIN_TOKENS}
