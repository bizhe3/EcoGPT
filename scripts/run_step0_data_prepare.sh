#!/bin/bash
# ============================================================
# EcoGPT Step 0: Data Preparation Pipeline
# ============================================================
# This script processes raw financial SFT data into training format.
# Run each step sequentially.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
SCRIPTS="${PROJECT_ROOT}/scripts"

echo "============================================"
echo "  EcoGPT Step 0: Data Preparation"
echo "============================================"

# --- Step 0.1: Format conversion (Alpaca → conversations) ---
echo "[Step 0.1] Converting Alpaca format to conversations..."
python "${SCRIPTS}/data_processing/apaca2conversation.py" \
    --input "${DATA_DIR}/sft/raw/your_alpaca_data.json" \
    --output_dir "${DATA_DIR}/sft/processed"

# --- Step 0.2: Deduplication ---
echo "[Step 0.2] Running deduplication..."
# Edit check2.py DATA_PATH before running, or use the decontaminate script
# python "${SCRIPTS}/data_processing/check2.py"

# --- Step 0.3: Token length filtering ---
echo "[Step 0.3] Filtering by token length..."
# Edit data_filter.py paths before running
# python "${SCRIPTS}/data_processing/data_filter.py"

# --- Step 0.4: Decontamination (train vs eval) ---
echo "[Step 0.4] Decontaminating against eval benchmarks..."
python "${SCRIPTS}/data_processing/decontaminate.py" \
    --train_dir "${DATA_DIR}/sft/processed" \
    --eval_dirs "${DATA_DIR}/eval/financeiq,${DATA_DIR}/eval/fineval,${DATA_DIR}/eval/ceval_finance,${DATA_DIR}/eval/disc_fin_eval" \
    --threshold 0.7 \
    --output_dir "${DATA_DIR}/sft/processed" \
    --report_path "${DATA_DIR}/sft/contamination_report.json"

# --- Step 0.5: Train/Val split ---
echo "[Step 0.5] Splitting train/val..."
python "${SCRIPTS}/data_processing/data_split.py" \
    --in "${DATA_DIR}/sft/processed/train_clean.jsonl" \
    --out_dir "${DATA_DIR}/sft" \
    --valid_ratio 0.05 \
    --seed 42

echo ""
echo "Data preparation complete!"
echo "  Train: ${DATA_DIR}/sft/train/"
echo "  Val:   ${DATA_DIR}/sft/val/"
echo "  Report: ${DATA_DIR}/sft/contamination_report.json"
