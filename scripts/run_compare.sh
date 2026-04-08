#!/bin/bash
# ============================================================
# EcoGPT - Compare Base vs SFT/GRPO Model
# ============================================================
# Sample validation data and generate side-by-side answers
# from base model and fine-tuned model.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Paths (EDIT THESE) ----
BASE_MODEL="${PROJECT_ROOT}/models/base/Qwen2.5-7B-Instruct"
LORA_MODEL="${PROJECT_ROOT}/models/sft_merged"
VALID_FILE="${PROJECT_ROOT}/data/sft/val/valid.jsonl"
OUTPUT="${PROJECT_ROOT}/outputs/eval_results/base_vs_sft.jsonl"
SYSTEM_PROMPT="你是一个专业的金融分析助手。"

# Sampling
N_RANDOM=15
N_HARD=5
HARD_POOL=200
SEED=42

# Generation
MAX_NEW_TOKENS=1024
REPETITION_PENALTY=1.1
NO_REPEAT_NGRAM=3

echo "============================================"
echo "  EcoGPT: Compare Base vs Fine-tuned"
echo "============================================"
echo "  Base:   ${BASE_MODEL}"
echo "  LoRA:   ${LORA_MODEL}"
echo "  Valid:  ${VALID_FILE}"
echo "  Output: ${OUTPUT}"
echo ""

mkdir -p "$(dirname "${OUTPUT}")"

python "${PROJECT_ROOT}/scripts/evaluation/compare.py" \
    --base_model "${BASE_MODEL}" \
    --lora_model "${LORA_MODEL}" \
    --valid_file "${VALID_FILE}" \
    --output "${OUTPUT}" \
    --system_prompt "${SYSTEM_PROMPT}" \
    --seed ${SEED} \
    --n_random ${N_RANDOM} \
    --n_hard ${N_HARD} \
    --hard_pool ${HARD_POOL} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --repetition_penalty ${REPETITION_PENALTY} \
    --no_repeat_ngram_size ${NO_REPEAT_NGRAM}
