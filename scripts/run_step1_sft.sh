#!/bin/bash
# ============================================================
# EcoGPT Step 1: SFT (Supervised Fine-Tuning)
# ============================================================
# Financial domain SFT with LoRA on Qwen3-4B
# 2×RTX PRO 6000 96GB, ~42K training samples

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Paths ----
MODEL_PATH="${PROJECT_ROOT}/models/base/Qwen3-4B"
# data_split.py outputs train.jsonl/valid.jsonl directly in data/sft/
# supervised_finetuning.py scans directories for *.jsonl files
TRAIN_DIR="${PROJECT_ROOT}/data/sft"
VAL_DIR="${PROJECT_ROOT}/data/sft"
OUTPUT_DIR="${PROJECT_ROOT}/models/sft_lora"
LOG_DIR="${PROJECT_ROOT}/logs/sft"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# Move data files into subdirectories so train/val don't mix
mkdir -p "${PROJECT_ROOT}/data/sft/train" "${PROJECT_ROOT}/data/sft/val"
if [ -f "${PROJECT_ROOT}/data/sft/train.jsonl" ]; then
    mv "${PROJECT_ROOT}/data/sft/train.jsonl" "${PROJECT_ROOT}/data/sft/train/train.jsonl"
fi
if [ -f "${PROJECT_ROOT}/data/sft/valid.jsonl" ]; then
    mv "${PROJECT_ROOT}/data/sft/valid.jsonl" "${PROJECT_ROOT}/data/sft/val/valid.jsonl"
fi

TRAIN_DIR="${PROJECT_ROOT}/data/sft/train"
VAL_DIR="${PROJECT_ROOT}/data/sft/val"

echo "============================================"
echo "  EcoGPT Step 1: SFT Training (Qwen3-4B)"
echo "============================================"
echo "  Model:  ${MODEL_PATH}"
echo "  Train:  ${TRAIN_DIR}"
echo "  Val:    ${VAL_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

python "${PROJECT_ROOT}/scripts/training/supervised_finetuning.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --template_name qwen \
    --train_file_dir "${TRAIN_DIR}" \
    --validation_file_dir "${VAL_DIR}" \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 2 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --model_max_length 2048 \
    --logging_steps 10 \
    --eval_steps 500 \
    --save_steps 500 \
    --eval_strategy steps \
    --save_strategy steps \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --bf16 True \
    --gradient_checkpointing True \
    --use_peft True \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all \
    --torch_dtype bfloat16 \
    --output_dir "${OUTPUT_DIR}" \
    --logging_dir "${LOG_DIR}" \
    --report_to tensorboard \
    --trust_remote_code True

echo ""
echo "SFT training complete! LoRA saved to: ${OUTPUT_DIR}"
echo ""
echo "Next step: Merge LoRA with:"
echo "  python scripts/utils/merge_lora.py \\"
echo "      --base_model ${MODEL_PATH} \\"
echo "      --adapter ${OUTPUT_DIR} \\"
echo "      --output ${PROJECT_ROOT}/models/sft_merged"
