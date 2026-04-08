#!/bin/bash
# ============================================================
# EcoGPT Step 1: SFT (Supervised Fine-Tuning)
# ============================================================
# Financial domain SFT with LoRA on Qwen2.5-7B-Instruct
# Optimized hyperparameters per v2 plan.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Paths (EDIT THESE) ----
MODEL_PATH="${PROJECT_ROOT}/models/base/Qwen2.5-7B-Instruct"
TRAIN_DIR="${PROJECT_ROOT}/data/sft/train"
VAL_DIR="${PROJECT_ROOT}/data/sft/val"
OUTPUT_DIR="${PROJECT_ROOT}/models/sft_lora"
LOG_DIR="${PROJECT_ROOT}/logs/sft"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "============================================"
echo "  EcoGPT Step 1: SFT Training"
echo "============================================"
echo "  Model:  ${MODEL_PATH}"
echo "  Train:  ${TRAIN_DIR}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

python "${PROJECT_ROOT}/scripts/training/supervised_finetuning.py" \
    --model_name_or_path "${MODEL_PATH}" \
    --template_name qwen \
    --train_file_dir "${TRAIN_DIR}" \
    --validation_file_dir "${VAL_DIR}" \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --model_max_length 2048 \
    --logging_steps 10 \
    --eval_steps 200 \
    --save_steps 200 \
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
