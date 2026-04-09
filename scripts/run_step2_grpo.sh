#!/bin/bash
# ============================================================
# EcoGPT Step 2: GRPO (Group Relative Policy Optimization)
# ============================================================
# Financial reasoning RL with verifiable rewards.
# Requires: SFT merged model from Step 1 + sanity check pass.

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Paths (EDIT THESE) ----
SFT_MODEL="${PROJECT_ROOT}/models/sft_merged"
TRAIN_DATA="${PROJECT_ROOT}/data/grpo/train"
VAL_DATA="${PROJECT_ROOT}/data/grpo/val"
OUTPUT_DIR="${PROJECT_ROOT}/models/grpo_lora"
LOG_DIR="${PROJECT_ROOT}/logs/grpo"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "============================================"
echo "  EcoGPT Step 2: GRPO Training"
echo "============================================"
echo "  SFT Model: ${SFT_MODEL}"
echo "  GRPO Data: ${TRAIN_DATA}"
echo "  Output:    ${OUTPUT_DIR}"
echo ""

python "${PROJECT_ROOT}/scripts/training/grpo_training.py" \
    --model_name_or_path "${SFT_MODEL}" \
    --train_data_path "${TRAIN_DATA}" \
    --val_data_path "${VAL_DATA}" \
    --output_dir "${OUTPUT_DIR}" \
    --use_peft True \
    --lora_rank 8 \
    --lora_alpha 16.0 \
    --lora_dropout 0.05 \
    --num_generations 8 \
    --temperature 0.9 \
    --use_vllm True \
    --max_completion_length 1024 \
    --beta 0.04 \
    --loss_type grpo \
    --learning_rate 5e-6 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_train_epochs 2 \
    --max_grad_norm 0.5 \
    --bf16 True \
    --logging_steps 10 \
    --save_steps 100 \
    --logging_dir "${LOG_DIR}" \
    --report_to tensorboard \
    --format_reward_weight 1.0 \
    --accuracy_reward_weight 2.0 \
    --length_reward_weight 0.5

echo ""
echo "GRPO training complete! LoRA saved to: ${OUTPUT_DIR}"
echo ""
echo "Next step: Merge GRPO LoRA with:"
echo "  python scripts/utils/merge_lora.py \\"
echo "      --base_model ${SFT_MODEL} \\"
echo "      --adapter ${OUTPUT_DIR} \\"
echo "      --output ${PROJECT_ROOT}/models/grpo_merged"
