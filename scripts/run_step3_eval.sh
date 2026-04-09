#!/bin/bash
# ============================================================
# EcoGPT Step 3: Evaluation Pipeline
# ============================================================
# Run all benchmarks on baseline, SFT, and SFT+GRPO models.

set -e

# HuggingFace 中国镜像
export HF_ENDPOINT="https://hf-mirror.com"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/outputs/eval_results"
mkdir -p "${RESULTS_DIR}"

# ---- Models to evaluate (EDIT THESE) ----
BASE_MODEL="${PROJECT_ROOT}/models/base/Qwen3-4B"
SFT_MODEL="${PROJECT_ROOT}/models/sft_merged"
GRPO_MODEL="${PROJECT_ROOT}/models/grpo_merged"

echo "============================================"
echo "  EcoGPT Step 3: Evaluation"
echo "============================================"

# --- 3.1 CEVAL Financial Subset ---
echo ""
echo "[3.1] CEVAL Financial Subset"
for MODEL_NAME in "base" "sft" "grpo"; do
    eval MODEL_PATH=\$${MODEL_NAME^^}_MODEL 2>/dev/null || true
    case $MODEL_NAME in
        base) MODEL_PATH="${BASE_MODEL}" ;;
        sft)  MODEL_PATH="${SFT_MODEL}" ;;
        grpo) MODEL_PATH="${GRPO_MODEL}" ;;
    esac

    echo "  Evaluating ${MODEL_NAME}: ${MODEL_PATH}"
    if [ -d "${MODEL_PATH}" ]; then
        lm_eval --model hf \
            --model_args "pretrained=${MODEL_PATH},trust_remote_code=True" \
            --tasks ceval-valid_accountant,ceval-valid_college_economics,ceval-valid_business_administration \
            --device cuda:0 \
            --batch_size 1 \
            --output_path "${RESULTS_DIR}/ceval_finance_${MODEL_NAME}" \
            2>&1 | tee "${RESULTS_DIR}/ceval_finance_${MODEL_NAME}.log" || echo "  [WARN] lm_eval failed for ${MODEL_NAME}"
    else
        echo "  [SKIP] Model not found: ${MODEL_PATH}"
    fi
done

# --- 3.2 CEVAL General (Regression Test) ---
echo ""
echo "[3.2] CEVAL General Regression Test"
for MODEL_NAME in "base" "sft" "grpo"; do
    case $MODEL_NAME in
        base) MODEL_PATH="${BASE_MODEL}" ;;
        sft)  MODEL_PATH="${SFT_MODEL}" ;;
        grpo) MODEL_PATH="${GRPO_MODEL}" ;;
    esac

    echo "  Evaluating ${MODEL_NAME}: ${MODEL_PATH}"
    if [ -d "${MODEL_PATH}" ]; then
        lm_eval --model hf \
            --model_args "pretrained=${MODEL_PATH},trust_remote_code=True" \
            --tasks ceval-valid_college_physics,ceval-valid_college_chemistry,ceval-valid_advanced_mathematics,ceval-valid_computer_architecture,ceval-valid_chinese_language_and_literature,ceval-valid_high_school_history,ceval-valid_law \
            --device cuda:0 \
            --batch_size 1 \
            --output_path "${RESULTS_DIR}/ceval_general_${MODEL_NAME}" \
            2>&1 | tee "${RESULTS_DIR}/ceval_general_${MODEL_NAME}.log" || echo "  [WARN] lm_eval failed for ${MODEL_NAME}"
    else
        echo "  [SKIP] Model not found: ${MODEL_PATH}"
    fi
done

# --- 3.3 PPL Evaluation ---
echo ""
echo "[3.3] Perplexity Evaluation"
EVAL_DATA="${PROJECT_ROOT}/data/sft/val/valid.jsonl"
if [ -f "${EVAL_DATA}" ]; then
    for MODEL_NAME in "base" "sft" "grpo"; do
        case $MODEL_NAME in
            base) MODEL_PATH="${BASE_MODEL}" ;;
            sft)  MODEL_PATH="${SFT_MODEL}" ;;
            grpo) MODEL_PATH="${GRPO_MODEL}" ;;
        esac

        if [ -d "${MODEL_PATH}" ]; then
            echo "  PPL for ${MODEL_NAME}..."
            python "${PROJECT_ROOT}/scripts/evaluation/eval_ppl_sft_jsonl.py" \
                --model "${MODEL_PATH}" \
                --data "${EVAL_DATA}" \
                --template qwen \
                --system "你是一个专业的金融分析助手。" \
                --mode sliding \
                2>&1 | tee "${RESULTS_DIR}/ppl_${MODEL_NAME}.log" || echo "  [WARN] PPL eval failed for ${MODEL_NAME}"
        fi
    done
else
    echo "  [SKIP] No eval data found at ${EVAL_DATA}"
fi

echo ""
echo "============================================"
echo "  Evaluation complete!"
echo "  Results: ${RESULTS_DIR}/"
echo "============================================"
