#!/bin/bash
# ============================================================
# EcoGPT Step 3: Full Benchmark Evaluation
# ============================================================
# Benchmark plan (aligned with XuanYuan evaluation):
#
# Part 1 - Financial Knowledge:
#   3.1  FinanceIQ (7173 questions, 10 categories)  [主指标]
#   3.2  CEVAL Financial (3 subjects)
#
# Part 2 - General Capability (Regression Test):
#   3.3  CEVAL General (7 subjects)
#   3.4  CMMLU (full, ~11K questions)
#
# Part 3 - GRPO Specific:
#   3.5  Financial Calculation Accuracy (274 questions)
#
# All benchmarks run on: base / SFT / GRPO

set -e

export HF_ENDPOINT="https://hf-mirror.com"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/outputs/eval_results"
mkdir -p "${RESULTS_DIR}"

# ---- Models ----
BASE_MODEL="${PROJECT_ROOT}/models/base/Qwen3-4B"
SFT_MODEL="${PROJECT_ROOT}/models/sft_merged"
GRPO_MODEL="${PROJECT_ROOT}/models/grpo_merged"

MODELS=("base:${BASE_MODEL}" "sft:${SFT_MODEL}" "grpo:${GRPO_MODEL}")

echo "============================================"
echo "  EcoGPT Step 3: Full Benchmark Evaluation"
echo "============================================"
echo "  Base:  ${BASE_MODEL}"
echo "  SFT:   ${SFT_MODEL}"
echo "  GRPO:  ${GRPO_MODEL}"
echo ""

# ============================================================
# 3.1 FinanceIQ (Main Financial Benchmark)
# ============================================================
echo ""
echo "[3.1] FinanceIQ (7173 questions, 10 financial exam categories)"
FINANCEIQ_DIR="${PROJECT_ROOT}/data/eval/financeiq"
if [ -d "${FINANCEIQ_DIR}" ] && [ "$(ls -A ${FINANCEIQ_DIR} 2>/dev/null)" ]; then
    for entry in "${MODELS[@]}"; do
        MODEL_NAME="${entry%%:*}"
        MODEL_PATH="${entry##*:}"
        echo "  Evaluating ${MODEL_NAME}: ${MODEL_PATH}"
        if [ -d "${MODEL_PATH}" ]; then
            python "${PROJECT_ROOT}/scripts/evaluation/eval_financeiq.py" \
                --model "${MODEL_PATH}" \
                --data "${FINANCEIQ_DIR}" \
                --output "${RESULTS_DIR}/financeiq_${MODEL_NAME}.json" \
                --device cuda:0 \
                2>&1 | tee "${RESULTS_DIR}/financeiq_${MODEL_NAME}.log" \
                || echo "  [WARN] FinanceIQ eval failed for ${MODEL_NAME}"
        else
            echo "  [SKIP] Model not found: ${MODEL_PATH}"
        fi
    done
else
    echo "  [SKIP] FinanceIQ data not found. Run:"
    echo "    git clone https://github.com/Duxiaoman-DI/XuanYuan.git /tmp/xuanyuan"
    echo "    cp -r /tmp/xuanyuan/FinanceIQ/data/* data/eval/financeiq/"
fi

# ============================================================
# 3.2 CEVAL Financial Subset
# ============================================================
echo ""
echo "[3.2] CEVAL Financial (accountant, economics, business admin)"
for entry in "${MODELS[@]}"; do
    MODEL_NAME="${entry%%:*}"
    MODEL_PATH="${entry##*:}"
    echo "  Evaluating ${MODEL_NAME}"
    if [ -d "${MODEL_PATH}" ]; then
        lm_eval --model hf \
            --model_args "pretrained=${MODEL_PATH},trust_remote_code=True" \
            --tasks ceval-valid_accountant,ceval-valid_college_economics,ceval-valid_business_administration \
            --device cuda:0 \
            --batch_size 1 \
            --output_path "${RESULTS_DIR}/ceval_finance_${MODEL_NAME}" \
            2>&1 | tee "${RESULTS_DIR}/ceval_finance_${MODEL_NAME}.log" \
            || echo "  [WARN] CEVAL finance failed for ${MODEL_NAME}"
    else
        echo "  [SKIP] ${MODEL_PATH}"
    fi
done

# ============================================================
# 3.3 CEVAL General (Regression Test)
# ============================================================
echo ""
echo "[3.3] CEVAL General (7 subjects, regression test)"
for entry in "${MODELS[@]}"; do
    MODEL_NAME="${entry%%:*}"
    MODEL_PATH="${entry##*:}"
    echo "  Evaluating ${MODEL_NAME}"
    if [ -d "${MODEL_PATH}" ]; then
        lm_eval --model hf \
            --model_args "pretrained=${MODEL_PATH},trust_remote_code=True" \
            --tasks ceval-valid_college_physics,ceval-valid_college_chemistry,ceval-valid_advanced_mathematics,ceval-valid_computer_architecture,ceval-valid_chinese_language_and_literature,ceval-valid_high_school_history,ceval-valid_law \
            --device cuda:0 \
            --batch_size 1 \
            --output_path "${RESULTS_DIR}/ceval_general_${MODEL_NAME}" \
            2>&1 | tee "${RESULTS_DIR}/ceval_general_${MODEL_NAME}.log" \
            || echo "  [WARN] CEVAL general failed for ${MODEL_NAME}"
    else
        echo "  [SKIP] ${MODEL_PATH}"
    fi
done

# ============================================================
# 3.4 CMMLU (Full General Capability)
# ============================================================
echo ""
echo "[3.4] CMMLU (full, ~11K questions, general capability)"
for entry in "${MODELS[@]}"; do
    MODEL_NAME="${entry%%:*}"
    MODEL_PATH="${entry##*:}"
    echo "  Evaluating ${MODEL_NAME}"
    if [ -d "${MODEL_PATH}" ]; then
        lm_eval --model hf \
            --model_args "pretrained=${MODEL_PATH},trust_remote_code=True" \
            --tasks cmmlu \
            --device cuda:0 \
            --batch_size 1 \
            --output_path "${RESULTS_DIR}/cmmlu_${MODEL_NAME}" \
            2>&1 | tee "${RESULTS_DIR}/cmmlu_${MODEL_NAME}.log" \
            || echo "  [WARN] CMMLU failed for ${MODEL_NAME}"
    else
        echo "  [SKIP] ${MODEL_PATH}"
    fi
done

# ============================================================
# 3.5 Math Reasoning (GSM8K + MGSM-zh)
# ============================================================
echo ""
echo "[3.5] Math Reasoning (GSM8K + MGSM-zh)"
for entry in "${MODELS[@]}"; do
    MODEL_NAME="${entry%%:*}"
    MODEL_PATH="${entry##*:}"
    echo "  Evaluating ${MODEL_NAME}"
    if [ -d "${MODEL_PATH}" ]; then
        lm_eval --model hf \
            --model_args "pretrained=${MODEL_PATH},trust_remote_code=True" \
            --tasks gsm8k,mgsm_direct_zh \
            --device cuda:0 \
            --batch_size 1 \
            --output_path "${RESULTS_DIR}/math_${MODEL_NAME}" \
            2>&1 | tee "${RESULTS_DIR}/math_${MODEL_NAME}.log" \
            || echo "  [WARN] Math eval failed for ${MODEL_NAME}"
    else
        echo "  [SKIP] ${MODEL_PATH}"
    fi
done

# ============================================================
# 3.6 GRPO Financial Calculation Accuracy
# ============================================================
echo ""
echo "[3.6] Financial Calculation Accuracy (GRPO validation set)"
GRPO_VAL="${PROJECT_ROOT}/data/grpo/val/valid.jsonl"
if [ -f "${GRPO_VAL}" ]; then
    for entry in "${MODELS[@]}"; do
        MODEL_NAME="${entry%%:*}"
        MODEL_PATH="${entry##*:}"
        # Skip base model (not trained on this format)
        if [ "${MODEL_NAME}" = "base" ]; then
            continue
        fi
        echo "  Evaluating ${MODEL_NAME}"
        if [ -d "${MODEL_PATH}" ]; then
            python "${PROJECT_ROOT}/scripts/evaluation/eval_grpo_calc.py" \
                --model "${MODEL_PATH}" \
                --data "${GRPO_VAL}" \
                --output "${RESULTS_DIR}/grpo_calc_${MODEL_NAME}.json" \
                --device cuda:0 \
                2>&1 | tee "${RESULTS_DIR}/grpo_calc_${MODEL_NAME}.log" \
                || echo "  [WARN] GRPO calc eval failed for ${MODEL_NAME}"
        else
            echo "  [SKIP] ${MODEL_PATH}"
        fi
    done
else
    echo "  [SKIP] GRPO validation data not found: ${GRPO_VAL}"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "============================================"
echo "  Full Benchmark Evaluation Complete!"
echo "============================================"
echo "  Results directory: ${RESULTS_DIR}/"
echo ""
echo "  Benchmarks completed:"
echo "    3.1  FinanceIQ        (financial knowledge, 7173 questions)"
echo "    3.2  CEVAL Financial  (financial knowledge, 3 subjects)"
echo "    3.3  CEVAL General    (regression test, 7 subjects)"
echo "    3.4  CMMLU            (general capability, ~11K questions)"
echo "    3.5  GSM8K + MGSM-zh  (math reasoning)"
echo "    3.6  GRPO Calculation (financial reasoning, 274 questions)"
echo "============================================"
