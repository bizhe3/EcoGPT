#!/bin/bash
# ============================================================
# EcoGPT - Self-QA Data Generation
# ============================================================
# Generate financial instruction data from raw corpus using LLM.
# Inspired by XuanYuan's Self-QA methodology.
#
# Prerequisites:
#   1. Download FinCorpus: huggingface-cli download Duxiaoman-DI/FinCorpus --repo-type dataset
#   2. Or prepare your own financial text corpus as jsonl with "text" field
#   3. Start a local vLLM server, or use local_hf / api mode

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Paths (EDIT THESE) ----
INPUT="${PROJECT_ROOT}/data/sft/raw/fincorpus_sample.jsonl"
OUTPUT="${PROJECT_ROOT}/data/sft/processed/self_qa.jsonl"
MODEL="Qwen/Qwen2.5-72B-Instruct"  # or local path

# Generation mode: local_vllm (fastest), local_hf (no vllm needed), api (remote)
MODE="local_vllm"

# API settings (only for api mode)
API_BASE="http://localhost:8000/v1"
API_KEY="dummy"

echo "============================================"
echo "  EcoGPT: Self-QA Data Generation"
echo "============================================"
echo "  Input:  ${INPUT}"
echo "  Output: ${OUTPUT}"
echo "  Model:  ${MODEL}"
echo "  Mode:   ${MODE}"
echo ""

python "${PROJECT_ROOT}/scripts/data_processing/self_qa_generate.py" \
    --input "${INPUT}" \
    --output "${OUTPUT}" \
    --model "${MODEL}" \
    --mode "${MODE}" \
    --api_base "${API_BASE}" \
    --api_key "${API_KEY}" \
    --text_field "text" \
    --num_questions 3 \
    --max_samples 5000 \
    --include_reasoning \
    --temperature 0.7 \
    --batch_size 32 \
    --seed 42
