#!/bin/bash
# ============================================================
# EcoGPT - SFT Data Clustering Analysis
# ============================================================
# Auto cluster + Qwen3-14B labeling for SFT training data.
#
# Usage:
#   bash scripts/run_cluster_analysis.sh           # Default (Qwen3-14B labels)
#   bash scripts/run_cluster_analysis.sh --no-llm  # Use TF-IDF keywords (faster)

set -e

# ---- HuggingFace China mirror ----
export HF_ENDPOINT="https://hf-mirror.com"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- Paths ----
INPUT="${PROJECT_ROOT}/data/sft/train/train.jsonl"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/cluster_analysis"
EMBED_MODEL="${PROJECT_ROOT}/models/embed/bge-large-zh-v1.5"
LLM_MODEL="${PROJECT_ROOT}/models/base/Qwen3-14B"

# ---- Auto-detect GPU count ----
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
else
    GPU_COUNT=0
fi

if [ "${GPU_COUNT}" -ge 2 ]; then
    TP_SIZE=2
elif [ "${GPU_COUNT}" -eq 1 ]; then
    TP_SIZE=1
else
    TP_SIZE=1
    echo "[WARN] No GPU detected. BGE will fall back to CPU (slow)."
fi

# ---- Parse args ----
USE_LLM=true
for arg in "$@"; do
    case $arg in
        --no-llm) USE_LLM=false; shift ;;
        --tp=*) TP_SIZE="${arg#*=}"; shift ;;
    esac
done

# ---- Sanity checks ----
if [ ! -f "${INPUT}" ]; then
    echo "[ERROR] Input file not found: ${INPUT}"
    echo "        Run Step 0 data preparation first: bash scripts/run_step0_data_prepare.sh"
    exit 1
fi

if [ ! -d "${EMBED_MODEL}" ]; then
    echo "[INFO] BGE model not found at ${EMBED_MODEL}"
    echo "       Downloading from ${HF_ENDPOINT}..."
    huggingface-cli download BAAI/bge-large-zh-v1.5 \
        --local-dir "${EMBED_MODEL}" || {
        echo "[ERROR] Failed to download embedding model"
        exit 1
    }
fi

if [ "${USE_LLM}" = true ] && [ ! -d "${LLM_MODEL}" ]; then
    echo "[WARN] LLM model not found at ${LLM_MODEL}"
    echo "       Falling back to TF-IDF keyword labeling"
    USE_LLM=false
fi

mkdir -p "${OUTPUT_DIR}"

# ---- Print config ----
echo "============================================"
echo "  EcoGPT: SFT Data Clustering Analysis"
echo "============================================"
echo "  Input:        ${INPUT}"
echo "  Output:       ${OUTPUT_DIR}"
echo "  Embed model:  ${EMBED_MODEL}"
if [ "${USE_LLM}" = true ]; then
    echo "  LLM labels:   ${LLM_MODEL} (TP=${TP_SIZE})"
else
    echo "  LLM labels:   disabled (using TF-IDF keywords)"
fi
echo "  GPU count:    ${GPU_COUNT}"
echo "  HF endpoint:  ${HF_ENDPOINT}"
echo "============================================"
echo ""

# ---- Check Python deps ----
python -c "import hdbscan, umap, sklearn, sentence_transformers" 2>/dev/null || {
    echo "[ERROR] Missing Python dependencies. Install with:"
    echo "        pip install sentence-transformers umap-learn hdbscan jieba scikit-learn matplotlib"
    exit 1
}

# ---- Run ----
set -o pipefail   # propagate failure through pipe
if [ "${USE_LLM}" = true ]; then
    python "${PROJECT_ROOT}/scripts/data_processing/cluster_analysis.py" \
        --input "${INPUT}" \
        --output_dir "${OUTPUT_DIR}" \
        --embed_model "${EMBED_MODEL}" \
        --llm_model "${LLM_MODEL}" \
        --tensor_parallel_size "${TP_SIZE}" \
        2>&1 | tee "${OUTPUT_DIR}/run.log"
else
    python "${PROJECT_ROOT}/scripts/data_processing/cluster_analysis.py" \
        --input "${INPUT}" \
        --output_dir "${OUTPUT_DIR}" \
        --embed_model "${EMBED_MODEL}" \
        --llm_model none \
        2>&1 | tee "${OUTPUT_DIR}/run.log"
fi

PYTHON_EXIT=$?
if [ "${PYTHON_EXIT}" -ne 0 ]; then
    echo ""
    echo "[ERROR] Cluster analysis failed with exit code ${PYTHON_EXIT}"
    echo "        See log: ${OUTPUT_DIR}/run.log"
    exit "${PYTHON_EXIT}"
fi

echo ""
echo "============================================"
echo "  Cluster analysis complete!"
echo "============================================"
echo "  Results:"
echo "    - JSON report:   ${OUTPUT_DIR}/cluster_report.json"
echo "    - Markdown:      ${OUTPUT_DIR}/cluster_summary.md"
echo "    - Visualization: ${OUTPUT_DIR}/cluster_umap.png"
echo "    - Run log:       ${OUTPUT_DIR}/run.log"
echo ""
echo "  Quick view:"
echo "    cat ${OUTPUT_DIR}/cluster_summary.md"
echo "============================================"
