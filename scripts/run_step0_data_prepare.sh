#!/bin/bash
# ============================================================
# EcoGPT Step 0: Data Preparation Pipeline (Full)
# ============================================================
# 完整的数据处理流水线，从原始数据到可训练数据。
# 按顺序执行每一步。
#
# 前置条件:
#   - 已执行 run_download_data.sh 下载原始数据
#   - 已执行 run_translate.sh 翻译英文数据 (如需)
#   - 已执行 run_self_qa.sh 生成 Self-QA 数据 (如需)
#   - 评测数据已放入 data/eval/ 目录

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data"
SCRIPTS="${PROJECT_ROOT}/scripts"
PROCESSED="${DATA_DIR}/sft/processed"
# Auto-detect tokenizer: prefer Qwen3-4B, fallback to others
if [ -d "${PROJECT_ROOT}/models/base/Qwen3-4B" ]; then
    TOKENIZER="${PROJECT_ROOT}/models/base/Qwen3-4B"
elif [ -d "${PROJECT_ROOT}/models/base/Qwen2.5-7B-Instruct" ]; then
    TOKENIZER="${PROJECT_ROOT}/models/base/Qwen2.5-7B-Instruct"
else
    TOKENIZER=""
fi

mkdir -p "${PROCESSED}"

echo "============================================"
echo "  EcoGPT Step 0: Data Preparation"
echo "============================================"

# --- Step 0.1: BAAI 数据集质量过滤 ---
echo ""
echo "[Step 0.1] Filtering BAAI IndustryInstruction by quality scores..."
echo "           deita_score >= 5.0, rw_score >= 0.0"
python "${SCRIPTS}/data_processing/filter_baai.py" \
    --input "${DATA_DIR}/sft/raw/baai_finance" \
    --output "${PROCESSED}/baai_finance.jsonl" \
    --min_deita_score 5.0 \
    --min_rw_score 0.0 \
    --min_turns 2 \
    --min_answer_len 20

# --- Step 0.2: 格式转换 (如有 Alpaca 格式数据) ---
echo ""
echo "[Step 0.2] Converting Alpaca format to conversations..."
if [ -f "${DATA_DIR}/sft/raw/finance_alpaca/data.json" ]; then
    python "${SCRIPTS}/data_processing/apaca2conversation.py" \
        --input "${DATA_DIR}/sft/raw/finance_alpaca/data.json" \
        --output_dir "${PROCESSED}"
else
    echo "  [SKIP] No Alpaca data found, skipping conversion"
fi

# --- Step 0.3: 多源合并 (按配比采样) ---
echo ""
echo "[Step 0.3] Merging SFT datasets with target ratios..."
python "${SCRIPTS}/data_processing/merge_sft_data.py" \
    --sources \
        "baai_finance:${PROCESSED}/baai_finance.jsonl:0.35" \
        "self_qa:${PROCESSED}/self_qa.jsonl:0.20" \
        "alpaca_zh:${PROCESSED}/finance_alpaca_zh.jsonl:0.20" \
        "sujet_zh:${PROCESSED}/sujet_finance_zh.jsonl:0.05" \
        "general_zh:${PROCESSED}/general_zh.jsonl:0.20" \
    --total 50000 \
    --output "${PROCESSED}/merged_sft.jsonl" \
    --seed 42

# --- Step 0.4: 去重 (精确去重 + 分析报告) ---
echo ""
echo "[Step 0.4] Deduplicating..."
# 精确去重: 按 human 问题内容去重，保留第一条
python -c "
import json, hashlib, re, sys

def normalize(text):
    return re.sub(r'\s+', '', text.lower().strip())

seen = set()
kept = 0
total = 0
with open('${PROCESSED}/merged_sft.jsonl', 'r', encoding='utf-8') as fin, \
     open('${PROCESSED}/merged_sft_dedup.jsonl', 'w', encoding='utf-8') as fout:
    for line in fin:
        total += 1
        item = json.loads(line.strip())
        convs = item.get('conversations', [])
        human = next((m['value'] for m in convs if m.get('from') == 'human'), '')
        h = hashlib.sha1(normalize(human).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            fout.write(line)
            kept += 1
print(f'Dedup: {total} -> {kept} ({total - kept} duplicates removed)')
"
mv "${PROCESSED}/merged_sft_dedup.jsonl" "${PROCESSED}/merged_sft.jsonl"

# 去重分析报告（可选）
python "${SCRIPTS}/data_processing/check2.py" \
    --data "${PROCESSED}/merged_sft.jsonl" \
    --out_dir "${PROJECT_ROOT}/outputs/dup_report" \
    --threshold 0.9

# --- Step 0.5: Token 长度过滤 ---
echo ""
echo "[Step 0.5] Filtering by token length..."
if [ -n "${TOKENIZER}" ] && [ -d "${TOKENIZER}" ]; then
    python "${SCRIPTS}/data_processing/data_filter.py" \
        --input "${PROCESSED}/merged_sft.jsonl" \
        --output "${PROCESSED}/merged_sft_filtered.jsonl" \
        --tokenizer "${TOKENIZER}" \
        --max_total_tokens 2048 \
        --min_total_tokens 10
else
    echo "  [SKIP] Tokenizer not found at ${TOKENIZER}, skipping length filter"
    cp "${PROCESSED}/merged_sft.jsonl" "${PROCESSED}/merged_sft_filtered.jsonl"
fi

# --- Step 0.6: 评测集去污染 ---
echo ""
echo "[Step 0.6] Decontaminating against eval benchmarks..."
python "${SCRIPTS}/data_processing/decontaminate.py" \
    --train_dir "${PROCESSED}/merged_sft_filtered.jsonl" \
    --eval_dirs "${DATA_DIR}/eval/financeiq,${DATA_DIR}/eval/fineval,${DATA_DIR}/eval/ceval_finance,${DATA_DIR}/eval/disc_fin_eval" \
    --threshold 0.7 \
    --output_dir "${PROCESSED}" \
    --report_path "${DATA_DIR}/sft/contamination_report.json"

# --- Step 0.7: 训练/验证集划分 ---
echo ""
echo "[Step 0.7] Splitting train/val (95/5)..."
python "${SCRIPTS}/data_processing/data_split.py" \
    --in "${PROCESSED}/train_clean.jsonl" \
    --out_dir "${DATA_DIR}/sft" \
    --valid_ratio 0.05 \
    --seed 42

# --- Summary ---
echo ""
echo "============================================"
echo "  Step 0 Complete!"
echo "============================================"
TRAIN_COUNT=$(wc -l < "${DATA_DIR}/sft/train.jsonl" 2>/dev/null || echo "?")
VAL_COUNT=$(wc -l < "${DATA_DIR}/sft/valid.jsonl" 2>/dev/null || echo "?")
echo "  Train samples:    ${TRAIN_COUNT}"
echo "  Val samples:      ${VAL_COUNT}"
echo "  Contamination:    ${DATA_DIR}/sft/contamination_report.json"
echo "  Dup report:       ${PROJECT_ROOT}/outputs/dup_report/"
echo ""
echo "  Next: bash scripts/run_step1_sft.sh"
