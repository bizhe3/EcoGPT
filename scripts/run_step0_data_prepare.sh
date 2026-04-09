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
    --lang zh \
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

# --- Step 0.2.5: 清洗各数据源 ---
echo ""
echo "[Step 0.2.5] Cleaning data sources..."
python -c "
import json, re, sys

THINKING_STARTS = ['嗯', '好的', '用户', '首先', '接下来', '让我', '我现在', '我需要']

def is_dirty(item):
    convs = item.get('conversations', [])
    if len(convs) < 2:
        return True
    human = next((m['value'] for m in convs if m.get('from') == 'human'), '')
    gpt = next((m['value'] for m in convs if m.get('from') == 'gpt'), '')
    # Filter: human starts with thinking keywords
    for kw in THINKING_STARTS:
        if human.strip().startswith(kw + '，') or human.strip().startswith(kw + ','):
            return True
    # Filter: gpt has heavy repetition (same 50-char block appears 3+ times)
    if len(gpt) > 200:
        block = gpt[:50]
        if gpt.count(block) >= 3:
            return True
    # Filter: gpt too short (< 2 chars)
    if len(gpt.strip()) < 2:
        return True
    # Filter: empty human
    if len(human.strip()) < 5:
        return True
    return False

for src in ['self_qa']:
    path = '${PROCESSED}/' + src + '.jsonl'
    try:
        with open(path, 'r', encoding='utf-8') as f:
            items = [json.loads(l) for l in f if l.strip()]
        before = len(items)
        items = [i for i in items if not is_dirty(i)]
        with open(path, 'w', encoding='utf-8') as f:
            for i in items:
                f.write(json.dumps(i, ensure_ascii=False) + '\n')
        print(f'  {src}: {before} -> {len(items)} ({before - len(items)} dirty removed)')
    except FileNotFoundError:
        print(f'  {src}: not found, skipping')
"

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

# --- Step 0.3.5: 全局质量过滤 ---
echo ""
echo "[Step 0.3.5] Global quality filtering..."
python -c "
import json, re

THINKING_STARTS = ['嗯', '好的，我', '用户让我', '首先，我需要', '接下来', '让我', '我现在需要', '我需要']

kept = 0
dropped = 0
with open('${PROCESSED}/merged_sft.jsonl', 'r', encoding='utf-8') as fin, \
     open('${PROCESSED}/merged_sft_clean.jsonl', 'w', encoding='utf-8') as fout:
    for line in fin:
        item = json.loads(line.strip())
        convs = item.get('conversations', [])
        if len(convs) < 2:
            dropped += 1
            continue
        human = next((m['value'] for m in convs if m.get('from') == 'human'), '')
        gpt = next((m['value'] for m in convs if m.get('from') == 'gpt'), '')

        skip = False
        # 1. Human is thinking content
        for kw in THINKING_STARTS:
            if human.strip().startswith(kw):
                if '，' in human[:len(kw)+5] or '。' in human[:len(kw)+5]:
                    skip = True
                    break
        # 2. GPT has heavy repetition
        if not skip and len(gpt) > 300:
            block = gpt[50:100]
            if block and gpt.count(block) >= 3:
                skip = True
        # 3. GPT too short
        if not skip and len(gpt.strip()) < 2:
            skip = True
        # 4. Human too short
        if not skip and len(human.strip()) < 5:
            skip = True
        # 5. Contains thinking tags in GPT
        if not skip and '<think>' in gpt:
            skip = True

        if skip:
            dropped += 1
        else:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
            kept += 1

print(f'  Global filter: kept={kept}, dropped={dropped}')
"
mv "${PROCESSED}/merged_sft_clean.jsonl" "${PROCESSED}/merged_sft.jsonl"

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
