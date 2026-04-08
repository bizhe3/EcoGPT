#!/bin/bash
# ============================================================
# EcoGPT - Filter BAAI IndustryInstruction Finance Dataset
# ============================================================
# Filter by quality scores to keep only high-quality samples.
#
# BAAI 数据集自带两个质量分数:
#   deita_score (2.75~27.7): 指令复杂度 + 回答质量
#   rw_score    (-45~69):    奖励模型偏好分
#
# 默认过滤: deita >= 5.0, rw >= 0.0
# 预计保留 60~80K / 122K (约 50~65%)

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- 配置 ----
INPUT="${PROJECT_ROOT}/data/sft/raw/baai_finance"
OUTPUT="${PROJECT_ROOT}/data/sft/processed/baai_finance.jsonl"

# 质量阈值 (可调整)
MIN_DEITA=5.0     # 过滤低复杂度/低质量回答
MIN_RW=0.0        # 过滤奖励模型不认可的样本

# 语言: zh = 仅中文, en = 仅英文, 不传 = 中英双语都保留
LANG=""  # 建议保留双语, 英文部分也有训练价值

echo "============================================"
echo "  EcoGPT: Filter BAAI Finance Dataset"
echo "============================================"

CMD="python ${PROJECT_ROOT}/scripts/data_processing/filter_baai.py \
    --input ${INPUT} \
    --output ${OUTPUT} \
    --min_deita_score ${MIN_DEITA} \
    --min_rw_score ${MIN_RW} \
    --min_turns 2 \
    --min_answer_len 20"

# 如果指定了语言过滤
if [ -n "${LANG}" ]; then
    CMD="${CMD} --lang ${LANG}"
fi

eval ${CMD}
