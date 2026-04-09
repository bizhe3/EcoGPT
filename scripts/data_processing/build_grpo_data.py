#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - Build GRPO training data.

Two modes:
  1. Extract: Clean existing Self-QA reasoning data (extract short answers via LLM)
  2. Generate: Generate new financial calculation QA pairs from scratch

Usage:
    # Extract short answers from Self-QA reasoning
    python build_grpo_data.py \
        --mode extract \
        --input data/sft/processed/self_qa_reasoning.jsonl \
        --output data/grpo/train/reasoning.jsonl \
        --model models/base/Qwen3-14B

    # Generate new calculation QA pairs
    python build_grpo_data.py \
        --mode generate \
        --output data/grpo/train/generated.jsonl \
        --model models/base/Qwen3-14B \
        --num_samples 5000

    # Both: extract + generate, then merge
    python build_grpo_data.py \
        --mode both \
        --input data/sft/processed/self_qa_reasoning.jsonl \
        --output data/grpo/train/grpo_all.jsonl \
        --model models/base/Qwen3-14B \
        --num_samples 5000
"""

import argparse
import hashlib
import json
import os
import random
import re
from typing import List

from loguru import logger


# ============================================================
# Generation prompts (15 financial calculation topics)
# ============================================================

TOPICS = [
    "利润率计算（毛利率、净利率、营业利润率）",
    "增长率计算（同比增长、复合增长率CAGR）",
    "投资回报率（ROI、ROE、ROA）",
    "复利终值/现值计算",
    "债券收益率（到期收益率YTM、当期收益率）",
    "股票估值（PE、PB、股息折现DDM）",
    "财务比率分析（流动比率、速动比率、资产负债率）",
    "期权定价基础（内在价值、时间价值）",
    "折旧计算（直线法、双倍余额递减法）",
    "税费计算（增值税、所得税、印花税）",
    "汇率换算与套利",
    "基差计算（现货价格-期货价格）",
    "久期与凸性基础计算",
    "资本成本（WACC加权平均资本成本）",
    "净现值NPV与内部收益率IRR",
]

GENERATE_PROMPT = """你是一名金融考试出题专家。请生成一道金融计算题，要求：

1. 题目必须包含具体数值，需要数学计算才能得出答案
2. 计算类型：{topic}
3. 题目应清晰明确，答案唯一
4. 难度为大学金融专业水平

请严格按以下格式输出（不要输出其他内容）：
题目：<你出的题目>
答案：<最终数值答案，如 72万元、15.3%、-5>"""

EXTRACT_PROMPT = "从以下文本中提取最终的数值答案，只输出答案本身（如：72、11580元、-5、13.56万元、15.3%），不要输出任何其他内容。\n\n文本：{text}"


# ============================================================
# vLLM helpers
# ============================================================

def init_vllm(model_path: str, tp: int = 1, gpu_util: float = 0.9):
    from vllm import LLM
    llm = LLM(
        model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tp,
        gpu_memory_utilization=gpu_util,
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def build_chat_prompt(tokenizer, user_content: str) -> str:
    messages = [{"role": "user", "content": user_content}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


# ============================================================
# Mode 1: Extract short answers from existing reasoning data
# ============================================================

def extract_answers(args, llm, tokenizer):
    from vllm import SamplingParams

    items = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    logger.info(f"Loaded {len(items)} reasoning samples")

    short_items = []
    long_items = []
    for item in items:
        gt = item.get("ground_truth", "").strip()
        if len(gt) < 30:
            short_items.append(item)
        else:
            long_items.append(item)

    logger.info(f"Short answers (keep as-is): {len(short_items)}")
    logger.info(f"Long answers (need extraction): {len(long_items)}")

    if long_items:
        prompts = [
            build_chat_prompt(tokenizer, EXTRACT_PROMPT.format(text=item["ground_truth"]))
            for item in long_items
        ]
        params = SamplingParams(temperature=0, max_tokens=50)
        outputs = llm.generate(prompts, params)

        for item, out in zip(long_items, outputs):
            answer = out.outputs[0].text.strip()
            if answer and len(answer) < 30:
                short_items.append({
                    "prompt": item["prompt"],
                    "ground_truth": answer,
                })

    logger.info(f"Extracted total: {len(short_items)} samples")
    return short_items


# ============================================================
# Mode 2: Generate new calculation QA pairs
# ============================================================

def generate_qa(args, llm, tokenizer):
    from vllm import SamplingParams

    num_samples = args.num_samples
    batch_size = args.batch_size
    logger.info(f"Generating {num_samples} financial calculation QA pairs...")

    all_prompts = []
    for _ in range(num_samples):
        topic = random.choice(TOPICS)
        user_msg = GENERATE_PROMPT.format(topic=topic)
        all_prompts.append(build_chat_prompt(tokenizer, user_msg))

    params = SamplingParams(temperature=0.8, max_tokens=512, top_p=0.95)

    results = []
    for i in range(0, len(all_prompts), batch_size):
        batch = all_prompts[i:i + batch_size]
        outputs = llm.generate(batch, params)

        for out in outputs:
            text = out.outputs[0].text.strip()
            # Strip thinking tags if present
            if "</think>" in text:
                text = text.split("</think>")[-1].strip()

            # Parse "题目：... 答案：..."
            q_match = re.search(r'题目[：:]\s*(.*?)(?=\n*答案[：:])', text, re.DOTALL)
            a_match = re.search(r'答案[：:]\s*(.*?)(?:\n|$)', text, re.DOTALL)

            if q_match and a_match:
                prompt = q_match.group(1).strip()
                answer = a_match.group(1).strip()
                if len(prompt) > 15 and 0 < len(answer) < 30:
                    results.append({
                        "prompt": prompt,
                        "ground_truth": answer,
                    })

        done = min(i + batch_size, len(all_prompts))
        logger.info(f"Generated {done}/{num_samples}, valid: {len(results)}")

    logger.info(f"Generated total: {len(results)} valid QA pairs")
    return results


# ============================================================
# Dedup
# ============================================================

def dedup(items: List[dict]) -> List[dict]:
    seen = set()
    result = []
    for item in items:
        h = hashlib.sha1(item["prompt"].strip().encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            result.append(item)
    logger.info(f"Dedup: {len(items)} -> {len(result)}")
    return result


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Build GRPO training data")
    ap.add_argument("--mode", choices=["extract", "generate", "both"], default="both")
    ap.add_argument("--input", default=None, help="Input Self-QA reasoning jsonl (for extract mode)")
    ap.add_argument("--output", required=True, help="Output GRPO jsonl")
    ap.add_argument("--model", required=True, help="LLM model path")
    ap.add_argument("--num_samples", type=int, default=5000, help="Number of QA pairs to generate")
    ap.add_argument("--batch_size", type=int, default=256, help="vLLM batch size for generation")
    ap.add_argument("--tensor_parallel", type=int, default=1)
    ap.add_argument("--gpu_utilization", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    logger.info(f"Loading model: {args.model}")
    llm, tokenizer = init_vllm(args.model, args.tensor_parallel, args.gpu_utilization)

    all_items = []

    # Extract
    if args.mode in ("extract", "both"):
        if args.input and os.path.exists(args.input):
            extracted = extract_answers(args, llm, tokenizer)
            all_items.extend(extracted)
        else:
            logger.warning(f"No input file for extraction: {args.input}")

    # Generate
    if args.mode in ("generate", "both"):
        generated = generate_qa(args, llm, tokenizer)
        all_items.extend(generated)

    # Dedup and save
    all_items = dedup(all_items)

    with open(args.output, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 50}")
    print(f"  GRPO Data Build Report")
    print(f"{'=' * 50}")
    print(f"  Total:  {len(all_items)}")
    print(f"  Output: {args.output}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
