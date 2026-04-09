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
# Cross-validation: verify answers with a different model
# ============================================================

VERIFY_PROMPT = "请计算以下金融题目的答案，只输出最终数值结果（如：72、11580元、-5、13.56万元、15.3%），不要输出推理过程或其他内容。\n\n题目：{prompt}"


def answer_matches(a: str, b: str) -> bool:
    """Check if two answers match (numerical or text)."""
    a = a.strip().replace(",", "").replace(" ", "")
    b = b.strip().replace(",", "").replace(" ", "")

    # Exact match
    if a == b:
        return True

    # Try numerical match with tolerance
    def extract_num(s):
        s = s.replace("万元", "").replace("亿元", "").replace("元", "")
        s = s.replace("%", "").replace("$", "").replace("¥", "")
        s = s.replace("万", "").replace("亿", "")
        try:
            return float(s)
        except ValueError:
            return None

    na, nb = extract_num(a), extract_num(b)
    if na is not None and nb is not None:
        if abs(na - nb) < 0.5 or (abs(na - nb) / max(abs(nb), 0.01)) < 0.05:
            return True

    # Containment match
    if a in b or b in a:
        return True

    return False


def cross_validate(items: List[dict], verify_model_path: str, tp: int, gpu_util: float):
    """Use a different model to re-solve each problem, keep only consistent answers."""
    from vllm import SamplingParams

    logger.info(f"Cross-validating {len(items)} items with {verify_model_path}")

    # Release generation model GPU memory first
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()

    # Load verification model
    verify_llm, verify_tokenizer = init_vllm(verify_model_path, tp, gpu_util)

    prompts = []
    for item in items:
        user_msg = VERIFY_PROMPT.format(prompt=item["prompt"])
        prompts.append(build_chat_prompt(verify_tokenizer, user_msg))

    params = SamplingParams(temperature=0, max_tokens=50)
    outputs = verify_llm.generate(prompts, params)

    verified = []
    mismatch = 0
    for item, out in zip(items, outputs):
        verify_answer = out.outputs[0].text.strip()
        # Strip thinking tags
        if "</think>" in verify_answer:
            verify_answer = verify_answer.split("</think>")[-1].strip()

        if answer_matches(item["ground_truth"], verify_answer):
            verified.append(item)
        else:
            mismatch += 1

    logger.info(f"Cross-validation: {len(items)} -> {len(verified)} "
                f"({mismatch} mismatches removed, {len(verified)/len(items):.1%} kept)")

    # Clean up verify model
    del verify_llm
    gc.collect()
    torch.cuda.empty_cache()

    return verified


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
    ap.add_argument("--model", required=True, help="LLM model path (for generation)")
    ap.add_argument("--verify_model", default=None,
                    help="Verification model path (e.g. DeepSeek-R1-Distill). If set, cross-validates answers.")
    ap.add_argument("--num_samples", type=int, default=10000, help="Number of QA pairs to generate")
    ap.add_argument("--batch_size", type=int, default=256, help="vLLM batch size for generation")
    ap.add_argument("--tensor_parallel", type=int, default=1)
    ap.add_argument("--gpu_utilization", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Phase 1: Generate with primary model
    logger.info(f"Loading generation model: {args.model}")
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

    # Release generation model
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Released generation model from GPU")

    # Dedup before verification (save compute)
    all_items = dedup(all_items)

    # Phase 2: Cross-validate with verification model
    before_verify = len(all_items)
    if args.verify_model and os.path.exists(args.verify_model):
        all_items = cross_validate(
            all_items, args.verify_model,
            args.tensor_parallel, args.gpu_utilization,
        )
    else:
        if args.verify_model:
            logger.warning(f"Verify model not found: {args.verify_model}, skipping cross-validation")

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 50}")
    print(f"  GRPO Data Build Report")
    print(f"{'=' * 50}")
    print(f"  Before verify: {before_verify}")
    print(f"  After verify:  {len(all_items)}")
    print(f"  Pass rate:     {len(all_items)/max(before_verify,1):.1%}")
    print(f"  Output:        {args.output}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
