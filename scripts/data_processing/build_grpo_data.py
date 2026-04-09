#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - Build GRPO training data using DeepSeek-R1-Distill.

Single model handles all steps:
  1. Extract: Clean Self-QA reasoning data (extract short answers)
  2. Generate: Create new financial calculation QA pairs
  3. Self-verify: Re-solve each problem, keep only consistent answers

Usage:
    python build_grpo_data.py \
        --mode both \
        --input data/sft/processed/self_qa_reasoning.jsonl \
        --output data/grpo/train/grpo_all.jsonl \
        --model models/base/DeepSeek-R1-Distill-Qwen-14B \
        --self_verify \
        --num_samples 10000
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
# Prompts
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

EXTRACT_PROMPT = "从以下文本中提取最终的数值答案，只输出答案本身（如：72、11580元、-5、13.56万元、15.3%），不要输出任何其他内容。\n\n文本：{text}\n\n答案是："

VERIFY_PROMPT = "请计算以下金融题目的答案，只输出最终数值结果（如：72、11580元、-5、13.56万元、15.3%），不要输出推理过程。\n\n题目：{prompt}\n\n答案是："


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


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> from output."""
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    # If thinking started but didn't finish, extract from tail
    if "<think>" in text and "</think>" not in text:
        nums = re.findall(r'[-+]?\d+\.?\d*\s*(?:万元|亿元|元|%|万|亿)?', text[-200:])
        if nums:
            return nums[-1].strip()
        return ""
    return text.strip()


def batch_generate(llm, tokenizer, user_messages: List[str],
                   temperature: float = 0, max_tokens: int = 512) -> List[str]:
    """Generate responses for a list of user messages."""
    from vllm import SamplingParams

    prompts = [build_chat_prompt(tokenizer, msg) for msg in user_messages]
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                            top_p=0.95 if temperature > 0 else 1.0)
    outputs = llm.generate(prompts, params)
    return [strip_thinking(o.outputs[0].text) for o in outputs]


# ============================================================
# Answer matching
# ============================================================

def extract_number(s: str):
    """Extract the first number from a string, handling Chinese units."""
    s = re.sub(r'[*#\s，。]', '', s)
    match = re.search(r'[-+]?\d*\.?\d+', s)
    if match:
        num = float(match.group())
        after = s[match.end():]
        if '万亿' in after:
            num *= 1e12
        elif '亿' in after:
            num *= 1e8
        elif '万' in after:
            num *= 1e4
        return num
    return None


def answer_matches(a: str, b: str) -> bool:
    """Check if two answers match."""
    if not a or not b:
        return False

    a = re.sub(r'[*#\s]', '', a.strip())
    b = re.sub(r'[*#\s]', '', b.strip())

    # Exact match
    if a == b:
        return True

    # Numerical match with tolerance
    na, nb = extract_number(a), extract_number(b)
    if na is not None and nb is not None:
        if na == nb == 0:
            return True
        if abs(na - nb) < 0.5:
            return True
        if abs(na - nb) / max(abs(nb), 0.01) < 0.05:
            return True

    # Containment match
    if len(a) > 1 and len(b) > 1:
        if a in b or b in a:
            return True

    return False


# ============================================================
# Mode 1: Extract short answers from existing reasoning data
# ============================================================

def extract_answers(items: List[dict], llm, tokenizer) -> List[dict]:
    logger.info(f"Loaded {len(items)} reasoning samples")

    short_items = []
    long_items = []
    for item in items:
        gt = item.get("ground_truth", "").strip()
        if not gt:
            continue
        if len(gt) < 30:
            short_items.append(item)
        else:
            long_items.append(item)

    logger.info(f"Short answers (keep as-is): {len(short_items)}")
    logger.info(f"Long answers (need extraction): {len(long_items)}")

    if long_items:
        from vllm import SamplingParams
        # Use raw prompts to avoid R1 thinking mode
        raw_prompts = [EXTRACT_PROMPT.format(text=item["ground_truth"]) for item in long_items]
        params = SamplingParams(temperature=0, max_tokens=50)
        outputs = llm.generate(raw_prompts, params)
        answers = [strip_thinking(o.outputs[0].text) for o in outputs]

        for item, answer in zip(long_items, answers):
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

def generate_qa(num_samples: int, batch_size: int, llm, tokenizer) -> List[dict]:
    logger.info(f"Generating {num_samples} financial calculation QA pairs...")

    all_messages = []
    for _ in range(num_samples):
        topic = random.choice(TOPICS)
        all_messages.append(GENERATE_PROMPT.format(topic=topic))

    results = []
    for i in range(0, len(all_messages), batch_size):
        batch = all_messages[i:i + batch_size]
        responses = batch_generate(llm, tokenizer, batch, temperature=0.8, max_tokens=512)

        for text in responses:
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

        done = min(i + batch_size, len(all_messages))
        logger.info(f"Generated {done}/{num_samples}, valid: {len(results)}")

    logger.info(f"Generated total: {len(results)} valid QA pairs")
    return results


# ============================================================
# Self-verification: re-solve and compare
# ============================================================

def self_verify(items: List[dict], llm, tokenizer) -> List[dict]:
    """Re-solve each problem with the same model, keep consistent answers."""
    from vllm import SamplingParams

    logger.info(f"Self-verifying {len(items)} items...")

    # Use raw prompts (not chat template) to avoid R1's long thinking
    # The "答案是：" suffix forces the model to output the answer directly
    prompts = [VERIFY_PROMPT.format(prompt=item["prompt"]) for item in items]
    params = SamplingParams(temperature=0, max_tokens=2048)
    outputs = llm.generate(prompts, params)
    answers = [strip_thinking(o.outputs[0].text) for o in outputs]

    verified = []
    mismatch = 0
    mismatch_examples = []

    for item, verify_answer in zip(items, answers):
        gt = item["ground_truth"]
        if not gt or not verify_answer:
            mismatch += 1
            continue

        if answer_matches(gt, verify_answer):
            verified.append(item)
        else:
            mismatch += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append({
                    "prompt": item["prompt"][:60],
                    "gt": gt,
                    "verify": verify_answer[:60],
                })

    logger.info(f"Self-verification: {len(items)} -> {len(verified)} "
                f"({mismatch} mismatches, {len(verified)/max(len(items),1):.1%} kept)")

    if mismatch_examples:
        logger.info("Mismatch examples:")
        for ex in mismatch_examples:
            logger.info(f"  prompt: {ex['prompt']}...")
            logger.info(f"  gt:     {ex['gt']}")
            logger.info(f"  verify: {ex['verify']}")
            logger.info("")

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
    ap.add_argument("--input", default=None, help="Input Self-QA reasoning jsonl")
    ap.add_argument("--output", required=True, help="Output GRPO jsonl")
    ap.add_argument("--model", required=True, help="Model path (DeepSeek-R1-Distill recommended)")
    ap.add_argument("--self_verify", action="store_true",
                    help="Re-solve each problem to verify answer consistency")
    ap.add_argument("--verify_model", default=None, help="[Deprecated] Use --self_verify instead")
    ap.add_argument("--num_samples", type=int, default=10000)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--tensor_parallel", type=int, default=1)
    ap.add_argument("--gpu_utilization", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load single model for all steps
    logger.info(f"Loading model: {args.model}")
    llm, tokenizer = init_vllm(args.model, args.tensor_parallel, args.gpu_utilization)

    all_items = []

    # Extract
    if args.mode in ("extract", "both"):
        if args.input and os.path.exists(args.input):
            items = []
            with open(args.input, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        items.append(json.loads(line))
            extracted = extract_answers(items, llm, tokenizer)
            all_items.extend(extracted)
        else:
            logger.warning(f"No input file: {args.input}")

    # Generate
    if args.mode in ("generate", "both"):
        generated = generate_qa(args.num_samples, args.batch_size, llm, tokenizer)
        all_items.extend(generated)

    # Dedup
    all_items = dedup(all_items)
    before_verify = len(all_items)

    # Self-verify
    if args.self_verify:
        all_items = self_verify(all_items, llm, tokenizer)

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'=' * 50}")
    print(f"  GRPO Data Build Report")
    print(f"{'=' * 50}")
    if args.self_verify:
        print(f"  Before verify: {before_verify}")
        print(f"  After verify:  {len(all_items)}")
        print(f"  Pass rate:     {len(all_items)/max(before_verify,1):.1%}")
    else:
        print(f"  Total:         {len(all_items)}")
    print(f"  Output:        {args.output}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
