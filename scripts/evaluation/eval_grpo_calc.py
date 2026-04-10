#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - GRPO Financial Calculation Accuracy Evaluation

Test model's financial calculation reasoning ability using GRPO validation set.
Model generates answers with thinking enabled, then we extract and verify.

Usage:
    python eval_grpo_calc.py \
        --model models/grpo_merged \
        --data data/grpo/val/valid.jsonl \
        --output outputs/eval_results/grpo_calc.json
"""

import argparse
import json
import os
import re

from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def extract_number(s: str):
    """Extract first number from string, handling Chinese units."""
    s = re.sub(r'[,，\s]', '', s)
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


def answer_matches(prediction: str, ground_truth: str) -> bool:
    """Check if prediction matches ground truth."""
    gt_num = extract_number(ground_truth)
    if gt_num is not None:
        # Extract answer part (after </think> if present)
        if '</think>' in prediction:
            prediction = prediction.split('</think>')[-1]

        nums = re.findall(r'[-+]?\d*\.?\d+', prediction.replace(',', '').replace('，', ''))
        for n_str in nums:
            try:
                n_val = float(n_str)
                if gt_num == 0 and n_val == 0:
                    return True
                if abs(n_val - gt_num) < 0.5:
                    return True
                if gt_num != 0 and abs(n_val - gt_num) / abs(gt_num) < 0.05:
                    return True
            except ValueError:
                continue
        return False

    # Text matching
    gt_norm = re.sub(r'[，。\s]', '', ground_truth.lower())
    pred_norm = re.sub(r'[，。\s]', '', prediction.lower())
    return gt_norm in pred_norm


def main():
    ap = argparse.ArgumentParser(description="Evaluate GRPO calculation accuracy")
    ap.add_argument("--model", required=True, help="Model path")
    ap.add_argument("--data", required=True, help="GRPO validation jsonl")
    ap.add_argument("--output", required=True, help="Output JSON file")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--tensor_parallel_size", type=int, default=1, help="Number of GPUs for tensor parallelism")
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load data
    samples = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if args.max_samples:
        samples = samples[:args.max_samples]

    logger.info(f"Loaded {len(samples)} calculation problems")

    # Load tokenizer (for prompt formatting)
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    # Prepare all prompts
    all_prompts = []
    all_gts = []
    for item in samples:
        messages = [
            {"role": "system", "content": "你是一个专业的金融分析助手。请仔细分析问题并给出准确的答案。"},
            {"role": "user", "content": item["prompt"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(text)
        all_gts.append(item["ground_truth"])

    # Initialize vLLM engine
    logger.info(f"Loading vLLM model: {args.model}")
    llm = LLM(
        model=args.model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=2048,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    sampling_params = SamplingParams(
        max_tokens=1024,
        temperature=0.1,
    )

    # Batch generate
    logger.info("Running batch inference...")
    outputs = llm.generate(all_prompts, sampling_params)

    # Score
    correct = 0
    total = 0
    details = []

    for output, gt, item in zip(outputs, all_gts, samples):
        response = output.outputs[0].text
        matched = answer_matches(response, gt)
        if matched:
            correct += 1
        total += 1

        details.append({
            "prompt": item["prompt"][:100],
            "ground_truth": gt,
            "prediction": response[:200],
            "correct": matched,
        })

    accuracy = correct / max(total, 1)

    report = {
        "model": args.model,
        "benchmark": "GRPO_Financial_Calculation",
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "details": details[:20],  # Save first 20 for inspection
    }

    print(f"\n{'=' * 50}")
    print(f"  GRPO Financial Calculation Evaluation")
    print(f"{'=' * 50}")
    print(f"  Model:    {args.model}")
    print(f"  Accuracy: {correct}/{total} = {accuracy:.4f}")
    print(f"{'=' * 50}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
