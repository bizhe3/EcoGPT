#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - FinanceIQ Evaluation (aligned with XuanYuan official implementation)

Evaluate models on FinanceIQ benchmark using generate + extract_choice method.
Supports few-shot evaluation with dev set examples.

Source: https://github.com/Duxiaoman-DI/XuanYuan/tree/main/FinanceIQ

Usage:
    # 0-shot (for chat models)
    python eval_financeiq.py \
        --model models/grpo_merged \
        --data data/eval/financeiq \
        --output outputs/eval_results/financeiq_grpo.json

    # 5-shot (for base models)
    python eval_financeiq.py \
        --model models/base/Qwen3-4B \
        --data data/eval/financeiq \
        --output outputs/eval_results/financeiq_base.json \
        --num_few_shot 5
"""

import argparse
import json
import os
import re
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

CHOICES = ["A", "B", "C", "D"]

# ============================================================
# Prompt construction (following XuanYuan official implementation)
# ============================================================

def format_example(df, idx, subject, include_answer=True):
    """Format a single question as prompt text."""
    question = df.iloc[idx, 0]
    prompt = "题目：" + str(question)
    k = df.shape[1] - 2  # number of options
    for j in range(k):
        option_choice = CHOICES[j]
        option_content = df.iloc[idx, j + 1]
        prompt += f"\n{option_choice}. {option_content}"
    prompt += "\n答案是："
    if include_answer:
        answer = df.iloc[idx, k + 1]
        prompt += f"{answer}\n\n"
    return prompt


def gen_prompt(dev_df, subject, prompt_end, num_few_shot=0, tokenizer=None, max_length=2048):
    """Generate full prompt with few-shot examples."""
    prompt = f"以下是关于{subject}的单项选择题，请直接给出正确答案的选项。\n\n"

    if num_few_shot > 0 and dev_df is not None:
        for i in range(min(num_few_shot, len(dev_df))):
            example = format_example(dev_df, i, subject, include_answer=True)
            prompt += example

    return prompt + prompt_end


# ============================================================
# Answer extraction (following XuanYuan official implementation)
# ============================================================

def extract_choice(response):
    """Extract choice from model response using regex patterns."""
    response = str(response).strip()

    # Strip thinking tags if present (Qwen3)
    if '</think>' in response:
        response = response.split('</think>')[-1].strip()

    if response and response[0] in CHOICES:
        return response[0]

    # Pattern matching (from XuanYuan official)
    patterns = [
        (r'答案(选项)?(是|为)：? ?([ABCD])', 3),
        (r'答案(是|为)选项 ?([ABCD])', 2),
        (r'故?选择?：? ?([ABCD])', 1),
        (r'([ABCD]) ?选?项(是|为)?正确', 1),
        (r'正确的?选项(是|为) ?([ABCD])', 2),
        (r'答案(应该)?(是|为)([ABCD])', 3),
        (r'选项 ?([ABCD]) ?(是|为)?正确', 1),
        (r'选择答案 ?([ABCD])', 1),
        (r'答案?：?([ABCD])', 1),
        (r'([ABCD])(选?项)?是?符合题意', 1),
        (r'答案选项：? ?([ABCD])', 1),
        (r'答案(选项)?为(.*?)([ABCD])', 3),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            answer = m.group(idx)
            if answer in CHOICES:
                return answer

    # Recursive match
    patterns = [
        (r'([ABCD])(.*?)当选', 1),
        (r'([ABCD])(.*?)正确', 1),
    ]
    for pattern, idx in patterns:
        m = re.search(pattern, response, re.M)
        if m:
            while m:
                answer = m.group(idx)
                m = re.search(pattern, m.group(0)[1:], re.M)
            if answer in CHOICES:
                return answer

    # Check only mentioned choice
    pattern = r'^[^ABCD]*([ABCD])[^ABCD]*$'
    m = re.match(pattern, response)
    if m:
        return m.group(1)

    # Random fallback (ensures fair comparison)
    return CHOICES[random.randint(0, 3)]


# ============================================================
# Evaluation
# ============================================================

def evaluate_generate(model, tokenizer, dev_df, test_df, subject,
                      num_few_shot, max_length, device, use_chat_template=False):
    """Evaluate using generate + extract_choice (XuanYuan method)."""
    cors = []
    all_preds = []

    for i in range(len(test_df)):
        prompt_end = format_example(test_df, i, subject, include_answer=False)
        prompt = gen_prompt(dev_df, subject, prompt_end, num_few_shot, tokenizer, max_length)
        label = str(test_df.iloc[i, test_df.shape[1] - 1]).strip()

        # For chat models, wrap in chat template
        if use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        else:
            text = prompt

        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        pred = extract_choice(response)
        all_preds.append(pred)

        if label in CHOICES:
            cors.append(pred == label)

    acc = np.mean(cors) if cors else 0.0
    return acc, cors, all_preds


def main():
    ap = argparse.ArgumentParser(description="Evaluate on FinanceIQ benchmark")
    ap.add_argument("--model", required=True, help="Model path")
    ap.add_argument("--data", required=True, help="FinanceIQ data directory (contains dev/ and test/)")
    ap.add_argument("--output", required=True, help="Output JSON file")
    ap.add_argument("--num_few_shot", type=int, default=0, help="Number of few-shot examples (0 for chat models, 5 for base)")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--use_chat_template", action="store_true", default=True,
                    help="Use chat template for chat models")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Discover subjects from test directory
    test_dir = os.path.join(args.data, "test")
    dev_dir = os.path.join(args.data, "dev")

    if not os.path.isdir(test_dir):
        logger.error(f"Test directory not found: {test_dir}")
        return

    subjects = []
    for f in sorted(os.listdir(test_dir)):
        if f.endswith(".csv"):
            subjects.append(os.path.splitext(f)[0])

    logger.info(f"Found {len(subjects)} subjects: {subjects}")

    # Load model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=args.device,
    ).eval()

    # Evaluate each subject
    all_acc = {}
    total_correct = 0
    total_questions = 0

    for i, subject in enumerate(subjects):
        logger.info(f"Evaluating {i+1}/{len(subjects)}: {subject}")

        test_df = pd.read_csv(os.path.join(test_dir, subject + ".csv"), header=0, index_col=0)
        dev_df = None
        dev_path = os.path.join(dev_dir, subject + ".csv")
        if os.path.exists(dev_path):
            dev_df = pd.read_csv(dev_path, header=0, index_col=0)

        acc, cors, preds = evaluate_generate(
            model, tokenizer, dev_df, test_df, subject,
            args.num_few_shot, args.max_length, args.device,
            use_chat_template=args.use_chat_template,
        )

        all_acc[subject] = acc
        total_correct += sum(cors)
        total_questions += len(cors)
        logger.info(f"  {subject}: {sum(cors)}/{len(cors)} = {acc:.4f}")

    overall_acc = total_correct / max(total_questions, 1)

    # Report
    report = {
        "model": args.model,
        "benchmark": "FinanceIQ",
        "num_few_shot": args.num_few_shot,
        "overall": {"accuracy": overall_acc, "correct": total_correct, "total": total_questions},
        "categories": {s: {"accuracy": a} for s, a in all_acc.items()},
    }

    print(f"\n{'=' * 60}")
    print(f"  FinanceIQ Evaluation Report ({args.num_few_shot}-shot)")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model}")
    print(f"  Overall: {total_correct}/{total_questions} = {overall_acc:.4f}")
    print(f"{'=' * 60}")
    for subject in sorted(all_acc.keys()):
        print(f"  {subject:40s} {all_acc[subject]:.4f}")
    print(f"{'=' * 60}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
