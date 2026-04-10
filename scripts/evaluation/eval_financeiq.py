#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - FinanceIQ Evaluation

Evaluate models on FinanceIQ benchmark (7173 multiple-choice questions
across 10 financial exam categories and 36 subcategories).

Source: https://github.com/Duxiaoman-DI/XuanYuan/tree/main/FinanceIQ

Usage:
    python eval_financeiq.py \
        --model models/grpo_merged \
        --data data/eval/financeiq \
        --output outputs/eval_results/financeiq_grpo.json
"""

import argparse
import json
import os
import re
from collections import defaultdict
from glob import glob

import torch
from loguru import logger
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_financeiq(data_dir: str):
    """Load FinanceIQ data from CSV files."""
    import csv

    all_samples = []
    categories = defaultdict(list)

    csv_files = glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    if not csv_files:
        # Try jsonl format
        jsonl_files = glob(os.path.join(data_dir, "**/*.jsonl"), recursive=True)
        for fpath in jsonl_files:
            category = os.path.splitext(os.path.basename(fpath))[0]
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    item["category"] = category
                    all_samples.append(item)
                    categories[category].append(item)
        return all_samples, dict(categories)

    for fpath in csv_files:
        category = os.path.splitext(os.path.basename(fpath))[0]
        with open(fpath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                row["category"] = category
                all_samples.append(row)
                categories[category].append(row)

    return all_samples, dict(categories)


def format_question(item: dict) -> str:
    """Format a multiple-choice question as prompt."""
    question = item.get("question", item.get("题目", ""))
    options = []

    # Try different field names
    for key in ["A", "B", "C", "D", "E"]:
        val = item.get(key, item.get(f"选项{key}", ""))
        if val:
            options.append(f"{key}. {val}")

    prompt = f"{question}\n" + "\n".join(options) + "\n答案："
    return prompt


def get_answer(item: dict) -> str:
    """Extract correct answer from item."""
    answer = item.get("answer", item.get("答案", ""))
    # Normalize to single letter
    answer = answer.strip().upper()
    if answer and answer[0] in "ABCDE":
        return answer[0]
    return answer


def evaluate_loglikelihood(model, tokenizer, samples, device, batch_size=1):
    """Evaluate using log-likelihood of each option (like lm-eval)."""
    correct = 0
    total = 0
    category_results = defaultdict(lambda: {"correct": 0, "total": 0})

    for item in tqdm(samples, desc="Evaluating"):
        question = format_question(item)
        gt = get_answer(item)
        category = item.get("category", "unknown")

        if not gt:
            continue

        # Score each option
        option_scores = {}
        for opt in ["A", "B", "C", "D"]:
            text = question + opt
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                # Get log probability of the last token (the option letter)
                logits = outputs.logits[0, -1, :]
                log_probs = torch.log_softmax(logits, dim=-1)
                opt_token_id = tokenizer.encode(opt, add_special_tokens=False)[-1]
                option_scores[opt] = log_probs[opt_token_id].item()

        predicted = max(option_scores, key=option_scores.get)

        if predicted == gt:
            correct += 1
            category_results[category]["correct"] += 1

        total += 1
        category_results[category]["total"] += 1

    return correct, total, dict(category_results)


def main():
    ap = argparse.ArgumentParser(description="Evaluate on FinanceIQ benchmark")
    ap.add_argument("--model", required=True, help="Model path")
    ap.add_argument("--data", required=True, help="FinanceIQ data directory")
    ap.add_argument("--output", required=True, help="Output JSON file")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load data
    logger.info(f"Loading FinanceIQ from {args.data}")
    samples, categories = load_financeiq(args.data)
    logger.info(f"Loaded {len(samples)} questions across {len(categories)} categories")

    if not samples:
        logger.error("No samples found!")
        return

    # Load model
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True,
        torch_dtype=torch.bfloat16, device_map=args.device,
    ).eval()

    # Evaluate
    correct, total, cat_results = evaluate_loglikelihood(
        model, tokenizer, samples, args.device,
    )

    overall_acc = correct / max(total, 1)

    # Build report
    report = {
        "model": args.model,
        "benchmark": "FinanceIQ",
        "overall": {"accuracy": overall_acc, "correct": correct, "total": total},
        "categories": {},
    }

    print(f"\n{'=' * 60}")
    print(f"  FinanceIQ Evaluation Report")
    print(f"{'=' * 60}")
    print(f"  Model: {args.model}")
    print(f"  Overall: {correct}/{total} = {overall_acc:.4f}")
    print(f"{'=' * 60}")

    for cat in sorted(cat_results.keys()):
        r = cat_results[cat]
        acc = r["correct"] / max(r["total"], 1)
        report["categories"][cat] = {
            "accuracy": acc, "correct": r["correct"], "total": r["total"]
        }
        print(f"  {cat:40s} {r['correct']:4d}/{r['total']:4d} = {acc:.4f}")

    print(f"{'=' * 60}")

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
