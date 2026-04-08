#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - Build GRPO training data from various sources.

Converts financial QA data into GRPO format: {"prompt": "...", "ground_truth": "..."}
Supports multiple input formats.

Usage:
    python build_grpo_data.py \
        --inputs data/grpo/raw/fin_r1.jsonl data/grpo/raw/disc_computing.jsonl \
        --output data/grpo/processed/grpo_train.jsonl \
        --format auto \
        --system_prompt "你是一个专业的金融分析助手。请先在<think>标签中分析推理，然后在<answer>标签中给出最终答案。"
"""

import argparse
import json
import os
import random
import re
from typing import Optional

from loguru import logger


GRPO_SYSTEM_PROMPT = (
    "你是一个专业的金融分析助手。请先在<think>标签中进行分析推理，"
    "然后在<answer>标签中给出最终答案。\n"
    "格式：<think>你的推理过程</think><answer>最终答案</answer>"
)


def extract_from_alpaca(item: dict, system_prompt: str) -> Optional[dict]:
    """Convert Alpaca format to GRPO format."""
    instr = (item.get("instruction") or "").strip()
    inp = (item.get("input") or "").strip()
    out = (item.get("output") or "").strip()

    if not instr or not out:
        return None

    prompt = f"{system_prompt}\n\n{instr}" if not inp else f"{system_prompt}\n\n{instr}\n{inp}"

    # Try to extract a concise answer from output (for ground_truth)
    # If output contains <answer> tags, use that
    answer_match = re.search(r"<answer>(.*?)</answer>", out, re.DOTALL)
    if answer_match:
        gt = answer_match.group(1).strip()
    else:
        # Use the full output as ground truth (works for short answers)
        gt = out.strip()

    return {"prompt": prompt, "ground_truth": gt}


def extract_from_conversations(item: dict, system_prompt: str) -> Optional[dict]:
    """Convert conversations format to GRPO format."""
    convs = item.get("conversations", [])
    if len(convs) < 2:
        return None

    human = next((m["value"] for m in convs if m.get("from") == "human"), None)
    gpt = next((m["value"] for m in convs if m.get("from") == "gpt"), None)

    if not human or not gpt:
        return None

    prompt = f"{system_prompt}\n\n{human}"

    answer_match = re.search(r"<answer>(.*?)</answer>", gpt, re.DOTALL)
    gt = answer_match.group(1).strip() if answer_match else gpt.strip()

    return {"prompt": prompt, "ground_truth": gt}


def extract_from_grpo(item: dict, system_prompt: str) -> Optional[dict]:
    """Pass-through for already GRPO-formatted data."""
    if "prompt" in item and "ground_truth" in item:
        prompt = item["prompt"]
        if system_prompt and system_prompt not in prompt:
            prompt = f"{system_prompt}\n\n{prompt}"
        return {"prompt": prompt, "ground_truth": item["ground_truth"]}
    return None


def detect_format(item: dict) -> str:
    """Auto-detect data format."""
    if "prompt" in item and "ground_truth" in item:
        return "grpo"
    elif "conversations" in item:
        return "conversations"
    elif "instruction" in item:
        return "alpaca"
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description="Build GRPO training data")
    ap.add_argument("--inputs", nargs="+", required=True, help="Input files (jsonl)")
    ap.add_argument("--output", required=True, help="Output GRPO jsonl file")
    ap.add_argument("--format", default="auto", choices=["auto", "alpaca", "conversations", "grpo"],
                    help="Input format (auto-detect by default)")
    ap.add_argument("--system_prompt", default=GRPO_SYSTEM_PROMPT, help="System prompt to prepend")
    ap.add_argument("--max_gt_length", type=int, default=200,
                    help="Max ground_truth character length (skip overly long answers)")
    ap.add_argument("--min_gt_length", type=int, default=1, help="Min ground_truth length")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    extractors = {
        "alpaca": extract_from_alpaca,
        "conversations": extract_from_conversations,
        "grpo": extract_from_grpo,
    }

    results = []
    total_read = 0
    skipped = 0

    for input_path in args.inputs:
        logger.info(f"Processing {input_path}")
        file_count = 0

        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue

                total_read += 1

                # Detect or use specified format
                fmt = detect_format(item) if args.format == "auto" else args.format
                extractor = extractors.get(fmt)
                if not extractor:
                    skipped += 1
                    continue

                result = extractor(item, args.system_prompt)
                if result is None:
                    skipped += 1
                    continue

                gt = result["ground_truth"]
                if len(gt) < args.min_gt_length or len(gt) > args.max_gt_length:
                    skipped += 1
                    continue

                results.append(result)
                file_count += 1

        logger.info(f"  -> {file_count} GRPO pairs extracted")

    # Shuffle and save
    random.shuffle(results)

    with open(args.output, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print("\n==================== GRPO DATA BUILD ====================")
    print(f"Total read:    {total_read}")
    print(f"Skipped:       {skipped}")
    print(f"GRPO pairs:    {len(results)}")
    print(f"Output:        {args.output}")


if __name__ == "__main__":
    main()
