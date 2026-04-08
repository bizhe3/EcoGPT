#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - Merge and sample multiple SFT datasets with target ratios.

Usage:
    python merge_sft_data.py \
        --sources consulting:data/disc_consulting.jsonl:0.25 \
                  task:data/disc_task.jsonl:0.35 \
                  computing:data/disc_computing.jsonl:0.15 \
                  retrieval:data/disc_retrieval.jsonl:0.10 \
                  self_qa:data/self_qa.jsonl:0.05 \
                  general:data/general_zh.jsonl:0.10 \
        --total 250000 \
        --output data/sft/processed/merged_sft.jsonl \
        --seed 42
"""

import argparse
import json
import os
import random
from loguru import logger


def load_jsonl(path: str) -> list:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return data


def main():
    ap = argparse.ArgumentParser(description="Merge and sample SFT datasets")
    ap.add_argument("--sources", nargs="+", required=True,
                    help="Format: name:path:ratio (ratio 0~1, sum ~1.0)")
    ap.add_argument("--total", type=int, default=250000, help="Total target samples")
    ap.add_argument("--output", required=True, help="Output merged jsonl")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    sources = []
    for s in args.sources:
        parts = s.split(":")
        if len(parts) != 3:
            logger.error(f"Invalid source: {s}. Expected name:path:ratio")
            return
        sources.append({"name": parts[0], "path": parts[1], "ratio": float(parts[2])})

    total_ratio = sum(s["ratio"] for s in sources)

    merged = []
    stats = []

    for src in sources:
        target_count = int(args.total * src["ratio"] / total_ratio)
        data = load_jsonl(src["path"])
        available = len(data)

        if available == 0:
            logger.warning(f"[{src['name']}] No data at {src['path']}, skipping")
            stats.append({"name": src["name"], "available": 0, "sampled": 0})
            continue

        sampled = random.sample(data, min(target_count, available))
        if available < target_count:
            logger.warning(f"[{src['name']}] Only {available}/{target_count} available, using all")

        merged.extend(sampled)
        stats.append({"name": src["name"], "available": available, "sampled": len(sampled)})
        logger.info(f"[{src['name']}] {len(sampled)}/{available} (ratio: {src['ratio']:.0%})")

    random.shuffle(merged)

    with open(args.output, "w", encoding="utf-8") as f:
        for item in merged:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n{'Source':<20} {'Available':>10} {'Sampled':>10}")
    print("-" * 42)
    for s in stats:
        print(f"{s['name']:<20} {s['available']:>10} {s['sampled']:>10}")
    print("-" * 42)
    print(f"{'TOTAL':<20} {'':>10} {len(merged):>10}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
