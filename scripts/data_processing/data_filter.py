#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - Token length filter for SFT data.

Usage:
    python data_filter.py \
        --input data/sft/processed/train.jsonl \
        --output data/sft/processed/train_filtered.jsonl \
        --tokenizer models/base/Qwen2.5-7B-Instruct \
        --max_total_tokens 2048
"""

import argparse
import json
from tqdm import tqdm
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser(description="Filter data by token length")
    ap.add_argument("--input", required=True, help="Input jsonl file")
    ap.add_argument("--output", required=True, help="Output jsonl file")
    ap.add_argument("--tokenizer", required=True, help="Tokenizer path")
    ap.add_argument("--max_total_tokens", type=int, default=2048, help="Max total tokens (prompt + completion)")
    ap.add_argument("--min_total_tokens", type=int, default=1, help="Min total tokens")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    def tok_len(text: str) -> int:
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    kept = 0
    dropped = 0
    stats = []

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Filtering"):
            ex = json.loads(line)

            instr = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            out = (ex.get("output") or "").strip()

            if not instr or not out:
                dropped += 1
                continue

            prompt_text = instr if not inp else f"{inp}\n{instr}"

            prompt_len = tok_len(prompt_text)
            completion_len = tok_len(out)
            total_len = prompt_len + completion_len

            if args.min_total_tokens <= total_len <= args.max_total_tokens:
                fout.write(json.dumps(ex, ensure_ascii=False) + "\n")
                kept += 1
                stats.append(total_len)
            else:
                dropped += 1

    stats.sort()
    n = len(stats)

    def p(x):
        return stats[int(n * x)]

    print("\n==================== RESULT ====================")
    print(f"input file:   {args.input}")
    print(f"output file:  {args.output}")
    print(f"kept:         {kept}")
    print(f"dropped:      {dropped}")
    print(f"keep ratio:   {kept / (kept + dropped):.4f}")

    if n > 0:
        print("\n[token stats of kept samples]")
        print(f"  min: {stats[0]}")
        print(f"  p50: {p(0.50)}")
        print(f"  p90: {p(0.90)}")
        print(f"  p95: {p(0.95)}")
        print(f"  p99: {p(0.99)}")
        print(f"  max: {stats[-1]}")


if __name__ == "__main__":
    main()
