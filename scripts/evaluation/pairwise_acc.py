#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - Pairwise accuracy evaluation for reward models.

Usage:
    python pairwise_acc.py \
        --rm_path models/reward_model/merged \
        --valid_dir data/grpo/val \
        --max_length 1024
"""

import argparse
import json
import glob

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def load_jsonl_or_json(files):
    data = []
    for fp in files:
        if fp.endswith(".jsonl"):
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    data.append(json.loads(line))
        else:
            with open(fp, "r", encoding="utf-8") as f:
                obj = json.load(f)
                if isinstance(obj, list):
                    data.extend(obj)
                else:
                    data.append(obj)
    return data


@torch.no_grad()
def score(model, tokenizer, text, device, max_length):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    r = out.logits.squeeze().float().item()
    return r


def main():
    ap = argparse.ArgumentParser(description="Evaluate reward model pairwise accuracy")
    ap.add_argument("--rm_path", required=True, help="Path to reward model")
    ap.add_argument("--valid_dir", required=True, help="Validation data directory (DPO format)")
    ap.add_argument("--max_length", type=int, default=1024, help="Max token length for scoring")
    args = ap.parse_args()

    files = glob.glob(args.valid_dir + "/**/*.json", recursive=True) + \
            glob.glob(args.valid_dir + "/**/*.jsonl", recursive=True)
    data = load_jsonl_or_json(files)

    if not data:
        print(f"No data found in {args.valid_dir}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.rm_path, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.rm_path, trust_remote_code=True
    ).to(device)
    model.eval()

    correct = 0
    margins = []

    for ex in data:
        chosen_text = ex.get("chosen_prompt", None)
        rejected_text = ex.get("rejected_prompt", None)

        if chosen_text is None or rejected_text is None:
            chosen_text = f"{ex.get('question', '')}\n{ex.get('response_chosen', '')}"
            rejected_text = f"{ex.get('question', '')}\n{ex.get('response_rejected', '')}"

        rc = score(model, tokenizer, chosen_text, device, args.max_length)
        rr = score(model, tokenizer, rejected_text, device, args.max_length)
        m = rc - rr
        margins.append(m)
        if m > 0:
            correct += 1

    acc = correct / len(data)
    avg_margin = sum(margins) / len(margins)
    margins_sorted = sorted(margins)
    p50 = margins_sorted[len(margins_sorted) // 2]
    p90 = margins_sorted[int(len(margins_sorted) * 0.9)]

    print(f"Samples     : {len(data)}")
    print(f"Pairwise Acc: {acc:.4f}")
    print(f"Avg Margin  : {avg_margin:.4f}")
    print(f"Margin p50  : {p50:.4f}")
    print(f"Margin p90  : {p90:.4f}")


if __name__ == "__main__":
    main()
