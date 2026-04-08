#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - Convert Alpaca format to ShareGPT conversations format.

Usage:
    python apaca2conversation.py \
        --input data/sft/raw/finance_alpaca.json \
        --output_dir data/sft/processed
"""

import os
import json
import argparse
from typing import Dict, Any
from tqdm import tqdm


def build_human_value(item: Dict[str, Any]) -> str:
    instruction = (item.get("instruction") or "").strip()
    inp = (item.get("input") or "").strip()

    if instruction and inp:
        return f"{instruction}\n{inp}"
    return instruction or inp


def convert_line(item: Dict[str, Any]) -> Dict[str, Any]:
    human_value = build_human_value(item)
    gpt_value = (item.get("output") or "").strip()

    return {
        "conversations": [
            {"from": "human", "value": human_value},
            {"from": "gpt", "value": gpt_value},
        ]
    }


def main():
    parser = argparse.ArgumentParser(description="Convert Alpaca format to conversations format")
    parser.add_argument("--input", required=True, help="Input Alpaca data file (json or jsonl)")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--output_name", default=None, help="Output filename (default: auto-generated)")
    args = parser.parse_args()

    in_path = args.input
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.basename(in_path)
    if args.output_name:
        out_name = args.output_name
    else:
        stem = base
        for ext in [".jsonl", ".json"]:
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        out_name = f"{stem}_conversations.jsonl"

    out_path = os.path.join(out_dir, out_name)

    total = 0
    kept = 0
    bad = 0

    def write_one(fout, obj):
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    with open(out_path, "w", encoding="utf-8") as fout:
        if in_path.endswith(".json"):
            with open(in_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Input .json must be a list[dict]")
            for item in tqdm(data, desc="Converting (json list)"):
                total += 1
                try:
                    conv = convert_line(item)
                    hv = conv["conversations"][0]["value"].strip()
                    gv = conv["conversations"][1]["value"].strip()
                    if not hv or not gv:
                        bad += 1
                        continue
                    write_one(fout, conv)
                    kept += 1
                except Exception:
                    bad += 1
        else:
            with open(in_path, "r", encoding="utf-8") as f:
                for line in tqdm(f, desc="Converting (jsonl)"):
                    line = line.strip()
                    if not line:
                        continue
                    total += 1
                    try:
                        item = json.loads(line)
                        conv = convert_line(item)
                        hv = conv["conversations"][0]["value"].strip()
                        gv = conv["conversations"][1]["value"].strip()
                        if not hv or not gv:
                            bad += 1
                            continue
                        write_one(fout, conv)
                        kept += 1
                    except Exception:
                        bad += 1

    print("\n===== DONE =====")
    print("Input :", in_path)
    print("Output:", out_path)
    print("Total read:", total)
    print("Kept:", kept)
    print("Skipped (bad/empty):", bad)


if __name__ == "__main__":
    main()
