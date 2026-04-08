#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - Filter BAAI/IndustryInstruction_Finance-Economics dataset.

Filters by deita_score and rw_score to keep only high-quality samples,
extracts conversations, and optionally filters by language.

Usage:
    python filter_baai.py \
        --input data/sft/raw/baai_finance \
        --output data/sft/processed/baai_finance.jsonl \
        --min_deita_score 5.0 \
        --min_rw_score 0.0 \
        --lang zh
"""

import argparse
import json
import os
from glob import glob

from loguru import logger

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def load_baai_data(path: str) -> list:
    """Load BAAI dataset from directory (parquet/json/jsonl)."""
    items = []

    if os.path.isdir(path):
        files = glob(os.path.join(path, "**/*.parquet"), recursive=True) + \
                glob(os.path.join(path, "**/*.jsonl"), recursive=True) + \
                glob(os.path.join(path, "**/*.json"), recursive=True)
    else:
        files = [path]

    for fpath in files:
        if fpath.endswith(".parquet"):
            if not HAS_PANDAS:
                logger.error(f"pandas + pyarrow required for parquet: pip install pandas pyarrow")
                continue
            df = pd.read_parquet(fpath)
            items.extend(df.to_dict("records"))
        elif fpath.endswith(".jsonl"):
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            items.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        elif fpath.endswith(".json"):
            with open(fpath, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        items.extend(data)
                except json.JSONDecodeError:
                    continue

    return items


def main():
    ap = argparse.ArgumentParser(description="Filter BAAI IndustryInstruction Finance dataset")
    ap.add_argument("--input", required=True, help="Input directory or file (BAAI dataset)")
    ap.add_argument("--output", required=True, help="Output jsonl file (conversations format)")
    ap.add_argument("--min_deita_score", type=float, default=5.0,
                    help="Min deita_score (instruction complexity + response quality). Default 5.0")
    ap.add_argument("--min_rw_score", type=float, default=0.0,
                    help="Min rw_score (reward model preference). Default 0.0")
    ap.add_argument("--lang", default=None, choices=["zh", "en", None],
                    help="Filter by language. None = keep both")
    ap.add_argument("--min_turns", type=int, default=2,
                    help="Min conversation turns (2 = at least 1 QA pair)")
    ap.add_argument("--min_answer_len", type=int, default=20,
                    help="Min answer character length")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load
    logger.info(f"Loading BAAI data from {args.input}")
    raw = load_baai_data(args.input)
    logger.info(f"Loaded {len(raw)} raw samples")

    # Filter
    kept = 0
    dropped_score = 0
    dropped_lang = 0
    dropped_content = 0

    score_dist = {"deita": [], "rw": []}

    with open(args.output, "w", encoding="utf-8") as fout:
        for item in raw:
            deita = item.get("deita_score", 0) or 0
            rw = item.get("rw_score", 0) or 0
            lang = item.get("lang", "")
            convs = item.get("conversations", [])

            # Score filter
            if deita < args.min_deita_score or rw < args.min_rw_score:
                dropped_score += 1
                continue

            # Language filter
            if args.lang and lang != args.lang:
                dropped_lang += 1
                continue

            # Content quality filter
            if len(convs) < args.min_turns:
                dropped_content += 1
                continue

            # Check answer length
            gpt_texts = [m.get("value", "") for m in convs if m.get("from") == "gpt"]
            if not gpt_texts or len(gpt_texts[0]) < args.min_answer_len:
                dropped_content += 1
                continue

            # Check human question is non-empty
            human_texts = [m.get("value", "") for m in convs if m.get("from") == "human"]
            if not human_texts or len(human_texts[0].strip()) < 5:
                dropped_content += 1
                continue

            # Output: only keep conversations field (drop metadata)
            out_item = {"conversations": convs}
            fout.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            kept += 1

            score_dist["deita"].append(deita)
            score_dist["rw"].append(rw)

    # Statistics
    total = len(raw)
    print(f"\n{'=' * 55}")
    print(f"  BAAI IndustryInstruction Filter Report")
    print(f"{'=' * 55}")
    print(f"  Input:            {total}")
    print(f"  Kept:             {kept} ({kept/total:.1%})")
    print(f"  Dropped (score):  {dropped_score}")
    print(f"  Dropped (lang):   {dropped_lang}")
    print(f"  Dropped (content):{dropped_content}")
    print(f"{'=' * 55}")
    print(f"  Filters applied:")
    print(f"    deita_score >= {args.min_deita_score}")
    print(f"    rw_score    >= {args.min_rw_score}")
    print(f"    lang        =  {args.lang or 'all'}")
    print(f"    min_turns   =  {args.min_turns}")
    print(f"    min_answer  =  {args.min_answer_len} chars")

    if score_dist["deita"]:
        d = sorted(score_dist["deita"])
        r = sorted(score_dist["rw"])
        n = len(d)
        print(f"\n  Kept samples score distribution:")
        print(f"    deita_score: min={d[0]:.1f}  p50={d[n//2]:.1f}  p90={d[int(n*0.9)]:.1f}  max={d[-1]:.1f}")
        print(f"    rw_score:    min={r[0]:.1f}  p50={r[n//2]:.1f}  p90={r[int(n*0.9)]:.1f}  max={r[-1]:.1f}")

    print(f"\n  Output: {args.output}")


if __name__ == "__main__":
    main()
