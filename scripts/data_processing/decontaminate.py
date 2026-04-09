# -*- coding: utf-8 -*-
"""
EcoGPT - Data Decontamination Script
Detect and remove training samples that overlap with evaluation benchmarks.
Uses MinHash LSH for approximate matching + exact SHA1 for strict matching.

Usage:
    python decontaminate.py \
        --train_dir /path/to/sft_data \
        --eval_dirs /path/to/financeiq,/path/to/fineval,/path/to/ceval \
        --threshold 0.7 \
        --output_dir /path/to/clean_data \
        --report_path /path/to/contamination_report.json
"""

import argparse
import hashlib
import json
import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path

from loguru import logger


def normalize_text(s: str) -> str:
    """Normalize text for comparison: lowercase, remove punctuation/whitespace."""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[。．，,、;；:：!?！？\"\"''（）()\[\]【】<>《》\-—_\n\r\t]+", "", s)
    return s


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def char_ngrams(s: str, n: int = 5) -> list:
    """Character n-grams for MinHash (works well for Chinese without tokenizer)."""
    s = normalize_text(s)
    if len(s) <= n:
        return [s] if s else []
    return [s[i:i + n] for i in range(len(s) - n + 1)]


def extract_text_from_sample(sample: dict) -> str:
    """Extract the main text content from a data sample (various formats)."""
    parts = []

    # Alpaca format
    if "instruction" in sample:
        parts.append(sample.get("instruction", ""))
        parts.append(sample.get("input", ""))
    # Conversation format
    elif "conversations" in sample:
        for msg in sample["conversations"]:
            if msg.get("from") == "human":
                parts.append(msg.get("value", ""))
    # GRPO format
    elif "prompt" in sample:
        parts.append(sample.get("prompt", ""))
    # Eval format (question-based)
    elif "question" in sample:
        parts.append(sample.get("question", ""))

    return " ".join(p for p in parts if p).strip()


def load_jsonl_texts(path: str) -> list:
    """Load text from a jsonl file or directory of jsonl/json files."""
    texts = []
    if os.path.isdir(path):
        files = glob(os.path.join(path, "**/*.jsonl"), recursive=True) + \
                glob(os.path.join(path, "**/*.json"), recursive=True)
    else:
        files = [path]

    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            if fpath.endswith(".json") and not fpath.endswith(".jsonl"):
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            texts.append(extract_text_from_sample(item))
                        continue
                except json.JSONDecodeError:
                    pass
            # jsonl
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    texts.append(extract_text_from_sample(sample))
                except json.JSONDecodeError:
                    continue

    return [t for t in texts if t]


def main():
    ap = argparse.ArgumentParser(description="Decontaminate training data against eval benchmarks")
    ap.add_argument("--train_dir", required=True, help="Training data directory or file")
    ap.add_argument("--eval_dirs", required=True, help="Comma-separated eval data directories/files")
    ap.add_argument("--threshold", type=float, default=0.7, help="MinHash Jaccard similarity threshold")
    ap.add_argument("--num_perm", type=int, default=128, help="MinHash permutation count")
    ap.add_argument("--ngram_size", type=int, default=5, help="Character n-gram size")
    ap.add_argument("--output_dir", required=True, help="Output directory for clean data")
    ap.add_argument("--report_path", default=None, help="Path to save contamination report JSON")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Load eval benchmark texts
    eval_dirs = [d.strip() for d in args.eval_dirs.split(",")]
    eval_texts = []
    eval_source_map = {}
    for eval_dir in eval_dirs:
        texts = load_jsonl_texts(eval_dir)
        eval_name = os.path.basename(eval_dir.rstrip("/\\"))
        for t in texts:
            eval_source_map[len(eval_texts)] = eval_name
            eval_texts.append(t)
    logger.info(f"Loaded {len(eval_texts)} eval samples from {len(eval_dirs)} benchmarks")

    # Step 2: Load training data (preserving original lines for output)
    train_files = []
    if os.path.isdir(args.train_dir):
        train_files = glob(os.path.join(args.train_dir, "**/*.jsonl"), recursive=True) + \
                      glob(os.path.join(args.train_dir, "**/*.json"), recursive=True)
    else:
        train_files = [args.train_dir]

    train_samples = []
    train_texts = []
    for fpath in train_files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    text = extract_text_from_sample(sample)
                    train_samples.append((fpath, line, sample))
                    train_texts.append(text)
                except json.JSONDecodeError:
                    continue

    logger.info(f"Loaded {len(train_samples)} training samples")

    # Step 3: Exact hash matching
    eval_hashes = {sha1(normalize_text(t)) for t in eval_texts}
    exact_contaminated = set()
    for i, t in enumerate(train_texts):
        if sha1(normalize_text(t)) in eval_hashes:
            exact_contaminated.add(i)
    logger.info(f"Exact hash matches: {len(exact_contaminated)}")

    # Step 4: MinHash LSH approximate matching
    approx_contaminated = set()
    try:
        from datasketch import MinHash, MinHashLSH

        lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)

        # Index eval samples
        for i, t in enumerate(eval_texts):
            mh = MinHash(num_perm=args.num_perm)
            for g in char_ngrams(t, args.ngram_size):
                mh.update(g.encode("utf-8"))
            lsh.insert(f"eval_{i}", mh)

        # Query training samples
        for i, t in enumerate(train_texts):
            if i in exact_contaminated:
                continue
            mh = MinHash(num_perm=args.num_perm)
            for g in char_ngrams(t, args.ngram_size):
                mh.update(g.encode("utf-8"))
            results = lsh.query(mh)
            if results:
                approx_contaminated.add(i)

        logger.info(f"Approximate matches (threshold={args.threshold}): {len(approx_contaminated)}")

    except ImportError:
        logger.warning("datasketch not installed. Skipping approximate dedup. Install: pip install datasketch")

    # Step 5: Remove contaminated and write clean data
    all_contaminated = exact_contaminated | approx_contaminated
    clean_count = 0
    contaminated_count = len(all_contaminated)

    out_path = os.path.join(args.output_dir, "train_clean.jsonl")
    contaminated_path = os.path.join(args.output_dir, "contaminated_removed.jsonl")

    with open(out_path, "w", encoding="utf-8") as f_clean, \
            open(contaminated_path, "w", encoding="utf-8") as f_contam:
        for i, (fpath, line, sample) in enumerate(train_samples):
            if i in all_contaminated:
                f_contam.write(line + "\n")
            else:
                f_clean.write(line + "\n")
                clean_count += 1

    logger.info(f"Clean samples: {clean_count}")
    logger.info(f"Contaminated removed: {contaminated_count}")
    logger.info(f"Contamination rate: {contaminated_count / max(len(train_samples), 1):.4f}")
    logger.info(f"Clean data saved to: {out_path}")
    logger.info(f"Removed samples saved to: {contaminated_path}")

    # Step 6: Save report
    report = {
        "total_train_samples": len(train_samples),
        "total_eval_samples": len(eval_texts),
        "exact_matches": len(exact_contaminated),
        "approx_matches": len(approx_contaminated),
        "total_contaminated": contaminated_count,
        "clean_samples": clean_count,
        "contamination_rate": contaminated_count / max(len(train_samples), 1),
        "threshold": args.threshold,
        "eval_sources": {name: sum(1 for v in eval_source_map.values() if v == name)
                         for name in set(eval_source_map.values())},
    }

    report_path = args.report_path or os.path.join(args.output_dir, "contamination_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
