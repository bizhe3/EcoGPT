#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - Deduplication analysis (exact + approximate via MinHash LSH).

Usage:
    python check2.py \
        --data data/sft/processed/train.jsonl \
        --out_dir outputs/dup_report \
        --threshold 0.9 \
        --max_samples 200000
"""

import argparse
import re
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_dataset


def normalize_text(s: str) -> str:
    _ws = re.compile(r"\s+")
    _punc = re.compile(r"[。．，,、;；:：!?！？\"""'''（）()\[\]【】<>《》\-—_]+")
    s = s.strip().lower()
    s = _ws.sub(" ", s)
    s = _punc.sub("", s)
    return s


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def get_fields(ex):
    if "instruction" in ex:
        instr = ex.get("instruction") or ""
        inp = ex.get("input") or ""
        out = ex.get("output") or ""
        return instr, inp, out
    if "conversations" in ex:
        convs = ex["conversations"]
        human = " ".join(m.get("value", "") for m in convs if m.get("from") == "human")
        gpt = " ".join(m.get("value", "") for m in convs if m.get("from") == "gpt")
        return human, "", gpt
    if "prompt" in ex:
        return ex.get("prompt") or "", "", ex.get("completion") or ex.get("ground_truth") or ""
    return str(ex), "", ""


def dup_stats(cnt: Counter, name: str):
    total = sum(cnt.values())
    unique = len(cnt)
    dup_instances = total - unique
    dup_rate = dup_instances / total if total else 0.0
    clusters = sum(1 for k, v in cnt.items() if v >= 2)
    max_dup = max(cnt.values()) if cnt else 0
    print(f"\n[{name}]")
    print(f"total: {total}")
    print(f"unique: {unique}")
    print(f"dup_instances (total-unique): {dup_instances}")
    print(f"dup_rate: {dup_rate:.4f}")
    print(f"dup_clusters (count>=2): {clusters}")
    print(f"max_cluster_size: {max_dup}")


def main():
    ap = argparse.ArgumentParser(description="Deduplication analysis")
    ap.add_argument("--data", required=True, help="Input jsonl data file")
    ap.add_argument("--out_dir", default="./dup_report", help="Output directory for reports")
    ap.add_argument("--max_samples", type=int, default=None, help="Max samples to process (None=all)")
    ap.add_argument("--threshold", type=float, default=0.9, help="MinHash LSH Jaccard threshold")
    ap.add_argument("--num_perm", type=int, default=128, help="MinHash permutation count")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    ds = load_dataset("json", data_files=args.data, split="train")
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    print("Loaded:", len(ds), "rows")
    print("Columns:", ds.column_names)

    # Exact duplicate statistics
    q_hashes = []
    qa_hashes = []

    for ex in ds:
        instr, inp, out = get_fields(ex)
        q_norm = normalize_text((inp + "\n" + instr).strip())
        qa_norm = normalize_text((inp + "\n" + instr + "\n" + out).strip())
        q_hashes.append(sha1(q_norm))
        qa_hashes.append(sha1(qa_norm))

    q_cnt = Counter(q_hashes)
    qa_cnt = Counter(qa_hashes)

    dup_stats(q_cnt, "Exact duplicates by QUESTION (normalized)")
    dup_stats(qa_cnt, "Exact duplicates by QA pair (normalized)")

    # Save top duplicate clusters
    top_q = q_cnt.most_common(50)
    q_to_indices = defaultdict(list)
    for i, h in enumerate(q_hashes):
        q_to_indices[h].append(i)

    sample_path = Path(args.out_dir) / "top_question_duplicate_clusters.txt"
    with open(sample_path, "w", encoding="utf-8") as f:
        for h, c in top_q:
            if c < 2:
                continue
            f.write("\n" + "=" * 90 + "\n")
            f.write(f"Q_HASH={h}  count={c}\n")
            for j, idx in enumerate(q_to_indices[h][:5]):
                instr, inp, out = get_fields(ds[idx])
                f.write(f"\n--- example {j + 1} (idx={idx}) ---\n")
                f.write("[INSTRUCTION]\n" + str(instr) + "\n")
                if inp.strip():
                    f.write("\n[INPUT]\n" + str(inp) + "\n")
                f.write("\n[OUTPUT]\n" + str(out)[:800] + ("\n...<truncated>...\n" if len(str(out)) > 800 else "\n"))
    print(f"\nSaved exact-dup cluster samples to: {sample_path}")

    # Approximate duplicates (MinHash LSH)
    try:
        from datasketch import MinHash, MinHashLSH
    except ImportError:
        print("\n[Approx duplicates] Missing dependency: datasketch")
        print("Install with: pip install datasketch")
        return

    def char_ngrams(s: str, n=3):
        s = normalize_text(s)
        if len(s) <= n:
            return [s] if s else []
        return [s[i:i + n] for i in range(len(s) - n + 1)]

    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    mhs = []

    for i, ex in enumerate(ds):
        instr, inp, _ = get_fields(ex)
        q = (inp + "\n" + instr).strip()
        grams = char_ngrams(q, n=3)
        mh = MinHash(num_perm=args.num_perm)
        for g in grams:
            mh.update(g.encode("utf-8"))
        lsh.insert(f"q{i}", mh)
        mhs.append(mh)

    parent = list(range(len(ds)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(ds)):
        candidates = lsh.query(mhs[i])
        for ck in candidates:
            j = int(ck[1:])
            if j != i:
                union(i, j)

    clusters = defaultdict(list)
    for i in range(len(ds)):
        clusters[find(i)].append(i)

    approx_clusters = [c for c in clusters.values() if len(c) >= 2]
    approx_clusters.sort(key=len, reverse=True)

    print(f"\n[Approx duplicates by QUESTION] threshold={args.threshold}")
    print("num_clusters>=2:", len(approx_clusters))
    print("largest_cluster_size:", len(approx_clusters[0]) if approx_clusters else 0)

    approx_path = Path(args.out_dir) / f"approx_question_duplicate_clusters_thr_{args.threshold}.txt"
    with open(approx_path, "w", encoding="utf-8") as f:
        for k, idxs in enumerate(approx_clusters[:50]):
            f.write("\n" + "=" * 90 + "\n")
            f.write(f"CLUSTER {k + 1}  size={len(idxs)}\n")
            for j, idx in enumerate(idxs[:5]):
                instr, inp, out = get_fields(ds[idx])
                f.write(f"\n--- example {j + 1} (idx={idx}) ---\n")
                f.write("[INSTRUCTION]\n" + str(instr) + "\n")
                if inp.strip():
                    f.write("\n[INPUT]\n" + str(inp) + "\n")
                f.write("\n[OUTPUT]\n" + str(out)[:400] + ("\n...<truncated>...\n" if len(str(out)) > 400 else "\n"))
    print(f"Saved approx-dup cluster samples to: {approx_path}")


if __name__ == "__main__":
    main()
