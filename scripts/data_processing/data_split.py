#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import os

#use seed=42
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="input jsonl")
    ap.add_argument("--out_dir", required=True, help="output dir")
    ap.add_argument("--valid_ratio", type=float, default=0.05, help="validation ratio")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.in_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    n = len(lines)
    idx = list(range(n))
    random.shuffle(idx)

    n_valid = int(n * args.valid_ratio)
    valid_idx = set(idx[:n_valid])

    os.makedirs(args.out_dir, exist_ok=True)

    train_path = os.path.join(args.out_dir, "train.jsonl")
    valid_path = os.path.join(args.out_dir, "valid.jsonl")

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(valid_path, "w", encoding="utf-8") as f_valid:
        for i, line in enumerate(lines):
            if i in valid_idx:
                f_valid.write(line)
            else:
                f_train.write(line)

    print(f"Total: {n}")
    print(f"Train: {n - n_valid}")
    print(f"Valid: {n_valid}")
    print(f"Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
