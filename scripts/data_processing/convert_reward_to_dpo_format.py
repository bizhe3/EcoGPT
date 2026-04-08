#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from glob import glob
from typing import Dict, Any, List


def normalize_history(x) -> List[List[str]]:
    """
    Ensure history is List[List[str,str]].
    If missing or invalid -> [].
    """
    if not x:
        return []
    if isinstance(x, list):
        ok = True
        for item in x:
            if not (isinstance(item, list) and len(item) == 2):
                ok = False
                break
        return x if ok else []
    return []


def convert_obj(
    obj: Dict[str, Any],
    default_system: str,
    question_field: str,
    chosen_field: str,
    rejected_field: str,
) -> Dict[str, Any]:
    q = (obj.get(question_field) or "").strip()
    c = (obj.get(chosen_field) or "").strip()
    r = (obj.get(rejected_field) or "").strip()

    if not q or not c or not r:
        raise ValueError(f"Missing required fields or empty text. "
                         f"q={bool(q)} chosen={bool(c)} rejected={bool(r)} keys={list(obj.keys())}")

    # If original already has system/history, keep them; otherwise fill defaults
    system = (obj.get("system") if "system" in obj else default_system) or ""
    history = normalize_history(obj.get("history")) if "history" in obj else []

    out = {
        "system": system,
        "history": history,
        "question": q,
        "response_chosen": c,
        "response_rejected": r,
    }

    # keep extra fields if you want (optional)
    # for k, v in obj.items():
    #     if k not in out:
    #         out[k] = v

    return out


def iter_input_files(path: str) -> List[str]:
    if os.path.isdir(path):
        files = glob(os.path.join(path, "**/*.jsonl"), recursive=True) + glob(os.path.join(path, "**/*.json"), recursive=True)
        files = sorted(list(set(files)))
        if not files:
            raise FileNotFoundError(f"No .jsonl/.json files found under: {path}")
        return files
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return [path]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True, help="Input file or directory (json/jsonl)")
    ap.add_argument("--out_path", required=True, help="Output jsonl file")
    ap.add_argument("--default_system", default="你是一个有帮助的医疗助手。", help="Default system prompt if missing")

    ap.add_argument("--question_field", default="question")
    ap.add_argument("--chosen_field", default="response_chosen")
    ap.add_argument("--rejected_field", default="response_rejected")

    ap.add_argument("--skip_bad", action="store_true", help="Skip bad lines instead of crashing")
    args = ap.parse_args()

    in_files = iter_input_files(args.in_path)
    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    kept = 0
    skipped = 0

    with open(args.out_path, "w", encoding="utf-8") as w:
        for fp in in_files:
            # Support .json (list) and .jsonl (lines)
            if fp.endswith(".json") and not fp.endswith(".jsonl"):
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    data = [data]
                for obj in data:
                    try:
                        out = convert_obj(
                            obj,
                            default_system=args.default_system,
                            question_field=args.question_field,
                            chosen_field=args.chosen_field,
                            rejected_field=args.rejected_field,
                        )
                        w.write(json.dumps(out, ensure_ascii=False) + "\n")
                        kept += 1
                    except Exception as e:
                        skipped += 1
                        if not args.skip_bad:
                            raise
                        print(f"[SKIP] file={fp} err={e}")
            else:
                with open(fp, "r", encoding="utf-8") as f:
                    for line_no, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                            out = convert_obj(
                                obj,
                                default_system=args.default_system,
                                question_field=args.question_field,
                                chosen_field=args.chosen_field,
                                rejected_field=args.rejected_field,
                            )
                            w.write(json.dumps(out, ensure_ascii=False) + "\n")
                            kept += 1
                        except Exception as e:
                            skipped += 1
                            if not args.skip_bad:
                                raise
                            print(f"[SKIP] file={fp} line={line_no} err={e}")

    print(f"Done. wrote={kept} skipped={skipped} -> {args.out_path}")


if __name__ == "__main__":
    main()
