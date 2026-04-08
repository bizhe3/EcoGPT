# -*- coding: utf-8 -*-
import argparse, json, os
from typing import Any, Dict, Iterable, List


def iter_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                raise ValueError(f"Bad JSONL at {path}:{i}: {e}")


def read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    if path.endswith(".jsonl"):
        return list(iter_jsonl(path))
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
            return obj["data"]
        raise ValueError("JSON must be list or {'data': list}")
    raise ValueError("Input must be .json/.jsonl")


def ensure_history(history: Any) -> List[List[str]]:
    if not history:
        return []
    out = []
    if isinstance(history, list):
        for x in history:
            if isinstance(x, (list, tuple)) and len(x) == 2:
                out.append([str(x[0]), str(x[1])])
    return out


def to_sharegpt(system_prompt: str, history: List[List[str]], question: str, answer: str = None):
    conv = []
    for q, a in history:
        q = (q or "").strip()
        a = (a or "").strip()
        if q:
            conv.append({"from": "human", "value": q})
        if a:
            conv.append({"from": "gpt", "value": a})
    question = (question or "").strip()
    if question:
        conv.append({"from": "human", "value": question})
    if answer is not None:
        ans = (answer or "").strip()
        if ans:
            conv.append({"from": "gpt", "value": ans})
    return {"system_prompt": (system_prompt or "").strip(), "conversations": conv}


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--mode", required=True, choices=["ppo_sharegpt", "sft_sharegpt", "rm"])
    args = ap.parse_args()

    data = read_json_or_jsonl(args.in_file)
    out = []
    dropped = 0

    for ex in data:
        system = ex.get("system", "") or ex.get("system_prompt", "") or ""
        history = ensure_history(ex.get("history", []))
        question = ex.get("question", "") or ""
        chosen = ex.get("response_chosen", "") or ex.get("chosen", "") or ""
        rejected = ex.get("response_rejected", "") or ex.get("rejected", "") or ""

        if not question:
            dropped += 1
            continue

        if args.mode == "ppo_sharegpt":
            # 对话以 human 结尾，PPO 负责生成
            out.append(to_sharegpt(system, history, question, answer=None))

        elif args.mode == "sft_sharegpt":
            if not chosen:
                dropped += 1
                continue
            out.append(to_sharegpt(system, history, question, answer=chosen))

        elif args.mode == "rm":
            if not chosen or not rejected:
                dropped += 1
                continue
            out.append({
                "system": system,
                "history": history,
                "question": question,
                "response_chosen": chosen,
                "response_rejected": rejected
            })

    write_jsonl(args.out_file, out)
    print(f"Done mode={args.mode} in={len(data)} out={len(out)} dropped={dropped}")
    print("Saved:", args.out_file)


if __name__ == "__main__":
    main()
