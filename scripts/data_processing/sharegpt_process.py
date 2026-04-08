#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import re
from typing import Any, Dict, List, Tuple, Iterable, Optional


def norm_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"[ \t]+\n", "\n", x)
    x = re.sub(r"\n{3,}", "\n\n", x)
    return x.strip()


def load_any(path: str) -> Iterable[Dict[str, Any]]:
    """
    Load JSONL or JSON array. If file starts with '[' -> JSON array else JSONL.
    """
    with open(path, "r", encoding="utf-8") as f:
        head = f.read(1)
        f.seek(0)
        if head == "[":
            data = json.load(f)
            if isinstance(data, list):
                for obj in data:
                    if isinstance(obj, dict):
                        yield obj
            return

        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


def extract_sharegpt_conversations(obj: Dict[str, Any]) -> List[Tuple[str, str]]:
    """
    Return list of (role, content) where role in {"user","assistant"} from ShareGPT:
    {"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
    Also accepts {"messages":[{"role":"user","content":"..."}...]} as fallback.
    """
    msgs = []

    if isinstance(obj.get("conversations"), list):
        for m in obj["conversations"]:
            if not isinstance(m, dict):
                continue
            frm = (m.get("from") or "").strip().lower()
            val = norm_text(m.get("value"))
            if not val:
                continue
            if frm in {"human", "user"}:
                msgs.append(("user", val))
            elif frm in {"gpt", "assistant", "bot"}:
                msgs.append(("assistant", val))
            else:
                # ignore unknown roles
                continue
        return msgs

    if isinstance(obj.get("messages"), list):
        for m in obj["messages"]:
            if not isinstance(m, dict):
                continue
            role = (m.get("role") or "").strip().lower()
            content = norm_text(m.get("content"))
            if not content:
                continue
            if role in {"user"}:
                msgs.append(("user", content))
            elif role in {"assistant"}:
                msgs.append(("assistant", content))
        return msgs

    return []


def format_history(history: List[Tuple[str, str]]) -> str:
    """
    history: list of (role, content) with role user/assistant.
    """
    parts = []
    for r, c in history:
        if r == "user":
            parts.append(f"用户：{c}")
        else:
            parts.append(f"助手：{c}")
    return "\n".join(parts).strip()


def truncate_tail(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    if len(s) <= max_chars:
        return s
    return s[-max_chars:].lstrip()


def build_sft_from_dialog(
    turns: List[Tuple[str, str]],
    max_history_user_turns: int,
    max_input_chars: int,
    include_current_user_in_input: bool,
) -> List[Dict[str, str]]:
    """
    Sliding window:
    For each user->assistant pair, create one sample:
      instruction = current user message
      input = formatted history (optional, last K user turns) (+ current user if include_current_user_in_input)
      output = assistant reply
    """
    samples = []
    rolling: List[Tuple[str, str]] = []

    i = 0
    while i < len(turns) - 1:
        r1, c1 = turns[i]
        r2, c2 = turns[i + 1]

        if r1 == "user" and r2 == "assistant":
            # keep last K user turns => last 2*K messages approx
            keep = 2 * max_history_user_turns if max_history_user_turns > 0 else 0
            hist = rolling[-keep:] if keep > 0 else []
            hist_str = format_history(hist)

            if include_current_user_in_input:
                # input contains history + current user at tail
                if hist_str:
                    inp = hist_str + "\n" + f"用户：{c1}"
                else:
                    inp = f"用户：{c1}"
                instruction = c1
            else:
                # match your current style: no history => input=""
                instruction = c1
                inp = hist_str  # may be ""

            inp = truncate_tail(inp, max_input_chars)

            samples.append({
                "instruction": instruction,
                "input": inp,
                "output": c2
            })

            rolling.append((r1, c1))
            rolling.append((r2, c2))
            i += 2
        else:
            rolling.append((r1, c1))
            i += 1

    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="ShareGPT json/jsonl file")
    ap.add_argument("--out", dest="out_path", required=True, help="Output jsonl path")
    ap.add_argument("--max_history_turns", type=int, default=4, help="Keep last N user turns as history (default 4)")
    ap.add_argument("--max_input_chars", type=int, default=4000, help="Max chars for input (tail kept). 0 disables.")
    ap.add_argument("--include_current_user_in_input", action="store_true",
                    help="Put current user message into input. If false, instruction holds it (recommended).")
    ap.add_argument("--add_meta", action="store_true", help="Add meta fields (source, convo_id, turn_index)")
    ap.add_argument("--drop_short_output", type=int, default=1, help="Drop samples with output shorter than N chars")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)

    total_items = 0
    total_samples = 0
    dropped_items = 0
    dropped_samples = 0

    with open(args.out_path, "w", encoding="utf-8") as fout:
        for idx, obj in enumerate(load_any(args.in_path)):
            total_items += 1
            turns = extract_sharegpt_conversations(obj)
            if not turns:
                dropped_items += 1
                continue

            samples = build_sft_from_dialog(
                turns=turns,
                max_history_user_turns=args.max_history_turns,
                max_input_chars=args.max_input_chars,
                include_current_user_in_input=args.include_current_user_in_input,
            )

            if not samples:
                dropped_items += 1
                continue

            convo_id = str(obj.get("id") or obj.get("conversation_id") or obj.get("uuid") or f"sharegpt_{idx}")

            for t_i, s in enumerate(samples):
                if len(norm_text(s["output"])) < args.drop_short_output:
                    dropped_samples += 1
                    continue

                if args.add_meta:
                    s["meta"] = {
                        "source": "sharegpt",
                        "convo_id": convo_id,
                        "turn_index": t_i
                    }

                fout.write(json.dumps(s, ensure_ascii=False) + "\n")
                total_samples += 1

    print(f"[done] items={total_items}, samples={total_samples}, "
          f"dropped_items={dropped_items}, dropped_samples={dropped_samples}")
    print(f"[saved] {args.out_path}")


if __name__ == "__main__":
    main()
