#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - Compare Base model vs LoRA/SFT model outputs.

Samples random + hard examples from validation set, generates answers
from both models, and saves side-by-side comparison.

Usage:
    python compare.py \
        --base_model models/base/Qwen2.5-7B-Instruct \
        --lora_model models/sft_merged \
        --valid_file data/sft/val/valid.jsonl \
        --output outputs/eval_results/base_vs_sft.jsonl \
        --system_prompt "你是一个专业的金融分析助手。" \
        --n_random 15 --n_hard 5
"""

import os
import json
import math
import random
import argparse
from typing import Dict, Any, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_user_text(ex: Dict[str, Any]) -> str:
    instr = (ex.get("instruction") or "").strip()
    ctx = (ex.get("input") or "").strip()
    return f"{ctx}\n{instr}".strip() if ctx else instr


def apply_chat(tokenizer, messages: List[Dict[str, str]], add_generation_prompt: bool) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


@torch.no_grad()
def generate_answer(model, tokenizer, prompt_text: str, max_new_tokens: int,
                    do_sample: bool, temperature: float, top_p: float,
                    repetition_penalty: float, no_repeat_ngram_size: int) -> str:
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "repetition_penalty": repetition_penalty,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    out = model.generate(**gen_kwargs)
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def assistant_only_loss(model, tokenizer, ex: Dict[str, Any], system_prompt: str) -> float:
    ans = (ex.get("output") or "").strip()
    if not ans:
        return float("-inf")

    user_text = build_user_text(ex)

    full_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ans},
    ]
    prompt_msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ""},
    ]

    full_text = apply_chat(tokenizer, full_msgs, add_generation_prompt=False)
    prompt_text = apply_chat(tokenizer, prompt_msgs, add_generation_prompt=False)

    full_ids = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)

    labels = full_ids.clone()
    pl = min(prompt_ids.size(1), labels.size(1))
    labels[:, :pl] = -100

    loss = model(input_ids=full_ids, labels=labels).loss.item()
    return float(loss)


def main():
    ap = argparse.ArgumentParser(description="Compare base model vs LoRA/SFT model")
    ap.add_argument("--base_model", required=True, help="Path to base model")
    ap.add_argument("--lora_model", required=True, help="Path to LoRA/SFT model (merged or adapter)")
    ap.add_argument("--adapter", default=None, help="Path to LoRA adapter (if lora_model is base for adapter)")
    ap.add_argument("--valid_file", required=True, help="Validation jsonl file")
    ap.add_argument("--output", required=True, help="Output jsonl file for comparison")
    ap.add_argument("--system_prompt", default="你是一个专业的金融分析助手。", help="System prompt")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_random", type=int, default=15, help="Number of random samples")
    ap.add_argument("--n_hard", type=int, default=5, help="Number of hard samples (by base loss)")
    ap.add_argument("--hard_pool", type=int, default=200, help="Pool size for picking hard samples")
    ap.add_argument("--max_new_tokens", type=int, default=1024)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling (default: greedy)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print("Loading tokenizer:", args.base_model)
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading base model:", args.base_model)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()

    print("Loading comparison model:", args.lora_model)
    lora_base = AutoModelForCausalLM.from_pretrained(
        args.lora_model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto",
    ).eval()

    if args.adapter:
        print("Loading LoRA adapter:", args.adapter)
        lora_base = PeftModel.from_pretrained(lora_base, args.adapter).eval()

    print("Loading validation data:", args.valid_file)
    ds = load_dataset("json", data_files=args.valid_file, split="train")
    n = len(ds)
    print("Valid size:", n)

    idxs = list(range(n))
    random.shuffle(idxs)

    # Random samples
    picked = idxs[:min(args.n_random, n)]

    # Hard samples
    cand = idxs[len(picked):len(picked) + min(args.hard_pool, n - len(picked))]
    scored: List[Tuple[float, int]] = []
    print(f"Scoring hard pool (size={len(cand)}) with base loss...")
    for i in cand:
        ex = ds[i]
        if not (ex.get("output") or "").strip():
            continue
        loss = assistant_only_loss(base, tok, ex, args.system_prompt)
        if math.isfinite(loss):
            scored.append((loss, i))

    scored.sort(reverse=True, key=lambda x: x[0])
    hard = [i for _, i in scored[:args.n_hard]]

    final_idxs = picked + hard
    print("Selected idxs:", final_idxs)

    def render_prompt(ex):
        msgs = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": build_user_text(ex)},
        ]
        return apply_chat(tok, msgs, add_generation_prompt=True)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )

    print(f"Writing outputs to: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        for rank, i in enumerate(final_idxs):
            ex = ds[i]
            prompt = render_prompt(ex)

            ans_base = generate_answer(base, tok, prompt, **gen_kwargs)
            ans_lora = generate_answer(lora_base, tok, prompt, **gen_kwargs)

            row = {
                "rank": rank,
                "idx_in_valid": int(i),
                "instruction": ex.get("instruction", ""),
                "input": ex.get("input", ""),
                "reference_output": ex.get("output", ""),
                "base_answer": ans_base,
                "lora_answer": ans_lora,
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("Done. Output:", args.output)


if __name__ == "__main__":
    main()
