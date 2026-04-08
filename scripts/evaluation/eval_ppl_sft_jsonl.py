#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
from typing import Dict, List, Tuple, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def build_prompt_plain(instruction: str, inp: str) -> str:
    instruction = (instruction or "").strip()
    inp = (inp or "").strip()
    if inp:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


def build_prompt_qwen_chat(tokenizer, instruction: str, inp: str, system: Optional[str]) -> str:
    """
    Use tokenizer chat_template to create a prompt ending with assistant turn (generation prompt).
    This gives you "prompt_text" (no answer included).
    """
    instruction = (instruction or "").strip()
    inp = (inp or "").strip()

    user_content = instruction if not inp else f"{instruction}\n\n{inp}"

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_content})

    # add_generation_prompt=True makes the final assistant prefix included, ready for answer tokens
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt_text


def load_jsonl(path: str, limit: int = 0) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            data.append(obj)
            if limit and len(data) >= limit:
                break
    if not data:
        raise RuntimeError("No valid samples found in jsonl.")
    return data


@torch.no_grad()
def ppl_conditional_sliding(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    device: torch.device,
    max_length: int,
    stride: int,
) -> Tuple[float, int]:
    """
    Conditional PPL with sliding window:
    - We evaluate ONLY answer tokens, but the model can attend to prompt + previous answer tokens.
    - Implementation: create full sequence = prompt + answer, set labels=-100 for prompt,
      then sliding-window over full sequence while keeping correct masking for overlap.
    Returns (total_nll, total_answer_tokens)
    """
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)  # [1, L]
    prompt_len = prompt_ids.size(1)
    full_len = full_ids.size(1)

    # labels: ignore prompt tokens; compute loss on answer tokens (next-token positions)
    labels_full = full_ids.clone()
    labels_full[:, :prompt_len] = -100  # ignore prompt positions

    total_nll = 0.0
    total_tokens = 0

    # slide over full sequence
    for start in range(0, full_len - 1, stride):
        end = min(start + max_length, full_len)
        input_chunk = full_ids[:, start:end].to(device)
        labels_chunk = labels_full[:, start:end].to(device)

        # For overlapping windows, ignore the tokens already evaluated in previous window
        if start > 0:
            overlap = max_length - stride
            ignore_len = min(overlap, labels_chunk.size(1) - 1)
            labels_chunk[:, :ignore_len] = -100

        out = model(input_ids=input_chunk, labels=labels_chunk)
        loss = out.loss  # mean over non-ignored tokens

        # count valid predicted tokens (positions excluding -100 in labels[:, 1:])
        valid = (labels_chunk[:, 1:] != -100).sum().item()
        if valid <= 0:
            continue

        total_nll += loss.item() * valid
        total_tokens += valid

    return total_nll, total_tokens


@torch.no_grad()
def ppl_conditional_simple(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    answer_ids: torch.Tensor,
    device: torch.device,
    max_length: int,
) -> Tuple[float, int]:
    """
    Conditional PPL without sliding:
    - Truncate from the left if too long (keeps the tail, usually includes answer)
    - Evaluate only answer tokens by label masking.
    Returns (total_nll, total_answer_tokens)
    """
    full_ids = torch.cat([prompt_ids, answer_ids], dim=1)  # [1, L]
    prompt_len = prompt_ids.size(1)
    full_len = full_ids.size(1)

    if full_len > max_length:
        # keep the last max_length tokens
        cut = full_len - max_length
        full_ids = full_ids[:, cut:]
        # prompt_len shifts
        prompt_len = max(0, prompt_len - cut)

    labels = full_ids.clone()
    labels[:, :prompt_len] = -100

    full_ids = full_ids.to(device)
    labels = labels.to(device)

    out = model(input_ids=full_ids, labels=labels)
    loss = out.loss
    valid = (labels[:, 1:] != -100).sum().item()
    if valid <= 0:
        return 0.0, 0

    return loss.item() * valid, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model path or repo id")
    ap.add_argument("--data", required=True, help="jsonl with fields: instruction, input, output")
    ap.add_argument("--limit", type=int, default=0, help="limit samples (0=all)")

    ap.add_argument("--template", choices=["qwen", "plain"], default="qwen",
                    help="qwen: use tokenizer chat_template; plain: simple Instruction/Input/Response format")
    ap.add_argument("--system", type=str, default="你是一个有帮助的助手。", help="system prompt (only for qwen template)")

    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument("--dtype", choices=["auto", "fp32", "fp16", "bf16"], default="auto")

    ap.add_argument("--mode", choices=["sliding", "simple"], default="sliding")
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=512, help="only for sliding mode")

    ap.add_argument("--add_eos_to_answer", action="store_true",
                    help="append eos_token to answer for evaluation (recommended for SFT-style data)")

    args = ap.parse_args()

    # device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # dtype
    if args.dtype == "fp32":
        torch_dtype = torch.float32
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        elif device.type == "cuda":
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    model.eval()
    model.to(device)

    data = load_jsonl(args.data, limit=args.limit)

    total_nll = 0.0
    total_tokens = 0

    for obj in data:
        instruction = (obj.get("instruction") or "").strip()
        inp = (obj.get("input") or "").strip()
        output = (obj.get("output") or "").strip()

        if not output:
            continue

        # build prompt text (NO answer included)
        if args.template == "qwen":
            prompt_text = build_prompt_qwen_chat(tokenizer, instruction, inp, args.system)
        else:
            prompt_text = build_prompt_plain(instruction, inp)

        # tokenize prompt and answer separately
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"]
        ans_text = output + (tokenizer.eos_token if args.add_eos_to_answer and tokenizer.eos_token else "")
        answer_ids = tokenizer(ans_text, return_tensors="pt", add_special_tokens=False)["input_ids"]

        if answer_ids.size(1) < 1:
            continue

        if args.mode == "sliding":
            nll, ntok = ppl_conditional_sliding(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                answer_ids=answer_ids,
                device=device,
                max_length=args.max_length,
                stride=args.stride,
            )
        else:
            nll, ntok = ppl_conditional_simple(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                answer_ids=answer_ids,
                device=device,
                max_length=args.max_length,
            )

        total_nll += nll
        total_tokens += ntok

    if total_tokens == 0:
        raise RuntimeError("No valid answer tokens were evaluated. Check data / template / max_length.")

    ppl = math.exp(total_nll / total_tokens)

    print(f"Model: {args.model}")
    print(f"Data:  {args.data} (samples={len(data)})")
    print(f"Template: {args.template} | Mode: {args.mode}")
    print(f"Evaluated answer tokens: {total_tokens}")
    print(f"Conditional PPL (answer-only): {ppl:.6f}")


if __name__ == "__main__":
    main()
