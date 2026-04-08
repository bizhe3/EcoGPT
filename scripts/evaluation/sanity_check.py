# -*- coding: utf-8 -*-
"""
EcoGPT - Sanity Check: SFT → GRPO Transition Validator
Evaluates pass@K on the GRPO training set using the SFT model.

Decision matrix:
  pass@K < 10%  → Cold start failure. Need Reasoning SFT first.
  10% ~ 80%     → Optimal range. Proceed to GRPO.
  > 80%         → Tasks too easy. Use harder GRPO data.

Usage:
    python sanity_check.py \
        --model_path /path/to/sft_merged \
        --data_path /path/to/grpo/train \
        --num_samples 200 \
        --num_generations 8 \
        --output_dir /path/to/outputs/sanity_check
"""

import argparse
import json
import os
import re
import sys
from typing import List, Optional

import torch
from loguru import logger


def extract_answer(text: str) -> Optional[str]:
    """Extract content from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def normalize(s: str) -> str:
    s = re.sub(r'[，。、；：""''！？\s,.:;!?\-\'\"()]', '', s)
    return s.lower().strip()


def check_accuracy(answer: str, ground_truth: str) -> bool:
    """Check if answer matches ground truth (numerical or text)."""
    if answer is None:
        return False

    gt_str = str(ground_truth).strip()

    # Numerical
    try:
        a_val = float(answer.replace(",", "").replace("%", "").replace("亿", "").replace("万", ""))
        g_val = float(gt_str.replace(",", "").replace("%", "").replace("亿", "").replace("万", ""))
        abs_ok = abs(a_val - g_val) < 0.5
        rel_ok = (abs(a_val - g_val) / max(abs(g_val), 1.0)) < 0.02
        return abs_ok or rel_ok
    except ValueError:
        pass

    # Text
    return normalize(answer) == normalize(gt_str) or \
           normalize(gt_str) in normalize(answer) or \
           normalize(answer) in normalize(gt_str)


def main():
    ap = argparse.ArgumentParser(description="Sanity check SFT model on GRPO dataset")
    ap.add_argument("--model_path", required=True, help="Path to SFT merged model")
    ap.add_argument("--data_path", required=True, help="GRPO training data (jsonl with prompt + ground_truth)")
    ap.add_argument("--num_samples", type=int, default=200, help="Number of samples to evaluate")
    ap.add_argument("--num_generations", type=int, default=8, help="K for pass@K")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--output_dir", default="./outputs/sanity_check")
    ap.add_argument("--use_vllm", action="store_true", help="Use vLLM for faster inference")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    from glob import glob
    if os.path.isdir(args.data_path):
        files = glob(os.path.join(args.data_path, "**/*.jsonl"), recursive=True) + \
                glob(os.path.join(args.data_path, "**/*.json"), recursive=True)
    else:
        files = [args.data_path]

    samples = []
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "prompt" in obj and "ground_truth" in obj:
                        samples.append(obj)
                except json.JSONDecodeError:
                    continue

    if not samples:
        logger.error(f"No valid samples found in {args.data_path}")
        sys.exit(1)

    samples = samples[:args.num_samples]
    logger.info(f"Evaluating {len(samples)} samples with pass@{args.num_generations}")

    # Generate completions
    if args.use_vllm:
        try:
            from vllm import LLM, SamplingParams
            model = LLM(args.model_path, trust_remote_code=True)
            sampling_params = SamplingParams(
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                n=args.num_generations,
            )
            prompts = [s["prompt"] for s in samples]
            all_outputs = model.generate(prompts, sampling_params)

            results = []
            for sample, request_output in zip(samples, all_outputs):
                completions = [o.text for o in request_output.outputs]
                gt = sample["ground_truth"]
                pass_k = any(check_accuracy(extract_answer(c), gt) for c in completions)
                results.append({
                    "prompt": sample["prompt"][:200],
                    "ground_truth": gt,
                    "pass_at_k": pass_k,
                    "num_correct": sum(1 for c in completions if check_accuracy(extract_answer(c), gt)),
                    "sample_answers": [extract_answer(c) for c in completions[:3]],
                })
        except ImportError:
            logger.error("vLLM not installed. Use --use_vllm=false or install: pip install vllm")
            sys.exit(1)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        )
        model.eval()

        results = []
        for idx, sample in enumerate(samples):
            prompt = sample["prompt"]
            gt = sample["ground_truth"]
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            completions = []
            with torch.no_grad():
                for _ in range(args.num_generations):
                    output = model.generate(
                        **inputs,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    gen_text = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                    completions.append(gen_text)

            pass_k = any(check_accuracy(extract_answer(c), gt) for c in completions)
            results.append({
                "prompt": prompt[:200],
                "ground_truth": gt,
                "pass_at_k": pass_k,
                "num_correct": sum(1 for c in completions if check_accuracy(extract_answer(c), gt)),
                "sample_answers": [extract_answer(c) for c in completions[:3]],
            })

            if (idx + 1) % 20 == 0:
                current_rate = sum(r["pass_at_k"] for r in results) / len(results)
                logger.info(f"Progress: {idx + 1}/{len(samples)}, current pass@{args.num_generations} = {current_rate:.2%}")

    # Compute statistics
    pass_rate = sum(r["pass_at_k"] for r in results) / len(results)
    avg_correct = sum(r["num_correct"] for r in results) / len(results)
    format_rate = sum(1 for r in results if any(a is not None for a in r["sample_answers"])) / len(results)

    # Decision
    if pass_rate < 0.10:
        decision = "COLD_START_FAILURE"
        advice = "pass@K < 10%. Model cannot solve these tasks. Need Reasoning SFT with CoT data first."
    elif pass_rate > 0.80:
        decision = "TASKS_TOO_EASY"
        advice = "pass@K > 80%. Tasks are too simple for GRPO. Use harder financial reasoning data."
    else:
        decision = "READY_FOR_GRPO"
        advice = f"pass@K = {pass_rate:.2%} is in the optimal range [10%, 80%]. Proceed to GRPO training."

    # Report
    report = {
        "model": args.model_path,
        "data": args.data_path,
        "num_samples": len(samples),
        "num_generations_K": args.num_generations,
        "temperature": args.temperature,
        f"pass_at_{args.num_generations}": pass_rate,
        "avg_correct_per_prompt": avg_correct,
        "format_compliance_rate": format_rate,
        "decision": decision,
        "advice": advice,
    }

    # Print results
    print("\n" + "=" * 60)
    print("  EcoGPT Sanity Check Report")
    print("=" * 60)
    print(f"  Model:         {args.model_path}")
    print(f"  Samples:       {len(samples)}")
    print(f"  pass@{args.num_generations}:        {pass_rate:.2%}")
    print(f"  Avg correct:   {avg_correct:.2f} / {args.num_generations}")
    print(f"  Format rate:   {format_rate:.2%}")
    print(f"  Decision:      {decision}")
    print(f"  Advice:        {advice}")
    print("=" * 60)

    # Save
    report_path = os.path.join(args.output_dir, "sanity_check_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    details_path = os.path.join(args.output_dir, "sanity_check_details.jsonl")
    with open(details_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info(f"Report saved to {report_path}")
    logger.info(f"Details saved to {details_path}")


if __name__ == "__main__":
    main()
