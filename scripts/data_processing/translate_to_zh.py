#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - Translate English financial SFT data to Chinese using LLM.

Translates instruction-response pairs while preserving financial terminology.
Supports vLLM (fast batch), HuggingFace transformers, or OpenAI-compatible API.

Usage:
    # Using local vLLM (recommended for speed)
    python translate_to_zh.py \
        --input data/sft/raw/finance_alpaca \
        --output data/sft/processed/finance_alpaca_zh.jsonl \
        --model Qwen/Qwen2.5-72B-Instruct \
        --mode local_vllm \
        --batch_size 64

    # Using OpenAI-compatible API (e.g. vLLM server, Ollama)
    python translate_to_zh.py \
        --input data/sft/raw/sujet_finance \
        --output data/sft/processed/sujet_zh.jsonl \
        --model Qwen2.5-72B-Instruct \
        --mode api \
        --api_base http://localhost:8000/v1
"""

import argparse
import json
import os
import re
from glob import glob
from typing import List, Optional

from loguru import logger
from tqdm import tqdm

TRANSLATE_PROMPT = """请将以下英文金融问答对翻译为中文。要求：
1. 保留金融专业术语的准确性（如 P/E ratio → 市盈率, EBITDA → 息税折旧摊销前利润）
2. 数字、公式、百分比保持不变
3. 翻译应自然流畅，符合中文金融行业表达习惯
4. 直接输出翻译结果，不要添加解释

原文问题：
{instruction}

原文回答：
{output}

中文翻译：
问题："""


def load_data(path: str) -> list:
    """Load data from file or directory, supporting multiple formats."""
    items = []

    if os.path.isdir(path):
        files = glob(os.path.join(path, "**/*.jsonl"), recursive=True) + \
                glob(os.path.join(path, "**/*.json"), recursive=True) + \
                glob(os.path.join(path, "**/*.parquet"), recursive=True) + \
                glob(os.path.join(path, "**/*.csv"), recursive=True)
    else:
        files = [path]

    for fpath in files:
        if fpath.endswith(".parquet") or fpath.endswith(".csv"):
            try:
                import pandas as pd
                if fpath.endswith(".parquet"):
                    df = pd.read_parquet(fpath)
                else:
                    df = pd.read_csv(fpath)
                items.extend(df.to_dict("records"))
            except ImportError:
                logger.warning(f"pandas needed for {fpath}")
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            if fpath.endswith(".json") and not fpath.endswith(".jsonl"):
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        items.extend(data)
                    continue
                except json.JSONDecodeError:
                    pass

            for line in f:
                line = line.strip()
                if line:
                    try:
                        items.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    return items


def extract_instruction_output(item: dict) -> Optional[tuple]:
    """Extract instruction and output from various formats."""
    # Alpaca format
    if "instruction" in item:
        instr = item.get("instruction", "") or ""
        inp = item.get("input", "") or ""
        out = item.get("output", "") or ""
        full_instr = f"{instr}\n{inp}".strip() if inp else instr.strip()
        return full_instr, out.strip()

    # Conversations format
    if "conversations" in item:
        convs = item["conversations"]
        human = next((m["value"] for m in convs if m.get("from") == "human"), None)
        gpt = next((m["value"] for m in convs if m.get("from") == "gpt"), None)
        if human and gpt:
            return human.strip(), gpt.strip()

    # Sujet format
    if "user_prompt" in item:
        return (item.get("user_prompt") or "").strip(), (item.get("answer") or "").strip()

    # Generic
    if "question" in item and "answer" in item:
        return (item["question"] or "").strip(), (item["answer"] or "").strip()

    return None


def parse_translation(response: str) -> Optional[tuple]:
    """Parse translated question and answer from LLM response."""
    # Strip Qwen3 thinking tags if present
    response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()

    # Try to find "问题：" and "回答：" markers
    q_match = re.search(r'问题[：:]\s*(.*?)(?=\n*回答[：:]|\Z)', response, re.DOTALL)
    a_match = re.search(r'回答[：:]\s*(.*)', response, re.DOTALL)

    if q_match and a_match:
        return q_match.group(1).strip(), a_match.group(1).strip()

    # Fallback: split by first double newline
    parts = response.strip().split("\n\n", 1)
    if len(parts) == 2:
        q = re.sub(r'^问题[：:]\s*', '', parts[0]).strip()
        a = re.sub(r'^回答[：:]\s*', '', parts[1]).strip()
        if q and a:
            return q, a

    return None


def translate_batch_vllm(prompts: List[str], model_path: str, max_tokens: int,
                         tensor_parallel: int = 4, gpu_utilization: float = 0.90) -> List[str]:
    from vllm import LLM, SamplingParams
    llm = LLM(
        model_path,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel,
        gpu_memory_utilization=gpu_utilization,
        max_model_len=4096,
    )
    params = SamplingParams(temperature=0.3, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


def translate_batch_hf(prompts: List[str], model_path: str, max_tokens: int) -> List[str]:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    ).eval()

    results = []
    for prompt in tqdm(prompts, desc="Translating (HF)"):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_tokens, temperature=0.3,
                                 do_sample=True, pad_token_id=tokenizer.eos_token_id)
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append(gen)
    return results


def translate_batch_api(prompts: List[str], model: str, api_base: str,
                        api_key: str, max_tokens: int) -> List[str]:
    from openai import OpenAI
    client = OpenAI(base_url=api_base, api_key=api_key)
    results = []
    for prompt in tqdm(prompts, desc="Translating (API)"):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens, temperature=0.3,
            )
            results.append(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"API error: {e}")
            results.append("")
    return results


def is_chinese(text: str) -> bool:
    """Check if text is predominantly Chinese."""
    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
    return chinese_chars > len(text) * 0.3


def main():
    ap = argparse.ArgumentParser(description="Translate English financial data to Chinese")
    ap.add_argument("--input", required=True, help="Input file or directory")
    ap.add_argument("--output", required=True, help="Output jsonl file")
    ap.add_argument("--model", required=True, help="LLM model path for translation")
    ap.add_argument("--mode", choices=["local_vllm", "local_hf", "api"], default="local_vllm")
    ap.add_argument("--api_base", default="http://localhost:8000/v1")
    ap.add_argument("--api_key", default="dummy")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--max_samples", type=int, default=None, help="Limit input samples")
    ap.add_argument("--tensor_parallel", type=int, default=4,
                    help="vLLM tensor parallel size (number of GPUs)")
    ap.add_argument("--gpu_utilization", type=float, default=0.90,
                    help="vLLM GPU memory utilization ratio")
    ap.add_argument("--skip_chinese", action="store_true", default=True,
                    help="Skip already-Chinese samples (pass through directly)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load
    logger.info(f"Loading data from {args.input}")
    raw_items = load_data(args.input)
    logger.info(f"Loaded {len(raw_items)} items")

    if args.max_samples:
        raw_items = raw_items[:args.max_samples]

    # Extract and separate Chinese vs English
    to_translate = []  # (index, instruction, output)
    chinese_passthrough = []  # Already Chinese, keep as-is

    for i, item in enumerate(raw_items):
        pair = extract_instruction_output(item)
        if not pair or not pair[0] or not pair[1]:
            continue
        instr, out = pair

        if args.skip_chinese and is_chinese(instr):
            chinese_passthrough.append({"conversations": [
                {"from": "human", "value": instr},
                {"from": "gpt", "value": out},
            ]})
        else:
            to_translate.append((i, instr, out))

    logger.info(f"Chinese passthrough: {len(chinese_passthrough)}")
    logger.info(f"To translate: {len(to_translate)}")

    # Initialize model once (outside loop)
    vllm_engine = None
    vllm_params = None
    hf_model = None
    hf_tokenizer = None

    if args.mode == "local_vllm":
        from vllm import LLM, SamplingParams
        logger.info(f"Loading vLLM engine: {args.model} (tp={args.tensor_parallel}, gpu_util={args.gpu_utilization})")
        vllm_engine = LLM(
            args.model,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_utilization,
            max_model_len=4096,
        )
        vllm_params = SamplingParams(temperature=0.3, max_tokens=args.max_tokens)
        logger.info("vLLM engine ready")
    elif args.mode == "local_hf":
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        logger.info(f"Loading HF model: {args.model}")
        hf_tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        logger.info("HF model ready")

    # Translate in batches
    translated = []
    all_prompts = [TRANSLATE_PROMPT.format(instruction=instr, output=out) for _, instr, out in to_translate]

    if args.mode == "local_vllm":
        # vLLM: single call with all prompts (continuous batching handles internally)
        logger.info(f"Generating {len(all_prompts)} translations with vLLM...")
        outputs = vllm_engine.generate(all_prompts, vllm_params)
        responses = [o.outputs[0].text for o in outputs]

        for (_, orig_instr, orig_out), resp in zip(to_translate, responses):
            parsed = parse_translation(resp)
            if parsed:
                zh_q, zh_a = parsed
                if len(zh_q) > 5 and len(zh_a) > 10:
                    translated.append({"conversations": [
                        {"from": "human", "value": zh_q},
                        {"from": "gpt", "value": zh_a},
                    ]})
    else:
        # HF / API: process in batches
        for batch_start in range(0, len(to_translate), args.batch_size):
            batch = to_translate[batch_start:batch_start + args.batch_size]
            prompts = [TRANSLATE_PROMPT.format(instruction=instr, output=out) for _, instr, out in batch]

            if args.mode == "local_hf":
                responses = translate_batch_hf(prompts, args.model, args.max_tokens)
            elif args.mode == "api":
                responses = translate_batch_api(prompts, args.model, args.api_base, args.api_key, args.max_tokens)

            for (_, orig_instr, orig_out), resp in zip(batch, responses):
                parsed = parse_translation(resp)
                if parsed:
                    zh_q, zh_a = parsed
                    if len(zh_q) > 5 and len(zh_a) > 10:
                        translated.append({"conversations": [
                            {"from": "human", "value": zh_q},
                            {"from": "gpt", "value": zh_a},
                        ]})

            done = min(batch_start + args.batch_size, len(to_translate))
            logger.info(f"Translated {done}/{len(to_translate)}, success: {len(translated)}")

    # Merge and save
    all_data = chinese_passthrough + translated

    with open(args.output, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n==================== TRANSLATION RESULT ====================")
    print(f"Chinese passthrough:  {len(chinese_passthrough)}")
    print(f"Translated:           {len(translated)}")
    print(f"Total output:         {len(all_data)}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
