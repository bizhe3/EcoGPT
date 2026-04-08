#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - Self-QA Data Generator (inspired by XuanYuan)

从非结构化金融文档中，使用大模型自动生成 instruction-response 对。
三步流程：
  1. 从文档中提取段落
  2. 用 LLM 为每段生成问题
  3. 用 LLM 基于原文生成答案

支持本地模型 (vLLM) 或 API 调用。

Usage:
    python self_qa_generate.py \
        --input data/sft/raw/fincorpus_sample.jsonl \
        --output data/sft/processed/self_qa.jsonl \
        --model Qwen/Qwen2.5-72B-Instruct \
        --mode local \
        --num_questions 3 \
        --max_samples 5000
"""

import argparse
import json
import os
import random
import re
from typing import List, Dict, Optional

from loguru import logger


# ============================================================
# Prompt templates for Self-QA
# ============================================================

QUESTION_GEN_PROMPT = """你是一名金融领域数据标注专家。请基于以下金融文本段落，生成 {n} 个高质量的问答问题。

要求：
1. 问题应覆盖文本中的关键金融信息（数据、概念、逻辑关系）
2. 问题应有明确答案，避免过于开放
3. 问题类型多样化：包括事实型、计算型、分析型
4. 每个问题单独一行，以"Q:"开头

金融文本：
{text}

请生成 {n} 个问题："""

ANSWER_GEN_PROMPT = """你是一名专业的金融分析师。请基于以下参考文本，准确回答问题。

要求：
1. 答案应完全基于参考文本中的信息
2. 如涉及数据，请准确引用
3. 回答需清晰、结构化
4. 如果参考文本无法回答该问题，请回答"根据提供的信息无法回答"

参考文本：
{text}

问题：{question}

回答："""

REASONING_QA_PROMPT = """你是一名金融教育专家。请基于以下金融文本，生成一个需要推理计算的问答对。

要求：
1. 问题应包含具体数值，需要计算推导才能得出答案
2. 答案应包含完整的推理过程，用 <think>推理过程</think><answer>最终答案</answer> 格式输出
3. 计算类型可以是：增长率、利润率、复合收益、估值倍数等

金融文本：
{text}

请生成一个计算推理问答对：
Q: """


# ============================================================
# Text extraction helpers
# ============================================================

def extract_paragraphs(text: str, min_len: int = 100, max_len: int = 1500) -> List[str]:
    """Split text into meaningful paragraphs."""
    # Split by double newline or common separators
    raw_paras = re.split(r'\n{2,}|。\s*\n', text)
    paragraphs = []
    for p in raw_paras:
        p = p.strip()
        if min_len <= len(p) <= max_len:
            paragraphs.append(p)
        elif len(p) > max_len:
            # Split long paragraphs at sentence boundaries
            sentences = re.split(r'(?<=[。！？])', p)
            chunk = ""
            for s in sentences:
                if len(chunk) + len(s) > max_len:
                    if len(chunk) >= min_len:
                        paragraphs.append(chunk.strip())
                    chunk = s
                else:
                    chunk += s
            if len(chunk) >= min_len:
                paragraphs.append(chunk.strip())
    return paragraphs


def load_corpus(path: str, text_field: str = "text", max_samples: Optional[int] = None) -> List[str]:
    """Load text corpus from jsonl/json/txt."""
    texts = []
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            texts = extract_paragraphs(content)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get(text_field, "") or obj.get("content", "") or obj.get("passage", "")
                    if text:
                        texts.extend(extract_paragraphs(text))
                except json.JSONDecodeError:
                    continue

    if max_samples:
        random.shuffle(texts)
        texts = texts[:max_samples]

    return texts


# ============================================================
# LLM generation backends
# ============================================================

def generate_local_vllm(prompts: List[str], model_path: str, max_tokens: int = 1024,
                        temperature: float = 0.7) -> List[str]:
    """Generate using local vLLM."""
    from vllm import LLM, SamplingParams

    llm = LLM(model_path, trust_remote_code=True, dtype="bfloat16")
    params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.generate(prompts, params)
    return [o.outputs[0].text for o in outputs]


def generate_local_hf(prompts: List[str], model_path: str, max_tokens: int = 1024,
                      temperature: float = 0.7) -> List[str]:
    """Generate using HuggingFace transformers (slower, lower memory)."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                do_sample=True, pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append(gen)
    return results


def generate_api(prompts: List[str], model: str, api_base: str, api_key: str,
                 max_tokens: int = 1024, temperature: float = 0.7) -> List[str]:
    """Generate using OpenAI-compatible API (works with vLLM server, Ollama, etc.)."""
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)
    results = []
    for prompt in prompts:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        results.append(resp.choices[0].message.content)
    return results


# ============================================================
# Self-QA pipeline
# ============================================================

def parse_questions(response: str) -> List[str]:
    """Parse generated questions from LLM response."""
    questions = []
    for line in response.strip().split("\n"):
        line = line.strip()
        # Match lines starting with Q: or number prefix
        line = re.sub(r'^(Q\s*[:：]\s*|\d+[.、)\s]+)', '', line).strip()
        if len(line) > 10:  # Skip too-short fragments
            questions.append(line)
    return questions


def main():
    ap = argparse.ArgumentParser(description="Self-QA financial data generator")
    ap.add_argument("--input", required=True, help="Input corpus file (jsonl/json/txt)")
    ap.add_argument("--output", required=True, help="Output jsonl file")
    ap.add_argument("--model", required=True, help="LLM model path or name")
    ap.add_argument("--mode", choices=["local_vllm", "local_hf", "api"], default="local_vllm",
                    help="Generation backend")
    ap.add_argument("--api_base", default="http://localhost:8000/v1", help="API base URL (for api mode)")
    ap.add_argument("--api_key", default="dummy", help="API key (for api mode)")
    ap.add_argument("--text_field", default="text", help="JSON field containing text content")
    ap.add_argument("--num_questions", type=int, default=3, help="Questions per paragraph")
    ap.add_argument("--max_samples", type=int, default=5000, help="Max paragraphs to process")
    ap.add_argument("--include_reasoning", action="store_true",
                    help="Also generate reasoning QA pairs (for GRPO data)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--batch_size", type=int, default=32, help="Batch size for vLLM generation")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load corpus
    logger.info(f"Loading corpus from {args.input}")
    paragraphs = load_corpus(args.input, args.text_field, args.max_samples)
    logger.info(f"Extracted {len(paragraphs)} paragraphs")

    if not paragraphs:
        logger.error("No paragraphs extracted. Check input file and text_field.")
        return

    # Select generation backend
    def generate(prompts):
        if args.mode == "local_vllm":
            return generate_local_vllm(prompts, args.model, temperature=args.temperature)
        elif args.mode == "local_hf":
            return generate_local_hf(prompts, args.model, temperature=args.temperature)
        elif args.mode == "api":
            return generate_api(prompts, args.model, args.api_base, args.api_key,
                                temperature=args.temperature)

    # Step 1: Generate questions
    logger.info("Step 1/2: Generating questions...")
    q_prompts = [
        QUESTION_GEN_PROMPT.format(text=p, n=args.num_questions)
        for p in paragraphs
    ]

    all_qa_pairs = []

    # Process in batches
    for batch_start in range(0, len(q_prompts), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(q_prompts))
        batch_prompts = q_prompts[batch_start:batch_end]
        batch_paragraphs = paragraphs[batch_start:batch_end]

        q_responses = generate(batch_prompts)

        # Parse questions and prepare answer prompts
        a_prompts = []
        qa_meta = []

        for para, q_resp in zip(batch_paragraphs, q_responses):
            questions = parse_questions(q_resp)
            for q in questions[:args.num_questions]:
                a_prompts.append(ANSWER_GEN_PROMPT.format(text=para, question=q))
                qa_meta.append({"paragraph": para, "question": q})

        # Step 2: Generate answers
        if a_prompts:
            a_responses = generate(a_prompts)

            for meta, answer in zip(qa_meta, a_responses):
                answer = answer.strip()
                if len(answer) < 20 or "无法回答" in answer:
                    continue
                all_qa_pairs.append({
                    "conversations": [
                        {"from": "human", "value": meta["question"]},
                        {"from": "gpt", "value": answer},
                    ]
                })

        logger.info(f"  Processed {batch_end}/{len(paragraphs)} paragraphs, "
                     f"generated {len(all_qa_pairs)} QA pairs so far")

    # Optional: Generate reasoning QA for GRPO
    if args.include_reasoning:
        logger.info("Generating reasoning QA pairs for GRPO...")
        r_prompts = [
            REASONING_QA_PROMPT.format(text=p) for p in paragraphs[:len(paragraphs) // 3]
        ]

        reasoning_output_path = args.output.replace(".jsonl", "_reasoning.jsonl")
        reasoning_pairs = []

        for batch_start in range(0, len(r_prompts), args.batch_size):
            batch = r_prompts[batch_start:batch_start + args.batch_size]
            r_responses = generate(batch)

            for resp in r_responses:
                # Try to parse Q: ... and answer with <think><answer> tags
                q_match = re.search(r'Q\s*[:：]\s*(.+?)(?:\n|$)', resp)
                a_match = re.search(r'<answer>(.*?)</answer>', resp, re.DOTALL)
                if q_match and a_match:
                    reasoning_pairs.append({
                        "prompt": q_match.group(1).strip(),
                        "ground_truth": a_match.group(1).strip(),
                    })

        if reasoning_pairs:
            with open(reasoning_output_path, "w", encoding="utf-8") as f:
                for pair in reasoning_pairs:
                    f.write(json.dumps(pair, ensure_ascii=False) + "\n")
            logger.info(f"Saved {len(reasoning_pairs)} reasoning pairs to {reasoning_output_path}")

    # Save SFT data
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in all_qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    logger.info(f"Done! Generated {len(all_qa_pairs)} SFT pairs")
    logger.info(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
