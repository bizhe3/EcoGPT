# -*- coding: utf-8 -*-
"""
EcoGPT - Merge LoRA-trained Reward Model into base model.

Usage:
    python merge_rm.py \
        --base_model models/base/Qwen2.5-7B-Instruct \
        --adapter models/reward_model/lora \
        --output models/reward_model/merged
"""

import argparse
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from loguru import logger


def main():
    ap = argparse.ArgumentParser(description="Merge LoRA reward model into base model")
    ap.add_argument("--base_model", required=True, help="Path to base model")
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    ap.add_argument("--output", required=True, help="Output directory for merged model")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    args = ap.parse_args()

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    logger.info(f"Loading base model: {args.base_model}")
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=1,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    )

    logger.info(f"Loading LoRA adapter: {args.adapter}")
    model = PeftModel.from_pretrained(model, args.adapter)

    logger.info("Merging weights...")
    model = model.merge_and_unload()

    os.makedirs(args.output, exist_ok=True)
    tok.save_pretrained(args.output)
    model.save_pretrained(args.output)

    logger.info(f"Merged RM saved to: {args.output}")
    logger.info(f"num_labels: {model.config.num_labels}")
    logger.info(f"architectures: {model.config.architectures}")


if __name__ == "__main__":
    main()
