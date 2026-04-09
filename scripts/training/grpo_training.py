# -*- coding: utf-8 -*-
"""
EcoGPT - GRPO (Group Relative Policy Optimization) Training Script
Based on TRL GRPOTrainer for financial domain reinforcement learning.

Reference:
  - DeepSeekMath (2024): GRPO algorithm
  - Fin-R1 (arxiv:2503.16252): Financial reasoning with GRPO
  - TRL documentation: https://huggingface.co/docs/trl/main/en/grpo_trainer
"""

import os
import re
import sys
import json
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import load_dataset
from loguru import logger
from peft import LoraConfig, TaskType
from transformers import AutoTokenizer, HfArgumentParser

from trl import GRPOTrainer, GRPOConfig

os.environ["TOKENIZERS_PARALLELISM"] = "FALSE"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# Project paths (adapt to your environment)
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs", "grpo")


# ============================================================
# Arguments
# ============================================================
@dataclass
class GRPOScriptArguments:
    """Arguments for GRPO training script."""

    # Model
    model_name_or_path: str = field(
        default=os.path.join(MODELS_DIR, "sft_merged"),
        metadata={"help": "Path to the SFT-merged model (Stage 1 output)."},
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Tokenizer path. Defaults to model_name_or_path."},
    )
    trust_remote_code: bool = field(default=True)

    # Data
    train_data_path: str = field(
        default=os.path.join(DATA_DIR, "grpo", "train"),
        metadata={"help": "Directory or file containing GRPO training data (jsonl with prompt + ground_truth)."},
    )
    val_data_path: Optional[str] = field(
        default=os.path.join(DATA_DIR, "grpo", "val"),
        metadata={"help": "Optional validation data path."},
    )
    max_train_samples: Optional[int] = field(default=None)

    # LoRA
    use_peft: bool = field(default=True, metadata={"help": "Use LoRA for GRPO (recommended for memory)."})
    lora_rank: int = field(default=8)
    lora_alpha: float = field(default=16.0)
    lora_dropout: float = field(default=0.05)

    # GRPO specific
    num_generations: int = field(default=8, metadata={"help": "Group size G: completions per prompt."})
    temperature: float = field(default=0.9, metadata={"help": "Sampling temperature for rollout."})
    max_completion_length: int = field(default=1024, metadata={"help": "Max tokens for each completion."})
    beta: float = field(default=0.04, metadata={"help": "KL penalty coefficient. 0.04 is safe default."})
    loss_type: str = field(default="grpo", metadata={"help": "Loss variant: grpo, dapo, dr_grpo."})
    use_vllm: bool = field(default=False, metadata={"help": "Use vLLM for fast rollout (recommended)."})

    # Reward weights
    format_reward_weight: float = field(default=1.0)
    accuracy_reward_weight: float = field(default=2.0)
    length_reward_weight: float = field(default=0.5)

    # Training
    output_dir: str = field(default=os.path.join(MODELS_DIR, "grpo_lora"))
    learning_rate: float = field(default=5e-6, metadata={"help": "LoRA GRPO learning rate."})
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=16)
    num_train_epochs: int = field(default=2)
    max_grad_norm: float = field(default=0.5, metadata={"help": "Lower for training stability."})
    logging_steps: int = field(default=10)
    save_steps: int = field(default=200)
    bf16: bool = field(default=True)
    report_to: str = field(default="tensorboard")
    logging_dir: str = field(default=LOGS_DIR)


# ============================================================
# Reward Functions (Financial Domain)
# ============================================================
# No dependency on <think>/<answer> tags.
# Works with any model output format.

def _extract_number(s: str):
    """Extract first number from string, handling Chinese units."""
    s = re.sub(r'[,，\s]', '', s)
    match = re.search(r'[-+]?\d*\.?\d+', s)
    if match:
        num = float(match.group())
        after = s[match.end():]
        if '万亿' in after:
            num *= 1e12
        elif '亿' in after:
            num *= 1e8
        elif '万' in after:
            num *= 1e4
        return num
    return None


def _normalize_text(s: str) -> str:
    """Normalize text for comparison."""
    s = re.sub(r'[，。、；：""''！？\s,.:;!?\-\'\"()（）]', '', s)
    return s.lower().strip()


def accuracy_reward(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    """
    Check if the completion contains the correct answer.
    Searches the entire completion for the ground truth value.
    - Numerical: extract all numbers, check if any match gt (5% tolerance)
    - Text: check if gt text appears in completion
    Weight: 2.0
    """
    rewards = []
    for c, gt in zip(completions, ground_truth):
        gt_str = str(gt).strip()
        if not gt_str:
            rewards.append(0.0)
            continue

        # Try numerical matching
        gt_num = _extract_number(gt_str)
        if gt_num is not None:
            # Extract all numbers from completion
            nums = re.findall(r'[-+]?\d*\.?\d+', c.replace(',', '').replace('，', ''))
            matched = False
            for n_str in nums:
                try:
                    n_val = float(n_str)
                    # Check with tolerance
                    if gt_num == 0 and n_val == 0:
                        matched = True
                        break
                    if abs(n_val - gt_num) < 0.5:
                        matched = True
                        break
                    if gt_num != 0 and abs(n_val - gt_num) / abs(gt_num) < 0.05:
                        matched = True
                        break
                except ValueError:
                    continue
            rewards.append(1.0 if matched else 0.0)
            continue

        # Text matching: check if normalized gt appears in completion
        norm_gt = _normalize_text(gt_str)
        norm_c = _normalize_text(c)

        if norm_gt in norm_c:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def length_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Reward appropriate response length.
    Too short = probably no reasoning. Too long = verbose/repetitive.
    Weight: 0.5
    """
    rewards = []
    for c in completions:
        length = len(c)
        if length < 20:
            rewards.append(0.0)
        elif length < 50:
            rewards.append(0.3)
        elif length > 2000:
            rewards.append(0.3)
        else:
            rewards.append(1.0)
    return rewards


# ============================================================
# Data Loading
# ============================================================
def load_grpo_data(path: str, max_samples: Optional[int] = None):
    """
    Load GRPO dataset. Expected format (jsonl):
    {"prompt": "...", "ground_truth": "..."}

    Or with system message:
    {"prompt": "...", "ground_truth": "...", "system": "..."}
    """
    if os.path.isdir(path):
        from glob import glob
        files = glob(os.path.join(path, "**/*.jsonl"), recursive=True) + \
                glob(os.path.join(path, "**/*.json"), recursive=True)
        if not files:
            raise FileNotFoundError(f"No jsonl/json files found in {path}")
        dataset = load_dataset("json", data_files=files, split="train")
    else:
        dataset = load_dataset("json", data_files=path, split="train")

    if max_samples is not None and max_samples > 0:
        dataset = dataset.select(range(min(len(dataset), max_samples)))

    logger.info(f"Loaded {len(dataset)} GRPO samples from {path}")
    return dataset


# ============================================================
# Main
# ============================================================
def main():
    parser = HfArgumentParser(GRPOScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    logger.info(f"GRPO args: {args}")

    # Load tokenizer
    tokenizer_path = args.tokenizer_name_or_path or args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset and format prompts with system instruction
    SYSTEM_PROMPT = "你是一个专业的金融分析助手。请仔细分析问题并给出准确的答案。"

    def format_prompt(example):
        """Wrap raw prompt into chat format with system instruction."""
        prompt = example["prompt"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        return {"prompt": formatted}

    train_dataset = load_grpo_data(args.train_data_path, args.max_train_samples)
    train_dataset = train_dataset.map(format_prompt)

    eval_dataset = None
    if args.val_data_path and os.path.exists(args.val_data_path):
        eval_dataset = load_grpo_data(args.val_data_path)
        eval_dataset = eval_dataset.map(format_prompt)

    # Configure GRPO
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        num_generations=args.num_generations,
        temperature=args.temperature,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
        loss_type=args.loss_type,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=True,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        logging_dir=args.logging_dir,
        save_steps=args.save_steps,
        report_to=args.report_to,
        use_vllm=args.use_vllm,
    )

    # Configure LoRA
    peft_config = None
    if args.use_peft:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            inference_mode=False,
        )
        logger.info(f"Using LoRA: rank={args.lora_rank}, alpha={args.lora_alpha}")

    # Build weighted reward function
    acc_w = args.accuracy_reward_weight
    len_w = args.length_reward_weight
    logger.info(f"Reward weights: accuracy={acc_w}, length={len_w}")

    def combined_reward(completions: list, ground_truth: list = None, **kwargs) -> list:
        acc_scores = accuracy_reward(completions, ground_truth=ground_truth, **kwargs)
        len_scores = length_reward(completions, **kwargs)
        return [
            acc_w * a + len_w * l
            for a, l in zip(acc_scores, len_scores)
        ]

    # Initialize trainer
    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        args=grpo_config,
        reward_funcs=combined_reward,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Train
    logger.info("*** Starting GRPO Training ***")
    logger.info(f"  Model: {args.model_name_or_path}")
    logger.info(f"  Num samples: {len(train_dataset)}")
    logger.info(f"  Num generations (G): {args.num_generations}")
    logger.info(f"  Beta (KL): {args.beta}")
    logger.info(f"  Loss type: {args.loss_type}")
    logger.info(f"  LR: {args.learning_rate}")

    trainer.train()

    # Save
    logger.info(f"Saving GRPO model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("GRPO training complete.")


if __name__ == "__main__":
    main()
