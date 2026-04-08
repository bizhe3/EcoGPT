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

def format_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Enforce <think>...</think><answer>...</answer> structure.
    Weight: 1.0 (configurable via args)
    """
    pattern = r"<think>[\s\S]+?</think>\s*<answer>[\s\S]+?</answer>"
    rewards = []
    for c in completions:
        if re.search(pattern, c):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards


def accuracy_reward(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    """
    Extract content from <answer> tags and compare with ground truth.
    - Numerical: absolute tolerance (0.5) OR relative tolerance (2%), whichever is more lenient
    - Text: normalized exact match, with partial credit for containment
    Weight: 2.0 (configurable via args)
    """
    rewards = []
    for c, gt in zip(completions, ground_truth):
        match = re.search(r"<answer>(.*?)</answer>", c, re.DOTALL)
        if not match:
            rewards.append(0.0)
            continue
        answer = match.group(1).strip()
        gt_str = str(gt).strip()

        # Try numerical matching
        try:
            a_val = float(answer.replace(",", "").replace("%", "").replace("亿", "").replace("万", ""))
            g_val = float(gt_str.replace(",", "").replace("%", "").replace("亿", "").replace("万", ""))
            abs_ok = abs(a_val - g_val) < 0.5
            rel_ok = (abs(a_val - g_val) / max(abs(g_val), 1.0)) < 0.02
            if abs_ok or rel_ok:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
            continue
        except ValueError:
            pass

        # Text matching: normalize then compare
        def normalize(s):
            s = re.sub(r'[，。、；：""''！？\s,.:;!?\-\'\"()]', '', s)
            return s.lower().strip()

        norm_a = normalize(answer)
        norm_g = normalize(gt_str)

        if norm_a == norm_g:
            rewards.append(1.0)
        elif norm_g in norm_a or norm_a in norm_g:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def length_reward(completions: List[str], **kwargs) -> List[float]:
    """
    Penalize too-short (<50 chars) or too-long (>2000 chars) reasoning.
    Encourages natural reasoning emergence via accuracy signal.
    Weight: 0.5 (configurable via args)
    """
    rewards = []
    for c in completions:
        think = re.search(r"<think>(.*?)</think>", c, re.DOTALL)
        if not think:
            rewards.append(0.0)
            continue
        length = len(think.group(1))
        if length < 50:
            rewards.append(0.2)
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

    # Load dataset
    train_dataset = load_grpo_data(args.train_data_path, args.max_train_samples)

    eval_dataset = None
    if args.val_data_path and os.path.exists(args.val_data_path):
        eval_dataset = load_grpo_data(args.val_data_path)

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

    # Build reward functions with weights
    reward_funcs = [format_reward, accuracy_reward, length_reward]
    reward_weights = [args.format_reward_weight, args.accuracy_reward_weight, args.length_reward_weight]
    logger.info(f"Reward weights: format={reward_weights[0]}, accuracy={reward_weights[1]}, length={reward_weights[2]}")

    # Initialize trainer
    trainer = GRPOTrainer(
        model=args.model_name_or_path,
        args=grpo_config,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
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
