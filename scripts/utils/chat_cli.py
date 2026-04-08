# chat_cli.py
# -*- coding: utf-8 -*-

import argparse
import sys
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from peft import PeftModel
except Exception:
    PeftModel = None


def build_prompt(tokenizer, system_prompt, history, user_query):
    """
    兼容两种方式：
    1) tokenizer.apply_chat_template (推荐，Qwen/Llama等多支持)
    2) 退化为简单拼接
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for u, a in history:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})

    messages.append({"role": "user", "content": user_query})

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # fallback：非常朴素的拼接（如果模型无chat_template）
        text = ""
        if system_prompt:
            text += f"[SYSTEM]\n{system_prompt}\n"
        for u, a in history:
            text += f"[USER]\n{u}\n[ASSISTANT]\n{a}\n"
        text += f"[USER]\n{user_query}\n[ASSISTANT]\n"
        return text


@torch.inference_mode()
def generate_one(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.9, repetition_penalty=1.05):
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
    )
    out = model.generate(**inputs, **gen_kwargs)

    # 只截取新生成部分
    gen_ids = out[0][inputs["input_ids"].shape[-1]:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Base模型路径或HF repo id")
    parser.add_argument("--lora_path", type=str, default="", help="可选：LoRA适配器路径（PEFT）")
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--device_map", type=str, default="auto", help="通常用auto")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--system", type=str, default="你是一个严谨的医疗助手，回答需说明不确定性并建议就医。")
    parser.add_argument("--history_turns", type=int, default=6, help="保留最近N轮对话")
    args = parser.parse_args()

    # dtype
    if args.dtype == "auto":
        torch_dtype = None
    elif args.dtype == "fp16":
        torch_dtype = torch.float16
    elif args.dtype == "bf16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)

    # pad_token处理（很多因果LM没有pad）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model: {args.model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
    )
    model.eval()

    if args.lora_path:
        if PeftModel is None:
            raise RuntimeError("未安装peft，但你传了--lora_path。请先 pip install peft")
        print(f"Loading LoRA adapter: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model.eval()

    print("\n✅ Ready. 输入问题开始对话。输入 /reset 清空历史，/exit 退出。\n")

    history = []
    while True:
        try:
            user_query = input("User> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break

        if not user_query:
            continue
        if user_query.lower() in ["/exit", "exit", "quit", "q"]:
            print("Bye.")
            break
        if user_query.lower() in ["/reset", "reset", "clear"]:
            history = []
            print("History cleared.\n")
            continue

        prompt = build_prompt(tokenizer, args.system, history, user_query)
        answer = generate_one(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
        )

        print(f"Assistant> {answer}\n")

        history.append((user_query, answer))
        if len(history) > args.history_turns:
            history = history[-args.history_turns:]


if __name__ == "__main__":
    main()
