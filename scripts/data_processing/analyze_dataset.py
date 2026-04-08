#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
analyze_conversation_dataset.py

用于分析 conversation 格式数据集中的回答分布，重点排查：
1. 复读（ngram repetition / sentence repetition）
2. 空洞模板化
3. 长度分布
4. 关键词覆盖
5. 独特词比例

支持常见格式：
1) {"conversations":[{"from":"human","value":"..."},{"from":"gpt","value":"..."}]}
2) {"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}

默认逻辑：
- 提取最后一条 assistant/gpt 回答作为 answer
- 提取该回答之前的全部 user/human 内容拼接为 prompt
"""

import argparse
import json
import os
import random
import re
from collections import Counter
from typing import Dict, List, Any, Optional, Tuple

import numpy as np


EMPTY_PATTERNS = [
    "建议及时就医",
    "根据具体情况",
    "在医生指导下",
    "具体分析",
    "因人而异",
    "视情况而定",
    "需要结合临床",
    "进一步检查",
    "建议咨询医生",
    "建议到医院就诊",
]

STOPWORDS = {
    "什么", "哪些", "如何", "为什么", "怎么", "需要", "可以", "吗",
    "的", "了", "和", "是", "在", "有", "与", "及", "一个", "一种",
    "是否", "进行", "对于", "应当", "应该", "请问", "一下", "有关"
}


def simple_tokenize_zh(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9_]+', text.lower())


def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    sents = re.split(r'[。！？；;\n]+', text)
    return [s.strip() for s in sents if s.strip()]


def ngram_repeat_score(text: str, n: int = 3) -> float:
    tokens = simple_tokenize_zh(text)
    if len(tokens) < n:
        return 0.0

    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    counter = Counter(ngrams)
    total = len(ngrams)
    repeated = sum(cnt - 1 for cnt in counter.values() if cnt > 1)
    return repeated / total if total > 0 else 0.0


def sentence_repeat_score(text: str) -> float:
    sents = split_sentences(text)
    if len(sents) <= 1:
        return 0.0
    counter = Counter(sents)
    repeated = sum(cnt - 1 for cnt in counter.values() if cnt > 1)
    return repeated / len(sents)


def empty_pattern_score(text: str) -> float:
    if not text:
        return 0.0
    count = sum(text.count(p) for p in EMPTY_PATTERNS)
    return count / max(len(text), 1) * 100.0


def extract_keywords_from_prompt(prompt: str) -> List[str]:
    if not prompt:
        return []
    tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z0-9_]+', prompt.lower())
    kws = []
    for t in tokens:
        if t not in STOPWORDS and len(t) >= 2:
            kws.append(t)
    return list(dict.fromkeys(kws))


def keyword_coverage(prompt: str, answer: str) -> float:
    kws = extract_keywords_from_prompt(prompt)
    if not kws:
        return 0.0
    answer_lower = answer.lower() if answer else ""
    hit = sum(1 for kw in kws if kw in answer_lower)
    return hit / len(kws)


def answer_length(text: str) -> int:
    return len(simple_tokenize_zh(text))


def unique_token_ratio(text: str) -> float:
    tokens = simple_tokenize_zh(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def evaluate_answer(prompt: str, answer: str) -> Dict[str, Any]:
    rep2 = ngram_repeat_score(answer, n=2)
    rep3 = ngram_repeat_score(answer, n=3)
    sent_rep = sentence_repeat_score(answer)
    empty_score = empty_pattern_score(answer)
    kw_cov = keyword_coverage(prompt, answer)
    length = answer_length(answer)
    uniq_ratio = unique_token_ratio(answer)

    return {
        "rep2": round(rep2, 6),
        "rep3": round(rep3, 6),
        "sent_rep": round(sent_rep, 6),
        "empty_score": round(empty_score, 6),
        "kw_cov": round(kw_cov, 6),
        "length": length,
        "uniq_ratio": round(uniq_ratio, 6),
    }


def normalize_role(role: str) -> str:
    role = (role or "").strip().lower()
    mapping = {
        "human": "user",
        "user": "user",
        "assistant": "assistant",
        "gpt": "assistant",
        "bot": "assistant",
        "system": "system",
    }
    return mapping.get(role, role)


def extract_from_messages(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """
    从 messages / conversations 中提取：
    - prompt: 最后一条 assistant 回答之前的所有 user 内容拼接
    - answer: 最后一条 assistant 内容
    """
    if not messages:
        return None, None

    normalized = []
    for m in messages:
        role = normalize_role(str(m.get("role", m.get("from", ""))))
        content = m.get("content", m.get("value", ""))
        content = "" if content is None else str(content).strip()
        if not content:
            continue
        normalized.append({"role": role, "content": content})

    if not normalized:
        return None, None

    last_assistant_idx = None
    for i in range(len(normalized) - 1, -1, -1):
        if normalized[i]["role"] == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        return None, None

    answer = normalized[last_assistant_idx]["content"]

    prompt_parts = []
    for i in range(last_assistant_idx):
        if normalized[i]["role"] == "user":
            prompt_parts.append(normalized[i]["content"])

    prompt = "\n".join(prompt_parts).strip()
    return prompt, answer


def extract_prompt_answer(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """
    自动兼容多种 conversation 格式。
    """
    if "conversations" in item and isinstance(item["conversations"], list):
        return extract_from_messages(item["conversations"])

    if "messages" in item and isinstance(item["messages"], list):
        return extract_from_messages(item["messages"])

    # 兼容已经是扁平结构的情况
    prompt = item.get("prompt")
    answer = item.get("answer")
    if prompt is not None and answer is not None:
        return str(prompt), str(answer)

    response = item.get("response")
    if prompt is not None and response is not None:
        return str(prompt), str(response)

    return None, None


def show_stats(name: str, arr: List[float]) -> None:
    if not arr:
        print(f"\n{name}\n  (no data)")
        return
    arr_np = np.array(arr, dtype=float)
    print(f"\n{name}")
    print(f"  count: {len(arr_np)}")
    print(f"  mean : {arr_np.mean():.6f}")
    print(f"  p50  : {np.percentile(arr_np, 50):.6f}")
    print(f"  p75  : {np.percentile(arr_np, 75):.6f}")
    print(f"  p90  : {np.percentile(arr_np, 90):.6f}")
    print(f"  p95  : {np.percentile(arr_np, 95):.6f}")
    print(f"  max  : {arr_np.max():.6f}")


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
                data.append(item)
            except json.JSONDecodeError as e:
                print(f"[WARN] 第 {idx} 行 JSON 解析失败，已跳过：{e}")
    return data


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def analyze_dataset(
    input_path: str,
    output_dir: str,
    high_rep_threshold: float,
    high_empty_threshold: float,
    sample_size: int,
) -> None:
    data = load_jsonl(input_path)
    if not data:
        print("[ERROR] 没有读取到有效数据。")
        return

    ensure_dir(output_dir)

    rep2_list = []
    rep3_list = []
    sent_rep_list = []
    empty_list = []
    kw_cov_list = []
    length_list = []
    uniq_list = []

    detailed_rows = []
    high_rep_samples = []
    high_empty_samples = []
    skipped = 0

    for i, item in enumerate(data):
        prompt, answer = extract_prompt_answer(item)

        if not answer or not answer.strip():
            skipped += 1
            continue

        prompt = prompt or ""
        m = evaluate_answer(prompt, answer)

        rep2_list.append(m["rep2"])
        rep3_list.append(m["rep3"])
        sent_rep_list.append(m["sent_rep"])
        empty_list.append(m["empty_score"])
        kw_cov_list.append(m["kw_cov"])
        length_list.append(m["length"])
        uniq_list.append(m["uniq_ratio"])

        row = {
            "id": i,
            "prompt": prompt,
            "answer": answer,
            **m
        }
        detailed_rows.append(row)

        if m["rep3"] >= high_rep_threshold or m["sent_rep"] >= 0.15:
            high_rep_samples.append(row)

        if m["empty_score"] >= high_empty_threshold:
            high_empty_samples.append(row)

    print("=" * 80)
    print("Conversation 数据集分析结果")
    print(f"输入文件: {input_path}")
    print(f"有效样本数: {len(detailed_rows)}")
    print(f"跳过样本数: {skipped}")

    show_stats("rep2", rep2_list)
    show_stats("rep3", rep3_list)
    show_stats("sent_rep", sent_rep_list)
    show_stats("empty_score", empty_list)
    show_stats("kw_cov", kw_cov_list)
    show_stats("length", length_list)
    show_stats("uniq_ratio", uniq_list)

    detailed_path = os.path.join(output_dir, "detailed_metrics.jsonl")
    with open(detailed_path, "w", encoding="utf-8") as f:
        for row in detailed_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    high_rep_path = os.path.join(output_dir, "high_repetition_samples.jsonl")
    with open(high_rep_path, "w", encoding="utf-8") as f:
        for row in high_rep_samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    high_empty_path = os.path.join(output_dir, "high_empty_samples.jsonl")
    with open(high_empty_path, "w", encoding="utf-8") as f:
        for row in high_empty_samples:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n" + "=" * 80)
    print("已保存文件：")
    print(f"- 全量详细指标: {detailed_path}")
    print(f"- 高复读样本:   {high_rep_path}")
    print(f"- 高空洞样本:   {high_empty_path}")

    if high_rep_samples:
        print("\n" + "=" * 80)
        print(f"随机展示 {min(sample_size, len(high_rep_samples))} 条高复读样本：")
        samples = random.sample(high_rep_samples, min(sample_size, len(high_rep_samples)))
        for row in samples:
            print("-" * 80)
            print("prompt:")
            print(row["prompt"])
            print("\nanswer:")
            print(row["answer"])
            print("\nmetrics:", {
                "rep2": row["rep2"],
                "rep3": row["rep3"],
                "sent_rep": row["sent_rep"],
                "empty_score": row["empty_score"],
                "kw_cov": row["kw_cov"],
                "length": row["length"],
                "uniq_ratio": row["uniq_ratio"],
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 conversation 格式数据中的复读/空洞/长度/覆盖分布")
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL 文件路径")
    parser.add_argument("--output-dir", type=str, default="analysis_output", help="输出目录")
    parser.add_argument("--high-rep-threshold", type=float, default=0.10, help="高复读阈值，默认 0.10")
    parser.add_argument("--high-empty-threshold", type=float, default=0.08, help="高空洞阈值，默认 0.08")
    parser.add_argument("--sample-size", type=int, default=10, help="终端随机展示样本数")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        high_rep_threshold=args.high_rep_threshold,
        high_empty_threshold=args.high_empty_threshold,
        sample_size=args.sample_size,
    )


if __name__ == "__main__":
    main()