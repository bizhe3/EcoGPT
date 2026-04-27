#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - SFT Data Clustering Analysis (Auto)

Fully automatic clustering analysis:
  1. Load SFT data (jsonl with conversations format)
  2. Embed user questions using BGE-large-zh-v1.5
  3. Auto-detect cluster count (HDBSCAN by default, or K-Means with auto-K via silhouette)
  4. Auto-label every cluster (LLM if available, else TF-IDF keywords)
  5. Generate balance diagnostics + UMAP visualization

Usage:
    # Fully automatic (HDBSCAN + Qwen3-14B labels by default)
    python cluster_analysis.py \
        --input data/sft/train.jsonl \
        --output_dir outputs/cluster_analysis

    # Use a different LLM
    python cluster_analysis.py \
        --input data/sft/train.jsonl \
        --output_dir outputs/cluster_analysis \
        --llm_model models/base/Qwen3-7B

    # Disable LLM, use TF-IDF keywords only (faster, no GPU needed)
    python cluster_analysis.py \
        --input data/sft/train.jsonl \
        --output_dir outputs/cluster_analysis \
        --llm_model none
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger


# ============================================================
# Data loading
# ============================================================

def load_sft_data(input_path: str, max_samples: int = None):
    """Load SFT data from jsonl. Extract user questions."""
    texts = []
    raw_items = []
    with open(input_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            try:
                item = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            convs = item.get("conversations", [])
            human = next(
                (m["value"] for m in convs if m.get("from") == "human"), ""
            )
            if human:
                texts.append(human)
                raw_items.append(item)
    logger.info(f"Loaded {len(texts)} samples from {input_path}")
    return texts, raw_items


# ============================================================
# Embedding
# ============================================================

def compute_embeddings(texts, model_name, cache_path=None, batch_size=64, device="cuda"):
    """Compute embeddings using BGE-large-zh-v1.5. Defaults to CUDA; falls back to CPU only if GPU unavailable."""
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    import torch
    from sentence_transformers import SentenceTransformer

    if device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available - falling back to CPU.")
        logger.warning("For 40K samples on CPU, expect ~30-60 minutes.")
        device = "cpu"
        batch_size = min(batch_size, 16)

    logger.info(f"Loading embedding model: {model_name} on {device}")
    model = SentenceTransformer(model_name, device=device)

    logger.info(f"Encoding {len(texts)} texts (batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
        np.save(cache_path, embeddings)
        logger.info(f"Cached embeddings to {cache_path}")

    return embeddings


# ============================================================
# Auto clustering
# ============================================================

def auto_cluster_hdbscan(embeddings, min_cluster_size=None, total=None):
    """
    Run HDBSCAN with auto-tuned min_cluster_size.

    Heuristic: min_cluster_size = max(50, sqrt(N) / 2)
    HDBSCAN automatically determines the number of clusters.
    """
    import time
    import hdbscan

    n = total or len(embeddings)
    if min_cluster_size is None:
        min_cluster_size = max(50, int(np.sqrt(n) / 2))
    logger.info(f"HDBSCAN: min_cluster_size={min_cluster_size} (auto-tuned for N={n})")
    logger.info(f"HDBSCAN: starting clustering on {n} points × {embeddings.shape[1]} dims...")
    logger.info(f"HDBSCAN: this is CPU-intensive, expected duration: 3-8 minutes for 30K-50K samples")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
        core_dist_n_jobs=-1,  # use all CPU cores
    )

    t0 = time.time()
    labels = clusterer.fit_predict(embeddings)
    elapsed = time.time() - t0

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = sum(1 for x in labels if x == -1)
    logger.info(f"HDBSCAN finished in {elapsed:.1f}s: {n_clusters} clusters, {n_noise} noise points ({n_noise/n*100:.1f}%)")
    return labels, None


def auto_cluster_kmeans(embeddings, k_min=5, k_max=50, sample_size=5000):
    """
    Auto-detect K via silhouette score over a wide range.

    Searches k_min to k_max with step=2, picks the K with highest silhouette.
    """
    import time
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from tqdm import tqdm

    if len(embeddings) > sample_size:
        idx = np.random.RandomState(42).choice(len(embeddings), sample_size, replace=False)
        sample = embeddings[idx]
    else:
        sample = embeddings

    scores = {}
    candidates = list(range(k_min, k_max + 1, 2))
    logger.info(f"Searching optimal K in {k_min}-{k_max} (step=2, {len(candidates)} values, sample={len(sample)})...")

    for k in tqdm(candidates, desc="K-Means search"):
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(sample)
        score = silhouette_score(sample, labels, metric="cosine")
        scores[k] = score

    for k in candidates:
        logger.info(f"  k={k}: silhouette={scores[k]:.4f}")
    best_k = max(scores, key=scores.get)
    logger.info(f"Best k = {best_k} (silhouette={scores[best_k]:.4f})")

    logger.info(f"Running final K-Means with k={best_k} on full dataset...")
    t0 = time.time()
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    logger.info(f"K-Means finished in {time.time() - t0:.1f}s")
    return labels, km.cluster_centers_


# ============================================================
# Cluster analysis
# ============================================================

def get_cluster_representatives(embeddings, centers, labels, texts, top_k=5):
    """For each cluster, find the K samples closest to the center."""
    representatives = {}
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:
            continue
        cluster_idx = np.where(labels == cluster_id)[0]
        if centers is not None:
            center = centers[cluster_id]
        else:
            center = embeddings[cluster_idx].mean(axis=0)
        distances = np.linalg.norm(embeddings[cluster_idx] - center, axis=1)
        top_idx = cluster_idx[distances.argsort()[:top_k]]
        representatives[int(cluster_id)] = [texts[i] for i in top_idx]
    return representatives


def gini_coefficient(sizes):
    """Calculate Gini coefficient for distribution inequality."""
    sizes = np.array(sorted(sizes))
    n = len(sizes)
    if n == 0 or sizes.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * (index * sizes).sum() - (n + 1) * sizes.sum()) / (n * sizes.sum())


def diagnose_balance(labels, total):
    """Generate balance diagnostics."""
    counter = Counter(int(x) for x in labels if x != -1)
    sizes = sorted(counter.values(), reverse=True)

    if not sizes:
        return {"error": "No valid clusters"}

    gini = gini_coefficient(sizes)
    top3_pct = sum(sizes[:3]) / total * 100
    max_min_ratio = sizes[0] / sizes[-1] if sizes[-1] > 0 else float("inf")
    noise_count = sum(1 for x in labels if x == -1)

    return {
        "total_samples": total,
        "n_clusters": len(counter),
        "noise_samples": noise_count,
        "noise_pct": noise_count / total * 100,
        "max_cluster_size": sizes[0],
        "min_cluster_size": sizes[-1],
        "max_cluster_pct": sizes[0] / total * 100,
        "max_min_ratio": float(max_min_ratio),
        "gini_coefficient": float(gini),
        "top3_pct": top3_pct,
        "balance_verdict": (
            "均衡" if gini < 0.3
            else "中等" if gini < 0.5
            else "失衡"
        ),
    }


# ============================================================
# Auto labeling - Method 1: TF-IDF keywords (no LLM needed)
# ============================================================

# Stopwords for Chinese
ZH_STOPWORDS = set("""
的 了 在 是 我 有 和 就 不 人 都 一 上 也 很 到 说 要 去 你 会 着 没有 看 好 自己 这
那 它 他 她 们 个 之 与 及 以 或 但 而 等 啊 哦 嗯 呢 吧 吗 呀 哈 嘛 哟 唉 啦 罢 已经
什么 怎么 为什么 如何 哪里 哪个 哪些 多少 几 谁 何 那么 这么 这样 那样 这种 那种
请 是否 可以 能否 应该 需要 必须 最 比较 一些 一下 一点 几个 一般 通常 经常
""".split())


def tokenize_zh(text):
    """Simple Chinese + English tokenizer using jieba if available, else char-level."""
    try:
        import jieba
        return [w for w in jieba.cut(text) if len(w) > 1 and w not in ZH_STOPWORDS]
    except ImportError:
        # Fallback: extract Chinese 2-grams + English words
        tokens = []
        # English words
        tokens.extend(re.findall(r"[A-Za-z]+", text))
        # Chinese 2-grams
        chinese = re.findall(r"[一-鿿]+", text)
        for chunk in chinese:
            for i in range(len(chunk) - 1):
                bigram = chunk[i:i+2]
                if bigram not in ZH_STOPWORDS:
                    tokens.append(bigram)
        return tokens


def label_clusters_tfidf(representatives, top_n=5):
    """
    Label clusters using TF-IDF: extract top-N keywords per cluster.

    Treats each cluster's concatenated representatives as a "document",
    then identifies words distinctive to that cluster.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_ids = sorted(representatives.keys())
    documents = []
    for cid in cluster_ids:
        # Concatenate representatives of this cluster
        doc = " ".join([" ".join(tokenize_zh(s)) for s in representatives[cid]])
        documents.append(doc)

    if not documents or all(not d.strip() for d in documents):
        return {cid: f"Cluster {cid}" for cid in cluster_ids}

    vectorizer = TfidfVectorizer(
        max_features=2000,
        token_pattern=r"\S+",  # already tokenized
        min_df=1,
    )
    try:
        tfidf = vectorizer.fit_transform(documents)
    except ValueError:
        return {cid: f"Cluster {cid}" for cid in cluster_ids}

    feature_names = vectorizer.get_feature_names_out()
    labels = {}
    for i, cid in enumerate(cluster_ids):
        scores = tfidf[i].toarray().flatten()
        top_indices = scores.argsort()[-top_n:][::-1]
        keywords = [feature_names[idx] for idx in top_indices if scores[idx] > 0]
        labels[cid] = " / ".join(keywords[:top_n]) if keywords else f"Cluster {cid}"
    return labels


# ============================================================
# Auto labeling - Method 2: LLM (best quality)
# ============================================================

def label_clusters_with_llm(representatives, llm_model, tensor_parallel_size=1):
    """Use Qwen3-14B via vLLM to label each cluster's theme."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    logger.info(f"Loading LLM for cluster labeling: {llm_model}")
    tokenizer = AutoTokenizer.from_pretrained(llm_model, trust_remote_code=True)
    llm = LLM(
        model=llm_model,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.85,
    )

    prompts = []
    cluster_ids = []
    for cid, samples in sorted(representatives.items()):
        text = "\n".join([f"{i+1}. {s[:200]}" for i, s in enumerate(samples)])
        user_prompt = (
            f"以下是同一类问题的5个代表性样本，请用一个简短的主题标签（10字以内）总结它们的共同主题：\n\n"
            f"{text}\n\n"
            f"主题标签（只输出标签，不要解释）："
        )
        messages = [{"role": "user", "content": user_prompt}]
        try:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            chat_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(chat_text)
        cluster_ids.append(cid)

    logger.info(f"Labeling {len(prompts)} clusters with LLM...")
    sampling_params = SamplingParams(max_tokens=32, temperature=0.1)
    outputs = llm.generate(prompts, sampling_params)

    labels = {}
    for cid, output in zip(cluster_ids, outputs):
        text = output.outputs[0].text.strip()
        text = text.split("\n")[0].strip()[:30]
        labels[cid] = text if text else f"Cluster {cid}"
    return labels


# ============================================================
# Visualization
# ============================================================

def visualize(embeddings, labels, output_path, cluster_names):
    """Generate UMAP 2D visualization. Always uses cluster_names."""
    import time
    import matplotlib.pyplot as plt
    import umap

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    logger.info(f"Running UMAP on {len(embeddings)} points (CPU-intensive, expected 1-3 min)...")
    t0 = time.time()
    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42,
        verbose=True,  # show UMAP progress
    )
    emb_2d = reducer.fit_transform(embeddings)
    logger.info(f"UMAP finished in {time.time() - t0:.1f}s")

    plt.figure(figsize=(14, 10))
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", max(len(unique_labels), 2))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        if label == -1:
            color = "lightgray"
            name = "noise"
        else:
            color = cmap(i)
            label_name = cluster_names.get(label, f"#{label}")
            name = f"#{label} {label_name}"
        plt.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            s=2, alpha=0.5, c=[color], label=name
        )

    plt.title("SFT Data Cluster Distribution (UMAP)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, markerscale=3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved visualization to {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SFT data clustering analysis (auto)")
    parser.add_argument("--input", required=True, help="Input jsonl file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--embed_model",
        default="BAAI/bge-large-zh-v1.5",
        help="Sentence embedding model",
    )
    parser.add_argument(
        "--method",
        choices=["hdbscan", "kmeans"],
        default="hdbscan",
        help="Clustering algorithm (hdbscan: truly automatic; kmeans: auto-K via silhouette)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples (for debugging)",
    )
    parser.add_argument(
        "--llm_model",
        default="models/base/Qwen3-14B",
        help="LLM path for cluster labeling. Default: Qwen3-14B. Set to 'none' to use TF-IDF instead.",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=2,
        help="vLLM tensor parallel size for LLM labeling (default: 2 for dual GPU)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("[1/6] Loading SFT data...")
    logger.info("=" * 60)
    texts, _ = load_sft_data(args.input, max_samples=args.max_samples)

    logger.info("=" * 60)
    logger.info("[2/6] Computing embeddings...")
    logger.info("=" * 60)
    cache_path = output_dir / "embeddings.npy"
    embeddings = compute_embeddings(
        texts, args.embed_model, cache_path=str(cache_path)
    )

    logger.info("=" * 60)
    logger.info(f"[3/6] Clustering with {args.method.upper()}...")
    logger.info("=" * 60)
    if args.method == "hdbscan":
        labels, centers = auto_cluster_hdbscan(embeddings, total=len(texts))
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters < 3:
            logger.warning(f"HDBSCAN only found {n_clusters} clusters, falling back to K-Means")
            labels, centers = auto_cluster_kmeans(embeddings)
    else:
        labels, centers = auto_cluster_kmeans(embeddings)

    logger.info("=" * 60)
    logger.info("[4/6] Selecting cluster representatives...")
    logger.info("=" * 60)
    representatives = get_cluster_representatives(
        embeddings, centers, labels, texts, top_k=5
    )

    logger.info("=" * 60)
    logger.info("[5/6] Labeling clusters...")
    logger.info("=" * 60)
    use_llm = args.llm_model and args.llm_model.lower() != "none"
    if use_llm:
        logger.info(f"Labeling clusters with LLM: {args.llm_model}")
        try:
            cluster_names = label_clusters_with_llm(
                representatives, args.llm_model, args.tensor_parallel_size
            )
        except Exception as e:
            logger.warning(f"LLM labeling failed: {e}, falling back to TF-IDF")
            cluster_names = label_clusters_tfidf(representatives)
    else:
        logger.info("Labeling clusters with TF-IDF keywords (LLM explicitly disabled)...")
        cluster_names = label_clusters_tfidf(representatives)

    logger.info("=" * 60)
    logger.info("[6/6] Diagnosing balance + generating reports + visualization...")
    logger.info("=" * 60)
    # ---- Step 6: Diagnose balance ----
    diagnosis = diagnose_balance(labels, len(texts))
    logger.info("=" * 60)
    logger.info("Balance Diagnosis:")
    for k, v in diagnosis.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)

    # ---- Step 7: Save reports ----
    cluster_stats = []
    counter = Counter(int(x) for x in labels)
    for cid in sorted(counter.keys()):
        if cid == -1:
            continue
        stat = {
            "cluster_id": cid,
            "size": counter[cid],
            "percentage": counter[cid] / len(texts) * 100,
            "label": cluster_names.get(cid, f"Cluster {cid}"),
            "representatives": [s[:200] for s in representatives.get(cid, [])],
        }
        cluster_stats.append(stat)

    cluster_stats.sort(key=lambda x: -x["size"])

    report = {
        "input": args.input,
        "method": args.method,
        "n_clusters": len([c for c in counter if c != -1]),
        "labeling_method": "llm" if use_llm else "tfidf",
        "diagnosis": diagnosis,
        "clusters": cluster_stats,
    }

    report_path = output_dir / "cluster_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved report to {report_path}")

    # ---- Markdown summary ----
    md_path = output_dir / "cluster_summary.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SFT Data Cluster Analysis\n\n")
        f.write("## Diagnosis\n\n")
        f.write(f"- Method: {args.method.upper()}\n")
        f.write(f"- Labeling: {'LLM (' + args.llm_model + ')' if use_llm else 'TF-IDF keywords'}\n")
        f.write(f"- Total samples: {diagnosis['total_samples']}\n")
        f.write(f"- Clusters: {diagnosis['n_clusters']}\n")
        f.write(f"- Noise: {diagnosis['noise_samples']} ({diagnosis['noise_pct']:.1f}%)\n")
        f.write(f"- Largest cluster: {diagnosis['max_cluster_size']} ({diagnosis['max_cluster_pct']:.1f}%)\n")
        f.write(f"- Top-3 coverage: {diagnosis['top3_pct']:.1f}%\n")
        f.write(f"- Gini coefficient: {diagnosis['gini_coefficient']:.3f}\n")
        f.write(f"- **Balance verdict: {diagnosis['balance_verdict']}**\n\n")
        f.write("## Clusters (sorted by size)\n\n")
        for stat in cluster_stats:
            f.write(f"### Cluster #{stat['cluster_id']}: {stat['label']} ")
            f.write(f"({stat['size']} samples, {stat['percentage']:.1f}%)\n\n")
            for i, rep in enumerate(stat["representatives"][:3]):
                f.write(f"  {i+1}. {rep}\n")
            f.write("\n")
    logger.info(f"Saved markdown summary to {md_path}")

    # ---- Step 8: Visualize ----
    viz_path = output_dir / "cluster_umap.png"
    try:
        visualize(embeddings, labels, str(viz_path), cluster_names)
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    np.save(output_dir / "labels.npy", labels)
    logger.info(f"Done. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
