#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EcoGPT - SFT Data Clustering Analysis

Explore the structure and balance of SFT training data through clustering.

Pipeline:
  1. Load SFT data (jsonl with conversations format)
  2. Embed user questions using BGE-large-zh-v1.5
  3. Reduce dimensionality with UMAP
  4. Cluster with K-Means (or HDBSCAN)
  5. Label clusters using Qwen3-14B (optional, requires vLLM)
  6. Generate balance diagnostics + visualization

Usage:
    # Basic usage
    python cluster_analysis.py \
        --input data/sft/train.jsonl \
        --output_dir outputs/cluster_analysis

    # With LLM labeling
    python cluster_analysis.py \
        --input data/sft/train.jsonl \
        --output_dir outputs/cluster_analysis \
        --label_with_llm \
        --llm_model models/base/Qwen3-14B
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm


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

def compute_embeddings(texts, model_name, cache_path=None, batch_size=64):
    """Compute embeddings using BGE-large-zh-v1.5."""
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached embeddings from {cache_path}")
        return np.load(cache_path)

    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name, device="cuda")

    logger.info(f"Encoding {len(texts)} texts...")
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
# Clustering
# ============================================================

def find_optimal_k(embeddings, k_range=(10, 15, 20, 25, 30, 40), sample_size=5000):
    """Use silhouette score to find optimal K."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    if len(embeddings) > sample_size:
        idx = np.random.RandomState(42).choice(len(embeddings), sample_size, replace=False)
        sample = embeddings[idx]
    else:
        sample = embeddings

    scores = {}
    logger.info("Searching for optimal K...")
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(sample)
        score = silhouette_score(sample, labels, metric="cosine")
        scores[k] = score
        logger.info(f"  k={k}: silhouette={score:.4f}")

    best_k = max(scores, key=scores.get)
    logger.info(f"Best k = {best_k} (silhouette={scores[best_k]:.4f})")
    return best_k, scores


def cluster_kmeans(embeddings, n_clusters):
    """Run K-Means clustering."""
    from sklearn.cluster import KMeans

    logger.info(f"Running K-Means with n_clusters={n_clusters}...")
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    return labels, km.cluster_centers_


def cluster_hdbscan(embeddings, min_cluster_size=200):
    """Run HDBSCAN clustering (auto-determine cluster count)."""
    import hdbscan

    logger.info(f"Running HDBSCAN with min_cluster_size={min_cluster_size}...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings)
    return labels, None


# ============================================================
# Cluster analysis
# ============================================================

def get_cluster_representatives(embeddings, centers, labels, texts, top_k=5):
    """For each cluster, find the K samples closest to the center."""
    representatives = {}
    for cluster_id in sorted(set(labels)):
        if cluster_id == -1:  # HDBSCAN noise
            continue
        cluster_idx = np.where(labels == cluster_id)[0]
        if centers is not None:
            center = centers[cluster_id]
            distances = np.linalg.norm(embeddings[cluster_idx] - center, axis=1)
            top_idx = cluster_idx[distances.argsort()[:top_k]]
        else:
            # For HDBSCAN: use cluster centroid
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

    diagnosis = {
        "total_samples": total,
        "n_clusters": len(counter),
        "noise_samples": noise_count,
        "noise_pct": noise_count / total * 100,
        "max_cluster_size": sizes[0],
        "min_cluster_size": sizes[-1],
        "max_cluster_pct": sizes[0] / total * 100,
        "max_min_ratio": max_min_ratio,
        "gini_coefficient": gini,
        "top3_pct": top3_pct,
        "balance_verdict": (
            "均衡" if gini < 0.3
            else "中等" if gini < 0.5
            else "失衡"
        ),
    }
    return diagnosis


# ============================================================
# LLM labeling (optional)
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
        labels[cid] = text

    return labels


# ============================================================
# Visualization
# ============================================================

def visualize(embeddings, labels, output_path, cluster_names=None):
    """Generate UMAP 2D visualization."""
    import matplotlib.pyplot as plt
    import umap

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    logger.info("Running UMAP for visualization...")
    reducer = umap.UMAP(
        n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42
    )
    emb_2d = reducer.fit_transform(embeddings)

    plt.figure(figsize=(14, 10))
    unique_labels = sorted(set(labels))
    cmap = plt.cm.get_cmap("tab20", len(unique_labels))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        color = "lightgray" if label == -1 else cmap(i)
        name = (
            "noise" if label == -1
            else f"#{label} {cluster_names[label]}" if cluster_names and label in cluster_names
            else f"#{label}"
        )
        plt.scatter(
            emb_2d[mask, 0], emb_2d[mask, 1],
            s=2, alpha=0.5, c=[color], label=name
        )

    plt.title("SFT Data Cluster Distribution (UMAP)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, markerscale=3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved visualization to {output_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="SFT data clustering analysis")
    parser.add_argument("--input", required=True, help="Input jsonl file")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument(
        "--embed_model",
        default="BAAI/bge-large-zh-v1.5",
        help="Sentence embedding model",
    )
    parser.add_argument(
        "--method",
        choices=["kmeans", "hdbscan"],
        default="kmeans",
        help="Clustering algorithm",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=None,
        help="Number of clusters for K-Means (auto if not set)",
    )
    parser.add_argument(
        "--min_cluster_size",
        type=int,
        default=200,
        help="Minimum cluster size for HDBSCAN",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples (for debugging)",
    )
    parser.add_argument(
        "--label_with_llm",
        action="store_true",
        help="Use LLM to label cluster themes",
    )
    parser.add_argument(
        "--llm_model",
        default=None,
        help="LLM path for cluster labeling",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size for LLM labeling",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Load data ----
    texts, _ = load_sft_data(args.input, max_samples=args.max_samples)

    # ---- Step 2: Compute embeddings ----
    cache_path = output_dir / "embeddings.npy"
    embeddings = compute_embeddings(
        texts, args.embed_model, cache_path=str(cache_path)
    )

    # ---- Step 3: Cluster ----
    if args.method == "kmeans":
        if args.n_clusters is None:
            best_k, _ = find_optimal_k(embeddings)
            args.n_clusters = best_k
        labels, centers = cluster_kmeans(embeddings, args.n_clusters)
    else:
        labels, centers = cluster_hdbscan(embeddings, args.min_cluster_size)

    # ---- Step 4: Get representatives ----
    representatives = get_cluster_representatives(
        embeddings, centers, labels, texts, top_k=5
    )

    # ---- Step 5: Optional LLM labeling ----
    cluster_names = None
    if args.label_with_llm:
        if not args.llm_model:
            logger.error("--llm_model required when --label_with_llm")
            return
        cluster_names = label_clusters_with_llm(
            representatives, args.llm_model, args.tensor_parallel_size
        )

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
            "name": cluster_names[cid] if cluster_names else f"Cluster {cid}",
            "representatives": [s[:200] for s in representatives.get(cid, [])],
        }
        cluster_stats.append(stat)

    cluster_stats.sort(key=lambda x: -x["size"])

    report = {
        "input": args.input,
        "method": args.method,
        "n_clusters": args.n_clusters if args.method == "kmeans" else len(counter),
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
        f.write(f"# SFT Data Cluster Analysis\n\n")
        f.write(f"## Diagnosis\n\n")
        f.write(f"- Total samples: {diagnosis['total_samples']}\n")
        f.write(f"- Clusters: {diagnosis['n_clusters']}\n")
        f.write(f"- Largest cluster: {diagnosis['max_cluster_size']} ({diagnosis['max_cluster_pct']:.1f}%)\n")
        f.write(f"- Top-3 coverage: {diagnosis['top3_pct']:.1f}%\n")
        f.write(f"- Gini coefficient: {diagnosis['gini_coefficient']:.3f}\n")
        f.write(f"- **Balance verdict: {diagnosis['balance_verdict']}**\n\n")
        f.write(f"## Clusters (sorted by size)\n\n")
        for stat in cluster_stats:
            f.write(f"### Cluster #{stat['cluster_id']}: {stat['name']} ")
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

    # ---- Save labels for downstream use ----
    np.save(output_dir / "labels.npy", labels)
    logger.info(f"Done. Output dir: {output_dir}")


if __name__ == "__main__":
    main()
