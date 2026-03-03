"""
Global Retrieval Evaluation
============================
Metrics:  Recall@1, Recall@5, Recall@10, MRR, ACG

Protocol:
  - Build a database of text embeddings from the test split.
  - For each query image, compute a normalised DTW cost against every text
    in the database and rank them from lowest cost to highest.
  - The 'correct' text is the one that was paired with the query image.

Usage (from project root):
    python Evaluation/eval_retrieval.py \\
        --weights model_epoch_80.pth \\
        --data-dir DataSet/Synthetic_Arabic \\
        --n-samples 500
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Evaluation._eval_utils import (
    load_image_model, load_text_model,
    get_image_embedding, get_text_embedding,
    compute_sim_matrix, soft_dtw_cost,
    load_test_pairs,
)
from Parameters import device


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def recall_at_k(ranks: list, k: int) -> float:
    """Fraction of queries where the correct item is in the top-k."""
    return float(sum(r <= k for r in ranks)) / len(ranks)


def mean_reciprocal_rank(ranks: list) -> float:
    return float(np.mean([1.0 / r for r in ranks]))


def averaged_contrastive_gap(pos_costs: list, neg_costs_per_query: list) -> float:
    """
    ACG = mean over queries of (hardest_neg_cost - pos_cost).
    A higher ACG means the embedding space is more robustly separated.
    """
    gaps = []
    for pos, negs in zip(pos_costs, neg_costs_per_query):
        hardest_neg = min(negs)   # lowest cost = most similar = hardest
        gaps.append(hardest_neg - pos)
    return float(np.mean(gaps))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(weights_path: str, data_dir: str,
             n_samples: int = 500):

    print(f"Loading models …")
    img_model  = load_image_model(weights_path, device)
    txt_model  = load_text_model(device)

    # Collect test pairs
    pairs = list(load_test_pairs(data_dir, split='test', n_samples=n_samples))
    n = len(pairs)
    print(f"Evaluating on {n} test pairs …\n")

    # Pre-compute all image and text embeddings
    img_embs  = []
    txt_embs  = []
    texts     = []

    for img1_path, text1, _img2, _t2 in tqdm(pairs, desc='Embedding'):
        img_embs.append(get_image_embedding(img_model, img1_path, device))
        txt_embs.append(get_text_embedding(txt_model, text1, device))
        texts.append(text1)

    # For every query image, compute DTW cost with every text in the database
    all_ranks      = []
    pos_costs      = []
    neg_costs_all  = []

    for q_idx in tqdm(range(n), desc='Retrieval'):
        img_emb  = img_embs[q_idx]
        costs    = []

        for db_idx in range(n):
            sim  = compute_sim_matrix(img_emb, txt_embs[db_idx])
            cost = soft_dtw_cost(sim)
            # hard_dtw_cost returns a score (higher = better).
            # For ranking we negate so we can sort ascending.
            costs.append(-cost)

        # Rank: lower transformed cost (= higher original score) is better
        sorted_indices = np.argsort(costs)        # ascending
        rank = int(np.where(sorted_indices == q_idx)[0][0]) + 1   # 1-based

        all_ranks.append(rank)
        pos_costs.append(costs[q_idx])
        neg_costs_all.append([costs[j] for j in range(n) if j != q_idx])

    # Compute metrics
    r1   = recall_at_k(all_ranks, 1)  * 100
    r5   = recall_at_k(all_ranks, 5)  * 100
    r10  = recall_at_k(all_ranks, 10) * 100
    mrr  = mean_reciprocal_rank(all_ranks)
    acg  = averaged_contrastive_gap(pos_costs, neg_costs_all)

    print("\n========================================")
    print("       Global Retrieval Results         ")
    print("========================================")
    print(f"  Eval samples : {n}")
    print(f"  Recall@1     : {r1:.2f}%")
    print(f"  Recall@5     : {r5:.2f}%")
    print(f"  Recall@10    : {r10:.2f}%")
    print(f"  MRR          : {mrr:.4f}")
    print(f"  ACG          : {acg:.4f}  (higher = better separated)")
    print("========================================\n")

    return dict(R1=r1, R5=r5, R10=r10, MRR=mrr, ACG=acg, ranks=all_ranks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Global Retrieval Evaluation')
    parser.add_argument('--weights',   type=str, default='model_epoch_80.pth',
                        help='Path to model weights (.pth)')
    parser.add_argument('--data-dir',  type=str, default='DataSet/Synthetic_Arabic',
                        help='Root of the dataset (contains images/ and texts/)')
    parser.add_argument('--n-samples', type=int, default=500,
                        help='Number of test pairs to evaluate (default 500)')
    args = parser.parse_args()

    evaluate(
        weights_path=args.weights,
        data_dir=args.data_dir,
        n_samples=args.n_samples,
    )


if __name__ == '__main__':
    main()
