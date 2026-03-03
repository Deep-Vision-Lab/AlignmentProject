"""
Visualization C – Positive vs Negative DTW Cost Distribution Histograms
=========================================================================
Plots two overlapping histograms:
  - Blue  (positive pairs): normalised DTW cost for correct (image, text) pairs.
  - Red   (negative pairs): normalised DTW cost for randomly mismatched pairs.

If the contrastive training worked correctly, the two bell-curves should be
completely separated with near-zero overlap.

Usage (from project root):
    python Evaluation/viz_distance_histogram.py \\
        --weights model_epoch_80.pth \\
        --data-dir DataSet/Synthetic_Arabic \\
        --n-samples 500 \\
        --output Results/Evaluation/distance_histogram.png
"""

import argparse
import os
import sys
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
# Main
# ---------------------------------------------------------------------------

def build_histograms(weights_path: str, data_dir: str,
                     n_samples: int = 500,
                     output_path: str = 'Results/Evaluation/distance_histogram.png',
                     negs_per_sample: int = 3):

    print("Loading models …")
    img_model = load_image_model(weights_path, device)
    txt_model = load_text_model(device)

    pairs = list(load_test_pairs(data_dir, split='test', n_samples=n_samples))
    n     = len(pairs)
    print(f"Computing DTW costs for {n} pairs …\n")

    # Pre-compute embeddings
    img_embs, txt_embs = [], []
    for img1, text1, _img2, _t2 in tqdm(pairs, desc='Embedding'):
        img_embs.append(get_image_embedding(img_model, img1, device))
        txt_embs.append(get_text_embedding(txt_model, text1, device))

    pos_costs = []
    neg_costs = []

    for i in tqdm(range(n), desc='DTW costs'):
        # Positive cost
        sim_pos = compute_sim_matrix(img_embs[i], txt_embs[i])
        pos_costs.append(soft_dtw_cost(sim_pos))

        # Negative costs: pick `negs_per_sample` random non-matching texts
        neg_indices = random.sample([j for j in range(n) if j != i],
                                    min(negs_per_sample, n - 1))
        for j in neg_indices:
            sim_neg = compute_sim_matrix(img_embs[i], txt_embs[j])
            neg_costs.append(soft_dtw_cost(sim_neg))

    pos_costs = np.array(pos_costs)
    neg_costs = np.array(neg_costs)

    # ---- stats ----
    print(f"  Positive costs : mean={pos_costs.mean():.4f}  std={pos_costs.std():.4f}")
    print(f"  Negative costs : mean={neg_costs.mean():.4f}  std={neg_costs.std():.4f}")
    gap = neg_costs.mean() - pos_costs.mean()
    print(f"  Mean gap (neg-pos) : {gap:.4f}  (higher = better)")

    # ---- Overlap coefficient (lower = better separation) ----
    all_min = min(pos_costs.min(), neg_costs.min())
    all_max = max(pos_costs.max(), neg_costs.max())
    bins    = np.linspace(all_min, all_max, 80)
    pos_h, _ = np.histogram(pos_costs, bins=bins, density=True)
    neg_h, _ = np.histogram(neg_costs, bins=bins, density=True)
    bin_w    = bins[1] - bins[0]
    overlap  = np.sum(np.minimum(pos_h, neg_h)) * bin_w
    print(f"  Histogram overlap  : {overlap:.4f}  (lower = better)")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(pos_costs, bins=50, density=True, alpha=0.6, color='steelblue',
            edgecolor='white', linewidth=0.4, label='Positive pairs (correct match)')
    ax.hist(neg_costs, bins=50, density=True, alpha=0.6, color='tomato',
            edgecolor='white', linewidth=0.4, label='Negative pairs (mismatch)')

    ax.axvline(pos_costs.mean(), color='steelblue', linewidth=2.0,
               linestyle='--', label=f'Pos mean={pos_costs.mean():.3f}')
    ax.axvline(neg_costs.mean(), color='tomato', linewidth=2.0,
               linestyle='--', label=f'Neg mean={neg_costs.mean():.3f}')

    ax.set_xlabel('Normalised DTW Cost  (higher = better alignment)',
                  fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Positive vs Negative DTW Costs',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)

    # annotation: mean gap and overlap
    ax.text(0.98, 0.96,
            f"Mean gap = {gap:.3f}\nHistogram overlap = {overlap:.3f}",
            ha='right', va='top', transform=ax.transAxes,
            fontsize=10, bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nSaved: {output_path}")

    return dict(pos_mean=pos_costs.mean(), neg_mean=neg_costs.mean(),
                gap=gap, overlap=overlap)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='DTW Cost Distribution Histograms')
    parser.add_argument('--weights',        type=str, default='model_epoch_80.pth')
    parser.add_argument('--data-dir',       type=str, default='DataSet/Synthetic_Arabic')
    parser.add_argument('--n-samples',      type=int, default=500)
    parser.add_argument('--negs-per-sample',type=int, default=3,
                        help='Number of negative texts sampled per query (default 3)')
    parser.add_argument('--output',         type=str,
                        default='Results/Evaluation/distance_histogram.png')
    args = parser.parse_args()

    build_histograms(
        weights_path   =args.weights,
        data_dir       =args.data_dir,
        n_samples      =args.n_samples,
        output_path    =args.output,
        negs_per_sample=args.negs_per_sample,
    )


if __name__ == '__main__':
    main()
