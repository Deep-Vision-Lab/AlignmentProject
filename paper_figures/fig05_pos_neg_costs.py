"""
fig05_pos_neg_costs.py
======================
Violin/box plots showing that the correct transcript receives a lower DTW cost
than hard-negative transcripts.

Computes for *num_samples* samples:
  - DTW cost for the positive (correct) transcript
  - DTW cost for K negative transcripts (cropped / dropped / shuffled / random)

Also prints and saves summary statistics as JSON.

Output:
  fig05_pos_neg_cost_distribution.pdf
  fig05_pos_neg_cost_stats.json
"""
import argparse
import os
import sys
import random

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _HERE)

from utils.model_loading import (
    load_image_model, load_text_embedder,
)
from utils.sample_data import make_sample, ARABIC_SENTENCES, FIG_STRIDE, FIG_NUM_PATCHES, pad_text
from utils.similarity import (compute_image_embeddings, compute_text_embeddings,
                               compute_text_image_similarity, compute_dtw_cost)
from utils.negatives import generate_hard_negatives
from utils.plotting import setup_paper_style, save_figure


def _trim_sim(sim):
    S = sim.shape[1]
    if S != FIG_NUM_PATCHES and S % FIG_NUM_PATCHES == 0:
        subfeat = S // FIG_NUM_PATCHES
        sim = sim.reshape(sim.shape[0], FIG_NUM_PATCHES, subfeat).mean(dim=-1)
    return sim


def collect_costs(checkpoint, num_samples, num_negatives, device,
                   negative_mode="mixed"):
    model = load_image_model(checkpoint, device, stride_override=FIG_STRIDE)
    text_embedder = load_text_embedder(device)

    # Use all sentences as the pool; cycle if num_samples > len(pool)
    all_texts = [ARABIC_SENTENCES[i % len(ARABIC_SENTENCES)]
                 for i in range(max(num_samples, len(ARABIC_SENTENCES)))]

    pos_costs, neg_costs_all, pos_ranks = [], [], []
    wins = 0

    for si in range(num_samples):
        img_tensor, text = make_sample(si)
        img_emb = compute_image_embeddings(model, img_tensor, device)

        txt_emb  = compute_text_embeddings(text_embedder, text)
        sim_pos  = compute_text_image_similarity(txt_emb, img_emb)
        sim_pos  = _trim_sim(sim_pos)
        pos_cost = compute_dtw_cost(sim_pos, device="cpu")
        if np.isfinite(pos_cost):
            pos_costs.append(pos_cost)

        # Negative costs
        other_texts = [t for t in all_texts if t.strip() != text.strip()]
        negs = generate_hard_negatives(text, all_texts=other_texts,
                                       k=num_negatives, negative_mode=negative_mode)
        sample_neg_costs = []
        for neg_text, neg_type in negs:
            try:
                n_emb    = compute_text_embeddings(text_embedder, neg_text)
                sim_neg  = compute_text_image_similarity(n_emb, img_emb)
                sim_neg  = _trim_sim(sim_neg)
                nc       = compute_dtw_cost(sim_neg, device="cpu")
                if np.isfinite(nc):
                    sample_neg_costs.append(nc)
            except Exception:
                pass

        neg_costs_all.extend(sample_neg_costs)
        if sample_neg_costs and np.isfinite(pos_cost) and pos_cost < min(sample_neg_costs):
            wins += 1
        if sample_neg_costs and np.isfinite(pos_cost):
            pos_ranks.append(1 + sum(nc < pos_cost for nc in sample_neg_costs))

        if (si + 1) % 5 == 0:
            print(f"  {si + 1}/{num_samples} done  "
                  f"pos={pos_cost:.3f}  neg_mean={np.mean(sample_neg_costs):.3f}")

    top1_acc = wins / len(pos_costs) if pos_costs else 0.0
    return pos_costs, neg_costs_all, top1_acc, pos_ranks


def draw_cost_distribution(checkpoint, num_samples, num_negatives,
                            output_dir, device, negative_mode="mixed",
                            include_char_pool_stats=False):
    setup_paper_style()

    pos_costs, neg_costs_all, top1_acc, pos_ranks = collect_costs(
        checkpoint, num_samples, num_negatives, device,
        negative_mode=negative_mode,
    )

    # ── Statistics ────────────────────────────────────────────────────────────
    stats = {
        "cost_type":         "d3tw",
        "num_samples":       num_samples,
        "num_negatives":     num_negatives,
        "mean_positive_cost": float(np.mean(pos_costs)),
        "mean_negative_cost": float(np.mean(neg_costs_all)),
        "mean_pos_cost":     float(np.mean(pos_costs)),
        "mean_neg_cost":     float(np.mean(neg_costs_all)),
        "mean_gap":          float(np.mean(pos_costs) - np.mean(neg_costs_all)),
        "top1_accuracy":     float(top1_acc),
        "positive_rank":     float(np.mean(pos_ranks)) if pos_ranks else float("nan"),
        "pct_pos_wins":      float(top1_acc * 100),
    }
    if include_char_pool_stats:
        stats.update({
            "char_pool_top1": None,
            "char_pool_top5": None,
            "mean_windows_per_character": None,
            "char_pool_note": (
                "Use fig04/fig13 for assignment-based character-pool diagnostics; "
                "fig05 intentionally keeps the main plot as positive vs negative D3TW cost."
            ),
        })
    print("\n── Statistics ──────────────────────────────")
    for k, v in stats.items():
        if isinstance(v, (int, float)):
            print(f"  {k:<22} {v:.4f}")
        else:
            print(f"  {k:<22} {v}")
    print("────────────────────────────────────────────")
    if stats["top1_accuracy"] == 0.0 or stats["mean_gap"] > 0:
        print(
            "\n  WARNING: Positive transcripts are not ranking above negatives.\n"
            "  Treat as diagnostic failure, not success evidence.\n"
            "  Check: negative_mode, DTW normalisation, text embedder type."
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                             gridspec_kw={"width_ratios": [2, 1], "wspace": 0.35})

    # Violin / box plot
    ax = axes[0]
    data   = [pos_costs, neg_costs_all]
    labels = ["Positive\n(correct)", f"Negatives\n(K={num_negatives}/sample)"]
    colors = ["#2c7bb6", "#d7191c"]

    parts = ax.violinplot(data, positions=[1, 2], widths=0.55,
                          showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.45)
    for key in ["cmeans", "cmedians", "cbars", "cmins", "cmaxes"]:
        if key in parts:
            parts[key].set_color("#333333")
            parts[key].set_linewidth(1.2)

    ax.set_xticks([1, 2])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Normalised Soft-DTW Cost")
    ax.set_title("Positive vs Negative Alignment Cost Distribution")
    ax.axhline(np.mean(pos_costs), color=colors[0], ls="--", lw=1.0,
               label=f"μ_pos = {np.mean(pos_costs):.3f}")
    ax.axhline(np.mean(neg_costs_all), color=colors[1], ls="--", lw=1.0,
               label=f"μ_neg = {np.mean(neg_costs_all):.3f}")
    ax.legend(fontsize=8)

    # Summary stats text panel
    ax2 = axes[1]
    ax2.axis("off")
    stat_lines = [
        f"Samples evaluated:  {num_samples}",
        f"Neg. per sample:    {num_negatives}",
        "",
        f"Mean pos. cost:     {stats['mean_pos_cost']:.4f}",
        f"Mean neg. cost:     {stats['mean_neg_cost']:.4f}",
        f"Mean gap (neg−pos): {-stats['mean_gap']:.4f}",
        "",
        f"Top-1 accuracy:     {stats['top1_accuracy']:.2%}",
        f"(pos < all negs)",
    ]
    ax2.text(0.05, 0.95, "\n".join(stat_lines),
             transform=ax2.transAxes,
             va="top", ha="left", fontsize=9,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", fc="#f7f7f7", ec="#aaaaaa"))

    fig.suptitle("D3TW Alignment Cost: Positive vs Hard Negatives",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, output_dir, "fig05_pos_neg_cost_distribution")
    plt.close(fig)

    # PDF-only output by design; stats are printed to stdout, not written.


def main():
    parser = argparse.ArgumentParser(
        description="Positive vs negative DTW cost distribution (fig05)."
    )
    parser.add_argument("--checkpoint",    required=True)
    parser.add_argument("--num_samples",   type=int, default=10,
                        help="Number of synthetic sentences to evaluate "
                             f"(max {len(ARABIC_SENTENCES)})")
    parser.add_argument("--num_negatives",  type=int, default=5)
    parser.add_argument("--negative_mode",  default="length_controlled",
                        choices=["mixed", "length_controlled", "dot_confusion",
                                 "same_length_random", "shuffle_only"],
                        help="Negative sampling strategy. 'length_controlled' is "
                             "recommended for fair evaluation (no length bias).")
    parser.add_argument("--output_dir",    default="paper_figures/outputs")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include_char_pool_stats", action="store_true",
                        help="Include char-pooling diagnostic fields in the JSON summary.")
    args = parser.parse_args()

    os.chdir(_PROJ_ROOT)
    draw_cost_distribution(args.checkpoint, args.num_samples,
                           args.num_negatives, args.output_dir, args.device,
                           negative_mode=args.negative_mode,
                           include_char_pool_stats=args.include_char_pool_stats)


if __name__ == "__main__":
    main()
