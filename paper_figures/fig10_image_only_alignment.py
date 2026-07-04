"""
fig10_image_only_alignment.py
==============================
Demonstrates image-only evaluation: no text branch required.

  Image A → visual embeddings [S_a, D]
  Image B → visual embeddings [S_b, D]
  Similarity matrix [S_a, S_b] + hard-DTW path

This is the key evaluation figure showing that after training the model can
align two manuscript images without any transcript information.

Output:
  fig10_image_only_alignment_{idx_a}_{idx_b}.pdf
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _HERE)

from utils.model_loading import load_image_model
from utils.sample_data import make_fig10_pair, FIG10_SENTENCE_PAIRS, FIG_STRIDE, FIG_NUM_PATCHES, get_fig_pair_windows
from utils.similarity import compute_image_embeddings, compute_image_image_similarity
from utils.alignment import hard_dtw_path_restricted
from utils.plotting import setup_paper_style, save_figure, plot_similarity_with_path, attach_window_strip, arabic_label


def _collapse(emb):
    """Mean-pool sub-features: [FIG_NUM_PATCHES*k, D] → [FIG_NUM_PATCHES, D].

    This collapse is not D3TW character pooling. Character pooling requires a
    transcript-dependent assignment A [T,S].
    """
    S = emb.shape[0]
    if S != FIG_NUM_PATCHES and S % FIG_NUM_PATCHES == 0:
        subfeat = S // FIG_NUM_PATCHES
        emb = emb.reshape(FIG_NUM_PATCHES, subfeat, -1).mean(dim=1)
    return emb


def draw_image_only_alignment(checkpoint, pair_idx, output_dir, device):
    setup_paper_style()

    model = load_image_model(checkpoint, device, stride_override=FIG_STRIDE)

    img_a, text_a, img_b, text_b, pil_a, pil_b = make_fig10_pair(pair_idx)

    emb_a  = _collapse(compute_image_embeddings(model, img_a, device))
    emb_b  = _collapse(compute_image_embeddings(model, img_b, device))
    sim    = compute_image_image_similarity(emb_a, emb_b)       # [64, 64]
    sim_np = sim.detach().cpu().numpy()
    path   = hard_dtw_path_restricted(sim_np)
    S_a, S_b = sim_np.shape

    # ── Highlight the shared words in title ────────────────────────────────────
    text_a_disp = arabic_label(text_a)
    text_b_disp = arabic_label(text_b)

    fig = plt.figure(figsize=(30, 24))
    gs  = GridSpec(3, 1, figure=fig,
                   height_ratios=[1.0, 1.0, 4.0],
                   hspace=0.10)

    ax_a = fig.add_subplot(gs[0])
    ax_a.imshow(np.array(pil_a), aspect="auto")
    ax_a.set_title(f"Image A  —  \"{text_a_disp}\"", fontsize=10, pad=4)
    ax_a.axis("off")

    ax_b = fig.add_subplot(gs[1])
    ax_b.imshow(np.array(pil_b), aspect="auto")
    ax_b.set_title(f"Image B  —  \"{text_b_disp}\"", fontsize=10, pad=4)
    ax_b.axis("off")

    ax_sim = fig.add_subplot(gs[2])
    im = plot_similarity_with_path(
        ax_sim, sim_np, path,
        title=(
            f"Image-only alignment uses the visual branch without transcript input\n"
            f"D3TW-guided character pooling is a training-time supervision mechanism"
        ),
        xlabel=f"Image B windows  (S={S_b})",
        ylabel=f"Image A windows  (S={S_a})",
        cmap="viridis",
        path_color="red", path_lw=1.8,
    )
    fig.colorbar(im, ax=ax_sim, fraction=0.025, pad=0.02,
                 label="Cosine similarity")

    fig.suptitle(
        "Image-Only Alignment — Evaluation Phase\n"
        "(No text branch used; NW alignment on visual embeddings)",
        fontsize=12, fontweight="bold",
    )
    _, wins_b = get_fig_pair_windows(pair_idx)
    plt.tight_layout()
    fig.subplots_adjust(bottom=max(0.08, 0.9 / fig.get_figheight()))
    attach_window_strip(ax_sim, wins_b)
    save_figure(fig, output_dir, f"fig10_image_only_pair{pair_idx}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Image-only NW alignment between two shared-content line images (fig10)."
    )
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--pair_idx",    type=int, default=None,
                        help=f"Index into FIG10_SENTENCE_PAIRS "
                             f"(0..{len(FIG10_SENTENCE_PAIRS)-1})")
    parser.add_argument("--pair_indices", type=int, nargs="+", default=[0, 1, 2],
                        help="Multiple image-pair indices to render.")
    parser.add_argument("--output_dir",  default="paper_figures/outputs")
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.chdir(_PROJ_ROOT)
    indices = [args.pair_idx] if args.pair_idx is not None else args.pair_indices
    for idx in indices:
        draw_image_only_alignment(args.checkpoint, idx,
                                  args.output_dir, args.device)


if __name__ == "__main__":
    main()
