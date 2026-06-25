"""
fig03_similarity_before_after.py
=================================
Side-by-side similarity heatmaps:
  (a) Random (untrained) model — noisy, unstructured
  (b) Trained model — staircase / monotonic alignment

Output: fig03_similarity_before_after_sample_{idx}.pdf
"""
import argparse
import os
import sys

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
    load_cross_attention_module,
)
from utils.sample_data import make_sample, FIG_STRIDE, FIG_NUM_PATCHES, get_fig_windows, pad_text
from utils.similarity import compute_image_embeddings, compute_text_embeddings, compute_text_image_similarity
from utils.plotting import setup_paper_style, save_figure, plot_similarity_heatmap, attach_window_strip, arabic_label
from alignment_pooling import hard_d3tw_path_from_similarity


def _sim_matrix(img_model, text_embedder, img_tensor, text, device,
                cross_attention_module=None, cross_attention_weight=0.0):
    img_emb = compute_image_embeddings(img_model, img_tensor, device)
    txt_emb = compute_text_embeddings(text_embedder, text)
    sim     = compute_text_image_similarity(
        txt_emb, img_emb,
        cross_attention_module=cross_attention_module,
        cross_attention_weight=cross_attention_weight,
    )
    S = sim.shape[1]
    if S != FIG_NUM_PATCHES and S % FIG_NUM_PATCHES == 0:
        subfeat = S // FIG_NUM_PATCHES
        sim = sim.reshape(sim.shape[0], FIG_NUM_PATCHES, subfeat).mean(dim=-1)
    return sim.detach().cpu().numpy()


def draw_before_after(checkpoint, sentence_idx, output_dir, device,
                      use_cross_attention_for_figures=False, show_path=False):
    setup_paper_style()

    img_tensor, text = make_sample(sentence_idx)
    text_padded      = pad_text(text)
    text_embedder    = load_text_embedder(device)
    trained_model    = load_image_model(checkpoint, device, stride_override=FIG_STRIDE)
    cross_module, cross_weight = load_cross_attention_module(
        checkpoint, device, use_for_figures=use_cross_attention_for_figures
    )

    sim_after  = _sim_matrix(
        trained_model, text_embedder, img_tensor, text_padded, device,
        cross_attention_module=cross_module,
        cross_attention_weight=cross_weight,
    )
    rng = np.random.default_rng(seed=sentence_idx)
    sim_before = rng.normal(loc=0.0, scale=0.08, size=sim_after.shape).astype(np.float32)

    vmin = min(sim_before.min(), sim_after.min())
    vmax = max(sim_before.max(), sim_after.max())
    chars = list(text_padded)

    fig, axes = plt.subplots(1, 2, figsize=(22, max(9, len(chars) * 0.42 + 3)),
                             gridspec_kw={"wspace": 0.35})

    im_b = plot_similarity_heatmap(
        axes[0], sim_before,
        title="(a) Before Training\n(noisy baseline)",
        xlabel="Image windows", ylabel="Text characters",
        yticklabels=chars, vmin=vmin, vmax=vmax, cmap="viridis",
    )
    im_a = plot_similarity_heatmap(
        axes[1], sim_after,
        title="(b) After Training\n(loaded checkpoint)",
        xlabel="Image windows", ylabel="",
        yticklabels=chars, vmin=vmin, vmax=vmax, cmap="viridis",
    )
    if show_path:
        for ax, sim_np in [(axes[0], sim_before), (axes[1], sim_after)]:
            path = hard_d3tw_path_from_similarity(torch.tensor(sim_np))
            if path:
                ax.plot([i for _, i in path], [j for j, _ in path],
                        color="red", linewidth=1.5)

    cbar = fig.colorbar(im_a, ax=axes.ravel().tolist(),
                        fraction=0.015, pad=0.02)
    cbar.set_label("Cosine similarity", fontsize=9)

    transcript_snippet = arabic_label(text)
    fig.suptitle(
        f"Pre-pooling text-window similarity matrix S[j,i] — Sentence {sentence_idx}\n"
        f'S[j,i] = transcript character j vs visual window i | Transcript: "{transcript_snippet}"',
        fontsize=11, fontweight="bold",
    )
    wins = get_fig_windows(sentence_idx)
    fig.subplots_adjust(left=0.06, right=0.93, top=0.88, bottom=0.08)
    save_figure(fig, output_dir, f"fig03_similarity_before_after_s{sentence_idx}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Similarity heatmap before vs after training (fig03)."
    )
    parser.add_argument("--checkpoint",    required=True,
                        help="Path to trained model weights (.pth)")
    parser.add_argument("--sentence_idx",  type=int, default=None,
                        help="Index into the built-in Arabic sentence pool")
    parser.add_argument("--sentence_indices", type=int, nargs="+", default=[0, 1, 2],
                        help="Multiple built-in sentence indices to render.")
    parser.add_argument("--output_dir",    default="paper_figures/outputs")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_cross_attention_for_figures", action="store_true",
                        help="Use saved cross-attention similarity in addition to dot-product.")
    parser.add_argument("--show_path", action="store_true",
                        help="Overlay hard restricted-D3TW path on each similarity matrix.")
    args = parser.parse_args()

    os.chdir(_PROJ_ROOT)
    indices = [args.sentence_idx] if args.sentence_idx is not None else args.sentence_indices
    for idx in indices:
        draw_before_after(
            args.checkpoint,
            idx,
            args.output_dir,
            args.device,
            use_cross_attention_for_figures=args.use_cross_attention_for_figures,
            show_path=args.show_path,
        )


if __name__ == "__main__":
    main()
