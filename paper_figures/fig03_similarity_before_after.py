"""
fig03_similarity_before_after.py
=================================
Side-by-side similarity heatmaps:
  (a) Random (untrained) model — noisy, unstructured
  (b) Trained model — staircase / monotonic alignment

Output: fig03_similarity_before_after_sample_{idx}.png / .pdf
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

from utils.model_loading import load_image_model, load_random_model, load_text_embedder
from utils.sample_data import make_sample, FIG_STRIDE, FIG_NUM_PATCHES, get_fig_windows, pad_text
from utils.similarity import compute_image_embeddings, compute_text_embeddings, compute_text_image_similarity
from utils.plotting import setup_paper_style, save_figure, plot_similarity_heatmap, attach_window_strip, arabic_label


def _sim_matrix(img_model, text_embedder, img_tensor, text, device):
    img_emb = compute_image_embeddings(img_model, img_tensor, device)
    txt_emb = compute_text_embeddings(text_embedder, text)
    sim     = compute_text_image_similarity(txt_emb, img_emb)
    S = sim.shape[1]
    if S != FIG_NUM_PATCHES and S % FIG_NUM_PATCHES == 0:
        subfeat = S // FIG_NUM_PATCHES
        sim = sim.reshape(sim.shape[0], FIG_NUM_PATCHES, subfeat).mean(dim=-1)
    return sim.detach().cpu().numpy()


def draw_before_after(checkpoint, sentence_idx, output_dir, device):
    setup_paper_style()

    img_tensor, text = make_sample(sentence_idx)
    text_padded      = pad_text(text)
    text_embedder    = load_text_embedder(device)
    random_model     = load_random_model(device,   stride_override=FIG_STRIDE)
    trained_model    = load_image_model(checkpoint, device, stride_override=FIG_STRIDE)

    sim_before = _sim_matrix(random_model,  text_embedder, img_tensor, text_padded, device)
    sim_after  = _sim_matrix(trained_model, text_embedder, img_tensor, text_padded, device)

    vmin = min(sim_before.min(), sim_after.min())
    vmax = max(sim_before.max(), sim_after.max())
    chars = list(text_padded)

    fig, axes = plt.subplots(1, 2, figsize=(38, max(14, len(chars) * 0.65 + 4)),
                             gridspec_kw={"wspace": 0.35})

    im_b = plot_similarity_heatmap(
        axes[0], sim_before,
        title="(a) Before Training\n(random model)",
        xlabel="Image windows", ylabel="Text characters",
        yticklabels=chars, vmin=vmin, vmax=vmax, cmap="viridis",
    )
    im_a = plot_similarity_heatmap(
        axes[1], sim_after,
        title="(b) After Training\n(loaded checkpoint)",
        xlabel="Image windows", ylabel="",
        yticklabels=chars, vmin=vmin, vmax=vmax, cmap="viridis",
    )

    cbar = fig.colorbar(im_a, ax=axes.ravel().tolist(),
                        fraction=0.015, pad=0.02)
    cbar.set_label("Cosine similarity", fontsize=9)

    transcript_snippet = arabic_label(text)
    fig.suptitle(
        f"Text-Image Similarity Matrix — Sentence {sentence_idx}\n"
        f'Transcript: "{transcript_snippet}"',
        fontsize=11, fontweight="bold",
    )
    wins = get_fig_windows(sentence_idx)
    plt.tight_layout()
    # Reserve bottom space for the window strip
    fig.subplots_adjust(bottom=max(0.08, 0.9 / fig.get_figheight()))
    attach_window_strip(axes[0], wins)
    attach_window_strip(axes[1], wins)
    save_figure(fig, output_dir, f"fig03_similarity_before_after_s{sentence_idx}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Similarity heatmap before vs after training (fig03)."
    )
    parser.add_argument("--checkpoint",    required=True,
                        help="Path to trained model weights (.pth)")
    parser.add_argument("--sentence_idx",  type=int, default=0,
                        help="Index into the built-in Arabic sentence pool")
    parser.add_argument("--output_dir",    default="paper_figures/outputs")
    parser.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.chdir(_PROJ_ROOT)
    draw_before_after(args.checkpoint, args.sentence_idx,
                      args.output_dir, args.device)


if __name__ == "__main__":
    main()
