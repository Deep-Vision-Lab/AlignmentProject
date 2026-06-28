"""
fig07_cnn_vs_bilstm.py
=======================
Side-by-side similarity heatmaps:
  (a) CNN-only model  (use_bilstm=False)
  (b) CNN + BiLSTM model

If --checkpoint_cnn_only is omitted, the script initialises a fresh (random)
CNN-only model — this demonstrates the architectural difference qualitatively
rather than comparing two trained checkpoints.

Output:
  fig07_cnn_vs_bilstm_sample_{idx}.png / .pdf
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

from utils.model_loading import (load_image_model, load_random_model,
                                  load_text_embedder,
                                  load_cross_attention_module)
from utils.sample_data import make_sample, FIG_STRIDE, FIG_NUM_PATCHES, get_fig_windows, pad_text
from utils.similarity import (compute_image_embeddings, compute_text_embeddings,
                               compute_text_image_similarity)
from utils.plotting import setup_paper_style, save_figure, plot_similarity_heatmap, attach_window_strip, arabic_label


def _get_sim(img_model, text_embedder, img_tensor, text, device,
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


def draw_cnn_vs_bilstm(checkpoint_cnn, checkpoint_bilstm,
                        sentence_idx, output_dir, device,
                        use_cross_attention_for_figures=False):
    setup_paper_style()

    img_tensor, text = make_sample(sentence_idx)
    text_padded      = pad_text(text)
    text_embedder    = load_text_embedder(device)
    chars            = list(text_padded)

    random_cnn = not (checkpoint_cnn and os.path.isfile(checkpoint_cnn))
    if not random_cnn:
        model_cnn = load_image_model(checkpoint_cnn, device,
                                     use_bilstm_override=False,
                                     stride_override=FIG_STRIDE)
        cnn_cross_module, cnn_cross_weight = load_cross_attention_module(
            checkpoint_cnn, device, use_for_figures=use_cross_attention_for_figures
        )
        cnn_label = "(a) CNN-only\n(trained checkpoint)"
    else:
        model_cnn = load_random_model(device, use_bilstm_override=False,
                                      stride_override=FIG_STRIDE)
        cnn_cross_module, cnn_cross_weight = None, 0.0
        cnn_label = "(a) CNN-only\n(randomly initialised)"
        print("  [fig07] No CNN-only checkpoint provided; using random initialisation.")

    model_bilstm = load_image_model(checkpoint_bilstm, device,
                                    stride_override=FIG_STRIDE)
    bilstm_cross_module, bilstm_cross_weight = load_cross_attention_module(
        checkpoint_bilstm, device, use_for_figures=use_cross_attention_for_figures
    )
    bilstm_label = "(b) CNN + BiLSTM\n(trained checkpoint)"

    sim_cnn    = _get_sim(
        model_cnn, text_embedder, img_tensor, text_padded, device,
        cross_attention_module=cnn_cross_module,
        cross_attention_weight=cnn_cross_weight,
    )
    sim_bilstm = _get_sim(
        model_bilstm, text_embedder, img_tensor, text_padded, device,
        cross_attention_module=bilstm_cross_module,
        cross_attention_weight=bilstm_cross_weight,
    )

    vmin = min(sim_cnn.min(), sim_bilstm.min())
    vmax = max(sim_cnn.max(), sim_bilstm.max())

    T = len(chars)
    fig, axes = plt.subplots(1, 2, figsize=(38, max(14, T * 0.65 + 4)),
                             gridspec_kw={"wspace": 0.35})

    im1 = plot_similarity_heatmap(
        axes[0], sim_cnn, title=cnn_label,
        xlabel="Image windows", ylabel="Text characters",
        yticklabels=chars, vmin=vmin, vmax=vmax, cmap="viridis",
    )
    im2 = plot_similarity_heatmap(
        axes[1], sim_bilstm, title=bilstm_label,
        xlabel="Image windows", ylabel="",
        yticklabels=chars, vmin=vmin, vmax=vmax, cmap="viridis",
    )

    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(),
                        fraction=0.015, pad=0.02)
    cbar.set_label("Cosine similarity", fontsize=9)

    fig.suptitle(
        f"CNN-only vs CNN+BiLSTM  |  \"{arabic_label(text)}\"",
        fontsize=12, fontweight="bold",
    )
    if random_cnn:
        axes[0].text(
            0.5, 1.02,
            "⚠ CNN-only random init — not a fair trained ablation",
            transform=axes[0].transAxes,
            ha="center", va="bottom", fontsize=8,
            color="#c0392b", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fdecea", ec="#c0392b", alpha=0.85),
        )

    wins = get_fig_windows(sentence_idx)
    plt.tight_layout()
    fig.subplots_adjust(bottom=max(0.08, 0.9 / fig.get_figheight()))
    attach_window_strip(axes[0], wins)
    attach_window_strip(axes[1], wins)
    save_figure(fig, output_dir, f"fig07_cnn_vs_bilstm_s{sentence_idx}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="CNN-only vs CNN+BiLSTM heatmap comparison (fig07)."
    )
    parser.add_argument("--checkpoint_cnn_only", default="",
                        help="CNN-only checkpoint (omit to use random init)")
    parser.add_argument("--checkpoint_bilstm",   required=True,
                        help="CNN+BiLSTM trained checkpoint")
    parser.add_argument("--sentence_idx", type=int, default=0,
                        help="Index into the built-in Arabic sentence pool")
    parser.add_argument("--output_dir",   default="paper_figures/outputs")
    parser.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--use_cross_attention_for_figures", action="store_true",
                        help="Use saved cross-attention similarity in addition to dot-product.")
    args = parser.parse_args()

    os.chdir(_PROJ_ROOT)
    draw_cnn_vs_bilstm(
        args.checkpoint_cnn_only or None,
        args.checkpoint_bilstm,
        args.sentence_idx,
        args.output_dir, args.device,
        use_cross_attention_for_figures=args.use_cross_attention_for_figures,
    )


if __name__ == "__main__":
    main()
