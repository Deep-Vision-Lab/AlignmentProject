"""
fig11_success_failure_cases.py
================================
Grid of qualitative alignment examples:
  Each row = one case (success / partial / failure)
  Columns: line image | similarity heatmap | D3TW path overlay | note

Output:
  fig11_success_failure_cases.png / .pdf
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

from utils.model_loading import load_image_model, load_text_embedder
from utils.sample_data import make_sample, FIG_STRIDE, FIG_NUM_PATCHES, get_fig_windows, pad_text
from utils.similarity import (compute_image_embeddings, compute_text_embeddings,
                               compute_text_image_similarity)
from utils.alignment import hard_dtw_path_restricted
from utils.plotting import setup_paper_style, save_figure, plot_similarity_with_path, attach_window_strip, arabic_label


# Row-label colours
_CASE_COLORS = {
    "success": "#1a9641",
    "partial": "#fdae61",
    "failure": "#d7191c",
}


def _get_sim(model, text_embedder, img_tensor, text, device):
    img_emb = compute_image_embeddings(model, img_tensor, device)
    txt_emb = compute_text_embeddings(text_embedder, text)
    sim     = compute_text_image_similarity(txt_emb, img_emb)
    S = sim.shape[1]
    if S != FIG_NUM_PATCHES and S % FIG_NUM_PATCHES == 0:
        subfeat = S // FIG_NUM_PATCHES
        sim = sim.reshape(sim.shape[0], FIG_NUM_PATCHES, subfeat).mean(dim=-1)
    return sim.detach().cpu().numpy()


def draw_success_failure(checkpoint, sentence_indices, case_labels,
                          notes, output_dir, device):
    setup_paper_style()

    model         = load_image_model(checkpoint, device, stride_override=FIG_STRIDE)
    text_embedder = load_text_embedder(device)

    n_cases = len(sentence_indices)
    n_cols  = 3   # image | heatmap | heatmap+path

    col_w   = [4.0, 10.0, 10.0]
    fig_w   = sum(col_w) + 1.5
    row_h   = 8.0
    fig_h   = n_cases * row_h + 2.0

    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(
        n_cases, n_cols, figure=fig,
        width_ratios=col_w,
        hspace=0.45, wspace=0.30,
    )

    _heatmap_axes = []   # [(ax_sim, ax_path, wins), ...] – populated in loop

    for row, (si, label, note) in enumerate(
            zip(sentence_indices, case_labels, notes)):
        pil_img, text  = make_sample(si, transform=False)
        img_tensor, _  = make_sample(si, transform=True)

        text_padded = pad_text(text)
        sim_np = _get_sim(model, text_embedder, img_tensor, text_padded, device)
        path   = hard_dtw_path_restricted(sim_np)
        chars  = list(text_padded)
        color  = _CASE_COLORS.get(label.lower(), "#555555")

        # Col 0: line image ────────────────────────────────────────────────────
        ax_img = fig.add_subplot(gs[row, 0])
        from PIL import Image as _PIL
        ax_img.imshow(np.array(pil_img.resize((708, 128))), aspect="auto")
        ax_img.axis("off")
        ax_img.set_title(
            f"[{label.upper()}]  Sentence {si}\n"
            f'"{arabic_label(text)}"',
            fontsize=7.5, color=color, pad=3,
        )
        # Coloured left border
        ax_img.axvline(x=0, color=color, linewidth=5, solid_capstyle="butt")

        # Col 1: similarity heatmap ────────────────────────────────────────────
        ax_sim = fig.add_subplot(gs[row, 1])
        T_s, S_s = sim_np.shape
        v_lo, v_hi = sim_np.min(), sim_np.max()
        span = max(v_hi - v_lo, 1e-8)
        im = ax_sim.imshow(sim_np, aspect="auto", cmap="viridis",
                           interpolation="nearest")
        y_labels = ["_" if ch == " " else arabic_label(ch) for ch in chars]
        show_vals = max(T_s, S_s) <= 64
        fsize = max(6, min(11, int(320 / max(T_s, S_s))))

        ax_sim.set_title("Similarity matrix", fontsize=9, pad=3)
        ax_sim.set_ylabel("Chars", fontsize=8)
        ax_sim.set_yticks(range(T_s))
        ax_sim.set_yticklabels(y_labels, fontsize=7)
        ax_sim.tick_params(labelsize=6)
        fig.colorbar(im, ax=ax_sim, fraction=0.04, pad=0.02)
        ax_sim.set_xticks(np.arange(-0.5, S_s, 1), minor=True)
        ax_sim.set_yticks(np.arange(-0.5, T_s, 1), minor=True)
        ax_sim.grid(which="minor", color="white", linewidth=0.4, alpha=0.6)
        ax_sim.tick_params(which="minor", bottom=False, left=False)
        ax_sim.set_ylim(T_s - 0.5 + 0.2, -0.5 - 0.2)
        if show_vals:
            for r in range(T_s):
                for c in range(S_s):
                    norm = (sim_np[r, c] - v_lo) / span
                    clr  = "white" if norm < 0.6 else "#111111"
                    ax_sim.text(c, r, f"{sim_np[r,c]:.2f}", ha="center", va="center",
                                fontsize=fsize, color=clr, fontweight="bold")

        # Col 2: heatmap + path ────────────────────────────────────────────────
        ax_path = fig.add_subplot(gs[row, 2])
        im2 = ax_path.imshow(sim_np, aspect="auto", cmap="viridis",
                              interpolation="nearest")
        if path:
            rows_p = [p[0] for p in path]
            cols_p = [p[1] for p in path]
            ax_path.plot(cols_p, rows_p, color="red", linewidth=1.5, alpha=0.85)
        ax_path.set_title(f"Path overlay\nNote: {note[:35]}", fontsize=9, pad=3)
        ax_path.set_yticks(range(T_s))
        ax_path.set_yticklabels(y_labels, fontsize=7)
        ax_path.tick_params(labelsize=6)
        fig.colorbar(im2, ax=ax_path, fraction=0.04, pad=0.02)
        ax_path.set_xticks(np.arange(-0.5, S_s, 1), minor=True)
        ax_path.set_yticks(np.arange(-0.5, T_s, 1), minor=True)
        ax_path.grid(which="minor", color="white", linewidth=0.4, alpha=0.6)
        ax_path.tick_params(which="minor", bottom=False, left=False)
        ax_path.set_ylim(T_s - 0.5 + 0.2, -0.5 - 0.2)
        if show_vals:
            for r in range(T_s):
                for c in range(S_s):
                    norm = (sim_np[r, c] - v_lo) / span
                    clr  = "white" if norm < 0.6 else "#111111"
                    ax_path.text(c, r, f"{sim_np[r,c]:.2f}", ha="center", va="center",
                                 fontsize=fsize, color=clr, fontweight="bold")

        # Store axes + windows for strip attachment after layout
        _heatmap_axes.append((ax_sim, ax_path, get_fig_windows(si)))

    fig.suptitle("Qualitative Success and Failure Cases",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    # Reserve bottom space for the lowest row's window strip
    fig.subplots_adjust(bottom=max(0.07, 0.9 / fig.get_figheight()))
    for ax_s, ax_p, wins in _heatmap_axes:
        attach_window_strip(ax_s, wins)
        attach_window_strip(ax_p, wins)
    save_figure(fig, output_dir, "fig11_success_failure_cases")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Success/partial/failure case grid (fig11)."
    )
    parser.add_argument("--checkpoint",         required=True)
    parser.add_argument("--sentence_indices",   type=int, nargs="+", default=[0, 1, 2],
                        help="Indices into the built-in Arabic sentence pool")
    parser.add_argument("--case_labels",        nargs="+",
                        default=["success", "partial", "failure"])
    parser.add_argument("--notes",              nargs="+",
                        default=["Good monotonic alignment",
                                 "Minor ambiguity near dots",
                                 "Failure due to degradation"])
    parser.add_argument("--output_dir",         default="paper_figures/outputs")
    parser.add_argument("--device",             default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    n = len(args.sentence_indices)
    labels = (args.case_labels + ["unknown"] * n)[:n]
    notes  = (args.notes       + [""]        * n)[:n]

    os.chdir(_PROJ_ROOT)
    draw_success_failure(args.checkpoint,
                         args.sentence_indices, labels, notes,
                         args.output_dir, args.device)


if __name__ == "__main__":
    main()
