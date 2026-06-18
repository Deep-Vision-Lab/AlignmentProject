"""
fig02_sliding_windows.py
========================
Shows how an Arabic line image is decomposed into overlapping sliding windows.

Figure layout:
  Row 0 : full image with highlighted window boxes (RTL labels)
  Row 1 : arrow row + "Right-to-Left sequence order" annotation
  Row 2+: grid of extracted window patches (first N windows)

Output: fig02_sliding_windows_sample_{idx}.png / .pdf
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE      = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _PROJ_ROOT)
sys.path.insert(0, _HERE)

from Parameters import lang
from utils.sample_data import make_sample, FIG_STRIDE, FIG_WINDOW_SIZE
from utils.plotting import setup_paper_style, save_figure


def _extract_windows(img_arr, ws, stride):
    """Return list of [H, ws, C] numpy patches."""
    W = img_arr.shape[1]
    windows = []
    j = 0
    while j + ws <= W:
        windows.append(img_arr[:, j:j + ws, :])
        j += stride
    return windows


def draw_sliding_windows(sentence_idx, output_dir, show_highlighted=10,
                         show_patches=16):
    setup_paper_style()

    stride    = FIG_STRIDE   # non-overlapping for display
    pil_img, text = make_sample(sentence_idx, transform=False)
    pil_img   = pil_img.resize((708, 128), Image.LANCZOS)
    img_arr   = np.array(pil_img)
    H, W, _   = img_arr.shape

    ws        = FIG_WINDOW_SIZE
    windows   = _extract_windows(img_arr, ws, stride)
    num_win   = len(windows)

    # For Arabic RTL: window index 0 is rightmost in the image.
    # We display windows in visual left-to-right order but label them RTL.
    is_arabic = lang.lower() == "arabic"

    # ── Layout ────────────────────────────────────────────────────────────────
    n_patch_cols = min(show_patches, num_win)
    patch_rows   = 1
    fig_h        = 1.4 + patch_rows * 1.0 + 0.5
    fig, axes    = plt.subplots(
        2 + patch_rows, 1,
        figsize=(14, fig_h),
        gridspec_kw={"height_ratios": [2.5] + [0.4] + [1.5] * patch_rows,
                     "hspace": 0.05},
    )

    # ── Row 0: full image with boxes ─────────────────────────────────────────
    ax_img = axes[0]
    ax_img.imshow(img_arr, aspect="auto")
    overlap_pct = int((1 - stride / ws) * 100)
    ax_img.set_title(
        f"Arabic Line Image  |  {num_win} overlapping windows "
        f"(window_size={ws}, stride={stride}, {overlap_pct}% overlap)",
        fontsize=11, pad=5,
    )
    ax_img.axis("off")

    # Choose evenly-spaced windows to highlight
    highlight_idxs = np.unique(
        np.linspace(0, num_win - 1, min(show_highlighted, num_win), dtype=int)
    )
    colors = plt.cm.Set2(np.linspace(0, 1, len(highlight_idxs)))

    for rank, vis_idx in enumerate(highlight_idxs):
        x = int(vis_idx) * stride
        clr = colors[rank]
        ax_img.add_patch(mpatches.Rectangle(
            (x, 0), ws, H,
            linewidth=0, facecolor=clr, alpha=0.20,
        ))
        ax_img.add_patch(mpatches.Rectangle(
            (x, 0), ws, H,
            linewidth=1.8, edgecolor=clr, facecolor="none",
        ))
        seq_label = (num_win - 1 - int(vis_idx)) if is_arabic else int(vis_idx)
        ax_img.text(
            x + ws / 2, H * 0.08, f"w{seq_label}",
            ha="center", va="top", fontsize=7, color="white",
            bbox=dict(boxstyle="square,pad=0.1", fc="black", alpha=0.55),
        )

    # ── Row 1: RTL arrow annotation ───────────────────────────────────────────
    ax_arrow = axes[1]
    ax_arrow.set_xlim(0, 1)
    ax_arrow.set_ylim(0, 1)
    ax_arrow.axis("off")

    ax_arrow.annotate(
        "Right-to-Left Sequence Order (Arabic)",
        xy=(0.02, 0.5), xytext=(0.98, 0.5),
        ha="right", va="center",
        fontsize=9.5, color="#c0392b", style="italic",
        arrowprops=dict(arrowstyle="<-", color="#c0392b", lw=1.5),
    )
    ax_arrow.text(0.50, 0.05,
                  "Window index 0 = rightmost patch (first Arabic character)",
                  ha="center", va="bottom", fontsize=8, color="#555555",
                  style="italic")

    # ── Row 2: patch grid ─────────────────────────────────────────────────────
    ax_patches = axes[2]
    ax_patches.set_xlim(0, n_patch_cols * (ws + 4))
    ax_patches.set_ylim(0, H + 20)
    ax_patches.axis("off")
    ax_patches.set_title("Extracted Window Patches (first shown left→right, RTL labelled)",
                         fontsize=9, pad=2)

    for rank in range(n_patch_cols):
        vis_idx   = rank
        patch_arr = windows[vis_idx]
        x_off     = rank * (ws + 4)
        seq_label = (num_win - 1 - vis_idx) if is_arabic else vis_idx
        ax_patches.imshow(
            patch_arr,
            extent=[x_off, x_off + ws, 0, H],
            aspect="auto",
        )
        ax_patches.text(
            x_off + ws / 2, H + 10,
            f"w{seq_label}", ha="center", va="bottom",
            fontsize=6.5, color="#222222",
        )

    fig.subplots_adjust(hspace=0.08, top=0.95, bottom=0.02, left=0.02, right=0.98)
    name = f"fig02_sliding_windows_sentence_{sentence_idx}"
    save_figure(fig, output_dir, name)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate sliding-window decomposition figure (fig02)."
    )
    parser.add_argument("--sentence_idx", type=int, default=0,
                        help="Index into the built-in Arabic sentence pool")
    parser.add_argument("--output_dir", default="paper_figures/outputs")
    args = parser.parse_args()

    os.chdir(_PROJ_ROOT)
    draw_sliding_windows(args.sentence_idx, args.output_dir)


if __name__ == "__main__":
    main()
