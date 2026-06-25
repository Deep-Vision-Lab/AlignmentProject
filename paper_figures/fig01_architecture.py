"""
fig01_architecture.py
=====================
Generates a conceptual architecture diagram showing:
  - Training phase with D3TW-guided character pooling
  - Evaluation phase (image branch only)

Output: fig01_architecture.pdf
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_HERE      = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from utils.plotting import setup_paper_style, save_figure


# ── Drawing helpers ───────────────────────────────────────────────────────────

def _box(ax, x, y, w, h, label, sublabel="", fc="#4C72B0", tc="white",
         fontsize=9, radius=0.02, alpha=0.92):
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad={radius}",
        facecolor=fc, edgecolor="white", linewidth=0.8, alpha=alpha,
    )
    ax.add_patch(rect)
    ax.text(x, y + (0.01 if sublabel else 0), label,
            ha="center", va="center", fontsize=fontsize,
            color=tc, fontweight="bold")
    if sublabel:
        ax.text(x, y - 0.035, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=tc, style="italic")


def _arrow(ax, x0, y0, x1, y1, color="#444444", lw=1.2,
           arrowstyle="->", connectionstyle="arc3,rad=0.0"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(
            arrowstyle=arrowstyle,
            color=color,
            lw=lw,
            connectionstyle=connectionstyle,
        ),
    )


def _label(ax, x, y, text, fontsize=8.5, color="#222222", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            color=color, style="italic")


def _section_bg(ax, x, y, w, h, title, fc="#f0f4fa", fontsize=10):
    rect = mpatches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01",
        facecolor=fc, edgecolor="#aaaaaa", linewidth=1.0, zorder=0,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h + 0.02, title,
            ha="center", va="bottom", fontsize=fontsize,
            fontweight="bold", color="#333333")


# ── Main drawing ──────────────────────────────────────────────────────────────

def draw_architecture(output_dir):
    setup_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.patch.set_facecolor("white")

    # ------------------------------------------------------------------ TRAINING
    ax = axes[0]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _section_bg(ax, 0.01, 0.02, 0.97, 0.93, "Training Phase", fc="#eef3fb")

    C_IMG  = "#2c7bb6"   # blue  — image branch
    C_TXT  = "#d7191c"   # red   — text branch (frozen)
    C_LOSS = "#1a9641"   # green — loss / alignment
    C_OUT  = "#756bb1"   # purple — output

    # Image branch ─────────────────────────────────────────────────────────────
    bw, bh = 0.17, 0.10
    _box(ax, 0.12, 0.77, bw, bh, "Arabic Line", "Image",       fc=C_IMG)
    _box(ax, 0.12, 0.60, bw, bh, "Sliding", "Windows",         fc=C_IMG)
    _box(ax, 0.12, 0.43, bw, bh, "CNN", "+ BiLSTM/Transformer", fc=C_IMG)
    _box(ax, 0.12, 0.26, bw, bh, "Window Emb.", "V₁…Vₛ  [S,D]", fc=C_OUT)
    _box(ax, 0.12, 0.09, bw, bh, "Image Branch", "trainable",   fc=C_IMG)

    for y0, y1 in [(0.72, 0.65), (0.55, 0.48), (0.38, 0.31), (0.21, 0.14)]:
        _arrow(ax, 0.12, y0, 0.12, y1)

    # Text branch ──────────────────────────────────────────────────────────────
    _box(ax, 0.42, 0.77, bw, bh, "Transcript", "String",        fc=C_TXT)
    _box(ax, 0.42, 0.57, bw + 0.02, bh + 0.03,
         "Frozen Text", "Embedder",  fc=C_TXT)
    ax.text(0.42, 0.48, "(frozen)", ha="center", va="top",
            fontsize=7.5, color=C_TXT, style="italic")
    _box(ax, 0.42, 0.34, bw, bh, "Char Emb.", "E₁…Eₜ  [T,D]",   fc=C_OUT)
    _arrow(ax, 0.42, 0.72, 0.42, 0.605)
    _arrow(ax, 0.42, 0.535, 0.42, 0.39)

    # Similarity matrix ────────────────────────────────────────────────────────
    _box(ax, 0.68, 0.62, 0.19, bh, "Similarity", "S = E @ Vᵀ",  fc=C_LOSS)
    _arrow(ax, 0.205, 0.26, 0.59, 0.62,
           connectionstyle="arc3,rad=-0.25")
    _arrow(ax, 0.505, 0.34, 0.59, 0.62,
           connectionstyle="arc3,rad=0.15")

    # D3TW alignment ──────────────────────────────────────────────────────────
    _box(ax, 0.68, 0.47, 0.19, bh, "Contrastive", "D3TW loss",  fc=C_LOSS)
    _box(ax, 0.68, 0.32, 0.19, bh, "Assignment", "A [T,S]",     fc=C_LOSS)
    _box(ax, 0.68, 0.17, 0.22, bh, "Char Pool", "M = norm(A) @ V", fc=C_LOSS)
    _box(ax, 0.68, 0.06, 0.22, bh, "InfoNCE", "char-level loss", fc=C_LOSS)
    for y0, y1 in [(0.57, 0.525), (0.42, 0.375), (0.27, 0.225), (0.12, 0.105)]:
        _arrow(ax, 0.68, y0, 0.68, y1)
    ax.text(0.50, 0.905, "L_total = λ_d3tw L_D3TW + λ_char L_char_pool",
            ha="center", va="center", fontsize=8.5, color="#1a9641",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#e6f4ea", ec=C_LOSS, lw=0.8))

    # Negatives arrow ──────────────────────────────────────────────────────────
    ax.annotate(
        "", xy=(0.77, 0.26), xytext=(0.87, 0.40),
        arrowprops=dict(arrowstyle="->", color="#888888", lw=1.0,
                        connectionstyle="arc3,rad=0.3"),
    )
    ax.text(0.90, 0.38, "K neg.\ntranscripts",
            ha="left", va="center", fontsize=7.5, color="#888888")

    # Gradient flows only to image branch ─────────────────────────────────────
    ax.text(0.50, 0.04,
            "Text branch is supervision only; gradients update image branch",
            ha="center", va="bottom", fontsize=8, color=C_IMG,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#dceeff", ec=C_IMG, lw=0.8))

    # ---------------------------------------------------------------- EVALUATION
    ax = axes[1]
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _section_bg(ax, 0.01, 0.02, 0.97, 0.93, "Evaluation Phase (Image-Only)", fc="#eefbf0")

    C_IMG2 = "#1a7a4a"

    _box(ax, 0.50, 0.82, 0.30, bh, "Arabic Line", "Image",        fc=C_IMG2)
    _box(ax, 0.50, 0.65, 0.30, bh, "Sliding", "Windows",          fc=C_IMG2)
    _box(ax, 0.50, 0.48, 0.30, bh, "CNN", "+ BiLSTM/Transformer",  fc=C_IMG2)
    _box(ax, 0.50, 0.31, 0.30, bh, "Visual", "Window Embeddings",  fc=C_IMG2)
    _box(ax, 0.50, 0.14, 0.30, bh, "Visual", "Embeddings",         fc=C_OUT)

    for y0, y1 in [(0.77, 0.70), (0.60, 0.53), (0.43, 0.36), (0.26, 0.19)]:
        _arrow(ax, 0.50, y0, 0.50, y1, color="#1a7a4a")

    ax.text(0.50, 0.04,
            "Image-only alignment uses the visual branch; no transcript input",
            ha="center", va="bottom", fontsize=8, color=C_IMG2,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#d7f5e3", ec=C_IMG2, lw=0.8))

    # Cross-out where text branch would be ────────────────────────────────────
    ax.text(0.82, 0.57, "Text/D3TW/\nCharPool only\nfor training\nand diagnostics",
            ha="center", va="center", fontsize=9, color="#cc0000",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff0f0", ec="#cc0000", lw=0.8))

    fig.suptitle(
        "D3TW-Guided Character Pooling for Arabic Transcript-Image Alignment\n"
        "Training uses transcript supervision; final evaluation is image-only",
        fontsize=12, fontweight="bold", y=1.01,
    )

    plt.tight_layout()
    save_figure(fig, output_dir, "fig01_architecture")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate architecture diagram (fig01)."
    )
    parser.add_argument("--output_dir", default="paper_figures/outputs")
    args = parser.parse_args()
    draw_architecture(args.output_dir)


if __name__ == "__main__":
    main()
