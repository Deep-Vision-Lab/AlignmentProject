"""
Visualization B – Direct Image-to-Text Mapping ("Show-Off" Graphic)
====================================================================
Layout (two-column figure):
  LEFT  – word mapping panels:
    • TOP    : raw manuscript line image with one coloured region per word
    • MIDDLE : coloured connector lines (word ↔ image patch)
    • BOTTOM : Arabic text, one proportional-width cell per word (RTL)
  RIGHT – alignment matrices:
    • TOP    : cosine-similarity heatmap  [T × S]
    • BOTTOM : same heatmap with DTW path overlaid

Usage (from project root):
    python Evaluation/viz_image_to_text_mapping.py \\
        --weights model_epoch_80.pth \\
        --image   DataSet/Synthetic_Arabic/images/img1_1.png \\
        --text    DataSet/Synthetic_Arabic/texts/text1_1.txt \\
        --output  Results/Evaluation/image_to_text_map.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Evaluation._eval_utils import (
    load_image_model, load_text_model,
    get_image_embedding, get_text_embedding,
    compute_sim_matrix, soft_dtw_path,
)
from Parameters import device


# ---------------------------------------------------------------------------
# Colour palette for connector lines (cycles if text > 20 chars)
# ---------------------------------------------------------------------------
PALETTE = plt.cm.tab20.colors  # 20 distinct colours


def _word_info(text: str):
    """Return list of (word_str, char_start, char_end_inclusive_with_space) from text."""
    words = []
    char_pos = 0
    raw_words = text.split()
    for i, w in enumerate(raw_words):
        while char_pos < len(text) and text[char_pos] == ' ':
            char_pos += 1
        start = char_pos
        char_pos += len(w)
        # Advance to include following spaces until next word (or end of text)
        end = char_pos - 1
        while char_pos < len(text) and text[char_pos] == ' ':
            end = char_pos
            char_pos += 1
        words.append((w, start, end))
    return words


def _word_patch_range(char_start, char_end, path, S, T):
    """Return (s_min, s_max, s_center) patch range for a word via DTW path."""
    # Collect all s-indices in the path that correspond to t-indices in [char_start, char_end]
    s_vals = [s for (t, s) in path if char_start <= t <= char_end]
    if not s_vals:
        # Fallback to proportional if path segment is missing
        s_vals = [
            round(char_start * (S - 1) / max(T - 1, 1)),
            round(char_end   * (S - 1) / max(T - 1, 1)),
        ]
    return min(s_vals), max(s_vals), int(np.median(s_vals))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_image_to_text(weights_path: str, image_path: str, text: str,
                       output_path: str):

    # ---- load models + embeddings ----
    img_model = load_image_model(weights_path, device)
    txt_model = load_text_model(device)

    img_emb  = get_image_embedding(img_model, image_path, device)    # [1, S, D]
    txt_emb  = get_text_embedding(txt_model, text, device)           # [1, T, D]
    sim      = compute_sim_matrix(img_emb, txt_emb)                  # [T, S]
    T, S     = sim.shape
    sim_np   = sim.detach().cpu().numpy() if hasattr(sim, 'detach') else np.array(sim)

    path, R_sdtw = soft_dtw_path(sim)   # D3TW via SoftDTW; R_sdtw = [T,S] cost

    # ---- word-level grouping ----
    words_info   = _word_info(text)
    N_words      = len(words_info)
    word_patches = [
        _word_patch_range(cs, ce, path, S, T)
        for (_, cs, ce) in words_info
    ]  # [(s_min, s_max, s_center), ...]

    # ---- original image ----
    orig = Image.open(image_path).convert('RGB')
    W_img, H_img = orig.size

    # ---- figure: left = word mapping (55 %), right = heatmaps (42 %) ----
    fig_w = max(16, W_img / 60)
    fig   = plt.figure(figsize=(fig_w, 7))
    fig.suptitle('Image ↔ Text Word Alignment Mapping',
                 fontsize=13, fontweight='bold', y=1.01)

    L  = 0.04   # left margin for mapping column
    LW = 0.52   # width of mapping column
    G  = 0.05   # gap between columns
    RX = L + LW + G   # right column x start
    RW = 1.0 - RX - 0.04  # right column width

    # mapping column rows (bottom-up)
    txt_b = 0.06;  txt_h = 0.14
    con_b = txt_b + txt_h + 0.01;  con_h = 0.20
    img_b = con_b + con_h + 0.01;  img_h = 0.50

    ax_img = fig.add_axes([L, img_b, LW, img_h])
    ax_con = fig.add_axes([L, con_b, LW, con_h])
    ax_txt = fig.add_axes([L, txt_b, LW, txt_h])

    # right column: two equal heatmap panels
    hm_h = 0.42
    ax_sim = fig.add_axes([RX, 0.52, RW, hm_h])
    ax_dtw = fig.add_axes([RX, 0.06, RW, hm_h])

    # ── image strip ─────────────────────────────────────────────────────
    ax_img.imshow(orig, aspect='auto')
    for wi, (s_min, s_max, _) in enumerate(word_patches):
        col = PALETTE[wi % len(PALETTE)]
        # RTL mirror: patch p -> x = (S-1-p)/S * W
        px0 = (S - 1 - s_max) / S * W_img
        px1 = (S - s_min)     / S * W_img
        ax_img.add_patch(mpatches.Rectangle(
            (px0, 2), max(px1 - px0, 2), H_img - 4,
            linewidth=1.5, edgecolor=col, facecolor=col, alpha=0.30
        ))
    ax_img.set_title('Manuscript Line', fontsize=10, fontweight='bold', pad=3)
    ax_img.axis('off')

    # ── connector panel ──────────────────────────────────────────────────
    ax_con.set_xlim(0, 1); ax_con.set_ylim(0, 1)
    ax_con.axis('off')

    # ── text bar ─────────────────────────────────────────────────────────
    ax_txt.set_xlim(0, 1); ax_txt.set_ylim(0, 1)
    ax_txt.axis('off')

    total_chars = sum(len(w) for w, _, _ in words_info) or 1
    word_x0, word_xc = [], []
    cum = 0.0
    for wi in range(N_words - 1, -1, -1):
        frac = len(words_info[wi][0]) / total_chars
        word_x0.insert(0, cum)
        word_xc.insert(0, cum + frac / 2)
        cum += frac

    for wi, (word, _, _) in enumerate(words_info):
        frac = len(word) / total_chars
        col  = PALETTE[wi % len(PALETTE)]
        ax_txt.add_patch(mpatches.FancyBboxPatch(
            (word_x0[wi], 0.0), frac - 0.002, 0.95,
            boxstyle='round,pad=0.01',
            linewidth=0.8, edgecolor='grey',
            facecolor=col, alpha=0.35,
            transform=ax_txt.transAxes
        ))
        ax_txt.text(word_xc[wi], 0.5, word,
                    ha='center', va='center',
                    fontsize=max(6, min(12, int(120 / N_words))),
                    transform=ax_txt.transAxes)
    ax_txt.set_title('Arabic Words (RTL)', fontsize=9, pad=2)

    # ── connectors ────────────────────────────────────────────────────────
    for wi in range(N_words):
        col         = PALETTE[wi % len(PALETTE)]
        _, _, s_ctr = word_patches[wi]
        x_txt_c   = word_xc[wi]
        x_patch_c = 1.0 - (s_ctr + 0.5) / S
        ax_con.add_line(Line2D(
            [x_txt_c, x_patch_c], [0.0, 1.0],
            transform=ax_con.transAxes,
            color=col, linewidth=1.8, alpha=0.75
        ))

    # ── similarity heatmap ────────────────────────────────────────────────
    im1 = ax_sim.imshow(sim_np, aspect='auto', origin='upper',
                        cmap='viridis', interpolation='nearest')
    ax_sim.set_title('Cosine Similarity  [T × S]', fontsize=10, fontweight='bold', pad=3)
    ax_sim.set_xlabel('Image patches (S)', fontsize=8)
    ax_sim.set_ylabel('Text chars (T)', fontsize=8)
    ax_sim.tick_params(labelsize=7)
    fig.colorbar(im1, ax=ax_sim, fraction=0.046, pad=0.04)

    # ── SoftDTW accumulated-cost heatmap + path ───────────────────────────
    im2 = ax_dtw.imshow(R_sdtw, aspect='auto', origin='upper',
                        cmap='magma_r', interpolation='nearest', alpha=0.85)
    if path:
        pts = np.array(path)          # [[t, s], ...]
        ax_dtw.plot(pts[:, 1], pts[:, 0], color='white',
                    linewidth=1.5, alpha=0.9)
        ax_dtw.plot(pts[:, 1], pts[:, 0], color='cyan',
                    linewidth=0.8, alpha=0.7)
    ax_dtw.set_title('SoftDTW Accumulated Cost + Path', fontsize=10, fontweight='bold', pad=3)
    ax_dtw.set_xlabel('Image patches (S)', fontsize=8)
    ax_dtw.set_ylabel('Text chars (T)', fontsize=8)
    ax_dtw.tick_params(labelsize=7)
    fig.colorbar(im2, ax=ax_dtw, fraction=0.046, pad=0.04)

    # ── save ─────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Image-to-Text Mapping Visualization')
    parser.add_argument('--weights',   type=str, default='model_epoch_80.pth')
    parser.add_argument('--image',     type=str, required=True)
    parser.add_argument('--text',      type=str, default=None)
    parser.add_argument('--text-file', type=str, default=None)
    parser.add_argument('--output',    type=str,
                        default='Results/Evaluation/image_to_text_map.png')
    args = parser.parse_args()

    if args.text is None and args.text_file is None:
        guess = args.image.replace('/images/', '/texts/') \
                          .replace('img1_', 'text1_') \
                          .replace('img2_', 'text2_') \
                          .replace('.png', '.txt')
        if os.path.exists(guess):
            args.text_file = guess
        else:
            parser.error('Provide --text or --text-file')

    if args.text is None:
        with open(args.text_file, encoding='utf-8') as f:
            args.text = f.read().strip()

    plot_image_to_text(
        weights_path=args.weights,
        image_path  =args.image,
        text        =args.text,
        output_path =args.output,
    )


if __name__ == '__main__':
    main()
