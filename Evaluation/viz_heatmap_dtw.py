"""
Visualization A – Similarity Heatmap + DTW Path Overlay
=========================================================
Produces a figure containing:
  - Top row   : the raw manuscript / handwriting line image
  - Main panel: the N×M cosine-similarity heatmap (text chars × image patches)
  - Over the heatmap: the optimal hard-DTW path drawn as a bright red line.
  - Bottom bar: the Arabic text string (one cell per character)

Reviewers expect to see a crisp "dark valley" diagonal in the heatmap with
the red path following it.

Usage (from project root):
    python Evaluation/viz_heatmap_dtw.py \\
        --weights model_epoch_80.pth \\
        --image   DataSet/Synthetic_Arabic/images/img1_1.png \\
        --text    DataSet/Synthetic_Arabic/texts/text1_1.txt \\
        --output  Results/Evaluation/heatmap_dtw.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Evaluation._eval_utils import (
    load_image_model, load_text_model,
    get_image_embedding, get_text_embedding,
    compute_sim_matrix, soft_dtw_path,
)
from Parameters import device


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_heatmap_dtw(weights_path: str, image_path: str, text: str,
                     output_path: str, cmap: str = 'viridis'):

    # ---- embeddings ----
    img_model = load_image_model(weights_path, device)
    txt_model = load_text_model(device)

    img_emb = get_image_embedding(img_model, image_path, device)   # [1, S, D]
    txt_emb = get_text_embedding(txt_model, text, device)          # [1, T, D]
    sim     = compute_sim_matrix(img_emb, txt_emb)                 # [T, S]

    sim_np  = sim.detach().cpu().numpy()
    T, S    = sim_np.shape

    # ---- DTW path ----
    path, _ = soft_dtw_path(sim)   # list of (t, s)
    t_coords = [p[0] for p in path]
    s_coords = [p[1] for p in path]

    # ---- original image ----
    orig_img = np.array(Image.open(image_path).convert('RGB'))

    # ---- figure layout ----
    fig = plt.figure(figsize=(max(12, S * 0.25), 10))
    gs  = gridspec.GridSpec(3, 1, height_ratios=[1.5, 5, 0.8],
                            hspace=0.35)

    # --- Row 0: original image ---
    ax_img = fig.add_subplot(gs[0])
    ax_img.imshow(orig_img, aspect='auto')
    ax_img.set_title('Input Manuscript Line', fontsize=12, fontweight='bold')
    ax_img.axis('off')

    # --- Row 1: similarity heatmap + path ---
    ax_heat = fig.add_subplot(gs[1])
    im = ax_heat.imshow(sim_np, aspect='auto', cmap=cmap,
                        origin='upper', interpolation='nearest')

    # DTW path as red line
    ax_heat.plot(s_coords, t_coords, color='red', linewidth=2.0,
                 label='DTW path', alpha=0.85)
    ax_heat.legend(loc='upper left', fontsize=9, framealpha=0.6)

    ax_heat.set_xlabel('Image Patch Index  →', fontsize=11)
    ax_heat.set_ylabel('← Text Character Index', fontsize=11)
    ax_heat.set_title('Cosine Similarity Heatmap  (text × image patches)',
                      fontsize=12, fontweight='bold')

    # Y-ticks: one per character
    step = max(1, T // 20)
    ax_heat.set_yticks(range(0, T, step))
    ax_heat.set_yticklabels([text[i] if i < len(text) else '' for i in range(0, T, step)],
                             fontsize=9)

    plt.colorbar(im, ax=ax_heat, fraction=0.03, pad=0.02, label='Cosine Similarity')

    # --- Row 2: text characters bar ---
    ax_txt = fig.add_subplot(gs[2])
    ax_txt.set_xlim(-0.5, T - 0.5)
    ax_txt.set_ylim(0, 1)
    for i, ch in enumerate(text):
        ax_txt.text(i, 0.5, ch, ha='center', va='center',
                    fontsize=max(6, min(14, int(200 / T))),
                    fontfamily='DejaVu Sans')
    ax_txt.set_xticks([])
    ax_txt.set_yticks([])
    ax_txt.set_xlabel('Text Characters', fontsize=11)
    ax_txt.spines['top'].set_visible(False)
    ax_txt.spines['right'].set_visible(False)

    # ---- save ----
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Similarity Heatmap + DTW Path')
    parser.add_argument('--weights', type=str, default='model_epoch_80.pth')
    parser.add_argument('--image',   type=str, required=True,
                        help='Path to the manuscript image')
    parser.add_argument('--text',    type=str, default=None,
                        help='Arabic text string. If omitted, --text-file must be given.')
    parser.add_argument('--text-file', type=str, default=None,
                        help='Path to a .txt file containing the text')
    parser.add_argument('--output',  type=str,
                        default='Results/Evaluation/heatmap_dtw.png')
    parser.add_argument('--cmap',    type=str,   default='viridis',
                        help='Matplotlib colormap (e.g. viridis, plasma, YlOrRd)')
    args = parser.parse_args()

    if args.text is None and args.text_file is None:
        # try to infer text file from image path
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

    plot_heatmap_dtw(
        weights_path=args.weights,
        image_path  =args.image,
        text        =args.text,
        output_path =args.output,
        cmap        =args.cmap,
    )


if __name__ == '__main__':
    main()
