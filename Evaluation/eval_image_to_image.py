"""
Image-to-Image Word Alignment
================================
Maps *words* from img1_{i}.png to their corresponding regions in img2_{i}.png
using the learned embeddings + hard DTW alignment path.

Layout of the output figure:
  - TOP    : img1 with one coloured box per word
  - MIDDLE : coloured connector lines linking each img1 word box to its
             matched region in img2
  - BOTTOM : img2 with the matched word region highlighted in the same colour

Word-to-patch mapping:
  - text1 / text2 are split on whitespace.
  - Each word is assigned a proportional patch range on its image.
  - For img2 the matched region is derived from the DTW path: for all img1
    patches that belong to a word, the corresponding img2 patch indices are
    collected and their extent becomes the word box on img2.

Usage (from project root):
    # Single pair (text files are inferred automatically)
    python Evaluation/eval_image_to_image.py \\
        --weights   model_epoch_80.pth \\
        --index     1 \\
        --data-dir  DataSet/Synthetic_Arabic \\
        --output    Results/Evaluation/img_to_img_1.png

    # Batch statistics over the test split
    python Evaluation/eval_image_to_image.py \\
        --weights   model_epoch_80.pth \\
        --data-dir  DataSet/Synthetic_Arabic \\
        --batch     --n-samples 200 \\
        --output-dir Results/Evaluation/ImageToImage
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
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Evaluation._eval_utils import (
    load_image_model,
    get_image_embedding,
    compute_sim_matrix,
    dtw_path,
    hard_dtw_cost,
    load_test_pairs,
)
from Parameters import device

PALETTE = [
    '#e6194b','#3cb44b','#4363d8','#f58231','#911eb4',
    '#42d4f4','#f032e6','#bfef45','#fabed4','#469990',
    '#dcbeff','#9A6324','#fffac8','#800000','#aaffc3',
    '#808000','#ffd8b1','#000075','#a9a9a9','#cccccc',
]


# ---------------------------------------------------------------------------
# Core alignment
# ---------------------------------------------------------------------------

def align_two_images(model, img1_path: str, img2_path: str):
    emb1 = get_image_embedding(model, img1_path, device)   # [1, S1, D]
    emb2 = get_image_embedding(model, img2_path, device)   # [1, S2, D]
    sim  = compute_sim_matrix(emb2, emb1)                  # [S1, S2]
    # S1 is vertical (text1/img1), S2 is horizontal (text2/img2)
    path, H = dtw_path(sim, gap_penalty=-10.0, match_score=10.0, mismatch_score=-27.0)
    return sim, path, H, emb1.shape[1], emb2.shape[1]


# ---------------------------------------------------------------------------
# Word-to-patch mapping helpers
# ---------------------------------------------------------------------------

def word_patch_ranges(text: str, n_patches: int):
    """
    Returns [(word_str, patch_start, patch_end), ...] (inclusive, 0-based).
    Patch ranges are proportional to character position in the full text string.
    """
    words = text.split()
    if not words or n_patches == 0:
        return []

    text_len = len(text)
    ranges   = []
    char_pos = 0

    for w in words:
        # advance past leading spaces
        while char_pos < text_len and text[char_pos] == ' ':
            char_pos += 1
        w_start = char_pos
        w_end   = char_pos + len(w) - 1
        char_pos += len(w)

        ps = int(round(w_start / max(text_len - 1, 1) * (n_patches - 1)))
        pe = int(round(w_end   / max(text_len - 1, 1) * (n_patches - 1)))
        ps = max(0, min(ps, n_patches - 1))
        pe = max(ps, min(pe, n_patches - 1))
        ranges.append((w, ps, pe))

    return ranges


def _patch_to_pixels(p_start: int, p_end: int, n_patches: int, img_width: int):
    """
    Convert embedding patch indices to image pixel coordinates.
    The model flips patches internally (RTL Arabic support), so patch index 0
    corresponds to the RIGHTMOST part of the image. We mirror here.
    """
    # Mirror: patch p  ->  pixel = (S-1-p) / S * W
    # For a range [p_start, p_end], mirrored pixel range is:
    px_start = (n_patches - 1 - p_end)   / n_patches * img_width
    px_end   = (n_patches - 1 - p_start + 1) / n_patches * img_width
    return px_start, px_end


def map_word_to_img2(p_start: int, p_end: int, path: list, S2: int):
    """
    Collect all img2 patches that the DTW path maps to for img1 patches
    in [p_start, p_end] and return (min_s2, max_s2).
    """
    s2_vals = [s2 for (s1, s2) in path if p_start <= s1 <= p_end]
    if not s2_vals:
        last_s1 = max(p[0] for p in path) if path else 1
        s2_vals = [
            int(round(p_start / max(last_s1, 1) * S2)),
            int(round(p_end   / max(last_s1, 1) * S2)),
        ]
    return max(0, min(s2_vals)), max(0, min(max(s2_vals), S2 - 1))


def word_sim_score(p_start: int, p_end: int, path: list,
                   sim: np.ndarray) -> float:
    """
    Average cosine similarity between img1 word patches [p_start, p_end]
    and their DTW-matched img2 patches.  Returns -1 if no path segment found.
    """
    vals = [sim[s1, s2] for (s1, s2) in path if p_start <= s1 <= p_end]
    if not vals:
        return -1.0
    return float(np.mean(vals))


# ---------------------------------------------------------------------------
# Single-pair visualisation  (word level, no heatmap)
# ---------------------------------------------------------------------------

def visualise_pair(model, img1_path: str, img2_path: str,
                   text1: str, text2: str,
                   output_path: str,
                   sim_threshold: float = 0.5):

    sim, path, H_mat, S1, S2 = align_two_images(model, img1_path, img2_path)
    sim_np = sim.detach().cpu().numpy() if hasattr(sim, 'detach') else np.array(sim)

    img1_arr = np.array(Image.open(img1_path).convert('RGB'))
    img2_arr = np.array(Image.open(img2_path).convert('RGB'))
    H1, W1   = img1_arr.shape[:2]
    H2, W2   = img2_arr.shape[:2]

    words1 = word_patch_ranges(text1, S1)
    n_words   = len(words1)
    img2_ranges  = [map_word_to_img2(ps, pe, path, S2)
                    for (_, ps, pe) in words1]
    word_scores  = [word_sim_score(ps, pe, path, sim_np)
                    for (_, ps, pe) in words1]

    # ── figure: img1 / connectors / img2 / heatmap ──────────────────────
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        f"Word-Level Image Alignment (NW) — "
        f"{os.path.basename(img1_path)} ↔ {os.path.basename(img2_path)}",
        fontsize=11, fontweight='bold', y=0.98
    )
    # gs = fig.add_gridspec(3, 1, height_ratios=[2.5, 0.8, 2.5], hspace=0.05)
    gs = fig.add_gridspec(4, 1, height_ratios=[2.0, 0.6, 2.0, 3.5], hspace=0.15)

    ax1   = fig.add_subplot(gs[0])
    ax_cn = fig.add_subplot(gs[1])
    ax2   = fig.add_subplot(gs[2])
    ax_hm = fig.add_subplot(gs[3])

    for ax in (ax1, ax2, ax_cn):
        ax.set_xticks([]); ax.set_yticks([])

    # ── img1 ────────────────────────────────────────────────────────────
    ax1.imshow(img1_arr, aspect='auto')
    ax1.set_ylabel('img1', fontsize=9, rotation=0, labelpad=28, va='center')

    for wi, (word, ps, pe) in enumerate(words1):
        col  = PALETTE[wi % len(PALETTE)]
        x0, x1_w = _patch_to_pixels(ps, pe, S1, W1)
        ax1.add_patch(mpatches.Rectangle(
            (x0, 2), max(x1_w - x0, 2), H1 - 4,
            linewidth=2, edgecolor=col, facecolor=col, alpha=0.28
        ))
        ax1.text((x0 + x1_w) / 2, H1 * 0.5, word,
                 ha='center', va='center', fontsize=7,
                 color='white', fontweight='bold',
                 bbox=dict(facecolor=col, alpha=0.65, pad=1, boxstyle='round'))

    # ── connectors ──────────────────────────────────────────────────────
    ax_cn.set_xlim(0, 1); ax_cn.set_ylim(0, 1)
    ax_cn.spines[:].set_visible(False)

    for wi in range(n_words):
        if word_scores[wi] < sim_threshold:
            continue                       # skip low-similarity pairs
        col       = PALETTE[wi % len(PALETTE)]
        _, ps, pe = words1[wi]
        ms, me    = img2_ranges[wi]

        # Mirror patch centres for RTL
        x_top = 1.0 - ((ps + pe) / 2 + 0.5) / S1
        x_bot = 1.0 - ((ms + me) / 2 + 0.5) / S2
        ax_cn.add_line(Line2D(
            [x_top, x_bot], [1.0, 0.0],
            transform=ax_cn.transAxes,
            color=col, linewidth=1.8, alpha=0.75
        ))

    # ── img2 ────────────────────────────────────────────────────────────
    ax2.imshow(img2_arr, aspect='auto')
    ax2.set_ylabel('img2', fontsize=9, rotation=0, labelpad=28, va='center')

    for wi in range(n_words):
        if word_scores[wi] < sim_threshold:
            continue                       # skip low-similarity pairs
        col      = PALETTE[wi % len(PALETTE)]
        ms, me   = img2_ranges[wi]

        x0, x1_w = _patch_to_pixels(ms, me, S2, W2)
        ax2.add_patch(mpatches.Rectangle(
            (x0, 2), max(x1_w - x0, 2), H2 - 4,
            linewidth=2, edgecolor=col, facecolor=col, alpha=0.28
        ))
        # ADD THE WORD TEXT BAG TO IMG2 TOO
        word = words1[wi][0]
        ax2.text((x0 + x1_w) / 2, H2 * 0.5, word,
                 ha='center', va='center', fontsize=7,
                 color='white', fontweight='bold',
                 bbox=dict(facecolor=col, alpha=0.65, pad=1, boxstyle='round'))

    # ── Heatmap ──────────────────────────────────────────────────────────
    im_hm = ax_hm.imshow(H_mat.T, aspect='auto', cmap='magma', origin='lower')
    ax_hm.set_title("NW Score Matrix & Optimal Path", fontsize=9, pad=5)
    ax_hm.set_xlabel("img1 patches", fontsize=8)
    ax_hm.set_ylabel("img2 patches", fontsize=8)
    plt.colorbar(im_hm, ax=ax_hm, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)

    # Plot the path on top of the heatmap
    px = [p[0] for p in path]
    py = [p[1] for p in path]
    ax_hm.plot(px, py, color='cyan', linewidth=1.5, alpha=0.8, label='Optimal Path')
    ax_hm.legend(fontsize=7, loc='upper left')

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
                exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Batch statistics
# ---------------------------------------------------------------------------

def diagonal_deviation(path, S1, S2):
    errs = [abs(s2 - s1 * (S2 - 1) / max(S1 - 1, 1)) for (s1, s2) in path]
    return float(np.mean(errs)) if errs else 0.0


def mean_matched_similarity(sim, path):
    s = sim.detach().cpu().numpy()
    vals = [s[s1, s2] for (s1, s2) in path
            if 0 <= s1 < s.shape[0] and 0 <= s2 < s.shape[1]]
    return float(np.mean(vals)) if vals else 0.0


def batch_evaluate(weights_path, data_dir, n_samples=200,
                   output_dir=None, save_figures=False, sim_threshold=0.5):
    print("Loading model …")
    model = load_image_model(weights_path, device)
    pairs = list(load_test_pairs(data_dir, split='test', n_samples=n_samples))
    print(f"Evaluating {len(pairs)} image pairs …\n")

    devs, sims, costs = [], [], []
    for img1_path, text1, img2_path, text2 in tqdm(pairs, desc='Pairs'):
        sim, path, H_mat, S1, S2 = align_two_images(model, img1_path, img2_path)
        devs.append(diagonal_deviation(path, S1, S2))
        sims.append(mean_matched_similarity(sim, path))
        costs.append(hard_dtw_cost(sim, gap_penalty=-10.0, match_score=10.0, mismatch_score=-27.0))

        if save_figures and output_dir:
            i_num = os.path.basename(img1_path).replace('img1_', '').replace('.png', '')
            visualise_pair(model, img1_path, img2_path, text1, text2,
                           os.path.join(output_dir, f'pair_{i_num}.png'),
                           sim_threshold=sim_threshold)

    print("\n=====================================================")
    print("     Image-to-Image Alignment Batch Results         ")
    print("=====================================================")
    print(f"  Pairs evaluated          : {len(pairs)}")
    print(f"  Mean diagonal deviation  : {np.mean(devs):.3f} patches  (lower = more linear)")
    print(f"  Mean matched cosine sim  : {np.mean(sims):.4f}          (higher = better)")
    print(f"  Mean normalised DTW cost : {np.mean(costs):.4f}")
    print("=====================================================\n")
    return dict(diag_dev=np.mean(devs), matched_sim=np.mean(sims), dtw_cost=np.mean(costs))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_text(path):
    with open(path, encoding='utf-8') as f:
        return f.read().strip()


def _infer_text_path(img_path, txt_dir):
    name = os.path.basename(img_path)
    txt  = name.replace('.png', '.txt') \
               .replace('img1_', 'text1_') \
               .replace('img2_', 'text2_')
    return os.path.join(txt_dir, txt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Image-to-Image Word-Level Alignment'
    )
    parser.add_argument('--weights',    type=str,  default='model_epoch_80.pth')
    parser.add_argument('--data-dir',   type=str,  default='DataSet/Synthetic_Arabic')
    parser.add_argument('--index',      type=int,  default=None)
    parser.add_argument('--output',     type=str,  default='Results/Evaluation/img_to_img.png')
    parser.add_argument('--batch',      action='store_true')
    parser.add_argument('--n-samples',  type=int,  default=200)
    parser.add_argument('--output-dir', type=str,  default='Results/Evaluation/ImageToImage')
    parser.add_argument('--save-figs',  action='store_true')
    parser.add_argument('--sim-threshold', type=float, default=0.5,
                        help='Minimum word similarity to draw connectors (default: 0.5)')
    args = parser.parse_args()

    img_dir = os.path.join(args.data_dir, 'images')
    txt_dir = os.path.join(args.data_dir, 'texts')

    if args.batch:
        batch_evaluate(args.weights, args.data_dir, args.n_samples,
                       args.output_dir, args.save_figs,
                       sim_threshold=args.sim_threshold)
    else:
        idx       = args.index or 1
        img1_path = os.path.join(img_dir, f'img1_{idx}.png')
        img2_path = os.path.join(img_dir, f'img2_{idx}.png')
        txt1_path = _infer_text_path(img1_path, txt_dir)
        txt2_path = _infer_text_path(img2_path, txt_dir)

        for p in (img1_path, img2_path, txt1_path, txt2_path):
            if not os.path.exists(p):
                raise FileNotFoundError(f'Not found: {p}')

        model = load_image_model(args.weights, device)
        visualise_pair(model, img1_path, img2_path,
                       _read_text(txt1_path), _read_text(txt2_path),
                       args.output,
                       sim_threshold=args.sim_threshold)


if __name__ == '__main__':
    main()
