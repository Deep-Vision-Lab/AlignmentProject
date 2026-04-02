"""
Image-to-Image Segment Alignment  (no text required)
======================================================
Finds aligned *regions* between two manuscript images purely from the
NW alignment path and the cosine similarity matrix — no ground-truth text
is needed.

Algorithm
---------
1. Embed both images with the loaded model.
2. Compute a cosine similarity matrix  sim[s1, s2].
3. Run Needleman-Wunsch to get an alignment path  [(s1, s2), ...].
4. Walk the path:
   - if sim[s1, s2] >= ``--threshold``  → the step is *aligned*
   - otherwise                          → the step is a *gap*
5. Consecutive aligned steps become a **segment**.
   Segments shorter than ``--min-seg`` patches are discarded.
6. Each segment is drawn as a coloured rectangle on both images,
   linked by a connector line in the middle panel.

Usage (from project root)
--------------------------
    # single pair
    python Evaluation/eval_img_align_segments.py \\
        --weights  model_epoch_80.pth \\
        --index    3 \\
        --output   Results/Evaluation/seg_align_3.png

    # batch  (saves one figure per pair)
    python Evaluation/eval_img_align_segments.py \\
        --weights    model_epoch_80.pth \\
        --batch      --n-samples 50 \\
        --output-dir Results/Evaluation/SegAlign \\
        --threshold  0.45
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
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#cccccc',
]


# ---------------------------------------------------------------------------
# Core alignment
# ---------------------------------------------------------------------------

def align_images(model, img1_path: str, img2_path: str):
    """Return (sim_np, path, S1, S2)."""
    emb1 = get_image_embedding(model, img1_path, device)   # [1, S1, D]
    emb2 = get_image_embedding(model, img2_path, device)   # [1, S2, D]
    sim  = compute_sim_matrix(emb2, emb1)                  # [S1, S2]
    path = dtw_path(sim, gap_penalty=-10.0, match_score=10.0, mismatch_score=-27.0)
    sim_np = sim.detach().cpu().numpy() if hasattr(sim, 'detach') else np.array(sim)
    return sim_np, path, emb1.shape[1], emb2.shape[1]


# ---------------------------------------------------------------------------
# Segment extraction
# ---------------------------------------------------------------------------

def find_aligned_segments(path: list, sim_np: np.ndarray,
                           threshold: float = 0.45,
                           min_length: int = 2) -> list:
    """
    Walk the NW alignment path and group consecutive steps whose cosine
    similarity exceeds *threshold* into segments.

    Parameters
    ----------
    path       : list of (s1, s2) from dtw_path()
    sim_np     : numpy similarity matrix [S1, S2]
    threshold  : minimum cosine similarity for a step to be "aligned"
    min_length : discard segments shorter than this many path steps

    Returns
    -------
    List of dicts, each containing:
        s1_min, s1_max   – patch range on img1
        s2_min, s2_max   – patch range on img2
        mean_sim         – average cosine similarity along the segment
        steps            – list of (s1, s2) that make up the segment
    """
    segments = []
    current  = []

    for s1, s2 in path:
        score = float(sim_np[s1, s2])
        if score >= threshold:
            current.append((s1, s2))
        else:
            if len(current) >= min_length:
                segments.append(current)
            current = []

    # flush last segment
    if len(current) >= min_length:
        segments.append(current)

    result = []
    for seg in segments:
        s1s = [p[0] for p in seg]
        s2s = [p[1] for p in seg]
        sims = [sim_np[p[0], p[1]] for p in seg]
        result.append(dict(
            s1_min=min(s1s), s1_max=max(s1s),
            s2_min=min(s2s), s2_max=max(s2s),
            mean_sim=float(np.mean(sims)),
            steps=seg,
        ))
    return result


# ---------------------------------------------------------------------------
# Pixel conversion (RTL mirror — patch 0 = rightmost)
# ---------------------------------------------------------------------------

def _patch_to_pixels(p_start: int, p_end: int, n_patches: int,
                     img_width: int):
    """Map embedding patch range to pixel x-coordinates (RTL mirrored)."""
    px_start = (n_patches - 1 - p_end)       / n_patches * img_width
    px_end   = (n_patches - 1 - p_start + 1) / n_patches * img_width
    return px_start, px_end


# ---------------------------------------------------------------------------
# Single-pair visualisation
# ---------------------------------------------------------------------------

def visualise_pair(model,
                   img1_path: str,
                   img2_path: str,
                   output_path: str,
                   threshold: float = 0.45,
                   min_length: int = 2):
    """
    Produce a 3-panel figure:
      TOP    – img1 with coloured segment boxes
      MIDDLE – connector lines
      BOTTOM – img2 with matched segment boxes (same colours)
    """
    sim_np, path, S1, S2 = align_images(model, img1_path, img2_path)
    segments = find_aligned_segments(path, sim_np,
                                     threshold=threshold,
                                     min_length=min_length)

    img1_arr = np.array(Image.open(img1_path).convert('RGB'))
    img2_arr = np.array(Image.open(img2_path).convert('RGB'))
    H1, W1   = img1_arr.shape[:2]
    H2, W2   = img2_arr.shape[:2]

    n_seg = len(segments)

    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        f"Segment Alignment  (threshold={threshold:.2f})  —  "
        f"{os.path.basename(img1_path)}  ↔  {os.path.basename(img2_path)}\n"
        f"{n_seg} aligned segment(s) found",
        fontsize=10, fontweight='bold', y=1.02,
    )
    gs  = fig.add_gridspec(3, 1, height_ratios=[2.5, 0.8, 2.5], hspace=0.05)
    ax1   = fig.add_subplot(gs[0])
    ax_cn = fig.add_subplot(gs[1])
    ax2   = fig.add_subplot(gs[2])

    for ax in (ax1, ax2, ax_cn):
        ax.set_xticks([]); ax.set_yticks([])

    # ── img1 ────────────────────────────────────────────────────────────
    ax1.imshow(img1_arr, aspect='auto')
    ax1.set_ylabel('img1', fontsize=9, rotation=0, labelpad=28, va='center')

    # ── connectors ──────────────────────────────────────────────────────
    ax_cn.set_xlim(0, 1); ax_cn.set_ylim(0, 1)
    ax_cn.spines[:].set_visible(False)

    # ── img2 ────────────────────────────────────────────────────────────
    ax2.imshow(img2_arr, aspect='auto')
    ax2.set_ylabel('img2', fontsize=9, rotation=0, labelpad=28, va='center')

    for si, seg in enumerate(segments):
        col = PALETTE[si % len(PALETTE)]

        # ── img1 box ──
        x0_1, x1_1 = _patch_to_pixels(seg['s1_min'], seg['s1_max'], S1, W1)
        ax1.add_patch(mpatches.Rectangle(
            (x0_1, 2), max(x1_1 - x0_1, 2), H1 - 4,
            linewidth=2, edgecolor=col, facecolor=col, alpha=0.28,
        ))
        ax1.text((x0_1 + x1_1) / 2, H1 * 0.5,
                 f'{seg["mean_sim"]:.2f}',
                 ha='center', va='center', fontsize=7,
                 color='white', fontweight='bold',
                 bbox=dict(facecolor=col, alpha=0.65, pad=1, boxstyle='round'))

        # ── img2 box ──
        x0_2, x1_2 = _patch_to_pixels(seg['s2_min'], seg['s2_max'], S2, W2)
        ax2.add_patch(mpatches.Rectangle(
            (x0_2, 2), max(x1_2 - x0_2, 2), H2 - 4,
            linewidth=2, edgecolor=col, facecolor=col, alpha=0.28,
        ))
        ax2.text((x0_2 + x1_2) / 2, H2 * 0.5,
                 f'{seg["mean_sim"]:.2f}',
                 ha='center', va='center', fontsize=7,
                 color='white', fontweight='bold',
                 bbox=dict(facecolor=col, alpha=0.65, pad=1, boxstyle='round'))

        # ── connector ──
        x_top = 1.0 - ((seg['s1_min'] + seg['s1_max']) / 2 + 0.5) / S1
        x_bot = 1.0 - ((seg['s2_min'] + seg['s2_max']) / 2 + 0.5) / S2
        ax_cn.add_line(Line2D(
            [x_top, x_bot], [1.0, 0.0],
            transform=ax_cn.transAxes,
            color=col, linewidth=1.8, alpha=0.80,
        ))

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
        exist_ok=True,
    )
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[{n_seg} segments]  Saved: {output_path}")
    return segments


# ---------------------------------------------------------------------------
# Batch run
# ---------------------------------------------------------------------------

def batch_run(weights_path: str, data_dir: str,
              n_samples: int = 200,
              output_dir: str = 'Results/Evaluation/SegAlign',
              threshold: float = 0.45,
              min_length: int = 2):

    print("Loading model …")
    model = load_image_model(weights_path, device)
    pairs = list(load_test_pairs(data_dir, split='test', n_samples=n_samples))
    print(f"Processing {len(pairs)} pairs …\n")

    os.makedirs(output_dir, exist_ok=True)
    seg_counts, mean_sims, nw_costs = [], [], []

    for img1_path, _text1, img2_path, _text2 in tqdm(pairs, desc='Pairs'):
        sim_np, path, S1, S2 = align_images(model, img1_path, img2_path)

        i_num    = os.path.basename(img1_path).replace('img1_', '').replace('.png', '')
        out_path = os.path.join(output_dir, f'seg_{i_num}.png')

        # build a throwaway sim tensor for hard_dtw_cost
        import torch
        sim_t = torch.from_numpy(sim_np)
        cost  = hard_dtw_cost(sim_t,
                              gap_penalty=-10.0,
                              match_score=10.0,
                              mismatch_score=-27.0)

        segs = find_aligned_segments(path, sim_np,
                                     threshold=threshold,
                                     min_length=min_length)
        seg_counts.append(len(segs))
        if segs:
            mean_sims.append(np.mean([s['mean_sim'] for s in segs]))
        nw_costs.append(cost)

        # ── save figure ──
        img1_arr = np.array(Image.open(img1_path).convert('RGB'))
        img2_arr = np.array(Image.open(img2_path).convert('RGB'))
        H1, W1   = img1_arr.shape[:2]
        H2, W2   = img2_arr.shape[:2]

        fig = plt.figure(figsize=(14, 7))
        fig.suptitle(
            f"Segment Alignment  (thr={threshold:.2f})  pair {i_num}\n"
            f"{len(segs)} segment(s)  |  NW cost={cost:.4f}",
            fontsize=9, fontweight='bold', y=1.02,
        )
        gs    = fig.add_gridspec(3, 1, height_ratios=[2.5, 0.8, 2.5], hspace=0.05)
        ax1   = fig.add_subplot(gs[0])
        ax_cn = fig.add_subplot(gs[1])
        ax2   = fig.add_subplot(gs[2])
        for ax in (ax1, ax2, ax_cn):
            ax.set_xticks([]); ax.set_yticks([])

        ax1.imshow(img1_arr, aspect='auto');  ax1.set_ylabel('img1', fontsize=9, rotation=0, labelpad=28, va='center')
        ax2.imshow(img2_arr, aspect='auto');  ax2.set_ylabel('img2', fontsize=9, rotation=0, labelpad=28, va='center')
        ax_cn.set_xlim(0, 1); ax_cn.set_ylim(0, 1); ax_cn.spines[:].set_visible(False)

        for si, seg in enumerate(segs):
            col = PALETTE[si % len(PALETTE)]
            x0_1, x1_1 = _patch_to_pixels(seg['s1_min'], seg['s1_max'], S1, W1)
            x0_2, x1_2 = _patch_to_pixels(seg['s2_min'], seg['s2_max'], S2, W2)
            for ax, x0, x1, H in [(ax1, x0_1, x1_1, H1), (ax2, x0_2, x1_2, H2)]:
                ax.add_patch(mpatches.Rectangle(
                    (x0, 2), max(x1 - x0, 2), H - 4,
                    linewidth=2, edgecolor=col, facecolor=col, alpha=0.28))
                ax.text((x0 + x1) / 2, H * 0.5, f'{seg["mean_sim"]:.2f}',
                        ha='center', va='center', fontsize=7, color='white',
                        fontweight='bold',
                        bbox=dict(facecolor=col, alpha=0.65, pad=1, boxstyle='round'))
            x_top = 1.0 - ((seg['s1_min'] + seg['s1_max']) / 2 + 0.5) / S1
            x_bot = 1.0 - ((seg['s2_min'] + seg['s2_max']) / 2 + 0.5) / S2
            ax_cn.add_line(Line2D([x_top, x_bot], [1.0, 0.0],
                                  transform=ax_cn.transAxes,
                                  color=col, linewidth=1.8, alpha=0.80))

        plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()

    print("\n=====================================================")
    print("       Segment Alignment  —  Batch Results          ")
    print("=====================================================")
    print(f"  Pairs evaluated          : {len(pairs)}")
    print(f"  Mean segments / pair     : {np.mean(seg_counts):.1f}")
    print(f"  Mean segment cosine sim  : {np.mean(mean_sims):.4f}")
    print(f"  Mean NW cost             : {np.mean(nw_costs):.4f}")
    print("=====================================================\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Image-to-Image Segment Alignment (no text required)'
    )
    p.add_argument('--weights',    type=str,   default='model_epoch_80.pth')
    p.add_argument('--data-dir',   type=str,   default='DataSet/Synthetic_Arabic')
    p.add_argument('--index',      type=int,   default=None)
    p.add_argument('--output',     type=str,   default='Results/Evaluation/seg_align.png')
    p.add_argument('--threshold',  type=float, default=0.45,
                   help='Minimum cosine similarity to count a path step as aligned')
    p.add_argument('--min-seg',    type=int,   default=2,
                   help='Minimum number of path steps to keep a segment')
    p.add_argument('--batch',      action='store_true')
    p.add_argument('--n-samples',  type=int,   default=200)
    p.add_argument('--output-dir', type=str,   default='Results/Evaluation/SegAlign')
    args = p.parse_args()

    img_dir = os.path.join(args.data_dir, 'images')

    if args.batch:
        batch_run(
            args.weights, args.data_dir,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            threshold=args.threshold,
            min_length=args.min_seg,
        )
    else:
        idx       = args.index or 1
        img1_path = os.path.join(img_dir, f'img1_{idx}.png')
        img2_path = os.path.join(img_dir, f'img2_{idx}.png')

        for path_ in (img1_path, img2_path):
            if not os.path.exists(path_):
                raise FileNotFoundError(f'Not found: {path_}')

        model = load_image_model(args.weights, device)
        visualise_pair(
            model, img1_path, img2_path,
            args.output,
            threshold=args.threshold,
            min_length=args.min_seg,
        )


if __name__ == '__main__':
    main()
