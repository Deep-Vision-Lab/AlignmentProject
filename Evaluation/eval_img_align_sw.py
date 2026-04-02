"""
Image-to-Image Alignment via Smith-Waterman  (no text required)
================================================================
Uses the **Smith-Waterman** local sequence alignment algorithm on the
cosine-similarity matrix to find the *longest / best-scoring contiguous
aligned region* between two manuscript images — no ground-truth text needed.

Why Smith-Waterman instead of Needleman-Wunsch?
-----------------------------------------------
NW is a *global* aligner: it forces every patch to be aligned or gapped.
SW is a *local* aligner: it resets to 0 whenever the running score turns
negative, so it returns only the highest-scoring contiguous block where
the two images genuinely agree.

Scoring
-------
  substitution(s1, s2) = sim[s1, s2] - threshold
      → positive  when patches are more similar than the threshold
      → negative  when patches are dissimilar (penalises bad matches)
  gap penalty          = --gap  (default -0.3, applied per gap step)

The highest-scoring (longest above-threshold) local alignment is then the
"best segment".  All other local alignments are found by masking out the
best one and re-running SW (--n-local, default 1 = only the best).

Layout of the output figure
---------------------------
  TOP    : img1 with one coloured rectangle per local alignment
  MIDDLE : connector lines
  BOTTOM : img2 with the matched region in the same colour

Usage (from project root)
--------------------------
    # single pair — best alignment only
    python Evaluation/eval_img_align_sw.py \\
        --weights  model_epoch_80.pth \\
        --index    3

    # top-3 local alignments
    python Evaluation/eval_img_align_sw.py \\
        --weights  model_epoch_80.pth \\
        --index    3 --n-local 3

    # batch
    python Evaluation/eval_img_align_sw.py \\
        --weights    model_epoch_80.pth \\
        --batch      --n-samples 50 \\
        --output-dir Results/Evaluation/SW_Align \\
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
# Smith-Waterman local alignment
# ---------------------------------------------------------------------------

def smith_waterman(sim_np: np.ndarray,
                   threshold: float = 0.45,
                   gap_penalty: float = -0.3):
    """
    Smith-Waterman local alignment on a cosine-similarity matrix.

    Substitution score  = sim[s1, s2] - threshold
      Positive when similarity exceeds the threshold → extends the alignment.
      Negative otherwise → discourages bad matches (may cause a reset to 0).

    Gap penalty is applied when one sequence is advanced without a match.

    Parameters
    ----------
    sim_np      : (S1, S2) cosine similarity  matrix
    threshold   : pivot score; pairs above this are rewarded, below penalised
    gap_penalty : cost per gap step (should be negative)

    Returns
    -------
    path      : list of (s1, s2) for the highest-scoring local alignment
    score     : the SW score of this alignment
    H         : the full (S1+1) x (S2+1) score matrix (for heatmap / debug)
    """
    S1, S2 = sim_np.shape
    H      = np.zeros((S1 + 1, S2 + 1), dtype=np.float32)
    # traceback codes: 0=stop, 1=diagonal, 2=gap-in-s2 (↑), 3=gap-in-s1 (←)
    tb     = np.zeros((S1 + 1, S2 + 1), dtype=np.int8)

    for i in range(1, S1 + 1):
        for j in range(1, S2 + 1):
            sub    = float(sim_np[i - 1, j - 1]) - threshold
            diag   = H[i - 1, j - 1] + sub
            up     = H[i - 1, j    ] + gap_penalty
            left   = H[i,     j - 1] + gap_penalty
            best   = max(0.0, diag, up, left)
            H[i, j] = best
            if   best == 0.0:  tb[i, j] = 0
            elif best == diag: tb[i, j] = 1
            elif best == up:   tb[i, j] = 2
            else:              tb[i, j] = 3

    # --- traceback from the global maximum ---
    max_score = float(H.max())
    if max_score <= 0:
        return [], max_score, H

    i, j  = map(int, np.unravel_index(np.argmax(H), H.shape))
    path  = []
    while H[i, j] > 0:
        code = int(tb[i, j])
        if   code == 0: break
        elif code == 1: path.append((i - 1, j - 1)); i -= 1; j -= 1
        elif code == 2: i -= 1          # gap in s2 (skip s1 patch)
        else:           j -= 1          # gap in s1 (skip s2 patch)

    path.reverse()
    return path, max_score, H


def smith_waterman_top_k(sim_np: np.ndarray,
                         k: int = 1,
                         threshold: float = 0.45,
                         gap_penalty: float = -0.3,
                         mask_radius: int = 2):
    """
    Find the top-k non-overlapping local alignments by iteratively masking
    out the best alignment and re-running SW.

    mask_radius : patches around the best alignment that are zeroed out
                  in the sim matrix before searching for the next one.
    """
    sim_work = sim_np.copy()
    results  = []

    for _ in range(k):
        path, score, H = smith_waterman(sim_work, threshold, gap_penalty)
        if not path or score <= 0:
            break
        results.append((path, score))

        # mask out the used region so next iteration finds a different one
        s1s = [p[0] for p in path]
        s2s = [p[1] for p in path]
        r1_lo = max(0, min(s1s) - mask_radius)
        r1_hi = min(sim_work.shape[0], max(s1s) + mask_radius + 1)
        r2_lo = max(0, min(s2s) - mask_radius)
        r2_hi = min(sim_work.shape[1], max(s2s) + mask_radius + 1)
        sim_work[r1_lo:r1_hi, r2_lo:r2_hi] = 0.0

    return results     # [(path, score), ...]


# ---------------------------------------------------------------------------
# Pixel conversion  (RTL mirror — patch 0 = rightmost)
# ---------------------------------------------------------------------------

def _patch_to_pixels(p_start: int, p_end: int,
                     n_patches: int, img_width: int):
    px_start = (n_patches - 1 - p_end)       / n_patches * img_width
    px_end   = (n_patches - 1 - p_start + 1) / n_patches * img_width
    return px_start, px_end


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def get_sim(model, img1_path: str, img2_path: str):
    emb1   = get_image_embedding(model, img1_path, device)   # [1, S1, D]
    emb2   = get_image_embedding(model, img2_path, device)   # [1, S2, D]
    sim_t  = compute_sim_matrix(emb2, emb1)                  # [S1, S2]
    sim_np = sim_t.detach().cpu().numpy() if hasattr(sim_t, 'detach') \
             else np.array(sim_t)
    return sim_np, emb1.shape[1], emb2.shape[1]


# ---------------------------------------------------------------------------
# Single-pair visualisation
# ---------------------------------------------------------------------------

def visualise_pair(model,
                   img1_path: str,
                   img2_path: str,
                   output_path: str,
                   threshold: float = 0.45,
                   gap_penalty: float = -0.3,
                   n_local: int = 1,
                   show_heatmap: bool = False):
    """
    3-panel figure (or 5-panel with heatmap) highlighting the best SW
    local alignment(s) between img1 and img2.
    """
    sim_np, S1, S2 = get_sim(model, img1_path, img2_path)

    # --- run SW, get top-k local alignments ---
    results = smith_waterman_top_k(sim_np, k=n_local,
                                   threshold=threshold,
                                   gap_penalty=gap_penalty)

    # also keep the score matrix of the FIRST SW run for heatmap
    _, best_score, H_full = smith_waterman(sim_np, threshold, gap_penalty)

    img1_arr = np.array(Image.open(img1_path).convert('RGB'))
    img2_arr = np.array(Image.open(img2_path).convert('RGB'))
    H1, W1   = img1_arr.shape[:2]
    H2, W2   = img2_arr.shape[:2]

    n_cols  = 2 if show_heatmap else 1
    fig_w   = 14 * n_cols
    fig, axes = plt.subplots(
        3, n_cols,
        figsize=(fig_w, 7),
        gridspec_kw=dict(height_ratios=[2.5, 0.8, 2.5], hspace=0.05,
                         wspace=0.03 if show_heatmap else 0),
    )
    if n_cols == 1:
        axes = axes[:, np.newaxis]   # shape -> (3, 1)

    ax1, ax_cn, ax2 = axes[0, 0], axes[1, 0], axes[2, 0]

    fig.suptitle(
        f"Smith-Waterman Local Alignment  (thr={threshold:.2f}, gap={gap_penalty:.2f})\n"
        f"{os.path.basename(img1_path)}  ↔  {os.path.basename(img2_path)}  "
        f"—  {len(results)} alignment(s) found  |  best score={best_score:.3f}",
        fontsize=10, fontweight='bold', y=1.02,
    )

    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])

    ax1.imshow(img1_arr, aspect='auto')
    ax1.set_ylabel('img1', fontsize=9, rotation=0, labelpad=28, va='center')
    ax2.imshow(img2_arr, aspect='auto')
    ax2.set_ylabel('img2', fontsize=9, rotation=0, labelpad=28, va='center')
    ax_cn.set_xlim(0, 1); ax_cn.set_ylim(0, 1)
    ax_cn.spines[:].set_visible(False)

    for ri, (path, score) in enumerate(results):
        if not path:
            continue
        col  = PALETTE[ri % len(PALETTE)]
        s1s  = [p[0] for p in path]
        s2s  = [p[1] for p in path]
        sims = [sim_np[p[0], p[1]] for p in path]
        mean_s = float(np.mean(sims))

        s1_min, s1_max = min(s1s), max(s1s)
        s2_min, s2_max = min(s2s), max(s2s)

        # img1 box
        x0_1, x1_1 = _patch_to_pixels(s1_min, s1_max, S1, W1)
        ax1.add_patch(mpatches.Rectangle(
            (x0_1, 2), max(x1_1 - x0_1, 2), H1 - 4,
            linewidth=2, edgecolor=col, facecolor=col, alpha=0.28,
        ))
        label = f'#{ri+1}  {mean_s:.2f}'
        ax1.text((x0_1 + x1_1) / 2, H1 * 0.5, label,
                 ha='center', va='center', fontsize=7,
                 color='white', fontweight='bold',
                 bbox=dict(facecolor=col, alpha=0.65, pad=1, boxstyle='round'))

        # img2 box
        x0_2, x1_2 = _patch_to_pixels(s2_min, s2_max, S2, W2)
        ax2.add_patch(mpatches.Rectangle(
            (x0_2, 2), max(x1_2 - x0_2, 2), H2 - 4,
            linewidth=2, edgecolor=col, facecolor=col, alpha=0.28,
        ))
        ax2.text((x0_2 + x1_2) / 2, H2 * 0.5, label,
                 ha='center', va='center', fontsize=7,
                 color='white', fontweight='bold',
                 bbox=dict(facecolor=col, alpha=0.65, pad=1, boxstyle='round'))

        # connector
        x_top = 1.0 - ((s1_min + s1_max) / 2 + 0.5) / S1
        x_bot = 1.0 - ((s2_min + s2_max) / 2 + 0.5) / S2
        ax_cn.add_line(Line2D(
            [x_top, x_bot], [1.0, 0.0],
            transform=ax_cn.transAxes,
            color=col, linewidth=2.0, alpha=0.85,
        ))

    # ── optional heatmap column ──────────────────────────────────────────
    if show_heatmap:
        ax_sim = axes[0, 1]
        ax_sw  = axes[1, 1]
        ax_leg = axes[2, 1]
        for ax in (ax_sim, ax_sw, ax_leg):
            ax.set_xticks([]); ax.set_yticks([])

        ax_sim.imshow(sim_np.T, origin='upper', aspect='auto',
                      cmap='viridis', vmin=0, vmax=1)
        ax_sim.set_title('Cosine similarity  (img1 × img2)', fontsize=8, pad=3)
        ax_sim.set_xlabel('img1 patches →', fontsize=7)
        ax_sim.set_ylabel('img2 patches →', fontsize=7)

        ax_sw.imshow(H_full[1:, 1:].T, origin='upper', aspect='auto',
                     cmap='hot')
        ax_sw.set_title('SW score matrix  H', fontsize=8, pad=3)

        # draw path on SW heatmap
        for ri, (path, _) in enumerate(results):
            if not path:
                continue
            col = PALETTE[ri % len(PALETTE)]
            xs  = [p[0] for p in path]
            ys  = [p[1] for p in path]
            ax_sw.plot(xs, ys, color=col, linewidth=1.5, alpha=0.9)
            ax_sim.plot(xs, ys, color=col, linewidth=1.5, alpha=0.9)

        ax_leg.axis('off')

    os.makedirs(
        os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
        exist_ok=True,
    )
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[best_score={best_score:.3f}, {len(results)} alignment(s)]  Saved: {output_path}")
    return results


# ---------------------------------------------------------------------------
# Batch run
# ---------------------------------------------------------------------------

def batch_run(weights_path: str, data_dir: str,
              n_samples: int = 200,
              output_dir: str = 'Results/Evaluation/SW_Align',
              threshold: float = 0.45,
              gap_penalty: float = -0.3,
              n_local: int = 1,
              show_heatmap: bool = False):

    print("Loading model …")
    model = load_image_model(weights_path, device)
    pairs = list(load_test_pairs(data_dir, split='test', n_samples=n_samples))
    print(f"Processing {len(pairs)} pairs …\n")
    os.makedirs(output_dir, exist_ok=True)

    all_scores, all_lengths = [], []

    for img1_path, _t1, img2_path, _t2 in tqdm(pairs, desc='Pairs'):
        sim_np, S1, S2 = get_sim(model, img1_path, img2_path)
        results        = smith_waterman_top_k(sim_np, k=n_local,
                                              threshold=threshold,
                                              gap_penalty=gap_penalty)
        if results:
            best_path, best_score = results[0]
            all_scores.append(best_score)
            all_lengths.append(len(best_path))

        i_num    = os.path.basename(img1_path).replace('img1_', '').replace('.png', '')
        out_path = os.path.join(output_dir, f'sw_{i_num}.png')
        visualise_pair(model, img1_path, img2_path, out_path,
                       threshold=threshold, gap_penalty=gap_penalty,
                       n_local=n_local, show_heatmap=show_heatmap)

    print("\n=====================================================")
    print("    Smith-Waterman Alignment  —  Batch Results      ")
    print("=====================================================")
    print(f"  Pairs evaluated          : {len(pairs)}")
    if all_scores:
        print(f"  Mean best SW score       : {np.mean(all_scores):.4f}")
        print(f"  Mean best path length    : {np.mean(all_lengths):.1f} steps")
        print(f"  Max best path length     : {int(np.max(all_lengths))} steps")
    print("=====================================================\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Image-to-Image Local Alignment via Smith-Waterman (no text)'
    )
    p.add_argument('--weights',     type=str,   default='model_epoch_80.pth')
    p.add_argument('--data-dir',    type=str,   default='DataSet/Synthetic_Arabic')
    p.add_argument('--index',       type=int,   default=None)
    p.add_argument('--output',      type=str,   default='Results/Evaluation/sw_align.png')
    p.add_argument('--threshold',   type=float, default=0.45,
                   help='Cosine-similarity pivot for SW substitution score')
    p.add_argument('--gap',         type=float, default=-0.3,
                   help='Gap penalty (negative value)')
    p.add_argument('--n-local',     type=int,   default=1,
                   help='Number of top non-overlapping local alignments to find')
    p.add_argument('--heatmap',     action='store_true',
                   help='Add a second column showing the similarity and SW score matrices')
    p.add_argument('--batch',       action='store_true')
    p.add_argument('--n-samples',   type=int,   default=200)
    p.add_argument('--output-dir',  type=str,   default='Results/Evaluation/SW_Align')
    args = p.parse_args()

    img_dir = os.path.join(args.data_dir, 'images')

    if args.batch:
        batch_run(
            args.weights, args.data_dir,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            threshold=args.threshold,
            gap_penalty=args.gap,
            n_local=args.n_local,
            show_heatmap=args.heatmap,
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
            gap_penalty=args.gap,
            n_local=args.n_local,
            show_heatmap=args.heatmap,
        )


if __name__ == '__main__':
    main()
