"""
Search for pairs where the best Smith-Waterman local alignment
starts near the BEGINNING of img1 and ends near the END of img2.

Arabic / RTL note
-----------------
The embedding model flips patches internally so that patch 0 = the
RIGHTMOST region of the image, which is the START of an Arabic line.
Therefore:
  "alignment starts at beginning of text1"  →  s1_min  near  0
  "alignment ends at end of text2"          →  s2_max  near  S2 - 1

Scoring
-------
Each candidate pair receives a "position score":
  pos_score = (1 - s1_min / S1) * (s2_max / (S2 - 1))
              ───────────────────   ─────────────────────
              1 if s1_min == 0      1 if s2_max == S2-1

So pos_score = 1.0 only for a perfect case; lower values indicate
the alignment is off-position.  We also require a minimum SW score
to filter out spurious short matches.

Usage (from project root)
--------------------------
    python Evaluation/search_start_end_alignment.py \\
        --weights  model_epoch_210.pth \\
        --data-dir DataSet/Synthetic_Arabic \\
        --top-k    10 \\
        --threshold 0.45 \\
        --min-sw-score 1.0 \\
        --output-dir Results/Evaluation/SearchStartEnd
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
]


# ---------------------------------------------------------------------------
# Smith-Waterman (same as eval_img_align_sw.py)
# ---------------------------------------------------------------------------

def smith_waterman(sim_np: np.ndarray,
                   threshold: float = 0.45,
                   gap_penalty: float = -0.3):
    S1, S2 = sim_np.shape
    H      = np.zeros((S1 + 1, S2 + 1), dtype=np.float32)
    tb     = np.zeros((S1 + 1, S2 + 1), dtype=np.int8)

    for i in range(1, S1 + 1):
        for j in range(1, S2 + 1):
            sub   = float(sim_np[i - 1, j - 1]) - threshold
            diag  = H[i - 1, j - 1] + sub
            up    = H[i - 1, j    ] + gap_penalty
            left  = H[i,     j - 1] + gap_penalty
            best  = max(0.0, diag, up, left)
            H[i, j] = best
            if   best == 0.0:  tb[i, j] = 0
            elif best == diag: tb[i, j] = 1
            elif best == up:   tb[i, j] = 2
            else:              tb[i, j] = 3

    max_score = float(H.max())
    if max_score <= 0:
        return [], max_score

    i, j = map(int, np.unravel_index(np.argmax(H), H.shape))
    path = []
    while H[i, j] > 0:
        code = int(tb[i, j])
        if   code == 0: break
        elif code == 1: path.append((i - 1, j - 1)); i -= 1; j -= 1
        elif code == 2: i -= 1
        else:           j -= 1

    path.reverse()
    return path, max_score


def get_sim(model, img1_path: str, img2_path: str):
    emb1   = get_image_embedding(model, img1_path, device)
    emb2   = get_image_embedding(model, img2_path, device)
    sim_t  = compute_sim_matrix(emb2, emb1)
    sim_np = sim_t.detach().cpu().numpy() if hasattr(sim_t, 'detach') \
             else np.array(sim_t)
    return sim_np, emb1.shape[1], emb2.shape[1]


# ---------------------------------------------------------------------------
# Position score
# ---------------------------------------------------------------------------

def position_score(s1_min: int, s2_max: int, S1: int, S2: int) -> float:
    """
    1.0  → perfect: alignment starts exactly at patch 0 of img1
                     AND ends at last patch of img2.
    0.0  → alignment is in the wrong place for both images.
    """
    start_score = 1.0 - s1_min / max(S1 - 1, 1)   # 1 if s1_min == 0
    end_score   = s2_max       / max(S2 - 1, 1)   # 1 if s2_max == S2-1
    return start_score * end_score


# ---------------------------------------------------------------------------
# Pixel helpers (RTL mirror)
# ---------------------------------------------------------------------------

def _patch_to_pixels(p_start, p_end, n_patches, img_width):
    px_s = (n_patches - 1 - p_end)       / n_patches * img_width
    px_e = (n_patches - 1 - p_start + 1) / n_patches * img_width
    return px_s, px_e


# ---------------------------------------------------------------------------
# Visualise one candidate pair
# ---------------------------------------------------------------------------

def visualise(img1_path, img2_path, path, sim_np,
              S1, S2, sw_score, pos_score_val,
              text1, text2, output_path, threshold):

    s1s = [p[0] for p in path]
    s2s = [p[1] for p in path]
    s1_min, s1_max = min(s1s), max(s1s)
    s2_min, s2_max = min(s2s), max(s2s)
    mean_sim = float(np.mean([sim_np[p[0], p[1]] for p in path]))

    img1_arr = np.array(Image.open(img1_path).convert('RGB'))
    img2_arr = np.array(Image.open(img2_path).convert('RGB'))
    H1, W1   = img1_arr.shape[:2]
    H2, W2   = img2_arr.shape[:2]

    col = PALETTE[0]

    fig = plt.figure(figsize=(14, 8))
    fig.suptitle(
        f"Start-of-img1 ↔ End-of-img2  Alignment\n"
        f"pair: {os.path.basename(img1_path)} ↔ {os.path.basename(img2_path)}\n"
        f'text1: "{text1}"\n'
        f'text2: "{text2}"\n'
        f"SW score={sw_score:.3f}  |  pos_score={pos_score_val:.3f}  |  "
        f"mean_sim={mean_sim:.3f}  |  thr={threshold:.2f}  |  "
        f"img1 patches [{s1_min}–{s1_max}]/{S1}   img2 patches [{s2_min}–{s2_max}]/{S2}",
        fontsize=9, fontweight='bold', y=1.02,
    )

    gs    = fig.add_gridspec(3, 1, height_ratios=[2.5, 0.6, 2.5], hspace=0.05)
    ax1   = fig.add_subplot(gs[0])
    ax_cn = fig.add_subplot(gs[1])
    ax2   = fig.add_subplot(gs[2])

    for ax in (ax1, ax2, ax_cn):
        ax.set_xticks([]); ax.set_yticks([])

    ax1.imshow(img1_arr, aspect='auto')
    ax1.set_ylabel('img1\n(start)', fontsize=8, rotation=0, labelpad=36, va='center')
    ax2.imshow(img2_arr, aspect='auto')
    ax2.set_ylabel('img2\n(end)', fontsize=8, rotation=0, labelpad=36, va='center')
    ax_cn.set_xlim(0, 1); ax_cn.set_ylim(0, 1)
    ax_cn.spines[:].set_visible(False)

    # img1 box (start of line)
    x0_1, x1_1 = _patch_to_pixels(s1_min, s1_max, S1, W1)
    ax1.add_patch(mpatches.Rectangle(
        (x0_1, 2), max(x1_1 - x0_1, 2), H1 - 4,
        linewidth=2.5, edgecolor=col, facecolor=col, alpha=0.30))
    ax1.text((x0_1 + x1_1) / 2, H1 * 0.5,
             f'start  sim={mean_sim:.2f}',
             ha='center', va='center', fontsize=8, color='white', fontweight='bold',
             bbox=dict(facecolor=col, alpha=0.7, pad=2, boxstyle='round'))

    # img2 box (end of line)
    x0_2, x1_2 = _patch_to_pixels(s2_min, s2_max, S2, W2)
    ax2.add_patch(mpatches.Rectangle(
        (x0_2, 2), max(x1_2 - x0_2, 2), H2 - 4,
        linewidth=2.5, edgecolor=col, facecolor=col, alpha=0.30))
    ax2.text((x0_2 + x1_2) / 2, H2 * 0.5,
             f'end  sim={mean_sim:.2f}',
             ha='center', va='center', fontsize=8, color='white', fontweight='bold',
             bbox=dict(facecolor=col, alpha=0.7, pad=2, boxstyle='round'))

    # connector
    x_top = 1.0 - ((s1_min + s1_max) / 2 + 0.5) / S1
    x_bot = 1.0 - ((s2_min + s2_max) / 2 + 0.5) / S2
    ax_cn.add_line(Line2D([x_top, x_bot], [1.0, 0.0],
                          transform=ax_cn.transAxes,
                          color=col, linewidth=2.2, alpha=0.85))

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.',
                exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main search loop
# ---------------------------------------------------------------------------

def search(weights_path, data_dir,
           n_samples=5000,
           top_k=10,
           threshold=0.45,
           gap_penalty=-0.3,
           min_sw_score=0.5,
           output_dir='Results/Evaluation/SearchStartEnd'):

    print("Loading model …")
    model = load_image_model(weights_path, device)
    pairs = list(load_test_pairs(data_dir, split=None, n_samples=n_samples))
    print(f"Scanning {len(pairs)} pairs …\n")

    candidates = []

    for img1_path, text1, img2_path, text2 in tqdm(pairs, desc='Search'):
        try:
            sim_np, S1, S2 = get_sim(model, img1_path, img2_path)
        except Exception:
            continue

        path, sw_score = smith_waterman(sim_np, threshold, gap_penalty)
        if not path or sw_score < min_sw_score:
            continue

        s1s = [p[0] for p in path]
        s2s = [p[1] for p in path]
        s1_min = min(s1s)
        s2_max = max(s2s)

        ps = position_score(s1_min, s2_max, S1, S2)
        candidates.append(dict(
            img1_path=img1_path, img2_path=img2_path,
            text1=text1, text2=text2,
            path=path, sw_score=sw_score, pos_score=ps,
            s1_min=s1_min, s1_max=max(s1s),
            s2_min=min(s2s), s2_max=s2_max,
            S1=S1, S2=S2,
            sim_np=sim_np,
        ))

    if not candidates:
        print("No candidates found — try lowering --threshold or --min-sw-score.")
        return

    # sort by position score (desc), then by SW score (desc)
    candidates.sort(key=lambda c: (c['pos_score'], c['sw_score']), reverse=True)
    top = candidates[:top_k]

    print(f"\nTop {len(top)} candidates (start-of-img1 ↔ end-of-img2):\n")
    print(f"{'Rank':<5} {'pair':>8} {'SW':>7} {'pos':>7} "
          f"{'s1_min':>7} {'s2_max':>7} {'S1':>5} {'S2':>5}  text1 / text2")
    print("-" * 100)

    os.makedirs(output_dir, exist_ok=True)

    for rank, c in enumerate(top, 1):
        i_num = os.path.basename(c['img1_path']).replace('img1_', '').replace('.png', '')
        print(f"  {rank:<4} {i_num:>8}  {c['sw_score']:>6.3f}  {c['pos_score']:>6.3f}  "
              f"{c['s1_min']:>6}  {c['s2_max']:>6}  {c['S1']:>4}  {c['S2']:>4}  "
              f'"{c["text1"]}" / "{c["text2"]}"')

        out_path = os.path.join(output_dir, f'rank{rank:02d}_pair{i_num}.png')
        visualise(
            c['img1_path'], c['img2_path'],
            c['path'], c['sim_np'],
            c['S1'], c['S2'],
            c['sw_score'], c['pos_score'],
            c['text1'], c['text2'],
            out_path, threshold,
        )

    print(f"\nFigures saved to: {output_dir}")
    return top


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description='Search for pairs aligned start-of-img1 ↔ end-of-img2'
    )
    p.add_argument('--weights',       type=str,   default='model_epoch_80.pth')
    p.add_argument('--data-dir',      type=str,   default='DataSet/Synthetic_Arabic')
    p.add_argument('--n-samples',     type=int,   default=5000,
                   help='Max number of pairs to scan (0 = all)')
    p.add_argument('--top-k',         type=int,   default=10,
                   help='Number of best candidates to save')
    p.add_argument('--threshold',     type=float, default=0.45,
                   help='SW cosine-similarity pivot')
    p.add_argument('--gap',           type=float, default=-0.3,
                   help='SW gap penalty')
    p.add_argument('--min-sw-score',  type=float, default=0.5,
                   help='Discard pairs whose best SW score is below this')
    p.add_argument('--output-dir',    type=str,
                   default='Results/Evaluation/SearchStartEnd')
    args = p.parse_args()

    search(
        args.weights,
        args.data_dir,
        n_samples=args.n_samples if args.n_samples > 0 else 10**9,
        top_k=args.top_k,
        threshold=args.threshold,
        gap_penalty=args.gap,
        min_sw_score=args.min_sw_score,
        output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
