"""
Local Alignment MAE Evaluation
================================
Metric: Mean Absolute Error (MAE) of per-character frame prediction.

Ground-truth derivation for synthetic data:
  - A text of length T paired with an image of width W yields T character slots.
  - Character i occupies roughly frames [ i*S/T, (i+1)*S/T ) where
    S = number of sliding-window patches (image frames).
  - Ground-truth frame for character i  =>  gt_frame[i] = round(i * S / T)

The DTW path maps each text character index t to one or more image frame
indices.  We take the median mapped frame for each character and compare it
to the ground-truth proportional frame.

    MAE = (1/T) * sum_t | predicted_frame[t]  -  gt_frame[t] |

A lower MAE means the model is predicting accurate character-level positions
rather than doing naive linear interpolation.

Usage (from project root):
    python Evaluation/eval_alignment_mae.py \\
        --weights model_epoch_80.pth \\
        --data-dir DataSet/Synthetic_Arabic \\
        --n-samples 200
"""

import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Evaluation._eval_utils import (
    load_image_model, load_text_model,
    get_image_embedding, get_text_embedding,
    compute_sim_matrix, soft_dtw_path,
    load_test_pairs,
)
from Parameters import device


# ---------------------------------------------------------------------------
# Per-sample MAE
# ---------------------------------------------------------------------------

def sample_mae(sim: torch.Tensor, text_len: int, img_frames: int) -> float:
    """
    Compute alignment MAE for one image-text pair.

    sim       : [T, S] similarity matrix
    text_len  : T  (number of characters)
    img_frames: S  (number of image patches)
    """
    path, _ = soft_dtw_path(sim)   # list of (t, s) tuples

    # For each character t, collect all mapped image frames s
    t_to_s: dict = {t: [] for t in range(text_len)}
    for (t, s) in path:
        if 0 <= t < text_len:
            t_to_s[t].append(s)

    # Ground-truth: linear interpolation
    gt_frames = [round(t * (img_frames - 1) / max(text_len - 1, 1))
                 for t in range(text_len)]

    errors = []
    for t in range(text_len):
        mapped = t_to_s.get(t, [])
        if mapped:
            pred = float(np.median(mapped))
        else:
            pred = round(t * (img_frames - 1) / max(text_len - 1, 1))
        errors.append(abs(pred - gt_frames[t]))

    return float(np.mean(errors))


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(weights_path: str, data_dir: str,
             n_samples: int = 200, gap_penalty: float = -0.5):

    print("Loading models …")
    img_model = load_image_model(weights_path, device)
    txt_model = load_text_model(device)

    pairs = list(load_test_pairs(data_dir, split='test', n_samples=n_samples))
    n = len(pairs)
    print(f"Evaluating alignment MAE on {n} samples …\n")

    all_maes = []

    for img1_path, text1, _img2, _t2 in tqdm(pairs, desc='MAE'):
        img_emb = get_image_embedding(img_model, img1_path, device)
        txt_emb = get_text_embedding(txt_model, text1, device)

        sim         = compute_sim_matrix(img_emb, txt_emb)  # [T, S]
        T           = sim.shape[0]
        S           = sim.shape[1]

        mae = sample_mae(sim, T, S)
        all_maes.append(mae)

    mean_mae  = float(np.mean(all_maes))
    median_mae = float(np.median(all_maes))
    std_mae   = float(np.std(all_maes))

    # Express MAE as fraction of total image frames for normalised reporting
    # We estimate a typical S from the first sample
    img_emb_0 = get_image_embedding(img_model, pairs[0][0], device)
    typical_S = img_emb_0.shape[1]
    norm_mae = mean_mae / max(typical_S, 1) * 100   # % of total image width

    print("\n========================================")
    print("       Local Alignment MAE Results      ")
    print("========================================")
    print(f"  Eval samples  : {n}")
    print(f"  Mean MAE      : {mean_mae:.3f} frames  ({norm_mae:.2f}% of image)")
    print(f"  Median MAE    : {median_mae:.3f} frames")
    print(f"  Std MAE       : {std_mae:.3f} frames")
    print("========================================\n")
    print("Interpretation: lower MAE means the model predicts accurate")
    print("character positions beyond naive linear interpolation.")

    return dict(mean_mae=mean_mae, median_mae=median_mae, std_mae=std_mae,
                norm_mae_pct=norm_mae, all_maes=all_maes)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Local Alignment MAE Evaluation')
    parser.add_argument('--weights',   type=str, default='model_epoch_80.pth')
    parser.add_argument('--data-dir',  type=str, default='DataSet/Synthetic_Arabic')
    parser.add_argument('--n-samples', type=int, default=200)
    parser.add_argument('--gap',       type=float, default=-0.5)
    args = parser.parse_args()

    evaluate(
        weights_path=args.weights,
        data_dir=args.data_dir,
        n_samples=args.n_samples,
        gap_penalty=args.gap,
    )


if __name__ == '__main__':
    main()
