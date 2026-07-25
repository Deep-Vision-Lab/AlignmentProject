#!/usr/bin/env python3
"""Checkpoint-compatible Smith-Waterman local image alignment."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation._eval_utils import (
    compute_similarity,
    get_image_features,
    load_evaluation_models,
    patch_range_to_pixels,
    synthetic_pair_paths,
)


def smith_waterman(similarity: np.ndarray, threshold=0.45, gap_penalty=-0.30):
    n, m = similarity.shape
    score = np.zeros((n + 1, m + 1), dtype=np.float32)
    trace = np.zeros((n + 1, m + 1), dtype=np.uint8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + float(similarity[i - 1, j - 1]) - threshold
            up = score[i - 1, j] + gap_penalty
            left = score[i, j - 1] + gap_penalty
            values = (0.0, diag, up, left)
            best = int(np.argmax(values))
            score[i, j] = values[best]
            trace[i, j] = best
    i, j = map(int, np.unravel_index(np.argmax(score), score.shape))
    best_score = float(score[i, j])
    path = []
    while i > 0 and j > 0 and score[i, j] > 0:
        code = int(trace[i, j])
        if code == 1:
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif code == 2:
            i -= 1
        elif code == 3:
            j -= 1
        else:
            break
    path.reverse()
    return path, best_score, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="DataSet/Synthetic_Arabic")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--image1")
    parser.add_argument("--image2")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dataset-type",
        choices=("synthetic", "real"),
        default="synthetic",
    )
    parser.add_argument(
        "--feature",
        choices=("contextual", "local", "grouped"),
        default="contextual",
    )
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--gap", type=float, default=-0.30)
    parser.add_argument(
        "--output",
        default="Results/Evaluation/SW/smith_waterman.png",
    )
    args = parser.parse_args()

    pair = synthetic_pair_paths(args.data_dir, args.index)
    image1 = Path(args.image1) if args.image1 else pair.image1
    image2 = Path(args.image2) if args.image2 else pair.image2
    models = load_evaluation_models(args.weights, args.device, load_text_model=False)
    features1 = get_image_features(models, image1, args.dataset_type)
    features2 = get_image_features(models, image2, args.dataset_type)
    sim = compute_similarity(
        features1.select(args.feature),
        features2.select(args.feature),
    ).cpu().numpy()
    path, score, _ = smith_waterman(sim, args.threshold, args.gap)

    with Image.open(image1) as opened:
        arr1 = np.asarray(opened.convert("RGB"))
    with Image.open(image2) as opened:
        arr2 = np.asarray(opened.convert("RGB"))
    fig, axes = plt.subplots(2, 1, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(arr1, aspect="auto")
    axes[1].imshow(arr2, aspect="auto")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    if path:
        i_values, j_values = zip(*path)
        x01, x11 = patch_range_to_pixels(
            min(i_values),
            max(i_values) + 1,
            len(features1.contextual),
            arr1.shape[1],
            models.image_model.use_flip,
        )
        x02, x12 = patch_range_to_pixels(
            min(j_values),
            max(j_values) + 1,
            len(features2.contextual),
            arr2.shape[1],
            models.image_model.use_flip,
        )
        axes[0].add_patch(
            Rectangle(
                (x01, 1),
                x11 - x01,
                arr1.shape[0] - 2,
                facecolor="red",
                edgecolor="red",
                alpha=0.28,
                linewidth=2,
            )
        )
        axes[1].add_patch(
            Rectangle(
                (x02, 1),
                x12 - x02,
                arr2.shape[0] - 2,
                facecolor="red",
                edgecolor="red",
                alpha=0.28,
                linewidth=2,
            )
        )
    fig.suptitle(f"Smith-Waterman local image alignment | score={score:.4f}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved={output} score={score:.6f} path_steps={len(path)}")


if __name__ == "__main__":
    main()
