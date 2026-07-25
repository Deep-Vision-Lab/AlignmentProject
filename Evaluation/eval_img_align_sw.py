#!/usr/bin/env python3
"""Checkpoint-compatible Smith-Waterman local image alignment."""
from __future__ import annotations

import argparse
import csv
import json
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


def _save_visualization(
    image1,
    image2,
    features1,
    features2,
    path,
    score,
    output,
    use_flip,
):
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
            use_flip,
        )
        x02, x12 = patch_range_to_pixels(
            min(j_values),
            max(j_values) + 1,
            len(features2.contextual),
            arr2.shape[1],
            use_flip,
        )
        for ax, array, x0, x1 in (
            (axes[0], arr1, x01, x11),
            (axes[1], arr2, x02, x12),
        ):
            ax.add_patch(
                Rectangle(
                    (x0, 1),
                    x1 - x0,
                    array.shape[0] - 2,
                    facecolor="red",
                    edgecolor="red",
                    alpha=0.28,
                    linewidth=2,
                )
            )

    fig.suptitle(f"Smith-Waterman local image alignment | score={score:.4f}")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _evaluate_sample(
    models,
    image1,
    image2,
    index,
    dataset_type,
    feature,
    threshold,
    gap,
    output,
):
    features1 = get_image_features(models, image1, dataset_type)
    features2 = get_image_features(models, image2, dataset_type)
    similarity = compute_similarity(
        features1.select(feature),
        features2.select(feature),
    ).cpu().numpy()
    path, score, _score_matrix = smith_waterman(similarity, threshold, gap)

    _save_visualization(
        image1,
        image2,
        features1,
        features2,
        path,
        score,
        output,
        models.image_model.use_flip,
    )

    path_similarities = [float(similarity[i, j]) for i, j in path]
    row = {
        "index": int(index),
        "status": "ok",
        "score": float(score),
        "path_steps": len(path),
        "mean_path_cosine": float(np.mean(path_similarities)) if path_similarities else 0.0,
        "line1_windows": int(similarity.shape[0]),
        "line2_windows": int(similarity.shape[1]),
        "line1_path_start": int(path[0][0]) if path else -1,
        "line1_path_end": int(path[-1][0]) if path else -1,
        "line2_path_start": int(path[0][1]) if path else -1,
        "line2_path_end": int(path[-1][1]) if path else -1,
        "feature": str(feature),
        "threshold": float(threshold),
        "gap": float(gap),
        "flipped": bool(models.image_model.use_flip),
        "image1": str(image1),
        "image2": str(image2),
        "output": str(output),
        "error": "",
    }
    print(
        f"[{index}] score={score:.6f} path_steps={len(path)} "
        f"mean_cosine={row['mean_path_cosine']:.4f} saved={output}",
        flush=True,
    )
    return row


def _aggregate(rows):
    successful = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    scores = [float(row["score"]) for row in successful]
    path_steps = [float(row["path_steps"]) for row in successful]
    path_cosines = [float(row["mean_path_cosine"]) for row in successful]
    return {
        "samples": len(rows),
        "successful": len(successful),
        "failed": len(failed),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "mean_path_steps": float(np.mean(path_steps)) if path_steps else 0.0,
        "mean_path_cosine": float(np.mean(path_cosines)) if path_cosines else 0.0,
        "failed_indices": [int(row["index"]) for row in failed],
    }


def _write_batch_outputs(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "index",
        "status",
        "score",
        "path_steps",
        "mean_path_cosine",
        "line1_windows",
        "line2_windows",
        "line1_path_start",
        "line1_path_end",
        "line2_path_start",
        "line2_path_end",
        "feature",
        "threshold",
        "gap",
        "flipped",
        "image1",
        "image2",
        "output",
        "error",
    ]
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(_aggregate(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="DataSet/Synthetic_Arabic")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch", action="store_true")
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
    parser.add_argument(
        "--output-dir",
        default="Results/Evaluation/SW/windows",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch and (args.image1 or args.image2):
        raise SystemExit("--image1/--image2 are single-sample options and cannot be used with --batch")
    if args.batch and args.dataset_type != "synthetic":
        raise SystemExit("Batch discovery currently supports only --dataset-type synthetic")
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be greater than zero")

    models = load_evaluation_models(args.weights, args.device, load_text_model=False)

    if not args.batch:
        pair = synthetic_pair_paths(args.data_dir, args.index)
        image1 = Path(args.image1) if args.image1 else pair.image1
        image2 = Path(args.image2) if args.image2 else pair.image2
        _evaluate_sample(
            models,
            image1,
            image2,
            args.index,
            args.dataset_type,
            args.feature,
            args.threshold,
            args.gap,
            Path(args.output),
        )
        return

    output_dir = Path(args.output_dir)
    rows = []
    for index in range(args.start_index, args.start_index + args.n_samples):
        output = output_dir / f"pair_{index}.png"
        try:
            pair = synthetic_pair_paths(args.data_dir, index)
            row = _evaluate_sample(
                models,
                pair.image1,
                pair.image2,
                index,
                args.dataset_type,
                args.feature,
                args.threshold,
                args.gap,
                output,
            )
        except Exception as exc:
            row = {
                "index": int(index),
                "status": "error",
                "score": 0.0,
                "path_steps": 0,
                "mean_path_cosine": 0.0,
                "line1_windows": 0,
                "line2_windows": 0,
                "line1_path_start": -1,
                "line1_path_end": -1,
                "line2_path_start": -1,
                "line2_path_end": -1,
                "feature": str(args.feature),
                "threshold": float(args.threshold),
                "gap": float(args.gap),
                "flipped": bool(models.image_model.use_flip),
                "image1": "",
                "image2": "",
                "output": str(output),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[{index}] failed: {row['error']}", file=sys.stderr, flush=True)
        rows.append(row)

    _write_batch_outputs(rows, output_dir)
    summary = _aggregate(rows)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
