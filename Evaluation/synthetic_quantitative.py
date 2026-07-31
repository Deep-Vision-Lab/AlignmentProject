#!/usr/bin/env python3
"""Quantitative image-alignment evaluation on the held-out synthetic split.

Measures (1) shared-region localization against generator masks and (2) true-pair
recognition against random held-out pairings. The 60/20/20 split exactly matches
DataLoader.py's seeded torch.utils.data.random_split protocol.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


def seeded_split_indices(dataset_size: int, seed: int = 42):
    """Return zero-based train/valid/test indices matching DataLoader.py."""
    size = int(dataset_size)
    if size <= 0:
        raise ValueError("dataset_size must be positive")
    train_size = int(0.6 * size)
    valid_size = int(0.2 * size)
    order = torch.randperm(
        size, generator=torch.Generator().manual_seed(int(seed))
    ).tolist()
    return (
        order[:train_size],
        order[train_size : train_size + valid_size],
        order[train_size + valid_size :],
    )


def mask_interval(mask_path: Path) -> tuple[int, int]:
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened.convert("L"))
    active = np.flatnonzero(np.any(mask > 0, axis=0))
    if not len(active):
        raise ValueError(f"Mask contains no active columns: {mask_path}")
    return int(active[0]), int(active[-1] + 1)


def interval_metrics(predicted, reference) -> dict[str, float]:
    """Metrics for two half-open intervals."""
    p0, p1 = map(float, predicted)
    g0, g1 = map(float, reference)
    pred_len = max(0.0, p1 - p0)
    gt_len = max(0.0, g1 - g0)
    intersection = max(0.0, min(p1, g1) - max(p0, g0))
    union = pred_len + gt_len - intersection
    precision = intersection / pred_len if pred_len else 0.0
    recall = intersection / gt_len if gt_len else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "iou": intersection / union if union else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "start_error": abs(p0 - g0),
        "end_error": abs(p1 - g1),
        "center_error": abs((p0 + p1 - g0 - g1) / 2.0),
        "length_error": abs(pred_len - gt_len),
        "pred_length": pred_len,
        "gt_length": gt_len,
    }


def gt_window_interval(gt_pixels, n_windows: int, width: int, flipped: bool):
    """Convert a pixel mask to logical window bins used by the evaluator."""
    centers = (np.arange(n_windows, dtype=np.float64) + 0.5) * width / n_windows
    active = np.flatnonzero((centers >= gt_pixels[0]) & (centers < gt_pixels[1]))
    if not len(active):
        midpoint = 0.5 * (gt_pixels[0] + gt_pixels[1])
        nearest = int(np.argmin(np.abs(centers - midpoint)))
        physical = (nearest, nearest + 1)
    else:
        physical = (int(active[0]), int(active[-1] + 1))
    if not flipped:
        return physical
    return n_windows - physical[1], n_windows - physical[0]


def predicted_pixel_interval(start, end_inclusive, n_windows, width, flipped):
    if int(start) < 0 or int(end_inclusive) < int(start):
        return 0.0, 0.0
    start = max(0, min(int(start), n_windows - 1))
    end = max(start + 1, min(int(end_inclusive) + 1, n_windows))
    if flipped:
        left = (n_windows - end) / n_windows * width
        right = (n_windows - start) / n_windows * width
    else:
        left = start / n_windows * width
        right = end / n_windows * width
    return float(min(left, right)), float(max(left, right))


def localization_metrics(*, pred_start, pred_end, gt_pixels, n_windows, width, flipped):
    empty = int(pred_start) < 0 or int(pred_end) < int(pred_start)
    pred_windows = (0, 0) if empty else (int(pred_start), int(pred_end) + 1)
    gt_windows = gt_window_interval(gt_pixels, n_windows, width, flipped)
    window = interval_metrics(pred_windows, gt_windows)
    pred_pixels = predicted_pixel_interval(
        pred_start, pred_end, n_windows, width, flipped
    )
    pixel = interval_metrics(pred_pixels, gt_pixels)
    result = {f"window_{key}": float(value) for key, value in window.items()}
    result.update({f"pixel_{key}": float(value) for key, value in pixel.items()})
    result.update(
        {
            "gt_window_start": int(gt_windows[0]),
            "gt_window_end": int(gt_windows[1] - 1),
            "pred_pixel_start": float(pred_pixels[0]),
            "pred_pixel_end": float(pred_pixels[1]),
            "gt_pixel_start": int(gt_pixels[0]),
            "gt_pixel_end": int(gt_pixels[1]),
        }
    )
    return result


def random_oracle_length_iou(gt_window, n_windows, trials, rng):
    """Chance localization using the correct span length but a random position."""
    gt_start, gt_end = map(int, gt_window)
    length = max(1, gt_end - gt_start)
    max_start = max(0, int(n_windows) - length)
    starts = rng.integers(0, max_start + 1, size=max(1, int(trials)))
    return float(
        np.mean(
            [
                interval_metrics((int(start), int(start) + length), gt_window)["iou"]
                for start in starts
            ]
        )
    )


def bootstrap_mean_ci(values: Iterable[float], seed: int, iterations: int):
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    means = np.empty(max(1, int(iterations)), dtype=np.float64)
    for index in range(len(means)):
        sample = rng.integers(0, len(array), size=len(array))
        means[index] = array[sample].mean()
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054):
    if total <= 0:
        return float("nan"), float("nan")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def roc_auc(positive_scores: Iterable[float], negative_scores: Iterable[float]):
    """Rank-based AUROC with average ranks for ties."""
    positive = np.asarray(list(positive_scores), dtype=np.float64)
    negative = np.asarray(list(negative_scores), dtype=np.float64)
    if not len(positive) or not len(negative):
        return float("nan")
    values = np.concatenate([positive, negative])
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * ((start + 1) + end)
        start = end
    rank_sum = ranks[: len(positive)].sum()
    return float(
        (rank_sum - len(positive) * (len(positive) + 1) / 2.0)
        / (len(positive) * len(negative))
    )


def mean(rows, key):
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return float(np.mean(values)) if values else float("nan")


def write_csv(path: Path, rows: list[dict]):
    fieldnames, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_commit(root: Path):
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return ""


def plot_reports(rows, positives, negatives, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ious = np.asarray([float(row["pair_mean_window_iou"]) for row in rows])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ious, bins=np.linspace(0, 1, 21))
    ax.set(xlabel="Pair mean window IoU", ylabel="Samples", title="Held-out synthetic localization")
    fig.tight_layout()
    fig.savefig(output_dir / "localization_iou_histogram.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(positives, bins=30, alpha=0.6, label="true pairs")
    ax.hist(negatives, bins=30, alpha=0.6, label="random pairs")
    ax.set(xlabel="Smith-Waterman local score", ylabel="Count", title="True versus random pair scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "pair_score_distributions.png", dpi=180)
    plt.close(fig)

    thresholds = np.linspace(0, 1, 101)
    minimum_ious = np.asarray([float(row["pair_min_window_iou"]) for row in rows])
    rates = [(minimum_ious >= threshold).mean() for threshold in thresholds]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thresholds, rates)
    ax.set(xlabel="Required IoU on both lines", ylabel="Success rate", title="Localization success curve")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_dir / "localization_success_curve.png", dpi=180)
    plt.close(fig)


def markdown_report(summary):
    p, loc, disc = summary["protocol"], summary["localization"], summary["pair_discrimination"]
    return "\n".join(
        [
            "# Synthetic alignment quantitative report",
            "",
            "## Protocol",
            "",
            f"- Checkpoint: `{p['weights']}`",
            f"- Git commit: `{p['git_commit']}`",
            f"- Exact held-out split: 60/20/20, seed **{p['split_seed']}**",
            f"- Test samples evaluated: **{p['evaluated_test_samples']}**",
            f"- Random negatives per query: **{p['negatives_per_query']}**",
            "",
            "## Localization",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| Mean pair window IoU | {loc['mean_pair_window_iou']:.4f} |",
            f"| 95% bootstrap CI | [{loc['mean_pair_window_iou_ci95'][0]:.4f}, {loc['mean_pair_window_iou_ci95'][1]:.4f}] |",
            f"| Mean pair pixel IoU | {loc['mean_pair_pixel_iou']:.4f} |",
            f"| Mean pair window F1 | {loc['mean_pair_window_f1']:.4f} |",
            f"| Both lines IoU >= 0.50 | {loc['both_iou_at_50']:.2%} |",
            f"| Both lines IoU >= 0.75 | {loc['both_iou_at_75']:.2%} |",
            f"| Both lines IoU >= 0.90 | {loc['both_iou_at_90']:.2%} |",
            f"| All four boundaries within one stride | {loc['all_boundaries_within_one_stride']:.2%} |",
            f"| Mean boundary error | {loc['mean_boundary_error_px']:.2f}px |",
            f"| Full-line baseline IoU | {loc['full_line_baseline_iou']:.4f} |",
            f"| Random-location oracle-length baseline IoU | {loc['random_oracle_length_baseline_iou']:.4f} |",
            "",
            "## Pair discrimination",
            "",
            "| Metric | Result |",
            "|---|---:|",
            f"| AUROC: true vs random pairs | {disc['auroc']:.4f} |",
            f"| Retrieval top-1 | {disc['top1_accuracy']:.2%} |",
            f"| Mean reciprocal rank | {disc['mean_reciprocal_rank']:.4f} |",
            f"| Mean true-minus-best-negative margin | {disc['mean_best_negative_margin']:.4f} |",
            "",
            "The primary metric is mean pair window IoU. The strict IoU >= 0.75 rate requires both line regions to be accurate. Pair AUROC and retrieval accuracy test whether the model identifies the correct partner instead of merely finding generic similar Arabic strokes.",
            "",
        ]
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="DataSet/Synthetic_Arabic")
    parser.add_argument("--dataset-size", type=int, default=0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--feature", choices=("contextual", "local", "grouped"), default="contextual")
    parser.add_argument("--score-mode", choices=("auto", "raw", "centered", "mutual-z"), default="auto")
    parser.add_argument("--score-clip", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--gap", type=float, default=-0.30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--negatives-per-query", type=int, default=9)
    parser.add_argument("--random-baseline-trials", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--output-dir", default="Results/Evaluation/Synthetic_Quantitative")
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    data_dir = Path(args.data_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    available = sorted(int(path.stem.split("_")[-1]) for path in (data_dir / "images").glob("img1_*.png"))
    if not available:
        raise SystemExit(f"No synthetic samples under {data_dir / 'images'}")
    dataset_size = int(args.dataset_size) if args.dataset_size > 0 else len(available)
    if dataset_size > len(available):
        raise SystemExit(f"dataset-size={dataset_size}, detected={len(available)}")

    _train, _valid, test_positions = seeded_split_indices(dataset_size, args.split_seed)
    test_indices = [position + 1 for position in test_positions]
    if args.max_test_samples > 0:
        test_indices = test_indices[: args.max_test_samples]
    if not test_indices:
        raise SystemExit("Held-out test split is empty")

    # Importing the canonical evaluator installs geometry and backend-aware loading.
    from Evaluation import eval_img_align_sw as evaluation
    from Evaluation.sw_core import alignment_region_metrics, dense_alignment_region
    from Evaluation.zero_shot_sw import ink_aware_match_scores

    models = evaluation.load_evaluation_models(args.weights, args.device, load_text_model=False)
    resolved_mode = evaluation.resolve_score_mode(args.score_mode, "synthetic")
    config = dict(models.config)
    stride = int(config.get("stride", 0))
    if stride <= 0:
        stride = max(1, int(config.get("window_size", 32) * float(config.get("stride_ratio", 0.5))))

    rng = np.random.default_rng(args.split_seed)
    rows, cache = [], []
    for ordinal, sample_index in enumerate(test_indices, 1):
        image1 = data_dir / "images" / f"img1_{sample_index}.png"
        image2 = data_dir / "images" / f"img2_{sample_index}.png"
        mask1 = data_dir / "masks" / f"mask1_{sample_index}.png"
        mask2 = data_dir / "masks" / f"mask2_{sample_index}.png"
        for path in (image1, image2, mask1, mask2):
            if not path.is_file():
                raise FileNotFoundError(path)

        features1 = evaluation.get_image_features(models, image1, "synthetic")
        features2 = evaluation.get_image_features(models, image2, "synthetic")
        selected1, selected2 = features1.select(args.feature), features2.select(args.feature)
        raw = evaluation.compute_similarity(selected1, selected2).detach().cpu().numpy()
        match = evaluation.build_match_scores(raw, resolved_mode, args.score_clip, args.threshold)
        match = ink_aware_match_scores(match, features1.ink.cpu().numpy(), features2.ink.cpu().numpy())
        path, score, _dp, traceback = evaluation.smith_waterman(
            raw, threshold=args.threshold, gap_penalty=args.gap,
            return_traceback=True, match_scores=match,
        )
        region = dense_alignment_region(path, traceback)
        region_stats = alignment_region_metrics(path, traceback, raw.shape)
        with Image.open(image1) as opened:
            width1 = opened.width
        with Image.open(image2) as opened:
            width2 = opened.width
        line1 = localization_metrics(
            pred_start=region.line1_start, pred_end=region.line1_end,
            gt_pixels=mask_interval(mask1), n_windows=raw.shape[0], width=width1,
            flipped=bool(models.image_model.use_flip),
        )
        line2 = localization_metrics(
            pred_start=region.line2_start, pred_end=region.line2_end,
            gt_pixels=mask_interval(mask2), n_windows=raw.shape[1], width=width2,
            flipped=bool(models.image_model.use_flip),
        )
        gt1 = (line1["gt_window_start"], line1["gt_window_end"] + 1)
        gt2 = (line2["gt_window_start"], line2["gt_window_end"] + 1)
        random_iou = 0.5 * (
            random_oracle_length_iou(gt1, raw.shape[0], args.random_baseline_trials, rng)
            + random_oracle_length_iou(gt2, raw.shape[1], args.random_baseline_trials, rng)
        )
        full_line_iou = 0.5 * ((gt1[1] - gt1[0]) / raw.shape[0] + (gt2[1] - gt2[0]) / raw.shape[1])
        boundary_errors = [
            line1["pixel_start_error"], line1["pixel_end_error"],
            line2["pixel_start_error"], line2["pixel_end_error"],
        ]
        row = {
            "sample_index": sample_index,
            "test_ordinal": ordinal,
            "score": float(score),
            "score_per_window": float(score) / max(raw.shape),
            **region_stats,
            **{f"line1_{key}": value for key, value in line1.items()},
            **{f"line2_{key}": value for key, value in line2.items()},
            "pair_mean_window_iou": 0.5 * (line1["window_iou"] + line2["window_iou"]),
            "pair_min_window_iou": min(line1["window_iou"], line2["window_iou"]),
            "pair_mean_window_f1": 0.5 * (line1["window_f1"] + line2["window_f1"]),
            "pair_mean_pixel_iou": 0.5 * (line1["pixel_iou"] + line2["pixel_iou"]),
            "pair_min_pixel_iou": min(line1["pixel_iou"], line2["pixel_iou"]),
            "mean_boundary_error_px": float(np.mean(boundary_errors)),
            "max_boundary_error_px": float(max(boundary_errors)),
            "all_boundaries_within_one_stride": bool(max(boundary_errors) <= stride),
            "random_oracle_length_iou": random_iou,
            "full_line_baseline_iou": full_line_iou,
        }
        rows.append(row)
        cache.append(
            {
                "left": selected1.detach().cpu().numpy().astype(np.float32),
                "right": selected2.detach().cpu().numpy().astype(np.float32),
                "ink_left": features1.ink.detach().cpu().numpy().astype(np.float32),
                "ink_right": features2.ink.detach().cpu().numpy().astype(np.float32),
            }
        )
        print(
            f"[{ordinal}/{len(test_indices)}] sample={sample_index} "
            f"pair_iou={row['pair_mean_window_iou']:.4f} "
            f"min_iou={row['pair_min_window_iou']:.4f} "
            f"boundary_mae={row['mean_boundary_error_px']:.2f}px",
            flush=True,
        )

    def cached_score(left_index, right_index):
        left, right = cache[left_index], cache[right_index]
        raw = left["left"] @ right["right"].T
        match = evaluation.build_match_scores(raw, resolved_mode, args.score_clip, args.threshold)
        match = ink_aware_match_scores(match, left["ink_left"], right["ink_right"])
        _path, score, _dp, _trace = evaluation.smith_waterman(
            raw, threshold=args.threshold, gap_penalty=args.gap,
            return_traceback=True, match_scores=match,
        )
        return float(score)

    positives = [float(row["score"]) for row in rows]
    negatives, ranks = [], []
    k = min(max(0, args.negatives_per_query), max(0, len(rows) - 1))
    for query, row in enumerate(rows):
        candidates = np.delete(np.arange(len(rows)), query)
        chosen = rng.choice(candidates, size=k, replace=False) if k else []
        scores = [cached_score(query, int(candidate)) for candidate in chosen]
        negatives.extend(scores)
        rank = 1 + sum(value > row["score"] for value in scores)
        ranks.append(rank)
        row["retrieval_rank"] = rank
        row["retrieval_top1"] = rank == 1
        row["best_negative_score"] = max(scores) if scores else 0.0
        row["best_negative_margin"] = row["score"] - max(scores) if scores else 0.0

    positives = [float(row["score"]) for row in rows]
    pair_ious = [row["pair_mean_window_iou"] for row in rows]
    pair_min_ious = [row["pair_min_window_iou"] for row in rows]
    ci = bootstrap_mean_ci(pair_ious, args.split_seed + 101, args.bootstrap_samples)
    counts = {threshold: sum(value >= threshold for value in pair_min_ious) for threshold in (0.5, 0.75, 0.9)}
    cis = {threshold: wilson_interval(count, len(rows)) for threshold, count in counts.items()}
    test_hash = hashlib.sha256(",".join(map(str, test_indices)).encode()).hexdigest()

    summary = {
        "protocol": {
            "weights": str(Path(args.weights).expanduser().resolve()),
            "git_commit": git_commit(root),
            "dataset_dir": str(data_dir),
            "dataset_size": dataset_size,
            "split_seed": args.split_seed,
            "test_split_size": len(test_positions),
            "evaluated_test_samples": len(rows),
            "test_indices_sha256": test_hash,
            "feature": args.feature,
            "score_mode": resolved_mode,
            "threshold": args.threshold,
            "gap": args.gap,
            "ink_aware": os.environ.get("SW_INK_AWARE", "1"),
            "negatives_per_query": k,
            "window_size": int(config.get("window_size", 32)),
            "stride": stride,
            "flipped": bool(models.image_model.use_flip),
            "model_backend": config.get("model_backend", config.get("visual_encoder_type", "")),
        },
        "localization": {
            "primary_metric": "mean_pair_window_iou",
            "mean_pair_window_iou": mean(rows, "pair_mean_window_iou"),
            "mean_pair_window_iou_ci95": list(ci),
            "mean_pair_min_window_iou": mean(rows, "pair_min_window_iou"),
            "mean_pair_window_f1": mean(rows, "pair_mean_window_f1"),
            "mean_pair_pixel_iou": mean(rows, "pair_mean_pixel_iou"),
            "mean_pair_min_pixel_iou": mean(rows, "pair_min_pixel_iou"),
            "mean_boundary_error_px": mean(rows, "mean_boundary_error_px"),
            "mean_max_boundary_error_px": mean(rows, "max_boundary_error_px"),
            "all_boundaries_within_one_stride": float(np.mean([row["all_boundaries_within_one_stride"] for row in rows])),
            "both_iou_at_50": counts[0.5] / len(rows),
            "both_iou_at_75": counts[0.75] / len(rows),
            "both_iou_at_90": counts[0.9] / len(rows),
            "both_iou_at_50_ci95": list(cis[0.5]),
            "both_iou_at_75_ci95": list(cis[0.75]),
            "both_iou_at_90_ci95": list(cis[0.9]),
            "full_line_baseline_iou": mean(rows, "full_line_baseline_iou"),
            "random_oracle_length_baseline_iou": mean(rows, "random_oracle_length_iou"),
            "lift_over_full_line": mean(rows, "pair_mean_window_iou") - mean(rows, "full_line_baseline_iou"),
            "lift_over_random_oracle_length": mean(rows, "pair_mean_window_iou") - mean(rows, "random_oracle_length_iou"),
            "mean_warp_steps": mean(rows, "warp_steps"),
            "mean_path_steps": mean(rows, "path_steps"),
        },
        "pair_discrimination": {
            "auroc": roc_auc(positives, negatives),
            "top1_accuracy": float(np.mean([rank == 1 for rank in ranks])) if ranks else 0.0,
            "mean_reciprocal_rank": float(np.mean([1 / rank for rank in ranks])) if ranks else 0.0,
            "mean_positive_score": float(np.mean(positives)),
            "mean_random_negative_score": float(np.mean(negatives)) if negatives else 0.0,
            "mean_best_negative_margin": mean(rows, "best_negative_margin"),
            "random_pair_count": len(negatives),
        },
        "worst_samples": [
            {
                "sample_index": row["sample_index"],
                "pair_min_window_iou": row["pair_min_window_iou"],
                "pair_mean_window_iou": row["pair_mean_window_iou"],
                "mean_boundary_error_px": row["mean_boundary_error_px"],
            }
            for row in sorted(rows, key=lambda item: item["pair_min_window_iou"])[:25]
        ],
    }

    write_csv(output_dir / "samples.csv", rows)
    write_csv(output_dir / "worst_cases.csv", sorted(rows, key=lambda item: item["pair_min_window_iou"])[:25])
    (output_dir / "test_indices.json").write_text(json.dumps(test_indices, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(markdown_report(summary), encoding="utf-8")
    plot_reports(rows, positives, negatives, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
