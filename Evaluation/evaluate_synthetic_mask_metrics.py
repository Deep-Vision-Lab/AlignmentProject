#!/usr/bin/env python3
"""Evaluate synthetic alignment masks with precision, recall, IoU, Dice, and F1.

The image-alignment model predicts horizontal window spans, not vertical boxes.
For a fair comparison with the synthetic masks, each ground-truth PNG mask is
collapsed over height to a 1-D foreground-column mask. Predicted NW components
are converted to the same image-width column mask. Metrics are therefore
alignment-region segmentation metrics along the manuscript line direction.

The current component-aware Needleman-Wunsch evaluation patches are installed by
importing Evaluation.eval_img_align_nw before predictions are computed.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Importing this entry point installs the same geometry, checkpoint loader, and
# component-aware region interpretation used by the current NW visual evaluator.
from Evaluation import eval_img_align_nw as nw_eval
from Evaluation._eval_utils import (
    compute_similarity,
    get_image_features,
    load_evaluation_models,
    needleman_wunsch,
    patch_range_to_pixels,
)
from Evaluation.sw_core import build_match_scores, resolve_score_mode
from Evaluation.zero_shot_sw import ink_aware_match_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=27000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--test-start", type=int, default=1)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=0,
        help="Number of held-out test pairs. 0 means all remaining test pairs.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--feature", choices=("contextual", "local", "grouped"), default="contextual"
    )
    parser.add_argument(
        "--score-mode", choices=("auto", "raw", "centered", "mutual-z"), default="raw"
    )
    parser.add_argument("--score-clip", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--gap", type=float, default=-0.30)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--save-predicted-masks",
        action="store_true",
        help="Save 1-D predicted/GT masks expanded to image height for inspection.",
    )
    return parser.parse_args()


def exact_test_indices(total: int, seed: int) -> list[int]:
    train_size = int(0.6 * total)
    valid_size = int(0.2 * total)
    permutation = torch.randperm(
        total, generator=torch.Generator().manual_seed(int(seed))
    ).tolist()
    return [int(value) + 1 for value in permutation[train_size + valid_size :]]


def selected_test_indices(args: argparse.Namespace) -> list[int]:
    all_test = exact_test_indices(args.num_samples, args.split_seed)
    start = args.test_start - 1
    if start < 0 or start >= len(all_test):
        raise ValueError(
            f"test-start={args.test_start} is outside the {len(all_test)}-pair test split"
        )
    end = len(all_test) if args.n_samples == 0 else start + args.n_samples
    chosen = all_test[start:end]
    if args.n_samples > 0 and len(chosen) != args.n_samples:
        raise ValueError(
            f"Requested {args.n_samples} samples from test position {args.test_start}, "
            f"but only {len(chosen)} remain"
        )
    return chosen


def mask_path(root: Path, line: int, index: int) -> Path:
    return root / "masks" / f"mask{line}_{index}.png"


def image_path(root: Path, line: int, index: int) -> Path:
    return root / "images" / f"img{line}_{index}.png"


def gt_column_mask(path: Path) -> np.ndarray:
    with Image.open(path) as opened:
        array = np.asarray(opened.convert("L"))
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D mask at {path}, got {array.shape}")
    return np.any(array > 0, axis=0)


def path_runs(path) -> list[list[tuple[int, int]]]:
    explicit = getattr(path, "runs", None)
    if explicit is not None:
        return [list(run) for run in explicit if run]
    return [list(path)] if path else []


def predicted_column_mask(
    region_path,
    axis: int,
    n_windows: int,
    width: int,
    use_flip: bool,
) -> np.ndarray:
    mask = np.zeros(int(width), dtype=bool)
    for run in path_runs(region_path):
        values = [int(pair[axis]) for pair in run]
        if not values:
            continue
        left, right = patch_range_to_pixels(
            min(values), max(values) + 1, int(n_windows), int(width), bool(use_flip)
        )
        start = max(0, min(int(width), int(np.floor(min(left, right)))))
        end = max(0, min(int(width), int(np.ceil(max(left, right)))))
        if end > start:
            mask[start:end] = True
    return mask


def confusion(predicted: np.ndarray, target: np.ndarray) -> dict[str, int]:
    if predicted.shape != target.shape:
        raise ValueError(f"Mask shape mismatch: predicted={predicted.shape}, target={target.shape}")
    predicted = predicted.astype(bool, copy=False)
    target = target.astype(bool, copy=False)
    return {
        "tp": int(np.logical_and(predicted, target).sum()),
        "fp": int(np.logical_and(predicted, np.logical_not(target)).sum()),
        "fn": int(np.logical_and(np.logical_not(predicted), target).sum()),
        "tn": int(np.logical_and(np.logical_not(predicted), np.logical_not(target)).sum()),
    }


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int = 0) -> dict[str, float]:
    del tn
    tp, fp, fn = int(tp), int(fp), int(fn)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "dice": float(dice),
        "f1": float(f1),
    }


def add_counts(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
    return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in ("tp", "fp", "fn", "tn")}


def save_mask_preview(path: Path, columns: np.ndarray, height: int = 128) -> None:
    image = np.where(columns[None, :], 255, 0).astype(np.uint8)
    image = np.repeat(image, max(1, int(height)), axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(path)


def predict_pair(models, args, root: Path, dataset_index: int):
    image1 = image_path(root, 1, dataset_index)
    image2 = image_path(root, 2, dataset_index)
    mask1 = mask_path(root, 1, dataset_index)
    mask2 = mask_path(root, 2, dataset_index)
    for path in (image1, image2, mask1, mask2):
        if not path.is_file():
            raise FileNotFoundError(path)

    features1 = get_image_features(models, image1, "synthetic")
    features2 = get_image_features(models, image2, "synthetic")
    raw_similarity = compute_similarity(
        features1.select(args.feature), features2.select(args.feature)
    ).cpu().numpy()

    score_mode = resolve_score_mode(args.score_mode, "synthetic")
    match_scores = build_match_scores(
        raw_similarity, score_mode, args.score_clip, args.threshold
    )
    match_scores = ink_aware_match_scores(
        match_scores,
        features1.ink.detach().cpu().numpy(),
        features2.ink.detach().cpu().numpy(),
    )
    result = needleman_wunsch(
        match_scores, gap_penalty=args.gap, similarity_offset=0.0
    )
    trace = nw_eval.trace_alignment(result, match_scores, args.gap)

    gt1 = gt_column_mask(mask1)
    gt2 = gt_column_mask(mask2)
    pred1 = predicted_column_mask(
        trace.region_path,
        axis=0,
        n_windows=raw_similarity.shape[0],
        width=len(gt1),
        use_flip=models.image_model.use_flip,
    )
    pred2 = predicted_column_mask(
        trace.region_path,
        axis=1,
        n_windows=raw_similarity.shape[1],
        width=len(gt2),
        use_flip=models.image_model.use_flip,
    )

    counts1 = confusion(pred1, gt1)
    counts2 = confusion(pred2, gt2)
    pair_counts = add_counts(counts1, counts2)
    return {
        "result": result,
        "trace": trace,
        "raw_similarity": raw_similarity,
        "gt1": gt1,
        "gt2": gt2,
        "pred1": pred1,
        "pred2": pred2,
        "counts1": counts1,
        "counts2": counts2,
        "pair_counts": pair_counts,
    }


def flatten_metrics(prefix: str, counts: dict[str, int]) -> dict:
    metrics = metrics_from_counts(**counts)
    result = {f"{prefix}_{key}": int(value) for key, value in counts.items()}
    result.update({f"{prefix}_{key}": float(value) for key, value in metrics.items()})
    return result


def mean_metric(rows: list[dict], key: str) -> float:
    values = [float(row[key]) for row in rows if key in row and row[key] != ""]
    return float(np.mean(values)) if values else 0.0


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")
    if args.test_start <= 0:
        raise SystemExit("--test-start must be positive")
    if args.n_samples < 0:
        raise SystemExit("--n-samples must be >= 0")

    root = Path(args.data_dir).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    indices = selected_test_indices(args)
    (output / "selected_test_indices.json").write_text(
        json.dumps(
            {
                "split": "test",
                "split_seed": int(args.split_seed),
                "total_dataset_pairs": int(args.num_samples),
                "test_start": int(args.test_start),
                "evaluated_pairs": len(indices),
                "selected_dataset_indices": indices,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    models = load_evaluation_models(args.weights, args.device, load_text_model=False)
    rows: list[dict] = []
    total_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    total_line_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    no_prediction_pairs = 0

    for ordinal, dataset_index in enumerate(indices, start=1):
        prediction = predict_pair(models, args, root, dataset_index)
        result = prediction["result"]
        trace = prediction["trace"]
        counts1 = prediction["counts1"]
        counts2 = prediction["counts2"]
        pair_counts = prediction["pair_counts"]
        total_counts = add_counts(total_counts, pair_counts)
        total_line_counts = add_counts(total_line_counts, counts1)
        total_line_counts = add_counts(total_line_counts, counts2)

        pair_predicted = int(prediction["pred1"].sum() + prediction["pred2"].sum())
        if pair_predicted == 0:
            no_prediction_pairs += 1

        row = {
            "sample": ordinal,
            "dataset_index": int(dataset_index),
            "nw_score": float(result.score),
            "nw_normalized_score": float(result.normalized_score),
            "predicted_components": len(path_runs(trace.region_path)),
            "line1_predicted_columns": int(prediction["pred1"].sum()),
            "line1_gt_columns": int(prediction["gt1"].sum()),
            "line2_predicted_columns": int(prediction["pred2"].sum()),
            "line2_gt_columns": int(prediction["gt2"].sum()),
            **flatten_metrics("line1", counts1),
            **flatten_metrics("line2", counts2),
            **flatten_metrics("pair", pair_counts),
        }
        rows.append(row)

        if args.save_predicted_masks:
            preview = output / "mask_previews"
            save_mask_preview(preview / f"pred1_{dataset_index}.png", prediction["pred1"])
            save_mask_preview(preview / f"gt1_{dataset_index}.png", prediction["gt1"])
            save_mask_preview(preview / f"pred2_{dataset_index}.png", prediction["pred2"])
            save_mask_preview(preview / f"gt2_{dataset_index}.png", prediction["gt2"])

        pair_metrics = metrics_from_counts(**pair_counts)
        print(
            f"[{ordinal}/{len(indices)}] dataset_index={dataset_index} "
            f"components={row['predicted_components']} "
            f"precision={pair_metrics['precision']:.4f} "
            f"recall={pair_metrics['recall']:.4f} "
            f"iou={pair_metrics['iou']:.4f} "
            f"dice={pair_metrics['dice']:.4f}",
            flush=True,
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with (output / "mask_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    micro = metrics_from_counts(**total_counts)
    macro_pair = {
        key: mean_metric(rows, f"pair_{key}")
        for key in ("precision", "recall", "iou", "dice", "f1")
    }
    macro_line = {
        key: float(
            np.mean(
                [
                    mean_metric(rows, f"line1_{key}"),
                    mean_metric(rows, f"line2_{key}"),
                ]
            )
        )
        for key in ("precision", "recall", "iou", "dice", "f1")
    }
    summary = {
        "metric_space": "horizontal_mask_columns",
        "algorithm": "needleman_wunsch_component_aware",
        "evaluated_pairs": len(rows),
        "evaluated_lines": 2 * len(rows),
        "no_prediction_pairs": int(no_prediction_pairs),
        "micro_counts": total_counts,
        "micro": micro,
        "macro_pair": macro_pair,
        "macro_line": macro_line,
        "dice_equals_f1_for_binary_masks": True,
        "feature": args.feature,
        "score_mode": args.score_mode,
        "score_clip": float(args.score_clip),
        "threshold": float(args.threshold),
        "gap": float(args.gap),
        "split_seed": int(args.split_seed),
        "test_start": int(args.test_start),
        "requested_n_samples": int(args.n_samples),
    }
    (output / "mask_metrics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
