#!/usr/bin/env python3
"""Label-free Point-4/5 diagnostics for real image-image alignment.

Cycle consistency and perturbation stability are diagnostics only. They measure
self-consistency/robustness and must not be reported as localization accuracy.
"""
from __future__ import annotations

import math
from pathlib import Path
import random
import tempfile

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _safe_cv(values):
    vals = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=np.float64)
    if vals.size < 2:
        return 0.0
    mean = float(np.mean(vals))
    if abs(mean) < 1e-12:
        return None
    return float(np.std(vals, ddof=0) / abs(mean))


def _interval_iou(a, b):
    inter = max(0.0, min(float(a[1]), float(b[1])) - max(float(a[0]), float(b[0])))
    union = max(float(a[1]), float(b[1])) - min(float(a[0]), float(b[0]))
    return float(inter / union) if union > 0 else 0.0


def _path_iou(first, second):
    a = {tuple(map(int, p)) for p in first}
    b = {tuple(map(int, p)) for p in second}
    union = a | b
    return float(len(a & b) / len(union)) if union else 1.0


def _region_intervals(runtime, models, alignment, width=1024):
    region = alignment["region"]
    if region.empty:
        return (0.0, 0.0), (0.0, 0.0)
    left = runtime.utils.patch_range_to_pixels(
        region.line1_start,
        region.line1_end + 1,
        alignment["line1_windows"],
        width,
        bool(models.image_model.use_flip),
    )
    right = runtime.utils.patch_range_to_pixels(
        region.line2_start,
        region.line2_end + 1,
        alignment["line2_windows"],
        width,
        bool(models.image_model.use_flip),
    )
    return tuple(map(float, left)), tuple(map(float, right))


def run_cycle_consistency(runtime, models, pairs, args, output: Path):
    """Nearest-neighbor A->B->A cycle diagnostic over valid/ink windows."""
    from Evaluation.quantitative_real import _feature_cache, _mean, _median, _write_csv

    selected = list(pairs)
    random.Random(args.seed + 222).shuffle(selected)
    selected = selected[: min(args.cycle_pairs, len(selected))]
    rows = []
    with tempfile.TemporaryDirectory(prefix="cycle_quant_") as tmp:
        get_features = _feature_cache(runtime, models, "real", Path(tmp))
        for order, pair in enumerate(selected, start=1):
            left = get_features(pair.image1)
            right = get_features(pair.image2)
            similarity = runtime.utils.compute_similarity(
                left.select(args.feature), right.select(args.feature)
            ).detach().cpu().numpy()

            left_valid = np.flatnonzero(left.ink.detach().cpu().numpy() >= args.min_ink)
            right_valid = np.flatnonzero(right.ink.detach().cpu().numpy() >= args.min_ink)
            if left_valid.size == 0:
                left_valid = np.arange(similarity.shape[0])
            if right_valid.size == 0:
                right_valid = np.arange(similarity.shape[1])

            errors_windows = []
            errors_norm = []
            for i in left_valid:
                j = int(right_valid[np.argmax(similarity[int(i), right_valid])])
                i_hat = int(left_valid[np.argmax(similarity[left_valid, j])])
                error = abs(int(i) - i_hat)
                errors_windows.append(error)
                errors_norm.append(error / max(1, similarity.shape[0]))

            rows.append(
                {
                    "query_order": order,
                    "pair_id": pair.pair_id,
                    "label_type": pair.label_type,
                    "image1": str(pair.image1),
                    "image2": str(pair.image2),
                    "cycles": len(errors_windows),
                    "mean_cycle_error_windows": _mean(errors_windows),
                    "median_cycle_error_windows": _median(errors_windows),
                    "mean_cycle_error_normalized": _mean(errors_norm),
                    "within_1_window": _mean(e <= 1 for e in errors_windows),
                    "within_2_windows": _mean(e <= 2 for e in errors_windows),
                    "within_4_windows": _mean(e <= 4 for e in errors_windows),
                }
            )

    _write_csv(output / "cycle_consistency.csv", rows)
    summary = {
        "diagnostic_only": True,
        "pairs": len(rows),
        "mean_cycle_error_windows": _mean(r["mean_cycle_error_windows"] for r in rows),
        "median_cycle_error_windows": _median(
            r["median_cycle_error_windows"] for r in rows
        ),
        "mean_cycle_error_normalized": _mean(
            r["mean_cycle_error_normalized"] for r in rows
        ),
        "within_1_window": _mean(r["within_1_window"] for r in rows),
        "within_2_windows": _mean(r["within_2_windows"] for r in rows),
        "within_4_windows": _mean(r["within_4_windows"] for r in rows),
        "warning": "Cycle consistency can be low even for a consistently wrong mapping.",
    }
    return rows, summary


def _background_rgb(array):
    edges = np.concatenate(
        [
            array[0].reshape(-1, 3),
            array[-1].reshape(-1, 3),
            array[:, 0].reshape(-1, 3),
            array[:, -1].reshape(-1, 3),
        ],
        axis=0,
    )
    return tuple(np.median(edges, axis=0).astype(np.uint8).tolist())


def _morphology(image: Image.Image, *, dilate_ink: bool) -> Image.Image:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    border = float(
        np.median(np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]]))
    )
    dark_ink = border > 127.5
    working = Image.fromarray(255 - gray if dark_ink else gray)
    filt = ImageFilter.MaxFilter(3) if dilate_ink else ImageFilter.MinFilter(3)
    result = np.asarray(working.filter(filt), dtype=np.uint8)
    if dark_ink:
        result = 255 - result
    return Image.fromarray(result, mode="L").convert("RGB")


def _perturb(array, mode, rng):
    image = Image.fromarray(array).convert("RGB")
    mode = str(mode).strip().lower()
    if mode == "blur":
        return np.asarray(image.filter(ImageFilter.GaussianBlur(radius=1.2)))
    if mode == "contrast":
        return np.asarray(ImageEnhance.Contrast(image).enhance(0.70))
    if mode == "brightness":
        return np.asarray(ImageEnhance.Brightness(image).enhance(0.82))
    if mode == "noise":
        values = np.asarray(image, dtype=np.float32)
        noise = rng.normal(0.0, 10.0, values.shape)
        return np.clip(values + noise, 0, 255).astype(np.uint8)
    if mode == "horizontal_scale":
        width, height = image.size
        new_width = max(8, int(round(width * 0.97)))
        resized = image.resize((new_width, height), Image.BILINEAR)
        background = _background_rgb(np.asarray(image))
        canvas = Image.new("RGB", (width, height), background)
        x = (width - new_width) // 2
        canvas.paste(resized, (x, 0))
        return np.asarray(canvas)
    if mode == "vertical_shift":
        width, height = image.size
        background = _background_rgb(np.asarray(image))
        canvas = Image.new("RGB", (width, height), background)
        canvas.paste(image, (0, 3))
        return np.asarray(canvas)
    if mode == "erosion":
        return np.asarray(_morphology(image, dilate_ink=False))
    if mode == "dilation":
        return np.asarray(_morphology(image, dilate_ink=True))
    raise ValueError(f"Unsupported robustness perturbation: {mode}")


def run_robustness(runtime, models, pairs, args, output: Path):
    """Perturb only line 2 and compare predicted alignment with baseline."""
    from Evaluation.quantitative_real import _alignment, _feature_cache, _mean, _write_csv

    modes = [item.strip() for item in args.robustness_modes.split(",") if item.strip()]
    selected = list(pairs)
    random.Random(args.seed + 333).shuffle(selected)
    selected = selected[: min(args.robustness_pairs, len(selected))]
    rng = np.random.default_rng(args.seed + 333)
    rows = []

    with tempfile.TemporaryDirectory(prefix="robustness_quant_") as tmp:
        root = Path(tmp)
        get_features = _feature_cache(runtime, models, "real", root)
        for order, pair in enumerate(selected, start=1):
            left_features = get_features(pair.image1)
            right_features = get_features(pair.image2)
            baseline = _alignment(runtime, left_features, right_features, args)
            base_left, base_right = _region_intervals(runtime, models, baseline)
            line2 = runtime.dataset.display_image(pair.image2, "real")
            score_values = [float(baseline["normalized_score"])]
            path_lengths = [len(baseline["path"])]
            pair_rows = []

            for mode in modes:
                perturbed = _perturb(line2, mode, rng)
                path = root / f"perturb_{order:04d}_{mode}.png"
                Image.fromarray(perturbed).save(path)
                perturbed_features = runtime.utils.get_image_features(
                    models, path, "synthetic"
                )
                aligned = _alignment(
                    runtime, left_features, perturbed_features, args
                )
                pred_left, pred_right = _region_intervals(runtime, models, aligned)
                left_drift = (
                    abs(base_left[0] - pred_left[0]) + abs(base_left[1] - pred_left[1])
                ) / (2.0 * 1024.0)
                right_drift = (
                    abs(base_right[0] - pred_right[0]) + abs(base_right[1] - pred_right[1])
                ) / (2.0 * 1024.0)
                row = {
                    "query_order": order,
                    "pair_id": pair.pair_id,
                    "label_type": pair.label_type,
                    "perturbation": mode,
                    "endpoint_drift_line1": left_drift,
                    "endpoint_drift_line2": right_drift,
                    "mean_endpoint_drift": 0.5 * (left_drift + right_drift),
                    "line1_interval_iou_vs_baseline": _interval_iou(base_left, pred_left),
                    "line2_interval_iou_vs_baseline": _interval_iou(base_right, pred_right),
                    "mean_interval_iou_vs_baseline": 0.5
                    * (
                        _interval_iou(base_left, pred_left)
                        + _interval_iou(base_right, pred_right)
                    ),
                    "path_cell_iou_vs_baseline": _path_iou(
                        baseline["path"], aligned["path"]
                    ),
                    "baseline_normalized_sw_score": baseline["normalized_score"],
                    "perturbed_normalized_sw_score": aligned["normalized_score"],
                    "baseline_path_length": len(baseline["path"]),
                    "perturbed_path_length": len(aligned["path"]),
                    "image1": str(pair.image1),
                    "image2": str(pair.image2),
                }
                pair_rows.append(row)
                score_values.append(float(aligned["normalized_score"]))
                path_lengths.append(len(aligned["path"]))

            score_cv = _safe_cv(score_values)
            length_cv = _safe_cv(path_lengths)
            for row in pair_rows:
                row["sw_score_coefficient_of_variation"] = score_cv
                row["path_length_coefficient_of_variation"] = length_cv
                rows.append(row)

    _write_csv(output / "robustness.csv", rows)
    summary = {
        "diagnostic_only": True,
        "pairs": len(selected),
        "perturbations": modes,
        "mean_endpoint_drift": _mean(r["mean_endpoint_drift"] for r in rows),
        "mean_prediction_interval_iou": _mean(
            r["mean_interval_iou_vs_baseline"] for r in rows
        ),
        "mean_path_cell_iou": _mean(r["path_cell_iou_vs_baseline"] for r in rows),
        "mean_sw_score_coefficient_of_variation": _mean(
            r["sw_score_coefficient_of_variation"] for r in rows
        ),
        "mean_path_length_coefficient_of_variation": _mean(
            r["path_length_coefficient_of_variation"] for r in rows
        ),
        "warning": "Perturbation stability measures robustness, not absolute correctness.",
    }
    return rows, summary
