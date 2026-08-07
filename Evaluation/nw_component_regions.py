"""Component-aware supported-region extraction for Needleman-Wunsch evaluation.

The global Needleman-Wunsch DP/traceback is unchanged.  This module interprets
that fixed path as up to three connected aligned components, matching the
synthetic generator's 1/2/3-shared-region design.  Ground-truth masks are used
only for reporting metrics, never for selecting predicted components.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from Evaluation import nw_discontinuous_regions as base


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _mutual_stats(matrix: np.ndarray, row: int, col: int) -> tuple[float, float]:
    """Return mutual z-score and mutual percentile for one candidate cell."""
    value = float(matrix[row, col])
    row_values = np.asarray(matrix[row], dtype=np.float32)
    col_values = np.asarray(matrix[:, col], dtype=np.float32)

    row_std = max(float(np.std(row_values)), 1e-4)
    col_std = max(float(np.std(col_values)), 1e-4)
    row_z = (value - float(np.mean(row_values))) / row_std
    col_z = (value - float(np.mean(col_values))) / col_std

    row_pct = float(np.mean(row_values <= value))
    col_pct = float(np.mean(col_values <= value))
    return float(min(row_z, col_z)), float(min(row_pct, col_pct))


def _diagonal_records(steps, matrix: np.ndarray):
    records = []
    for position, step in enumerate(steps):
        if step.index1 is None or step.index2 is None:
            continue
        row, col = int(step.index1), int(step.index2)
        score = float(matrix[row, col])
        mutual_z, mutual_pct = _mutual_stats(matrix, row, col)
        records.append(
            {
                "position": int(position),
                "row": row,
                "col": col,
                "score": score,
                "mutual_z": mutual_z,
                "mutual_pct": mutual_pct,
            }
        )
    return records


def _can_join(left, right, max_path_gap: int, max_window_gap: int) -> bool:
    path_gap = int(right["position"]) - int(left["position"]) - 1
    row_gap = int(right["row"]) - int(left["row"]) - 1
    col_gap = int(right["col"]) - int(left["col"]) - 1
    return (
        path_gap <= max_path_gap
        and row_gap <= max_window_gap
        and col_gap <= max_window_gap
    )


def _component_record(group, seeds) -> dict:
    positions = [int(item["position"]) for item in group]
    scores = np.asarray([float(item["score"]) for item in group], dtype=np.float32)
    z_values = np.asarray([float(item["mutual_z"]) for item in group], dtype=np.float32)
    percentiles = np.asarray(
        [float(item["mutual_pct"]) for item in group], dtype=np.float32
    )
    rows = [int(item["row"]) for item in group]
    cols = [int(item["col"]) for item in group]
    seed_positions = {int(item["position"]) for item in seeds}
    seed_count = sum(position in seed_positions for position in positions)
    step_span = max(1, max(positions) - min(positions) + 1)
    row_span = max(rows) - min(rows) + 1
    col_span = max(cols) - min(cols) + 1
    balance = min(row_span, col_span) / max(1, max(row_span, col_span))
    positive_sum = float(np.maximum(scores, 0.0).sum())
    density = float(len(group) / step_span)
    mean_score = float(scores.mean())
    mean_z = float(z_values.mean())
    mean_pct = float(percentiles.mean())
    quality = positive_sum * density * mean_pct * max(0.25, min(2.0, mean_z + 0.5))
    return {
        "items": list(group),
        "positions": positions,
        "path": [(int(item["row"]), int(item["col"])) for item in group],
        "traceback_steps": int(step_span),
        "score": positive_sum,
        "mean_match_score": mean_score,
        "mean_mutual_z": mean_z,
        "mean_mutual_pct": mean_pct,
        "support_density": density,
        "seed_count": int(seed_count),
        "row_span": int(row_span),
        "col_span": int(col_span),
        "span_balance": float(balance),
        "quality": float(quality),
        "step_start": int(min(positions)),
        "step_end": int(max(positions) + 1),
    }


def _merge_component_records(left, right, seeds):
    merged = sorted(
        {int(item["position"]): item for item in (*left["items"], *right["items"])}.values(),
        key=lambda item: int(item["position"]),
    )
    return _component_record(merged, seeds)


def _supported_runs(steps, match_scores: np.ndarray, gap_penalty: float):
    """Find up to three connected, distinctive components on the full NW path."""
    del gap_penalty
    if not steps:
        return [], 0.0

    matrix = np.asarray(match_scores, dtype=np.float32)
    records = _diagonal_records(steps, matrix)
    if not records:
        return [], 0.0

    seed_score = _env_float("NW_COMPONENT_SEED_SCORE", 0.22)
    seed_z = _env_float("NW_COMPONENT_SEED_MUTUAL_Z", 0.25)
    seed_pct = _env_float("NW_COMPONENT_SEED_PERCENTILE", 0.82)
    support_score = _env_float("NW_COMPONENT_SUPPORT_SCORE", 0.04)
    support_z = _env_float("NW_COMPONENT_SUPPORT_MUTUAL_Z", -0.10)
    support_pct = _env_float("NW_COMPONENT_SUPPORT_PERCENTILE", 0.62)

    max_path_gap = max(0, _env_int("NW_COMPONENT_MAX_PATH_GAP", 3))
    max_window_gap = max(0, _env_int("NW_COMPONENT_MAX_WINDOW_GAP", 2))
    merge_path_gap = max(0, _env_int("NW_COMPONENT_MERGE_PATH_GAP", 3))
    merge_window_gap = max(0, _env_int("NW_COMPONENT_MERGE_WINDOW_GAP", 2))

    min_matches = max(1, _env_int("NW_COMPONENT_MIN_MATCHES", 4))
    min_seeds = max(1, _env_int("NW_COMPONENT_MIN_SEEDS", 2))
    min_mean_score = _env_float("NW_COMPONENT_MIN_MEAN_SCORE", 0.12)
    min_mean_z = _env_float("NW_COMPONENT_MIN_MEAN_MUTUAL_Z", 0.10)
    min_mean_pct = _env_float("NW_COMPONENT_MIN_MEAN_PERCENTILE", 0.72)
    min_density = _env_float("NW_COMPONENT_MIN_DENSITY", 0.50)
    min_balance = _env_float("NW_COMPONENT_MIN_SPAN_BALANCE", 0.55)
    min_quality = _env_float("NW_COMPONENT_MIN_QUALITY", 1.25)
    min_relative_quality = _env_float("NW_COMPONENT_MIN_RELATIVE_QUALITY", 0.35)
    max_components = max(1, _env_int("NW_COMPONENT_MAX_COMPONENTS", 3))

    seeds = [
        item
        for item in records
        if float(item["score"]) >= seed_score
        and float(item["mutual_z"]) >= seed_z
        and float(item["mutual_pct"]) >= seed_pct
    ]
    if not seeds:
        return [], 0.0

    support = [
        item
        for item in records
        if float(item["score"]) >= support_score
        and float(item["mutual_z"]) >= support_z
        and float(item["mutual_pct"]) >= support_pct
    ]
    if not support:
        return [], 0.0

    # Group low-threshold support into connected path components.  Using the
    # entire NW traceback is essential because the generator can contain 2-3
    # separated shared phrases with real unaligned text between them.
    groups = []
    group = [support[0]]
    for item in support[1:]:
        if _can_join(group[-1], item, max_path_gap, max_window_gap):
            group.append(item)
        else:
            groups.append(group)
            group = [item]
    groups.append(group)

    components = [_component_record(group, seeds) for group in groups]
    components = [
        component
        for component in components
        if len(component["path"]) >= min_matches
        and int(component["seed_count"]) >= min_seeds
        and float(component["mean_match_score"]) >= min_mean_score
        and float(component["mean_mutual_z"]) >= min_mean_z
        and float(component["mean_mutual_pct"]) >= min_mean_pct
        and float(component["support_density"]) >= min_density
        and float(component["span_balance"]) >= min_balance
        and float(component["quality"]) >= min_quality
    ]
    if not components:
        return [], 0.0

    # Morphological closing at component level: neighboring confident pieces
    # are one aligned phrase even when one or two noisy windows disappear.
    components.sort(key=lambda component: int(component["step_start"]))
    merged = [components[0]]
    for component in components[1:]:
        previous = merged[-1]
        left = previous["items"][-1]
        right = component["items"][0]
        if _can_join(left, right, merge_path_gap, merge_window_gap):
            merged[-1] = _merge_component_records(previous, component, seeds)
        else:
            merged.append(component)

    strongest = max(float(component["quality"]) for component in merged)
    relative_floor = strongest * max(0.0, min(1.0, min_relative_quality))
    credible = [
        component
        for component in merged
        if float(component["quality"]) >= max(min_quality, relative_floor)
    ]

    # The generator has at most three true shared regions.  Keep the strongest
    # credible components only, then restore reading/path order for rendering.
    credible.sort(key=lambda component: float(component["quality"]), reverse=True)
    credible = credible[:max_components]
    credible.sort(key=lambda component: int(component["step_start"]))

    total_score = float(sum(float(component["score"]) for component in credible))
    return credible, total_score


def _mask_intervals(mask_path: Path | None) -> list[tuple[float, float]]:
    """Return every connected horizontal component in a synthetic mask."""
    if mask_path is None or not Path(mask_path).is_file():
        return []
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened.convert("L"))
    columns = np.any(mask > 0, axis=0)
    indices = np.flatnonzero(columns)
    if not len(indices):
        return []
    intervals = []
    start = previous = int(indices[0])
    for value in map(int, indices[1:]):
        if value != previous + 1:
            intervals.append((float(start), float(previous + 1)))
            start = value
        previous = value
    intervals.append((float(start), float(previous + 1)))
    return intervals


def _union_length(intervals) -> float:
    return float(sum(max(0.0, right - left) for left, right in intervals))


def _union_intersection(left, right) -> float:
    return float(
        sum(
            max(0.0, min(a1, b1) - max(a0, b0))
            for a0, a1 in left
            for b0, b1 in right
        )
    )


def _union_iou(predicted, target):
    predicted = base._merge_intervals(predicted)
    target = base._merge_intervals(target)
    if not predicted or not target:
        return None
    intersection = _union_intersection(predicted, target)
    union = _union_length(predicted) + _union_length(target) - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def synthetic_mask_region_metrics(
    path,
    traceback,
    similarity_shape,
    image1,
    image2,
    image_width1: int,
    image_width2: int,
    use_flip: bool,
) -> dict:
    """Compare predicted component unions with the true multi-component masks."""
    del traceback
    keys = {}
    for prefix in ("line1", "line2"):
        keys.update(
            {
                f"{prefix}_pred_start_px": None,
                f"{prefix}_pred_end_px": None,
                f"{prefix}_gt_start_px": None,
                f"{prefix}_gt_end_px": None,
                f"{prefix}_region_iou": None,
                f"{prefix}_start_error_px": None,
                f"{prefix}_end_error_px": None,
            }
        )
    keys["mean_region_iou"] = None

    runs = base._path_runs(path)
    if not runs:
        return keys

    n1, n2 = map(int, similarity_shape)
    specifications = (
        ("line1", 0, n1, int(image_width1), Path(image1)),
        ("line2", 1, n2, int(image_width2), Path(image2)),
    )
    ious = []
    for prefix, axis, n_windows, width, image_path in specifications:
        predicted = []
        for run in runs:
            indices = [int(pair[axis]) for pair in run]
            if not indices:
                continue
            left, right = base.patch_range_to_pixels(
                min(indices), max(indices) + 1, n_windows, width, use_flip
            )
            predicted.append((min(left, right), max(left, right)))
        predicted = base._merge_intervals(predicted)
        if not predicted:
            continue

        target = _mask_intervals(base.sw_core._synthetic_mask_path(image_path))
        pred_start = min(left for left, _ in predicted)
        pred_end = max(right for _, right in predicted)
        keys[f"{prefix}_pred_start_px"] = float(pred_start)
        keys[f"{prefix}_pred_end_px"] = float(pred_end)
        if not target:
            continue

        gt_start = min(left for left, _ in target)
        gt_end = max(right for _, right in target)
        iou = _union_iou(predicted, target)
        keys[f"{prefix}_gt_start_px"] = int(round(gt_start))
        keys[f"{prefix}_gt_end_px"] = int(round(gt_end))
        keys[f"{prefix}_region_iou"] = iou
        keys[f"{prefix}_start_error_px"] = abs(float(pred_start) - gt_start)
        keys[f"{prefix}_end_error_px"] = abs(float(pred_end) - gt_end)
        if iou is not None:
            ious.append(float(iou))

    if ious:
        keys["mean_region_iou"] = float(np.mean(ious))
    return keys


def install(runner) -> None:
    """Install component-aware region interpretation; global NW DP is unchanged."""
    base._supported_runs = _supported_runs
    base.synthetic_mask_region_metrics = synthetic_mask_region_metrics
    base.install(runner)
