"""Runtime integration for bbox.json subword-box quantitative evaluation.

Besides loading bbox.json through the exact crop/resize geometry, this patch
teaches the real-box scorer about disconnected SW/NW component masks.  When the
evaluation row contains ``line*_component_intervals_px`` the metrics use the
UNION of those intervals and preserve holes; older single-span rows still use
the legacy fallback.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from Evaluation import real_subword_box_metrics as box_metrics
from Evaluation.real_subword_box_geometry import load_line_annotations


# The metric module resolves these globals at evaluation time. Replace its simple
# fallback with bbox.json loading plus the exact crop/resize/pad mapping.
box_metrics.load_line_annotations = load_line_annotations

_LEGACY_PREDICTED_INTERVAL = box_metrics._predicted_interval


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _merge_intervals(values) -> list[tuple[float, float]]:
    intervals = []
    for item in values or ():
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        try:
            left, right = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(left) and math.isfinite(right)):
            continue
        left, right = min(left, right), max(left, right)
        if right > left:
            intervals.append((left, right))
    intervals.sort()
    if not intervals:
        return []
    merged = [list(intervals[0])]
    for left, right in intervals[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return [(float(left), float(right)) for left, right in merged]


def _predicted_intervals(row: dict, prefix: str, width: int, use_flip: bool):
    """Prefer exact component intervals; fall back to the legacy outer span."""
    raw = row.get(f"{prefix}_component_intervals_px")
    parsed = None
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = None
    elif isinstance(raw, (list, tuple)):
        parsed = raw
    intervals = _merge_intervals(parsed)
    if intervals:
        return intervals

    legacy = _LEGACY_PREDICTED_INTERVAL(row, prefix, width, use_flip)
    return _merge_intervals([legacy] if legacy is not None else [])


def _box_union_coverage(box, intervals) -> float:
    if box.width <= 0:
        return 0.0
    overlap = sum(
        max(0.0, min(float(box.x1), right) - max(float(box.x0), left))
        for left, right in intervals
    )
    return min(1.0, float(overlap / box.width))


def _predicted_indices_union(boxes, intervals):
    rule = os.environ.get("REAL_BOX_IN_MASK_RULE", "center").strip().lower()
    minimum = max(0.0, min(1.0, _number("REAL_BOX_MIN_COVERAGE", 0.50)))
    predicted, coverages = set(), []
    for index, box in enumerate(boxes):
        coverage = _box_union_coverage(box, intervals)
        coverages.append(coverage)
        center_inside = any(left <= box.center_x <= right for left, right in intervals)
        inside = center_inside if rule == "center" else coverage >= minimum
        if rule in {"center_or_coverage", "either"}:
            inside = center_inside or coverage >= minimum
        if inside:
            predicted.add(index)
    return predicted, coverages


def _box_intervals(boxes, indices) -> list[tuple[float, float]]:
    return _merge_intervals(
        [(boxes[index].x0, boxes[index].x1) for index in sorted(indices)]
    )


def _interval_union_length(intervals) -> float:
    return float(sum(max(0.0, right - left) for left, right in intervals))


def _interval_union_iou(left, right):
    left = _merge_intervals(left)
    right = _merge_intervals(right)
    if not left or not right:
        return None
    intersection = sum(
        max(0.0, min(a1, b1) - max(a0, b0))
        for a0, a1 in left
        for b0, b1 in right
    )
    union = _interval_union_length(left) + _interval_union_length(right) - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _raster_union_metrics(boxes, gt_indices, intervals, width: int, height: int):
    if not gt_indices:
        return {"precision": None, "recall": None, "f1": None, "iou": None}
    gt = np.zeros((height, width), dtype=bool)
    for index in gt_indices:
        box = boxes[index]
        x0, x1 = int(math.floor(box.x0)), int(math.ceil(box.x1))
        y0, y1 = int(math.floor(box.y0)), int(math.ceil(box.y1))
        gt[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = True

    pred = np.zeros_like(gt)
    for left, right in intervals:
        x0, x1 = int(math.floor(left)), int(math.ceil(right))
        pred[:, max(0, x0):min(width, x1)] = True

    intersection = int(np.logical_and(gt, pred).sum())
    pred_area, gt_area = int(pred.sum()), int(gt.sum())
    union = int(np.logical_or(gt, pred).sum())
    precision = box_metrics._ratio(
        intersection, pred_area, 1.0 if gt_area == 0 else 0.0
    )
    recall = box_metrics._ratio(intersection, gt_area, 1.0)
    return {
        "precision": precision,
        "recall": recall,
        "f1": box_metrics._ratio(2 * precision * recall, precision + recall, 0.0),
        "iou": box_metrics._ratio(intersection, union, 1.0),
    }


def _line_metrics_union(
    prefix: str,
    annotations,
    gt_indices,
    intervals,
    width: int,
    height: int,
):
    intervals = _merge_intervals(intervals)
    boxes = annotations.boxes
    predicted, coverages = _predicted_indices_union(boxes, intervals)
    tp = len(predicted & gt_indices)
    fp = len(predicted - gt_indices)
    fn = len(gt_indices - predicted)
    tn = len(set(range(len(boxes))) - predicted - gt_indices)
    binary = box_metrics._binary_metrics(tp, fp, fn, tn)

    gt_intervals = _box_intervals(boxes, gt_indices)
    interval_iou = _interval_union_iou(intervals, gt_intervals)
    raster = _raster_union_metrics(boxes, gt_indices, intervals, width, height)

    pred_start = min((left for left, _ in intervals), default=None)
    pred_end = max((right for _, right in intervals), default=None)
    gt_start = min((left for left, _ in gt_intervals), default=None)
    gt_end = max((right for _, right in gt_intervals), default=None)

    return {
        f"{prefix}_box_annotation_path": annotations.workbook,
        f"{prefix}_box_annotation_sheet": annotations.sheet,
        f"{prefix}_box_annotation_status": annotations.status,
        f"{prefix}_box_annotation_error": annotations.error,
        f"{prefix}_box_count": len(boxes),
        f"{prefix}_shared_gt_boxes": len(gt_indices),
        f"{prefix}_predicted_mask_boxes": len(predicted),
        f"{prefix}_box_tp": tp,
        f"{prefix}_box_fp": fp,
        f"{prefix}_box_fn": fn,
        f"{prefix}_box_tn": tn,
        f"{prefix}_box_precision": binary["precision"],
        f"{prefix}_box_recall": binary["recall"],
        f"{prefix}_box_f1": binary["f1"],
        f"{prefix}_box_specificity": binary["specificity"],
        f"{prefix}_box_accuracy": binary["accuracy"],
        f"{prefix}_mean_box_mask_coverage": (
            float(np.mean(coverages)) if coverages else None
        ),
        f"{prefix}_shared_box_mask_coverage": (
            float(np.mean([coverages[index] for index in gt_indices]))
            if gt_indices else None
        ),
        f"{prefix}_box_interval_iou": interval_iou,
        f"{prefix}_box_pixel_precision": raster["precision"],
        f"{prefix}_box_pixel_recall": raster["recall"],
        f"{prefix}_box_pixel_f1": raster["f1"],
        f"{prefix}_box_pixel_iou": raster["iou"],
        # Keep scalar outer limits for backward-compatible CSV consumers, while
        # IoU/box/raster metrics above use the true disconnected union.
        f"{prefix}_pred_start_px": pred_start,
        f"{prefix}_pred_end_px": pred_end,
        f"{prefix}_gt_start_px": gt_start,
        f"{prefix}_gt_end_px": gt_end,
        f"{prefix}_region_iou": interval_iou,
        f"{prefix}_start_error_px": (
            None if pred_start is None or gt_start is None else abs(pred_start - gt_start)
        ),
        f"{prefix}_end_error_px": (
            None if pred_end is None or gt_end is None else abs(pred_end - gt_end)
        ),
    }


# metrics_from_evaluation_row resolves these names from the module at call time.
# Patching them here means both SW and any evaluator that opts into the box patch
# are scored with disconnected masks without changing the annotation loader.
box_metrics._predicted_interval = _predicted_intervals
box_metrics._line_metrics = _line_metrics_union


def install(sw_runner_module) -> None:
    if getattr(sw_runner_module, "_real_subword_box_patch_installed", False):
        return

    original_evaluate = sw_runner_module.evaluate_sample
    original_aggregate = sw_runner_module.aggregate
    original_fieldnames = sw_runner_module.batch_fieldnames

    def evaluate_sample(*args, **kwargs):
        row = original_evaluate(*args, **kwargs)
        dataset_type = kwargs.get("dataset_type")
        if dataset_type is None and len(args) >= 3:
            dataset_type = args[2]
        if str(dataset_type or row.get("dataset_type", "")).lower() != "real":
            return row

        models = kwargs.get("models") if "models" in kwargs else (args[0] if args else None)
        pair = kwargs.get("pair") if "pair" in kwargs else (args[1] if len(args) > 1 else None)
        if models is None or pair is None:
            return row

        try:
            metrics = box_metrics.metrics_from_evaluation_row(row, pair, models)
        except FileNotFoundError as exc:
            if _flag("REAL_REQUIRE_BOX_ANNOTATIONS", False):
                raise SystemExit(
                    "Strict bbox.json evaluation aborted on the first unresolved "
                    f"annotation: {exc}"
                ) from exc
            raise
        row.update(metrics)
        print(
            f"[{row.get('index')}] real-box status={row.get('real_box_status')} "
            f"shared={row.get('shared_subword_matches')} "
            f"precision={row.get('pair_box_precision')} "
            f"recall={row.get('pair_box_recall')} "
            f"f1={row.get('pair_box_f1')} "
            f"interval_iou={row.get('mean_box_interval_iou')}",
            flush=True,
        )
        return row

    def aggregate(rows):
        summary = original_aggregate(rows)
        summary.update(box_metrics.aggregate(rows))
        return summary

    def batch_fieldnames():
        names = list(original_fieldnames())
        for name in box_metrics.fieldnames():
            if name not in names:
                names.append(name)
        return names

    sw_runner_module.evaluate_sample = evaluate_sample
    sw_runner_module._evaluate_sample = evaluate_sample
    sw_runner_module.aggregate = aggregate
    sw_runner_module._aggregate = aggregate
    sw_runner_module.batch_fieldnames = batch_fieldnames
    sw_runner_module._batch_fieldnames = batch_fieldnames
    sw_runner_module._real_subword_box_patch_installed = True
