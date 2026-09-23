#!/usr/bin/env python3
"""Points 4-5: spatial correctness against external masks/intervals.

Consumes a completed fused Point-2/Yelda evaluation directory and checks whether
predicted aligned regions are actually localized in the source-image coordinate
system.

Point 4:
- compare prediction with GT masks/intervals;
- report explicit whole-line and empty-prediction controls.

Point 5:
- IoU, precision, recall, F1, center error;
- a GT region is counted as correctly matched only when one predicted interval
  has sufficient interval IoU AND at least N consecutive overlapping physical
  32-pixel windows at stride 16.

Internal NW/path scores are copied only for diagnosis and are not treated as
localization accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

WINDOW_WIDTH = 32
WINDOW_STRIDE = 16
WHOLE_LINE_THRESHOLD = 0.95


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    cols = np.any(mask > 0, axis=0)
    out: list[tuple[int, int]] = []
    start = None
    for x, on in enumerate(cols.tolist() + [False]):
        if on and start is None:
            start = x
        elif not on and start is not None:
            out.append((start, x))
            start = None
    return out


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    return arr > 0


def _interval_iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def _center(interval):
    return 0.5 * (float(interval[0]) + float(interval[1]))


def _best_center_errors(pred, gt, width):
    if not pred or not gt:
        return None, None
    errors = []
    for g in gt:
        p = max(pred, key=lambda item: _interval_iou(item, g))
        errors.append(abs(_center(p) - _center(g)))
    mean_px = float(np.mean(errors))
    return mean_px, mean_px / float(width)


def _column_labels(intervals, width):
    values = np.zeros(int(width), dtype=bool)
    for a, b in intervals:
        lo = max(0, min(int(width), int(math.floor(min(a, b)))))
        hi = max(0, min(int(width), int(math.ceil(max(a, b)))))
        if hi > lo:
            values[lo:hi] = True
    return values


def _window_labels(intervals, width, physical_windows=None):
    if physical_windows is not None:
        return np.asarray([any(max(a, x0) < min(b, x1) for x0, x1 in intervals)
                           for a, b in physical_windows], dtype=bool)
    n = max(1, 1 + max(0, int(width) - WINDOW_WIDTH) // WINDOW_STRIDE)
    labels = []
    for i in range(n):
        a = i * WINDOW_STRIDE
        b = min(width, a + WINDOW_WIDTH)
        labels.append(any(max(a, x0) < min(b, x1) for x0, x1 in intervals))
    return np.asarray(labels, dtype=bool)


def _binary_metrics(pred, gt, prefix=""):
    tp = int(np.sum(pred & gt))
    tn = int(np.sum(~pred & ~gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    p = f"{prefix}_" if prefix else ""
    return {
        f"{p}accuracy": accuracy,
        f"{p}precision": precision,
        f"{p}recall": recall,
        f"{p}f1": f1,
        f"{p}iou": iou,
        f"{p}dice": dice,
        f"{p}tp": tp,
        f"{p}tn": tn,
        f"{p}fp": fp,
        f"{p}fn": fn,
    }


def _max_consecutive_true(values):
    best = run = 0
    for value in values:
        if bool(value):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _region_matches(pred_intervals, gt_intervals, width, min_windows, min_iou, physical_windows=None):
    """Match every GT interval to its best predicted interval.

    The >=N-window rule is deliberately paired with interval IoU. Otherwise a
    trivial full-line prediction would satisfy every sufficiently wide GT region.
    """
    matches = []
    n_windows = max(1, 1 + max(0, int(width) - WINDOW_WIDTH) // WINDOW_STRIDE)

    for gt_index, gt in enumerate(gt_intervals):
        if pred_intervals:
            pred_index, pred = max(
                enumerate(pred_intervals),
                key=lambda item: _interval_iou(item[1], gt),
            )
            best_iou = _interval_iou(pred, gt)
        else:
            pred_index, pred, best_iou = None, None, 0.0

        overlap_windows = []
        windows = physical_windows if physical_windows is not None else [
            (i * WINDOW_STRIDE, min(width, i * WINDOW_STRIDE + WINDOW_WIDTH)) for i in range(n_windows)]
        for a, b in windows:
            overlaps_gt = max(a, gt[0]) < min(b, gt[1])
            overlaps_pred = (
                pred is not None and max(a, pred[0]) < min(b, pred[1])
            )
            overlap_windows.append(overlaps_gt and overlaps_pred)
        support = _max_consecutive_true(overlap_windows)
        pred_coverage = (
            max(0.0, min(width, pred[1]) - max(0.0, pred[0])) / float(width)
            if pred is not None and width > 0
            else 0.0
        )
        success = best_iou >= float(min_iou) and support >= int(min_windows)
        specific_success = success and pred_coverage < WHOLE_LINE_THRESHOLD
        matches.append(
            {
                "gt_index": gt_index,
                "pred_index": pred_index,
                "gt": gt,
                "pred": pred,
                "interval_iou": best_iou,
                "consecutive_window_support": support,
                "predicted_interval_coverage": pred_coverage,
                "success": bool(success),
                "specific_success": bool(specific_success),
            }
        )
    return matches


def source_window_intervals(geometry, window_size, stride):
    """Actual physical model windows mapped into source space, never source 32/16."""
    from Evaluation.yelda_geometry import source_intervals
    width = int(geometry["canvas_width"])
    result = []
    for x in range(0, width - int(window_size) + 1, int(stride)):
        mapped = source_intervals([[x, x + int(window_size)]], geometry)
        result.append(tuple(mapped[0]) if mapped else (0.0, 0.0))
    return result


def score_source_regions(pred, gt_path, geometry, window_size, stride, min_windows=5, min_iou=0.5):
    gt_mask = _load_mask(Path(gt_path))
    width = int(geometry["source_width"])
    if gt_mask.shape != (int(geometry["source_height"]), width):
        raise ValueError("GT mask must use original source-image coordinates")
    gt = _runs(gt_mask)
    metrics = _binary_metrics(_column_labels(pred, width), np.any(gt_mask, axis=0))
    metrics["center_error_px"] = _best_center_errors(pred, gt, width)[0]
    matches = _region_matches(pred, gt, width, min_windows, min_iou,
                              physical_windows=source_window_intervals(geometry, window_size, stride))
    metrics["region_success_rate"] = _mean(m["success"] for m in matches)
    return metrics


def _coverage(intervals, width):
    if not intervals or width <= 0:
        return 0.0
    merged = []
    for a, b in sorted(
        (max(0.0, float(a)), min(float(width), float(b))) for a, b in intervals
    ):
        if b <= a:
            continue
        if not merged or a > merged[-1][1]:
            merged.append([a, b])
        else:
            merged[-1][1] = max(merged[-1][1], b)
    covered = sum(b - a for a, b in merged)
    return covered / float(width)


def _mean(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-consecutive-windows", type=int, default=5)
    ap.add_argument("--min-region-iou", type=float, default=0.50)
    args = ap.parse_args()
    if args.min_consecutive_windows < 1:
        raise SystemExit("--min-consecutive-windows must be >= 1")
    if not 0.0 <= args.min_region_iou <= 1.0:
        raise SystemExit("--min-region-iou must be in [0,1]")

    root = Path(args.eval_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_metadata_path = root / "run.json"
    run_metadata = json.loads(run_metadata_path.read_text()) if run_metadata_path.is_file() else {}
    recorded_geometry = run_metadata.get("evaluation_contract", run_metadata.get("model_config", {}))

    rows = []
    region_rows = []
    for pair_dir in sorted(root.glob("pair_*")):
        summary_path = pair_dir / "summary.json"
        if not summary_path.is_file():
            continue
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "index": s.get("index"),
            "pair_id": s.get("pair_id"),
            "normalized_nw_score_diagnostic": s.get("normalized_nw_score"),
            "mean_path_cosine_diagnostic": s.get("mean_path_cosine"),
            "path_cosine_margin_diagnostic": s.get("path_cosine_margin"),
            "path_cosine_z_diagnostic": s.get("path_cosine_z"),
            "mean_mask_iou_existing": s.get("mean_mask_iou"),
        }

        all_region_matches = []
        for side in (1, 2):
            geometry = s["geometry"][side - 1]
            width = int(geometry["source_width"])
            pred = [
                tuple(map(float, p))
                for p in s.get(f"line{side}_source_intervals_px", [])
            ]
            gt_mask_path = pair_dir / f"line{side}_source_gt_mask.png"
            if not gt_mask_path.is_file():
                row[f"line{side}_gt_regions"] = None
                continue

            gt_mask = _load_mask(gt_mask_path)
            gt = _runs(gt_mask)
            center_px, center_norm = _best_center_errors(pred, gt, width)

            pred_columns = _column_labels(pred, width)
            gt_columns = np.any(gt_mask, axis=0)
            col = _binary_metrics(pred_columns, gt_columns, "column")

            if "window_size" not in recorded_geometry or "stride" not in recorded_geometry:
                raise ValueError("Source-space support requires checkpoint window_size/stride in run.json; no implicit source 32/16 grid")
            physical_windows = source_window_intervals(geometry, recorded_geometry["window_size"], recorded_geometry["stride"])
            pred_windows = _window_labels(pred, width, physical_windows)
            gt_windows = _window_labels(gt, width, physical_windows)
            win = _binary_metrics(pred_windows, gt_windows, "window")

            whole = _binary_metrics(
                _column_labels([(0.0, float(width))], width),
                gt_columns,
                "whole_line_control",
            )
            empty = _binary_metrics(
                _column_labels([], width),
                gt_columns,
                "empty_control",
            )

            matches = _region_matches(
                pred,
                gt,
                width,
                args.min_consecutive_windows,
                args.min_region_iou,
                physical_windows=physical_windows,
            )
            all_region_matches.extend(matches)
            for match in matches:
                region_rows.append(
                    {
                        "index": s.get("index"),
                        "pair_id": s.get("pair_id"),
                        "side": side,
                        "gt_region_index": match["gt_index"],
                        "pred_region_index": match["pred_index"],
                        "gt_start_px": match["gt"][0],
                        "gt_end_px": match["gt"][1],
                        "pred_start_px": (
                            match["pred"][0] if match["pred"] is not None else None
                        ),
                        "pred_end_px": (
                            match["pred"][1] if match["pred"] is not None else None
                        ),
                        "interval_iou": match["interval_iou"],
                        "consecutive_window_support": match[
                            "consecutive_window_support"
                        ],
                        "predicted_interval_coverage": match[
                            "predicted_interval_coverage"
                        ],
                        "success_iou_and_support": int(match["success"]),
                        "specific_success_not_whole_line": int(
                            match["specific_success"]
                        ),
                    }
                )

            pred_coverage = _coverage(pred, width)
            gt_coverage = _coverage(gt, width)
            row.update(
                {
                    f"line{side}_pred_regions": len(pred),
                    f"line{side}_gt_regions": len(gt),
                    f"line{side}_center_error_px": center_px,
                    f"line{side}_center_error_norm": center_norm,
                    f"line{side}_correct_regions": sum(
                        int(item["success"]) for item in matches
                    ),
                    f"line{side}_specific_correct_regions": sum(
                        int(item["specific_success"]) for item in matches
                    ),
                    f"line{side}_region_success_rate": (
                        _mean(item["success"] for item in matches)
                    ),
                    f"line{side}_specific_region_success_rate": (
                        _mean(item["specific_success"] for item in matches)
                    ),
                    f"line{side}_mean_best_region_iou": _mean(
                        item["interval_iou"] for item in matches
                    ),
                    f"line{side}_max_consecutive_support": (
                        max(
                            (
                                item["consecutive_window_support"]
                                for item in matches
                            ),
                            default=0,
                        )
                    ),
                    f"line{side}_pred_coverage": pred_coverage,
                    f"line{side}_gt_coverage": gt_coverage,
                    f"line{side}_whole_line_prediction": (
                        pred_coverage >= WHOLE_LINE_THRESHOLD
                    ),
                }
            )
            row.update({f"line{side}_{k}": v for k, v in col.items()})
            row.update({f"line{side}_{k}": v for k, v in win.items()})
            row.update({f"line{side}_{k}": v for k, v in whole.items()})
            row.update({f"line{side}_{k}": v for k, v in empty.items()})

        row["region_success_rate"] = _mean(
            item["success"] for item in all_region_matches
        )
        row["specific_region_success_rate"] = _mean(
            item["specific_success"] for item in all_region_matches
        )
        rows.append(row)

    if not rows:
        raise SystemExit(f"No pair summaries found under {root}")

    fields = sorted({k for row in rows for k in row})
    with (out / "point45_pair_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if region_rows:
        region_fields = sorted({k for row in region_rows for k in row})
        with (out / "point45_region_metrics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=region_fields)
            writer.writeheader()
            writer.writerows(region_rows)

    whole_flags = [
        bool(r.get(f"line{s}_whole_line_prediction"))
        for r in rows
        for s in (1, 2)
        if r.get(f"line{s}_whole_line_prediction") is not None
    ]
    summary = {
        "points": [4, 5],
        "pairs": len(rows),
        "regions": len(region_rows),
        "window_width": WINDOW_WIDTH,
        "window_stride": WINDOW_STRIDE,
        "min_consecutive_windows": args.min_consecutive_windows,
        "min_region_iou": args.min_region_iou,
        "whole_line_threshold": WHOLE_LINE_THRESHOLD,
        "mean_column_iou": _mean(
            r.get(f"line{s}_column_iou") for r in rows for s in (1, 2)
        ),
        "mean_column_precision": _mean(
            r.get(f"line{s}_column_precision") for r in rows for s in (1, 2)
        ),
        "mean_column_recall": _mean(
            r.get(f"line{s}_column_recall") for r in rows for s in (1, 2)
        ),
        "mean_column_f1": _mean(
            r.get(f"line{s}_column_f1") for r in rows for s in (1, 2)
        ),
        "mean_window_precision": _mean(
            r.get(f"line{s}_window_precision") for r in rows for s in (1, 2)
        ),
        "mean_window_recall": _mean(
            r.get(f"line{s}_window_recall") for r in rows for s in (1, 2)
        ),
        "mean_window_f1": _mean(
            r.get(f"line{s}_window_f1") for r in rows for s in (1, 2)
        ),
        "mean_center_error_px": _mean(
            r.get(f"line{s}_center_error_px") for r in rows for s in (1, 2)
        ),
        "mean_center_error_norm": _mean(
            r.get(f"line{s}_center_error_norm") for r in rows for s in (1, 2)
        ),
        "mean_region_success_rate": _mean(
            r.get("region_success_rate") for r in rows
        ),
        "mean_specific_region_success_rate": _mean(
            r.get("specific_region_success_rate") for r in rows
        ),
        "mean_best_region_iou": _mean(
            row.get("interval_iou") for row in region_rows
        ),
        "mean_predicted_line_coverage": _mean(
            r.get(f"line{s}_pred_coverage") for r in rows for s in (1, 2)
        ),
        "mean_gt_line_coverage": _mean(
            r.get(f"line{s}_gt_coverage") for r in rows for s in (1, 2)
        ),
        "whole_line_prediction_rate": (
            float(np.mean(whole_flags)) if whole_flags else None
        ),
        "whole_line_control_mean_iou": _mean(
            r.get(f"line{s}_whole_line_control_iou")
            for r in rows
            for s in (1, 2)
        ),
        "whole_line_control_mean_precision": _mean(
            r.get(f"line{s}_whole_line_control_precision")
            for r in rows
            for s in (1, 2)
        ),
        "whole_line_control_mean_recall": _mean(
            r.get(f"line{s}_whole_line_control_recall")
            for r in rows
            for s in (1, 2)
        ),
        "empty_control_mean_iou": _mean(
            r.get(f"line{s}_empty_control_iou")
            for r in rows
            for s in (1, 2)
        ),
        "note": (
            "A region success requires BOTH interval IoU >= min_region_iou and "
            ">= min_consecutive_windows consecutive overlapping 32px windows. "
            "specific_region_success additionally rejects >=95%-of-line predicted "
            "intervals. Whole-line/empty controls expose trivial baselines."
        ),
    }
    (out / "point45_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )

    # Backward-compatible aliases for the earlier Point-3 metrics filenames.
    (out / "point3_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (out / "point3_pair_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
