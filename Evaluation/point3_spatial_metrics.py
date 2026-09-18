#!/usr/bin/env python3
"""Point-3 spatial alignment validation.

Consumes a completed fused Point-2/Yelda evaluation directory and measures
whether predicted aligned regions are spatially correct in source-image space.

Metrics:
- existing source-mask IoU / Dice / precision / recall
- GT/predicted interval center error in pixels and normalized by source width
- explicit 128x32, stride-16 window classification accuracy / precision / recall / F1
- GT-region success requiring >= N consecutive predicted-positive windows

The script never changes model outputs.  It only audits spatial correctness.
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


def _iou_1d(a: tuple[float, float], b: tuple[float, float]) -> float:
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
        p = max(pred, key=lambda item: _iou_1d(item, g))
        errors.append(abs(_center(p) - _center(g)))
    mean_px = float(np.mean(errors))
    return mean_px, mean_px / float(width)


def _window_labels(intervals, width):
    n = max(1, 1 + max(0, int(width) - WINDOW_WIDTH) // WINDOW_STRIDE)
    labels = []
    for i in range(n):
        a = i * WINDOW_STRIDE
        b = min(width, a + WINDOW_WIDTH)
        labels.append(any(max(a, x0) < min(b, x1) for x0, x1 in intervals))
    return np.asarray(labels, dtype=bool)


def _binary_metrics(pred, gt):
    tp = int(np.sum(pred & gt))
    tn = int(np.sum(~pred & ~gt))
    fp = int(np.sum(pred & ~gt))
    fn = int(np.sum(~pred & gt))
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall > 0
        else None
    )
    return dict(window_accuracy=acc, window_precision=precision,
                window_recall=recall, window_f1=f1, window_tp=tp,
                window_tn=tn, window_fp=fp, window_fn=fn)


def _max_consecutive_true(values):
    best = run = 0
    for value in values:
        if bool(value):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _region_success(pred_labels, gt_intervals, width, min_windows):
    successes = []
    supports = []
    n = len(pred_labels)
    for x0, x1 in gt_intervals:
        relevant = []
        for i in range(n):
            a = i * WINDOW_STRIDE
            b = min(width, a + WINDOW_WIDTH)
            relevant.append(max(a, x0) < min(b, x1))
        relevant = np.asarray(relevant, dtype=bool)
        support = _max_consecutive_true(pred_labels & relevant)
        supports.append(int(support))
        successes.append(support >= min_windows)
    return successes, supports


def _mean(values):
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(vals)) if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--min-consecutive-windows", type=int, default=5)
    args = ap.parse_args()

    root = Path(args.eval_root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for pair_dir in sorted(root.glob("pair_*")):
        summary_path = pair_dir / "summary.json"
        if not summary_path.is_file():
            continue
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        row = {
            "index": s.get("index"),
            "pair_id": s.get("pair_id"),
            "normalized_nw_score": s.get("normalized_nw_score"),
            "mean_path_cosine": s.get("mean_path_cosine"),
            "path_cosine_margin": s.get("path_cosine_margin"),
            "path_cosine_z": s.get("path_cosine_z"),
            "mean_mask_iou": s.get("mean_mask_iou"),
        }

        side_successes = []
        for side in (1, 2):
            geometry = s["geometry"][side - 1]
            width = int(geometry["source_width"])
            pred = [tuple(map(float, p)) for p in s.get(f"line{side}_source_intervals_px", [])]
            gt_mask_path = pair_dir / f"line{side}_source_gt_mask.png"
            if not gt_mask_path.is_file():
                row[f"line{side}_gt_regions"] = None
                continue

            gt = _runs(_load_mask(gt_mask_path))
            center_px, center_norm = _best_center_errors(pred, gt, width)
            pred_labels = _window_labels(pred, width)
            gt_labels = _window_labels(gt, width)
            wm = _binary_metrics(pred_labels, gt_labels)
            successes, supports = _region_success(
                pred_labels, gt, width, args.min_consecutive_windows
            )

            row.update({
                f"line{side}_pred_regions": len(pred),
                f"line{side}_gt_regions": len(gt),
                f"line{side}_center_error_px": center_px,
                f"line{side}_center_error_norm": center_norm,
                f"line{side}_region_success_rate": (
                    sum(successes) / len(successes) if successes else None
                ),
                f"line{side}_max_consecutive_support": (
                    max(supports) if supports else 0
                ),
                f"line{side}_mask_iou": s.get(f"line{side}_mask_iou"),
            })
            row.update({f"line{side}_{k}": v for k, v in wm.items()})
            side_successes.extend(successes)

        row["word_level_success_rate"] = (
            sum(side_successes) / len(side_successes) if side_successes else None
        )
        rows.append(row)

    if not rows:
        raise SystemExit(f"No pair summaries found under {root}")

    fields = sorted({k for row in rows for k in row})
    with (out / "point3_pair_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    summary = {
        "pairs": len(rows),
        "window_width": WINDOW_WIDTH,
        "window_stride": WINDOW_STRIDE,
        "min_consecutive_windows": args.min_consecutive_windows,
        "mean_mask_iou": _mean(r.get("mean_mask_iou") for r in rows),
        "mean_center_error_px": _mean(
            r.get(f"line{s}_center_error_px") for r in rows for s in (1, 2)
        ),
        "mean_center_error_norm": _mean(
            r.get(f"line{s}_center_error_norm") for r in rows for s in (1, 2)
        ),
        "mean_window_accuracy": _mean(
            r.get(f"line{s}_window_accuracy") for r in rows for s in (1, 2)
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
        "mean_word_level_success_rate": _mean(
            r.get("word_level_success_rate") for r in rows
        ),
        "whole_line_prediction_rate": float(np.mean([
            any(
                len(r.get(f"line{s}_source_intervals_px", [])) == 1
                and r.get(f"line{s}_source_intervals_px", [[1, 0]])[0][0] <= 0
                for s in (1, 2)
            )
            for r in []
        ])) if False else None,
        "note": (
            "Center/window/word-support metrics are intended to catch broad whole-line "
            "predictions that can obtain nontrivial mask IoU without precise localization."
        ),
    }
    (out / "point3_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
