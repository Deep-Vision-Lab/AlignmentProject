#!/usr/bin/env python3
"""Aggregate the four true Point-2 representation evaluation runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median

MODES = ("local", "context", "fused", "fused_wrong_context")
METRICS = (
    "normalized_nw_score",
    "mean_path_cosine",
    "mean_mask_iou",
    "path_cosine_margin",
    "path_cosine_z",
    "component_count",
    "gap_steps",
)


def _number(value):
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _load_samples(path: Path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            key = str(row.get("pair_id") or row.get("index"))
            rows[key] = row
    return rows


def _aggregate(rows, metric):
    values = [_number(row.get(metric)) for row in rows.values()]
    values = [value for value in values if value is not None]
    if not values:
        return {"count": 0, "mean": None, "median": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "median": median(values),
    }


def _paired_delta(left, right, metric):
    shared = sorted(set(left) & set(right))
    deltas = []
    wins = ties = losses = 0
    for key in shared:
        a = _number(left[key].get(metric))
        b = _number(right[key].get(metric))
        if a is None or b is None:
            continue
        delta = a - b
        deltas.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    if not deltas:
        return {
            "count": 0,
            "mean_delta": None,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "win_rate": None,
        }
    return {
        "count": len(deltas),
        "mean_delta": sum(deltas) / len(deltas),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / len(deltas),
    }


def _fmt(value):
    return "n/a" if value is None else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--label", default="model")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    runs = {}
    selections = []
    for mode in MODES:
        directory = root / mode
        samples = directory / "samples.csv"
        selected = directory / "selected_pairs.json"
        if not samples.is_file():
            raise SystemExit(f"Missing Point-2 samples: {samples}")
        runs[mode] = _load_samples(samples)
        if selected.is_file():
            selections.append(selected.read_text(encoding="utf-8"))

    if selections and any(value != selections[0] for value in selections[1:]):
        raise SystemExit("Point-2 modes did not evaluate identical selected pairs")

    report = {
        "label": args.label,
        "root": str(root),
        "representations": {},
        "paired_comparisons": {},
    }
    for mode in MODES:
        report["representations"][mode] = {
            "successful_pairs": len(runs[mode]),
            "metrics": {
                metric: _aggregate(runs[mode], metric) for metric in METRICS
            },
        }

    for left, right in (
        ("fused", "local"),
        ("fused", "context"),
        ("fused", "fused_wrong_context"),
        ("context", "local"),
    ):
        report["paired_comparisons"][f"{left}_minus_{right}"] = {
            metric: _paired_delta(runs[left], runs[right], metric)
            for metric in (
                "normalized_nw_score",
                "mean_path_cosine",
                "mean_mask_iou",
                "path_cosine_margin",
                "path_cosine_z",
            )
        }

    json_path = root / "point2_representation_comparison.json"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = root / "point2_representation_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["representation", "metric", "count", "mean", "median"])
        for mode in MODES:
            for metric in METRICS:
                item = report["representations"][mode]["metrics"][metric]
                writer.writerow(
                    [mode, metric, item["count"], item["mean"], item["median"]]
                )

    print(f"\nPoint-2 representation comparison: {args.label}")
    print("=" * 94)
    print(
        f"{'representation':<22} {'NW':>10} {'path cos':>10} {'margin':>10} "
        f"{'path z':>10} {'mask IoU':>10} {'components':>12} {'pairs':>8}"
    )
    for mode in MODES:
        metrics = report["representations"][mode]["metrics"]
        print(
            f"{mode:<22} "
            f"{_fmt(metrics['normalized_nw_score']['mean']):>10} "
            f"{_fmt(metrics['mean_path_cosine']['mean']):>10} "
            f"{_fmt(metrics['path_cosine_margin']['mean']):>10} "
            f"{_fmt(metrics['path_cosine_z']['mean']):>10} "
            f"{_fmt(metrics['mean_mask_iou']['mean']):>10} "
            f"{_fmt(metrics['component_count']['mean']):>12} "
            f"{report['representations'][mode]['successful_pairs']:>8}"
        )

    print("\nKey paired comparisons")
    print("-" * 94)
    for name in (
        "fused_minus_local",
        "fused_minus_context",
        "fused_minus_fused_wrong_context",
    ):
        print(name)
        for metric in ("normalized_nw_score", "mean_path_cosine", "mean_mask_iou"):
            item = report["paired_comparisons"][name][metric]
            print(
                f"  {metric:<22} delta={_fmt(item['mean_delta'])} "
                f"wins/ties/losses={item['wins']}/{item['ties']}/{item['losses']} "
                f"win_rate={_fmt(item['win_rate'])}"
            )

    print(f"\nSaved: {json_path}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
