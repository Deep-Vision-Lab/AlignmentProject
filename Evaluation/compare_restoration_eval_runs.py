#!/usr/bin/env python3
"""Compare local/primary/joint restoration-branch Yelda evaluation runs.

Each run must contain summary.json and samples.csv produced by Evaluation.eval_yelda.
The script reports aggregate metrics plus paired per-sample win rates so we can
answer the architectural question: did the semantic DTW adapter improve
cross-image alignment over the primitive restoration representation?
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median


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
        deltas.append((delta, key, a, b))
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
            "best": [],
            "worst": [],
        }
    ordered = sorted(deltas)
    count = len(deltas)
    return {
        "count": count,
        "mean_delta": sum(item[0] for item in deltas) / count,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / count,
        "best": [
            {"pair_id": key, "delta": delta, "left": a, "right": b}
            for delta, key, a, b in reversed(ordered[-5:])
        ],
        "worst": [
            {"pair_id": key, "delta": delta, "left": a, "right": b}
            for delta, key, a, b in ordered[:5]
        ],
    }


def _fmt(value):
    return "n/a" if value is None else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    runs = {}
    for name in ("local", "primary", "joint"):
        directory = root / name
        samples_path = directory / "samples.csv"
        summary_path = directory / "summary.json"
        if not samples_path.is_file() or not summary_path.is_file():
            raise SystemExit(
                f"Missing {name} evaluation outputs under {directory}; "
                "expected samples.csv and summary.json"
            )
        runs[name] = {
            "directory": directory,
            "samples": _load_samples(samples_path),
            "summary": json.loads(summary_path.read_text(encoding="utf-8")),
        }

    report = {
        "root": str(root),
        "representations": {},
        "paired_comparisons": {},
    }
    for name, run in runs.items():
        report["representations"][name] = {
            "successful_pairs": len(run["samples"]),
            "metrics": {
                metric: _aggregate(run["samples"], metric)
                for metric in METRICS
            },
        }

    for left, right in (
        ("primary", "local"),
        ("joint", "local"),
        ("primary", "joint"),
    ):
        report["paired_comparisons"][f"{left}_minus_{right}"] = {
            metric: _paired_delta(
                runs[left]["samples"], runs[right]["samples"], metric
            )
            for metric in (
                "normalized_nw_score",
                "mean_path_cosine",
                "mean_mask_iou",
                "path_cosine_margin",
                "path_cosine_z",
            )
        }

    (root / "representation_comparison.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (root / "representation_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["representation", "metric", "count", "mean", "median"]
        )
        for name in ("local", "primary", "joint"):
            for metric in METRICS:
                item = report["representations"][name]["metrics"][metric]
                writer.writerow(
                    [name, metric, item["count"], item["mean"], item["median"]]
                )

    print("\nRestoration branch representation comparison")
    print("=" * 76)
    print(
        f"{'representation':<14} {'NW':>10} {'path cos':>10} "
        f"{'margin':>10} {'path z':>10} {'mask IoU':>10} "
        f"{'components':>12} {'pairs':>8}"
    )
    for name in ("local", "primary", "joint"):
        metrics = report["representations"][name]["metrics"]
        print(
            f"{name:<14} "
            f"{_fmt(metrics['normalized_nw_score']['mean']):>10} "
            f"{_fmt(metrics['mean_path_cosine']['mean']):>10} "
            f"{_fmt(metrics['path_cosine_margin']['mean']):>10} "
            f"{_fmt(metrics['path_cosine_z']['mean']):>10} "
            f"{_fmt(metrics['mean_mask_iou']['mean']):>10} "
            f"{_fmt(metrics['component_count']['mean']):>12} "
            f"{report['representations'][name]['successful_pairs']:>8}"
        )

    print("\nPrimary semantic L_i versus primitive P_i")
    print("-" * 76)
    comparison = report["paired_comparisons"]["primary_minus_local"]
    for metric in (
        "normalized_nw_score",
        "mean_path_cosine",
        "path_cosine_margin",
        "path_cosine_z",
        "mean_mask_iou",
    ):
        item = comparison[metric]
        print(
            f"{metric:<24} delta={_fmt(item['mean_delta'])} "
            f"wins/ties/losses={item['wins']}/{item['ties']}/{item['losses']} "
            f"win_rate={_fmt(item['win_rate'])}"
        )

    # Build a small human-readable list of pairs worth opening visually.
    interesting = []
    for metric in (
        "mean_mask_iou",
        "path_cosine_z",
        "path_cosine_margin",
        "mean_path_cosine",
        "normalized_nw_score",
    ):
        item = comparison[metric]
        if not item["count"]:
            continue
        interesting.append(f"## {metric}")
        interesting.append("")
        interesting.append("Primary improves most:")
        for row in item["best"]:
            interesting.append(
                f"- {row['pair_id']}: delta={row['delta']:.4f} "
                f"(primary={row['left']:.4f}, local={row['right']:.4f})"
            )
        interesting.append("")
        interesting.append("Primary degrades most:")
        for row in item["worst"]:
            interesting.append(
                f"- {row['pair_id']}: delta={row['delta']:.4f} "
                f"(primary={row['left']:.4f}, local={row['right']:.4f})"
            )
        interesting.append("")
    (root / "interesting_pairs.md").write_text(
        "\n".join(interesting) + "\n", encoding="utf-8"
    )

    print(f"\nSaved: {root / 'representation_comparison.json'}")
    print(f"Saved: {root / 'representation_comparison.csv'}")
    print(f"Saved: {root / 'interesting_pairs.md'}")


if __name__ == "__main__":
    main()
