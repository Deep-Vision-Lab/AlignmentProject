#!/usr/bin/env python3
"""Compare Point-2 evaluation metrics between old and physical-window ViT models."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

MODES = ("local", "context", "fused", "fused_wrong_context")
METRICS = (
    "normalized_primary_score",
    "normalized_nw_score",
    "mean_path_cosine",
    "mean_mask_iou",
    "path_cosine_margin",
    "path_cosine_z",
)


def _number(value):
    if value in (None, "", "None", "nan", "NaN"):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _load(path: Path):
    rows = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok":
                continue
            rows[str(row.get("pair_id") or row.get("index"))] = row
    return rows


def _paired_delta(new, old, metric):
    values = []
    wins = ties = losses = 0
    for key in sorted(set(new) & set(old)):
        a = _number(new[key].get(metric))
        b = _number(old[key].get(metric))
        if a is None or b is None:
            continue
        delta = a - b
        values.append(delta)
        if delta > 1e-9:
            wins += 1
        elif delta < -1e-9:
            losses += 1
        else:
            ties += 1
    return {
        "count": len(values),
        "mean_delta": sum(values) / len(values) if values else None,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / len(values) if values else None,
    }


def _fmt(value):
    return "n/a" if value is None else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", required=True)
    parser.add_argument("--new-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    old_root = Path(args.old_root).expanduser().resolve()
    new_root = Path(args.new_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "old_root": str(old_root),
        "new_root": str(new_root),
        "delta_definition": "physical_window_minus_resnet_token",
        "representations": {},
    }

    for mode in MODES:
        old_selected = old_root / mode / "selected_pairs.json"
        new_selected = new_root / mode / "selected_pairs.json"
        if old_selected.is_file() and new_selected.is_file():
            if old_selected.read_text(encoding="utf-8") != new_selected.read_text(encoding="utf-8"):
                raise SystemExit(
                    f"Architecture comparison is invalid: selected pairs differ for {mode}"
                )

        old = _load(old_root / mode / "samples.csv")
        new = _load(new_root / mode / "samples.csv")
        report["representations"][mode] = {
            metric: _paired_delta(new, old, metric) for metric in METRICS
        }

    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "representation",
                "metric",
                "count",
                "physical_minus_old_mean_delta",
                "physical_wins",
                "ties",
                "physical_losses",
                "physical_win_rate",
            ]
        )
        for mode in MODES:
            for metric in METRICS:
                item = report["representations"][mode][metric]
                writer.writerow(
                    [
                        mode,
                        metric,
                        item["count"],
                        item["mean_delta"],
                        item["wins"],
                        item["ties"],
                        item["losses"],
                        item["win_rate"],
                    ]
                )

    print("\nPoint-2 architecture comparison: physical-window minus old ResNet-token ViT")
    print("=" * 100)
    for mode in MODES:
        print(f"\n{mode}")
        for metric in ("normalized_primary_score", "normalized_nw_score", "mean_path_cosine", "mean_mask_iou"):
            item = report["representations"][mode][metric]
            print(
                f"  {metric:<22} delta={_fmt(item['mean_delta'])} "
                f"wins/ties/losses={item['wins']}/{item['ties']}/{item['losses']} "
                f"physical_win_rate={_fmt(item['win_rate'])}"
            )

    print(f"\nSaved: {output}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
