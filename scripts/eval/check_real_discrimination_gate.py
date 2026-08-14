#!/usr/bin/env python3
"""Gate continuation training using fixed-manifest real discrimination metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--baseline-steps-auc", type=float, default=0.6800)
    p.add_argument("--baseline-matched-auc", type=float, default=0.6725)
    p.add_argument("--min-positive-steps", type=float, default=8.0)
    p.add_argument("--min-step-gap", type=float, default=2.0)
    p.add_argument("--thresholds", nargs="*", default=["0.40", "0.50", "0.60", "0.65", "0.70"])
    p.add_argument("--no-fail", action="store_true", help="Print verdict but always exit 0.")
    return p.parse_args()


def metrics(root: Path, threshold: str):
    pos_path = root / f"raw_t{threshold}_positive" / "samples.csv"
    neg_path = root / f"raw_t{threshold}_negative" / "samples.csv"
    if not pos_path.is_file() or not neg_path.is_file():
        return None
    pos = pd.read_csv(pos_path)
    neg = pd.read_csv(neg_path)
    pos = pos[pos["status"] == "ok"].copy()
    neg = neg[neg["status"] == "ok"].copy()
    if pos.empty or neg.empty:
        return None
    pos["target"] = 1
    neg["target"] = 0
    df = pd.concat([pos, neg], ignore_index=True)
    y = df["target"].to_numpy()
    steps_auc = float(roc_auc_score(y, df["path_steps"].to_numpy()))
    matched_auc = float(roc_auc_score(y, df["line1_matched_fraction"].to_numpy()))
    score_auc = float(roc_auc_score(y, df["score"].to_numpy()))
    pos_steps = float(pos["path_steps"].mean())
    neg_steps = float(neg["path_steps"].mean())
    return {
        "threshold": threshold,
        "score_auc": score_auc,
        "steps_auc": steps_auc,
        "matched_auc": matched_auc,
        "positive_steps": pos_steps,
        "negative_steps": neg_steps,
        "step_gap": pos_steps - neg_steps,
    }


def main():
    args = parse_args()
    rows = [row for t in args.thresholds if (row := metrics(args.root, t)) is not None]
    if not rows:
        print(f"GATE ERROR: no usable evaluation outputs under {args.root}", file=sys.stderr)
        return 2

    for row in rows:
        structural_improvement = (
            row["steps_auc"] > args.baseline_steps_auc
            or row["matched_auc"] > args.baseline_matched_auc
        )
        healthy_paths = (
            row["positive_steps"] >= args.min_positive_steps
            and row["step_gap"] >= args.min_step_gap
        )
        row["structural_improvement"] = structural_improvement
        row["healthy_paths"] = healthy_paths
        row["pass"] = structural_improvement and healthy_paths

    # Prefer a passing threshold; otherwise report the strongest structural one.
    passing = [row for row in rows if row["pass"]]
    pool = passing or rows
    best = max(pool, key=lambda r: (max(r["steps_auc"], r["matched_auc"]), r["step_gap"]))
    verdict = bool(passing)

    result = {
        "pass": verdict,
        "reference": {
            "steps_auc": args.baseline_steps_auc,
            "matched_auc": args.baseline_matched_auc,
            "min_positive_steps": args.min_positive_steps,
            "min_step_gap": args.min_step_gap,
        },
        "best": best,
        "all_thresholds": rows,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "gate_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== REAL DISCRIMINATION GATE ===")
    print(f"root={args.root}")
    print(
        "reference: "
        f"steps_AUC>{args.baseline_steps_auc:.4f} OR matched_AUC>{args.baseline_matched_auc:.4f}; "
        f"positive_steps>={args.min_positive_steps:.2f}; step_gap>={args.min_step_gap:.2f}"
    )
    print(
        f"best T={best['threshold']} score_AUC={best['score_auc']:.4f} "
        f"steps_AUC={best['steps_auc']:.4f} matched_AUC={best['matched_auc']:.4f} "
        f"positive_steps={best['positive_steps']:.2f} negative_steps={best['negative_steps']:.2f} "
        f"step_gap={best['step_gap']:.2f}"
    )
    print("VERDICT=" + ("PASS" if verdict else "FAIL"))

    if verdict or args.no_fail:
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
