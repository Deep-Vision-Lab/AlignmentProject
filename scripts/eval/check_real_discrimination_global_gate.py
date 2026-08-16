#!/usr/bin/env python3
"""Gate a real-alignment candidate against a baseline's GLOBAL best threshold.

The older continuation gate compares candidate and baseline at the same threshold.
That is useful for stage-to-stage diagnostics but can hide a regression when the
baseline's best operating threshold is different.  This gate is intentionally
stricter for the partial-overlap experiment:

* search all fixed-manifest thresholds for both candidate and baseline;
* require a candidate operating point with long/selective positive paths;
* require that healthy candidate point to beat the baseline's global-best
  structural AUROC (max of steps-AUC and matched-fraction-AUC).

Exit 3 means a scientific gate failure, not an execution failure.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--min-positive-steps", type=float, default=8.0)
    p.add_argument("--min-step-gap", type=float, default=2.0)
    p.add_argument(
        "--min-structural-improvement",
        type=float,
        default=0.0,
        help="Required AUROC improvement over the baseline global best.",
    )
    p.add_argument(
        "--thresholds",
        nargs="*",
        default=["0.40", "0.50", "0.60", "0.65", "0.70"],
    )
    p.add_argument("--no-fail", action="store_true")
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
    row = {
        "threshold": threshold,
        "score_auc": float(roc_auc_score(y, df["score"].to_numpy())),
        "steps_auc": float(roc_auc_score(y, df["path_steps"].to_numpy())),
        "matched_auc": float(
            roc_auc_score(y, df["line1_matched_fraction"].to_numpy())
        ),
        "positive_steps": float(pos["path_steps"].mean()),
        "negative_steps": float(neg["path_steps"].mean()),
    }
    row["step_gap"] = row["positive_steps"] - row["negative_steps"]
    row["structural_auc"] = max(row["steps_auc"], row["matched_auc"])
    return row


def load_rows(root: Path, thresholds: list[str]):
    return [row for t in thresholds if (row := metrics(root, t)) is not None]


def main():
    args = parse_args()
    candidate_rows = load_rows(args.root, args.thresholds)
    baseline_rows = load_rows(args.baseline_root, args.thresholds)
    if not candidate_rows:
        print(f"GLOBAL GATE ERROR: no candidate outputs under {args.root}", file=sys.stderr)
        return 2
    if not baseline_rows:
        print(
            f"GLOBAL GATE ERROR: no baseline outputs under {args.baseline_root}",
            file=sys.stderr,
        )
        return 2

    baseline_best = max(
        baseline_rows,
        key=lambda r: (r["structural_auc"], r["step_gap"], r["score_auc"]),
    )

    for row in candidate_rows:
        row["healthy_paths"] = (
            row["positive_steps"] >= args.min_positive_steps
            and row["step_gap"] >= args.min_step_gap
        )

    healthy = [row for row in candidate_rows if row["healthy_paths"]]
    candidate_best_overall = max(
        candidate_rows,
        key=lambda r: (r["structural_auc"], r["step_gap"], r["score_auc"]),
    )
    candidate_best_healthy = (
        max(
            healthy,
            key=lambda r: (r["structural_auc"], r["step_gap"], r["score_auc"]),
        )
        if healthy
        else None
    )

    required_auc = baseline_best["structural_auc"] + args.min_structural_improvement
    verdict = bool(
        candidate_best_healthy is not None
        and candidate_best_healthy["structural_auc"] > required_auc
    )

    result = {
        "pass": verdict,
        "candidate_root": str(args.root),
        "baseline_root": str(args.baseline_root),
        "requirements": {
            "min_positive_steps": args.min_positive_steps,
            "min_step_gap": args.min_step_gap,
            "min_structural_improvement": args.min_structural_improvement,
            "required_structural_auc_strictly_greater_than": required_auc,
        },
        "baseline_best": baseline_best,
        "candidate_best_overall": candidate_best_overall,
        "candidate_best_healthy": candidate_best_healthy,
        "candidate_thresholds": candidate_rows,
        "baseline_thresholds": baseline_rows,
    }

    args.root.mkdir(parents=True, exist_ok=True)
    baseline_slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.baseline_root.name)
    result_path = args.root / f"global_gate_vs_{baseline_slug}.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== GLOBAL REAL DISCRIMINATION GATE ===")
    print(f"candidate_root={args.root}")
    print(f"baseline_root={args.baseline_root}")
    print(
        "healthy path requirements: "
        f"positive_steps>={args.min_positive_steps:.2f}; "
        f"step_gap>={args.min_step_gap:.2f}"
    )
    print(
        "baseline GLOBAL best: "
        f"T={baseline_best['threshold']} "
        f"structural_AUC={baseline_best['structural_auc']:.4f} "
        f"steps_AUC={baseline_best['steps_auc']:.4f} "
        f"matched_AUC={baseline_best['matched_auc']:.4f} "
        f"positive_steps={baseline_best['positive_steps']:.2f} "
        f"negative_steps={baseline_best['negative_steps']:.2f}"
    )
    print(
        "candidate GLOBAL best (regardless of path health): "
        f"T={candidate_best_overall['threshold']} "
        f"structural_AUC={candidate_best_overall['structural_auc']:.4f} "
        f"positive_steps={candidate_best_overall['positive_steps']:.2f} "
        f"negative_steps={candidate_best_overall['negative_steps']:.2f} "
        f"step_gap={candidate_best_overall['step_gap']:.2f}"
    )
    if candidate_best_healthy is None:
        print("candidate healthy operating point: NONE")
    else:
        print(
            "candidate best HEALTHY point: "
            f"T={candidate_best_healthy['threshold']} "
            f"structural_AUC={candidate_best_healthy['structural_auc']:.4f} "
            f"steps_AUC={candidate_best_healthy['steps_auc']:.4f} "
            f"matched_AUC={candidate_best_healthy['matched_auc']:.4f} "
            f"positive_steps={candidate_best_healthy['positive_steps']:.2f} "
            f"negative_steps={candidate_best_healthy['negative_steps']:.2f} "
            f"step_gap={candidate_best_healthy['step_gap']:.2f}"
        )
    print(f"required_structural_AUC>{required_auc:.4f}")
    print(f"result_json={result_path}")
    print("VERDICT=" + ("PASS" if verdict else "FAIL"))

    if verdict or args.no_fail:
        return 0
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
