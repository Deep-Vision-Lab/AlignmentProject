#!/usr/bin/env python3
"""Gate R0 adaptation by requiring real pair structure to be preserved.

R0 is intentionally trained without partner supervision, so it is not required
to beat Stage 1 yet.  It must avoid a meaningful structural AUROC drop and keep
substantial positive paths before R1 is allowed to start.
"""
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
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--max-auc-drop", type=float, default=0.03)
    p.add_argument("--min-positive-steps", type=float, default=8.0)
    p.add_argument("--thresholds", nargs="*", default=["0.40", "0.50", "0.60", "0.65", "0.70"])
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
    return {
        "threshold": threshold,
        "steps_auc": float(roc_auc_score(y, df["path_steps"].to_numpy())),
        "matched_auc": float(roc_auc_score(y, df["line1_matched_fraction"].to_numpy())),
        "score_auc": float(roc_auc_score(y, df["score"].to_numpy())),
        "positive_steps": float(pos["path_steps"].mean()),
        "negative_steps": float(neg["path_steps"].mean()),
    }


def main():
    args = parse_args()
    candidates = []
    for threshold in args.thresholds:
        row = metrics(args.root, threshold)
        base = metrics(args.baseline_root, threshold)
        if row is None or base is None:
            continue
        row["baseline_steps_auc"] = base["steps_auc"]
        row["baseline_matched_auc"] = base["matched_auc"]
        row["structural_auc"] = max(row["steps_auc"], row["matched_auc"])
        row["baseline_structural_auc"] = max(base["steps_auc"], base["matched_auc"])
        row["auc_drop"] = row["baseline_structural_auc"] - row["structural_auc"]
        row["pass"] = (
            row["auc_drop"] <= args.max_auc_drop
            and row["positive_steps"] >= args.min_positive_steps
        )
        candidates.append(row)

    if not candidates:
        print("PRESERVATION GATE ERROR: no comparable outputs", file=sys.stderr)
        return 2
    passing = [row for row in candidates if row["pass"]]
    best = max(passing or candidates, key=lambda r: (r["structural_auc"], r["positive_steps"]))
    verdict = bool(passing)
    result = {"pass": verdict, "best": best, "all_thresholds": candidates}
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "preservation_gate_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("=== R0 REAL DISCRIMINATION PRESERVATION GATE ===")
    print(f"root={args.root}")
    print(f"baseline_root={args.baseline_root}")
    print(f"allowed_structural_auc_drop<={args.max_auc_drop:.4f}")
    print(
        f"best T={best['threshold']} structural_AUC={best['structural_auc']:.4f} "
        f"baseline={best['baseline_structural_auc']:.4f} drop={best['auc_drop']:.4f} "
        f"positive_steps={best['positive_steps']:.2f}"
    )
    print("VERDICT=" + ("PASS" if verdict else "FAIL"))
    return 0 if verdict else 3


if __name__ == "__main__":
    raise SystemExit(main())
