#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "Results/Evaluation/Representation_Diagnostics"
)
thresholds = ["0.40", "0.50", "0.60", "0.65", "0.70"]

print(
    f"{'T':>5} {'score_AUC':>10} {'steps_AUC':>10} {'matched_AUC':>12} "
    f"{'AP':>8} {'best_cut':>10} {'pos_score':>10} {'neg_score':>10} "
    f"{'pos_steps':>10} {'neg_steps':>10}"
)

best = None
for threshold in thresholds:
    pos_path = root / f"raw_t{threshold}_positive" / "samples.csv"
    neg_path = root / f"raw_t{threshold}_negative" / "samples.csv"
    if not pos_path.is_file() or not neg_path.is_file():
        continue

    pos = pd.read_csv(pos_path)
    neg = pd.read_csv(neg_path)
    pos = pos[pos["status"] == "ok"].copy()
    neg = neg[neg["status"] == "ok"].copy()
    pos["target"] = 1
    neg["target"] = 0
    df = pd.concat([pos, neg], ignore_index=True)
    y = df["target"].to_numpy()
    score = df["score"].to_numpy()
    steps = df["path_steps"].to_numpy()
    matched = df["line1_matched_fraction"].to_numpy()

    score_auc = roc_auc_score(y, score)
    steps_auc = roc_auc_score(y, steps)
    matched_auc = roc_auc_score(y, matched)
    ap = average_precision_score(y, score)
    fpr, tpr, cuts = roc_curve(y, score)
    best_cut = cuts[(tpr - fpr).argmax()]

    print(
        f"{threshold:>5} {score_auc:10.4f} {steps_auc:10.4f} {matched_auc:12.4f} "
        f"{ap:8.4f} {best_cut:10.4f} {pos['score'].mean():10.4f} "
        f"{neg['score'].mean():10.4f} {pos['path_steps'].mean():10.2f} "
        f"{neg['path_steps'].mean():10.2f}"
    )
    candidate = (max(steps_auc, matched_auc), threshold, score_auc, steps_auc, matched_auc)
    if best is None or candidate > best:
        best = candidate

if best is not None:
    _, threshold, score_auc, steps_auc, matched_auc = best
    print()
    print(
        "Best structural threshold: "
        f"T={threshold} score_AUC={score_auc:.4f} "
        f"steps_AUC={steps_auc:.4f} matched_AUC={matched_auc:.4f}"
    )
    print("Phase-3 reference to beat: steps_AUC=0.6800 matched_AUC=0.6725 at T=0.65")
