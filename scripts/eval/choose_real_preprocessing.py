#!/usr/bin/env python3
"""Choose binarized or non-binarized real preprocessing from Stage-1 diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

THRESHOLDS = ("0.40", "0.50", "0.60", "0.65", "0.70")


def metrics(root: Path):
    rows = []
    for threshold in THRESHOLDS:
        pos_path = root / f"raw_t{threshold}_positive" / "samples.csv"
        neg_path = root / f"raw_t{threshold}_negative" / "samples.csv"
        if not pos_path.is_file() or not neg_path.is_file():
            continue
        pos = pd.read_csv(pos_path)
        neg = pd.read_csv(neg_path)
        pos = pos[pos["status"] == "ok"].copy()
        neg = neg[neg["status"] == "ok"].copy()
        if pos.empty or neg.empty:
            continue
        pos["target"] = 1
        neg["target"] = 0
        df = pd.concat([pos, neg], ignore_index=True)
        y = df["target"].to_numpy()
        steps = float(roc_auc_score(y, df["path_steps"].to_numpy()))
        matched = float(roc_auc_score(y, df["line1_matched_fraction"].to_numpy()))
        score = float(roc_auc_score(y, df["score"].to_numpy()))
        pos_steps = float(pos["path_steps"].mean())
        neg_steps = float(neg["path_steps"].mean())
        rows.append((threshold, steps, matched, score, pos_steps - neg_steps))
    if not rows:
        raise SystemExit(f"ERROR: no usable discrimination outputs under {root}")
    return max(rows, key=lambda row: (max(row[1], row[2]), row[4], row[3]))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--binarized-root", type=Path, required=True)
    p.add_argument("--gray-root", type=Path, required=True)
    p.add_argument("--output-env", type=Path, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    binary = metrics(args.binarized_root)
    gray = metrics(args.gray_root)
    binary_key = (max(binary[1], binary[2]), binary[4], binary[3])
    gray_key = (max(gray[1], gray[2]), gray[4], gray[3])
    use_binary = binary_key >= gray_key
    chosen = binary if use_binary else gray
    baseline_root = args.binarized_root if use_binary else args.gray_root

    args.output_env.parent.mkdir(parents=True, exist_ok=True)
    args.output_env.write_text(
        "\n".join(
            [
                f"export REAL_BINARIZE={'1' if use_binary else '0'}",
                f"export STAGE1_BASELINE_ROOT='{baseline_root}'",
                f"export STAGE1_BASELINE_THRESHOLD='{chosen[0]}'",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print("=== REAL PREPROCESSING SELECTION ===")
    print(
        f"binarized best: T={binary[0]} steps_AUC={binary[1]:.4f} "
        f"matched_AUC={binary[2]:.4f} score_AUC={binary[3]:.4f} step_gap={binary[4]:.2f}"
    )
    print(
        f"gray best:      T={gray[0]} steps_AUC={gray[1]:.4f} "
        f"matched_AUC={gray[2]:.4f} score_AUC={gray[3]:.4f} step_gap={gray[4]:.2f}"
    )
    print(f"selected REAL_BINARIZE={'1' if use_binary else '0'}")
    print(f"baseline_root={baseline_root}")
    print(f"env_file={args.output_env}")


if __name__ == "__main__":
    main()
