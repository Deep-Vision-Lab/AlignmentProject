#!/usr/bin/env python3
"""Build deterministic leakage-safe positive/no-shared diagnostic manifests.

The split mirrors the joint-real 80/10/10 pair_id grouping used for training.
Positive diagnostics come only from validation+test pair groups. No-shared
negatives are retained only when their pair_id or page touches those held-out
positive groups, which also guarantees they were excluded from joint-real train.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

POSITIVE_LABELS = {"high_match", "medium_match"}
NEGATIVE_LABEL = "no_shared_content"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--train-fraction", type=float, default=0.80)
    p.add_argument("--valid-fraction", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-per-class", type=int, default=100)
    p.add_argument("--min-per-class", type=int, default=20)
    return p.parse_args()


def label_of(row):
    return str(row.get("label_type", row.get("label", "")))


def page_ids(row):
    return {
        str(v)
        for v in (row.get("A_page_id"), row.get("B_page_id"))
        if v is not None
    }


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON at {path}:{line_no}: {exc}")
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    if not args.manifest.is_file():
        raise SystemExit(f"Missing manifest: {args.manifest}")
    test_fraction = 1.0 - args.train_fraction - args.valid_fraction
    if min(args.train_fraction, args.valid_fraction, test_fraction) <= 0:
        raise SystemExit("Split fractions must leave non-empty train/valid/test fractions")

    rows = load_jsonl(args.manifest)
    positives = [row for row in rows if label_of(row) in POSITIVE_LABELS]
    negatives = [row for row in rows if label_of(row) == NEGATIVE_LABEL]
    if not positives or not negatives:
        raise SystemExit(
            f"Need positive and no_shared rows; found positive={len(positives)} no_shared={len(negatives)}"
        )

    groups = defaultdict(list)
    for index, row in enumerate(positives):
        groups[str(row.get("pair_id", f"sample_{index}"))].append(index)
    if len(groups) < 3:
        raise SystemExit("Need at least three positive pair_id groups")

    group_ids = list(groups)
    random.Random(args.seed).shuffle(group_ids)
    train_target = args.train_fraction * len(positives)
    valid_target = args.valid_fraction * len(positives)
    train_idx, valid_idx, test_idx = [], [], []
    for group_id in group_ids:
        indices = groups[group_id]
        if len(train_idx) < train_target:
            train_idx.extend(indices)
        elif len(valid_idx) < valid_target:
            valid_idx.extend(indices)
        else:
            test_idx.extend(indices)
    if not train_idx or not valid_idx or not test_idx:
        raise SystemExit("Could not form non-empty 80/10/10 grouped split")

    eval_positive = [positives[i] for i in valid_idx + test_idx]
    eval_pair_ids = {str(row.get("pair_id", "")) for row in eval_positive}
    eval_pages = set().union(*(page_ids(row) for row in eval_positive))

    eval_negative = []
    for row in negatives:
        pair_id = str(row.get("pair_id", ""))
        if pair_id in eval_pair_ids or (page_ids(row) & eval_pages):
            eval_negative.append(row)

    # Deterministic ordering independent of source manifest order after candidate creation.
    rng_pos = random.Random(args.seed + 1001)
    rng_neg = random.Random(args.seed + 2001)
    rng_pos.shuffle(eval_positive)
    rng_neg.shuffle(eval_negative)
    eval_positive = eval_positive[: args.max_per_class]
    eval_negative = eval_negative[: args.max_per_class]

    if len(eval_positive) < args.min_per_class or len(eval_negative) < args.min_per_class:
        raise SystemExit(
            "Not enough held-out diagnostics: "
            f"positive={len(eval_positive)} no_shared={len(eval_negative)} "
            f"required={args.min_per_class}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pos_path = args.output_dir / "diagnostic_positive_rows.jsonl"
    neg_path = args.output_dir / "diagnostic_no_shared_rows.jsonl"
    write_jsonl(pos_path, eval_positive)
    write_jsonl(neg_path, eval_negative)

    metadata = {
        "source_manifest": str(args.manifest.resolve()),
        "seed": args.seed,
        "split": {
            "train": args.train_fraction,
            "valid": args.valid_fraction,
            "test": test_fraction,
        },
        "positive_total": len(positives),
        "positive_train_rows": len(train_idx),
        "positive_valid_rows": len(valid_idx),
        "positive_test_rows": len(test_idx),
        "diagnostic_positive_rows": len(eval_positive),
        "diagnostic_no_shared_rows": len(eval_negative),
        "held_out_pair_ids": len(eval_pair_ids),
        "held_out_page_ids": len(eval_pages),
    }
    (args.output_dir / "diagnostic_manifest_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"positive_manifest={pos_path}")
    print(f"negative_manifest={neg_path}")


if __name__ == "__main__":
    main()
