#!/usr/bin/env python3
"""CPU smoke test for train-only partial-overlap real pair synthesis."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=PROJECT_DIR / "DataSet/ArabicDataset")
    p.add_argument("--samples", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    full_manifest = root / "dataset_manifest_full_pairs.jsonl"
    if not full_manifest.is_file():
        raise SystemExit(
            f"ERROR: missing {full_manifest}; run scripts/data/build_full_line_pair_manifest.py first"
        )

    os.environ["REAL_MANIFEST_NAME"] = full_manifest.name
    os.environ.setdefault("REAL_TRAIN_FRACTION", "0.80")
    os.environ.setdefault("REAL_VALID_FRACTION", "0.10")
    os.environ.setdefault("DATASET_SPLIT_SEED", "42")
    os.environ.setdefault("REAL_AUGMENT", "1")
    os.environ.setdefault("REAL_PARTIAL_OVERLAP_MAX_SHARED_ISLANDS", "3")
    os.environ.setdefault("REAL_PARTIAL_OVERLAP_MULTI_ISLAND_PROB", "0.85")
    os.environ.setdefault("REAL_PARTIAL_OVERLAP_THREE_ISLAND_PROB", "0.25")

    import AugmentedRealDataLoader as augmented_loader
    from partial_overlap_runtime_fix import (
        FeasiblePartialOverlapRealPairDataset as PartialOverlapRealPairDataset,
    )
    import extra_real_training as legacy
    import joint_real_training_v5 as joint

    positive_dataset = legacy._manifest_dataset(root, legacy.POSITIVE_LABELS)
    train_raw, valid_raw, test_raw = joint._group_split(positive_dataset)
    train_positive, _stats = legacy._filter_feasible(
        positive_dataset, train_raw, "smoke_train_positive"
    )

    eval_pair_ids = joint._pair_ids(positive_dataset, valid_raw) | joint._pair_ids(
        positive_dataset, test_raw
    )
    eval_page_ids = legacy._sample_page_ids(positive_dataset, (valid_raw, test_raw))
    extra_dataset = legacy._manifest_dataset(root, (legacy.EXTRA_LABEL,))
    extra_indices = []
    for index, sample in enumerate(extra_dataset.samples):
        pair_id = str(sample.get("pair_id", index))
        if pair_id in eval_pair_ids:
            continue
        pages = {
            str(value)
            for value in (sample.get("A_page_id"), sample.get("B_page_id"))
            if value is not None
        }
        if pages & eval_page_ids:
            continue
        extra_indices.append(index)
    if not extra_indices:
        raise SystemExit("ERROR: no leakage-safe no-shared rows for smoke test")

    from torch.utils.data import Subset

    extra_train, _extra_stats = legacy._filter_feasible(
        extra_dataset, Subset(extra_dataset, extra_indices), "smoke_train_no_shared"
    )
    dataset = PartialOverlapRealPairDataset(
        positive_dataset=positive_dataset,
        positive_indices=train_positive.indices,
        distractor_dataset=extra_dataset,
        distractor_indices=extra_train.indices,
        transform=augmented_loader._train_real_transform(),
        target_length=max(1, int(args.samples)),
    )

    counts = {1: 0, 2: 0, 3: 0}
    print("=== PARTIAL OVERLAP SMOKE TEST ===")
    print(f"positive_train_rows={len(train_positive)}")
    print(f"partial_overlap_feasible_positive_anchors={len(dataset.positive_indices)}")
    print(
        "partial_overlap_rejected_long_positive_anchors="
        f"{dataset.partial_overlap_rejected_long_positive_anchors}"
    )
    print(f"no_shared_train_rows={len(extra_train)}")
    for index in range(max(1, int(args.samples))):
        sample = dataset[index]
        image1, image2 = sample["image1"], sample["image2"]
        if tuple(image1.shape) != (3, 128, 1024) or tuple(image2.shape) != (3, 128, 1024):
            raise RuntimeError(
                f"Bad tensor shape at sample {index}: {tuple(image1.shape)} / {tuple(image2.shape)}"
            )
        shared = int(sample["partial_overlap_shared_islands"])
        distractors = int(sample["partial_overlap_distractor_islands"])
        counts[shared] = counts.get(shared, 0) + 1
        print(
            f"sample={index:02d} shared_islands={shared} distractors={distractors} "
            f"text1_chars={len(sample['text1'].strip())} "
            f"text2_chars={len(sample['text2'].strip())} "
            f"shape={tuple(image1.shape)}"
        )
    print("shared_island_histogram=" + str(counts))
    print("SMOKE_TEST=PASS")


if __name__ == "__main__":
    main()
