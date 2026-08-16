#!/usr/bin/env python3
"""CPU smoke test for the production partial-overlap training mixture.

This exercises the exact loader used by
``NO_SHARED_IMAGE_OBJECTIVE=sequence_ranking_partial_overlap``.  It verifies the
25/25/50 train mixture, samples generated composites, checks fixed tensor shapes,
and relies on the production loader's path-level validation/test leakage guard.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_DIR / "DataSet/ArabicDataset",
        help="Real Arabic dataset root containing dataset_manifest.jsonl.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=12,
        help="Number of generated partial-overlap examples to inspect.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest_name = os.environ.get("REAL_MANIFEST_NAME", "dataset_manifest.jsonl")
    manifest = root / manifest_name
    if not manifest.is_file():
        raise SystemExit(f"ERROR: missing real manifest: {manifest}")

    # Keep the smoke test small while retaining an exactly divisible 25/25/50
    # mixture.  The production launcher defaults to 6000 samples/epoch.
    requested_samples = max(1, int(args.samples))
    smoke_total = max(12, requested_samples * 4)
    smoke_total += (-smoke_total) % 4

    os.environ["REAL_MANIFEST_NAME"] = manifest_name
    os.environ.setdefault("DATASET_SPLIT_SEED", "42")
    os.environ.setdefault("REAL_EXTRA_EXCLUDE_EVAL_PAGES", "1")
    os.environ.setdefault("REAL_VALIDATE_PATHS", "0")
    os.environ["REAL_TRAIN_SAMPLES_PER_EPOCH"] = str(smoke_total)
    os.environ.setdefault("REAL_PARTIAL_OVERLAP_MAX_SHARED_ISLANDS", "3")
    os.environ.setdefault("REAL_PARTIAL_OVERLAP_MULTI_ISLAND_PROB", "0.85")
    os.environ.setdefault("REAL_PARTIAL_OVERLAP_THREE_ISLAND_PROB", "0.25")

    from torch.utils.data import ConcatDataset

    from PartialOverlapRealAugmentation import PartialOverlapRealPairDataset
    from extra_real_training_partial_overlap import _build_partial_overlap_dataloaders

    train_loader, valid_loader, test_loader = _build_partial_overlap_dataloaders(root)
    train_dataset = train_loader.dataset
    if not isinstance(train_dataset, ConcatDataset) or len(train_dataset.datasets) != 3:
        raise RuntimeError(
            "Expected production train dataset to be ConcatDataset(original, partial, no_shared)"
        )

    original_dataset, partial_dataset, no_shared_dataset = train_dataset.datasets
    if not isinstance(partial_dataset, PartialOverlapRealPairDataset):
        raise RuntimeError("Middle train component is not PartialOverlapRealPairDataset")

    counts = (
        len(original_dataset),
        len(partial_dataset),
        len(no_shared_dataset),
    )
    total = sum(counts)
    expected = (total // 4, total // 4, total - 2 * (total // 4))
    if counts != expected:
        raise RuntimeError(f"Bad train mixture: counts={counts}, expected={expected}")

    print("=== PARTIAL OVERLAP PRODUCTION SMOKE TEST ===")
    print(f"manifest={manifest}")
    print(
        "train_mixture="
        f"original:{counts[0]} partial:{counts[1]} no_shared:{counts[2]} total:{total}"
    )
    print(f"valid_rows={len(valid_loader.dataset)} test_rows={len(test_loader.dataset)}")

    histogram: dict[int, int] = {1: 0, 2: 0, 3: 0}
    for index in range(requested_samples):
        sample = partial_dataset[index % len(partial_dataset)]
        image1, image2 = sample["image1"], sample["image2"]
        shape1, shape2 = tuple(image1.shape), tuple(image2.shape)
        if shape1 != (3, 128, 1024) or shape2 != (3, 128, 1024):
            raise RuntimeError(
                f"Bad tensor shape at sample {index}: {shape1} / {shape2}"
            )
        if sample.get("sample_type") != "synthetic_partial_overlap":
            raise RuntimeError(f"Bad sample_type at sample {index}: {sample.get('sample_type')!r}")
        if sample.get("label_type") not in {"high_match", "medium_match"}:
            raise RuntimeError(f"Synthetic sample is not positive: {sample.get('label_type')!r}")

        shared = int(sample["partial_overlap_shared_islands"])
        distractors = int(sample["partial_overlap_distractor_islands"])
        histogram[shared] = histogram.get(shared, 0) + 1
        print(
            f"sample={index:02d} shared_islands={shared} distractors={distractors} "
            f"text1_chars={len(sample['text1'].strip())} "
            f"text2_chars={len(sample['text2'].strip())} shape={shape1}"
        )

    print(f"shared_island_histogram={histogram}")
    print("LEAKAGE_GUARD=PASS")
    print("SMOKE_TEST=PASS")


if __name__ == "__main__":
    main()
