#!/usr/bin/env python3
"""CPU smoke test for the CNN+BiLSTM partial-overlap training mixture."""
from __future__ import annotations

import argparse
import os

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="DataSet/ArabicDataset")
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    os.environ.setdefault("DATASET_TYPE", "real")
    os.environ.setdefault("REAL_USE_EXTRA_NO_SHARED", "1")
    os.environ.setdefault("REAL_EXTRA_EXCLUDE_EVAL_PAGES", "1")
    os.environ["REAL_TRAIN_SAMPLES_PER_EPOCH"] = str(max(4, args.samples))
    os.environ.setdefault("DATALOADER_NUM_WORKERS", "0")

    import extra_real_training_partial_overlap as partial
    from PartialOverlapRealAugmentation import PartialOverlapRealPairDataset

    train_loader, valid_loader, test_loader = partial._build_partial_overlap_dataloaders(
        args.data_dir
    )
    train_dataset = train_loader.dataset
    if len(train_dataset.datasets) != 3:
        raise AssertionError("Expected original/partial/no-shared train components")

    original, synthetic, no_shared = train_dataset.datasets
    if not isinstance(synthetic, PartialOverlapRealPairDataset):
        raise AssertionError("Middle train component is not partial-overlap synthesis")

    total = len(train_dataset)
    expected_original = total // 4
    expected_partial = total // 4
    expected_no_shared = total - expected_original - expected_partial
    actual = (len(original), len(synthetic), len(no_shared))
    expected = (expected_original, expected_partial, expected_no_shared)
    if actual != expected:
        raise AssertionError(f"Mixture mismatch: actual={actual} expected={expected}")

    histogram = {}
    inspect_count = min(len(synthetic), max(1, args.samples))
    for index in range(inspect_count):
        sample = synthetic[index]
        if sample.get("label_type") != "medium_match":
            raise AssertionError("Synthetic sample must be a positive medium_match")
        if sample.get("sample_type") != "synthetic_partial_overlap":
            raise AssertionError("Synthetic marker missing")
        for key in ("image1", "image2"):
            image = sample[key]
            if not torch.is_tensor(image) or tuple(image.shape) != (3, 128, 1024):
                raise AssertionError(
                    f"Bad {key} shape/type: {type(image)} {getattr(image, 'shape', None)}"
                )
        islands = int(sample["partial_overlap_shared_islands"])
        distractors = int(sample["partial_overlap_distractor_islands"])
        histogram[islands] = histogram.get(islands, 0) + 1
        print(
            f"sample={index:02d} shared_islands={islands} "
            f"distractors={distractors} text1_chars={len(sample['text1'])} "
            f"text2_chars={len(sample['text2'])} shape={tuple(sample['image1'].shape)}"
        )

    # Eval loaders must remain canonical subsets, never synthetic datasets.
    for name, loader in (("valid", valid_loader), ("test", test_loader)):
        if isinstance(loader.dataset, PartialOverlapRealPairDataset):
            raise AssertionError(f"{name} unexpectedly uses synthetic samples")

    print(f"mixture_counts={actual} total={total}")
    print(f"shared_island_histogram={histogram}")
    print(f"valid={len(valid_loader.dataset)} test={len(test_loader.dataset)}")
    print("SMOKE_TEST=PASS")


if __name__ == "__main__":
    main()
