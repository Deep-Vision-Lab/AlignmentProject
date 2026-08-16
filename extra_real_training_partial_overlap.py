"""Train-only 25/25/50 partial-overlap mixture for the expanded-real recipe.

This module keeps the canonical positive train/valid/test split from
``extra_real_training`` and replaces only the training mixture:

* 25% original high/medium positive pairs
* 25% synthetic partial-overlap positive pairs
* 50% leakage-safe no_shared_content pairs

Validation and test remain the untouched canonical positive subsets.  The
sequence-ranking objective is inherited unchanged from ``extra_real_training_v4``.
"""
from __future__ import annotations

import os
import random

from torch.utils.data import ConcatDataset, Dataset, Subset

import DataLoader as base_loader
import extra_real_training as legacy
from PartialOverlapRealAugmentation import PartialOverlapRealPairDataset


class ExactLengthDataset(Dataset):
    """Deterministically sample/repeat a dataset to exactly ``target_length``."""

    def __init__(self, dataset, target_length: int, seed: int = 42):
        if len(dataset) <= 0:
            raise ValueError("Cannot resize an empty dataset")
        self.dataset = dataset
        self.target_length = max(1, int(target_length))
        rng = random.Random(int(seed))
        self.indices = []
        while len(self.indices) < self.target_length:
            block = list(range(len(dataset)))
            rng.shuffle(block)
            self.indices.extend(block)
        self.indices = self.indices[: self.target_length]

    def __len__(self):
        return self.target_length

    def __getitem__(self, index):
        return self.dataset[self.indices[int(index)]]


def _build_partial_overlap_dataloaders(data_dir):
    positive_dataset = legacy._manifest_dataset(data_dir, legacy.POSITIVE_LABELS)
    train_raw, valid_raw, test_raw = base_loader._group_split_real_dataset(
        positive_dataset
    )

    # Freeze the canonical group assignment first.  The no-shared pool is then
    # restricted to training pair IDs and (by default) cannot touch eval pages.
    train_pair_ids = legacy._sample_pair_ids(positive_dataset, train_raw)
    eval_page_ids = legacy._sample_page_ids(positive_dataset, (valid_raw, test_raw))

    train_positive, train_stats = legacy._filter_feasible(
        positive_dataset, train_raw, "train_positive"
    )
    valid_positive, valid_stats = legacy._filter_feasible(
        positive_dataset, valid_raw, "valid"
    )
    test_positive, test_stats = legacy._filter_feasible(
        positive_dataset, test_raw, "test"
    )

    extra_dataset = legacy._manifest_dataset(data_dir, (legacy.EXTRA_LABEL,))
    strict_eval_page_exclusion = legacy._flag("REAL_EXTRA_EXCLUDE_EVAL_PAGES", True)
    extra_indices = []
    excluded_pair = 0
    excluded_eval_page = 0
    for index, sample in enumerate(extra_dataset.samples):
        pair_id = str(sample.get("pair_id", index))
        if pair_id not in train_pair_ids:
            excluded_pair += 1
            continue
        if strict_eval_page_exclusion:
            sample_pages = {
                str(value)
                for value in (sample.get("A_page_id"), sample.get("B_page_id"))
                if value is not None
            }
            if sample_pages & eval_page_ids:
                excluded_eval_page += 1
                continue
        extra_indices.append(index)

    if not extra_indices:
        raise RuntimeError(
            "Partial-overlap training found no leakage-safe no_shared_content rows"
        )

    extra_raw = Subset(extra_dataset, extra_indices)
    extra_train, extra_stats = legacy._filter_feasible(
        extra_dataset, extra_raw, "train_no_shared"
    )
    if not train_positive.indices or not extra_train.indices:
        raise RuntimeError("Partial-overlap training needs both positive and no-shared rows")

    natural_total = len(train_positive) + len(extra_train)
    requested_total = int(os.environ.get("REAL_TRAIN_SAMPLES_PER_EPOCH", "0"))
    total = requested_total if requested_total > 0 else natural_total
    total = max(4, total)

    original_count = total // 4
    partial_count = total // 4
    no_shared_count = total - original_count - partial_count
    seed = int(os.environ.get("DATASET_SPLIT_SEED", "42"))

    original_positive = ExactLengthDataset(
        train_positive, original_count, seed=seed + 11
    )
    no_shared = ExactLengthDataset(
        extra_train, no_shared_count, seed=seed + 29
    )
    partial_overlap = PartialOverlapRealPairDataset(
        positive_dataset=positive_dataset,
        positive_indices=train_positive.indices,
        distractor_dataset=extra_dataset,
        distractor_indices=extra_train.indices,
        transform=base_loader.real_transform,
        target_length=partial_count,
    )
    combined_train = ConcatDataset(
        [original_positive, partial_overlap, no_shared]
    )

    # Defensive leakage assertions based on the raw line-image paths.
    eval_paths = legacy._line_paths(positive_dataset, valid_positive)
    eval_paths.update(legacy._line_paths(positive_dataset, test_positive))
    positive_paths = legacy._line_paths(positive_dataset, train_positive)
    no_shared_paths = legacy._line_paths(extra_dataset, extra_train)
    leaked = (positive_paths | no_shared_paths) & eval_paths
    if leaked:
        preview = sorted(leaked)[:5]
        raise RuntimeError(
            "Partial-overlap source leakage detected against validation/test: "
            f"{preview}"
        )

    print(
        "Partial-overlap training mixture: "
        f"original_positive={original_count} ({original_count / total:.1%}) "
        f"partial_positive={partial_count} ({partial_count / total:.1%}) "
        f"no_shared={no_shared_count} ({no_shared_count / total:.1%}) "
        f"total={len(combined_train)} "
        f"raw_positive_rows={len(train_positive)} "
        f"raw_no_shared_rows={len(extra_train)} "
        f"valid={len(valid_positive)} test={len(test_positive)} "
        f"exclude_eval_pages={strict_eval_page_exclusion} "
        f"excluded_nontrain_pair_rows={excluded_pair} "
        f"excluded_eval_page_rows={excluded_eval_page}",
        flush=True,
    )
    print(
        "Partial-overlap feasibility: "
        f"positive_removed={train_stats.removed} "
        f"extra_removed={extra_stats.removed} "
        f"valid_removed={valid_stats.removed} "
        f"test_removed={test_stats.removed}",
        flush=True,
    )

    return (
        legacy._make_loader(combined_train, shuffle=True),
        legacy._make_loader(valid_positive, shuffle=False),
        legacy._make_loader(test_positive, shuffle=False),
    )


def install(base) -> None:
    # ``legacy.install`` resolves build_dataloaders from its module globals when
    # training starts, so replace that one hook and leave all losses untouched.
    legacy.build_dataloaders = _build_partial_overlap_dataloaders

    from extra_real_training_v4 import install as install_sequence_ranking

    install_sequence_ranking(base)
