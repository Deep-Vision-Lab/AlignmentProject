"""Joint real training with online multi-island partial-overlap positives.

This is an opt-in extension of the established joint-real curriculum. Validation
and test remain clean canonical high/medium pairs. Training is balanced 50/50
between positive-class and no-shared rows. Inside the positive half, a configurable
fraction is replaced by online partial-overlap composites built only from
leakage-safe training lines.

Default effective class exposure:
  25% canonical high/medium positive pairs
  25% synthetic partial-overlap positive pairs
  50% no_shared_content negative pairs

Global order consistency should be set to zero for this mode because synthetic
composites intentionally contain unmatched distractor islands.
"""
from __future__ import annotations

import os
import random

from torch.utils.data import Dataset, Subset

import AugmentedRealDataLoader as augmented_loader
from PartialOverlapRealAugmentation import PartialOverlapRealPairDataset
from RealDataAugmentation import AugmentedRealSubset, RealLinePairAugmentor
import extra_real_training as legacy
import extra_real_training_v4 as sequence
import joint_real_training_v5 as joint


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class BalancedPartialOverlapMix(Dataset):
    """Balance canonical positives, synthetic positives, and no-shared negatives."""

    def __init__(
        self,
        clean_positive: Dataset,
        augmented_positive: Dataset,
        partial_positive: Dataset,
        clean_negative: Dataset,
        augmented_negative: Dataset,
        target_length: int,
        clean_views: int,
        augmented_views: int,
        partial_positive_fraction: float,
    ):
        for name, dataset in (
            ("clean_positive", clean_positive),
            ("augmented_positive", augmented_positive),
            ("partial_positive", partial_positive),
            ("clean_negative", clean_negative),
            ("augmented_negative", augmented_negative),
        ):
            if len(dataset) <= 0:
                raise ValueError(f"Partial-overlap joint training requires non-empty {name}")
        self.clean_positive = clean_positive
        self.augmented_positive = augmented_positive
        self.partial_positive = partial_positive
        self.clean_negative = clean_negative
        self.augmented_negative = augmented_negative
        self.clean_views = max(1, int(clean_views))
        self.augmented_views = max(1, int(augmented_views))
        self.view_cycle = self.clean_views + self.augmented_views
        self.target_length = max(4, int(target_length))
        self.partial_positive_fraction = max(0.0, min(0.9, float(partial_positive_fraction)))
        self.partial_cycle = 100
        self.partial_slots = int(round(self.partial_cycle * self.partial_positive_fraction))

    def __len__(self):
        return self.target_length

    def _canonical_positive(self, occurrence: int):
        view_slot = occurrence % self.view_cycle
        base_occurrence = occurrence // self.view_cycle
        use_augmented = view_slot >= self.clean_views
        dataset = self.augmented_positive if use_augmented else self.clean_positive
        return dataset[base_occurrence % len(dataset)]

    def _negative(self, occurrence: int):
        view_slot = occurrence % self.view_cycle
        base_occurrence = occurrence // self.view_cycle
        use_augmented = view_slot >= self.clean_views
        dataset = self.augmented_negative if use_augmented else self.clean_negative
        return dataset[base_occurrence % len(dataset)]

    def __getitem__(self, index):
        index = int(index)
        class_slot = index % 2
        occurrence = index // 2
        if class_slot == 1:
            return self._negative(occurrence)

        if self.partial_slots > 0 and occurrence % self.partial_cycle < self.partial_slots:
            return self.partial_positive[occurrence % len(self.partial_positive)]
        return self._canonical_positive(occurrence)


def build_partial_overlap_dataloaders(data_dir):
    positive_dataset = legacy._manifest_dataset(data_dir, legacy.POSITIVE_LABELS)
    train_raw, valid_raw, test_raw = joint._group_split(positive_dataset)

    valid_pair_ids = joint._pair_ids(positive_dataset, valid_raw)
    test_pair_ids = joint._pair_ids(positive_dataset, test_raw)
    eval_pair_ids = valid_pair_ids | test_pair_ids
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
    extra_indices: list[int] = []
    excluded_eval_pair = 0
    excluded_eval_page = 0
    for index, sample in enumerate(extra_dataset.samples):
        pair_id = str(sample.get("pair_id", index))
        if pair_id in eval_pair_ids:
            excluded_eval_pair += 1
            continue
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
        raise RuntimeError("Partial-overlap training found no leakage-safe no-shared rows")
    extra_raw = Subset(extra_dataset, extra_indices)
    extra_train, extra_stats = legacy._filter_feasible(
        extra_dataset, extra_raw, "train_no_shared"
    )

    augmentor = RealLinePairAugmentor.from_env()
    if not augmentor.enabled:
        raise RuntimeError("Partial-overlap joint training requires REAL_AUGMENT=1")
    train_transform = augmented_loader._train_real_transform()
    augmented_positive = AugmentedRealSubset(
        positive_dataset,
        train_positive.indices,
        train_transform,
        augmentor,
    )
    augmented_negative = AugmentedRealSubset(
        extra_dataset,
        extra_train.indices,
        train_transform,
        augmentor,
    )

    clean_views = _env_int("REAL_CLEAN_VIEWS_PER_CYCLE", 1)
    augmented_views = _env_int("REAL_AUG_VIEWS_PER_CYCLE", 1)
    multiplier = max(1, _env_int("REAL_EFFECTIVE_EPOCH_MULTIPLIER", 4))
    base_rows = len(train_positive) + len(extra_train)
    requested = _env_int("REAL_TRAIN_SAMPLES_PER_EPOCH", 0)
    target_length = requested if requested > 0 else base_rows * multiplier
    partial_fraction = _env_float("REAL_PARTIAL_OVERLAP_POSITIVE_FRACTION", 0.50)

    partial_positive = PartialOverlapRealPairDataset(
        positive_dataset=positive_dataset,
        positive_indices=train_positive.indices,
        distractor_dataset=extra_dataset,
        distractor_indices=extra_train.indices,
        transform=train_transform,
        target_length=max(len(train_positive), target_length // 4),
    )
    train_dataset = BalancedPartialOverlapMix(
        clean_positive=train_positive,
        augmented_positive=augmented_positive,
        partial_positive=partial_positive,
        clean_negative=extra_train,
        augmented_negative=augmented_negative,
        target_length=target_length,
        clean_views=clean_views,
        augmented_views=augmented_views,
        partial_positive_fraction=partial_fraction,
    )

    train_fraction, valid_fraction, test_fraction = joint._split_fractions()
    positive_paths = legacy._line_paths(positive_dataset, train_positive)
    extra_paths = legacy._line_paths(extra_dataset, extra_train)
    overall_partial = 0.5 * partial_fraction
    overall_canonical = 0.5 * (1.0 - partial_fraction)

    print(
        "Partial-overlap real training dataset: "
        f"source={data_dir} "
        f"split={train_fraction:.2f}/{valid_fraction:.2f}/{test_fraction:.2f} "
        f"positive_train_rows={len(train_positive)} "
        f"no_shared_train_rows={len(extra_train)} "
        f"positive_unique_lines={len(positive_paths)} "
        f"no_shared_unique_lines={len(extra_paths)} "
        f"effective_train_rows={len(train_dataset)} "
        f"canonical_positive_ratio={overall_canonical:.3f} "
        f"partial_overlap_positive_ratio={overall_partial:.3f} "
        f"no_shared_ratio=0.500 "
        f"partial_positive_fraction_within_positive={partial_fraction:.3f} "
        f"max_shared_islands={partial_positive.max_shared_islands} "
        f"multi_island_prob={partial_positive.multi_island_probability:.3f} "
        f"three_island_prob={partial_positive.three_island_probability:.3f} "
        f"valid={len(valid_positive)} test={len(test_positive)} "
        f"excluded_eval_pair_rows={excluded_eval_pair} "
        f"excluded_eval_page_rows={excluded_eval_page}",
        flush=True,
    )
    print(
        "Partial-overlap feasibility: "
        f"positive_removed={train_stats.removed} "
        f"no_shared_removed={extra_stats.removed} "
        f"valid_removed={valid_stats.removed} "
        f"test_removed={test_stats.removed}",
        flush=True,
    )

    return (
        legacy._make_loader(train_dataset, shuffle=True),
        legacy._make_loader(valid_positive, shuffle=False),
        legacy._make_loader(test_positive, shuffle=False),
    )


def install(base) -> None:
    legacy.build_dataloaders = build_partial_overlap_dataloaders
    sequence.install(base)

    previous_model_config = base.model_config

    def model_config(stride, args):
        config = dict(previous_model_config(stride, args))
        config.update(
            {
                "partial_overlap_real_training": True,
                "partial_overlap_positive_fraction": _env_float(
                    "REAL_PARTIAL_OVERLAP_POSITIVE_FRACTION", 0.50
                ),
                "partial_overlap_max_shared_islands": _env_int(
                    "REAL_PARTIAL_OVERLAP_MAX_SHARED_ISLANDS", 3
                ),
                "partial_overlap_multi_island_probability": _env_float(
                    "REAL_PARTIAL_OVERLAP_MULTI_ISLAND_PROB", 0.65
                ),
                "partial_overlap_three_island_probability": _env_float(
                    "REAL_PARTIAL_OVERLAP_THREE_ISLAND_PROB", 0.15
                ),
            }
        )
        return config

    base.model_config = model_config

    if getattr(base.CTX, "is_main", True):
        print(
            "Partial-overlap objective installed: canonical positives + online "
            "multi-island partial positives + no-shared sequence negatives; "
            "set SEQUENCE_CONSISTENCY_LOSS_WEIGHT=0 for unmatched distractors.",
            flush=True,
        )
