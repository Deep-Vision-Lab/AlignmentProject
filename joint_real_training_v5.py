"""Joint clean + online-augmented real training from the Stage-1 checkpoint.

This loader deliberately uses only the canonical original ArabicDataset manifest.
It creates fresh augmented views at __getitem__ time, keeps clean original views in
the same training epoch, uses an 80/10/10 page-pair split by default, and adds
safe no_shared_content rows only to training.  Validation/test stay clean.

The objective reuses the established image-text/local-hard-negative/positive-pair
losses and adds the differentiable sequence-ranking objective from v4.
"""
from __future__ import annotations

import os
import random

from torch.utils.data import ConcatDataset, Dataset, Subset

import AugmentedRealDataLoader as augmented_loader
from RealDataAugmentation import AugmentedRealSubset, RealLinePairAugmentor
import extra_real_training as legacy
import extra_real_training_v4 as sequence


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


def _split_fractions() -> tuple[float, float, float]:
    train = _env_float("REAL_TRAIN_FRACTION", 0.80)
    valid = _env_float("REAL_VALID_FRACTION", 0.10)
    test = 1.0 - train - valid
    if not (0.0 < train < 1.0 and 0.0 < valid < 1.0 and 0.0 < test < 1.0):
        raise ValueError(
            "REAL_TRAIN_FRACTION and REAL_VALID_FRACTION must leave a positive "
            f"test fraction; got train={train}, valid={valid}, test={test}."
        )
    return train, valid, test


def _group_split(dataset):
    """Split complete pair_id groups according to the configured real fractions."""
    groups: dict[str, list[int]] = {}
    for sample_index, sample in enumerate(dataset.samples):
        pair_id = str(sample.get("pair_id", f"sample_{sample_index}"))
        groups.setdefault(pair_id, []).append(sample_index)

    if len(groups) < 3:
        raise RuntimeError(
            "Joint real training requires at least three pair_id groups to keep "
            "train/validation/test leakage-free."
        )

    seed = _env_int("DATASET_SPLIT_SEED", 42)
    group_ids = list(groups)
    random.Random(seed).shuffle(group_ids)
    train_fraction, valid_fraction, _test_fraction = _split_fractions()
    total = len(dataset)
    train_target = train_fraction * total
    valid_target = valid_fraction * total

    train_indices: list[int] = []
    valid_indices: list[int] = []
    test_indices: list[int] = []
    for group_id in group_ids:
        indices = groups[group_id]
        if len(train_indices) < train_target:
            train_indices.extend(indices)
        elif len(valid_indices) < valid_target:
            valid_indices.extend(indices)
        else:
            test_indices.extend(indices)

    if not train_indices or not valid_indices or not test_indices:
        raise RuntimeError(
            "Could not create non-empty leakage-free real train/valid/test splits."
        )
    return (
        Subset(dataset, train_indices),
        Subset(dataset, valid_indices),
        Subset(dataset, test_indices),
    )


class CleanOnlineAugmentedMix(Dataset):
    """Exact epoch-level clean/online-augmented exposure ratio.

    Both datasets contain the same underlying rows in the same order.  The
    augmented dataset generates a new stochastic view every time it is indexed.
    DataLoader shuffling randomizes the clean/augmented slots within batches.
    """

    def __init__(
        self,
        clean_dataset: Dataset,
        augmented_dataset: Dataset,
        target_length: int,
        clean_views: int,
        augmented_views: int,
    ):
        if len(clean_dataset) <= 0 or len(clean_dataset) != len(augmented_dataset):
            raise ValueError(
                "Clean and augmented joint-real datasets must be non-empty and equal length."
            )
        self.clean_dataset = clean_dataset
        self.augmented_dataset = augmented_dataset
        self.base_length = len(clean_dataset)
        self.clean_views = max(1, int(clean_views))
        self.augmented_views = max(1, int(augmented_views))
        self.cycle = self.clean_views + self.augmented_views
        self.target_length = max(self.cycle, int(target_length))

    def __len__(self):
        return self.target_length

    def __getitem__(self, index):
        index = int(index)
        slot = index % self.cycle
        base_index = (index // self.cycle) % self.base_length
        if slot < self.clean_views:
            return self.clean_dataset[base_index]
        return self.augmented_dataset[base_index]


def _pair_ids(dataset, subset) -> set[str]:
    return {
        str(dataset.samples[int(index)].get("pair_id", index))
        for index in subset.indices
    }


def build_joint_dataloaders(data_dir):
    positive_dataset = legacy._manifest_dataset(data_dir, legacy.POSITIVE_LABELS)
    train_raw, valid_raw, test_raw = _group_split(positive_dataset)

    valid_pair_ids = _pair_ids(positive_dataset, valid_raw)
    test_pair_ids = _pair_ids(positive_dataset, test_raw)
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

    # Use every no-shared row that cannot leak an evaluation pair/page.  Unlike
    # the older loader, no_shared-only pair_ids are allowed into training.
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
        raise RuntimeError(
            "Joint real training found no leakage-safe no_shared_content rows."
        )
    extra_raw = Subset(extra_dataset, extra_indices)
    extra_train, extra_stats = legacy._filter_feasible(
        extra_dataset, extra_raw, "train_no_shared"
    )

    clean_train = ConcatDataset([train_positive, extra_train])
    base_rows = len(clean_train)

    augmentor = RealLinePairAugmentor.from_env()
    if not augmentor.enabled:
        raise RuntimeError(
            "Joint real training requires REAL_AUGMENT=1 so augmentations are "
            "generated online from the original ArabicDataset."
        )
    train_transform = augmented_loader._train_real_transform()
    augmented_train = ConcatDataset(
        [
            AugmentedRealSubset(
                positive_dataset,
                train_positive.indices,
                train_transform,
                augmentor,
            ),
            AugmentedRealSubset(
                extra_dataset,
                extra_train.indices,
                train_transform,
                augmentor,
            ),
        ]
    )

    clean_views = _env_int("REAL_CLEAN_VIEWS_PER_CYCLE", 1)
    augmented_views = _env_int("REAL_AUG_VIEWS_PER_CYCLE", 2)
    multiplier = max(1, _env_int("REAL_EFFECTIVE_EPOCH_MULTIPLIER", 6))
    requested = _env_int("REAL_TRAIN_SAMPLES_PER_EPOCH", 0)
    target_length = requested if requested > 0 else base_rows * multiplier
    train_dataset = CleanOnlineAugmentedMix(
        clean_train,
        augmented_train,
        target_length=target_length,
        clean_views=clean_views,
        augmented_views=augmented_views,
    )

    train_fraction, valid_fraction, test_fraction = _split_fractions()
    clean_ratio = clean_views / float(clean_views + augmented_views)
    positive_paths = legacy._line_paths(positive_dataset, train_positive)
    extra_paths = legacy._line_paths(extra_dataset, extra_train)

    print(
        "Joint real training dataset: "
        f"source={data_dir} "
        f"split={train_fraction:.2f}/{valid_fraction:.2f}/{test_fraction:.2f} "
        f"positive_train_rows={len(train_positive)} "
        f"no_shared_train_rows={len(extra_train)} "
        f"clean_base_rows={base_rows} "
        f"effective_train_rows={len(train_dataset)} "
        f"clean_ratio={clean_ratio:.3f} augmented_ratio={1.0-clean_ratio:.3f} "
        f"online_augmentation=True appearance_prob={augmentor.appearance_probability:.3f} "
        f"stitch_prob={augmentor.stitch_probability:.3f} "
        f"valid={len(valid_positive)} test={len(test_positive)} "
        f"positive_unique_lines={len(positive_paths)} "
        f"no_shared_unique_lines={len(extra_paths)} "
        f"excluded_eval_pair_rows={excluded_eval_pair} "
        f"excluded_eval_page_rows={excluded_eval_page}",
        flush=True,
    )
    print(
        "Joint-real feasibility: "
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
    # v4 calls legacy.install(); replacing this module-global loader before that
    # call gives v4 the clean+online-augmented data recipe without duplicating the
    # loss plumbing.
    legacy.build_dataloaders = build_joint_dataloaders
    sequence.install(base)

    previous_model_config = base.model_config

    def model_config(stride, args):
        config = dict(previous_model_config(stride, args))
        train_fraction, valid_fraction, test_fraction = _split_fractions()
        config.update(
            {
                "joint_real_training": True,
                "real_train_fraction": train_fraction,
                "real_valid_fraction": valid_fraction,
                "real_test_fraction": test_fraction,
                "real_clean_views_per_cycle": _env_int(
                    "REAL_CLEAN_VIEWS_PER_CYCLE", 1
                ),
                "real_aug_views_per_cycle": _env_int(
                    "REAL_AUG_VIEWS_PER_CYCLE", 2
                ),
                "real_effective_epoch_multiplier": _env_int(
                    "REAL_EFFECTIVE_EPOCH_MULTIPLIER", 6
                ),
                "online_real_augmentation": True,
            }
        )
        return config

    base.model_config = model_config

    if getattr(base.CTX, "is_main", True):
        print(
            "Joint real objective installed: image-text + local hard negatives + "
            "positive image-pair contrastive + sequence ranking; online augmentation only.",
            flush=True,
        )
