"""Leakage-safe pair_id group split shared by bridge training architectures."""
from __future__ import annotations

import os
import random

from torch.utils.data import Subset


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


def group_split(dataset):
    """Split complete pair_id groups into configured train/valid/test subsets."""
    groups: dict[str, list[int]] = {}
    for sample_index, sample in enumerate(dataset.samples):
        pair_id = str(sample.get("pair_id", f"sample_{sample_index}"))
        groups.setdefault(pair_id, []).append(sample_index)
    if len(groups) < 3:
        raise RuntimeError(
            "Bridge training requires at least three pair_id groups for train/valid/test."
        )

    train_fraction = _env_float("REAL_TRAIN_FRACTION", 0.80)
    valid_fraction = _env_float("REAL_VALID_FRACTION", 0.10)
    test_fraction = 1.0 - train_fraction - valid_fraction
    if not (
        0.0 < train_fraction < 1.0
        and 0.0 < valid_fraction < 1.0
        and 0.0 < test_fraction < 1.0
    ):
        raise ValueError(
            "REAL_TRAIN_FRACTION and REAL_VALID_FRACTION must leave a positive test split."
        )

    group_ids = list(groups)
    random.Random(_env_int("DATASET_SPLIT_SEED", 42)).shuffle(group_ids)
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
        raise RuntimeError("Could not create non-empty bridge train/valid/test group splits.")
    return (
        Subset(dataset, train_indices),
        Subset(dataset, valid_indices),
        Subset(dataset, test_indices),
    )
