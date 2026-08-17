"""Balanced training on an offline real-conditioned synthetic bridge manifest.

The manifest contains one positive synthetic span and several guaranteed-negative
synthetic spans for every real anchor.  This loader keeps entire anchor ``pair_id``
groups together, uses only positive rows for internal validation/test, and balances
positive vs no-shared rows 50/50 during training regardless of how many negatives
were generated per anchor.

The actual loss plumbing is delegated to ``extra_real_training_v4`` so the same
normalized local image embeddings and differentiable local-alignment objective used
by real evaluation remain active.  v4 also adds the bridge-specific cross-text loss
when ``REAL_SYNTHETIC_BRIDGE=1``.
"""
from __future__ import annotations

import os

from torch.utils.data import Dataset, Subset

import extra_real_training as legacy
import extra_real_training_v4 as sequence
from joint_real_training_v5 import _group_split


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


class BalancedOfflineBridgeMix(Dataset):
    """Alternate positive and negative slots, then let the DataLoader shuffle them."""

    def __init__(self, positive: Dataset, negative: Dataset, target_length: int):
        if len(positive) <= 0 or len(negative) <= 0:
            raise ValueError("Bridge training requires both positive and negative rows.")
        self.positive = positive
        self.negative = negative
        self.target_length = max(2, int(target_length))
        if self.target_length % 2:
            self.target_length += 1

    def __len__(self):
        return self.target_length

    def __getitem__(self, index):
        index = int(index)
        occurrence = index // 2
        if index % 2 == 0:
            return self.positive[occurrence % len(self.positive)]
        return self.negative[occurrence % len(self.negative)]


def _pair_ids(dataset, subset: Subset) -> set[str]:
    return {
        str(dataset.samples[int(index)].get("pair_id", index))
        for index in subset.indices
    }


def build_bridge_dataloaders(data_dir):
    positive_dataset = legacy._manifest_dataset(data_dir, legacy.POSITIVE_LABELS)
    train_raw, valid_raw, test_raw = _group_split(positive_dataset)

    train_positive, train_stats = legacy._filter_feasible(
        positive_dataset, train_raw, "bridge_train_positive"
    )
    valid_positive, valid_stats = legacy._filter_feasible(
        positive_dataset, valid_raw, "bridge_valid_positive"
    )
    test_positive, test_stats = legacy._filter_feasible(
        positive_dataset, test_raw, "bridge_test_positive"
    )

    train_pair_ids = _pair_ids(positive_dataset, train_raw)
    negative_dataset = legacy._manifest_dataset(data_dir, (legacy.EXTRA_LABEL,))
    negative_indices = [
        index
        for index, sample in enumerate(negative_dataset.samples)
        if str(sample.get("pair_id", index)) in train_pair_ids
    ]
    if not negative_indices:
        raise RuntimeError(
            "Offline bridge manifest has no no_shared_content rows belonging to "
            "the internal training anchors."
        )
    negative_raw = Subset(negative_dataset, negative_indices)
    train_negative, negative_stats = legacy._filter_feasible(
        negative_dataset, negative_raw, "bridge_train_negative"
    )

    # Default: expose every generated negative once per epoch and repeat positives
    # as needed to preserve exact 50/50 class balance.  The public launcher may cap
    # this with BRIDGE_TRAIN_SAMPLES_PER_EPOCH for fast pilots.
    natural_target = 2 * len(train_negative)
    requested_target = _env_int("BRIDGE_TRAIN_SAMPLES_PER_EPOCH", 0)
    target_length = requested_target if requested_target > 0 else natural_target
    train_dataset = BalancedOfflineBridgeMix(
        train_positive,
        train_negative,
        target_length=target_length,
    )

    print(
        "Offline real-synthetic bridge dataset: "
        f"source={data_dir} "
        f"positive_train={len(train_positive)} "
        f"negative_train={len(train_negative)} "
        f"train_per_epoch={len(train_dataset)} "
        f"positive_ratio=0.500 negative_ratio=0.500 "
        f"valid_positive={len(valid_positive)} test_positive={len(test_positive)} "
        "online_rendering=False online_augmentation=False",
        flush=True,
    )
    print(
        "Bridge feasibility: "
        f"positive_removed={train_stats.removed} "
        f"negative_removed={negative_stats.removed} "
        f"valid_removed={valid_stats.removed} test_removed={test_stats.removed}",
        flush=True,
    )

    return (
        legacy._make_loader(train_dataset, shuffle=True),
        legacy._make_loader(valid_positive, shuffle=False),
        legacy._make_loader(test_positive, shuffle=False),
    )


def install(base) -> None:
    # v4 calls legacy.install(), and legacy.install() consults this module-global
    # builder at runtime. Replace it first so no online real augmentation is used.
    legacy.build_dataloaders = build_bridge_dataloaders
    sequence.install(base)

    previous_model_config = base.model_config

    def model_config(stride, args):
        config = dict(previous_model_config(stride, args))
        config.update(
            {
                "real_synthetic_bridge": True,
                "bridge_offline_rendering": True,
                "bridge_class_balance_positive": 0.5,
                "bridge_class_balance_negative": 0.5,
                "bridge_cross_text_weight": float(
                    os.environ.get("BRIDGE_CROSS_TEXT_WEIGHT", "0.10")
                ),
                "bridge_cross_text_threshold": float(
                    os.environ.get("BRIDGE_CROSS_TEXT_THRESHOLD", "0.50")
                ),
            }
        )
        return config

    base.model_config = model_config
    if getattr(base.CTX, "is_main", True):
        print(
            "Real-conditioned synthetic bridge installed: offline 50/50 positive/"
            "negative pairs + image-text supervision + positive image-pair loss + "
            "sequence ranking + real-anchor/synthetic-text ranking.",
            flush=True,
        )
