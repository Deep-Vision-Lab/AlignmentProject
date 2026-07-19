"""Opt-in real-dataset DataLoader with training-only augmentations.

This module intentionally leaves DataLoader.py unchanged. It reuses its manifest
filtering, split logic, binarization, negative generation, and collate function,
then replaces only the training subset with ``AugmentedRealSubset``.
"""
from __future__ import annotations

import os

from torch.utils.data import Dataset
from torchvision import transforms

import DataLoader as base_loader
from RealDataAugmentation import (
    AugmentedRealSubset,
    BinaryInkAugment,
    RealLinePairAugmentor,
)


class RepeatToLengthDataset(Dataset):
    """Repeat a dataset to a requested number of samples per training epoch.

    The real manifest contains fewer unique pairs than a large synthetic dataset.
    Repeating the training subset is useful here because ``AugmentedRealSubset``
    applies fresh random appearance/stitching perturbations every time an item is
    fetched. Validation and test datasets are never wrapped by this class.
    """

    def __init__(self, dataset: Dataset, target_length: int):
        if len(dataset) <= 0:
            raise ValueError("Cannot repeat an empty training dataset.")
        self.dataset = dataset
        self.target_length = max(len(dataset), int(target_length))

    def __len__(self):
        return self.target_length

    def __getitem__(self, index):
        return self.dataset[int(index) % len(self.dataset)]


def _train_real_transform():
    """Return binarization plus post-binary training perturbations."""
    return transforms.Compose(
        [
            base_loader.ResizeAndBinarize(
                size=(128, 1024),
                enabled=base_loader._real_binarize,
                method=base_loader._real_binarize_method,
                fixed_threshold=base_loader._real_binarize_threshold,
                auto_invert=base_loader._real_binarize_auto_invert,
                autocontrast=base_loader._real_binarize_autocontrast,
            ),
            BinaryInkAugment.from_env(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def build_dataloaders(data_dir=None):
    """Build loaders, augmenting only the real-dataset training split."""
    if data_dir is None:
        data_dir = base_loader._default_data_dir

    if base_loader._detect_dataset_type(data_dir) != "real":
        return base_loader.build_dataloaders(data_dir)

    full_dataset = base_loader._build_real_dataset(data_dir)
    split_by_pair = os.environ.get(
        "REAL_SPLIT_BY_PAIR_ID",
        "1",
    ).lower() in {"1", "true", "yes", "on"}
    train_subset, valid_subset, test_subset = (
        base_loader._group_split_real_dataset(full_dataset)
        if split_by_pair
        else base_loader._random_split_seeded(full_dataset)
    )

    augmentor = RealLinePairAugmentor.from_env()
    if augmentor.enabled:
        train_dataset = AugmentedRealSubset(
            base_dataset=full_dataset,
            indices=train_subset.indices,
            transform=_train_real_transform(),
            augmentor=augmentor,
        )
    else:
        train_dataset = train_subset

    base_train_samples = len(train_dataset)
    target_train_samples = int(os.environ.get("REAL_TRAIN_SAMPLES_PER_EPOCH", "0"))
    if target_train_samples > base_train_samples:
        train_dataset = RepeatToLengthDataset(train_dataset, target_train_samples)

    print(
        "Loaded augmented real Arabic dataset: "
        f"samples={len(full_dataset)} base_train={base_train_samples} "
        f"train_per_epoch={len(train_dataset)} "
        f"valid={len(valid_subset)} test={len(test_subset)} "
        f"augment={augmentor.enabled} "
        f"stitch_prob={augmentor.stitch_probability:.3f} "
        f"appearance_prob={augmentor.appearance_probability:.3f} "
        f"stitch_max_chars={augmentor.stitch_max_text_chars} "
        f"binarize={base_loader._real_binarize} "
        f"split_by_pair_id={split_by_pair}",
        flush=True,
    )

    return (
        base_loader._make_loader(train_dataset, shuffle=True),
        base_loader._make_loader(valid_subset, shuffle=False),
        base_loader._make_loader(test_subset, shuffle=False),
    )
