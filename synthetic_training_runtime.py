"""Synthetic training split and controlled mild augmentation for the global trainer."""
from __future__ import annotations

import random

from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

import Parameters as P


class _SyntheticTrainingTransform:
    """Mild geometry/appearance augmentation applied only to selected train samples."""

    def __init__(self):
        self.pipeline = transforms.Compose(
            [
                transforms.Resize((int(P.line_height), int(P.line_width))),
                transforms.RandomAffine(
                    degrees=float(P.synthetic_aug_rotate_deg),
                    translate=(
                        float(P.synthetic_aug_translate_x),
                        float(P.synthetic_aug_translate_y),
                    ),
                    scale=(
                        float(P.synthetic_aug_scale_min),
                        float(P.synthetic_aug_scale_max),
                    ),
                    interpolation=InterpolationMode.BILINEAR,
                    fill=255,
                ),
                transforms.ColorJitter(
                    brightness=float(P.synthetic_aug_brightness),
                    contrast=float(P.synthetic_aug_contrast),
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __call__(self, image):
        return self.pipeline(image)


class _MixedSyntheticTrainDataset(Dataset):
    """Expose exactly N train items, with exactly a configured fraction augmented."""

    def __init__(
        self,
        clean_dataset,
        augmented_dataset,
        indices,
        augmentation_probability: float,
        seed: int,
    ):
        self.clean_dataset = clean_dataset
        self.augmented_dataset = augmented_dataset
        self.indices = tuple(int(index) for index in indices)
        probability = min(1.0, max(0.0, float(augmentation_probability)))
        self.augmented_count = int(round(len(self.indices) * probability))
        rng = random.Random(int(seed))
        positions = list(range(len(self.indices)))
        self.augmented_positions = frozenset(
            rng.sample(positions, self.augmented_count)
            if self.augmented_count > 0
            else []
        )

    @property
    def clean_count(self) -> int:
        return len(self.indices) - self.augmented_count

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        source_index = self.indices[int(position)]
        if int(position) in self.augmented_positions:
            return self.augmented_dataset[source_index]
        return self.clean_dataset[source_index]


def _configured_split(total: int):
    train_count = int(getattr(P, "synthetic_train_samples", 0))
    if train_count <= 0:
        return None
    if train_count > total - 2:
        raise ValueError(
            "synthetic_train_samples must leave at least two samples for "
            f"validation/test; requested train={train_count}, available={total}."
        )

    order = list(range(total))
    random.Random(int(P.dataset_split_seed)).shuffle(order)
    train_indices = order[:train_count]
    remaining = order[train_count:]
    valid_count = len(remaining) // 2
    valid_indices = remaining[:valid_count]
    test_indices = remaining[valid_count:]
    if not valid_indices or not test_indices:
        raise ValueError(
            f"Synthetic split produced an empty validation/test set: total={total}, "
            f"train={train_count}."
        )
    return train_indices, valid_indices, test_indices


def install(data_module) -> None:
    """Patch only the synthetic branch of DataLoader.build_dataloaders."""
    if getattr(data_module, "_controlled_synthetic_training_installed", False):
        return

    original_build_dataloaders = data_module.build_dataloaders

    def build_dataloaders(data_dir=None):
        if data_dir is None:
            data_dir = data_module._default_data_dir
        if data_module._detect_dataset_type(data_dir) != "synthetic":
            return original_build_dataloaders(data_dir)

        clean_dataset = data_module._build_synthetic_dataset(data_dir)
        split = _configured_split(len(clean_dataset))
        if split is None:
            return original_build_dataloaders(data_dir)
        train_indices, valid_indices, test_indices = split

        if bool(P.synthetic_augment) and float(P.synthetic_augment_probability) > 0:
            augmented_dataset = data_module.TextLineModern(
                new_dataset=clean_dataset.new_dataset,
                transform=_SyntheticTrainingTransform(),
                num_samples_override=len(clean_dataset),
            )
            train_dataset = _MixedSyntheticTrainDataset(
                clean_dataset,
                augmented_dataset,
                train_indices,
                P.synthetic_augment_probability,
                P.dataset_split_seed,
            )
            augmented_count = train_dataset.augmented_count
            clean_count = train_dataset.clean_count
        else:
            train_dataset = Subset(clean_dataset, train_indices)
            augmented_count = 0
            clean_count = len(train_indices)

        valid_dataset = Subset(clean_dataset, valid_indices)
        test_dataset = Subset(clean_dataset, test_indices)

        print(
            "Loaded controlled synthetic dataset: "
            f"samples={len(clean_dataset)} data_dir={data_dir}",
            flush=True,
        )
        print(
            "Synthetic split: "
            f"train={len(train_dataset)} valid={len(valid_dataset)} "
            f"test={len(test_dataset)}",
            flush=True,
        )
        print(
            "Synthetic augmentation: "
            f"enabled={bool(P.synthetic_augment)} "
            f"probability={float(P.synthetic_augment_probability):.2f} "
            f"augmented={augmented_count} clean={clean_count}",
            flush=True,
        )

        return (
            data_module._make_loader(train_dataset, shuffle=True),
            data_module._make_loader(valid_dataset, shuffle=False),
            data_module._make_loader(test_dataset, shuffle=False),
        )

    data_module.build_dataloaders = build_dataloaders
    data_module._controlled_synthetic_training_installed = True
