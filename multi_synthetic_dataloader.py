"""Balanced loading from several synthetic Arabic datasets.

The default connected-subword experiment can combine Synthetic_Arabic_1 through
Synthetic_Arabic_4 while taking the same number of samples from each source.
For direct subword supervision, training-only augmentations deliberately preserve
pixel geometry so renderer-derived subword intervals remain valid.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from types import ModuleType
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_INSTALLED = False


def flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def integer(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def resolve_synthetic_data_dirs(data_dir: str | os.PathLike[str]) -> list[Path]:
    """Resolve explicit comma-separated roots or discover Arabic_1..Arabic_4."""
    explicit = os.environ.get("SYNTHETIC_DATA_DIRS", "").strip()
    if explicit:
        roots = [
            Path(item.strip()).expanduser().resolve()
            for item in explicit.split(",")
            if item.strip()
        ]
        if not roots:
            raise ValueError("SYNTHETIC_DATA_DIRS did not contain any directories")
        return roots

    root = Path(data_dir).expanduser().resolve()
    if (root / "images").is_dir() and (root / "texts").is_dir():
        return [root]

    if root.name == "DataSet":
        candidates = [root / f"Synthetic_Arabic_{index}" for index in range(1, 5)]
    else:
        candidates = [Path(f"{root}_{index}") for index in range(1, 5)]
    if all((item / "images").is_dir() and (item / "texts").is_dir() for item in candidates):
        return [item.resolve() for item in candidates]
    return [root]


def _detected_samples(images_dir: Path) -> int:
    indices = {
        int(path.stem.rsplit("_", 1)[1])
        for path in images_dir.glob("img1_*.png")
        if path.stem.rsplit("_", 1)[-1].isdigit()
    }
    if not indices:
        return 0
    # TextLineModern addresses samples as 1..N, so require a contiguous prefix.
    contiguous = 0
    while contiguous + 1 in indices:
        contiguous += 1
    return contiguous


def _dataset_paths(root: Path) -> dict[str, str]:
    return {
        "images": str(root / "images"),
        "matrices": str(root / "matrices"),
        "diffNWmatrices": str(root / "diffNWmatrices"),
        "similarity_matrices": str(root / "similarity_matrices"),
        "texts": str(root / "texts"),
    }


def build_balanced_synthetic_dataset(loader_module: ModuleType, data_dir):
    """Build an equal-sized concatenation of all configured synthetic roots."""
    roots = resolve_synthetic_data_dirs(data_dir)
    explicit_multi = bool(os.environ.get("SYNTHETIC_DATA_DIRS", "").strip())
    if len(roots) == 1 and not explicit_multi:
        original = getattr(loader_module, "_multi_synthetic_original_builder", None)
        if original is None:
            raise RuntimeError("Original synthetic builder was not retained")
        return original(str(roots[0]))

    requested = integer("SYNTHETIC_SAMPLES_PER_DIR", 3000)
    if requested <= 0:
        raise ValueError("SYNTHETIC_SAMPLES_PER_DIR must be positive")
    strict = flag("SYNTHETIC_REQUIRE_FULL_PER_DIR", True)
    datasets = []
    summaries = []
    for root in roots:
        images_dir, texts_dir = root / "images", root / "texts"
        if not images_dir.is_dir() or not texts_dir.is_dir():
            raise FileNotFoundError(
                f"Synthetic source expects images/ and texts/: {root}"
            )
        detected = _detected_samples(images_dir)
        if strict and detected < requested:
            raise ValueError(
                f"Synthetic source {root} has only {detected} contiguous samples; "
                f"{requested} are required."
            )
        selected = min(requested, detected)
        if selected <= 0:
            raise ValueError(f"Synthetic source has no usable samples: {root}")
        dataset = loader_module.TextLineModern(
            new_dataset=_dataset_paths(root),
            transform=loader_module.synthetic_transform,
            num_samples_override=selected,
        )
        datasets.append(dataset)
        summaries.append(
            {"data_dir": str(root), "detected": detected, "selected": selected}
        )

    combined = ConcatDataset(datasets)
    combined.synthetic_source_summaries = summaries
    print(
        "Loaded balanced synthetic sources: "
        + ", ".join(
            f"{Path(item['data_dir']).name}={item['selected']}"
            for item in summaries
        )
        + f" total={len(combined)}",
        flush=True,
    )
    return combined


class BoxSafeSyntheticAugment:
    """Change appearance and strokes without moving interval coordinates."""

    def __init__(self):
        self.probability = max(
            0.0,
            min(1.0, float(os.environ.get("DIRECT_SUBWORD_AUGMENT_PROBABILITY", "0.85"))),
        )
        self.clean_probability = max(
            0.0,
            min(1.0, float(os.environ.get("DIRECT_SUBWORD_CLEAN_PROBABILITY", "0.15"))),
        )
        self.noise_std_max = max(
            0.0, float(os.environ.get("DIRECT_SUBWORD_NOISE_STD_MAX", "10.0"))
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        work = image.convert("RGB")
        original_size = work.size
        if random.random() > self.probability or random.random() < self.clean_probability:
            return work

        if random.random() < 0.65:
            work = ImageOps.autocontrast(work)
        if random.random() < 0.60:
            work = ImageEnhance.Contrast(work).enhance(random.uniform(0.70, 1.35))
        if random.random() < 0.45:
            work = ImageEnhance.Brightness(work).enhance(random.uniform(0.82, 1.18))
        if random.random() < 0.35:
            work = work.filter(ImageFilter.GaussianBlur(random.uniform(0.20, 0.85)))
        if random.random() < 0.30:
            # For black ink, MinFilter thickens strokes and MaxFilter erodes them.
            work = work.filter(
                ImageFilter.MinFilter(3)
                if random.random() < 0.5
                else ImageFilter.MaxFilter(3)
            )
        if self.noise_std_max > 0 and random.random() < 0.65:
            array = np.asarray(work, dtype=np.float32).copy()
            std = random.uniform(1.0, self.noise_std_max)
            array += np.random.normal(0.0, std, size=array.shape)
            work = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")

        if work.size != original_size:  # defensive: geometry must never change
            raise RuntimeError("Box-safe augmentation changed image geometry")
        return work


def _clean_tensor_transform():
    return transforms.Compose(
        [
            transforms.Resize((128, 1024)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def _direct_train_transform():
    return transforms.Compose([BoxSafeSyntheticAugment(), _clean_tensor_transform()])


class TransformViewDataset(Dataset):
    """Apply a split-specific image transform while preserving sidecar regions."""

    def __init__(self, dataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def _image(self, value):
        return value if torch.is_tensor(value) else self.transform(value)

    def __getitem__(self, index):
        item = self.dataset[index]
        if isinstance(item, dict):
            result = dict(item)
            result["image1"] = self._image(result["image1"])
            if result.get("image2") is not None:
                result["image2"] = self._image(result["image2"])
            return result
        if isinstance(item, tuple) and len(item) == 2:
            text, image = item
            return text, self._image(image)
        return item


def _install_direct_split_transforms(loader_module: ModuleType) -> None:
    if not flag("DIRECT_SUBWORD_SUPERVISION", False):
        return
    if not flag("DIRECT_SUBWORD_BOX_SAFE_AUGMENT", True):
        return
    original_build = loader_module.build_dataloaders

    def build_dataloaders(data_dir=None):
        resolved = data_dir or loader_module._default_data_dir
        if loader_module._detect_dataset_type(resolved) != "synthetic":
            return original_build(resolved)

        previous = loader_module.synthetic_transform
        loader_module.synthetic_transform = None
        try:
            full_dataset = loader_module._build_synthetic_dataset(resolved)
        finally:
            loader_module.synthetic_transform = previous
        train_subset, valid_subset, test_subset = loader_module._random_split_seeded(
            full_dataset
        )
        clean = _clean_tensor_transform()
        train_dataset = TransformViewDataset(train_subset, _direct_train_transform())
        valid_dataset = TransformViewDataset(valid_subset, clean)
        test_dataset = TransformViewDataset(test_subset, clean)
        print(
            "Direct-subword augmentation: training=box-safe appearance/stroke "
            "validation=test=clean geometry-preserving",
            flush=True,
        )
        return (
            loader_module._make_loader(train_dataset, shuffle=True),
            loader_module._make_loader(valid_dataset, shuffle=False),
            loader_module._make_loader(test_dataset, shuffle=False),
        )

    loader_module.build_dataloaders = build_dataloaders


def config(data_dir=None) -> dict:
    roots = resolve_synthetic_data_dirs(data_dir or os.environ.get("DATA_DIR", "DataSet"))
    return {
        "synthetic_data_dirs": [str(root) for root in roots],
        "synthetic_samples_per_dir": integer("SYNTHETIC_SAMPLES_PER_DIR", 3000),
        "synthetic_require_full_per_dir": flag("SYNTHETIC_REQUIRE_FULL_PER_DIR", True),
        "direct_subword_box_safe_augment": flag(
            "DIRECT_SUBWORD_BOX_SAFE_AUGMENT", True
        ),
    }


def install(loader_module: ModuleType) -> dict:
    """Install balanced source loading and direct-mode split-safe augmentation."""
    global _INSTALLED
    resolved = config()
    if _INSTALLED:
        return resolved
    loader_module._multi_synthetic_original_builder = (
        loader_module._build_synthetic_dataset
    )
    loader_module._build_synthetic_dataset = lambda data_dir: (
        build_balanced_synthetic_dataset(loader_module, data_dir)
    )
    _install_direct_split_transforms(loader_module)
    _INSTALLED = True
    return resolved
