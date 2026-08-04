"""Balanced multi-source synthetic loading and geometry-safe augmentation.

The connected-subword CNN experiments can train from four independently
rendered synthetic datasets while preserving equal representation from every
source.  By default, when ``Synthetic_Arabic_1`` through
``Synthetic_Arabic_4`` are available beside ``DATA_DIR``, exactly 3,000 samples
are loaded from each source.

Direct no-DTW supervision uses renderer-provided horizontal subword intervals.
For that mode this module installs a photometric augmentation profile that does
not crop, rotate, translate, or horizontally rescale the image, so the saved
intervals remain valid.  Span-DTW training continues to use the existing
zero-shot augmentation profile.
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms


DEFAULT_SOURCE_NAMES = tuple(f"Synthetic_Arabic_{index}" for index in range(1, 5))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_IMAGE_PATTERN = re.compile(r"img1_(\d+)\.png$")

try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    _BILINEAR = Image.BILINEAR


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


def number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def source_names() -> tuple[str, ...]:
    raw = os.environ.get("SYNTHETIC_SOURCE_DIRS", "").strip()
    if not raw:
        return DEFAULT_SOURCE_NAMES
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        raise ValueError("SYNTHETIC_SOURCE_DIRS did not contain any usable paths")
    return names


def _candidate_roots(data_dir: str | os.PathLike[str]) -> list[Path]:
    resolved = Path(data_dir).expanduser().resolve()
    roots = [resolved]
    if resolved.name in DEFAULT_SOURCE_NAMES or (
        (resolved / "images").is_dir() and (resolved / "texts").is_dir()
    ):
        roots.insert(0, resolved.parent)
    unique: list[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key not in seen:
            seen.add(key)
            unique.append(root)
    return unique


def resolve_source_dirs(data_dir: str | os.PathLike[str]) -> list[Path]:
    """Resolve configured source paths relative to ``data_dir`` or its parent."""
    roots = _candidate_roots(data_dir)
    resolved: list[Path] = []
    for token in source_names():
        path = Path(token).expanduser()
        if path.is_absolute():
            candidate = path.resolve()
        else:
            candidate = next(
                (root / path for root in roots if (root / path).is_dir()),
                roots[0] / path,
            ).resolve()
        resolved.append(candidate)
    return resolved


def _available_indices(source_dir: Path) -> list[int]:
    images_dir = source_dir / "images"
    texts_dir = source_dir / "texts"
    if not images_dir.is_dir() or not texts_dir.is_dir():
        raise FileNotFoundError(
            f"Synthetic source must contain images/ and texts/: {source_dir}"
        )
    indices = []
    for path in images_dir.iterdir():
        match = _IMAGE_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        index = int(match.group(1))
        if (texts_dir / f"text1_{index}.txt").is_file():
            indices.append(index)
    return sorted(set(indices))


def _validate_contiguous_prefix(source_dir: Path, count: int) -> None:
    available = set(_available_indices(source_dir))
    missing = [index for index in range(1, count + 1) if index not in available]
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise FileNotFoundError(
            f"{source_dir} cannot provide the requested first {count} samples; "
            f"missing indices: {preview}"
        )


def build_balanced_dataset(data_loader_module, data_dir: str | os.PathLike[str]):
    """Build a concatenation containing the same number of samples per source."""
    source_dirs = resolve_source_dirs(data_dir)
    samples_per_source = max(1, integer("SYNTHETIC_SAMPLES_PER_SOURCE", 3000))
    require_all = flag("SYNTHETIC_REQUIRE_ALL_SOURCES", True)

    usable: list[Path] = []
    errors = []
    for source_dir in source_dirs:
        try:
            _validate_contiguous_prefix(source_dir, samples_per_source)
            usable.append(source_dir)
        except (FileNotFoundError, ValueError) as exc:
            errors.append(str(exc))

    if errors and require_all:
        raise FileNotFoundError("\n".join(errors))
    if not usable:
        raise FileNotFoundError(
            "No usable synthetic sources were found. Checked: "
            + ", ".join(str(path) for path in source_dirs)
        )

    datasets = []
    for source_dir in usable:
        paths = {
            "images": str(source_dir / "images"),
            "matrices": str(source_dir / "matrices"),
            "diffNWmatrices": str(source_dir / "diffNWmatrices"),
            "similarity_matrices": str(source_dir / "similarity_matrices"),
            "texts": str(source_dir / "texts"),
        }
        datasets.append(
            data_loader_module.TextLineModern(
                new_dataset=paths,
                transform=data_loader_module.synthetic_transform,
                num_samples_override=samples_per_source,
            )
        )

    combined = ConcatDataset(datasets)
    combined.synthetic_source_dirs = [str(path) for path in usable]
    combined.samples_per_source = samples_per_source
    print(
        "Loaded balanced synthetic sources: "
        + ", ".join(f"{path.name}={samples_per_source}" for path in usable)
        + f" total={len(combined)}",
        flush=True,
    )
    return combined


def _border_mean(values: np.ndarray) -> float:
    height, width = values.shape
    border_h = max(1, int(round(height * 0.05)))
    border_w = max(1, int(round(width * 0.01)))
    border = np.concatenate(
        [
            values[:border_h, :].reshape(-1),
            values[-border_h:, :].reshape(-1),
            values[:, :border_w].reshape(-1),
            values[:, -border_w:].reshape(-1),
        ]
    )
    return float(border.mean()) if border.size else 255.0


def _otsu_threshold(gray: np.ndarray) -> int:
    values = np.asarray(gray, dtype=np.uint8)
    histogram = np.bincount(values.reshape(-1), minlength=256).astype(np.float64)
    total = float(values.size)
    if total <= 0:
        return 127
    levels = np.arange(256, dtype=np.float64)
    total_sum = float(np.dot(levels, histogram))
    left_weight = 0.0
    left_sum = 0.0
    best_variance = -1.0
    best_threshold = 127
    for threshold in range(256):
        left_weight += histogram[threshold]
        if left_weight <= 0:
            continue
        right_weight = total - left_weight
        if right_weight <= 0:
            break
        left_sum += threshold * histogram[threshold]
        left_mean = left_sum / left_weight
        right_mean = (total_sum - left_sum) / right_weight
        variance = left_weight * right_weight * (left_mean - right_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            best_threshold = threshold
    return int(best_threshold)


def _add_noise(image: Image.Image, std: float) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=np.float32)
    array += np.random.normal(0.0, float(std), size=array.shape)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="L")


def _add_scan_artifacts(image: Image.Image) -> Image.Image:
    array = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    height, width = array.shape
    for _ in range(random.randint(0, 4)):
        block_w = random.randint(1, max(2, width // 100))
        block_h = random.randint(1, max(2, height // 12))
        x0 = random.randint(0, max(0, width - block_w))
        y0 = random.randint(0, max(0, height - block_h))
        array[y0 : y0 + block_h, x0 : x0 + block_w] = 255
    count = int(array.size * random.uniform(0.0, 0.0025))
    if count:
        ys = np.random.randint(0, height, size=count)
        xs = np.random.randint(0, width, size=count)
        array[ys, xs] = np.random.choice([0, 40, 210, 255], size=count)
    return Image.fromarray(array, mode="L")


class BoxSafeSyntheticPreprocessor:
    """Photometric augmentation that leaves horizontal subword boxes unchanged."""

    def __init__(self, *, training: bool, size=(128, 1024)):
        self.training = bool(training)
        self.height, self.width = map(int, size)
        self.augment = flag("SYNTHETIC_MANUSCRIPT_AUGMENT", True)
        self.augment_probability = number("SYNTHETIC_AUGMENT_PROBABILITY", 0.85)
        self.clean_probability = number("SYNTHETIC_CLEAN_PROBABILITY", 0.20)
        self.threshold_jitter = max(0, integer("SYNTHETIC_THRESHOLD_JITTER", 24))
        self.binarize = flag("SYNTHETIC_BINARIZE", True)
        self.autocontrast = flag("SYNTHETIC_AUTOCONTRAST", True)

    def _augment(self, image: Image.Image) -> Image.Image:
        if not self.training or not self.augment:
            return image
        if random.random() < self.clean_probability:
            return image
        if random.random() > self.augment_probability:
            return image

        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.80, 1.20))
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.35))
        if random.random() < 0.45:
            image = image.filter(ImageFilter.GaussianBlur(random.uniform(0.25, 1.15)))
        if random.random() < 0.35:
            image = image.filter(
                ImageFilter.MinFilter(3)
                if random.random() < 0.5
                else ImageFilter.MaxFilter(3)
            )
        if random.random() < 0.70:
            image = _add_noise(image, random.uniform(2.0, 14.0))
        if random.random() < 0.65:
            image = _add_scan_artifacts(image)
        return image

    def __call__(self, image: Image.Image) -> Image.Image:
        work = image.convert("L").resize((self.width, self.height), _BILINEAR)
        if self.autocontrast:
            work = ImageOps.autocontrast(work)
        work = self._augment(work)
        if not self.binarize:
            return work.convert("RGB")

        gray = np.asarray(ImageOps.autocontrast(work), dtype=np.uint8)
        threshold = _otsu_threshold(gray)
        if self.training and self.threshold_jitter:
            threshold += random.randint(-self.threshold_jitter, self.threshold_jitter)
        threshold = max(0, min(255, int(threshold)))
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
        if _border_mean(binary) < 127.5:
            binary = 255 - binary
        return Image.fromarray(binary, mode="L").convert("RGB")


def build_box_safe_tensor_transform(training: bool):
    return transforms.Compose(
        [
            BoxSafeSyntheticPreprocessor(training=training),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class TransformViewDataset(Dataset):
    """Apply a split-specific transform to PIL images returned by a base dataset."""

    def __init__(self, dataset, transform: Callable):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def _transform_image(self, image):
        return image if torch.is_tensor(image) else self.transform(image)

    def __getitem__(self, index):
        item = self.dataset[index]
        if isinstance(item, dict):
            result = dict(item)
            if "image1" in result:
                result["image1"] = self._transform_image(result["image1"])
            if "image2" in result:
                result["image2"] = self._transform_image(result["image2"])
            return result
        if isinstance(item, tuple) and len(item) == 2:
            text, image = item
            return text, self._transform_image(image)
        return item


def _install_direct_box_safe_profile(data_loader_module) -> None:
    if not flag("DIRECT_SUBWORD_SUPERVISION", False):
        return
    if not flag("DIRECT_SUBWORD_SAFE_AUGMENT", True):
        return
    if getattr(data_loader_module, "_direct_box_safe_profile_installed", False):
        return

    original_build = data_loader_module.build_dataloaders

    def build_dataloaders(data_dir=None):
        resolved = data_dir or data_loader_module._default_data_dir
        if data_loader_module._detect_dataset_type(resolved) != "synthetic":
            return original_build(resolved)

        previous = data_loader_module.synthetic_transform
        data_loader_module.synthetic_transform = None
        try:
            full_dataset = data_loader_module._build_synthetic_dataset(resolved)
        finally:
            data_loader_module.synthetic_transform = previous

        train_subset, valid_subset, test_subset = (
            data_loader_module._random_split_seeded(full_dataset)
        )
        train_dataset = TransformViewDataset(
            train_subset, build_box_safe_tensor_transform(training=True)
        )
        clean_transform = build_box_safe_tensor_transform(training=False)
        valid_dataset = TransformViewDataset(valid_subset, clean_transform)
        test_dataset = TransformViewDataset(test_subset, clean_transform)
        print(
            "Direct-subword box-safe augmentation enabled: photometric-only; "
            "subword x-intervals remain unchanged.",
            flush=True,
        )
        return (
            data_loader_module._make_loader(train_dataset, shuffle=True),
            data_loader_module._make_loader(valid_dataset, shuffle=False),
            data_loader_module._make_loader(test_dataset, shuffle=False),
        )

    data_loader_module.build_dataloaders = build_dataloaders
    data_loader_module._direct_box_safe_profile_installed = True


def config(data_dir: str | os.PathLike[str] | None = None) -> dict:
    configured = [str(path) for path in resolve_source_dirs(data_dir or "DataSet")]
    return {
        "synthetic_multi_source": flag("SYNTHETIC_MULTI_SOURCE", True),
        "synthetic_source_dirs": configured,
        "synthetic_samples_per_source": max(
            1, integer("SYNTHETIC_SAMPLES_PER_SOURCE", 3000)
        ),
        "direct_subword_safe_augment": flag("DIRECT_SUBWORD_SAFE_AUGMENT", True),
        "direct_subword_augmentation_geometry": "fixed",
    }


def install() -> dict:
    """Patch the project's synthetic builder and optional direct-data profile."""
    global _INSTALLED
    import DataLoader as data_loader

    if _INSTALLED:
        return config()
    if not flag("SYNTHETIC_MULTI_SOURCE", True):
        return config()

    original_builder = data_loader._build_synthetic_dataset

    def build_synthetic_dataset(data_dir):
        source_dirs = resolve_source_dirs(data_dir)
        if all(path.is_dir() for path in source_dirs):
            return build_balanced_dataset(data_loader, data_dir)
        if flag("SYNTHETIC_REQUIRE_ALL_SOURCES", True):
            missing = [str(path) for path in source_dirs if not path.is_dir()]
            raise FileNotFoundError(
                "Missing configured synthetic sources: " + ", ".join(missing)
            )
        return original_builder(data_dir)

    data_loader._build_synthetic_dataset = build_synthetic_dataset
    _install_direct_box_safe_profile(data_loader)
    data_loader._multi_source_synthetic_installed = True
    _INSTALLED = True
    return config()
