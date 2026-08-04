"""Balanced augmented loader for Synthetic_Arabic_1 through Synthetic_Arabic_4.

The training split contains a fixed number of unique samples from every source
folder. Validation and test samples are selected only from the remaining rows,
so no raw image is shared across splits. Random scale/translation augmentation
is applied online to the training split only.
"""
from __future__ import annotations

import os
import random
import re
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
from torch.utils.data import ConcatDataset, Subset
from torchvision import transforms

import DataLoader as base_loader
from DataSet import TextLineModern


DEFAULT_SYNTHETIC_FOLDERS = tuple(
    f"Synthetic_Arabic_{index}" for index in range(1, 5)
)
_IMAGE_PATTERN = re.compile(r"^img1_(\d+)\.png$", re.IGNORECASE)

try:
    _RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    _RESAMPLE_BILINEAR = Image.BILINEAR


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def synthetic_folder_names() -> tuple[str, ...]:
    """Read the source-folder list from the environment."""
    raw = os.environ.get("SYNTHETIC_DATASET_FOLDERS", "").strip()
    if not raw:
        return DEFAULT_SYNTHETIC_FOLDERS
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        raise ValueError("SYNTHETIC_DATASET_FOLDERS did not contain any folder names.")
    return names


def _candidate_roots(data_dir: str | os.PathLike[str]) -> Iterable[Path]:
    override = os.environ.get("SYNTHETIC_DATA_ROOT", "").strip()
    if override:
        yield Path(override).expanduser()

    requested = Path(data_dir).expanduser()
    yield requested
    yield requested.parent


def resolve_synthetic_folders(
    data_dir: str | os.PathLike[str],
    folder_names: Sequence[str] | None = None,
) -> tuple[Path, tuple[Path, ...]]:
    """Resolve a parent containing every requested synthetic folder."""
    names = tuple(folder_names or synthetic_folder_names())
    checked: list[str] = []
    seen: set[Path] = set()

    for candidate in _candidate_roots(data_dir):
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        checked.append(str(candidate))
        folders = tuple(candidate / name for name in names)
        if all((folder / "images").is_dir() and (folder / "texts").is_dir() for folder in folders):
            return candidate, folders

    expected = ", ".join(names)
    locations = ", ".join(checked)
    raise FileNotFoundError(
        f"Could not find {expected}, each with images/ and texts/. "
        f"Checked: {locations}. Set SYNTHETIC_DATA_ROOT to the parent directory."
    )


def _border_background(image: Image.Image) -> tuple[int, int, int]:
    """Estimate canvas color from the image border (works for either polarity)."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    border_h = max(1, int(round(height * 0.05)))
    border_w = max(1, int(round(width * 0.01)))
    border = np.concatenate(
        (
            gray[:border_h, :].reshape(-1),
            gray[-border_h:, :].reshape(-1),
            gray[:, :border_w].reshape(-1),
            gray[:, -border_w:].reshape(-1),
        )
    )
    level = int(np.median(border)) if border.size else 255
    return level, level, level


def _paste_clipped(
    canvas: Image.Image,
    patch: Image.Image,
    left: int,
    top: int,
) -> None:
    """Paste a possibly shifted patch without relying on negative-box behavior."""
    right = left + patch.width
    bottom = top + patch.height
    dst_left = max(0, left)
    dst_top = max(0, top)
    dst_right = min(canvas.width, right)
    dst_bottom = min(canvas.height, bottom)
    if dst_left >= dst_right or dst_top >= dst_bottom:
        return

    src_left = dst_left - left
    src_top = dst_top - top
    src_right = src_left + (dst_right - dst_left)
    src_bottom = src_top + (dst_bottom - dst_top)
    canvas.paste(
        patch.crop((src_left, src_top, src_right, src_bottom)),
        (dst_left, dst_top),
    )


class SyntheticScaleTranslateAugment:
    """Resize to the model canvas, then randomly scale and translate the line.

    No rotation is used because rotating a line changes the horizontal sequence
    geometry that the alignment loss is learning. The empty canvas color is
    inferred from the border, so both black- and white-background PNGs work.
    """

    def __init__(
        self,
        size: tuple[int, int] = (128, 1024),
        scale_min: float = 0.85,
        scale_max: float = 1.0,
        translate_pct: float = 0.05,
        probability: float = 1.0,
    ) -> None:
        self.height = int(size[0])
        self.width = int(size[1])
        self.scale_min = float(scale_min)
        self.scale_max = float(scale_max)
        self.translate_pct = float(translate_pct)
        self.probability = float(probability)

        if self.height <= 0 or self.width <= 0:
            raise ValueError("Augmentation size must be positive.")
        if not 0.0 < self.scale_min <= self.scale_max:
            raise ValueError("Expected 0 < scale_min <= scale_max.")
        if self.translate_pct < 0.0:
            raise ValueError("translate_pct must be non-negative.")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("probability must be in [0, 1].")

    @classmethod
    def from_env(cls) -> "SyntheticScaleTranslateAugment":
        return cls(
            size=(
                _env_int("SYNTHETIC_IMAGE_HEIGHT", 128),
                _env_int("SYNTHETIC_IMAGE_WIDTH", 1024),
            ),
            scale_min=_env_float("SYNTHETIC_SCALE_MIN", 0.85),
            scale_max=_env_float("SYNTHETIC_SCALE_MAX", 1.0),
            translate_pct=_env_float("SYNTHETIC_TRANSLATE_PCT", 0.05),
            probability=_env_float("SYNTHETIC_AUGMENT_PROB", 1.0),
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB").resize(
            (self.width, self.height),
            _RESAMPLE_BILINEAR,
        )
        if random.random() > self.probability:
            return image

        scale = random.uniform(self.scale_min, self.scale_max)
        scaled_width = max(1, int(round(self.width * scale)))
        scaled_height = max(1, int(round(self.height * scale)))
        scaled = image.resize((scaled_width, scaled_height), _RESAMPLE_BILINEAR)

        max_dx = int(round(self.width * self.translate_pct))
        max_dy = int(round(self.height * self.translate_pct))
        left = (self.width - scaled_width) // 2 + random.randint(-max_dx, max_dx)
        top = (self.height - scaled_height) // 2 + random.randint(-max_dy, max_dy)

        canvas = Image.new("RGB", (self.width, self.height), _border_background(image))
        _paste_clipped(canvas, scaled, left, top)
        return canvas


def build_synthetic_train_transform():
    """Return the online transform used only for synthetic training samples."""
    if not _env_flag("SYNTHETIC_AUGMENT", True):
        return base_loader.synthetic_transform
    return transforms.Compose(
        [
            SyntheticScaleTranslateAugment.from_env(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def _sample_count(folder: Path) -> int:
    indices = []
    for path in (folder / "images").iterdir():
        match = _IMAGE_PATTERN.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    if not indices:
        raise FileNotFoundError(f"No img1_*.png files found in {folder / 'images'}.")

    indices.sort()
    expected = list(range(1, indices[-1] + 1))
    if indices != expected:
        available = set(indices)
        missing = [index for index in expected if index not in available]
        preview = ", ".join(str(index) for index in missing[:10])
        raise ValueError(
            f"{folder.name} has non-contiguous img1 indices; first missing: {preview}. "
            "TextLineModern expects indices 1..N."
        )
    return indices[-1]


def _dataset_paths(folder: Path) -> dict[str, str]:
    return {
        "images": str(folder / "images"),
        "matrices": str(folder / "matrices"),
        "diffNWmatrices": str(folder / "diffNWmatrices"),
        "similarity_matrices": str(folder / "similarity_matrices"),
        "texts": str(folder / "texts"),
    }


def _build_dataset(folder: Path, transform, count: int) -> TextLineModern:
    return TextLineModern(
        new_dataset=_dataset_paths(folder),
        transform=transform,
        num_samples_override=count,
    )


def _split_indices(
    total: int,
    train_count: int,
    valid_requested: int,
    test_requested: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """Take an exact train count and disjoint evaluation samples."""
    if train_count <= 0:
        raise ValueError("SYNTHETIC_TRAIN_SAMPLES_PER_FOLDER must be positive.")
    if total < train_count:
        raise ValueError(
            f"Requested {train_count} training samples but the folder has only {total}."
        )

    order = list(range(total))
    random.Random(seed).shuffle(order)
    train_indices = order[:train_count]
    remaining = order[train_count:]

    if len(remaining) < 2:
        raise ValueError(
            "At least two unused samples per folder are required for disjoint "
            "validation and test splits."
        )

    valid_requested = max(1, int(valid_requested))
    test_requested = max(1, int(test_requested))
    requested_total = valid_requested + test_requested
    if len(remaining) >= requested_total:
        valid_count = valid_requested
        test_count = test_requested
    else:
        valid_fraction = valid_requested / requested_total
        valid_count = max(1, int(round(len(remaining) * valid_fraction)))
        valid_count = min(valid_count, len(remaining) - 1)
        test_count = len(remaining) - valid_count

    valid_indices = remaining[:valid_count]
    test_indices = remaining[valid_count : valid_count + test_count]
    return train_indices, valid_indices, test_indices


def build_dataloaders(data_dir=None):
    """Build 4-folder balanced synthetic train/valid/test loaders.

    Defaults:
      * 3,000 unique training samples from each Synthetic_Arabic_1..4;
      * 500 validation and 500 test samples per folder when available;
      * online scale/translation augmentation on training samples only.
    """
    if data_dir is None:
        data_dir = "DataSet"

    folder_names = synthetic_folder_names()
    data_root, folders = resolve_synthetic_folders(data_dir, folder_names)
    train_count = _env_int("SYNTHETIC_TRAIN_SAMPLES_PER_FOLDER", 3000)
    valid_count = _env_int("SYNTHETIC_VALID_SAMPLES_PER_FOLDER", 500)
    test_count = _env_int("SYNTHETIC_TEST_SAMPLES_PER_FOLDER", 500)
    split_seed = _env_int(
        "SYNTHETIC_SPLIT_SEED",
        _env_int("DATASET_SPLIT_SEED", 42),
    )

    train_transform = build_synthetic_train_transform()
    train_parts = []
    valid_parts = []
    test_parts = []
    summaries = []

    for folder_offset, folder in enumerate(folders):
        count = _sample_count(folder)
        train_indices, valid_indices, test_indices = _split_indices(
            total=count,
            train_count=train_count,
            valid_requested=valid_count,
            test_requested=test_count,
            seed=split_seed + folder_offset,
        )
        train_dataset = _build_dataset(folder, train_transform, count)
        eval_dataset = _build_dataset(folder, base_loader.synthetic_transform, count)
        train_parts.append(Subset(train_dataset, train_indices))
        valid_parts.append(Subset(eval_dataset, valid_indices))
        test_parts.append(Subset(eval_dataset, test_indices))
        summaries.append(
            f"{folder.name}: available={count} train={len(train_indices)} "
            f"valid={len(valid_indices)} test={len(test_indices)}"
        )

    train_dataset = ConcatDataset(train_parts)
    valid_dataset = ConcatDataset(valid_parts)
    test_dataset = ConcatDataset(test_parts)

    print(
        "Loaded balanced augmented synthetic datasets from "
        f"{data_root}: " + " | ".join(summaries),
        flush=True,
    )
    print(
        f"Combined split sizes: train={len(train_dataset)} "
        f"valid={len(valid_dataset)} test={len(test_dataset)} "
        f"augment={_env_flag('SYNTHETIC_AUGMENT', True)} "
        f"scale=[{_env_float('SYNTHETIC_SCALE_MIN', 0.85):.3f}, "
        f"{_env_float('SYNTHETIC_SCALE_MAX', 1.0):.3f}] "
        f"translate_pct={_env_float('SYNTHETIC_TRANSLATE_PCT', 0.05):.3f}",
        flush=True,
    )

    return (
        base_loader._make_loader(train_dataset, shuffle=True),
        base_loader._make_loader(valid_dataset, shuffle=False),
        base_loader._make_loader(test_dataset, shuffle=False),
    )
