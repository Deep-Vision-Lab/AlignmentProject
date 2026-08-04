"""Lightweight multi-region augmentation for Synthetic_Arabic_1..3.

The loader takes 3,000 unique raw samples from each source folder and exposes
2 fresh augmented copies per raw sample by default (18,000 training items per
epoch). Every training item uses one of three alignment-focused modes:

* cross-line injection (50%): target and donor fragments in distant regions;
* same-line two-region composition (35%): two distant fragments of one line;
* mild full-line augmentation (15%): scale, translation, and contrast only.

Validation and test samples stay disjoint and unaugmented.
"""
from __future__ import annotations

import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import ConcatDataset, DataLoader as TorchDataLoader, Dataset, Subset
from torchvision import transforms

import DataLoader as base_loader
from DataSet import TextLineModern

DEFAULT_SYNTHETIC_FOLDERS = tuple(f"Synthetic_Arabic_{i}" for i in range(1, 4))
_IMAGE_PATTERN = re.compile(r"^img1_(\d+)\.png$", re.IGNORECASE)
_TO_TENSOR = transforms.ToTensor()
_NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    _BILINEAR = Image.BILINEAR


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


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def synthetic_folder_names() -> tuple[str, ...]:
    raw = os.environ.get("SYNTHETIC_DATASET_FOLDERS", "").strip()
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    return names or DEFAULT_SYNTHETIC_FOLDERS


def resolve_synthetic_folders(data_dir, folder_names=None):
    names = tuple(folder_names or synthetic_folder_names())
    requested = Path(data_dir).expanduser()
    roots = []
    if os.environ.get("SYNTHETIC_DATA_ROOT"):
        roots.append(Path(os.environ["SYNTHETIC_DATA_ROOT"]).expanduser())
    roots.extend((requested, requested.parent))
    checked = []
    for root in roots:
        root = root.resolve()
        if str(root) in checked:
            continue
        checked.append(str(root))
        folders = tuple(root / name for name in names)
        if all((folder / "images").is_dir() and (folder / "texts").is_dir() for folder in folders):
            return root, folders
    raise FileNotFoundError(
        f"Could not find {names}, each with images/ and texts/. Checked: {checked}"
    )


def _border_level(image: Image.Image) -> int:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    bh, bw = max(1, h // 20), max(1, w // 100)
    border = np.concatenate(
        (gray[:bh].ravel(), gray[-bh:].ravel(), gray[:, :bw].ravel(), gray[:, -bw:].ravel())
    )
    return int(np.median(border)) if border.size else 255


def _background(*images: Image.Image) -> tuple[int, int, int]:
    level = int(np.median([_border_level(image) for image in images]))
    return level, level, level


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _wrap_text(text: str) -> str:
    text = " ".join(text.strip().split())
    return f" {text} " if text else " "


def _sample_count(folder: Path) -> int:
    indices = sorted(
        int(match.group(1))
        for path in (folder / "images").iterdir()
        if (match := _IMAGE_PATTERN.match(path.name))
    )
    if not indices:
        raise FileNotFoundError(f"No img1_*.png files in {folder / 'images'}")
    expected = list(range(1, indices[-1] + 1))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))[:10]
        raise ValueError(f"{folder.name} has non-contiguous img1 indices; missing {missing}")
    return indices[-1]


def _split_indices(total, train_count, valid_requested, test_requested, seed):
    if train_count <= 0 or train_count > total:
        raise ValueError(f"Invalid train count {train_count} for {total} samples")
    order = list(range(total))
    random.Random(seed).shuffle(order)
    train, remaining = order[:train_count], order[train_count:]
    if len(remaining) < 2:
        raise ValueError("At least two unused samples per folder are required for validation/test")
    valid_requested, test_requested = max(1, valid_requested), max(1, test_requested)
    if len(remaining) >= valid_requested + test_requested:
        valid_count, test_count = valid_requested, test_requested
    else:
        valid_count = min(len(remaining) - 1, max(1, len(remaining) // 2))
        test_count = len(remaining) - valid_count
    return train, remaining[:valid_count], remaining[valid_count : valid_count + test_count]


@dataclass(frozen=True)
class SyntheticRecord:
    folder_name: str
    sample_index: int
    image1_path: Path
    text1_path: Path
    image2_path: Optional[Path]
    text2_path: Optional[Path]

    @property
    def paired(self) -> bool:
        return self.image2_path is not None and self.text2_path is not None


def _make_record(folder: Path, zero_index: int) -> SyntheticRecord:
    index = int(zero_index) + 1
    image2 = folder / "images" / f"img2_{index}.png"
    text2 = folder / "texts" / f"text2_{index}.txt"
    paired = image2.is_file() and text2.is_file()
    return SyntheticRecord(
        folder.name,
        index,
        folder / "images" / f"img1_{index}.png",
        folder / "texts" / f"text1_{index}.txt",
        image2 if paired else None,
        text2 if paired else None,
    )


def _dataset_paths(folder: Path) -> dict[str, str]:
    return {
        "images": str(folder / "images"),
        "texts": str(folder / "texts"),
        "matrices": str(folder / "matrices"),
        "diffNWmatrices": str(folder / "diffNWmatrices"),
        "similarity_matrices": str(folder / "similarity_matrices"),
    }


def _build_eval_dataset(folder: Path, count: int):
    return TextLineModern(
        new_dataset=_dataset_paths(folder),
        transform=base_loader.synthetic_transform,
        num_samples_override=count,
    )


def _ink_bounds(image: Image.Image) -> tuple[int, int]:
    gray = np.asarray(image.convert("L"), dtype=np.int16)
    difference = np.abs(gray - _border_level(image))
    active = np.flatnonzero((difference > 18).mean(axis=0) > 0.01)
    if not active.size:
        return 0, image.width
    pad = max(2, image.width // 200)
    return max(0, int(active[0]) - pad), min(image.width, int(active[-1]) + pad + 1)


def _random_span(min_fraction: float, max_fraction: float) -> tuple[float, float]:
    width = random.uniform(min_fraction, max_fraction)
    start = random.uniform(0.0, max(0.0, 1.0 - width))
    return start, min(1.0, start + width)


def _two_spans(min_fraction, max_fraction, min_gap):
    for _ in range(100):
        spans = sorted((_random_span(min_fraction, max_fraction), _random_span(min_fraction, max_fraction)))
        if spans[1][0] - spans[0][1] >= min_gap:
            return spans
    width = min(max_fraction, max(min_fraction, (1.0 - min_gap) / 3.0))
    return [(0.02, 0.02 + width), (0.98 - width, 0.98)]


def _text_fragment(text: str, span: tuple[float, float]) -> str:
    text = text.strip()
    if not text:
        return ""
    start = min(len(text) - 1, int(math.floor(span[0] * len(text))))
    end = min(len(text), max(start + 1, int(math.ceil(span[1] * len(text)))))
    return text[start:end].strip() or text[start : start + 1]


def _crop_span(image: Image.Image, span, rtl=True) -> Image.Image:
    left, right = _ink_bounds(image)
    width = max(1, right - left)
    if rtl:
        x1 = left + int((1.0 - span[1]) * width)
        x2 = left + int((1.0 - span[0]) * width)
    else:
        x1 = left + int(span[0] * width)
        x2 = left + int(span[1] * width)
    x1, x2 = max(0, x1), min(image.width, max(x1 + 1, x2))
    return image.crop((x1, 0, x2, image.height))


def _fit_patch(patch: Image.Image, target_height: int, max_width: int) -> Image.Image:
    scale = target_height / max(1, patch.height)
    width, height = max(1, int(round(patch.width * scale))), target_height
    if width > max_width:
        scale = max_width / width
        width, height = max_width, max(1, int(round(height * scale)))
    return patch.resize((width, height), _BILINEAR)


def compose_separated_regions(first, second, size=(128, 1024), rtl=True, min_gap_fraction=0.18):
    """Place two fragments in distant reading-order regions of one canvas."""
    height, width = int(size[0]), int(size[1])
    canvas = Image.new("RGB", (width, height), _background(first, second))
    patch_height = random.randint(int(height * 0.70), int(height * 0.92))
    first = _fit_patch(first, patch_height, int(width * 0.34))
    second = _fit_patch(second, patch_height, int(width * 0.34))
    margin, gap = int(width * 0.04), int(width * min_gap_fraction)
    usable = width - 2 * margin - gap
    if first.width + second.width > usable:
        factor = usable / max(1, first.width + second.width)
        first = first.resize((max(1, int(first.width * factor)), max(1, int(first.height * factor))), _BILINEAR)
        second = second.resize((max(1, int(second.width * factor)), max(1, int(second.height * factor))), _BILINEAR)
    right_pos = (width - margin - first.width, random.randint(0, height - first.height))
    left_pos = (margin, random.randint(0, height - second.height))
    positions = (right_pos, left_pos) if rtl else (left_pos, right_pos)
    canvas.paste(first, positions[0])
    canvas.paste(second, positions[1])
    return canvas


class LightweightLineAugment:
    """Only mild scale, translation, and contrast changes."""

    def __init__(self):
        self.height = _env_int("SYNTHETIC_IMAGE_HEIGHT", 128)
        self.width = _env_int("SYNTHETIC_IMAGE_WIDTH", 1024)
        self.scale_min = _env_float("SYNTHETIC_SCALE_MIN", 0.90)
        self.scale_max = _env_float("SYNTHETIC_SCALE_MAX", 1.00)
        self.translate_pct = _env_float("SYNTHETIC_TRANSLATE_PCT", 0.04)
        self.contrast = _env_float("SYNTHETIC_CONTRAST", 0.10)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB").resize((self.width, self.height), _BILINEAR)
        background = _background(image)
        scale = random.uniform(self.scale_min, self.scale_max)
        scaled = image.resize(
            (max(1, int(round(self.width * scale))), max(1, int(round(self.height * scale)))),
            _BILINEAR,
        )
        max_dx = int(round(self.width * self.translate_pct))
        max_dy = int(round(self.height * self.translate_pct))
        left = (self.width - scaled.width) // 2 + random.randint(-max_dx, max_dx)
        top = (self.height - scaled.height) // 2 + random.randint(-max_dy, max_dy)
        canvas = Image.new("RGB", (self.width, self.height), background)
        src_left, src_top = max(0, -left), max(0, -top)
        dst_left, dst_top = max(0, left), max(0, top)
        paste_w = min(scaled.width - src_left, self.width - dst_left)
        paste_h = min(scaled.height - src_top, self.height - dst_top)
        if paste_w > 0 and paste_h > 0:
            patch = scaled.crop((src_left, src_top, src_left + paste_w, src_top + paste_h))
            canvas.paste(patch, (dst_left, dst_top))
        amount = max(0.0, self.contrast)
        return ImageEnhance.Contrast(canvas).enhance(random.uniform(1.0 - amount, 1.0 + amount))


class SyntheticPairAugmentor:
    MODES = ("full_line", "two_region", "cross_injection")

    def __init__(self):
        self.appearance = LightweightLineAugment()
        self.injection_probability = _env_float("SYNTHETIC_INJECTION_PROB", 0.50)
        self.two_region_probability = _env_float("SYNTHETIC_TWO_REGION_PROB", 0.35)
        if self.injection_probability + self.two_region_probability > 1.0:
            raise ValueError("Injection + two-region probability cannot exceed 1")
        self.min_fraction = _env_float("SYNTHETIC_FRAGMENT_MIN_FRACTION", 0.12)
        self.max_fraction = _env_float("SYNTHETIC_FRAGMENT_MAX_FRACTION", 0.32)
        self.source_gap = _env_float("SYNTHETIC_SOURCE_GAP_FRACTION", 0.08)
        self.canvas_gap = _env_float("SYNTHETIC_CANVAS_GAP_FRACTION", 0.18)
        self.rtl = _env_flag("SYNTHETIC_TEXT_IS_RTL", True)

    @classmethod
    def from_env(cls):
        return cls()

    def choose_mode(self) -> str:
        draw = random.random()
        if draw < self.injection_probability:
            return "cross_injection"
        if draw < self.injection_probability + self.two_region_probability:
            return "two_region"
        return "full_line"

    def _compose(self, image_a, image_b, text_a, text_b, span_a, span_b):
        image = compose_separated_regions(
            _crop_span(image_a, span_a, self.rtl),
            _crop_span(image_b, span_b, self.rtl),
            size=(self.appearance.height, self.appearance.width),
            rtl=self.rtl,
            min_gap_fraction=self.canvas_gap,
        )
        text = _wrap_text(f"{_text_fragment(text_a, span_a)} {_text_fragment(text_b, span_b)}")
        return image, text

    def apply(self, target_images, target_texts, donor_images=None, donor_texts=None, mode=None):
        mode = mode or self.choose_mode()
        if mode not in self.MODES:
            raise ValueError(f"Unknown augmentation mode: {mode}")
        image1, image2 = target_images
        text1, text2 = target_texts
        output1, output2 = image1, image2
        out_text1 = _wrap_text(text1)
        out_text2 = _wrap_text(text2) if text2 else None
        if mode == "two_region":
            span_a, span_b = _two_spans(self.min_fraction, self.max_fraction, self.source_gap)
            output1, out_text1 = self._compose(image1, image1, text1, text1, span_a, span_b)
            if image2 is not None and text2 is not None:
                output2, out_text2 = self._compose(image2, image2, text2, text2, span_a, span_b)
        elif mode == "cross_injection":
            if donor_images is None or donor_texts is None:
                raise ValueError("cross_injection requires a donor sample")
            span_a = _random_span(self.min_fraction, self.max_fraction)
            span_b = _random_span(self.min_fraction, self.max_fraction)
            output1, out_text1 = self._compose(image1, donor_images[0], text1, donor_texts[0], span_a, span_b)
            if image2 is not None and text2 is not None:
                donor_image2 = donor_images[1] if donor_images[1] is not None else donor_images[0]
                donor_text2 = donor_texts[1] if donor_texts[1] else donor_texts[0]
                output2, out_text2 = self._compose(image2, donor_image2, text2, donor_text2, span_a, span_b)
        return {
            "mode": mode,
            "image1": self.appearance(output1),
            "text1": out_text1,
            "image2": self.appearance(output2) if output2 is not None else None,
            "text2": out_text2,
        }


def _to_tensor(image: Image.Image) -> torch.Tensor:
    return _NORMALIZE(_TO_TENSOR(image))


class ExpandedAugmentedSyntheticDataset(Dataset):
    def __init__(self, records: Sequence[SyntheticRecord], copies_per_sample: int, augmentor=None):
        if not records or copies_per_sample <= 0:
            raise ValueError("A non-empty record set and positive copy count are required")
        self.records = tuple(records)
        self.copies_per_sample = int(copies_per_sample)
        self.augmentor = augmentor or SyntheticPairAugmentor.from_env()

    @property
    def base_sample_count(self):
        return len(self.records)

    def __len__(self):
        return len(self.records) * self.copies_per_sample

    @staticmethod
    def _load(record):
        return (
            (_load_rgb(record.image1_path), _load_rgb(record.image2_path) if record.image2_path else None),
            (_read_text(record.text1_path), _read_text(record.text2_path) if record.text2_path else None),
        )

    def __getitem__(self, index):
        base_index = int(index) % len(self.records)
        target_images, target_texts = self._load(self.records[base_index])
        mode = self.augmentor.choose_mode()
        donor_images = donor_texts = None
        if mode == "cross_injection":
            donor_index = random.randrange(len(self.records) - 1) if len(self.records) > 1 else 0
            if len(self.records) > 1 and donor_index >= base_index:
                donor_index += 1
            donor_images, donor_texts = self._load(self.records[donor_index])
        result = self.augmentor.apply(target_images, target_texts, donor_images, donor_texts, mode)
        if result["image2"] is None:
            return result["text1"], _to_tensor(result["image1"])
        return {
            "text1": result["text1"],
            "image1": _to_tensor(result["image1"]),
            "text2": result["text2"],
            "image2": _to_tensor(result["image2"]),
        }


def _make_loader(dataset, shuffle):
    factory = getattr(base_loader, "_make_loader", None)
    if callable(factory):
        return factory(dataset, shuffle=shuffle)
    return TorchDataLoader(
        dataset,
        batch_size=getattr(base_loader, "batch_size", 32),
        shuffle=shuffle,
        collate_fn=getattr(base_loader, "custom_collate_fn", None),
        num_workers=_env_int("DATALOADER_NUM_WORKERS", 0),
        pin_memory=torch.cuda.is_available(),
    )


def build_dataloaders(data_dir=None):
    """Default train size: 3 folders x 3,000 raw x 2 copies = 18,000/epoch."""
    data_dir = data_dir or "DataSet"
    root, folders = resolve_synthetic_folders(data_dir)
    train_count = _env_int("SYNTHETIC_TRAIN_SAMPLES_PER_FOLDER", 3000)
    valid_count = _env_int("SYNTHETIC_VALID_SAMPLES_PER_FOLDER", 500)
    test_count = _env_int("SYNTHETIC_TEST_SAMPLES_PER_FOLDER", 500)
    copies = _env_int("SYNTHETIC_AUGMENT_COPIES_PER_SAMPLE", 2)
    seed = _env_int("SYNTHETIC_SPLIT_SEED", _env_int("DATASET_SPLIT_SEED", 42))
    train_records, valid_parts, test_parts, summaries = [], [], [], []
    for offset, folder in enumerate(folders):
        count = _sample_count(folder)
        train, valid, test = _split_indices(count, train_count, valid_count, test_count, seed + offset)
        train_records.extend(_make_record(folder, index) for index in train)
        eval_dataset = _build_eval_dataset(folder, count)
        valid_parts.append(Subset(eval_dataset, valid))
        test_parts.append(Subset(eval_dataset, test))
        summaries.append(f"{folder.name}: train={len(train)} valid={len(valid)} test={len(test)}")
    augmentor = SyntheticPairAugmentor.from_env()
    train_dataset = ExpandedAugmentedSyntheticDataset(train_records, copies, augmentor)
    valid_dataset, test_dataset = ConcatDataset(valid_parts), ConcatDataset(test_parts)
    print(f"Loaded Synthetic Arabic 1-3 from {root}: " + " | ".join(summaries), flush=True)
    print(
        f"Augmented train: raw={len(train_records)} copies={copies} virtual={len(train_dataset)} "
        f"cross_injection={augmentor.injection_probability:.2f} "
        f"two_region={augmentor.two_region_probability:.2f} "
        f"full_line={1.0 - augmentor.injection_probability - augmentor.two_region_probability:.2f}",
        flush=True,
    )
    return (
        _make_loader(train_dataset, shuffle=True),
        _make_loader(valid_dataset, shuffle=False),
        _make_loader(test_dataset, shuffle=False),
    )
