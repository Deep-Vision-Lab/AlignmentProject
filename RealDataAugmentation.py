"""Training-only augmentations for the real Arabic line-pair dataset.

The main augmentation is a paired, RTL-aware line stitch. Two positive manifest
samples are concatenated in the same logical order on both A and B sides:

    text:  first + second
    image: [second | gap | first]

The physical image order is reversed because Arabic is read right-to-left. This
keeps image/text order consistent after the model's Arabic flip. Partner samples
are selected only from the training split, with preference for adjacent lines from
the same page pair and then lines from the same surah.

All augmentations in this module are training-only. Validation and test samples
continue to use the deterministic real-data preprocessing from DataLoader.py.
"""
from __future__ import annotations

import math
import os
import random
from typing import Iterable, Optional, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset


try:
    _BILINEAR = Image.Resampling.BILINEAR
    _BICUBIC = Image.Resampling.BICUBIC
except AttributeError:  # Pillow < 9
    _BILINEAR = Image.BILINEAR
    _BICUBIC = Image.BICUBIC

try:
    _AFFINE = Image.Transform.AFFINE
except AttributeError:  # Pillow < 9
    _AFFINE = Image.AFFINE


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


def _bounded(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _strip_text_boundaries(text: str) -> str:
    return " ".join(str(text).strip().split())


def _with_text_boundaries(text: str) -> str:
    return " " + _strip_text_boundaries(text) + " "


def _crop_white_margins(
    image: Image.Image,
    threshold: int = 245,
    padding: int = 3,
) -> Image.Image:
    """Remove large empty margins without cutting dark handwriting pixels."""
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    foreground = gray < int(threshold)
    if not foreground.any():
        return image.convert("RGB")

    ys, xs = np.nonzero(foreground)
    left = max(0, int(xs.min()) - int(padding))
    right = min(image.width, int(xs.max()) + int(padding) + 1)
    top = max(0, int(ys.min()) - int(padding))
    bottom = min(image.height, int(ys.max()) + int(padding) + 1)
    return image.convert("RGB").crop((left, top, right, bottom))


def _resize_to_height(image: Image.Image, target_height: int) -> Image.Image:
    target_height = max(1, int(target_height))
    scale = target_height / max(1, image.height)
    target_width = max(1, int(round(image.width * scale)))
    return image.resize((target_width, target_height), _BILINEAR)


def stitch_rtl_lines(
    first: Image.Image,
    second: Image.Image,
    gap_ratio_min: float = 0.08,
    gap_ratio_max: float = 0.18,
    vertical_jitter_ratio: float = 0.025,
) -> Image.Image:
    """Join two logical Arabic segments while preserving RTL visual order.

    ``first`` is the first text segment and therefore appears on the RIGHT side
    of the produced image. ``second`` is placed on the LEFT.
    """
    first = _crop_white_margins(first)
    second = _crop_white_margins(second)

    target_height = max(24, first.height, second.height)
    first = _resize_to_height(first, target_height)
    second = _resize_to_height(second, target_height)

    gap_low = max(2, int(round(target_height * max(0.0, gap_ratio_min))))
    gap_high = max(
        gap_low,
        int(round(target_height * max(gap_ratio_min, gap_ratio_max))),
    )
    gap = random.randint(gap_low, gap_high)

    jitter = max(0, int(round(target_height * max(0.0, vertical_jitter_ratio))))
    y_first = random.randint(0, 2 * jitter) if jitter else 0
    y_second = random.randint(0, 2 * jitter) if jitter else 0
    canvas_height = target_height + 2 * jitter
    canvas_width = second.width + gap + first.width
    canvas = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

    # Arabic logical order: first segment on the right, second on the left.
    canvas.paste(second, (0, y_second))
    canvas.paste(first, (second.width + gap, y_first))
    return canvas


class BinaryInkAugment:
    """Light post-binarization ink perturbations.

    Black ink is dilated with ``MinFilter`` and eroded with ``MaxFilter``. A very
    small amount of black/white speckle can also imitate scan defects.
    """

    def __init__(
        self,
        enabled: bool = True,
        morphology_probability: float = 0.25,
        speckle_probability: float = 0.12,
        speckle_fraction: float = 0.0006,
    ):
        self.enabled = bool(enabled)
        self.morphology_probability = _bounded(
            morphology_probability,
            0.0,
            1.0,
        )
        self.speckle_probability = _bounded(speckle_probability, 0.0, 1.0)
        self.speckle_fraction = _bounded(speckle_fraction, 0.0, 0.02)

    @classmethod
    def from_env(cls) -> "BinaryInkAugment":
        return cls(
            enabled=_env_flag("REAL_AUGMENT", True),
            morphology_probability=_env_float("REAL_AUG_MORPH_PROB", 0.25),
            speckle_probability=_env_float("REAL_AUG_SPECKLE_PROB", 0.12),
            speckle_fraction=_env_float(
                "REAL_AUG_SPECKLE_FRACTION",
                0.0006,
            ),
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        if not self.enabled:
            return image.convert("RGB")

        gray = image.convert("L")
        if random.random() < self.morphology_probability:
            # Equal probability of slightly thicker or thinner handwriting.
            filter_cls = (
                ImageFilter.MinFilter
                if random.random() < 0.5
                else ImageFilter.MaxFilter
            )
            gray = gray.filter(filter_cls(3))

        if (
            random.random() < self.speckle_probability
            and self.speckle_fraction > 0
        ):
            pixels = np.array(gray, dtype=np.uint8, copy=True)
            count = max(1, int(round(pixels.size * self.speckle_fraction)))
            flat_indices = np.random.randint(0, pixels.size, size=count)
            flat = pixels.reshape(-1)
            half = count // 2
            flat[flat_indices[:half]] = 0
            flat[flat_indices[half:]] = 255
            gray = Image.fromarray(pixels, mode="L")

        return gray.convert("RGB")


class RealLinePairAugmentor:
    """Smart paired-line stitching plus conservative scan augmentations."""

    def __init__(
        self,
        enabled: bool = True,
        appearance_probability: float = 0.85,
        stitch_probability: float = 0.25,
        stitch_pool_size: int = 32,
        stitch_max_text_chars: int = 120,
        prefer_adjacent: bool = True,
        rotation_degrees: float = 1.25,
        translate_x_ratio: float = 0.012,
        translate_y_ratio: float = 0.035,
        brightness_delta: float = 0.12,
        contrast_delta: float = 0.18,
        blur_probability: float = 0.15,
        blur_radius_max: float = 0.8,
        noise_probability: float = 0.18,
        noise_std: float = 5.0,
        gap_ratio_min: float = 0.08,
        gap_ratio_max: float = 0.18,
    ):
        self.enabled = bool(enabled)
        self.appearance_probability = _bounded(
            appearance_probability,
            0.0,
            1.0,
        )
        self.stitch_probability = _bounded(stitch_probability, 0.0, 1.0)
        self.stitch_pool_size = max(1, int(stitch_pool_size))
        self.stitch_max_text_chars = max(8, int(stitch_max_text_chars))
        self.prefer_adjacent = bool(prefer_adjacent)
        self.rotation_degrees = max(0.0, float(rotation_degrees))
        self.translate_x_ratio = max(0.0, float(translate_x_ratio))
        self.translate_y_ratio = max(0.0, float(translate_y_ratio))
        self.brightness_delta = _bounded(brightness_delta, 0.0, 0.8)
        self.contrast_delta = _bounded(contrast_delta, 0.0, 0.8)
        self.blur_probability = _bounded(blur_probability, 0.0, 1.0)
        self.blur_radius_max = max(0.0, float(blur_radius_max))
        self.noise_probability = _bounded(noise_probability, 0.0, 1.0)
        self.noise_std = max(0.0, float(noise_std))
        self.gap_ratio_min = max(0.0, float(gap_ratio_min))
        self.gap_ratio_max = max(self.gap_ratio_min, float(gap_ratio_max))

    @classmethod
    def from_env(cls) -> "RealLinePairAugmentor":
        return cls(
            enabled=_env_flag("REAL_AUGMENT", True),
            appearance_probability=_env_float(
                "REAL_AUG_APPEARANCE_PROB",
                0.85,
            ),
            stitch_probability=_env_float("REAL_AUG_STITCH_PROB", 0.25),
            stitch_pool_size=_env_int("REAL_AUG_STITCH_POOL_SIZE", 32),
            stitch_max_text_chars=_env_int(
                "REAL_AUG_STITCH_MAX_TEXT_CHARS",
                120,
            ),
            prefer_adjacent=_env_flag(
                "REAL_AUG_STITCH_PREFER_ADJACENT",
                True,
            ),
            rotation_degrees=_env_float("REAL_AUG_ROTATE_DEG", 1.25),
            translate_x_ratio=_env_float("REAL_AUG_TRANSLATE_X", 0.012),
            translate_y_ratio=_env_float("REAL_AUG_TRANSLATE_Y", 0.035),
            brightness_delta=_env_float("REAL_AUG_BRIGHTNESS", 0.12),
            contrast_delta=_env_float("REAL_AUG_CONTRAST", 0.18),
            blur_probability=_env_float("REAL_AUG_BLUR_PROB", 0.15),
            blur_radius_max=_env_float("REAL_AUG_BLUR_RADIUS", 0.8),
            noise_probability=_env_float("REAL_AUG_NOISE_PROB", 0.18),
            noise_std=_env_float("REAL_AUG_NOISE_STD", 5.0),
            gap_ratio_min=_env_float("REAL_AUG_STITCH_GAP_MIN", 0.08),
            gap_ratio_max=_env_float("REAL_AUG_STITCH_GAP_MAX", 0.18),
        )

    def should_stitch(self) -> bool:
        return self.enabled and random.random() < self.stitch_probability

    @staticmethod
    def _side_line_index(sample: dict, side: str) -> int:
        try:
            return int((sample.get(side) or {}).get("line_idx", -10_000))
        except (TypeError, ValueError):
            return -10_000

    @staticmethod
    def _surah_id(sample: dict):
        return sample.get("surah_number", sample.get("surah_name"))

    def select_partner(
        self,
        base_dataset,
        sample_index: int,
        candidate_indices: Sequence[int],
        read_text,
    ) -> Optional[int]:
        """Choose a compatible partner only from the current training split."""
        if not self.enabled or len(candidate_indices) <= 1:
            return None

        candidates = [
            int(index)
            for index in candidate_indices
            if int(index) != int(sample_index)
        ]
        if not candidates:
            return None
        if len(candidates) > self.stitch_pool_size:
            candidates = random.sample(candidates, self.stitch_pool_size)

        current = base_dataset.samples[int(sample_index)]
        current_text_a = _strip_text_boundaries(
            read_text(current["A"][base_dataset.text_key])
        )
        current_text_b = _strip_text_boundaries(
            read_text(current["B"][base_dataset.text_key])
        )
        current_average = 0.5 * (len(current_text_a) + len(current_text_b))

        scored = []
        for candidate_index in candidates:
            candidate = base_dataset.samples[candidate_index]
            candidate_text_a = _strip_text_boundaries(
                read_text(candidate["A"][base_dataset.text_key])
            )
            candidate_text_b = _strip_text_boundaries(
                read_text(candidate["B"][base_dataset.text_key])
            )

            if (
                len(current_text_a) + 1 + len(candidate_text_a)
                > self.stitch_max_text_chars
                or len(current_text_b) + 1 + len(candidate_text_b)
                > self.stitch_max_text_chars
            ):
                continue

            candidate_average = 0.5 * (
                len(candidate_text_a) + len(candidate_text_b)
            )
            length_penalty = abs(
                math.log(
                    (candidate_average + 1.0) / (current_average + 1.0)
                )
            )
            pair_balance = abs(
                len(candidate_text_a) - len(candidate_text_b)
            ) / max(1.0, candidate_average)

            same_pair = str(candidate.get("pair_id", "")) == str(
                current.get("pair_id", "")
            )
            same_surah = self._surah_id(candidate) == self._surah_id(current)
            line_gap_a = abs(
                self._side_line_index(candidate, "A")
                - self._side_line_index(current, "A")
            )
            line_gap_b = abs(
                self._side_line_index(candidate, "B")
                - self._side_line_index(current, "B")
            )
            adjacent = same_pair and line_gap_a == 1 and line_gap_b == 1

            # Lower is better. Adjacent lines from the same paired pages are the
            # most natural long-line samples. Same-surah samples are next best.
            relation_penalty = 2.0
            if same_surah:
                relation_penalty = 0.8
            if same_pair:
                relation_penalty = 0.25
            if self.prefer_adjacent and adjacent:
                relation_penalty = -1.5

            score = (
                relation_penalty
                + 0.65 * length_penalty
                + 0.35 * pair_balance
                + random.uniform(0.0, 0.08)
            )
            scored.append((score, candidate_index))

        if not scored:
            return None
        scored.sort(key=lambda item: item[0])
        shortlist = scored[: min(4, len(scored))]
        return int(random.choice(shortlist)[1])

    def stitch_pair(
        self,
        image1: Image.Image,
        image2: Image.Image,
        text1: str,
        text2: str,
        partner_image1: Image.Image,
        partner_image2: Image.Image,
        partner_text1: str,
        partner_text2: str,
    ):
        stitched1 = stitch_rtl_lines(
            image1,
            partner_image1,
            gap_ratio_min=self.gap_ratio_min,
            gap_ratio_max=self.gap_ratio_max,
        )
        stitched2 = stitch_rtl_lines(
            image2,
            partner_image2,
            gap_ratio_min=self.gap_ratio_min,
            gap_ratio_max=self.gap_ratio_max,
        )
        return (
            stitched1,
            stitched2,
            _strip_text_boundaries(text1)
            + " "
            + _strip_text_boundaries(partner_text1),
            _strip_text_boundaries(text2)
            + " "
            + _strip_text_boundaries(partner_text2),
        )

    def augment_appearance(self, image: Image.Image) -> Image.Image:
        """Apply conservative scan-like perturbations before binarization."""
        image = image.convert("RGB")
        if (
            not self.enabled
            or random.random() >= self.appearance_probability
        ):
            return image

        if self.brightness_delta > 0:
            factor = random.uniform(
                1.0 - self.brightness_delta,
                1.0 + self.brightness_delta,
            )
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast_delta > 0:
            factor = random.uniform(
                1.0 - self.contrast_delta,
                1.0 + self.contrast_delta,
            )
            image = ImageEnhance.Contrast(image).enhance(factor)

        if self.rotation_degrees > 0:
            angle = random.uniform(
                -self.rotation_degrees,
                self.rotation_degrees,
            )
            image = image.rotate(
                angle,
                resample=_BICUBIC,
                expand=False,
                fillcolor=(255, 255, 255),
            )

        max_dx = int(round(image.width * self.translate_x_ratio))
        max_dy = int(round(image.height * self.translate_y_ratio))
        dx = random.randint(-max_dx, max_dx) if max_dx > 0 else 0
        dy = random.randint(-max_dy, max_dy) if max_dy > 0 else 0
        if dx or dy:
            image = image.transform(
                image.size,
                _AFFINE,
                (1.0, 0.0, -dx, 0.0, 1.0, -dy),
                resample=_BILINEAR,
                fillcolor=(255, 255, 255),
            )

        if (
            self.blur_radius_max > 0
            and random.random() < self.blur_probability
        ):
            radius = random.uniform(0.1, self.blur_radius_max)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))

        if self.noise_std > 0 and random.random() < self.noise_probability:
            pixels = np.asarray(image.convert("L"), dtype=np.float32)
            noise = np.random.normal(
                0.0,
                self.noise_std,
                size=pixels.shape,
            )
            pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
            image = Image.fromarray(pixels, mode="L").convert("RGB")

        return image


class AugmentedRealSubset(Dataset):
    """Training subset that reads raw manifest images before augmentation."""

    def __init__(
        self,
        base_dataset,
        indices: Iterable[int],
        transform,
        augmentor: RealLinePairAugmentor,
    ):
        self.base_dataset = base_dataset
        self.indices = [int(index) for index in indices]
        self.transform = transform
        self.augmentor = augmentor
        self._text_cache = {}

    def __len__(self) -> int:
        return len(self.indices)

    def _read_text(self, path_value) -> str:
        key = str(path_value)
        if key not in self._text_cache:
            self._text_cache[key] = self.base_dataset._read_text(
                path_value
            ).strip()
        return self._text_cache[key]

    def _read_image(self, path_value) -> Image.Image:
        path = self.base_dataset._resolve(path_value)
        with Image.open(path) as image:
            return image.convert("RGB").copy()

    def _load_raw_sample(self, sample_index: int):
        sample = self.base_dataset.samples[int(sample_index)]
        side_a = sample["A"]
        side_b = sample["B"]
        return (
            self._read_image(side_a["line_image_path"]),
            self._read_image(side_b["line_image_path"]),
            self._read_text(side_a[self.base_dataset.text_key]),
            self._read_text(side_b[self.base_dataset.text_key]),
            sample,
        )

    def __getitem__(self, local_index: int):
        sample_index = self.indices[int(local_index)]
        image1, image2, text1, text2, sample = self._load_raw_sample(
            sample_index
        )
        partner_index = None

        if self.base_dataset.paired and self.augmentor.should_stitch():
            partner_index = self.augmentor.select_partner(
                self.base_dataset,
                sample_index,
                self.indices,
                self._read_text,
            )
            if partner_index is not None:
                (
                    partner_image1,
                    partner_image2,
                    partner_text1,
                    partner_text2,
                    _partner,
                ) = self._load_raw_sample(partner_index)
                image1, image2, text1, text2 = self.augmentor.stitch_pair(
                    image1,
                    image2,
                    text1,
                    text2,
                    partner_image1,
                    partner_image2,
                    partner_text1,
                    partner_text2,
                )

        image1 = self.transform(self.augmentor.augment_appearance(image1))
        text1 = _with_text_boundaries(text1)

        if not self.base_dataset.paired:
            return text1, image1

        image2 = self.transform(self.augmentor.augment_appearance(image2))
        text2 = _with_text_boundaries(text2)
        scores = sample.get("scores") or {}
        return {
            "text1": text1,
            "image1": image1,
            "text2": text2,
            "image2": image2,
            "pair_id": str(sample.get("pair_id", sample_index)),
            "label_type": str(sample.get("label_type", "")),
            "text_score": float(scores.get("text_score", 0.0)),
            "avg_sim": float(scores.get("avg_sim", 0.0)),
            "augmentation": (
                "rtl_stitch" if partner_index is not None else "appearance"
            ),
            "stitch_partner_index": (
                int(partner_index) if partner_index is not None else -1
            ),
        }
