"""Geometry-preserving scan augmentation for real manuscript lines.

This module deliberately contains NO rotation, translation, scaling, cropping,
warping, or stitching.  It changes only image appearance so spatial alignment
labels and window coordinates remain valid.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    return image.convert("RGB").filter(
        ImageFilter.GaussianBlur(radius=max(0.0, float(radius)))
    )


def gaussian_luminance_noise(image: Image.Image, std: float) -> Image.Image:
    """Add the same Gaussian perturbation to R/G/B, preserving paper colour."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    sigma = max(0.0, float(std))
    if sigma <= 0.0:
        return image.convert("RGB")
    noise = np.random.normal(0.0, sigma, size=rgb.shape[:2]).astype(np.float32)
    rgb = np.clip(rgb + noise[..., None], 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def speckle_noise(image: Image.Image, fraction: float) -> Image.Image:
    """Sparse dark/light scan defects without moving any pixel coordinates."""
    rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    frac = min(0.02, max(0.0, float(fraction)))
    count = int(round(rgb.shape[0] * rgb.shape[1] * frac))
    if count <= 0:
        return Image.fromarray(rgb, mode="RGB")
    ys = np.random.randint(0, rgb.shape[0], size=count)
    xs = np.random.randint(0, rgb.shape[1], size=count)
    values = np.random.choice([0, 255], size=count).astype(np.uint8)
    rgb[ys, xs, :] = values[:, None]
    return Image.fromarray(rgb, mode="RGB")


def adjust_brightness_contrast(
    image: Image.Image,
    *,
    brightness_factor: float = 1.0,
    contrast_factor: float = 1.0,
) -> Image.Image:
    out = ImageEnhance.Brightness(image.convert("RGB")).enhance(
        max(0.1, float(brightness_factor))
    )
    return ImageEnhance.Contrast(out).enhance(max(0.1, float(contrast_factor)))


@dataclass
class ScanOnlyAugmentor:
    enabled: bool = False
    probability: float = 0.90
    brightness_delta: float = 0.06
    contrast_delta: float = 0.10
    blur_probability: float = 0.35
    blur_radius_min: float = 0.15
    blur_radius_max: float = 0.90
    gaussian_noise_probability: float = 0.50
    gaussian_noise_std_min: float = 2.0
    gaussian_noise_std_max: float = 7.0
    speckle_probability: float = 0.20
    speckle_fraction: float = 0.0005

    @classmethod
    def from_env(cls) -> "ScanOnlyAugmentor":
        return cls(
            enabled=_env_flag("REAL_SCAN_AUGMENT", False),
            probability=_env_float("REAL_SCAN_AUGMENT_PROB", 0.90),
            brightness_delta=_env_float("REAL_SCAN_BRIGHTNESS_DELTA", 0.06),
            contrast_delta=_env_float("REAL_SCAN_CONTRAST_DELTA", 0.10),
            blur_probability=_env_float("REAL_SCAN_BLUR_PROB", 0.35),
            blur_radius_min=_env_float("REAL_SCAN_BLUR_RADIUS_MIN", 0.15),
            blur_radius_max=_env_float("REAL_SCAN_BLUR_RADIUS_MAX", 0.90),
            gaussian_noise_probability=_env_float(
                "REAL_SCAN_GAUSSIAN_NOISE_PROB", 0.50
            ),
            gaussian_noise_std_min=_env_float(
                "REAL_SCAN_GAUSSIAN_NOISE_STD_MIN", 2.0
            ),
            gaussian_noise_std_max=_env_float(
                "REAL_SCAN_GAUSSIAN_NOISE_STD_MAX", 7.0
            ),
            speckle_probability=_env_float("REAL_SCAN_SPECKLE_PROB", 0.20),
            speckle_fraction=_env_float("REAL_SCAN_SPECKLE_FRACTION", 0.0005),
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.augment_with_metadata(image)[0]

    def augment_with_metadata(self, image: Image.Image):
        source = image.convert("RGB")
        out = source.copy()
        metadata = {
            "enabled": bool(self.enabled),
            "source_size": list(source.size),
            "output_size": list(source.size),
            "geometry_changed": False,
            "rotation_degrees": 0.0,
            "translation_x": 0,
            "translation_y": 0,
            "scale": 1.0,
            "blur_radius": 0.0,
            "gaussian_noise_std": 0.0,
            "speckle_fraction": 0.0,
            "brightness_factor": 1.0,
            "contrast_factor": 1.0,
        }
        if not self.enabled or random.random() >= max(0.0, min(1.0, self.probability)):
            return out, metadata

        brightness = 1.0
        contrast = 1.0
        if self.brightness_delta > 0:
            brightness = random.uniform(
                1.0 - self.brightness_delta, 1.0 + self.brightness_delta
            )
        if self.contrast_delta > 0:
            contrast = random.uniform(
                1.0 - self.contrast_delta, 1.0 + self.contrast_delta
            )
        out = adjust_brightness_contrast(
            out,
            brightness_factor=brightness,
            contrast_factor=contrast,
        )
        metadata["brightness_factor"] = float(brightness)
        metadata["contrast_factor"] = float(contrast)

        if (
            self.blur_radius_max > 0
            and random.random() < max(0.0, min(1.0, self.blur_probability))
        ):
            lo = max(0.0, min(self.blur_radius_min, self.blur_radius_max))
            hi = max(lo, self.blur_radius_max)
            radius = random.uniform(lo, hi)
            out = gaussian_blur(out, radius)
            metadata["blur_radius"] = float(radius)

        if (
            self.gaussian_noise_std_max > 0
            and random.random()
            < max(0.0, min(1.0, self.gaussian_noise_probability))
        ):
            lo = max(
                0.0,
                min(self.gaussian_noise_std_min, self.gaussian_noise_std_max),
            )
            hi = max(lo, self.gaussian_noise_std_max)
            std = random.uniform(lo, hi)
            out = gaussian_luminance_noise(out, std)
            metadata["gaussian_noise_std"] = float(std)

        if (
            self.speckle_fraction > 0
            and random.random() < max(0.0, min(1.0, self.speckle_probability))
        ):
            out = speckle_noise(out, self.speckle_fraction)
            metadata["speckle_fraction"] = float(self.speckle_fraction)

        if out.size != source.size:
            raise RuntimeError(
                "ScanOnlyAugmentor changed geometry, which is forbidden: "
                f"{source.size} -> {out.size}"
            )
        return out, metadata
