"""Geometry-preserving scan augmentation for real manuscript lines.

The augmentation is intentionally NON-GEOMETRIC:
  * no rotation
  * no translation
  * no scaling
  * no cropping
  * no warping

Only scan/appearance corruption is applied.  The input PIL mode is preserved,
so grayscale manuscript training remains truly one-channel.
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


def _mode(image: Image.Image) -> str:
    return "L" if image.mode == "L" else "RGB"


def gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    mode = _mode(image)
    return image.convert(mode).filter(
        ImageFilter.GaussianBlur(radius=max(0.0, float(radius)))
    )


def gaussian_luminance_noise(image: Image.Image, std: float) -> Image.Image:
    """Add Gaussian scan noise while preserving grayscale/RGB channel count."""
    mode = _mode(image)
    sigma = max(0.0, float(std))
    if sigma <= 0.0:
        return image.convert(mode)

    if mode == "L":
        pixels = np.asarray(image.convert("L"), dtype=np.float32)
        noise = np.random.normal(0.0, sigma, size=pixels.shape).astype(np.float32)
        pixels = np.clip(pixels + noise, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(pixels, mode="L")

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    # Same perturbation on all three channels: scan intensity noise, not random
    # colour noise.
    noise = np.random.normal(0.0, sigma, size=rgb.shape[:2]).astype(np.float32)
    rgb = np.clip(rgb + noise[..., None], 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def speckle_noise(image: Image.Image, fraction: float) -> Image.Image:
    """Sparse black/white dust/scan defects with unchanged geometry."""
    mode = _mode(image)
    frac = min(0.03, max(0.0, float(fraction)))
    if mode == "L":
        pixels = np.array(image.convert("L"), dtype=np.uint8, copy=True)
        count = int(round(pixels.size * frac))
        if count > 0:
            ys = np.random.randint(0, pixels.shape[0], size=count)
            xs = np.random.randint(0, pixels.shape[1], size=count)
            pixels[ys, xs] = np.random.choice([0, 255], size=count).astype(np.uint8)
        return Image.fromarray(pixels, mode="L")

    rgb = np.array(image.convert("RGB"), dtype=np.uint8, copy=True)
    count = int(round(rgb.shape[0] * rgb.shape[1] * frac))
    if count > 0:
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
    mode = _mode(image)
    out = ImageEnhance.Brightness(image.convert(mode)).enhance(
        max(0.1, float(brightness_factor))
    )
    return ImageEnhance.Contrast(out).enhance(max(0.1, float(contrast_factor)))


@dataclass
class ScanOnlyAugmentor:
    enabled: bool = False
    probability: float = 0.95
    brightness_delta: float = 0.12
    contrast_delta: float = 0.22
    blur_probability: float = 0.55
    blur_radius_min: float = 0.35
    blur_radius_max: float = 1.40
    gaussian_noise_probability: float = 0.75
    gaussian_noise_std_min: float = 5.0
    gaussian_noise_std_max: float = 16.0
    speckle_probability: float = 0.30
    speckle_fraction: float = 0.0010

    @classmethod
    def from_env(cls) -> "ScanOnlyAugmentor":
        return cls(
            enabled=_env_flag("REAL_SCAN_AUGMENT", False),
            probability=_env_float("REAL_SCAN_AUGMENT_PROB", 0.95),
            brightness_delta=_env_float("REAL_SCAN_BRIGHTNESS_DELTA", 0.12),
            contrast_delta=_env_float("REAL_SCAN_CONTRAST_DELTA", 0.22),
            blur_probability=_env_float("REAL_SCAN_BLUR_PROB", 0.55),
            blur_radius_min=_env_float("REAL_SCAN_BLUR_RADIUS_MIN", 0.35),
            blur_radius_max=_env_float("REAL_SCAN_BLUR_RADIUS_MAX", 1.40),
            gaussian_noise_probability=_env_float(
                "REAL_SCAN_GAUSSIAN_NOISE_PROB", 0.75
            ),
            gaussian_noise_std_min=_env_float(
                "REAL_SCAN_GAUSSIAN_NOISE_STD_MIN", 5.0
            ),
            gaussian_noise_std_max=_env_float(
                "REAL_SCAN_GAUSSIAN_NOISE_STD_MAX", 16.0
            ),
            speckle_probability=_env_float("REAL_SCAN_SPECKLE_PROB", 0.30),
            speckle_fraction=_env_float("REAL_SCAN_SPECKLE_FRACTION", 0.0010),
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        return self.augment_with_metadata(image)[0]

    def augment_with_metadata(self, image: Image.Image):
        source_mode = _mode(image)
        source = image.convert(source_mode)
        out = source.copy()
        metadata = {
            "enabled": bool(self.enabled),
            "source_mode": source_mode,
            "output_mode": source_mode,
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

        brightness = random.uniform(
            1.0 - max(0.0, self.brightness_delta),
            1.0 + max(0.0, self.brightness_delta),
        )
        contrast = random.uniform(
            1.0 - max(0.0, self.contrast_delta),
            1.0 + max(0.0, self.contrast_delta),
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

        if out.size != source.size or _mode(out) != source_mode:
            raise RuntimeError(
                "ScanOnlyAugmentor changed image geometry/channel contract: "
                f"mode {source_mode}->{out.mode}, size {source.size}->{out.size}"
            )
        return out, metadata
