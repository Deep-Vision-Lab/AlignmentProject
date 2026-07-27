"""Source-compatible geometry for synthetic-to-real line transfer.

The original synthetic pipeline feeds fixed 128x1024 line images to the model.
A fully aspect-preserving fit can make very long real lines extremely short in
height because the 1024-pixel width becomes the limiting dimension.  This
module keeps the actual ink at a stable target height and, only when necessary,
compresses the horizontal axis to the source-domain width.
"""
from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
from PIL import Image, ImageOps

try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    _BILINEAR = Image.BILINEAR


def _enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ink_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    """Return the foreground bbox after normalizing page polarity."""
    import zero_shot_preprocessing as base

    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    mask = base._ink_mask(gray)
    ys, xs = np.nonzero(mask)
    if xs.size < 4 or ys.size < 4:
        return 0, 0, image.width, image.height
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def source_compatible_pad(
    image: Image.Image,
    size=(128, 1024),
    target_ink_height_ratio=0.72,
    horizontal_jitter=0.0,
) -> Image.Image:
    """Normalize ink height while retaining the fixed synthetic canvas.

    Processing order:
      1. Normalize polarity so page background is white.
      2. Scale from the measured foreground height, not the cropped image height.
      3. Keep that vertical scale even for very long lines.
      4. If needed, compress only width to 1024, matching the fixed-width source.
      5. Center the foreground vertically and pad with white.

    The output is always ``(width=1024, height=128)`` by default.
    """
    import zero_shot_preprocessing as base

    target_h, target_w = map(int, size)
    work = ImageOps.autocontrast(image.convert("L"))
    gray = np.asarray(work, dtype=np.uint8)

    # Auto-inversion after white padding cannot reliably detect a dark source
    # background. Normalize polarity before geometry while the original border
    # is still available.
    if base._border_mean(gray) < 127.5:
        work = ImageOps.invert(work)

    x0, y0, x1, y1 = _ink_bbox(work)
    ink_h = max(1, y1 - y0)
    ratio = min(0.95, max(0.20, float(target_ink_height_ratio)))
    desired_ink_h = max(8, min(target_h - 2, int(round(target_h * ratio))))
    scale = desired_ink_h / float(ink_h)

    scaled_w = max(1, int(round(work.width * scale)))
    scaled_h = max(1, int(round(work.height * scale)))
    resized = work.resize((scaled_w, scaled_h), _BILINEAR)

    # Preserve the selected vertical scale. Long lines are compressed only in x
    # instead of shrinking both axes, which was the source of tiny real lines.
    if scaled_w > target_w:
        resized = resized.resize((target_w, scaled_h), _BILINEAR)
        new_w = target_w
    else:
        new_w = scaled_w

    canvas = Image.new("L", (target_w, target_h), color=255)
    max_x = max(0, target_w - new_w)
    centered_x = max_x // 2
    jitter = int(round(max_x * max(0.0, float(horizontal_jitter))))
    x = (
        min(max_x, max(0, centered_x + random.randint(-jitter, jitter)))
        if jitter
        else centered_x
    )

    # Scaling width after the first resize does not alter the vertical bbox.
    scaled_ink_center_y = 0.5 * (y0 + y1) * scale
    y = int(round(0.5 * target_h - scaled_ink_center_y))
    canvas.paste(resized, (x, y))
    return canvas


def install_source_compatible_geometry() -> bool:
    """Install the geometry in the shared train/evaluation preprocessor."""
    if not _enabled("ZERO_SHOT_SOURCE_GEOMETRY", True):
        return False

    import zero_shot_preprocessing as base

    if getattr(base, "_source_compatible_geometry_installed", False):
        return True

    base.aspect_preserving_pad = source_compatible_pad

    # train_optimized imports this function after sitecustomize runs, so the
    # resolved W&B/checkpoint config records the actual geometry used.
    original_config = base.zero_shot_config

    def zero_shot_config() -> dict[str, Any]:
        config = dict(original_config())
        config["zero_shot_geometry_mode"] = "source-compatible-height"
        config["zero_shot_source_geometry"] = True
        return config

    base.zero_shot_config = zero_shot_config
    base._source_compatible_geometry_installed = True
    return True
