"""Training preprocessing plus horizontal coordinates for source-image masks."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

import zero_shot_preprocessing as preprocessing
from zero_shot_geometry import _ink_bbox


def prepare_line(path, domain):
    processor = preprocessing.build_preprocessor(domain, training=False)
    with Image.open(path) as opened:
        original = opened.convert("RGB")
    processed = processor(original)
    # Reconstruct only the deterministic horizontal geometry, from the same
    # foreground detector and rounding rules as the training preprocessor.
    work = original.convert("L")
    if processor.autocontrast:
        work = ImageOps.autocontrast(work)
    crop_left = 0
    if processor.crop_foreground:
        work = ImageOps.autocontrast(work)
        mask = preprocessing._ink_mask(np.asarray(work, dtype=np.uint8))
        ys, xs = np.nonzero(mask)
        if xs.size >= 4 and ys.size >= 4:
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            dx = max(2, int(round((x1 - x0) * 0.025)))
            dy = max(2, int(round((y1 - y0) * 0.15)))
            crop_left = max(0, x0 - dx)
            work = work.crop((crop_left, max(0, y0 - dy), min(work.width, x1 + dx), min(work.height, y1 + dy)))
    height, width = processor.size
    if processor.preserve_aspect:
        work = ImageOps.autocontrast(work)
        if preprocessing._border_mean(np.asarray(work, dtype=np.uint8)) < 127.5:
            work = ImageOps.invert(work)
        _, y0, _, y1 = _ink_bbox(work)
        ratio = min(0.95, max(0.20, float(processor.target_ink_height_ratio)))
        desired = max(8, min(height - 2, int(round(height * ratio))))
        scale = desired / float(max(1, y1 - y0))
        new_width = min(width, max(1, int(round(work.width * scale))))
        offset = max(0, width - new_width) // 2
    else:
        new_width, offset = width, 0
    geometry = {
        "source_width": original.width, "source_height": original.height,
        "crop_left": crop_left, "crop_width": work.width,
        "scale_x": new_width / float(work.width), "offset_x": offset,
        "canvas_width": width, "canvas_height": height,
        "binarize": processor.binarize, "crop_foreground": processor.crop_foreground,
        "preserve_aspect": processor.preserve_aspect,
    }
    return processed, geometry


def source_intervals(intervals, geometry):
    result = []
    # Ignore model padding; it does not correspond to source pixels.
    left = geometry["offset_x"]
    right = left + geometry["crop_width"] * geometry["scale_x"]
    for start, end in intervals:
        start, end = max(left, start), min(right, end)
        if end > start:
            result.append([
                geometry["crop_left"] + (start - left) / geometry["scale_x"],
                geometry["crop_left"] + (end - left) / geometry["scale_x"],
            ])
    return result
