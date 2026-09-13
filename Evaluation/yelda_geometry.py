"""Full-image RGB or training preprocessing, with source-mask coordinates."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

import zero_shot_preprocessing as preprocessing
from zero_shot_geometry import _ink_bbox


def prepare_line(path, domain, image_preprocessing="original"):
    with Image.open(path) as opened:
        original = opened.convert("RGB")
    if image_preprocessing == "original":
        # Keep every source pixel in the field of view. Do not use the training
        # preprocessor: even binarize=False still grayscales/crops/scales ink.
        width, height = 1024, 128
        processed = (original.copy() if original.size == (width, height) else
                     original.resize((width, height), Image.Resampling.BILINEAR))
        return processed, {
            "image_preprocessing": "original", "color_mode": "RGB",
            "source_width": original.width, "source_height": original.height,
            "crop_left": 0, "crop_width": original.width,
            "scale_x": width / float(original.width), "offset_x": 0,
            "canvas_width": width, "canvas_height": height,
            "binarize": False, "crop_foreground": False, "preserve_aspect": False,
            "autocontrast": False, "auto_invert": False,
        }
    if image_preprocessing != "training":
        raise ValueError(f"Unknown image preprocessing: {image_preprocessing}")
    processor = preprocessing.build_preprocessor(domain, training=False)
    if hasattr(processor, "preprocess_with_metadata"):
        processed, geometry = processor.preprocess_with_metadata(original)
        geometry = dict(geometry)
        geometry["image_preprocessing"] = "training"
        geometry.setdefault("autocontrast", bool(processor.autocontrast))
        geometry.setdefault("auto_invert", bool(processor.auto_invert))
        return processed, geometry

    # Legacy fallback for old preprocessors.
    processed = processor(original)
    work = original.convert("L")
    crop_left = 0
    height, width = processor.size
    new_width, offset = width, 0
    geometry = {
        "image_preprocessing": "training",
        "source_width": original.width,
        "source_height": original.height,
        "crop_left": crop_left,
        "crop_width": work.width,
        "scale_x": new_width / float(max(1, work.width)),
        "offset_x": offset,
        "canvas_width": width,
        "canvas_height": height,
        "binarize": processor.binarize,
        "crop_foreground": processor.crop_foreground,
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
