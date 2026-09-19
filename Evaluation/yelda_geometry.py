"""Full-image RGB or training preprocessing, with source-mask coordinates."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps

import zero_shot_preprocessing as preprocessing
from zero_shot_geometry import _ink_bbox


def _tight_foreground_crop_with_metadata(image: Image.Image):
    """Crop to the detected foreground exactly, with no artificial outer margin."""
    source = image.convert("RGB")
    gray_for_mask = ImageOps.autocontrast(source.convert("L"))
    gray = np.asarray(gray_for_mask, dtype=np.uint8)
    mask = preprocessing._ink_mask(gray)
    ys, xs = np.nonzero(mask)
    if xs.size < 4 or ys.size < 4:
        box = (0, 0, source.width, source.height)
    else:
        box = (
            int(xs.min()),
            int(ys.min()),
            int(xs.max()) + 1,
            int(ys.max()) + 1,
        )
    cropped = source.crop(box)
    metadata = {
        "source_width": int(source.width),
        "source_height": int(source.height),
        "crop_left": int(box[0]),
        "crop_top": int(box[1]),
        "crop_right": int(box[2]),
        "crop_bottom": int(box[3]),
        "crop_width": int(box[2] - box[0]),
        "crop_height": int(box[3] - box[1]),
        "crop_margin_x": 0.0,
        "crop_margin_y": 0.0,
    }
    return cropped, metadata


def _tight_resize_to_height(image: Image.Image, metadata: dict, target_height: int = 128):
    """Resize a cropped line to the model height without any surrounding canvas."""
    source = image.convert("RGB")
    scale = float(target_height) / max(1.0, float(source.height))
    new_width = max(32, int(round(float(source.width) * scale)))
    resized = source.resize((new_width, int(target_height)), Image.Resampling.BILINEAR)
    geometry = dict(metadata)
    geometry.update(
        {
            "scale_x": float(new_width / max(1, source.width)),
            "scale_y": float(target_height / max(1, source.height)),
            "resize_scale": float(scale),
            "resized_width": int(new_width),
            "resized_height": int(target_height),
            "offset_x": 0,
            "offset_y": 0,
            "canvas_width": int(new_width),
            "canvas_height": int(target_height),
            "binarize": False,
            "crop_foreground": True,
            "preserve_aspect": True,
            "artificial_padding": False,
            "autocontrast": False,
            "auto_invert": False,
            "image_preprocessing": "tight",
            "color_mode": "RGB",
        }
    )
    return resized, geometry


def _crop_resize_fixed_1024(
    image: Image.Image,
    metadata: dict,
    *,
    target_width: int = 1024,
    target_height: int = 128,
):
    """Resize the exact foreground crop directly to 1024x128 with no padding.

    This intentionally does NOT preserve aspect ratio. The evaluation contract
    is: crop outer whitespace first, then use the exact resolution seen by the
    trained model. No white side/top/bottom canvas is added after the crop.
    """
    source = image.convert("RGB")
    resized = source.resize(
        (int(target_width), int(target_height)),
        Image.Resampling.BILINEAR,
    )
    geometry = dict(metadata)
    geometry.update(
        {
            "scale_x": float(target_width / max(1, source.width)),
            "scale_y": float(target_height / max(1, source.height)),
            "resize_scale": None,
            "resized_width": int(target_width),
            "resized_height": int(target_height),
            "offset_x": 0,
            "offset_y": 0,
            "canvas_width": int(target_width),
            "canvas_height": int(target_height),
            "binarize": False,
            "crop_foreground": True,
            "preserve_aspect": False,
            "artificial_padding": False,
            "autocontrast": False,
            "auto_invert": False,
            "image_preprocessing": "cropped_1024",
            "color_mode": "RGB",
        }
    )
    return resized, geometry


def prepare_line(path, domain, image_preprocessing="original"):
    with Image.open(path) as opened:
        original = opened.convert("RGB")
    if image_preprocessing == "tight":
        cropped, metadata = _tight_foreground_crop_with_metadata(original)
        return _tight_resize_to_height(cropped, metadata, target_height=128)
    if image_preprocessing == "cropped_1024":
        cropped, metadata = _tight_foreground_crop_with_metadata(original)
        return _crop_resize_fixed_1024(
            cropped,
            metadata,
            target_width=1024,
            target_height=128,
        )
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
        raise ValueError(
            f"Unknown image preprocessing: {image_preprocessing}; "
            "use original, training, tight, or cropped_1024"
        )
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
