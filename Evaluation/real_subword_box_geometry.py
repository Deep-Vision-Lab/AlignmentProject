"""Map bbox.json subword boxes through the exact evaluation crop/pad geometry."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from Evaluation.real_subword_box_json import load_json_annotations
from Evaluation.real_subword_box_metrics import (
    BoxAnnotations,
    SubwordBox,
    _fill_missing_text,
    _reading_order,
)
from zero_shot_preprocessing import _ink_mask


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _foreground_bounds(image: Image.Image) -> tuple[int, int, int, int]:
    gray_image = ImageOps.autocontrast(image.convert("L"))
    gray = np.asarray(gray_image, dtype=np.uint8)
    mask = _ink_mask(gray)
    ys, xs = np.nonzero(mask)
    if xs.size < 4 or ys.size < 4:
        return 0, 0, image.width, image.height
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    pad_x = max(2, int(round((x1 - x0) * 0.025)))
    pad_y = max(2, int(round((y1 - y0) * 0.15)))
    return (
        max(0, x0 - pad_x),
        max(0, y0 - pad_y),
        min(image.width, x1 + pad_x),
        min(image.height, y1 + pad_y),
    )


def _geometry(image: Image.Image, target_width: int, target_height: int):
    zero_shot = _flag("ZERO_SHOT_PREPROCESS", True)
    crop = zero_shot and _flag("ZERO_SHOT_FOREGROUND_CROP", True)
    preserve = zero_shot and _flag("ZERO_SHOT_PRESERVE_ASPECT", True)
    crop_box = _foreground_bounds(image) if crop else (0, 0, image.width, image.height)
    x0, y0, x1, y1 = crop_box
    crop_width = max(1, x1 - x0)
    crop_height = max(1, y1 - y0)

    if preserve:
        desired_height = max(
            8,
            int(round(target_height * _number("ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", 0.72))),
        )
        scale = min(desired_height / crop_height, target_width / crop_width)
        new_width = max(1, min(target_width, int(round(crop_width * scale))))
        new_height = max(1, min(target_height, int(round(crop_height * scale))))
        offset_x = max(0, target_width - new_width) // 2
        offset_y = max(0, target_height - new_height) // 2
        scale_x = new_width / crop_width
        scale_y = new_height / crop_height
    else:
        offset_x = offset_y = 0
        scale_x = target_width / crop_width
        scale_y = target_height / crop_height
    return crop_box, scale_x, scale_y, float(offset_x), float(offset_y)


def load_line_annotations(image_path, target_width: int, target_height: int) -> BoxAnnotations:
    """Load bbox.json boxes and apply the evaluation crop, resize and pad."""
    image_path = Path(image_path)
    annotations = load_json_annotations(image_path)
    if not annotations.boxes:
        return annotations

    coordinate_space = os.environ.get("REAL_BOX_COORDINATE_SPACE", "original").strip().lower()
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        source_width, source_height = image.size
        crop_box, scale_x, scale_y, offset_x, offset_y = _geometry(
            image, int(target_width), int(target_height)
        )
    crop_x0, crop_y0, _crop_x1, _crop_y1 = crop_box

    maximum = max(max(box.x1, box.y1) for box in annotations.boxes)
    if coordinate_space == "auto":
        coordinate_space = "normalized" if maximum <= 1.5 else "original"

    transformed = []
    for box in annotations.boxes:
        if coordinate_space == "normalized":
            source_x0, source_x1 = box.x0 * source_width, box.x1 * source_width
            source_y0, source_y1 = box.y0 * source_height, box.y1 * source_height
        elif coordinate_space == "model":
            mapped_x0, mapped_x1 = box.x0, box.x1
            mapped_y0, mapped_y1 = box.y0, box.y1
            source_x0 = source_x1 = source_y0 = source_y1 = None
        else:
            source_x0, source_x1 = box.x0, box.x1
            source_y0, source_y1 = box.y0, box.y1

        if coordinate_space != "model":
            mapped_x0 = (source_x0 - crop_x0) * scale_x + offset_x
            mapped_x1 = (source_x1 - crop_x0) * scale_x + offset_x
            mapped_y0 = (source_y0 - crop_y0) * scale_y + offset_y
            mapped_y1 = (source_y1 - crop_y0) * scale_y + offset_y

        x0 = max(0.0, min(float(target_width), mapped_x0))
        x1 = max(0.0, min(float(target_width), mapped_x1))
        y0 = max(0.0, min(float(target_height), mapped_y0))
        y1 = max(0.0, min(float(target_height), mapped_y1))
        if y1 <= y0:
            y0, y1 = 0.0, float(target_height)
        if x1 > x0:
            transformed.append(SubwordBox(box.text, x0, y0, x1, y1, box.source_row))

    transformed = _fill_missing_text(_reading_order(transformed), image_path)
    status, error = annotations.status, annotations.error
    if transformed and any(not box.text.strip() for box in transformed):
        status = "missing_text"
        error = "bbox.json text is missing and transcript subword count does not match box count"
    return BoxAnnotations(tuple(transformed), annotations.workbook, annotations.sheet, status, error)
