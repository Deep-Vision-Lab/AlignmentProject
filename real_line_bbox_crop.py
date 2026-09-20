"""Crop saved real line images to their XML text envelope on all four sides.

The source dataset builder creates each line as:
    img[y1_padded:y2_padded, :]
where y1_padded/y2_padded come from the XML line envelope with 25% vertical
padding, while the full source-page width is retained.  Therefore:
  * XML x coordinates map directly to the saved line x axis (modulo any resize);
  * XML y coordinates must be shifted by the padded line-envelope top.

This module reconstructs exactly that geometry and removes left/right/top/bottom
padding while keeping only a small safety margin around the XML text boxes.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


def parse_xml_boxes(xml_path):
    root = ET.parse(str(xml_path)).getroot()
    boxes = []
    for elem in root.findall(".//DocumentElement"):
        x = elem.findtext("X")
        y = elem.findtext("Y")
        w = elem.findtext("Width")
        h = elem.findtext("Height")
        if None in (x, y, w, h):
            continue
        x = int(float(x))
        y = int(float(y))
        w = int(float(w))
        h = int(float(h))
        boxes.append(
            {
                "x1": x,
                "y1": y,
                "x2": x + w,
                "y2": y + h,
                "w": w,
                "h": h,
                "cx": x + 0.5 * w,
                "cy": y + 0.5 * h,
                "text": elem.findtext("Transcript") or "",
            }
        )
    return boxes


def dynamic_line_threshold(boxes):
    if not boxes:
        return 30
    heights = sorted(box["h"] for box in boxes if box["h"] > 0)
    median_h = heights[len(heights) // 2] if heights else 40
    centers = sorted(box["cy"] for box in boxes)
    gaps = sorted(
        gap
        for gap in (
            centers[index + 1] - centers[index]
            for index in range(len(centers) - 1)
        )
        if gap > 0
    )
    median_gap = gaps[len(gaps) // 2] if gaps else median_h
    return max(int(max(0.6 * median_h, 0.6 * median_gap)), 25)


def group_boxes_into_lines(boxes, line_threshold=None):
    if not boxes:
        return [], 0
    threshold = (
        dynamic_line_threshold(boxes)
        if line_threshold is None
        else int(line_threshold)
    )
    ordered = sorted(boxes, key=lambda box: box["cy"])
    lines = []
    current = [ordered[0]]
    for box in ordered[1:]:
        if abs(box["cy"] - current[-1]["cy"]) < threshold:
            current.append(box)
        else:
            lines.append(current)
            current = [box]
    lines.append(current)
    lines.sort(key=lambda line: min(box["y1"] for box in line))
    return lines, threshold


def _source_page_size(side_dir: Path):
    candidates = sorted(
        path for path in side_dir.glob("original_image.*") if path.is_file()
    )
    if not candidates:
        return None, None, ""
    with Image.open(candidates[0]) as image:
        return int(image.width), int(image.height), str(candidates[0])


def _builder_vertical_envelope(line, source_height: int, pad_ratio: float = 0.25):
    """Reproduce build_new_quran_dataset.py::line_envelope exactly."""
    raw_y1 = min(box["y1"] for box in line)
    raw_y2 = max(box["y2"] for box in line)
    raw_height = max(1, raw_y2 - raw_y1)
    pad = int(float(pad_ratio) * raw_height)
    saved_y1 = max(0, raw_y1 - pad)
    saved_y2 = min(int(source_height), raw_y2 + pad)
    return raw_y1, raw_y2, saved_y1, saved_y2, pad


def line_text_bounds(
    side_dir,
    line_index: int,
    line_image_size,
    *,
    margin_ratio: float = 0.05,
    minimum_margin_px: int = 2,
    builder_vertical_pad_ratio: float = 0.25,
):
    """Return (left, top, right, bottom) text crop in saved line coordinates."""
    side_dir = Path(side_dir)
    xml_path = side_dir / "original.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"Source XML not found: {xml_path}")

    boxes = parse_xml_boxes(xml_path)
    lines, threshold = group_boxes_into_lines(boxes)
    index = int(line_index) - 1
    if index < 0 or index >= len(lines):
        raise IndexError(
            f"line_index={line_index} outside XML line count {len(lines)}"
        )

    line = lines[index]
    raw_x1 = min(box["x1"] for box in line)
    raw_x2 = max(box["x2"] for box in line)
    raw_y1 = min(box["y1"] for box in line)
    raw_y2 = max(box["y2"] for box in line)

    source_w, source_h, source_image = _source_page_size(side_dir)
    line_w, line_h = map(int, line_image_size)
    if source_w is None or source_h is None:
        raise FileNotFoundError(
            f"original_image.* is required to map XML coordinates for {side_dir}"
        )

    (
        _raw_y1,
        _raw_y2,
        saved_y1,
        saved_y2,
        builder_pad,
    ) = _builder_vertical_envelope(
        line,
        source_h,
        pad_ratio=float(builder_vertical_pad_ratio),
    )
    saved_height = max(1, int(saved_y2 - saved_y1))

    scale_x = float(line_w) / float(source_w)
    scale_y = float(line_h) / float(saved_height)

    text_x1 = float(raw_x1) * scale_x
    text_x2 = float(raw_x2) * scale_x
    text_y1 = float(raw_y1 - saved_y1) * scale_y
    text_y2 = float(raw_y2 - saved_y1) * scale_y

    text_box_height = max(1.0, text_y2 - text_y1)
    margin = max(
        int(minimum_margin_px),
        int(round(text_box_height * max(0.0, float(margin_ratio)))),
    )

    left = max(0, int(round(text_x1)) - margin)
    right = min(line_w, int(round(text_x2)) + margin)
    top = max(0, int(round(text_y1)) - margin)
    bottom = min(line_h, int(round(text_y2)) + margin)
    if right <= left or bottom <= top:
        raise RuntimeError(
            "Invalid four-sided XML crop "
            f"({left},{top})-({right},{bottom}) for line size {(line_w, line_h)}"
        )

    return (left, top, right, bottom), {
        "xml_path": str(xml_path),
        "source_image": source_image,
        "line_index": int(line_index),
        "xml_line_count": len(lines),
        "line_threshold": int(threshold),
        "line_box_count": len(line),
        "raw_source_x1": int(raw_x1),
        "raw_source_y1": int(raw_y1),
        "raw_source_x2": int(raw_x2),
        "raw_source_y2": int(raw_y2),
        "source_page_width": int(source_w),
        "source_page_height": int(source_h),
        "builder_saved_y1": int(saved_y1),
        "builder_saved_y2": int(saved_y2),
        "builder_vertical_pad": int(builder_pad),
        "line_image_width": int(line_w),
        "line_image_height": int(line_h),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "text_local_x1": float(text_x1),
        "text_local_y1": float(text_y1),
        "text_local_x2": float(text_x2),
        "text_local_y2": float(text_y2),
        "margin_ratio": float(margin_ratio),
        "margin_px": int(margin),
        "crop_left": int(left),
        "crop_top": int(top),
        "crop_right": int(right),
        "crop_bottom": int(bottom),
        "crop_width": int(right - left),
        "crop_height": int(bottom - top),
        "crop_method": "source-xml-four-sided-text-envelope",
    }


def bbox_crop_line(
    image: Image.Image,
    side_dir,
    line_index: int,
    *,
    margin_ratio: float = 0.05,
    minimum_margin_px: int = 2,
):
    source = image.copy()
    bounds, metadata = line_text_bounds(
        side_dir,
        line_index,
        source.size,
        margin_ratio=float(margin_ratio),
        minimum_margin_px=int(minimum_margin_px),
    )
    cropped = source.crop(bounds)
    metadata.update(
        {
            "source_mode": str(source.mode),
            "crop_mode": str(cropped.mode),
            "preserved_full_line_height": False,
            "all_four_sides_cropped": True,
        }
    )
    return cropped, metadata


# Backward-compatible name used by the first diagnostic revision.
def line_horizontal_bounds(
    side_dir,
    line_index: int,
    line_image_width: int,
    *,
    margin_px: int = 0,
    margin_ratio_of_line_height: float = 0.0,
    line_image_height: int | None = None,
):
    height = int(line_image_height or 128)
    bounds, metadata = line_text_bounds(
        side_dir,
        line_index,
        (int(line_image_width), height),
        margin_ratio=float(margin_ratio_of_line_height),
        minimum_margin_px=int(margin_px),
    )
    left, _top, right, _bottom = bounds
    return (left, right), metadata
