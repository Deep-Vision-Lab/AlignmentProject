"""Recover horizontal text bounds for real line images from source XML boxes.

The dataset builder groups PartOfWord XML boxes into lines, then saves each
line image as img[y1:y2, :] -- i.e. vertical crop only, full page width.
Therefore source-page X coordinates remain directly aligned with line_XX.png.
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


def _source_page_width(side_dir: Path):
    candidates = sorted(
        path
        for path in side_dir.glob("original_image.*")
        if path.is_file()
    )
    if not candidates:
        return None, ""
    with Image.open(candidates[0]) as image:
        return int(image.width), str(candidates[0])


def line_horizontal_bounds(
    side_dir,
    line_index: int,
    line_image_width: int,
    *,
    margin_px: int = 0,
    margin_ratio_of_line_height: float = 0.0,
    line_image_height: int | None = None,
):
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
    source_width, source_image = _source_page_width(side_dir)
    if source_width is None or source_width <= 0:
        source_width = int(line_image_width)

    scale_x = float(line_image_width) / float(source_width)
    x1 = int(round(raw_x1 * scale_x))
    x2 = int(round(raw_x2 * scale_x))

    extra = max(0, int(margin_px))
    if line_image_height is not None:
        extra = max(
            extra,
            int(round(float(line_image_height) * float(margin_ratio_of_line_height))),
        )
    left = max(0, x1 - extra)
    right = min(int(line_image_width), x2 + extra)
    if right <= left:
        raise RuntimeError(
            f"Invalid bbox line crop {left}:{right} for width={line_image_width}"
        )

    return (left, right), {
        "xml_path": str(xml_path),
        "source_image": source_image,
        "line_index": int(line_index),
        "xml_line_count": len(lines),
        "line_threshold": int(threshold),
        "line_box_count": len(line),
        "raw_source_x1": int(raw_x1),
        "raw_source_x2": int(raw_x2),
        "source_page_width": int(source_width),
        "line_image_width": int(line_image_width),
        "scale_x": float(scale_x),
        "unmargined_x1": int(x1),
        "unmargined_x2": int(x2),
        "margin_px": int(extra),
        "crop_left": int(left),
        "crop_right": int(right),
        "crop_width": int(right - left),
    }


def bbox_crop_line(
    image: Image.Image,
    side_dir,
    line_index: int,
    *,
    margin_ratio_of_line_height: float = 0.20,
    minimum_margin_px: int = 8,
):
    source = image.convert("RGB")
    bounds, metadata = line_horizontal_bounds(
        side_dir,
        line_index,
        source.width,
        margin_px=int(minimum_margin_px),
        margin_ratio_of_line_height=float(margin_ratio_of_line_height),
        line_image_height=source.height,
    )
    left, right = bounds
    cropped = source.crop((left, 0, right, source.height))
    metadata.update(
        {
            "crop_top": 0,
            "crop_bottom": int(source.height),
            "crop_height": int(source.height),
            "crop_method": "source-xml-line-envelope-horizontal",
            "preserved_full_line_height": True,
        }
    )
    return cropped, metadata
