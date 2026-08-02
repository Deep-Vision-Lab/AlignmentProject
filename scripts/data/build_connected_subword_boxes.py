#!/usr/bin/env python3
"""Build and validate renderer-derived connected-subword interval sidecars."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import arabic_reshaper
from bidi.algorithm import get_display
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from connected_subword_mode import connected_units

_DIACRITICS = re.compile("""ّ|َ|ً|ُ|ٌ|ِ|ٍ|ْ|ـ""", re.VERBOSE)
_RESHAPER_CONFIG = {
    "delete_harakat": True,
    "support_zwj": False,
    "delete_at_sign": True,
    "use_unshaped_instead_of_isolated": True,
}
_SCHEMA_VERSION = 2


def clean_text(text: str, *, strip: bool = True) -> str:
    value = _DIACRITICS.sub("", str(text))
    return value.strip() if strip else value


def visual_text(text: str, *, strip: bool = True) -> str:
    reshaped = arabic_reshaper.ArabicReshaper(
        configuration=_RESHAPER_CONFIG
    ).reshape(clean_text(text, strip=strip))
    reshaped = "".join(
        character
        for character in reshaped
        if character.isprintable() and ord(character) != 0x200C
    )
    return get_display(reshaped)


def logical_subword_spans(text: str) -> list[dict]:
    text = clean_text(text)
    cursor = word_index = logical_index = 0
    spans = []
    for unit in connected_units(text):
        if unit.kind == "space":
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            word_index += 1
            continue
        if unit.kind != "subword":
            continue
        start = text.find(unit.text, cursor)
        if start < 0:
            raise ValueError(
                f"Could not locate {unit.text!r} after index {cursor} in {text!r}"
            )
        end = start + len(unit.text)
        spans.append(
            {
                "text": unit.text,
                "logical_start": start,
                "logical_end": end,
                "logical_index": logical_index,
                "word_index": word_index,
            }
        )
        logical_index += 1
        cursor = end
    return spans


def _rendered_width(draw, text: str, font) -> float:
    if not text:
        return 0.0
    return float(draw.textlength(visual_text(text, strip=False), font=font))


def _snap_boundary(projection: np.ndarray, estimate: float, radius: int) -> float:
    if projection.size == 0 or radius <= 0:
        return float(estimate)
    center = int(round(estimate))
    left = max(0, center - radius)
    right = min(int(projection.size) - 1, center + radius)
    if right <= left:
        return float(estimate)
    local = projection[left : right + 1]
    candidates = np.flatnonzero(local == local.min()) + left
    return float(min(candidates.tolist(), key=lambda value: abs(value - estimate)))


def _image_ink(image_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with Image.open(image_path) as opened:
        gray = np.asarray(opened.convert("L"), dtype=np.float32)
    border = np.concatenate(
        [
            gray[:2, :].reshape(-1),
            gray[-2:, :].reshape(-1),
            gray[:, :2].reshape(-1),
            gray[:, -2:].reshape(-1),
        ]
    )
    background = float(np.median(border)) if border.size else float(np.median(gray))
    ink = np.abs(gray - background) >= 20.0
    return gray, ink


def _repair_neighbor_overlaps(payload: dict) -> None:
    boxes = sorted(payload["subwords"], key=lambda item: item["logical_index"])
    for right_box, left_box in zip(boxes, boxes[1:]):
        if float(left_box["x1"]) > float(right_box["x0"]):
            boundary = 0.5 * (float(left_box["x1"]) + float(right_box["x0"]))
            right_box["x0"] = boundary
            left_box["x1"] = boundary


def snap_boxes_to_image_ink(payload: dict, image_path: Path, radius: int) -> dict:
    if radius <= 0:
        return payload
    _gray, ink = _image_ink(image_path)
    projection = ink.sum(axis=0).astype(np.float32)
    for item in payload["subwords"]:
        old_x0, old_x1 = float(item["x0"]), float(item["x1"])
        x0 = _snap_boundary(projection, old_x0, radius)
        x1 = _snap_boundary(projection, old_x1, radius)
        if x1 - x0 >= 1.0:
            item["x0"], item["x1"], item["ink_snapped"] = x0, x1, True
        else:
            item["ink_snapped"] = False
    _repair_neighbor_overlaps(payload)
    payload["ink_snap_radius"] = int(radius)
    return payload


def measure_subword_boxes(text, *, font, canvas_width, canvas_height, padding):
    cleaned = clean_text(text)
    display = visual_text(cleaned)
    probe = Image.new("RGB", (1000, 1000), "black")
    draw = ImageDraw.Draw(probe)
    left, _top, right, _bottom = draw.textbbox((0, 0), display, font=font)
    text_width = max(1.0, float(right - left))
    natural_width = text_width + 2.0 * padding
    scale_x = float(canvas_width) / natural_width
    boxes = []
    for span in logical_subword_spans(cleaned):
        before = _rendered_width(
            draw, cleaned[: int(span["logical_start"])], font
        )
        through = _rendered_width(
            draw, cleaned[: int(span["logical_end"])], font
        )
        item = dict(span)
        item["x0"] = max(
            0.0, min(float(canvas_width), (padding + text_width - through) * scale_x)
        )
        item["x1"] = max(
            0.0, min(float(canvas_width), (padding + text_width - before) * scale_x)
        )
        if item["x1"] < item["x0"]:
            item["x0"], item["x1"] = item["x1"], item["x0"]
        boxes.append(item)
    return {
        "schema_version": _SCHEMA_VERSION,
        "text": cleaned,
        "display_text": display,
        "canvas_width": int(canvas_width),
        "canvas_height": int(canvas_height),
        "natural_width": natural_width,
        "padding": int(padding),
        "subwords": boxes,
    }


def validate_payload(payload: dict, image_path: Path) -> dict:
    _gray, ink = _image_ink(image_path)
    _height, width = ink.shape
    boxes = sorted(payload["subwords"], key=lambda item: item["logical_index"])
    errors = []
    centers = []
    union_columns = np.zeros(width, dtype=bool)
    per_box_ink = []

    for item in boxes:
        x0, x1 = float(item["x0"]), float(item["x1"])
        if not (0.0 <= x0 < x1 <= float(width)):
            errors.append(f"invalid_bounds:{item['logical_index']}")
            continue
        left = max(0, min(width - 1, int(np.floor(x0))))
        right = max(left + 1, min(width, int(np.ceil(x1))))
        union_columns[left:right] = True
        ink_pixels = int(ink[:, left:right].sum())
        ink_columns = int(ink[:, left:right].any(axis=0).sum())
        per_box_ink.append(
            {
                "logical_index": int(item["logical_index"]),
                "ink_pixels": ink_pixels,
                "ink_columns": ink_columns,
            }
        )
        if ink_pixels <= 0:
            errors.append(f"empty_box:{item['logical_index']}")
        centers.append(0.5 * (x0 + x1))

    if any(left + 1e-3 < right for left, right in zip(centers, centers[1:])):
        errors.append("logical_order_not_rtl")

    total_ink = int(ink.sum())
    covered_ink = int(ink[:, union_columns].sum()) if union_columns.any() else 0
    coverage = covered_ink / max(1, total_ink)
    if coverage < 0.70:
        errors.append("low_total_ink_coverage")

    max_overlap_ratio = 0.0
    for first, second in zip(boxes, boxes[1:]):
        overlap = max(
            0.0,
            min(float(first["x1"]), float(second["x1"]))
            - max(float(first["x0"]), float(second["x0"])),
        )
        denominator = max(
            1.0,
            min(
                float(first["x1"]) - float(first["x0"]),
                float(second["x1"]) - float(second["x0"]),
            ),
        )
        max_overlap_ratio = max(max_overlap_ratio, overlap / denominator)
    if max_overlap_ratio > 0.50:
        errors.append("excessive_neighbor_overlap")

    return {
        "valid": not errors,
        "errors": errors,
        "total_ink_pixels": total_ink,
        "covered_ink_pixels": covered_ink,
        "total_ink_coverage": coverage,
        "max_neighbor_overlap_ratio": max_overlap_ratio,
        "per_box_ink": per_box_ink,
    }


def generator_signature(args, font_path: Path) -> str:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "font_sha256": hashlib.sha256(font_path.read_bytes()).hexdigest(),
        "font_size": int(args.font_size),
        "padding": int(args.padding),
        "canvas_width": int(args.canvas_width),
        "canvas_height": int(args.canvas_height),
        "snap_radius": int(args.snap_radius),
        "reshaper": _RESHAPER_CONFIG,
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sidecar_is_current(path: Path, signature: str) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        payload.get("generator_signature") == signature
        and bool(payload.get("validation", {}).get("valid", False))
    )


def write_overlay(image_path: Path, payload: dict, output_path: Path) -> None:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    draw = ImageDraw.Draw(image)
    for item in payload["subwords"]:
        x0, x1 = float(item["x0"]), float(item["x1"])
        draw.rectangle((x0, 1, x1, image.height - 2), outline=(255, 0, 0), width=2)
        draw.text((x0 + 2, 2), str(item["logical_index"]), fill=(255, 255, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create validated connected-subword sidecars for Synthetic_Arabic"
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--font", default=str(PROJECT_DIR / "Fonts/Arslan_Wessam_B.ttf")
    )
    parser.add_argument("--font-size", type=int, default=90)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--canvas-width", type=int, default=1024)
    parser.add_argument("--canvas-height", type=int, default=128)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=0)
    parser.add_argument("--snap-radius", type=int, default=8)
    parser.add_argument("--overlay-count", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-invalid", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    images_dir, texts_dir = data_dir / "images", data_dir / "texts"
    output_dir = data_dir / "subword_boxes"
    if not images_dir.is_dir() or not texts_dir.is_dir():
        raise SystemExit(f"Expected images/ and texts/ under {data_dir}")
    font_path = Path(args.font).expanduser().resolve()
    if not font_path.is_file():
        raise SystemExit(f"Font not found: {font_path}")
    font = ImageFont.truetype(str(font_path), args.font_size)
    signature = generator_signature(args, font_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    detected = sorted(
        int(path.stem.split("_")[-1])
        for path in images_dir.glob("img1_*.png")
        if path.stem.split("_")[-1].isdigit()
    )
    if not detected:
        raise SystemExit(f"No img1_N.png files found in {images_dir}")
    end_index = args.end_index if args.end_index > 0 else detected[-1]
    written = skipped = invalid = 0
    overlays_written = 0
    for index in detected:
        if index < args.start_index or index > end_index:
            continue
        for line in (1, 2):
            image_path = images_dir / f"img{line}_{index}.png"
            text_path = texts_dir / f"text{line}_{index}.txt"
            output_path = output_dir / f"subwords{line}_{index}.json"
            if not image_path.is_file() or not text_path.is_file():
                raise FileNotFoundError(f"Incomplete synthetic sample {index}")
            if (
                output_path.exists()
                and not args.overwrite
                and sidecar_is_current(output_path, signature)
            ):
                skipped += 1
                continue
            payload = measure_subword_boxes(
                text_path.read_text(encoding="utf-8").strip(),
                font=font,
                canvas_width=args.canvas_width,
                canvas_height=args.canvas_height,
                padding=args.padding,
            )
            payload = snap_boxes_to_image_ink(
                payload, image_path, max(0, args.snap_radius)
            )
            payload.update(
                {
                    "image": str(image_path),
                    "text_file": str(text_path),
                    "font": str(font_path),
                    "font_size": args.font_size,
                    "generator_signature": signature,
                }
            )
            payload["validation"] = validate_payload(payload, image_path)
            if not payload["validation"]["valid"]:
                invalid += 1
                if not args.allow_invalid:
                    raise ValueError(
                        f"Invalid sidecar for {image_path}: "
                        f"{payload['validation']['errors']}"
                    )
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            if overlays_written < max(0, int(args.overlay_count)):
                write_overlay(
                    image_path,
                    payload,
                    output_dir / "overlays" / f"subwords{line}_{index}.png",
                )
                overlays_written += 1
            written += 1
    print(
        f"Connected-subword boxes ready: written={written} skipped={skipped} "
        f"invalid={invalid} overlays={overlays_written} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
