#!/usr/bin/env python3
"""Build renderer-derived synthetic connected-subword interval sidecars.

The script reproduces the geometry used by ``generateDataArabic.py``: Arabic
reshaping, bidi display, natural text canvas with padding, then resize to
1024x128. It writes one JSON sidecar per line image.
"""
from __future__ import annotations

import argparse
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


def snap_boxes_to_image_ink(payload: dict, image_path: Path, radius: int) -> dict:
    if radius <= 0:
        return payload
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
    projection = (np.abs(gray - background) >= 20.0).sum(axis=0).astype(np.float32)
    for item in payload["subwords"]:
        old_x0, old_x1 = float(item["x0"]), float(item["x1"])
        x0 = _snap_boundary(projection, old_x0, radius)
        x1 = _snap_boundary(projection, old_x1, radius)
        if x1 - x0 >= 1.0:
            item["x0"], item["x1"], item["ink_snapped"] = x0, x1, True
        else:
            item["ink_snapped"] = False
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
        # Arabic logical prefixes occupy the right side of the rendered line.
        # Prefix widths remain stable when the reshaper emits ligatures.
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
        "schema_version": 1,
        "text": cleaned,
        "display_text": display,
        "canvas_width": int(canvas_width),
        "canvas_height": int(canvas_height),
        "natural_width": natural_width,
        "padding": int(padding),
        "subwords": boxes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create connected-subword box sidecars for Synthetic_Arabic"
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
    parser.add_argument("--overwrite", action="store_true")
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
    output_dir.mkdir(parents=True, exist_ok=True)
    detected = sorted(
        int(path.stem.split("_")[-1])
        for path in images_dir.glob("img1_*.png")
        if path.stem.split("_")[-1].isdigit()
    )
    if not detected:
        raise SystemExit(f"No img1_N.png files found in {images_dir}")
    end_index = args.end_index if args.end_index > 0 else detected[-1]
    written = skipped = 0
    for index in detected:
        if index < args.start_index or index > end_index:
            continue
        for line in (1, 2):
            image_path = images_dir / f"img{line}_{index}.png"
            text_path = texts_dir / f"text{line}_{index}.txt"
            output_path = output_dir / f"subwords{line}_{index}.json"
            if not image_path.is_file() or not text_path.is_file():
                raise FileNotFoundError(f"Incomplete synthetic sample {index}")
            if output_path.exists() and not args.overwrite:
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
                }
            )
            output_path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            written += 1
    print(
        f"Connected-subword boxes ready: written={written} skipped={skipped} "
        f"output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
