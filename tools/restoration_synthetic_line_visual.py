#!/usr/bin/env python3
"""Visualize the restoration preprocessing/window pipeline on a real synthetic line.

No arguments are required. The script auto-selects a real line from the project's
synthetic datasets, preferring DataSet/Synthetic63 (the restoration branch's
current synthetic dataset), then DataSet/Synthetic_Arabic.

Outputs:
  Results/Diagnostics/restoration_points/synthetic_line/
"""
from __future__ import annotations

from pathlib import Path
import json
import math
import sys

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zero_shot_preprocessing import ManuscriptLinePreprocessor

WINDOW_SIZE = 32
STRIDE = 16
TARGET_SIZE = (128, 1024)
OUT_DIR = ROOT / "Results" / "Diagnostics" / "restoration_points" / "synthetic_line"


def natural_key(path: Path):
    text = path.stem
    parts = []
    current = ""
    digit = None
    for ch in text:
        is_digit = ch.isdigit()
        if digit is None or is_digit == digit:
            current += ch
        else:
            parts.append(int(current) if digit else current)
            current = ch
        digit = is_digit
    if current:
        parts.append(int(current) if digit else current)
    return parts


def find_synthetic_line() -> Path:
    exact = [
        ROOT / "DataSet" / "Synthetic63" / "images" / "img1_132.png",
        ROOT / "DataSet" / "Synthetic63" / "images" / "img1_1.png",
        ROOT / "DataSet" / "Synthetic_Arabic" / "images" / "img1_1.png",
    ]
    for path in exact:
        if path.is_file():
            return path

    roots = [
        ROOT / "DataSet" / "Synthetic63" / "images",
        ROOT / "DataSet" / "Synthetic_Arabic" / "images",
    ]
    roots.extend(sorted((ROOT / "DataSet").glob("Synthetic*/images")))

    seen = set()
    for folder in roots:
        folder = folder.resolve()
        if folder in seen or not folder.is_dir():
            continue
        seen.add(folder)
        candidates = []
        for suffix in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
            candidates.extend(folder.glob(suffix))
        if candidates:
            return sorted(candidates, key=natural_key)[0]

    raise FileNotFoundError(
        "No synthetic line image found. Expected e.g. "
        "DataSet/Synthetic63/images/img1_132.png or "
        "DataSet/Synthetic_Arabic/images/img1_1.png"
    )


def transcript_for(image_path: Path) -> Path | None:
    text_dir = image_path.parent.parent / "texts"
    stem = image_path.stem
    if stem.startswith("img"):
        candidate = text_dir / ("text" + stem[3:] + ".txt")
        if candidate.is_file():
            return candidate
    return None


def make_window_overlay(image: Image.Image) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    width, height = out.size
    count = 0
    for x0 in range(0, width - WINDOW_SIZE + 1, STRIDE):
        x1 = x0 + WINDOW_SIZE
        draw.rectangle((x0, 0, x1 - 1, height - 1), outline=(220, 20, 20), width=1)
        if count % 4 == 0:
            draw.text((x0 + 1, 2), str(count), fill=(10, 10, 180))
        count += 1
    return out


def extract_windows(image: Image.Image):
    image = image.convert("RGB")
    width, height = image.size
    windows = []
    for index, x0 in enumerate(range(0, width - WINDOW_SIZE + 1, STRIDE)):
        windows.append((index, x0, x0 + WINDOW_SIZE, image.crop((x0, 0, x0 + WINDOW_SIZE, height))))
    return windows


def save_window_sheet(windows, path: Path, *, rtl=False):
    entries = list(reversed(windows)) if rtl else list(windows)
    columns = 8
    rows = int(math.ceil(len(entries) / columns))
    scale = 2
    patch_w = WINDOW_SIZE * scale
    patch_h = 128 * scale
    label_h = 34
    canvas = Image.new("RGB", (columns * patch_w, rows * (patch_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for slot, (physical_index, x0, x1, patch) in enumerate(entries):
        row = slot // columns
        col = slot % columns
        logical_index = slot if rtl else physical_index
        ox = col * patch_w
        oy = row * (patch_h + label_h)
        draw.text(
            (ox + 2, oy + 2),
            f"L{logical_index} P{physical_index}\nx={x0}:{x1}",
            fill="black",
        )
        canvas.paste(patch.resize((patch_w, patch_h)), (ox, oy + label_h))
    canvas.save(path)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_path = find_synthetic_line()
    text_path = transcript_for(image_path)
    source = Image.open(image_path).convert("RGB")

    processor = ManuscriptLinePreprocessor(
        size=TARGET_SIZE,
        training=False,
        augment=False,
        binarize=False,
        preserve_aspect=True,
        crop_foreground=True,
        target_ink_height_ratio=0.72,
        autocontrast=False,
    )
    prepared, geometry = processor.preprocess_with_metadata(source)

    source.save(OUT_DIR / "01_original_synthetic_line.png")

    boxed = source.copy()
    draw = ImageDraw.Draw(boxed)
    box = (
        int(geometry["crop_left"]),
        int(geometry["crop_top"]),
        int(geometry["crop_right"]) - 1,
        int(geometry["crop_bottom"]) - 1,
    )
    draw.rectangle(box, outline=(255, 0, 0), width=2)
    boxed.save(OUT_DIR / "02_detected_crop_on_real_line.png")

    prepared.save(OUT_DIR / "03_cropped_resized_padded_line.png")
    make_window_overlay(prepared).save(OUT_DIR / "04_window_boundaries.png")

    windows = extract_windows(prepared)
    save_window_sheet(windows, OUT_DIR / "05_windows_physical_left_to_right.png", rtl=False)
    save_window_sheet(windows, OUT_DIR / "06_windows_arabic_logical_right_to_left.png", rtl=True)

    transcript = ""
    if text_path is not None:
        transcript = text_path.read_text(encoding="utf-8").strip()
        (OUT_DIR / "transcript.txt").write_text(transcript + "\n", encoding="utf-8")

    info = {
        "image": str(image_path),
        "text": str(text_path) if text_path is not None else None,
        "transcript": transcript,
        "window_size": WINDOW_SIZE,
        "stride": STRIDE,
        "window_count": len(windows),
        "processed_size": list(prepared.size),
        "geometry": geometry,
    }
    (OUT_DIR / "geometry.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = f"""REAL SYNTHETIC LINE VISUAL CHECK
================================

Source image:
  {image_path}

Transcript:
  {text_path if text_path is not None else "<not found>"}

Original size:
  {source.size}

Processed size:
  {prepared.size}

Crop box on original:
  {box}

Window size / stride:
  {WINDOW_SIZE} / {STRIDE}

Window count:
  {len(windows)}

Inspect in this order:
1. 01_original_synthetic_line.png
2. 02_detected_crop_on_real_line.png
3. 03_cropped_resized_padded_line.png
4. 04_window_boundaries.png
5. 05_windows_physical_left_to_right.png
6. 06_windows_arabic_logical_right_to_left.png

What to verify yourself:
- the crop removes only OUTER blank margins;
- Arabic dots/diacritics at the top/bottom are not cut;
- blank spaces BETWEEN letters/words are preserved;
- resize preserves aspect ratio rather than stretching the line;
- model input remains RGB (no binarization);
- every window is the complete 128x32 RGB slice;
- the Arabic logical sequence is the physical window sequence in reverse order.
"""
    (OUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Visual results saved to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
