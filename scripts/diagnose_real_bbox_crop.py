#!/usr/bin/env python3
"""Preview source-XML four-sided crop and grayscale model input."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image, ImageDraw

from RealDataSet import ArabicAllPageLinesDataset
from real_line_bbox_crop import bbox_crop_line
from zero_shot_preprocessing import aspect_preserving_pad_with_metadata


def _stack(images, labels, width=1024):
    rows = []
    for image, label in zip(images, labels):
        im = image.convert("RGB")
        scale = min(1.0, width / max(1, im.width))
        if scale != 1.0:
            im = im.resize(
                (
                    max(1, int(round(im.width * scale))),
                    max(1, int(round(im.height * scale))),
                ),
                Image.Resampling.BILINEAR,
            )
        row = Image.new("RGB", (width, im.height + 28), "white")
        row.paste(im, ((width - im.width) // 2, 28))
        ImageDraw.Draw(row).text((8, 6), label, fill="black")
        rows.append(row)
    out = Image.new("RGB", (width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="Results/Diagnostics/RealBBoxCrop")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin-ratio", type=float, default=0.05)
    args = parser.parse_args()

    dataset = ArabicAllPageLinesDataset(
        args.dataset, transform=None, validate_paths=False
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    summary = []
    selected = 0
    for index in indices:
        if selected >= max(1, args.samples):
            break
        sample = dataset.samples[index]
        side_dir = dataset._resolve(sample["page_dir"])
        xml_path = side_dir / "original.xml"
        if not xml_path.is_file() or int(sample["line_idx"]) <= 0:
            continue

        image_path = dataset._resolve(sample["line_image_path"])
        with Image.open(image_path) as handle:
            original = handle.convert("RGB").copy()

        try:
            bbox_crop, bbox_meta = bbox_crop_line(
                original,
                side_dir,
                int(sample["line_idx"]),
                margin_ratio=args.margin_ratio,
                minimum_margin_px=2,
            )
        except Exception as exc:
            summary.append(
                {
                    "dataset_index": int(index),
                    "image": str(image_path),
                    "status": "bbox_error",
                    "error": repr(exc),
                }
            )
            continue

        grayscale = bbox_crop.convert("L")
        model_input, resize_meta = aspect_preserving_pad_with_metadata(
            grayscale,
            size=(128, 1024),
            target_ink_height_ratio=0.72,
            horizontal_jitter=0.0,
        )

        overlay = original.copy()
        draw = ImageDraw.Draw(overlay)
        left = int(bbox_meta["crop_left"])
        top = int(bbox_meta["crop_top"])
        right = int(bbox_meta["crop_right"]) - 1
        bottom = int(bbox_meta["crop_bottom"]) - 1
        draw.rectangle((left, top, right, bottom), outline=(255, 0, 0), width=4)

        selected += 1
        sample_dir = output / (
            f"sample_{selected:02d}_{sample['source_pair_id']}_"
            f"{sample['side']}_line_{sample['line_idx']:02d}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        original.save(sample_dir / "01_original_line.png")
        overlay.save(sample_dir / "02_four_side_bbox_overlay.png")
        bbox_crop.save(sample_dir / "03_four_side_bbox_crop_rgb_reference.png")
        grayscale.save(sample_dir / "04_grayscale_crop.png")
        model_input.save(sample_dir / "05_grayscale_model_input.png")
        _stack(
            [original, overlay, bbox_crop, grayscale, model_input],
            [
                "1 original saved line (contains dataset padding)",
                "2 XML text envelope: all four sides",
                "3 four-side crop",
                "4 true grayscale crop",
                "5 grayscale 1024x128 model canvas",
            ],
        ).save(sample_dir / "comparison.png")

        record = {
            "dataset_index": int(index),
            "source_pair_id": sample["source_pair_id"],
            "side": sample["side"],
            "line_idx": int(sample["line_idx"]),
            "image": str(image_path),
            "status": "ok",
            "original_size": list(original.size),
            "bbox_crop_size": list(bbox_crop.size),
            "grayscale_mode": grayscale.mode,
            "model_input_mode": model_input.mode,
            "bbox": bbox_meta,
            "resize": resize_meta,
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary.append(record)

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {selected} four-sided grayscale crop previews to {output}")


if __name__ == "__main__":
    main()
