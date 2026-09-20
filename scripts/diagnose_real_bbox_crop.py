#!/usr/bin/env python3
"""Compare current detector crop with source-XML line-bbox crop."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image, ImageDraw

from RealDataSet import ArabicAllPageLinesDataset
from real_line_bbox_crop import bbox_crop_line
from zero_shot_preprocessing import (
    foreground_crop_with_metadata,
    aspect_preserving_pad_with_metadata,
)


def _stack(images, labels, width=1024):
    rows = []
    for image, label in zip(images, labels):
        im = image.convert("RGB")
        scale = min(1.0, width / max(1, im.width))
        if scale != 1.0:
            im = im.resize(
                (max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale)))),
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
    parser.add_argument("--margin-ratio", type=float, default=0.20)
    args = parser.parse_args()

    os.environ.setdefault("ZERO_SHOT_CROP_MODE", "vertical_borders")
    dataset = ArabicAllPageLinesDataset(args.dataset, transform=None, validate_paths=False)
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
                margin_ratio_of_line_height=args.margin_ratio,
                minimum_margin_px=8,
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

        current_crop, current_meta = foreground_crop_with_metadata(original)

        bbox_input, bbox_resize = aspect_preserving_pad_with_metadata(
            bbox_crop,
            size=(128, 1024),
            target_ink_height_ratio=0.72,
            horizontal_jitter=0.0,
        )
        current_input, current_resize = aspect_preserving_pad_with_metadata(
            current_crop,
            size=(128, 1024),
            target_ink_height_ratio=0.72,
            horizontal_jitter=0.0,
        )

        overlay = original.copy()
        draw = ImageDraw.Draw(overlay)
        left = int(bbox_meta["crop_left"])
        right = int(bbox_meta["crop_right"]) - 1
        draw.rectangle((left, 0, right, original.height - 1), outline=(255, 0, 0), width=4)

        selected += 1
        sample_dir = output / f"sample_{selected:02d}_{sample['source_pair_id']}_{sample['side']}_line_{sample['line_idx']:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        original.save(sample_dir / "original_line.png")
        overlay.save(sample_dir / "bbox_overlay.png")
        bbox_crop.save(sample_dir / "bbox_crop.png")
        current_crop.save(sample_dir / "current_detector_crop.png")
        bbox_input.save(sample_dir / "bbox_model_input.png")
        current_input.save(sample_dir / "current_model_input.png")
        _stack(
            [overlay, current_crop, bbox_crop, current_input, bbox_input],
            [
                "XML bbox overlay on original line",
                "current detector crop",
                "XML bbox crop",
                "current detector -> model input",
                "XML bbox -> model input",
            ],
        ).save(sample_dir / "comparison.png")

        record = {
            "dataset_index": int(index),
            "source_pair_id": sample["source_pair_id"],
            "side": sample["side"],
            "line_idx": int(sample["line_idx"]),
            "image": str(image_path),
            "status": "ok",
            "original_width": int(original.width),
            "current_crop_width": int(current_crop.width),
            "bbox_crop_width": int(bbox_crop.width),
            "current_width_fraction": float(current_crop.width / original.width),
            "bbox_width_fraction": float(bbox_crop.width / original.width),
            "current_crop_mode": current_meta.get("crop_mode"),
            "current_crop_detector": current_meta.get("crop_detector"),
            "bbox": bbox_meta,
            "bbox_resize": bbox_resize,
            "current_resize": current_resize,
        }
        (sample_dir / "metadata.json").write_text(
            json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        summary.append(record)

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {selected} bbox-crop comparisons to {output}")


if __name__ == "__main__":
    main()
