#!/usr/bin/env python3
"""Preview non-geometric real-data augmentations before enabling training."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from RealDataSet import ArabicAllPageLinesDataset
from real_scan_augmentation import (
    ScanOnlyAugmentor,
    gaussian_blur,
    gaussian_luminance_noise,
    speckle_noise,
    adjust_brightness_contrast,
)
from zero_shot_preprocessing import (
    foreground_crop_with_metadata,
    aspect_preserving_pad_with_metadata,
)


def _sheet(images, labels, width=1024):
    rows = []
    for image, label in zip(images, labels):
        im = image.convert("RGB")
        scale = min(1.0, width / max(1, im.width))
        if scale != 1.0:
            im = im.resize(
                (max(1, int(round(im.width * scale))), max(1, int(round(im.height * scale)))),
                Image.Resampling.BILINEAR,
            )
        canvas = Image.new("RGB", (width, im.height + 28), "white")
        canvas.paste(im, ((width - im.width) // 2, 28))
        ImageDraw.Draw(canvas).text((8, 6), label, fill="black")
        rows.append(canvas)
    height = sum(row.height for row in rows)
    out = Image.new("RGB", (width, height), "white")
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", default="Results/Diagnostics/RealScanAugmentation")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ.setdefault("ZERO_SHOT_CROP_MODE", "vertical_borders")
    dataset = ArabicAllPageLinesDataset(args.dataset, transform=None, validate_paths=False)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[: min(len(indices), max(1, args.samples))]

    summary = []
    for ordinal, index in enumerate(indices, start=1):
        sample = dataset.samples[index]
        image_path = dataset._resolve(sample["line_image_path"])
        with Image.open(image_path) as handle:
            original = handle.convert("RGB").copy()

        # Crop first on the clean image; augmentation must not alter crop geometry.
        cropped, crop_meta = foreground_crop_with_metadata(original)

        np.random.seed(args.seed + ordinal * 101)
        random.seed(args.seed + ordinal * 101)
        variants = [
            ("clean_crop", cropped),
            ("gaussian_blur_r0.45", gaussian_blur(cropped, 0.45)),
            ("gaussian_blur_r0.90", gaussian_blur(cropped, 0.90)),
            ("gaussian_noise_std3", gaussian_luminance_noise(cropped, 3.0)),
            ("gaussian_noise_std7", gaussian_luminance_noise(cropped, 7.0)),
            ("speckle_0.0005", speckle_noise(cropped, 0.0005)),
            (
                "mild_brightness_contrast",
                adjust_brightness_contrast(
                    cropped, brightness_factor=0.97, contrast_factor=1.06
                ),
            ),
        ]

        combined = adjust_brightness_contrast(
            cropped, brightness_factor=0.98, contrast_factor=1.07
        )
        combined = gaussian_blur(combined, 0.55)
        combined = gaussian_luminance_noise(combined, 4.5)
        combined = speckle_noise(combined, 0.00035)
        variants.append(("combined_candidate", combined))

        model_inputs = []
        for name, variant in variants:
            normalized, _ = aspect_preserving_pad_with_metadata(
                variant,
                size=(128, 1024),
                target_ink_height_ratio=0.72,
                horizontal_jitter=0.0,
            )
            model_inputs.append((name, normalized))

        sample_dir = output / f"sample_{ordinal:02d}_{sample['source_pair_id']}_{sample['side']}_line_{sample['line_idx']:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        original.save(sample_dir / "original_line.png")
        cropped.save(sample_dir / "clean_crop.png")
        for name, image in variants:
            image.save(sample_dir / f"{name}.png")
        for name, image in model_inputs:
            image.save(sample_dir / f"model_{name}.png")

        _sheet(
            [image for _, image in model_inputs],
            [name for name, _ in model_inputs],
        ).save(sample_dir / "model_input_comparison.png")

        summary.append(
            {
                "dataset_index": int(index),
                "source_pair_id": sample["source_pair_id"],
                "side": sample["side"],
                "line_idx": int(sample["line_idx"]),
                "image": str(image_path),
                "original_size": list(original.size),
                "crop_size": list(cropped.size),
                "crop_mode": crop_meta.get("crop_mode"),
                "crop_detector": crop_meta.get("crop_detector"),
                "geometry_changed_by_augmentation": False,
            }
        )

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(summary)} augmentation previews to {output}")


if __name__ == "__main__":
    main()
