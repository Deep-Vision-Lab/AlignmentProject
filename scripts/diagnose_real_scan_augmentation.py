#!/usr/bin/env python3
"""Preview strong non-geometric augmentation on four-side-cropped grayscale lines."""
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
    gaussian_blur,
    gaussian_luminance_noise,
    speckle_noise,
    adjust_brightness_contrast,
)
from zero_shot_preprocessing import aspect_preserving_pad_with_metadata


def _sheet(images, labels, width=1024):
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
        canvas = Image.new("RGB", (width, im.height + 28), "white")
        canvas.paste(im, ((width - im.width) // 2, 28))
        ImageDraw.Draw(canvas).text((8, 6), label, fill="black")
        rows.append(canvas)
    out = Image.new("RGB", (width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        out.paste(row, (0, y))
        y += row.height
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--output-dir", default="Results/Diagnostics/RealScanAugmentation"
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.environ["REAL_BBOX_CROP"] = "1"
    os.environ.setdefault("REAL_BBOX_MARGIN_RATIO", "0.05")
    os.environ.setdefault("REAL_BBOX_MIN_MARGIN_PX", "2")
    os.environ["VISUAL_GRAYSCALE"] = "1"
    os.environ["REAL_GRAYSCALE"] = "1"

    dataset = ArabicAllPageLinesDataset(
        args.dataset, transform=None, validate_paths=False
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = indices[: min(len(indices), max(1, args.samples))]

    summary = []
    for ordinal, index in enumerate(indices, start=1):
        sample = dataset.samples[index]
        _text, clean = dataset.read_prepared_pil(index)
        clean = clean.convert("L")

        np.random.seed(args.seed + ordinal * 101)
        random.seed(args.seed + ordinal * 101)

        variants = [
            ("clean_grayscale", clean),
            ("blur_r0.70", gaussian_blur(clean, 0.70)),
            ("blur_r1.40", gaussian_blur(clean, 1.40)),
            ("gaussian_noise_std8", gaussian_luminance_noise(clean, 8.0)),
            ("gaussian_noise_std16", gaussian_luminance_noise(clean, 16.0)),
            ("speckle_0.0010", speckle_noise(clean, 0.0010)),
            (
                "contrast_brightness_strong",
                adjust_brightness_contrast(
                    clean, brightness_factor=0.90, contrast_factor=1.22
                ),
            ),
        ]

        combined = adjust_brightness_contrast(
            clean, brightness_factor=0.94, contrast_factor=1.18
        )
        combined = gaussian_blur(combined, 1.05)
        combined = gaussian_luminance_noise(combined, 12.0)
        combined = speckle_noise(combined, 0.0008)
        variants.append(("combined_training_strength", combined))

        model_inputs = []
        for name, variant in variants:
            normalized, _ = aspect_preserving_pad_with_metadata(
                variant,
                size=(128, 1024),
                target_ink_height_ratio=0.72,
                horizontal_jitter=0.0,
            )
            model_inputs.append((name, normalized))

        sample_dir = output / (
            f"sample_{ordinal:02d}_{sample['source_pair_id']}_"
            f"{sample['side']}_line_{sample['line_idx']:02d}"
        )
        sample_dir.mkdir(parents=True, exist_ok=True)
        clean.save(sample_dir / "clean_grayscale_crop.png")
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
                "clean_size": list(clean.size),
                "clean_mode": clean.mode,
                "geometry_changed_by_augmentation": False,
                "strong_preview": {
                    "blur_radius": 1.05,
                    "gaussian_noise_std": 12.0,
                    "speckle_fraction": 0.0008,
                    "brightness_factor": 0.94,
                    "contrast_factor": 1.18,
                },
            }
        )

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {len(summary)} strong grayscale augmentation previews to {output}")


if __name__ == "__main__":
    main()
