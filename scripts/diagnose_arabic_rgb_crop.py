#!/usr/bin/env python3
"""Diagnostic for the exact RGB crop/model-input path used by real fine-tuning.

Nothing in this diagnostic changes the source appearance.  A temporary mask is
used only to locate the handwriting.  The crop itself and the 1024x128 model
canvas keep the original RGB pixels.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Exact real-data mode requested for fine-tuning.
os.environ["REAL_SYNTHETIC_STYLE"] = "0"
os.environ["REAL_BINARIZE"] = "0"
os.environ["REAL_BINARIZE_AUTOCONTRAST"] = "0"
os.environ["REAL_BINARIZE_AUTO_INVERT"] = "0"
os.environ["ZERO_SHOT_PREPROCESS"] = "1"
os.environ["ZERO_SHOT_FOREGROUND_CROP"] = "1"
os.environ["ZERO_SHOT_PRESERVE_ASPECT"] = "1"
os.environ["ZERO_SHOT_SOURCE_GEOMETRY"] = "0"
os.environ["ZERO_SHOT_CROP_MODE"] = "robust_projection"
os.environ.setdefault("ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", "0.72")

from RealDataSet import ArabicManifestIndependentLineDataset
from zero_shot_preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    aspect_preserving_pad_with_metadata,
    foreground_crop_with_metadata,
    foreground_detection_mask_with_metadata,
)


def _draw_bbox(image: Image.Image, box):
    out = image.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle(tuple(map(int, box)), outline=(255, 0, 0), width=3)
    return out


def _save_mask(mask: np.ndarray, path: Path):
    Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(path)


def _window_overlay(model_input: Image.Image, window_width=32, stride=16):
    out = model_input.convert("RGB").copy()
    draw = ImageDraw.Draw(out)
    width, height = out.size
    starts = list(range(0, max(1, width - window_width + 1), stride))
    if starts[-1] != width - window_width:
        starts.append(width - window_width)
    for index, start in enumerate(starts):
        draw.rectangle(
            (start, 0, min(width - 1, start + window_width - 1), height - 1),
            outline=(255, 0, 0),
            width=1,
        )
        if index % 4 == 0:
            draw.text((start + 1, 2), str(index), fill=(255, 0, 0))
    return out, starts


def _windows_contact_sheet(model_input: Image.Image, starts, window_width=32):
    tiles = []
    for index, start in enumerate(starts):
        crop = model_input.crop((start, 0, start + window_width, model_input.height))
        crop = crop.resize((48, 192), Image.Resampling.BILINEAR)
        tile = Image.new("RGB", (52, 210), "white")
        tile.paste(crop, (2, 16))
        draw = ImageDraw.Draw(tile)
        draw.text((2, 2), str(index), fill="black")
        tiles.append(tile)

    cols = 9
    rows = int(math.ceil(len(tiles) / cols))
    sheet = Image.new("RGB", (cols * 52, rows * 210), "white")
    for index, tile in enumerate(tiles):
        x = (index % cols) * 52
        y = (index // cols) * 210
        sheet.paste(tile, (x, y))
    return sheet


def _tensor_stats(model_input: Image.Image):
    tensor = transforms.ToTensor()(model_input)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    normalized = (tensor - mean) / std
    return {
        "shape": list(normalized.shape),
        "channel_min": [float(v) for v in normalized.amin(dim=(1, 2))],
        "channel_max": [float(v) for v in normalized.amax(dim=(1, 2))],
        "channel_mean": [float(v) for v in normalized.mean(dim=(1, 2))],
        "dtype": str(normalized.dtype),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--samples", type=int, default=10)
    args = ap.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    manifest = root / "dataset_manifest_full_pairs.jsonl"
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    dataset = ArabicManifestIndependentLineDataset(
        manifest,
        transform=None,
        text_key="text_original_path",
        validate_paths=False,
    )

    report = []
    for sample_index, sample in enumerate(dataset.samples[: args.samples]):
        image_path = dataset._resolve(sample["line_image_path"])
        text_path = dataset._resolve(sample["text_path"])

        with Image.open(image_path) as opened:
            original = opened.convert("RGB")

        # Detection is temporary and never fed into the network.
        mask, detection_meta = foreground_detection_mask_with_metadata(original)
        cropped, crop_meta = foreground_crop_with_metadata(original)
        model_input, resize_meta = aspect_preserving_pad_with_metadata(
            cropped,
            size=(128, 1024),
            target_ink_height_ratio=float(
                os.environ.get("ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", "0.72")
            ),
            horizontal_jitter=0.0,
        )

        box = (
            crop_meta["crop_left"],
            crop_meta["crop_top"],
            crop_meta["crop_right"] - 1,
            crop_meta["crop_bottom"] - 1,
        )

        stem = f"sample_{sample_index:03d}"
        original.save(output / f"{stem}_01_original_rgb.png")
        _save_mask(mask, output / f"{stem}_02_temporary_foreground_mask.png")
        _draw_bbox(original, box).save(output / f"{stem}_03_crop_bbox_overlay.png")
        cropped.save(output / f"{stem}_04_cropped_rgb.png")
        model_input.save(output / f"{stem}_05_model_input_1024x128_rgb.png")

        overlay, starts = _window_overlay(model_input)
        overlay.save(output / f"{stem}_06_model_input_window_overlay.png")
        _windows_contact_sheet(model_input, starts).save(
            output / f"{stem}_07_windows_contact_sheet.png"
        )

        with text_path.open("r", encoding="utf-8") as handle:
            transcript = handle.read().strip()

        report.append(
            {
                "sample_index": sample_index,
                "source_pair_id": sample.get("source_pair_id"),
                "source_page": sample.get("pair_id"),
                "side": sample.get("side"),
                "line_idx": sample.get("line_idx"),
                "source_image": str(image_path),
                "transcript_path": str(text_path),
                "transcript": transcript,
                "source_size": [original.width, original.height],
                "crop_box": [
                    crop_meta["crop_left"],
                    crop_meta["crop_top"],
                    crop_meta["crop_right"],
                    crop_meta["crop_bottom"],
                ],
                "cropped_size": [cropped.width, cropped.height],
                "model_input_size": [model_input.width, model_input.height],
                "binarization": False,
                "appearance": "original_rgb_preserved",
                "window_width": 32,
                "window_stride": 16,
                "window_count": len(starts),
                "crop": crop_meta,
                "detection": detection_meta,
                "resize": resize_meta,
                "normalized_tensor": _tensor_stats(model_input),
            }
        )

    (output / "diagnostic_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("RGB crop diagnostic complete")
    print("  samples      =", len(report))
    print("  output       =", output)
    print("  binarization = OFF")
    print("  crop mode    = robust_projection")
    print("  model input  = original RGB crop -> 1024x128")
    for row in report[:5]:
        print(
            f"  sample={row['sample_index']:03d} source={row['source_size']} "
            f"crop={row['crop_box']} cropped={row['cropped_size']} "
            f"windows={row['window_count']}"
        )


if __name__ == "__main__":
    main()
