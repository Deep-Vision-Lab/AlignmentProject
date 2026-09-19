#!/usr/bin/env python3
"""Preview ArabicDataset with the synthetic-style real preprocessing."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

os.environ.setdefault("REAL_SYNTHETIC_STYLE", "1")
os.environ.setdefault("ZERO_SHOT_PREPROCESS", "1")
os.environ.setdefault("ZERO_SHOT_FOREGROUND_CROP", "1")
os.environ.setdefault("ZERO_SHOT_PRESERVE_ASPECT", "1")
os.environ.setdefault("ZERO_SHOT_TARGET_INK_HEIGHT_RATIO", "0.72")
os.environ.setdefault("REAL_BINARIZE", "1")
os.environ.setdefault("REAL_BINARIZE_AUTO_INVERT", "1")
os.environ.setdefault("REAL_BINARIZE_AUTOCONTRAST", "1")

from RealDataSet import ArabicManifestIndependentLineDataset
from zero_shot_preprocessing import build_preprocessor


def _save_contact(raw: Image.Image, processed: Image.Image, path: Path) -> None:
    raw_rgb = raw.convert("RGB")
    target_h = 128
    scale = target_h / max(1, raw_rgb.height)
    raw_preview = raw_rgb.resize(
        (max(1, int(round(raw_rgb.width * scale))), target_h),
        Image.Resampling.BILINEAR,
    )
    canvas = Image.new(
        "RGB",
        (raw_preview.width + processed.width, max(raw_preview.height, processed.height)),
        "white",
    )
    canvas.paste(raw_preview, (0, 0))
    canvas.paste(processed, (raw_preview.width, 0))
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--samples", type=int, default=6)
    args = ap.parse_args()

    root = Path(args.dataset).expanduser().resolve()
    manifest = root / "dataset_manifest_full_pairs.jsonl"
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    dataset = ArabicManifestIndependentLineDataset(
        manifest,
        transform=None,
        text_key="text_original_path",
        max_samples=None,
        validate_paths=False,
    )
    preprocessor = build_preprocessor("real", training=False)

    report = []
    for sample_index, sample in enumerate(dataset.samples[: args.samples]):
        image_path = dataset._resolve(sample["line_image_path"])
        with Image.open(image_path) as opened:
            raw = opened.convert("RGB")
        processed, metadata = preprocessor.preprocess_with_metadata(raw)

        stem = f"sample_{sample_index:03d}"
        processed.save(output / f"{stem}_processed.png")
        _save_contact(raw, processed, output / f"{stem}_raw_vs_processed.png")

        pixels = np.asarray(processed.convert("L"), dtype=np.uint8)
        border = np.concatenate(
            [
                pixels[:4, :].reshape(-1),
                pixels[-4:, :].reshape(-1),
                pixels[:, :4].reshape(-1),
                pixels[:, -4:].reshape(-1),
            ]
        )
        report.append(
            {
                "sample_index": sample_index,
                "source_pair_id": sample.get("source_pair_id"),
                "source_page": sample.get("pair_id"),
                "side": sample.get("side"),
                "line_idx": sample.get("line_idx"),
                "source": str(image_path),
                "processed_size": list(processed.size),
                "border_mean": float(border.mean()),
                "foreground_max": int(pixels.max()),
                "foreground_mean": float(pixels.mean()),
                "metadata": metadata,
            }
        )

    (output / "preview_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved {len(report)} transformed sides to {output}")
    for row in report[:4]:
        print(
            f"{row['source_pair_id']} side={row['side']} line={row['line_idx']} "
            f"size={row['processed_size']} "
            f"border_mean={row['border_mean']:.2f} max={row['foreground_max']}"
        )


if __name__ == "__main__":
    main()
