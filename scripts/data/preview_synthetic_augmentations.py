#!/usr/bin/env python3
"""Save original-versus-augmented PNG previews for all synthetic sources.

Examples
--------
Direct no-DTW, geometry-preserving augmentation::

    python scripts/data/preview_synthetic_augmentations.py \
      --data-root DataSet --profile box-safe

Span-DTW zero-shot augmentation::

    python scripts/data/preview_synthetic_augmentations.py \
      --data-root DataSet --profile zero-shot
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_synthetic_dataloader import (  # noqa: E402
    BoxSafeSyntheticAugment,
    resolve_synthetic_data_dirs,
)
from zero_shot_preprocessing import build_preprocessor  # noqa: E402


_INDEX_PATTERN = re.compile(r"img1_(\d+)\.png$")
try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    _BILINEAR = Image.BILINEAR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create contact sheets and full-resolution PNG variants showing the "
            "actual synthetic training augmentations."
        )
    )
    parser.add_argument(
        "--data-root",
        default=str(ROOT / "DataSet"),
        help="DataSet directory containing Synthetic_Arabic_1 through _4.",
    )
    parser.add_argument(
        "--profile",
        choices=("box-safe", "zero-shot"),
        default="box-safe",
        help=(
            "box-safe preserves renderer subword intervals; zero-shot uses the "
            "crop/resize/rotate manuscript augmentation used with Span-DTW."
        ),
    )
    parser.add_argument("--samples-per-source", type=int, default=2)
    parser.add_argument("--augmentations", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "Results" / "AugmentationPreview"),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open each contact sheet with the operating-system image viewer.",
    )
    return parser.parse_args()


def numeric_image_paths(images_dir: Path) -> list[Path]:
    indexed = []
    for path in images_dir.glob("img1_*.png"):
        match = _INDEX_PATTERN.fullmatch(path.name)
        if match:
            indexed.append((int(match.group(1)), path))
    return [path for _index, path in sorted(indexed)]


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))


def label_tile(image: Image.Image, label: str, width=512, height=64) -> Image.Image:
    resized = image.convert("RGB").resize((width, height), _BILINEAR)
    label_height = 24
    tile = Image.new("RGB", (width, height + label_height), "white")
    tile.paste(resized, (0, label_height))
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default()
    draw.text((6, 6), label, fill="black", font=font)
    return tile


def make_contact_sheet(rows: list[list[tuple[str, Image.Image]]]) -> Image.Image:
    if not rows:
        raise ValueError("No preview rows were created")
    tile_width, tile_height = 512, 88
    columns = max(len(row) for row in rows)
    sheet = Image.new("RGB", (columns * tile_width, len(rows) * tile_height), "white")
    for row_index, row in enumerate(rows):
        for column_index, (label, image) in enumerate(row):
            sheet.paste(
                label_tile(image, label, tile_width, tile_height - 24),
                (column_index * tile_width, row_index * tile_height),
            )
    return sheet


def box_safe_variants(image: Image.Image, count: int, seed: int):
    target_size = (1024, 128)
    original = image.convert("RGB").resize(target_size, _BILINEAR)
    results = [("original", original)]
    augmenter = BoxSafeSyntheticAugment()
    for index in range(count):
        seed_everything(seed + index)
        # This matches training: appearance/stroke augmentation first, then the
        # fixed 1024x128 resize. No crop, rotation, translation, or random scale.
        augmented = augmenter(image.copy()).resize(target_size, _BILINEAR)
        results.append((f"augmented_{index + 1}", augmented))
    return results


def zero_shot_variants(image: Image.Image, count: int, seed: int):
    seed_everything(seed)
    clean = build_preprocessor("synthetic", training=False)(image.copy())
    results = [("clean", clean)]
    for index in range(count):
        seed_everything(seed + index)
        augmented = build_preprocessor("synthetic", training=True)(image.copy())
        results.append((f"augmented_{index + 1}", augmented))
    return results


def save_full_resolution_variants(
    output_dir: Path,
    source_name: str,
    sample_name: str,
    variants: list[tuple[str, Image.Image]],
) -> None:
    sample_dir = output_dir / source_name / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)
    for label, image in variants:
        image.save(sample_dir / f"{label}.png")


def main() -> None:
    args = parse_args()
    if args.samples_per_source <= 0:
        raise SystemExit("--samples-per-source must be positive")
    if args.augmentations <= 0:
        raise SystemExit("--augmentations must be positive")

    # Force discovery from the supplied root rather than an unrelated training
    # environment variable left in the shell.
    os.environ.pop("SYNTHETIC_DATA_DIRS", None)
    source_dirs = resolve_synthetic_data_dirs(args.data_root)
    if len(source_dirs) != 4:
        raise SystemExit(
            "Expected four sources Synthetic_Arabic_1 through _4, found: "
            + ", ".join(str(path) for path in source_dirs)
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_sheets = []

    for source_index, source_dir in enumerate(source_dirs):
        paths = numeric_image_paths(source_dir / "images")
        if len(paths) < args.samples_per_source:
            raise SystemExit(
                f"{source_dir} has only {len(paths)} img1 PNG files; "
                f"{args.samples_per_source} were requested"
            )
        chooser = random.Random(args.seed + source_index)
        selected = chooser.sample(paths, args.samples_per_source)
        rows = []
        for sample_offset, image_path in enumerate(selected):
            with Image.open(image_path) as opened:
                original = opened.convert("RGB")
            sample_seed = args.seed + source_index * 10000 + sample_offset * 100
            if args.profile == "box-safe":
                variants = box_safe_variants(
                    original, args.augmentations, sample_seed
                )
            else:
                variants = zero_shot_variants(
                    original, args.augmentations, sample_seed
                )
            save_full_resolution_variants(
                output_dir,
                source_dir.name,
                image_path.stem,
                variants,
            )
            rows.append(
                [(f"{image_path.name}: {label}", image) for label, image in variants]
            )

        sheet = make_contact_sheet(rows)
        sheet_path = output_dir / f"{source_dir.name}_{args.profile}_preview.png"
        sheet.save(sheet_path)
        contact_sheets.append((source_dir.name, sheet_path, sheet))
        print(f"Saved {sheet_path}")
        if args.show:
            sheet.show(title=f"{source_dir.name} {args.profile} augmentation")

    manifest = output_dir / f"{args.profile}_preview_files.txt"
    manifest.write_text(
        "\n".join(f"{name}: {path}" for name, path, _sheet in contact_sheets) + "\n",
        encoding="utf-8",
    )
    print(f"Full-resolution variants: {output_dir}/Synthetic_Arabic_*/img1_*/")
    print(f"Preview manifest: {manifest}")


if __name__ == "__main__":
    main()
