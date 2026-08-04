#!/usr/bin/env python3
"""Save PNG previews for dense aligned and unaligned augmentation modes."""
from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import SyntheticAugmentedDataLoader as augmented_loader  # noqa: E402
import DenseSyntheticAugmentation  # noqa: E402,F401  # patches augmented_loader


try:
    _BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    _BILINEAR = Image.BILINEAR


def _set_defaults() -> None:
    defaults = {
        "SYNTHETIC_DATASET_FOLDERS": (
            "Synthetic_Arabic_1,Synthetic_Arabic_2,Synthetic_Arabic_3"
        ),
        "SYNTHETIC_INJECTION_PROB": "0.35",
        "SYNTHETIC_TWO_REGION_PROB": "0.30",
        "SYNTHETIC_ALIGNED_UNALIGNED_PROB": "0.25",
        "SYNTHETIC_FRAGMENT_MIN_FRACTION": "0.20",
        "SYNTHETIC_FRAGMENT_MAX_FRACTION": "0.40",
        "SYNTHETIC_SOURCE_GAP_FRACTION": "0.04",
        "SYNTHETIC_CANVAS_GAP_FRACTION": "0.08",
        "SYNTHETIC_MISMATCH_SPAN_DISTANCE": "0.12",
        "SYNTHETIC_SCALE_MIN": "0.90",
        "SYNTHETIC_SCALE_MAX": "1.00",
        "SYNTHETIC_TRANSLATE_PCT": "0.04",
        "SYNTHETIC_CONTRAST": "0.10",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview full-line, two-region, cross-injection, and "
            "aligned-plus-unaligned augmentations."
        )
    )
    parser.add_argument("--data-root", default="DataSet")
    parser.add_argument("--folders", nargs="*", default=None)
    parser.add_argument("--samples-per-folder", type=int, default=3)
    parser.add_argument("--random-copies", type=int, default=2)
    parser.add_argument(
        "--output-dir",
        default="Results/AugmentationPreview_Arabic_1_3",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def _load(record):
    image1 = augmented_loader._load_rgb(record.image1_path)
    text1 = augmented_loader._read_text(record.text1_path)
    image2 = (
        augmented_loader._load_rgb(record.image2_path)
        if record.image2_path
        else None
    )
    text2 = (
        augmented_loader._read_text(record.text2_path)
        if record.text2_path
        else None
    )
    return (image1, image2), (text1, text2)


def _safe_text(text):
    return "" if text is None else " ".join(text.strip().split())


def main() -> None:
    _set_defaults()
    args = parse_args()
    if args.samples_per_folder <= 0:
        raise ValueError("--samples-per-folder must be positive")
    if args.random_copies < 0:
        raise ValueError("--random-copies cannot be negative")

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = (
        tuple(args.folders)
        if args.folders
        else augmented_loader.synthetic_folder_names()
    )
    _, folders = augmented_loader.resolve_synthetic_folders(
        args.data_root,
        names,
    )
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    random.seed(args.seed)
    augmentor = augmented_loader.SyntheticPairAugmentor.from_env()
    records = {}
    for folder in folders:
        count = augmented_loader._sample_count(folder)
        if count < args.samples_per_folder:
            raise ValueError(
                f"{folder.name} has {count} samples; "
                f"{args.samples_per_folder} requested"
            )
        records[folder.name] = [
            augmented_loader._make_record(folder, index)
            for index in rng.sample(range(count), args.samples_per_folder)
        ]

    fixed = [
        "original",
        "full_line",
        "two_region",
        "cross_injection",
        "aligned_unaligned",
    ]
    random_labels = [
        f"random_{index + 1}"
        for index in range(args.random_copies)
    ]
    columns = fixed + random_labels
    manifest = []

    for folder_index, folder in enumerate(folders):
        targets = records[folder.name]
        donor_folder = folders[(folder_index + 1) % len(folders)]
        donor_pool = records[donor_folder.name]
        paired_rows = 2 if targets[0].paired else 1
        figure, axes = plt.subplots(
            len(targets) * paired_rows,
            len(columns),
            figsize=(4.2 * len(columns), 2.25 * len(targets) * paired_rows),
            squeeze=False,
        )
        folder_output = output_root / folder.name
        folder_output.mkdir(parents=True, exist_ok=True)

        for offset, target_record in enumerate(targets):
            donor_record = donor_pool[offset % len(donor_pool)]
            target_images, target_texts = _load(target_record)
            donor_images, donor_texts = _load(donor_record)
            rendered = {
                "original": {
                    "mode": "original",
                    "image1": target_images[0].resize(
                        (augmentor.appearance.width, augmentor.appearance.height),
                        _BILINEAR,
                    ),
                    "image2": (
                        target_images[1].resize(
                            (
                                augmentor.appearance.width,
                                augmentor.appearance.height,
                            ),
                            _BILINEAR,
                        )
                        if target_images[1] is not None
                        else None
                    ),
                    "text1": target_texts[0],
                    "text2": target_texts[1],
                }
            }
            for mode in fixed[1:]:
                rendered[mode] = augmentor.apply(
                    target_images,
                    target_texts,
                    donor_images,
                    donor_texts,
                    mode,
                )
            for label in random_labels:
                rendered[label] = augmentor.apply(
                    target_images,
                    target_texts,
                    donor_images,
                    donor_texts,
                    None,
                )

            for column, label in enumerate(columns):
                result = rendered[label]
                for line_number, key in ((1, "image1"), (2, "image2")):
                    image = result[key]
                    if image is None:
                        continue
                    filename = (
                        f"sample_{target_record.sample_index:05d}_"
                        f"{label}_line{line_number}.png"
                    )
                    image.save(folder_output / filename)
                    row = offset * paired_rows + line_number - 1
                    axes[row, column].imshow(image)
                    axes[row, column].set_title(
                        f"{label}\nline {line_number}"
                        if line_number == 1
                        else f"line {line_number}"
                    )
                    axes[row, column].axis("off")

                manifest.append(
                    {
                        "target_folder": target_record.folder_name,
                        "target_sample": target_record.sample_index,
                        "donor_folder": donor_record.folder_name,
                        "donor_sample": donor_record.sample_index,
                        "preview_label": label,
                        "actual_mode": result["mode"],
                        "text1": _safe_text(result["text1"]),
                        "text2": _safe_text(result["text2"]),
                    }
                )

        figure.suptitle(
            f"{folder.name}: dense aligned and unaligned augmentation",
            fontsize=14,
        )
        figure.tight_layout()
        grid_path = (
            output_root
            / f"{folder.name}_all_augmentation_modes.png"
        )
        figure.savefig(grid_path, dpi=160, bbox_inches="tight")
        print(f"Saved {grid_path}", flush=True)
        if args.show:
            plt.show()
        plt.close(figure)

    manifest_path = output_root / "augmentation_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(manifest[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Saved {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
