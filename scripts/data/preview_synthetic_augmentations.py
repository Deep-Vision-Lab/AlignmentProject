#!/usr/bin/env python3
"""Save PNG previews for the lightweight synthetic augmentation modes.

The output includes original paired lines, mild full-line augmentation,
two separated regions from one line, cross-line donor injection, and optional
random mixed examples. The script works without a display on HPC nodes.
"""
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

from SyntheticAugmentedDataLoader import (  # noqa: E402
    SyntheticPairAugmentor,
    _load_rgb,
    _make_record,
    _read_text,
    _sample_count,
    resolve_synthetic_folders,
    synthetic_folder_names,
)

try:
    _RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    _RESAMPLE_BILINEAR = Image.BILINEAR


def _set_lightweight_defaults() -> None:
    """Match the default training launcher without overriding user choices."""
    defaults = {
        "SYNTHETIC_DATASET_FOLDERS": (
            "Synthetic_Arabic_1,Synthetic_Arabic_2,Synthetic_Arabic_3"
        ),
        "SYNTHETIC_AUGMENT_COPIES_PER_SAMPLE": "2",
        "SYNTHETIC_INJECTION_PROB": "0.50",
        "SYNTHETIC_TWO_REGION_PROB": "0.35",
        "SYNTHETIC_SCALE_MIN": "0.90",
        "SYNTHETIC_SCALE_MAX": "1.00",
        "SYNTHETIC_TRANSLATE_PCT": "0.04",
        "SYNTHETIC_CONTRAST": "0.10",
        "SYNTHETIC_ROTATION_DEGREES": "0",
        "SYNTHETIC_SHEAR_DEGREES": "0",
        "SYNTHETIC_BRIGHTNESS": "0",
        "SYNTHETIC_SHARPNESS": "0",
        "SYNTHETIC_BLUR_PROB": "0",
        "SYNTHETIC_NOISE_PROB": "0",
        "SYNTHETIC_MORPHOLOGY_PROB": "0",
        "SYNTHETIC_ERASE_PROB": "0",
    }
    for name, value in defaults.items():
        os.environ.setdefault(name, value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write PNG previews for mild full-line, two-region, and injection "
            "augmentations."
        )
    )
    parser.add_argument(
        "--data-root",
        default="DataSet",
        help="Parent directory of Synthetic_Arabic_1, _2, and _3.",
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="Optional folder-name override.",
    )
    parser.add_argument("--samples-per-folder", type=int, default=3)
    parser.add_argument(
        "--random-copies",
        type=int,
        default=2,
        help="Additional randomly selected augmentation modes per source sample.",
    )
    parser.add_argument(
        "--output-dir",
        default="Results/AugmentationPreview_Arabic_1_3",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open each comparison grid after saving it.",
    )
    return parser.parse_args()


def _load_record(record):
    image1 = _load_rgb(record.image1_path)
    text1 = _read_text(record.text1_path)
    image2 = _load_rgb(record.image2_path) if record.image2_path else None
    text2 = _read_text(record.text2_path) if record.text2_path else None
    return (image1, image2), (text1, text2)


def _safe_text(text: str | None) -> str:
    return "" if text is None else " ".join(text.strip().split())


def main() -> None:
    _set_lightweight_defaults()
    args = parse_args()
    if args.samples_per_folder <= 0:
        raise ValueError("--samples-per-folder must be positive.")
    if args.random_copies < 0:
        raise ValueError("--random-copies cannot be negative.")

    if not args.show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = tuple(args.folders) if args.folders else synthetic_folder_names()
    _, folders = resolve_synthetic_folders(args.data_root, names)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    random.seed(args.seed)
    augmentor = SyntheticPairAugmentor.from_env()

    records_by_folder = {}
    for folder in folders:
        count = _sample_count(folder)
        if count < args.samples_per_folder:
            raise ValueError(
                f"{folder.name} contains {count} samples, fewer than requested "
                f"{args.samples_per_folder}."
            )
        chosen = rng.sample(range(count), args.samples_per_folder)
        records_by_folder[folder.name] = [_make_record(folder, index) for index in chosen]

    manifest_rows = []
    fixed_modes = ["original", "full_line", "two_region", "cross_injection"]
    random_modes = [f"random_{index + 1}" for index in range(args.random_copies)]
    columns = fixed_modes + random_modes

    for folder_index, folder in enumerate(folders):
        targets = records_by_folder[folder.name]
        donor_folder = folders[(folder_index + 1) % len(folders)]
        donor_pool = records_by_folder[donor_folder.name]
        rows = len(targets) * (2 if targets[0].paired else 1)
        figure, axes = plt.subplots(
            rows,
            len(columns),
            figsize=(4.2 * len(columns), 2.25 * rows),
            squeeze=False,
        )
        folder_output = output_root / folder.name
        folder_output.mkdir(parents=True, exist_ok=True)

        for sample_offset, target_record in enumerate(targets):
            donor_record = donor_pool[sample_offset % len(donor_pool)]
            target_images, target_texts = _load_record(target_record)
            donor_images, donor_texts = _load_record(donor_record)
            rendered = {}

            original_image1 = target_images[0].resize(
                (augmentor.appearance.width, augmentor.appearance.height),
                _RESAMPLE_BILINEAR,
            )
            original_image2 = (
                target_images[1].resize(
                    (augmentor.appearance.width, augmentor.appearance.height),
                    _RESAMPLE_BILINEAR,
                )
                if target_images[1] is not None
                else None
            )
            rendered["original"] = {
                "image1": original_image1,
                "image2": original_image2,
                "text1": target_texts[0],
                "text2": target_texts[1],
                "mode": "original",
            }

            for mode in ("full_line", "two_region", "cross_injection"):
                rendered[mode] = augmentor.apply(
                    target_images=target_images,
                    target_texts=target_texts,
                    donor_images=donor_images,
                    donor_texts=donor_texts,
                    mode=mode,
                )
            for label in random_modes:
                rendered[label] = augmentor.apply(
                    target_images=target_images,
                    target_texts=target_texts,
                    donor_images=donor_images,
                    donor_texts=donor_texts,
                    mode=None,
                )

            for column_index, label in enumerate(columns):
                result = rendered[label]
                mode_name = str(result["mode"])
                for line_number, image_key in ((1, "image1"), (2, "image2")):
                    image = result[image_key]
                    if image is None:
                        continue
                    filename = (
                        f"sample_{target_record.sample_index:05d}_"
                        f"{label}_line{line_number}.png"
                    )
                    image.save(folder_output / filename)
                    row_index = (
                        sample_offset * (2 if target_record.paired else 1)
                        + line_number
                        - 1
                    )
                    axes[row_index, column_index].imshow(image)
                    axes[row_index, column_index].set_title(
                        f"{label}\nline {line_number}"
                        if line_number == 1
                        else f"line {line_number}"
                    )
                    axes[row_index, column_index].axis("off")

                manifest_rows.append(
                    {
                        "target_folder": target_record.folder_name,
                        "target_sample": target_record.sample_index,
                        "donor_folder": donor_record.folder_name,
                        "donor_sample": donor_record.sample_index,
                        "preview_label": label,
                        "actual_mode": mode_name,
                        "text1": _safe_text(result["text1"]),
                        "text2": _safe_text(result["text2"]),
                    }
                )

        figure.suptitle(
            f"{folder.name}: original, mild full-line, two regions, and injection",
            fontsize=14,
        )
        figure.tight_layout()
        grid_path = output_root / f"{folder.name}_all_augmentation_modes.png"
        figure.savefig(grid_path, dpi=160, bbox_inches="tight")
        print(f"Saved {grid_path}", flush=True)
        if args.show:
            plt.show()
        plt.close(figure)

    manifest_path = output_root / "augmentation_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(manifest_rows[0]),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Saved {manifest_path}", flush=True)
    print(
        "Each folder contains individual PNGs for every mode and paired line. "
        "The top-level grids make side-by-side inspection easy.",
        flush=True,
    )


if __name__ == "__main__":
    main()
