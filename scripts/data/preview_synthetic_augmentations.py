#!/usr/bin/env python3
"""Save and optionally display synthetic Arabic augmentation previews."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageOps


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SyntheticAugmentedDataLoader import (  # noqa: E402
    SyntheticScaleTranslateAugment,
    resolve_synthetic_folders,
    synthetic_folder_names,
)

try:
    _RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
except AttributeError:  # Pillow < 9
    _RESAMPLE_BILINEAR = Image.BILINEAR


def _natural_image_key(path: Path) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0


def _load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview the same scale/translation augmentation used by "
            "SyntheticAugmentedDataLoader."
        )
    )
    parser.add_argument(
        "--data-root",
        default="DataSet",
        help="Parent of Synthetic_Arabic_1..4 (default: DataSet).",
    )
    parser.add_argument(
        "--folders",
        nargs="*",
        default=None,
        help="Optional folder-name override.",
    )
    parser.add_argument("--samples-per-folder", type=int, default=3)
    parser.add_argument("--augmentations-per-image", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        default="Results/AugmentationPreview",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also open the Matplotlib preview window.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.samples_per_folder <= 0:
        raise ValueError("--samples-per-folder must be positive.")
    if args.augmentations_per_image <= 0:
        raise ValueError("--augmentations-per-image must be positive.")

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
    augmentor = SyntheticScaleTranslateAugment.from_env()

    for folder in folders:
        image_paths = sorted(
            (folder / "images").glob("img1_*.png"),
            key=_natural_image_key,
        )
        if len(image_paths) < args.samples_per_folder:
            raise ValueError(
                f"{folder.name} has {len(image_paths)} img1 PNGs, but "
                f"{args.samples_per_folder} were requested."
            )
        selected = rng.sample(image_paths, args.samples_per_folder)
        rows = args.samples_per_folder
        columns = args.augmentations_per_image + 1
        figure, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.0 * columns, 2.4 * rows),
            squeeze=False,
        )
        folder_output = output_root / folder.name
        folder_output.mkdir(parents=True, exist_ok=True)

        for row, source_path in enumerate(selected):
            original = _load_rgb(source_path)
            resized_original = original.resize(
                (augmentor.width, augmentor.height),
                _RESAMPLE_BILINEAR,
            )
            original_path = folder_output / f"{source_path.stem}_original.png"
            resized_original.save(original_path)

            axes[row, 0].imshow(resized_original)
            axes[row, 0].set_title(f"{source_path.name}\noriginal")
            axes[row, 0].axis("off")

            for column in range(1, columns):
                augmented = augmentor(original.copy())
                augmented_path = folder_output / (
                    f"{source_path.stem}_aug_{column:02d}.png"
                )
                augmented.save(augmented_path)
                axes[row, column].imshow(augmented)
                axes[row, column].set_title(f"augmentation {column}")
                axes[row, column].axis("off")

        figure.suptitle(
            f"{folder.name}: scale [{augmentor.scale_min:.2f}, "
            f"{augmentor.scale_max:.2f}], translation "
            f"±{augmentor.translate_pct:.1%}",
            fontsize=14,
        )
        figure.tight_layout()
        grid_path = output_root / f"{folder.name}_augmentation_grid.png"
        figure.savefig(grid_path, dpi=160, bbox_inches="tight")
        print(f"Saved {grid_path}", flush=True)
        if args.show:
            plt.show()
        plt.close(figure)


if __name__ == "__main__":
    main()
