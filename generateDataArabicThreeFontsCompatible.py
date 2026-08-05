#!/usr/bin/env python3
"""Run the three-font generator with generateDataArabic.py-compatible outputs.

This wrapper keeps the exact-text, compact three-font augmentation logic while
matching the original Arabic generator's image/mask conventions:

- 1024x128 images, white text on black;
- paired img1/img2, mask1/mask2, and text1/text2 filenames;
- masks cover the shared phrase's complete horizontal range over full height;
- no matrices, similarity matrices, or subword-box outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

import generateDataArabicThreeFonts as generator


def full_height_shared_mask(
    metadata: list[dict],
    image_size: tuple[int, int],
) -> Image.Image:
    """Create a full-height mask from shared-segment horizontal boxes."""
    width, height = image_size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    for segment in metadata:
        if segment.get("role") != "shared":
            continue
        box = segment.get("box")
        if not box or len(box) != 4:
            raise ValueError("Shared segment metadata is missing its render box")
        left, _, right, _ = (int(value) for value in box)
        draw.rectangle((left, 0, right, height), fill=255)
    return mask


def output_dirs(root: Path) -> dict[str, Path]:
    """Create only the directories used by generateDataArabic.py training data."""
    paths = {name: root / name for name in ("images", "masks", "texts")}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def render_line(plan, rng, args):
    image, _, metadata = _ORIGINAL_RENDER_LINE(plan, rng, args)
    mask = full_height_shared_mask(metadata, image.size)
    clean_metadata = [
        {
            "text": segment["text"],
            "role": segment["role"],
            "font": segment["font"],
        }
        for segment in metadata
    ]
    return image, mask, clean_metadata


def save(index, plan, rendered1, rendered2, paths, skip_matrices=False):
    """Save only paired images, full-height masks, exact texts, and metadata."""
    del skip_matrices
    image1, mask1, metadata1 = rendered1
    image2, mask2, metadata2 = rendered2

    image1.save(paths["images"] / f"img1_{index}.png")
    image2.save(paths["images"] / f"img2_{index}.png")
    mask1.save(paths["masks"] / f"mask1_{index}.png")
    mask2.save(paths["masks"] / f"mask2_{index}.png")
    (paths["texts"] / f"text1_{index}.txt").write_text(
        plan.line1.text, encoding="utf-8"
    )
    (paths["texts"] / f"text2_{index}.txt").write_text(
        plan.line2.text, encoding="utf-8"
    )

    return {
        "sample_index": index,
        "mode": plan.mode,
        "text1": plan.line1.text,
        "text2": plan.line2.text,
        "shared_text": plan.shared,
        "primary_font": plan.primary_font.name,
        "secondary_font": plan.secondary_font.name,
        "line1_segments": metadata1,
        "line2_segments": metadata2,
    }


def write_compatible_summary(root: Path) -> None:
    summary_path = root / "generation_summary.json"
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary.update(
        {
            "image_size": [1024, 128],
            "font_size": 90,
            "full_height_masks": True,
            "generated_directories": ["images", "masks", "texts"],
            "omitted_outputs": [
                "matrices",
                "similarity_matrices",
                "subword_boxes",
            ],
        }
    )
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    generator.output_dirs = output_dirs
    generator.render_line = render_line
    generator.save = save
    generator.main()

    args = generator.parse_args()
    write_compatible_summary(args.output_dir.expanduser().resolve())


_ORIGINAL_RENDER_LINE = generator.render_line


if __name__ == "__main__":
    main()
