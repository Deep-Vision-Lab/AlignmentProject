#!/usr/bin/env python3
"""Run the three-font generator with generateDataArabic.py-compatible outputs.

This wrapper keeps the exact-text, compact three-font augmentation logic while
matching the original Arabic generator's image/mask conventions:

- 1024x128 images, white text on black;
- paired img1/img2, mask1/mask2, and text1/text2 filenames;
- masks cover the shared phrase's complete horizontal range over full height;
- no matrices, similarity matrices, or subword-box outputs;
- Arabic words retain their internal characters without inserted spaces;
- the same box/control-character safeguards from generateDataArabic.py are
  applied before right-to-left rendering.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

import generateDataArabicThreeFonts as generator


_RENDER_CONTROL_CODEPOINTS = {
    0x200B,
    0x200C,
    0x200D,
    0x200E,
    0x200F,
    0x202A,
    0x202B,
    0x202C,
    0x202D,
    0x202E,
    0x2066,
    0x2067,
    0x2068,
    0x2069,
    0xFEFF,
}


def _remove_render_controls(text: str) -> str:
    """Remove invisible controls that can render as '?' or tofu boxes."""
    return "".join(
        character
        for character in text
        if character.isprintable()
        and ord(character) not in _RENDER_CONTROL_CODEPOINTS
    )


def clean_text(text: str) -> list[str]:
    """Normalize Arabic while preserving characters inside each word."""
    text = unicodedata.normalize("NFKC", text)
    text = generator.DIACRITICS.sub("", text)
    text = _remove_render_controls(text)
    return text.split()


def normalize(text: str) -> str:
    """Collapse real whitespace without separating Arabic characters."""
    return " ".join(clean_text(text))


def join_text(parts: Iterable[str]) -> str:
    """Join logical phrase segments using one real inter-phrase space."""
    return " ".join(
        part for part in (normalize(value) for value in parts) if part
    )


def visual_text(text: str) -> str:
    """Shape Arabic with the safeguards used by generateDataArabic.py."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError as exc:
        raise RuntimeError(
            "Install arabic-reshaper and python-bidi before generation."
        ) from exc

    reshaper = arabic_reshaper.ArabicReshaper(
        configuration={
            "delete_harakat": True,
            "support_zwj": False,
            "delete_at_sign": True,
            "use_unshaped_instead_of_isolated": True,
        }
    )
    shaped = reshaper.reshape(normalize(text))
    shaped = _remove_render_controls(shaped)
    return get_display(shaped)


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
            "connected_arabic_shaping": True,
            "generateDataArabic_box_cleanup": True,
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
    generator.clean_text = clean_text
    generator.normalize = normalize
    generator.join_text = join_text
    generator.visual_text = visual_text
    generator.output_dirs = output_dirs
    generator.render_line = render_line
    generator.save = save
    generator.main()

    args = generator.parse_args()
    write_compatible_summary(args.output_dir.expanduser().resolve())


_ORIGINAL_RENDER_LINE = generator.render_line


if __name__ == "__main__":
    main()
