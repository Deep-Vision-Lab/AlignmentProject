#!/usr/bin/env python3
"""Run line-to-part visualization with the same y-axis style as self-window cosine.

This wrapper keeps the original line-to-part logic, but replaces the heatmap y-axis
window strip with the same behavior used by:

    scripts/eval/window-similarity/visualize_line_self_window_cosine.py

Main differences from the older line-to-part y-axis:
  - same gap-aware stacked cells
  - same rotate-then-optional-flip behavior
  - optional mirroring controlled by the existing --heatmap-mirror-part-axis-windows
  - default y-axis rotation enabled, y-axis flip disabled

The CLI is inherited from visualize_line2_parts_in_line1.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import visualize_line2_parts_in_line1 as base  # noqa: E402


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.lower() in {"1", "true", "yes", "on"}


def make_y_strip_self_window_style(image: Image.Image, indices, n: int, args) -> np.ndarray:
    """Match visualize_line_self_window_cosine.py's y-axis strip style."""
    cells = []
    gap = int(args.heatmap_window_gap_pixels)
    mirror = bool(getattr(args, "heatmap_mirror_part_axis_windows", False))
    y_axis_rotate = _env_flag("HEATMAP_Y_AXIS_ROTATE", True)
    y_axis_flip = _env_flag("HEATMAP_Y_AXIS_FLIP", False)

    for model_idx in indices:
        crop = base.crop_visual_slice_for_model_window(image, int(model_idx), n, args)
        if mirror:
            crop = ImageOps.mirror(crop)
        if y_axis_rotate:
            crop = crop.transpose(base._ROTATE_90)
        if y_axis_flip:
            crop = ImageOps.mirror(crop)
        crop = crop.resize(
            (int(args.heatmap_part_strip_width), int(args.heatmap_axis_cell_pixels)),
            base._RESAMPLE,
        )
        cell = Image.new(
            "RGB",
            (
                int(args.heatmap_part_strip_width),
                int(args.heatmap_axis_cell_pixels) + gap,
            ),
            (255, 255, 255),
        )
        cell.paste(crop, (0, 0))
        cells.append(cell)

    if cells:
        cells[-1] = cells[-1].crop(
            (
                0,
                0,
                int(args.heatmap_part_strip_width),
                int(args.heatmap_axis_cell_pixels),
            )
        )
    return np.array(base.vstack(cells))


def main() -> None:
    base.make_y_strip = make_y_strip_self_window_style
    base.main()


if __name__ == "__main__":
    main()
