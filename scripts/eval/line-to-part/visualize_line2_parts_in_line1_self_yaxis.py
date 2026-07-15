#!/usr/bin/env python3
"""Run line-to-part visualization with self-window-cosine-style heatmap axes.

This wrapper keeps the original line-to-part logic, but replaces the heatmap
axis display with behavior close to:

    scripts/eval/window-similarity/visualize_line_self_window_cosine.py

It adds environment controls for independently reversing the displayed x/y axes
while keeping the heatmap values, SW path, selected cells, window thumbnails, and
token labels aligned with the displayed order.

Useful environment variables:
  HEATMAP_REVERSE_X_AXIS=1   reverse the displayed line1 x-axis/heatmap columns
  HEATMAP_REVERSE_Y_AXIS=1   reverse the displayed part y-axis/heatmap rows
  HEATMAP_Y_AXIS_ROTATE=1    rotate y-axis window thumbnails, self-cosine style
  HEATMAP_Y_AXIS_FLIP=1      flip y-axis thumbnails after rotation

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


def _axis_indices(num_windows: int, args, axis: str) -> np.ndarray:
    """Return the displayed model indices for one axis.

    The base script already handles model-vs-visual order. This wrapper adds the
    same independent reverse controls used by the self-window cosine script.
    """
    indices = np.asarray(base.display_indices(num_windows, args), dtype=np.int64)
    if axis == "x":
        reverse = _env_flag("HEATMAP_REVERSE_X_AXIS", False)
    elif axis == "y":
        reverse = _env_flag("HEATMAP_REVERSE_Y_AXIS", False)
    else:  # pragma: no cover - defensive only
        raise ValueError(f"unknown axis: {axis!r}")
    if reverse:
        indices = indices[::-1].copy()
    return indices


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


def save_heatmap_with_reversible_axes(
    sim,
    sw_path,
    seg,
    part: base.SourcePart,
    threshold: float,
    args,
    line1_display=None,
    part_display=None,
    line1_labels=None,
    part_labels=None,
) -> str:
    """Base save_heatmap with independent x/y reverse controls.

    This keeps all data structures aligned after reversing:
      - sim_display columns/rows
      - top/side window thumbnails
      - token labels
      - SW path and selected red cells
      - model-id tick labels
    """
    out = base.heatmap_output_path(args.output, args.heatmap_dir, part.part_id)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    axis_images = args.heatmap_axis_images and line1_display is not None and part_display is not None
    n_line1, n_part = int(sim.shape[0]), int(sim.shape[1])

    line1_indices = _axis_indices(n_line1, args, "x")
    part_indices = _axis_indices(n_part, args, "y")
    line1_pos = {int(model_idx): int(pos) for pos, model_idx in enumerate(line1_indices)}
    part_pos = {int(model_idx): int(pos) for pos, model_idx in enumerate(part_indices)}
    sim_display = sim[np.ix_(line1_indices, part_indices)]
    path_pairs_display = base.transform_pairs(base.diag_pairs(sw_path), line1_pos, part_pos)
    selected_display = set(base.transform_pairs(base.selected_cell_set(sw_path, seg), line1_pos, part_pos))
    line1_labels = list(line1_labels) if line1_labels is not None else [""] * n_line1
    part_labels = list(part_labels) if part_labels is not None else [""] * n_part

    reverse_x = _env_flag("HEATMAP_REVERSE_X_AXIS", False)
    reverse_y = _env_flag("HEATMAP_REVERSE_Y_AXIS", False)
    reverse_note = []
    if reverse_x:
        reverse_note.append("x reversed")
    if reverse_y:
        reverse_note.append("y reversed")
    reverse_label = " | " + ", ".join(reverse_note) if reverse_note else ""

    if axis_images:
        fig = base.plt.figure(
            figsize=(max(13, min(38, n_line1 * 0.38 + 5)), max(8.0, min(21, n_part * 0.88 + 6.0)))
        )
        gs = base.GridSpec(
            2,
            3,
            figure=fig,
            width_ratios=[2.55, 8.8, 0.34],
            height_ratios=[1.85, 6.0],
            wspace=0.08,
            hspace=0.10,
        )
        ax_corner = fig.add_subplot(gs[0, 0])
        ax_x = fig.add_subplot(gs[0, 1])
        ax_y = fig.add_subplot(gs[1, 0])
        ax_h = fig.add_subplot(gs[1, 1])
        cax = fig.add_subplot(gs[1, 2])
        ax_corner.axis("off")
        ax_corner.text(
            0.5,
            0.5,
            f"separated\nwindow slices\nmode={args.heatmap_axis_slice_mode}\n"
            f"display={args.heatmap_display_order}{reverse_label}",
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
        )
        ax_x.imshow(base.make_x_strip(line1_display, line1_indices, n_line1, args), extent=(-0.5, n_line1 - 0.5, 1, 0), aspect="auto")
        ax_y.imshow(make_y_strip_self_window_style(part_display, part_indices, n_part, args), extent=(0, 1, n_part - 0.5, -0.5), aspect="auto")
        ax_x.set_xlim(-0.5, n_line1 - 0.5)
        ax_x.set_ylim(1, 0)
        ax_x.set_yticks([])
        ax_y.set_xlim(0, 1)
        ax_y.set_ylim(n_part - 0.5, -0.5)
        ax_y.set_xticks([])
        base.draw_axis_tokens(ax_x, ax_y, line1_labels, part_labels, line1_indices, part_indices, args)

        xs = base.label_step(n_line1)
        ys = 1 if n_part <= 16 else base.label_step(n_part)
        ax_x.set_xticks(list(range(0, n_line1, xs)))
        ax_x.set_xticklabels([str(int(line1_indices[i])) for i in range(0, n_line1, xs)], fontsize=7)
        ax_x.tick_params(axis="x", labeltop=True, labelbottom=False, pad=1)
        ax_x.set_title("line1 windows: labels are model ids; images are unmirrored visual crops", fontsize=9)
        ax_y.set_yticks(list(range(0, n_part, ys)))
        ax_y.set_yticklabels([str(int(part_indices[i])) for i in range(0, n_part, ys)], fontsize=7)
        ax_y.set_ylabel("part windows\nmodel ids + tokens", fontsize=8)
        base.grid_cells(ax_x, n_line1, None, alpha=0.60, lw=0.55)
        for y in np.arange(-0.5, n_part + 0.5, 1.0):
            ax_y.axhline(y, color="white", lw=0.55, alpha=0.65, zorder=10)

        im = ax_h.imshow(sim_display.T, origin="upper", aspect="auto", vmin=args.heatmap_vmin, vmax=args.heatmap_vmax)
        base.plt.colorbar(im, cax=cax).set_label("cosine similarity")
        base.grid_cells(ax_h, n_line1, n_part, alpha=0.24, lw=0.40)
        base.annotate_heatmap_cells(ax_h, sim_display, threshold, selected_display, args)
        dx, dy = base.draw_path(ax_h, path_pairs_display, selected_display, seg)
        ax_h.set_xlim(-0.5, n_line1 - 0.5)
        ax_h.set_ylim(n_part - 0.5, -0.5)
        ax_h.set_xlabel(f"line1 displayed windows ({args.heatmap_display_order} order; tick=model id{'; reversed' if reverse_x else ''})")
        ax_h.set_ylabel(f"part displayed windows ({args.heatmap_display_order} order; tick=model id{'; reversed' if reverse_y else ''})")
        ax_h.set_xticks(list(range(0, n_line1, xs)))
        ax_h.set_xticklabels([str(int(line1_indices[i])) for i in range(0, n_line1, xs)], fontsize=8)
        ax_h.set_yticks(list(range(0, n_part, ys)))
        ax_h.set_yticklabels([str(int(part_indices[i])) for i in range(0, n_part, ys)], fontsize=8)
        if dx or dy or seg is not None:
            ax_h.legend(loc="upper right", fontsize=8)
    else:
        fig, ax_h = base.plt.subplots(figsize=(max(9, min(26, n_line1 / 3.6)), max(4.2, min(13, n_part / 1.8 + 2))))
        im = ax_h.imshow(sim_display.T, origin="upper", aspect="auto", vmin=args.heatmap_vmin, vmax=args.heatmap_vmax)
        base.plt.colorbar(im, ax=ax_h, fraction=0.025, pad=0.02).set_label("cosine similarity")
        base.grid_cells(ax_h, n_line1, n_part, alpha=0.24, lw=0.40)
        base.annotate_heatmap_cells(ax_h, sim_display, threshold, selected_display, args)
        dx, dy = base.draw_path(ax_h, path_pairs_display, selected_display, seg)
        ax_h.set_xlim(-0.5, n_line1 - 0.5)
        ax_h.set_ylim(n_part - 0.5, -0.5)
        ax_h.set_xlabel(f"line1 displayed window index{'; reversed' if reverse_x else ''}")
        ax_h.set_ylabel(f"part displayed window index{'; reversed' if reverse_y else ''}")
        if dx or dy or seg is not None:
            ax_h.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Part {part.part_id} vs line1 cosine similarity | "
        f"line2_x=[{part.x0_original},{part.x1_original}] | thr={threshold:.3f} | "
        f"orange=sim≥thr, red=chosen SW cells{reverse_label}",
        fontsize=11,
    )
    base.plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    base.plt.close(fig)
    print(f"Saved cosine heatmap for part {part.part_id}: {out}")
    return out


def main() -> None:
    base.make_y_strip = make_y_strip_self_window_style
    base.save_heatmap = save_heatmap_with_reversible_axes
    base.main()


if __name__ == "__main__":
    main()
