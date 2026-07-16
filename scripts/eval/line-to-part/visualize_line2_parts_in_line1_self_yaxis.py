#!/usr/bin/env python3
"""Run line-to-part visualization with self-window-cosine-style heatmap axes.

This wrapper keeps the original line-to-part search logic and changes only the
heatmap display/token helpers.

The y-axis is reversed by default so the last part window is displayed first
and the first part window is displayed last. The same ordering is applied to:

  - y-axis thumbnails
  - heatmap rows
  - token labels
  - model-index ticks
  - Smith-Waterman path
  - selected red cells

Environment controls:
  HEATMAP_REVERSE_X_AXIS=1   reverse line1 x-axis and heatmap columns
  HEATMAP_REVERSE_Y_AXIS=1   reverse part y-axis and heatmap rows (default)
  HEATMAP_Y_AXIS_ROTATE=1    rotate y-axis thumbnails (default)
  HEATMAP_Y_AXIS_FLIP=1      flip each thumbnail after rotation

Axis token labels first use hard Span-DTW. If the cropped part is too short for
a full transcript path, missing labels fall back to independent per-window
nearest-span assignment instead of being displayed as "?".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn.functional as F

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
    """Return model-window indices in the exact displayed order."""
    indices = np.asarray(base.display_indices(num_windows, args), dtype=np.int64)
    if axis == "x":
        reverse = _env_flag("HEATMAP_REVERSE_X_AXIS", False)
    elif axis == "y":
        # The desired line-to-part display is last part window first.
        reverse = _env_flag("HEATMAP_REVERSE_Y_AXIS", True)
    else:  # pragma: no cover
        raise ValueError(f"unknown axis: {axis!r}")
    return indices[::-1].copy() if reverse else indices.copy()


def _fallback_window_argmax_labels(
    encoding,
    image_embeddings_for_alignment: torch.Tensor,
) -> list[str]:
    """Assign the closest candidate text span independently to every window."""
    num_windows = int(image_embeddings_for_alignment.shape[0])
    texts = list(getattr(encoding, "texts", []) or [])
    span_embeddings = getattr(encoding, "embeddings", None)
    if num_windows <= 0:
        return []
    if span_embeddings is None or not texts:
        return [""] * num_windows

    with torch.no_grad():
        norm_img = F.normalize(image_embeddings_for_alignment.float(), p=2, dim=-1)
        norm_span = F.normalize(
            span_embeddings.float().to(norm_img.device),
            p=2,
            dim=-1,
        )
        best_span_indices = torch.matmul(norm_img, norm_span.T).argmax(dim=1)
        best_span_indices = best_span_indices.detach().cpu().tolist()

    return [
        str(texts[int(idx)]) if 0 <= int(idx) < len(texts) else "?"
        for idx in best_span_indices
    ]


def aligned_tokens_for_windows_robust(
    text_encoder,
    text: str | None,
    image_embeddings_for_alignment: torch.Tensor,
    args,
) -> list[str]:
    """Use hard Span-DTW labels and fill missing labels with per-window argmax."""
    num_windows = int(image_embeddings_for_alignment.shape[0])
    if text_encoder is None or not text:
        return [""] * num_windows

    try:
        encoding = base.encode_text(text_encoder, text)
    except Exception as exc:
        print(f"[warn] failed to encode axis text: {exc}", flush=True)
        return ["?"] * num_windows

    labels = ["?"] * num_windows
    filled = [False] * num_windows

    try:
        norm_img = F.normalize(image_embeddings_for_alignment.float(), p=2, dim=-1)
        path = base.hard_span_dtw_path(
            encoding,
            norm_img,
            temperature=float(args.token_temperature),
            max_windows=int(args.max_windows_per_span),
            window_count_penalty=float(args.window_count_penalty),
        )
        for step in path:
            span_idx = int(step.get("span_idx", -1))
            fallback = "?"
            if hasattr(encoding, "texts") and 0 <= span_idx < len(encoding.texts):
                fallback = str(encoding.texts[span_idx])
            token = str(step.get("text", fallback))
            start = max(0, int(step.get("window_start", 0)))
            end = min(num_windows, int(step.get("window_end", start + 1)))
            if end <= start:
                end = min(num_windows, start + 1)
            for window_idx in range(start, end):
                labels[window_idx] = token
                filled[window_idx] = True
    except Exception as exc:
        print(
            "[warn] hard Span-DTW axis tokens failed; "
            f"using per-window argmax fallback: {exc}",
            flush=True,
        )

    if not all(filled):
        try:
            fallback_labels = _fallback_window_argmax_labels(
                encoding,
                image_embeddings_for_alignment,
            )
            for window_idx, fallback in enumerate(fallback_labels):
                if window_idx >= num_windows:
                    break
                if not filled[window_idx] or labels[window_idx] == "?":
                    labels[window_idx] = fallback
                    filled[window_idx] = True
        except Exception as exc:
            print(f"[warn] per-window argmax axis-token fallback failed: {exc}", flush=True)

    return labels


def make_y_strip_self_window_style(
    image: Image.Image,
    indices,
    n: int,
    args,
) -> np.ndarray:
    """Draw y-axis thumbnails in the same style as self-window cosine."""
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
            (
                int(args.heatmap_part_strip_width),
                int(args.heatmap_axis_cell_pixels),
            ),
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
    """Save a heatmap with independent, fully synchronized x/y ordering."""
    out = base.heatmap_output_path(args.output, args.heatmap_dir, part.part_id)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    axis_images = (
        args.heatmap_axis_images
        and line1_display is not None
        and part_display is not None
    )
    n_line1, n_part = int(sim.shape[0]), int(sim.shape[1])

    line1_indices = _axis_indices(n_line1, args, "x")
    part_indices = _axis_indices(n_part, args, "y")
    line1_pos = {
        int(model_idx): int(pos)
        for pos, model_idx in enumerate(line1_indices)
    }
    part_pos = {
        int(model_idx): int(pos)
        for pos, model_idx in enumerate(part_indices)
    }

    # sim is [line1_window, part_window]. Reorder both dimensions before
    # transposing it for display as [part_row, line1_column].
    sim_display = sim[np.ix_(line1_indices, part_indices)]
    path_pairs_display = base.transform_pairs(
        base.diag_pairs(sw_path),
        line1_pos,
        part_pos,
    )
    selected_display = set(
        base.transform_pairs(
            base.selected_cell_set(sw_path, seg),
            line1_pos,
            part_pos,
        )
    )
    line1_labels = list(line1_labels) if line1_labels is not None else [""] * n_line1
    part_labels = list(part_labels) if part_labels is not None else [""] * n_part

    reverse_x = _env_flag("HEATMAP_REVERSE_X_AXIS", False)
    reverse_y = _env_flag("HEATMAP_REVERSE_Y_AXIS", True)
    reverse_note = []
    if reverse_x:
        reverse_note.append("x reversed")
    if reverse_y:
        reverse_note.append("y reversed")
    reverse_label = " | " + ", ".join(reverse_note) if reverse_note else ""

    if axis_images:
        fig = base.plt.figure(
            figsize=(
                max(13, min(38, n_line1 * 0.38 + 5)),
                max(8.0, min(21, n_part * 0.88 + 6.0)),
            )
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

        ax_x.imshow(
            base.make_x_strip(line1_display, line1_indices, n_line1, args),
            extent=(-0.5, n_line1 - 0.5, 1, 0),
            aspect="auto",
        )
        ax_y.imshow(
            make_y_strip_self_window_style(part_display, part_indices, n_part, args),
            extent=(0, 1, n_part - 0.5, -0.5),
            aspect="auto",
        )
        ax_x.set_xlim(-0.5, n_line1 - 0.5)
        ax_x.set_ylim(1, 0)
        ax_x.set_yticks([])
        ax_y.set_xlim(0, 1)
        ax_y.set_ylim(n_part - 0.5, -0.5)
        ax_y.set_xticks([])

        base.draw_axis_tokens(
            ax_x,
            ax_y,
            line1_labels,
            part_labels,
            line1_indices,
            part_indices,
            args,
        )

        xs = base.label_step(n_line1)
        ys = 1 if n_part <= 16 else base.label_step(n_part)
        ax_x.set_xticks(list(range(0, n_line1, xs)))
        ax_x.set_xticklabels(
            [str(int(line1_indices[i])) for i in range(0, n_line1, xs)],
            fontsize=7,
        )
        ax_x.tick_params(axis="x", labeltop=True, labelbottom=False, pad=1)
        ax_x.set_title(
            "line1 windows: labels are model ids; images are unmirrored visual crops",
            fontsize=9,
        )
        ax_y.set_yticks(list(range(0, n_part, ys)))
        ax_y.set_yticklabels(
            [str(int(part_indices[i])) for i in range(0, n_part, ys)],
            fontsize=7,
        )
        ax_y.set_ylabel("part windows\nmodel ids + tokens", fontsize=8)
        base.grid_cells(ax_x, n_line1, None, alpha=0.60, lw=0.55)
        for y in np.arange(-0.5, n_part + 0.5, 1.0):
            ax_y.axhline(y, color="white", lw=0.55, alpha=0.65, zorder=10)

        im = ax_h.imshow(
            sim_display.T,
            origin="upper",
            aspect="auto",
            vmin=args.heatmap_vmin,
            vmax=args.heatmap_vmax,
        )
        base.plt.colorbar(im, cax=cax).set_label("cosine similarity")
        base.grid_cells(ax_h, n_line1, n_part, alpha=0.24, lw=0.40)
        base.annotate_heatmap_cells(
            ax_h,
            sim_display,
            threshold,
            selected_display,
            args,
        )
        dx, dy = base.draw_path(
            ax_h,
            path_pairs_display,
            selected_display,
            seg,
        )
        ax_h.set_xlim(-0.5, n_line1 - 0.5)
        ax_h.set_ylim(n_part - 0.5, -0.5)
        ax_h.set_xlabel(
            f"line1 displayed windows ({args.heatmap_display_order} order; "
            f"tick=model id{'; reversed' if reverse_x else ''})"
        )
        ax_h.set_ylabel(
            f"part displayed windows ({args.heatmap_display_order} order; "
            f"tick=model id{'; reversed' if reverse_y else ''})"
        )
        ax_h.set_xticks(list(range(0, n_line1, xs)))
        ax_h.set_xticklabels(
            [str(int(line1_indices[i])) for i in range(0, n_line1, xs)],
            fontsize=8,
        )
        ax_h.set_yticks(list(range(0, n_part, ys)))
        ax_h.set_yticklabels(
            [str(int(part_indices[i])) for i in range(0, n_part, ys)],
            fontsize=8,
        )
        if dx or dy or seg is not None:
            ax_h.legend(loc="upper right", fontsize=8)
    else:
        fig, ax_h = base.plt.subplots(
            figsize=(
                max(9, min(26, n_line1 / 3.6)),
                max(4.2, min(13, n_part / 1.8 + 2)),
            )
        )
        im = ax_h.imshow(
            sim_display.T,
            origin="upper",
            aspect="auto",
            vmin=args.heatmap_vmin,
            vmax=args.heatmap_vmax,
        )
        base.plt.colorbar(im, ax=ax_h, fraction=0.025, pad=0.02).set_label(
            "cosine similarity"
        )
        base.grid_cells(ax_h, n_line1, n_part, alpha=0.24, lw=0.40)
        base.annotate_heatmap_cells(
            ax_h,
            sim_display,
            threshold,
            selected_display,
            args,
        )
        dx, dy = base.draw_path(
            ax_h,
            path_pairs_display,
            selected_display,
            seg,
        )
        ax_h.set_xlim(-0.5, n_line1 - 0.5)
        ax_h.set_ylim(n_part - 0.5, -0.5)
        ax_h.set_xlabel(
            f"line1 displayed window index{'; reversed' if reverse_x else ''}"
        )
        ax_h.set_ylabel(
            f"part displayed window index{'; reversed' if reverse_y else ''}"
        )
        if dx or dy or seg is not None:
            ax_h.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Part {part.part_id} vs line1 cosine similarity | "
        f"line2_x=[{part.x0_original},{part.x1_original}] | "
        f"thr={threshold:.3f} | orange=sim≥thr, red=chosen SW cells"
        f"{reverse_label}",
        fontsize=11,
    )
    base.plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    base.plt.close(fig)
    print(f"Saved cosine heatmap for part {part.part_id}: {out}")
    return out


def main() -> None:
    base.make_y_strip = make_y_strip_self_window_style
    base.save_heatmap = save_heatmap_with_reversible_axes
    base.aligned_tokens_for_windows = aligned_tokens_for_windows_robust
    base.main()


if __name__ == "__main__":
    main()
