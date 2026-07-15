#!/usr/bin/env python3
"""Run line-to-part visualization with the same y-axis style as self-window cosine.

This wrapper keeps the original line-to-part logic, but replaces the heatmap y-axis
window strip with the same behavior used by:

    scripts/eval/window-similarity/visualize_line_self_window_cosine.py

It also makes the axis token labels robust: if hard Span-DTW cannot assign a token
for every displayed window, it falls back to an independent per-window argmax over
all text spans. That avoids showing only "?" when the full transcript is too long
for a small cropped part or when Span-DTW constraints are infeasible.

The CLI is inherited from visualize_line2_parts_in_line1.py.
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


def _fallback_window_argmax_labels(encoding, image_embeddings_for_alignment: torch.Tensor) -> list[str]:
    """Assign one best text span to each window independently.

    This is used only as a visualization fallback when hard Span-DTW is infeasible
    or leaves windows unlabeled. It respects MAX_SPAN_CHARS because the candidate
    spans come from the text encoder configuration.
    """
    num_windows = int(image_embeddings_for_alignment.shape[0])
    texts = list(getattr(encoding, "texts", []) or [])
    span_embeddings = getattr(encoding, "embeddings", None)
    if num_windows <= 0:
        return []
    if span_embeddings is None or len(texts) == 0:
        return [""] * num_windows

    with torch.no_grad():
        norm_img = F.normalize(image_embeddings_for_alignment.float(), p=2, dim=-1)
        norm_span = F.normalize(span_embeddings.float().to(norm_img.device), p=2, dim=-1)
        sim = torch.matmul(norm_img, norm_span.T)
        best_span_indices = sim.argmax(dim=1).detach().cpu().tolist()

    return [str(texts[int(idx)]) if 0 <= int(idx) < len(texts) else "?" for idx in best_span_indices]


def aligned_tokens_for_windows_robust(text_encoder, text: str | None, image_embeddings_for_alignment: torch.Tensor, args) -> list[str]:
    """Hard Span-DTW labels with robust per-window fallback.

    The original base implementation returned "?" for every window when Span-DTW
    could not create a full path. This happens often for line-to-part heatmaps:
    the y-axis part has only a few image windows, but the available text is still
    the full line transcript. In that case, a full monotonic path is infeasible.
    Here we keep Span-DTW when it works, and fill any missing windows with the
    closest span by cosine similarity.
    """
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

    # First try the sequence-aware labels from hard Span-DTW.
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
            # hard_span_dtw_path uses exclusive window_end. Be defensive in case
            # another backend returns a zero-width or inclusive span.
            if end <= start:
                end = min(num_windows, start + 1)
            for window_idx in range(start, end):
                labels[window_idx] = token
                filled[window_idx] = True
    except Exception as exc:
        print(
            f"[warn] hard Span-DTW axis tokens failed; using per-window argmax fallback: {exc}",
            flush=True,
        )

    # Fill any remaining ? labels independently. This is especially important for
    # cropped parts, because full-line text cannot always be globally aligned to a
    # tiny part crop.
    if not all(filled):
        try:
            fallback_labels = _fallback_window_argmax_labels(encoding, image_embeddings_for_alignment)
            for window_idx, fallback in enumerate(fallback_labels):
                if window_idx >= len(labels):
                    break
                if not filled[window_idx] or labels[window_idx] == "?":
                    labels[window_idx] = fallback
                    filled[window_idx] = True
        except Exception as exc:
            print(f"[warn] per-window argmax axis-token fallback failed: {exc}", flush=True)

    return labels


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
    base.aligned_tokens_for_windows = aligned_tokens_for_windows_robust
    base.main()


if __name__ == "__main__":
    main()
