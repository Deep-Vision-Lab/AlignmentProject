#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image, ImageOps
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
LINE_TO_LINE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "line-to-line"))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, LINE_TO_LINE_DIR)

from visualize_sw_longest_alignment import (  # noqa: E402
    PALETTE,
    cosine_similarity_matrix,
    image_embeddings,
    load_image_model,
    preprocess_line_image,
    resolve_threshold,
    smith_waterman,
    window_range_to_pixels,
)

try:
    _RESAMPLE = Image.Resampling.BILINEAR
    _ROTATE_90 = Image.Transpose.ROTATE_90
except AttributeError:  # Pillow<9
    _RESAMPLE = Image.BILINEAR
    _ROTATE_90 = Image.ROTATE_90


@dataclass
class SourcePart:
    part_id: int
    path: str
    x0_original: int
    x1_original: int


@dataclass
class Segment:
    line1_start: int
    line1_end: int
    part_start: int
    part_end: int
    length: int
    mean_similarity: float
    min_similarity: float
    score: float
    match_kind: str


@dataclass
class Result:
    part: SourcePart
    found: bool
    match_x0_original: Optional[int] = None
    match_x1_original: Optional[int] = None
    part_match_x0_relative: Optional[int] = None
    part_match_x1_relative: Optional[int] = None
    run_x0_original: Optional[float] = None
    run_x1_original: Optional[float] = None
    line1_window_start: Optional[int] = None
    line1_window_end: Optional[int] = None
    part_window_start: Optional[int] = None
    part_window_end: Optional[int] = None
    part_window_count: int = 0
    run_length: int = 0
    mean_similarity: float = 0.0
    min_similarity: float = 0.0
    sw_score: float = 0.0
    score: float = 0.0
    match_kind: str = ""
    threshold_used: float = 0.0
    best_similarity: float = float("nan")
    best_sw_diag_mean: float = float("nan")
    embedding_space: str = ""
    heatmap_output: Optional[str] = None
    message: str = ""


def parse_part_starts(text: Optional[str]) -> Optional[List[int]]:
    if not text or not text.strip():
        return None
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def clamp_part_start(start: int, width: int, part_width: int) -> int:
    if width <= part_width:
        return 0
    return max(0, min(int(start), width - part_width))


def random_part_starts(width: int, part_width: int, n: int, rng: random.Random, min_gap: int = 0) -> List[int]:
    max_start = max(0, width - part_width)
    if max_start == 0:
        return [0 for _ in range(n)]
    if n * part_width + (n - 1) * min_gap <= width:
        best: List[int] = []
        for _attempt in range(5000):
            starts: List[int] = []
            for _i in range(n):
                cand = rng.randint(0, max_start)
                cend = cand + part_width
                if all(cend + min_gap <= s or s + part_width + min_gap <= cand for s in starts):
                    starts.append(cand)
            if len(starts) == n:
                return sorted(starts)
            if len(starts) > len(best):
                best = starts
        while len(best) < n:
            best.append(rng.randint(0, max_start))
        return sorted(best[:n])
    return sorted(rng.randint(0, max_start) for _ in range(n))


def default_part_starts(width: int, part_width: int, n: int, mode: str, rng: random.Random, min_gap: int) -> List[int]:
    max_start = max(0, width - part_width)
    if n <= 1:
        return [rng.randint(0, max_start)] if mode == "random" and max_start > 0 else [max_start // 2]
    if mode == "random":
        return random_part_starts(width, part_width, n, rng, min_gap)
    if mode == "even":
        return [int(round(x)) for x in np.linspace(0, max_start, n)]
    return [max(0, width - (k + 1) * part_width) for k in range(n)]


def resolve_line_paths(args: argparse.Namespace) -> Tuple[str, str]:
    if args.line1 and args.line2:
        return args.line1, args.line2
    if args.data_dir and args.index is not None:
        root = os.path.join(args.data_dir, "images")
        return os.path.join(root, f"img1_{args.index}.png"), os.path.join(root, f"img2_{args.index}.png")
    raise ValueError("Use either --line1/--line2 or --data-dir/--index")


def make_source_parts(line2_path: str, starts: Sequence[int], part_width: int, tmp_dir: str) -> List[SourcePart]:
    line2 = Image.open(line2_path).convert("RGB")
    width, height = line2.size
    parts: List[SourcePart] = []
    for part_id, raw_start in enumerate(starts, start=1):
        x0 = clamp_part_start(raw_start, width, part_width)
        x1 = min(width, x0 + part_width)
        part_path = os.path.join(tmp_dir, f"line2_part_{part_id}.png")
        line2.crop((x0, 0, x1, height)).save(part_path)
        parts.append(SourcePart(part_id=part_id, path=part_path, x0_original=x0, x1_original=x1))
    return parts


def clamp_segment_to_image(x0: float, x1: float, width: int) -> Tuple[int, int]:
    x0_i = int(round(max(0.0, min(float(x0), float(width)))))
    x1_i = int(round(max(float(x0_i) + 1.0, min(float(x1), float(width)))))
    if x1_i > width:
        x1_i = width
    if x0_i >= x1_i:
        x0_i = max(0, min(x0_i, max(0, width - 1)))
        x1_i = min(width, x0_i + 1)
    return x0_i, x1_i


def cap_segment_width(x0: float, x1: float, max_width: int, image_width: int) -> Tuple[int, int]:
    x0_i, x1_i = clamp_segment_to_image(x0, x1, image_width)
    if max_width <= 0 or image_width <= max_width or (x1_i - x0_i) <= max_width:
        return x0_i, x1_i
    start = int(round(0.5 * (x0_i + x1_i) - max_width / 2.0))
    start = max(0, min(start, image_width - max_width))
    return start, start + max_width


def diag_pairs(sw_path) -> List[Tuple[int, int]]:
    return [(i, j) for op, i, j in sw_path if op == "diag" and i is not None and j is not None]


def make_segment(pairs: Sequence[Tuple[int, int]], sim: np.ndarray, kind: str) -> Optional[Segment]:
    if not pairs:
        return None
    vals = np.asarray([float(sim[i, j]) for i, j in pairs], dtype=np.float32)
    line1_start = int(min(i for i, _j in pairs))
    line1_end = int(max(i for i, _j in pairs))
    part_start = int(min(j for _i, j in pairs))
    part_end = int(max(j for _i, j in pairs))
    mean = float(vals.mean())
    minv = float(vals.min())
    return Segment(line1_start, line1_end, part_start, part_end, len(pairs), mean, minv, mean + 1e-4 * len(pairs), kind)


def passes(seg: Segment, threshold: float, min_len: int, require_all: bool) -> bool:
    return seg.length >= min_len and seg.mean_similarity >= threshold and (not require_all or seg.min_similarity >= threshold)


def split_runs(pairs: Sequence[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    if not pairs:
        return []
    runs: List[List[Tuple[int, int]]] = []
    current = [pairs[0]]
    for prev, cur in zip(pairs[:-1], pairs[1:]):
        if cur[0] == prev[0] + 1 and cur[1] == prev[1] + 1:
            current.append(cur)
        else:
            runs.append(current)
            current = [cur]
    runs.append(current)
    return runs


def choose_segment(sw_path, sim: np.ndarray, threshold: float, min_len: int, require_all: bool) -> Optional[Segment]:
    pairs = diag_pairs(sw_path)
    if pairs and sim.ndim == 2:
        covered = {j for _i, j in pairs}
        if covered == set(range(sim.shape[1])):
            full = make_segment(pairs, sim, "whole_part")
            if full and passes(full, threshold, min_len, require_all):
                return full
    best: Optional[Segment] = None
    for run in split_runs(pairs):
        seg = make_segment(run, sim, "partial_segment")
        if seg and passes(seg, threshold, min_len, require_all) and (best is None or seg.score > best.score):
            best = seg
    return best


def highest_similarity(sim: np.ndarray) -> float:
    return float(np.max(sim)) if sim.size else float("nan")


def best_diag_mean(sw_path, sim: np.ndarray) -> float:
    pairs = diag_pairs(sw_path)
    if not pairs:
        return float("nan")
    return float(np.mean([float(sim[i, j]) for i, j in pairs]))


def span_to_original_pixels(start, end, n_windows, display_width, original_width, window_size, stride, use_flip, padding=0):
    start = max(0, int(start) - int(padding))
    end = min(n_windows - 1, int(end) + int(padding))
    x0, x1 = window_range_to_pixels(start, end, n_windows, display_width, window_size, stride, use_flip)
    scale = original_width / float(display_width)
    return x0 * scale, x1 * scale


def heatmap_output_path(main_output: str, heatmap_dir: Optional[str], part_id: int) -> str:
    base_dir = heatmap_dir or os.path.dirname(main_output) or "."
    stem = os.path.splitext(os.path.basename(main_output))[0]
    return os.path.join(base_dir, f"{stem}_part{part_id}_cosine_heatmap.png")


def ordered_image(image: Image.Image, use_flip: bool) -> Image.Image:
    image = image.convert("RGB")
    return ImageOps.mirror(image) if use_flip else image


def crop_visual_slice(image: Image.Image, idx: int, n: int, window_size: int, stride: int, mode: str) -> Image.Image:
    width, height = image.size
    if mode == "window":
        x0 = idx * stride
        x1 = idx * stride + window_size
    else:
        center = idx * stride + window_size / 2.0
        x0 = int(round(center - stride / 2.0))
        x1 = int(round(center + stride / 2.0))
        if idx == 0:
            x0 = max(0, x0)
        if idx == n - 1:
            x1 = min(width, x1)
    x0 = max(0, min(int(x0), max(0, width - 1)))
    x1 = max(x0 + 1, min(int(x1), width))
    return image.crop((x0, 0, x1, height))


def hstack(images: Sequence[Image.Image]) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), (255, 255, 255))
    canvas = Image.new("RGB", (sum(im.size[0] for im in images), max(im.size[1] for im in images)), (255, 255, 255))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.size[0]
    return canvas


def vstack(images: Sequence[Image.Image]) -> Image.Image:
    if not images:
        return Image.new("RGB", (1, 1), (255, 255, 255))
    canvas = Image.new("RGB", (max(im.size[0] for im in images), sum(im.size[1] for im in images)), (255, 255, 255))
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.size[1]
    return canvas


def make_x_strip(image: Image.Image, n: int, args) -> np.ndarray:
    im = ordered_image(image, args.use_flip)
    cells = []
    for idx in range(n):
        crop = crop_visual_slice(im, idx, n, args.window_size, args.stride, args.heatmap_axis_slice_mode)
        crop = crop.resize((args.heatmap_axis_cell_pixels, args.heatmap_line1_strip_height), _RESAMPLE)
        cell = Image.new("RGB", (args.heatmap_axis_cell_pixels + args.heatmap_window_gap_pixels, args.heatmap_line1_strip_height), (255, 255, 255))
        cell.paste(crop, (0, 0))
        cells.append(cell)
    return np.array(hstack(cells))


def make_y_strip(image: Image.Image, n: int, args) -> np.ndarray:
    im = ordered_image(image, args.use_flip)
    cells = []
    for idx in range(n):
        crop = crop_visual_slice(im, idx, n, args.window_size, args.stride, args.heatmap_axis_slice_mode)
        if args.heatmap_flip_part_axis_windows:
            # Reverted behavior: mirror the part slice first, then rotate it for the side axis.
            crop = ImageOps.mirror(crop)
        crop = crop.transpose(_ROTATE_90).resize((args.heatmap_part_strip_width, args.heatmap_axis_cell_pixels), _RESAMPLE)
        cell = Image.new("RGB", (args.heatmap_part_strip_width, args.heatmap_axis_cell_pixels + args.heatmap_window_gap_pixels), (255, 255, 255))
        cell.paste(crop, (0, 0))
        cells.append(cell)
    return np.array(vstack(cells))


def grid_cells(ax, nx: int, ny: Optional[int] = None, alpha=0.25, lw=0.4):
    for x in np.arange(-0.5, nx + 0.5, 1.0):
        ax.axvline(x, color="white", lw=lw, alpha=alpha, zorder=10)
    if ny is not None:
        for y in np.arange(-0.5, ny + 0.5, 1.0):
            ax.axhline(y, color="white", lw=lw, alpha=alpha, zorder=10)


def label_step(n: int, target=14) -> int:
    return max(1, int(np.ceil(max(1, n) / float(target))))


def selected_cell_set(sw_path, seg: Optional[Segment]) -> set:
    if seg is None:
        return set()
    selected = set()
    for i, j in diag_pairs(sw_path):
        if seg.line1_start <= i <= seg.line1_end and seg.part_start <= j <= seg.part_end:
            selected.add((int(i), int(j)))
    return selected


def annotate_heatmap_cells(ax, sim: np.ndarray, threshold: float, sw_path, seg: Optional[Segment], args):
    n_line1, n_part = int(sim.shape[0]), int(sim.shape[1])
    selected = selected_cell_set(sw_path, seg)
    for j in range(n_part):
        for i in range(n_line1):
            value = float(sim[i, j])
            if args.heatmap_mark_above_threshold and value >= threshold:
                ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, edgecolor="orange", linewidth=0.85, alpha=0.95, zorder=12))
            if args.heatmap_cell_values:
                text_color = "black" if value >= 0.62 else "white"
                ax.text(i, j, f"{value:.2f}", ha="center", va="center", fontsize=args.heatmap_cell_value_fontsize, color=text_color, weight="bold" if (i, j) in selected else "normal", zorder=14)
    for i, j in selected:
        ax.add_patch(Rectangle((i - 0.5, j - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=2.2, alpha=1.0, zorder=15))


def draw_path(ax, sw_path, seg: Optional[Segment]):
    dx = [i for op, i, j in sw_path if op == "diag" and i is not None and j is not None]
    dy = [j for op, i, j in sw_path if op == "diag" and i is not None and j is not None]
    if dx:
        ax.plot(dx, dy, color="white", lw=1.5, alpha=0.95, label="SW diag path", zorder=16)
    if seg is not None:
        ax.plot([seg.line1_start, seg.line1_end], [seg.part_start, seg.part_end], color="red", lw=2.6, alpha=0.95, label=f"selected {seg.match_kind}", zorder=17)
        ax.scatter([seg.line1_start, seg.line1_end], [seg.part_start, seg.part_end], color="red", s=22, zorder=18)
    return dx, dy


def save_heatmap(sim, sw_path, seg, part: SourcePart, threshold: float, args, line1_display=None, part_display=None) -> str:
    out = heatmap_output_path(args.output, args.heatmap_dir, part.part_id)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    axis_images = args.heatmap_axis_images and line1_display is not None and part_display is not None
    n_line1, n_part = int(sim.shape[0]), int(sim.shape[1])

    if axis_images:
        fig = plt.figure(figsize=(max(13, min(38, n_line1 * 0.38 + 5)), max(7.5, min(20, n_part * 0.88 + 5.5))))
        gs = GridSpec(2, 3, figure=fig, width_ratios=[2.2, 8.8, 0.34], height_ratios=[1.55, 6.0], wspace=0.08, hspace=0.08)
        ax_corner = fig.add_subplot(gs[0, 0])
        ax_x = fig.add_subplot(gs[0, 1])
        ax_y = fig.add_subplot(gs[1, 0])
        ax_h = fig.add_subplot(gs[1, 1])
        cax = fig.add_subplot(gs[1, 2])
        ax_corner.axis("off")
        ax_corner.text(0.5, 0.5, f"separated\nwindow slices\nmode={args.heatmap_axis_slice_mode}", ha="center", va="center", fontsize=8, weight="bold")
        ax_x.imshow(make_x_strip(line1_display, n_line1, args), extent=(-0.5, n_line1 - 0.5, 1, 0), aspect="auto")
        ax_y.imshow(make_y_strip(part_display, n_part, args), extent=(0, 1, n_part - 0.5, -0.5), aspect="auto")
        ax_x.set_xlim(-0.5, n_line1 - 0.5)
        ax_x.set_ylim(1, 0)
        ax_x.set_yticks([])
        ax_y.set_xlim(0, 1)
        ax_y.set_ylim(n_part - 0.5, -0.5)
        ax_y.set_xticks([])
        xs = label_step(n_line1)
        ys = 1 if n_part <= 16 else label_step(n_part)
        ax_x.set_xticks(list(range(0, n_line1, xs)))
        ax_x.set_xticklabels([str(i) for i in range(0, n_line1, xs)], fontsize=7)
        ax_x.tick_params(axis="x", labeltop=True, labelbottom=False, pad=1)
        ax_x.set_title("line1 separated window slices (same order as heatmap columns)", fontsize=9)
        ax_y.set_yticks(list(range(0, n_part, ys)))
        ax_y.set_yticklabels([str(i) for i in range(0, n_part, ys)], fontsize=7)
        ax_y.set_ylabel("part separated windows\nrotated 90° + flipped", fontsize=8)
        grid_cells(ax_x, n_line1, None, alpha=0.60, lw=0.55)
        for y in np.arange(-0.5, n_part + 0.5, 1.0):
            ax_y.axhline(y, color="white", lw=0.55, alpha=0.65, zorder=10)
        im = ax_h.imshow(sim.T, origin="upper", aspect="auto", vmin=args.heatmap_vmin, vmax=args.heatmap_vmax)
        plt.colorbar(im, cax=cax).set_label("cosine similarity")
        grid_cells(ax_h, n_line1, n_part, alpha=0.24, lw=0.40)
        annotate_heatmap_cells(ax_h, sim, threshold, sw_path, seg, args)
        dx, dy = draw_path(ax_h, sw_path, seg)
        ax_h.set_xlim(-0.5, n_line1 - 0.5)
        ax_h.set_ylim(n_part - 0.5, -0.5)
        ax_h.set_xlabel("line1 window index / x-axis slices")
        ax_h.set_ylabel("part window index / y-axis slices")
        ax_h.set_xticks(list(range(0, n_line1, xs)))
        ax_h.set_xticklabels([str(i) for i in range(0, n_line1, xs)], fontsize=8)
        ax_h.set_yticks(list(range(0, n_part, ys)))
        ax_h.set_yticklabels([str(i) for i in range(0, n_part, ys)], fontsize=8)
        if dx or dy or seg is not None:
            ax_h.legend(loc="upper right", fontsize=8)
    else:
        fig, ax_h = plt.subplots(figsize=(max(9, min(26, n_line1 / 3.6)), max(4.2, min(13, n_part / 1.8 + 2))))
        im = ax_h.imshow(sim.T, origin="upper", aspect="auto", vmin=args.heatmap_vmin, vmax=args.heatmap_vmax)
        plt.colorbar(im, ax=ax_h, fraction=0.025, pad=0.02).set_label("cosine similarity")
        grid_cells(ax_h, n_line1, n_part, alpha=0.24, lw=0.40)
        annotate_heatmap_cells(ax_h, sim, threshold, sw_path, seg, args)
        dx, dy = draw_path(ax_h, sw_path, seg)
        ax_h.set_xlim(-0.5, n_line1 - 0.5)
        ax_h.set_ylim(n_part - 0.5, -0.5)
        ax_h.set_xlabel("line1 window index")
        ax_h.set_ylabel("part window index")
        if dx or dy or seg is not None:
            ax_h.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Part {part.part_id} vs line1 cosine similarity | "
        f"line2_x=[{part.x0_original},{part.x1_original}] | thr={threshold:.3f} | "
        f"orange=sim≥thr, red=chosen SW cells",
        fontsize=11,
    )
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved cosine heatmap for part {part.part_id}: {out}")
    return out


def search_one_part(model, emb_line1, part: SourcePart, line1_num_windows: int, line1_display_width: int, line1_original_width: int, args, line1_display=None) -> Result:
    part_display, part_tensor = preprocess_line_image(part.path, target_height=args.height)
    emb_part = image_embeddings(model, part_tensor, args.device, embedding_space=args.embedding_space)
    sim = cosine_similarity_matrix(emb_line1, emb_part)
    threshold = resolve_threshold(sim, args)
    best_sim = highest_similarity(sim)
    sw_path, sw_score, _H = smith_waterman(sim, threshold=threshold, gap_penalty=args.gap, match_reward=args.match, mismatch_penalty=args.mismatch)
    diag_mean = best_diag_mean(sw_path, sim)
    seg = choose_segment(sw_path, sim, threshold, args.min_run_length, args.require_all_windows_above_threshold)
    heatmap_path = save_heatmap(sim, sw_path, seg, part, threshold, args, line1_display, part_display) if args.heatmap else None

    if seg is None:
        return Result(
            part=part,
            found=False,
            part_window_count=int(emb_part.shape[0]),
            sw_score=float(sw_score),
            score=best_sim,
            threshold_used=float(threshold),
            best_similarity=best_sim,
            best_sw_diag_mean=diag_mean,
            embedding_space=args.embedding_space,
            heatmap_output=heatmap_path,
            message="Smith-Waterman did not find the whole part or a fallback segment above threshold; displayed best is the highest cosine similarity in the matrix.",
        )

    rx0, rx1 = span_to_original_pixels(seg.line1_start, seg.line1_end, line1_num_windows, line1_display_width, line1_original_width, args.window_size, args.stride, args.use_flip, args.mask_padding_windows)
    mx0, mx1 = clamp_segment_to_image(rx0, rx1, line1_original_width)
    part_width = max(1, part.x1_original - part.x0_original)
    px0, px1 = span_to_original_pixels(seg.part_start, seg.part_end, int(emb_part.shape[0]), int(part_display.size[0]), part_width, args.window_size, args.stride, args.use_flip, 0)
    pr0, pr1 = cap_segment_width(px0, px1, part_width, part_width)
    return Result(
        part=part,
        found=True,
        match_x0_original=mx0,
        match_x1_original=mx1,
        part_match_x0_relative=pr0,
        part_match_x1_relative=pr1,
        run_x0_original=rx0,
        run_x1_original=rx1,
        line1_window_start=seg.line1_start,
        line1_window_end=seg.line1_end,
        part_window_start=seg.part_start,
        part_window_end=seg.part_end,
        part_window_count=int(emb_part.shape[0]),
        run_length=seg.length,
        mean_similarity=seg.mean_similarity,
        min_similarity=seg.min_similarity,
        sw_score=float(sw_score),
        score=seg.score,
        match_kind=seg.match_kind,
        threshold_used=float(threshold),
        best_similarity=best_sim,
        best_sw_diag_mean=diag_mean,
        embedding_space=args.embedding_space,
        heatmap_output=heatmap_path,
    )


def crop_or_blank(image: Image.Image, x0: int, x1: int, width: int) -> Image.Image:
    image = image.convert("RGB")
    x0 = max(0, min(int(x0), image.size[0]))
    x1 = max(x0, min(int(x1), image.size[0]))
    crop = image.crop((x0, 0, x1, image.size[1]))
    if crop.size[0] == width:
        return crop
    out = Image.new("RGB", (width, image.size[1]), (255, 255, 255))
    out.paste(crop, (0, 0))
    return out


def draw_results(line1: Image.Image, line2: Image.Image, results: Sequence[Result], output: str, title: str, part_width: int):
    arr1 = np.array(line1.convert("RGB"))
    h1, w1 = arr1.shape[:2]
    line2_h = line2.size[1]
    part_gap = max(18, int(round(part_width * 0.25)))
    n = max(1, len(results))
    total_parts_w = n * part_width + (n - 1) * part_gap
    canvas_w = max(w1, total_parts_w)
    xoff = 0 if canvas_w == w1 else int(round((canvas_w - w1) / 2.0))
    top, arrow_gap, bottom = 34, 95, 28
    y1_top, y1_bottom = top, top + h1
    yp_top, yp_bottom = y1_bottom + arrow_gap, y1_bottom + arrow_gap + line2_h

    fig, ax = plt.subplots(figsize=(max(12.0, line1.size[0] / 100.0), max(5.0, (line1.size[1] + line2.size[1] + 160) / 100.0)))
    ax.imshow(arr1, extent=(xoff, xoff + w1, y1_bottom, y1_top), zorder=1)
    ax.text(xoff, y1_top - 8, "Line 1: full searched line with masks", fontsize=11, weight="bold", va="bottom")
    ax.text(0, yp_top - 8, "Line 2: chosen parts only, no masks", fontsize=11, weight="bold", va="bottom")
    starts = [int(round((canvas_w - part_width) / 2.0))] if n == 1 else [int(round((canvas_w - total_parts_w) / 2.0)) + i * (part_width + part_gap) for i in range(n)]

    for idx, (res, px0) in enumerate(zip(results, starts)):
        color = PALETTE[idx % len(PALETTE)]
        part = res.part
        px1 = px0 + part_width
        ax.imshow(np.array(crop_or_blank(line2, part.x0_original, part.x1_original, part_width)), extent=(px0, px1, yp_bottom, yp_top), zorder=1)
        ax.add_patch(Rectangle((px0, yp_top), part_width, line2_h, facecolor="none", edgecolor=color, linewidth=2, alpha=0.95, zorder=3))
        pc = 0.5 * (px0 + px1)
        if not res.found:
            best = "nan" if np.isnan(res.best_similarity) else f"{res.best_similarity:.3f}"
            diag = "nan" if np.isnan(res.best_sw_diag_mean) else f"{res.best_sw_diag_mean:.3f}"
            ax.text(pc, yp_top + 0.5 * line2_h, f"part {part.part_id}\nnot found\nbest={best}", ha="center", va="center", fontsize=8, color="white", weight="bold", bbox=dict(facecolor=color, edgecolor="none", alpha=0.72, boxstyle="round,pad=0.25"), zorder=5)
            heat = f" heatmap={res.heatmap_output}" if res.heatmap_output else ""
            print(f"part {part.part_id}: NOT FOUND | embedding={res.embedding_space} best_sim={best} best_sw_diag_mean={diag} thr={res.threshold_used:.4f} sw_path_score={res.sw_score:.4f}{heat} | {res.message}")
            continue

        assert res.match_x0_original is not None and res.match_x1_original is not None
        mx0 = xoff + res.match_x0_original
        mx1 = xoff + res.match_x1_original
        ax.add_patch(Rectangle((mx0, y1_top), max(1, mx1 - mx0), h1, facecolor=color, edgecolor=color, linewidth=2, alpha=0.30, zorder=3))
        src_x0 = px0 + (res.part_match_x0_relative or 0)
        src_x1 = px0 + (res.part_match_x1_relative or part_width)
        sc = 0.5 * (src_x0 + src_x1)
        mc = 0.5 * (mx0 + mx1)
        ax.add_patch(FancyArrowPatch((sc, yp_top - 4), (mc, y1_bottom + 4), arrowstyle="->", mutation_scale=18, linewidth=2.4, color=color, alpha=0.95, zorder=4))
        ax.text(mc, y1_top + 0.5 * h1, f"part {part.part_id}\n{res.match_kind}\nsim={res.mean_similarity:.3f}\nwin={res.run_length}", ha="center", va="center", fontsize=8, color="white", weight="bold", bbox=dict(facecolor=color, edgecolor="none", alpha=0.72, boxstyle="round,pad=0.25"), zorder=5)
        ax.text(pc, yp_top + 0.5 * line2_h, f"part {part.part_id}\nx=[{part.x0_original},{part.x1_original}]", ha="center", va="center", fontsize=8, color="white", weight="bold", bbox=dict(facecolor=color, edgecolor="none", alpha=0.72, boxstyle="round,pad=0.25"), zorder=5)
        heat = f" | heatmap={res.heatmap_output}" if res.heatmap_output else ""
        print(f"part {part.part_id}: FOUND | kind={res.match_kind} | thr={res.threshold_used:.4f} | line2_x=[{part.x0_original},{part.x1_original}] | line1_mask_x=[{res.match_x0_original},{res.match_x1_original}] width={res.match_x1_original - res.match_x0_original} | line1_windows=[{res.line1_window_start},{res.line1_window_end}] | part_windows=[{res.part_window_start},{res.part_window_end}]/{res.part_window_count} | mean_sim={res.mean_similarity:.4f} | min_sim={res.min_similarity:.4f} | best_sim={res.best_similarity:.4f}{heat}")

    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(yp_bottom + bottom, 0)
    ax.axis("off")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Crop parts from line2 and find their SW-aligned segment in full line1.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--embedding-space", choices=("local", "contextual"), default="local")
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--adaptive-threshold", choices=("none", "percentile", "mean_std"), default="percentile")
    parser.add_argument("--threshold-percentile", type=float, default=90.0)
    parser.add_argument("--threshold-std-scale", type=float, default=1.0)
    parser.add_argument("--no-adaptive-threshold-floor", dest="adaptive_threshold_floor", action="store_false")
    parser.set_defaults(adaptive_threshold_floor=True)
    parser.add_argument("--match", type=float, default=1.0)
    parser.add_argument("--mismatch", type=float, default=-1.5)
    parser.add_argument("--gap", type=float, default=-0.15)
    parser.add_argument("--min-run-length", type=int, default=3)
    parser.add_argument("--require-all-windows-above-threshold", action="store_true")
    parser.add_argument("--mask-padding-windows", type=int, default=0)
    parser.add_argument("--use-flip", action="store_true")
    parser.add_argument("--no-bilstm", action="store_true")

    parser.add_argument("--heatmap", action="store_true")
    parser.add_argument("--heatmap-dir", default=None)
    parser.add_argument("--heatmap-vmin", type=float, default=0.0)
    parser.add_argument("--heatmap-vmax", type=float, default=1.0)
    parser.add_argument("--no-heatmap-axis-images", dest="heatmap_axis_images", action="store_false")
    parser.set_defaults(heatmap_axis_images=True)
    parser.add_argument("--heatmap-axis-cell-pixels", type=int, default=52)
    parser.add_argument("--heatmap-window-gap-pixels", type=int, default=10)
    parser.add_argument("--heatmap-axis-slice-mode", choices=("nonoverlap", "window"), default="nonoverlap")
    parser.add_argument("--heatmap-line1-strip-height", type=int, default=88)
    parser.add_argument("--heatmap-part-strip-width", type=int, default=112)
    parser.add_argument("--no-heatmap-flip-part-axis-windows", dest="heatmap_flip_part_axis_windows", action="store_false")
    parser.set_defaults(heatmap_flip_part_axis_windows=True)
    parser.add_argument("--no-heatmap-cell-values", dest="heatmap_cell_values", action="store_false")
    parser.set_defaults(heatmap_cell_values=True)
    parser.add_argument("--heatmap-cell-value-fontsize", type=float, default=4.2)
    parser.add_argument("--no-heatmap-mark-above-threshold", dest="heatmap_mark_above_threshold", action="store_false")
    parser.set_defaults(heatmap_mark_above_threshold=True)

    parser.add_argument("--line1", default=None)
    parser.add_argument("--line2", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--part-width", type=int, default=124)
    parser.add_argument("--num-parts", type=int, default=3)
    parser.add_argument("--part-starts", default=None)
    parser.add_argument("--part-mode", choices=("random", "rtl-blocks", "even"), default="random")
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--random-min-gap", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.part_width <= 0 or args.num_parts <= 0 or args.min_run_length <= 0:
        raise ValueError("part-width, num-parts, and min-run-length must be positive")
    if args.random_min_gap < 0 or args.mask_padding_windows < 0 or args.heatmap_window_gap_pixels < 0:
        raise ValueError("gap-like arguments must be >= 0")
    if args.heatmap_vmax <= args.heatmap_vmin:
        raise ValueError("--heatmap-vmax must be larger than --heatmap-vmin")
    if min(args.heatmap_axis_cell_pixels, args.heatmap_line1_strip_height, args.heatmap_part_strip_width) <= 0:
        raise ValueError("heatmap axis image sizes must be positive")
    if args.heatmap_cell_value_fontsize <= 0:
        raise ValueError("--heatmap-cell-value-fontsize must be positive")
    if args.gap > 0 or args.mismatch > 0 or args.match <= 0:
        raise ValueError("use negative --gap/--mismatch and positive --match")

    line1_path, line2_path = resolve_line_paths(args)
    if not os.path.exists(line1_path):
        raise FileNotFoundError(line1_path)
    if not os.path.exists(line2_path):
        raise FileNotFoundError(line2_path)
    print(f"Line1: {line1_path}\nLine2: {line2_path}")

    model = load_image_model(
        args.weights,
        args.device,
        window_size=args.window_size,
        stride=args.stride,
        vector_size=args.vector_size,
        use_bilstm=False if args.no_bilstm else None,
        use_flip=args.use_flip,
    )
    img1_display, tensor1 = preprocess_line_image(line1_path, target_height=args.height)
    emb_line1 = image_embeddings(model, tensor1, args.device, embedding_space=args.embedding_space)
    line1_original = Image.open(line1_path).convert("RGB")
    line2_original = Image.open(line2_path).convert("RGB")

    starts = parse_part_starts(args.part_starts)
    mode = "manual"
    if starts is None:
        starts = default_part_starts(line2_original.size[0], args.part_width, args.num_parts, args.part_mode, random.Random(args.random_seed), args.random_min_gap)
        mode = args.part_mode

    print(f"Part width: {args.part_width}\nPart selection mode: {mode}\nRandom seed: {args.random_seed if args.random_seed is not None else 'none'}\nPart starts: {starts}\nEmbedding space: {args.embedding_space}")
    print(f"Threshold mode: {args.adaptive_threshold}, base threshold={args.threshold}")
    if args.heatmap:
        print(
            f"Cosine heatmaps: enabled, dir={args.heatmap_dir or os.path.dirname(args.output) or '.'}, "
            f"separated windows gap={args.heatmap_window_gap_pixels}, slice_mode={args.heatmap_axis_slice_mode}, "
            f"flip_part_axis={args.heatmap_flip_part_axis_windows}, cell_values={args.heatmap_cell_values}, "
            f"mark_above_threshold={args.heatmap_mark_above_threshold}"
        )

    results: List[Result] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for part in make_source_parts(line2_path, starts, args.part_width, tmp_dir):
            results.append(
                search_one_part(
                    model,
                    emb_line1,
                    part,
                    int(emb_line1.shape[0]),
                    int(img1_display.size[0]),
                    int(line1_original.size[0]),
                    args,
                    line1_display=img1_display,
                )
            )

    sample = f" sample {args.index}" if args.index is not None else ""
    title = f"Line2 {mode} parts searched inside full line1{sample} | whole-part-first, embedding={args.embedding_space}, part_width={args.part_width}, threshold={args.adaptive_threshold}/{args.threshold}"
    draw_results(line1_original, line2_original, results, args.output, title, args.part_width)


if __name__ == "__main__":
    main()
