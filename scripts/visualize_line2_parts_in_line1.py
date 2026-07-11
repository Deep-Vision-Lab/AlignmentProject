#!/usr/bin/env python3
"""
Randomly choose fixed-width parts from line2 and search where they appear in line1.

The script:
  1. loads the trained image encoder,
  2. takes the whole first line image as the search target,
  3. randomly crops N fixed-width parts from the second line image by default,
  4. for every cropped part, runs Smith-Waterman between full line1 windows and
     the cropped part windows,
  5. first tries to align the whole chosen part,
  6. if the whole part is not found, falls back to the best aligned segment of
     that part,
  7. masks only the discovered aligned region in line1,
  8. shows the chosen line2 parts without masking them,
  9. connects each chosen part/segment to its line1 mask using the same color.

Important:
  - The selected line2 parts are displayed without a filled mask.
  - Masks are added only on line1.
  - The line1 mask follows the Smith-Waterman aligned full part or segment.
  - The line1 mask can be equal to or smaller than --part-width.
  - Unless --part-starts is given, chosen line2 parts are random on every run.
    Use --random-seed for reproducible random choices.

Example:
    python scripts/visualize_line2_parts_in_line1.py \
      --data-dir DataSet/Synthetic_Arabic \
      --index 10 \
      --weights Weights/span_jax_best_quality_win32_offline/model_latest.pth \
      --output Results/Evaluation/Part_Search/sample10_parts_random_sw.png \
      --part-width 124 \
      --num-parts 3 \
      --window-size 32 \
      --stride 16 \
      --height 128 \
      --threshold 0.86 \
      --match 1.0 \
      --mismatch -1.5 \
      --gap -0.15 \
      --min-run-length 3 \
      --use-flip

Manual part positions from the left side of line2 override random selection:
    --part-starts "100,350,600"
"""

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
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image
import torch

# Make imports work when running from the project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from visualize_sw_longest_alignment import (  # noqa: E402
    PALETTE,
    cosine_similarity_matrix,
    image_embeddings,
    load_image_model,
    preprocess_line_image,
    smith_waterman,
    window_range_to_pixels,
)


@dataclass
class SourcePart:
    part_id: int
    path: str
    x0_original: int
    x1_original: int


@dataclass
class SWAlignedSegment:
    line1_start: int
    line1_end: int
    part_start: int
    part_end: int
    length: int
    mean_similarity: float
    min_similarity: float
    score: float
    match_kind: str  # "full_part" or "partial_segment"


@dataclass
class PartSearchResult:
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
    message: str = ""


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def parse_part_starts(part_starts: Optional[str]) -> Optional[List[int]]:
    if part_starts is None or not part_starts.strip():
        return None
    starts: List[int] = []
    for item in part_starts.split(","):
        item = item.strip()
        if item:
            starts.append(int(item))
    return starts


def clamp_part_start(start: int, line_width: int, part_width: int) -> int:
    if line_width <= part_width:
        return 0
    return max(0, min(int(start), line_width - part_width))


def random_part_starts(
    line2_width: int,
    part_width: int,
    num_parts: int,
    rng: random.Random,
    min_gap: int = 0,
    max_attempts: int = 5000,
) -> List[int]:
    """Randomly choose part starts. Prefer non-overlapping crops when possible."""
    if num_parts <= 0:
        return []

    max_start = max(0, line2_width - part_width)
    if max_start == 0:
        return [0 for _ in range(num_parts)]

    can_fit_non_overlapping = num_parts * part_width + (num_parts - 1) * min_gap <= line2_width
    if can_fit_non_overlapping:
        best: List[int] = []
        for _ in range(max_attempts):
            starts: List[int] = []
            for _part_idx in range(num_parts):
                candidate = rng.randint(0, max_start)
                candidate_end = candidate + part_width
                ok = True
                for s in starts:
                    s_end = s + part_width
                    if not (candidate_end + min_gap <= s or s_end + min_gap <= candidate):
                        ok = False
                        break
                if ok:
                    starts.append(candidate)
            if len(starts) == num_parts:
                return sorted(starts)
            if len(starts) > len(best):
                best = starts
        if best:
            while len(best) < num_parts:
                best.append(rng.randint(0, max_start))
            return sorted(best[:num_parts])

    return sorted(rng.randint(0, max_start) for _ in range(num_parts))


def default_part_starts(
    line2_width: int,
    part_width: int,
    num_parts: int,
    mode: str,
    rng: random.Random,
    min_gap: int,
) -> List[int]:
    max_start = max(0, line2_width - part_width)
    if num_parts <= 1:
        if mode == "random":
            return [rng.randint(0, max_start)] if max_start > 0 else [0]
        return [max_start // 2]

    if mode == "random":
        return random_part_starts(
            line2_width=line2_width,
            part_width=part_width,
            num_parts=num_parts,
            rng=rng,
            min_gap=min_gap,
        )

    if mode == "even":
        return [int(round(x)) for x in np.linspace(0, max_start, num_parts)]

    # Arabic-friendly deterministic option: take parts from the right side moving left.
    starts = []
    for k in range(num_parts):
        starts.append(max(0, line2_width - (k + 1) * part_width))
    return starts


def resolve_line_paths(args: argparse.Namespace) -> Tuple[str, str]:
    if args.line1 and args.line2:
        return args.line1, args.line2

    if args.data_dir and args.index is not None:
        images_dir = os.path.join(args.data_dir, "images")
        return (
            os.path.join(images_dir, f"img1_{args.index}.png"),
            os.path.join(images_dir, f"img2_{args.index}.png"),
        )

    raise ValueError("Use either --line1/--line2 or --data-dir/--index")


def clamp_segment_to_image(x0: float, x1: float, image_width: int) -> Tuple[int, int]:
    if image_width <= 0:
        return 0, 1
    x0_i = int(round(max(0.0, min(float(x0), float(image_width)))))
    x1_i = int(round(max(float(x0_i) + 1.0, min(float(x1), float(image_width)))))
    if x1_i > image_width:
        x1_i = image_width
    if x0_i >= x1_i:
        x0_i = max(0, min(x0_i, image_width - 1))
        x1_i = min(image_width, x0_i + 1)
    return x0_i, x1_i


def cap_segment_width(
    x0: float,
    x1: float,
    max_width: int,
    image_width: int,
) -> Tuple[int, int]:
    """Keep a segment no wider than max_width, centered on the original segment."""
    x0_i, x1_i = clamp_segment_to_image(x0, x1, image_width)
    if max_width <= 0 or image_width <= max_width or (x1_i - x0_i) <= max_width:
        return x0_i, x1_i

    center = 0.5 * (x0_i + x1_i)
    start = int(round(center - max_width / 2.0))
    start = max(0, min(start, image_width - max_width))
    return start, start + max_width


def make_source_parts(
    line2_path: str,
    starts: Sequence[int],
    part_width: int,
    tmp_dir: str,
) -> List[SourcePart]:
    line2 = Image.open(line2_path).convert("RGB")
    width, height = line2.size

    parts: List[SourcePart] = []
    for part_id, raw_start in enumerate(starts, start=1):
        x0 = clamp_part_start(raw_start, width, part_width)
        x1 = min(width, x0 + part_width)
        crop = line2.crop((x0, 0, x1, height))

        part_path = os.path.join(tmp_dir, f"line2_part_{part_id}.png")
        crop.save(part_path)

        parts.append(
            SourcePart(
                part_id=part_id,
                path=part_path,
                x0_original=x0,
                x1_original=x1,
            )
        )
    return parts


# ---------------------------------------------------------------------------
# Smith-Waterman full-part / segment search
# ---------------------------------------------------------------------------


def _diag_pairs(sw_path: Sequence[Tuple[str, Optional[int], Optional[int]]]) -> List[Tuple[int, int]]:
    return [(i, j) for op, i, j in sw_path if op == "diag" and i is not None and j is not None]


def _segment_from_pairs(
    pairs: Sequence[Tuple[int, int]],
    sim: np.ndarray,
    match_kind: str,
) -> Optional[SWAlignedSegment]:
    if not pairs:
        return None
    sims = np.asarray([float(sim[i, j]) for i, j in pairs], dtype=np.float32)
    mean_sim = float(np.mean(sims))
    min_sim = float(np.min(sims))
    line1_start = int(min(i for i, _j in pairs))
    line1_end = int(max(i for i, _j in pairs))
    part_start = int(min(j for _i, j in pairs))
    part_end = int(max(j for _i, j in pairs))
    length = len(pairs)
    score = mean_sim + 1e-4 * length
    return SWAlignedSegment(
        line1_start=line1_start,
        line1_end=line1_end,
        part_start=part_start,
        part_end=part_end,
        length=length,
        mean_similarity=mean_sim,
        min_similarity=min_sim,
        score=float(score),
        match_kind=match_kind,
    )


def _passes_threshold(
    segment: SWAlignedSegment,
    threshold: float,
    min_run_length: int,
    require_all_windows_above_threshold: bool,
) -> bool:
    if segment.length < min_run_length:
        return False
    if segment.mean_similarity < threshold:
        return False
    if require_all_windows_above_threshold and segment.min_similarity < threshold:
        return False
    return True


def full_part_sw_segment(
    sw_path: Sequence[Tuple[str, Optional[int], Optional[int]]],
    sim: np.ndarray,
    threshold: float,
    min_run_length: int,
    require_all_windows_above_threshold: bool = False,
) -> Optional[SWAlignedSegment]:
    """Try to use the whole chosen part before falling back to a sub-segment.

    A full-part match means every part window has a diagonal match in the local
    Smith-Waterman path. The line1 span may have gaps, but the chosen line2 part
    is represented as a whole.
    """
    pairs = _diag_pairs(sw_path)
    if not pairs or sim.ndim != 2:
        return None

    _n_line1, n_part = sim.shape
    if n_part <= 0:
        return None

    covered_part_windows = {j for _i, j in pairs}
    if covered_part_windows != set(range(n_part)):
        return None

    segment = _segment_from_pairs(pairs, sim, match_kind="full_part")
    if segment is None:
        return None
    if not _passes_threshold(segment, threshold, min_run_length, require_all_windows_above_threshold):
        return None
    return segment


def best_partial_sw_segment(
    sw_path: Sequence[Tuple[str, Optional[int], Optional[int]]],
    sim: np.ndarray,
    threshold: float,
    min_run_length: int,
    require_all_windows_above_threshold: bool = False,
) -> Optional[SWAlignedSegment]:
    """Fallback: choose the best consecutive diagonal segment inside the part."""
    pairs = _diag_pairs(sw_path)
    if not pairs:
        return None

    runs: List[List[Tuple[int, int]]] = []
    current = [pairs[0]]
    for prev, cur in zip(pairs[:-1], pairs[1:]):
        if cur[0] == prev[0] + 1 and cur[1] == prev[1] + 1:
            current.append(cur)
        else:
            runs.append(current)
            current = [cur]
    runs.append(current)

    best: Optional[SWAlignedSegment] = None
    for run in runs:
        segment = _segment_from_pairs(run, sim, match_kind="partial_segment")
        if segment is None:
            continue
        if not _passes_threshold(segment, threshold, min_run_length, require_all_windows_above_threshold):
            continue
        if best is None or segment.score > best.score:
            best = segment
    return best


def choose_sw_segment(
    sw_path: Sequence[Tuple[str, Optional[int], Optional[int]]],
    sim: np.ndarray,
    threshold: float,
    min_run_length: int,
    require_all_windows_above_threshold: bool = False,
) -> Optional[SWAlignedSegment]:
    """First search for the full part; if not found, search for a segment."""
    full = full_part_sw_segment(
        sw_path,
        sim,
        threshold=threshold,
        min_run_length=min_run_length,
        require_all_windows_above_threshold=require_all_windows_above_threshold,
    )
    if full is not None:
        return full

    return best_partial_sw_segment(
        sw_path,
        sim,
        threshold=threshold,
        min_run_length=min_run_length,
        require_all_windows_above_threshold=require_all_windows_above_threshold,
    )


def best_possible_sw_segment_mean(
    sw_path: Sequence[Tuple[str, Optional[int], Optional[int]]],
    sim: np.ndarray,
) -> float:
    pairs = _diag_pairs(sw_path)
    if not pairs:
        return float("nan")
    sims = [float(sim[i, j]) for i, j in pairs]
    return float(np.mean(sims)) if sims else float("nan")


def window_span_to_original_pixels(
    start: int,
    end: int,
    num_windows: int,
    display_width: int,
    original_width: int,
    window_size: int,
    stride: int,
    use_flip: bool,
) -> Tuple[float, float]:
    x0_display, x1_display = window_range_to_pixels(
        start,
        end,
        num_windows,
        display_width,
        window_size,
        stride,
        use_flip,
    )
    display_to_original = original_width / float(display_width)
    return x0_display * display_to_original, x1_display * display_to_original


def search_one_part(
    model,
    emb_line1: torch.Tensor,
    part: SourcePart,
    line1_num_windows: int,
    line1_display_width: int,
    line1_original_width: int,
    args: argparse.Namespace,
) -> PartSearchResult:
    part_img_display, part_tensor = preprocess_line_image(part.path, target_height=args.height)
    emb_part = image_embeddings(model, part_tensor, args.device)

    # sim[i, j] = similarity between window i from full line1 and window j from the cropped line2 part.
    sim = cosine_similarity_matrix(emb_line1, emb_part)

    sw_path, sw_score, _H = smith_waterman(
        sim,
        threshold=args.threshold,
        gap_penalty=args.gap,
        match_reward=args.match,
        mismatch_penalty=args.mismatch,
    )

    segment = choose_sw_segment(
        sw_path,
        sim,
        threshold=args.threshold,
        min_run_length=args.min_run_length,
        require_all_windows_above_threshold=args.require_all_windows_above_threshold,
    )

    if segment is None:
        return PartSearchResult(
            part=part,
            found=False,
            part_window_count=int(emb_part.shape[0]),
            sw_score=float(sw_score),
            score=best_possible_sw_segment_mean(sw_path, sim),
            message=(
                "Smith-Waterman did not find the whole part or a fallback segment above --threshold; "
                "try lowering --threshold, making --mismatch less negative, making --gap less negative, "
                "or lowering --min-run-length"
            ),
        )

    # Convert the aligned line1 SW span to original line1 pixels. The mask is not
    # forced to --part-width, but capped to --part-width so it is equal or smaller
    # than the selected part.
    run_x0_original, run_x1_original = window_span_to_original_pixels(
        segment.line1_start,
        segment.line1_end,
        line1_num_windows,
        line1_display_width,
        line1_original_width,
        args.window_size,
        args.stride,
        args.use_flip,
    )
    match_x0, match_x1 = cap_segment_width(
        run_x0_original,
        run_x1_original,
        max_width=args.part_width,
        image_width=line1_original_width,
    )

    # Also locate where the full/partial match lies inside the chosen line2 part.
    # This is used only to place the arrow start. We do not draw a filled mask on line2.
    part_original_width = max(1, part.x1_original - part.x0_original)
    part_x0_rel, part_x1_rel = window_span_to_original_pixels(
        segment.part_start,
        segment.part_end,
        int(emb_part.shape[0]),
        int(part_img_display.size[0]),
        part_original_width,
        args.window_size,
        args.stride,
        args.use_flip,
    )
    part_x0_rel_i, part_x1_rel_i = cap_segment_width(
        part_x0_rel,
        part_x1_rel,
        max_width=part_original_width,
        image_width=part_original_width,
    )

    return PartSearchResult(
        part=part,
        found=True,
        match_x0_original=int(match_x0),
        match_x1_original=int(match_x1),
        part_match_x0_relative=int(part_x0_rel_i),
        part_match_x1_relative=int(part_x1_rel_i),
        run_x0_original=float(run_x0_original),
        run_x1_original=float(run_x1_original),
        line1_window_start=int(segment.line1_start),
        line1_window_end=int(segment.line1_end),
        part_window_start=int(segment.part_start),
        part_window_end=int(segment.part_end),
        part_window_count=int(emb_part.shape[0]),
        run_length=int(segment.length),
        mean_similarity=float(segment.mean_similarity),
        min_similarity=float(segment.min_similarity),
        sw_score=float(sw_score),
        score=float(segment.score),
        match_kind=segment.match_kind,
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def crop_or_blank(
    image: Image.Image,
    x0: int,
    x1: int,
    width: int,
    fill: int = 255,
) -> Image.Image:
    """Crop exactly width pixels when possible; otherwise pad to width."""
    image = image.convert("RGB")
    img_w, img_h = image.size
    x0 = max(0, min(int(x0), img_w))
    x1 = max(x0, min(int(x1), img_w))
    crop = image.crop((x0, 0, x1, img_h))

    if crop.size[0] == width:
        return crop

    canvas = Image.new("RGB", (width, img_h), (fill, fill, fill))
    canvas.paste(crop, (0, 0))
    return canvas


def draw_line2_part_on_full_line1_canvas(
    ax,
    line1: Image.Image,
    line2: Image.Image,
    results: Sequence[PartSearchResult],
    title: str,
    part_width: int,
):
    arr1 = np.array(line1.convert("RGB"))
    h1, w1 = arr1.shape[:2]
    line2_h = line2.size[1]

    part_gap = max(18, int(round(part_width * 0.25)))
    n_parts = max(1, len(results))
    total_parts_w = n_parts * part_width + (n_parts - 1) * part_gap

    canvas_w = max(w1, total_parts_w)
    x_offset_line1 = 0 if canvas_w == w1 else int(round((canvas_w - w1) / 2.0))

    top_margin = 34
    arrow_gap = 95
    bottom_margin = 28
    y1_top = top_margin
    y1_bottom = y1_top + h1
    y_parts_top = y1_bottom + arrow_gap
    y_parts_bottom = y_parts_top + line2_h
    canvas_h = y_parts_bottom + bottom_margin

    ax.imshow(arr1, extent=(x_offset_line1, x_offset_line1 + w1, y1_bottom, y1_top), zorder=1)

    ax.text(
        x_offset_line1,
        y1_top - 8,
        "Line 1: full searched line with masks",
        fontsize=11,
        weight="bold",
        va="bottom",
    )
    ax.text(
        0,
        y_parts_top - 8,
        "Line 2: chosen parts only, no masks",
        fontsize=11,
        weight="bold",
        va="bottom",
    )

    part_positions: List[Tuple[int, int]] = []
    if n_parts == 1:
        start = int(round((canvas_w - part_width) / 2.0))
        part_positions.append((start, start + part_width))
    else:
        start0 = int(round((canvas_w - total_parts_w) / 2.0))
        for idx in range(n_parts):
            px0 = start0 + idx * (part_width + part_gap)
            part_positions.append((px0, px0 + part_width))

    for idx, (result, (part_canvas_x0, part_canvas_x1)) in enumerate(zip(results, part_positions)):
        color = PALETTE[idx % len(PALETTE)]
        part = result.part

        part_crop = crop_or_blank(
            line2,
            part.x0_original,
            part.x1_original,
            part_width,
        )
        part_arr = np.array(part_crop)
        ax.imshow(
            part_arr,
            extent=(part_canvas_x0, part_canvas_x1, y_parts_bottom, y_parts_top),
            zorder=1,
        )

        # Border only around the chosen source crop. Do not mask/fill line2.
        ax.add_patch(
            Rectangle(
                (part_canvas_x0, y_parts_top),
                part_width,
                line2_h,
                facecolor="none",
                edgecolor=color,
                linewidth=2,
                alpha=0.95,
                zorder=3,
            )
        )

        part_center_x = 0.5 * (part_canvas_x0 + part_canvas_x1)

        if not result.found:
            ax.text(
                part_center_x,
                y_parts_top + 0.5 * line2_h,
                f"part {part.part_id}\nnot found",
                ha="center",
                va="center",
                fontsize=8,
                color="white",
                weight="bold",
                bbox=dict(facecolor=color, edgecolor="none", alpha=0.72, boxstyle="round,pad=0.25"),
                zorder=5,
            )
            print(f"part {part.part_id}: NOT FOUND | best_sw_diag_mean={result.score:.4f} | {result.message}")
            continue

        assert result.match_x0_original is not None and result.match_x1_original is not None
        assert result.part_match_x0_relative is not None and result.part_match_x1_relative is not None

        # Mask only the aligned SW segment on full line1.
        match_x0 = x_offset_line1 + result.match_x0_original
        match_x1 = x_offset_line1 + result.match_x1_original
        mask_width = max(1, match_x1 - match_x0)
        ax.add_patch(
            Rectangle(
                (match_x0, y1_top),
                mask_width,
                h1,
                facecolor=color,
                edgecolor=color,
                linewidth=2,
                alpha=0.30,
                zorder=3,
            )
        )

        match_kind_label = "full" if result.match_kind == "full_part" else "segment"
        ax.text(
            0.5 * (match_x0 + match_x1),
            y1_top + 0.5 * h1,
            f"part {part.part_id}\n{match_kind_label}\nsim={result.mean_similarity:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            weight="bold",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.72, boxstyle="round,pad=0.25"),
            zorder=5,
        )
        ax.text(
            part_center_x,
            y_parts_top + 0.5 * line2_h,
            f"part {part.part_id}\n{match_kind_label}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            weight="bold",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.72, boxstyle="round,pad=0.25"),
            zorder=5,
        )

        match_center_x = 0.5 * (match_x0 + match_x1)
        # The arrow starts from the aligned sub-range inside the chosen part, but
        # no visual mask is drawn there.
        src_x0 = part_canvas_x0 + result.part_match_x0_relative
        src_x1 = part_canvas_x0 + result.part_match_x1_relative
        src_center_x = 0.5 * (src_x0 + src_x1)
        ax.add_patch(
            FancyArrowPatch(
                (src_center_x, y_parts_top - 4),
                (match_center_x, y1_bottom + 4),
                arrowstyle="->",
                mutation_scale=18,
                linewidth=2.4,
                color=color,
                alpha=0.95,
                zorder=4,
            )
        )

        print(
            f"part {part.part_id}: FOUND | "
            f"kind={result.match_kind} | "
            f"line2_x=[{part.x0_original},{part.x1_original}] | "
            f"line2_aligned_rel_x=[{result.part_match_x0_relative},{result.part_match_x1_relative}] | "
            f"line1_mask_x=[{result.match_x0_original},{result.match_x1_original}] "
            f"width={result.match_x1_original - result.match_x0_original} | "
            f"line1_windows=[{result.line1_window_start},{result.line1_window_end}] | "
            f"part_windows=[{result.part_window_start},{result.part_window_end}]/{result.part_window_count} | "
            f"raw_window_x=[{result.run_x0_original:.1f},{result.run_x1_original:.1f}] | "
            f"sw_path_score={result.sw_score:.4f} | "
            f"mean_sim={result.mean_similarity:.4f} | "
            f"min_sim={result.min_similarity:.4f}"
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)
    ax.axis("off")


def draw_results(
    line1_original: Image.Image,
    line2_original: Image.Image,
    results: Sequence[PartSearchResult],
    output: str,
    title: str,
    part_width: int,
):
    line1_w, line1_h = line1_original.size
    n_parts = max(1, len(results))
    fig_w = max(12.0, line1_w / 100.0)
    fig_h = max(5.0, (line1_h + line2_original.size[1] + 160) / 100.0)
    if n_parts > 4:
        fig_h += 0.35 * (n_parts - 4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    draw_line2_part_on_full_line1_canvas(
        ax,
        line1_original,
        line2_original,
        results,
        title,
        part_width,
    )

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Randomly crop fixed-width parts from line2 and find their Smith-Waterman aligned full/partial segment in full line1."
    )
    parser.add_argument("--weights", required=True, help="Path to trained model .pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--match", type=float, default=1.0, help="Reward scale for similarities above threshold")
    parser.add_argument("--mismatch", type=float, default=-1.5, help="Penalty scale for similarities below threshold; should be negative")
    parser.add_argument("--gap", type=float, default=-0.15, help="Smith-Waterman gap penalty; should be negative")
    parser.add_argument("--min-run-length", type=int, default=3)
    parser.add_argument(
        "--require-all-windows-above-threshold",
        action="store_true",
        help="Require every diagonal window similarity in the chosen SW full/partial segment to be >= --threshold, not only the mean.",
    )
    parser.add_argument("--use-flip", action="store_true", help="Use when Arabic windows were encoded right-to-left")
    parser.add_argument("--no-bilstm", action="store_true", help="Disable BiLSTM even if checkpoint config is missing")

    # Input line options.
    parser.add_argument("--line1", default=None, help="Path to the full first line image")
    parser.add_argument("--line2", default=None, help="Path to the second line image to crop parts from")
    parser.add_argument("--data-dir", default=None, help="Dataset root containing images/img1_<idx>.png and images/img2_<idx>.png")
    parser.add_argument("--index", type=int, default=None, help="Dataset sample index")

    # Part selection options.
    parser.add_argument("--part-width", type=int, default=124, help="Width in original line2 pixels for each cropped part")
    parser.add_argument("--num-parts", type=int, default=3, help="Number of random parts to crop when --part-starts is not given")
    parser.add_argument(
        "--part-starts",
        default=None,
        help='Comma-separated x-start positions in original line2 pixels, e.g. "100,350,600". Overrides random selection.',
    )
    parser.add_argument(
        "--part-mode",
        choices=("random", "rtl-blocks", "even"),
        default="random",
        help="How to choose parts when --part-starts is not provided. Default is random every run.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for reproducible random part selection. Omit it to choose different random parts every run.",
    )
    parser.add_argument(
        "--random-min-gap",
        type=int,
        default=0,
        help="Minimum gap in pixels between random parts when non-overlapping selection is possible.",
    )

    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    if args.part_width <= 0:
        raise ValueError("--part-width must be positive")
    if args.num_parts <= 0:
        raise ValueError("--num-parts must be positive")
    if args.min_run_length <= 0:
        raise ValueError("--min-run-length must be positive")
    if args.random_min_gap < 0:
        raise ValueError("--random-min-gap must be >= 0")
    if args.gap > 0:
        raise ValueError("--gap should be negative, for example --gap -0.15")
    if args.mismatch > 0:
        raise ValueError("--mismatch should be negative, for example --mismatch -1.5")
    if args.match <= 0:
        raise ValueError("--match should be positive, for example --match 1.0")

    line1_path, line2_path = resolve_line_paths(args)
    if not os.path.exists(line1_path):
        raise FileNotFoundError(line1_path)
    if not os.path.exists(line2_path):
        raise FileNotFoundError(line2_path)

    print(f"Line1: {line1_path}")
    print(f"Line2: {line2_path}")

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
    emb_line1 = image_embeddings(model, tensor1, args.device)

    line1_original = Image.open(line1_path).convert("RGB")
    line2_original = Image.open(line2_path).convert("RGB")
    line2_original_width, _ = line2_original.size

    starts = parse_part_starts(args.part_starts)
    selection_mode = "manual"
    if starts is None:
        rng = random.Random(args.random_seed)
        starts = default_part_starts(
            line2_original_width,
            args.part_width,
            args.num_parts,
            args.part_mode,
            rng,
            args.random_min_gap,
        )
        selection_mode = args.part_mode

    print(f"Part width: {args.part_width}")
    print(f"Part selection mode: {selection_mode}")
    print(f"Random seed: {args.random_seed if args.random_seed is not None else 'none'}")
    print(f"Part starts: {starts}")
    print("Search method: Smith-Waterman; first full selected part, then fallback segment; masks only on line1")

    results: List[PartSearchResult] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        parts = make_source_parts(
            line2_path,
            starts,
            args.part_width,
            tmp_dir,
        )

        for part in parts:
            result = search_one_part(
                model,
                emb_line1,
                part,
                line1_num_windows=int(emb_line1.shape[0]),
                line1_display_width=int(img1_display.size[0]),
                line1_original_width=int(line1_original.size[0]),
                args=args,
            )
            results.append(result)

    sample_label = f" sample {args.index}" if args.index is not None else ""
    title = (
        f"Line2 {selection_mode} parts searched inside full line1{sample_label} | "
        f"SW full-part first, fallback segment, thr={args.threshold}, gap={args.gap}"
    )
    draw_results(line1_original, line2_original, results, args.output, title, args.part_width)


if __name__ == "__main__":
    main()
