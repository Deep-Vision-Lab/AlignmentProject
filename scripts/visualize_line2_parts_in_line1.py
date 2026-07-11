#!/usr/bin/env python3
"""
Search selected fixed-width parts from line2 inside the full line1 image.

The script:
  1. loads the trained image encoder,
  2. takes the whole first line image as the search target,
  3. crops N fixed-width parts from the second line image,
  4. for every cropped part, searches the full line1 for the consecutive
     window block with the highest diagonal similarity to that part,
  5. accepts the block only when its similarity is above --threshold,
  6. shows the full line1 image with fixed-width masks where each part was found,
  7. shows only the chosen line2 parts, not the whole line2 image,
  8. connects each chosen part to its masked match in line1 using the same color.

Important:
  The matched mask in line1 is forced to have exactly --part-width original pixels.
  The search uses consecutive line1 windows, not Smith-Waterman. For a cropped
  part with K windows, it checks every K-window consecutive block in line1 and
  chooses the block with the highest mean diagonal cosine similarity:

      score(start) = mean_k cosine(line1[start+k], part[k])

Example:
    python scripts/visualize_line2_parts_in_line1.py \
      --data-dir DataSet/Synthetic_Arabic \
      --index 10 \
      --weights Weights/span_jax_best_quality_win32_offline/model_latest.pth \
      --output Results/Evaluation/Part_Search/sample10_parts.png \
      --part-width 124 \
      --num-parts 3 \
      --window-size 32 \
      --stride 16 \
      --height 128 \
      --threshold 0.86 \
      --min-run-length 3 \
      --use-flip

Manual part positions from the left side of line2:
    --part-starts "100,350,600"
"""

from __future__ import annotations

import argparse
import os
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
    window_range_to_pixels,
)


@dataclass
class SourcePart:
    part_id: int
    path: str
    x0_original: int
    x1_original: int


@dataclass
class PartSearchResult:
    part: SourcePart
    found: bool
    match_x0_original: Optional[int] = None
    match_x1_original: Optional[int] = None
    run_x0_original: Optional[float] = None
    run_x1_original: Optional[float] = None
    line1_window_start: Optional[int] = None
    line1_window_end: Optional[int] = None
    part_window_count: int = 0
    run_length: int = 0
    mean_similarity: float = 0.0
    min_similarity: float = 0.0
    score: float = 0.0
    message: str = ""


@dataclass
class ConsecutiveWindowMatch:
    line1_start: int
    line1_end: int
    part_start: int
    part_end: int
    length: int
    mean_similarity: float
    min_similarity: float
    score: float


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


def default_part_starts(
    line2_width: int,
    part_width: int,
    num_parts: int,
    mode: str,
) -> List[int]:
    max_start = max(0, line2_width - part_width)
    if num_parts <= 1:
        return [max_start // 2]

    if mode == "even":
        return [int(round(x)) for x in np.linspace(0, max_start, num_parts)]

    # Arabic-friendly default: take parts from the right side moving left.
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


def clamp_exact_width_start(center: float, image_width: int, crop_width: int) -> int:
    """Return a crop start whose output width is exactly crop_width when possible."""
    if image_width <= crop_width:
        return 0
    start = int(round(center - crop_width / 2.0))
    return max(0, min(start, image_width - crop_width))


def clamp_part_start(start: int, line_width: int, part_width: int) -> int:
    if line_width <= part_width:
        return 0
    return max(0, min(int(start), line_width - part_width))


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
# Consecutive-window search
# ---------------------------------------------------------------------------


def best_consecutive_window_match(
    sim: np.ndarray,
    threshold: float,
    min_run_length: int,
    require_all_windows_above_threshold: bool = False,
) -> Optional[ConsecutiveWindowMatch]:
    """Find the consecutive line1 window block most similar to the part.

    sim has shape [line1_windows, part_windows]. For a part with K windows, we
    scan every K-window block in line1 and score it by the diagonal similarity:

        line1[start + k]  <->  part[k]

    A candidate is accepted when its mean similarity is >= threshold. When
    --require-all-windows-above-threshold is used, every diagonal element must
    also be >= threshold.
    """
    if sim.ndim != 2:
        raise ValueError(f"Expected 2D similarity matrix, got shape {sim.shape}")

    n_line1, n_part = sim.shape
    run_len = int(n_part)
    if run_len <= 0 or n_line1 <= 0:
        return None
    if run_len < min_run_length:
        return None
    if run_len > n_line1:
        return None

    best: Optional[ConsecutiveWindowMatch] = None

    for start in range(0, n_line1 - run_len + 1):
        diag = np.asarray([float(sim[start + k, k]) for k in range(run_len)], dtype=np.float32)
        mean_sim = float(np.mean(diag))
        min_sim = float(np.min(diag))

        if mean_sim < threshold:
            continue
        if require_all_windows_above_threshold and min_sim < threshold:
            continue

        candidate = ConsecutiveWindowMatch(
            line1_start=start,
            line1_end=start + run_len - 1,
            part_start=0,
            part_end=run_len - 1,
            length=run_len,
            mean_similarity=mean_sim,
            min_similarity=min_sim,
            score=mean_sim,
        )
        if best is None or candidate.score > best.score:
            best = candidate

    return best


def search_one_part(
    model,
    emb_line1: torch.Tensor,
    part: SourcePart,
    line1_num_windows: int,
    line1_display_width: int,
    line1_original_width: int,
    args: argparse.Namespace,
) -> PartSearchResult:
    _part_img, part_tensor = preprocess_line_image(part.path, target_height=args.height)
    emb_part = image_embeddings(model, part_tensor, args.device)

    # sim[i, j] = similarity between window i from full line1 and window j from the cropped line2 part.
    sim = cosine_similarity_matrix(emb_line1, emb_part)

    match = best_consecutive_window_match(
        sim,
        threshold=args.threshold,
        min_run_length=args.min_run_length,
        require_all_windows_above_threshold=args.require_all_windows_above_threshold,
    )

    if match is None:
        best_possible = float("nan")
        if sim.ndim == 2 and sim.shape[0] > 0 and sim.shape[1] > 0 and sim.shape[1] <= sim.shape[0]:
            n_part = sim.shape[1]
            possible_scores = [
                float(np.mean([sim[start + k, k] for k in range(n_part)]))
                for start in range(0, sim.shape[0] - n_part + 1)
            ]
            if possible_scores:
                best_possible = max(possible_scores)

        return PartSearchResult(
            part=part,
            found=False,
            part_window_count=int(emb_part.shape[0]),
            score=best_possible,
            message=(
                "no consecutive line1 window block had mean similarity above --threshold; "
                "try lowering --threshold or lowering --min-run-length"
            ),
        )

    # This is only used to find the center of the match. The final line1 mask
    # is forced to exactly --part-width original pixels.
    run_x0_display, run_x1_display = window_range_to_pixels(
        match.line1_start,
        match.line1_end,
        line1_num_windows,
        line1_display_width,
        args.window_size,
        args.stride,
        args.use_flip,
    )

    display_to_original = line1_original_width / float(line1_display_width)
    run_x0_original = run_x0_display * display_to_original
    run_x1_original = run_x1_display * display_to_original
    run_center_original = 0.5 * (run_x0_original + run_x1_original)

    fixed_x0 = clamp_exact_width_start(
        run_center_original,
        line1_original_width,
        args.part_width,
    )
    fixed_x1 = min(line1_original_width, fixed_x0 + args.part_width)

    return PartSearchResult(
        part=part,
        found=True,
        match_x0_original=int(fixed_x0),
        match_x1_original=int(fixed_x1),
        run_x0_original=float(run_x0_original),
        run_x1_original=float(run_x1_original),
        line1_window_start=int(match.line1_start),
        line1_window_end=int(match.line1_end),
        part_window_count=int(emb_part.shape[0]),
        run_length=int(match.length),
        mean_similarity=float(match.mean_similarity),
        min_similarity=float(match.min_similarity),
        score=float(match.score),
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
        "Line 1: full searched line",
        fontsize=11,
        weight="bold",
        va="bottom",
    )
    ax.text(
        0,
        y_parts_top - 8,
        "Line 2: chosen parts only",
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

        # Colored mask/border on the selected line2 part crop.
        ax.add_patch(
            Rectangle(
                (part_canvas_x0, y_parts_top),
                part_width,
                line2_h,
                facecolor=color,
                edgecolor=color,
                linewidth=2,
                alpha=0.28,
                zorder=3,
            )
        )
        ax.text(
            0.5 * (part_canvas_x0 + part_canvas_x1),
            y_parts_top + 0.5 * line2_h,
            f"part {part.part_id}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            weight="bold",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.72, boxstyle="round,pad=0.25"),
            zorder=5,
        )

        part_center_x = 0.5 * (part_canvas_x0 + part_canvas_x1)

        if not result.found:
            ax.text(
                part_center_x,
                y_parts_bottom + 12,
                "not found",
                ha="center",
                va="top",
                fontsize=8,
                color=color,
                weight="bold",
            )
            print(f"part {part.part_id}: NOT FOUND | best_mean={result.score:.4f} | {result.message}")
            continue

        assert result.match_x0_original is not None and result.match_x1_original is not None
        match_x0 = x_offset_line1 + result.match_x0_original
        match_x1 = x_offset_line1 + result.match_x1_original
        mask_width = match_x1 - match_x0

        # Fixed-width mask on full line1. Width equals --part-width.
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
        ax.text(
            0.5 * (match_x0 + match_x1),
            y1_top + 0.5 * h1,
            f"part {part.part_id}\nsim={result.mean_similarity:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            weight="bold",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.72, boxstyle="round,pad=0.25"),
            zorder=5,
        )

        match_center_x = 0.5 * (match_x0 + match_x1)
        ax.add_patch(
            FancyArrowPatch(
                (part_center_x, y_parts_top - 4),
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
            f"line1_mask_x=[{result.match_x0_original},{result.match_x1_original}] "
            f"width={result.match_x1_original - result.match_x0_original} | "
            f"line1_windows=[{result.line1_window_start},{result.line1_window_end}] | "
            f"part_windows={result.part_window_count} | "
            f"raw_window_x=[{result.run_x0_original:.1f},{result.run_x1_original:.1f}] | "
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
        description="Crop fixed-width parts from line2 and find the highest-similarity consecutive window block in full line1."
    )
    parser.add_argument("--weights", required=True, help="Path to trained model .pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--min-run-length", type=int, default=3)
    parser.add_argument(
        "--require-all-windows-above-threshold",
        action="store_true",
        help="Require every diagonal window similarity in the chosen block to be >= --threshold, not only the mean.",
    )
    parser.add_argument("--use-flip", action="store_true", help="Use when Arabic windows were encoded right-to-left")
    parser.add_argument("--no-bilstm", action="store_true", help="Disable BiLSTM even if checkpoint config is missing")

    # Kept for backward compatibility with older commands. They are not used by
    # the consecutive-window search.
    parser.add_argument("--match", type=float, default=1.0, help=argparse.SUPPRESS)
    parser.add_argument("--mismatch", type=float, default=-1.5, help=argparse.SUPPRESS)
    parser.add_argument("--gap", type=float, default=-0.15, help=argparse.SUPPRESS)

    # Input line options.
    parser.add_argument("--line1", default=None, help="Path to the full first line image")
    parser.add_argument("--line2", default=None, help="Path to the second line image to crop parts from")
    parser.add_argument("--data-dir", default=None, help="Dataset root containing images/img1_<idx>.png and images/img2_<idx>.png")
    parser.add_argument("--index", type=int, default=None, help="Dataset sample index")

    # Part selection options.
    parser.add_argument("--part-width", type=int, default=124, help="Width in original line2 pixels for each cropped part")
    parser.add_argument("--num-parts", type=int, default=3, help="Number of parts to crop when --part-starts is not given")
    parser.add_argument(
        "--part-starts",
        default=None,
        help='Comma-separated x-start positions in original line2 pixels, e.g. "100,350,600"',
    )
    parser.add_argument(
        "--part-mode",
        choices=("rtl-blocks", "even"),
        default="rtl-blocks",
        help="Default crop placement when --part-starts is not provided. rtl-blocks starts from the right side.",
    )

    parser.add_argument("--output", required=True, help="Output PNG path")
    args = parser.parse_args()

    if args.part_width <= 0:
        raise ValueError("--part-width must be positive")
    if args.min_run_length <= 0:
        raise ValueError("--min-run-length must be positive")

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
    if starts is None:
        starts = default_part_starts(
            line2_original_width,
            args.part_width,
            args.num_parts,
            args.part_mode,
        )

    print(f"Part width: {args.part_width}")
    print(f"Part starts: {starts}")
    print("Search method: highest mean similarity over consecutive line1 windows")

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
        f"Line2 chosen parts searched inside full line1{sample_label} | "
        f"part_width={args.part_width}, consecutive-window mean thr={args.threshold}"
    )
    draw_results(line1_original, line2_original, results, args.output, title, args.part_width)


if __name__ == "__main__":
    main()
