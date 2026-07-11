#!/usr/bin/env python3
"""
Search selected fixed-width parts from line2 inside the full line1 image.

The script:
  1. loads the trained image encoder,
  2. takes the whole first line image as the search target,
  3. crops N fixed-width parts from the second line image,
  4. aligns each cropped part against the whole first line using Smith-Waterman,
  5. displays only:
       - the selected line2 crop,
       - the fixed-width line1 crop where it was found,
     not the whole line images,
  6. masks both displayed crops with the same color and connects them with an arrow.

Important:
  The matched crop in line1 is forced to have exactly --part-width original pixels.
  It is centered around the Smith-Waterman matched run, but its width is not taken
  from the run length. This keeps the source part and the matched part equal width.

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
      --match 1.0 \
      --mismatch -1.5 \
      --gap -0.15 \
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
    longest_consecutive_diagonal_run,
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
class PartSearchResult:
    part: SourcePart
    found: bool
    match_x0_original: Optional[int] = None
    match_x1_original: Optional[int] = None
    run_x0_original: Optional[float] = None
    run_x1_original: Optional[float] = None
    run_length: int = 0
    mean_similarity: float = 0.0
    sw_score: float = 0.0
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
# Search
# ---------------------------------------------------------------------------


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

    sw_path, sw_score, _H = smith_waterman(
        sim,
        threshold=args.threshold,
        gap_penalty=args.gap,
        match_reward=args.match,
        mismatch_penalty=args.mismatch,
    )

    run = longest_consecutive_diagonal_run(
        sw_path,
        sim,
        sw_score,
        min_run_length=args.min_run_length,
    )

    if run is None:
        return PartSearchResult(
            part=part,
            found=False,
            sw_score=float(sw_score),
            message=(
                "no consecutive diagonal run found; try lowering --threshold, "
                "making --mismatch less negative, or lowering --min-run-length"
            ),
        )

    # This is only used to find the center of the match. The final displayed
    # line1 crop is forced to exactly --part-width original pixels.
    run_x0_display, run_x1_display = window_range_to_pixels(
        run.line1_start,
        run.line1_end,
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
        run_length=int(run.length),
        mean_similarity=float(run.mean_similarity),
        sw_score=float(run.sw_score),
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


def draw_masked_crop(ax, crop: Image.Image, color: str, label: str):
    arr = np.array(crop)
    h, w = arr.shape[:2]

    ax.imshow(arr, aspect="auto")
    ax.add_patch(
        Rectangle(
            (0, 0),
            w,
            h,
            facecolor=color,
            edgecolor=color,
            linewidth=2,
            alpha=0.30,
        )
    )
    ax.text(
        w * 0.5,
        h * 0.5,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color="white",
        weight="bold",
        bbox=dict(facecolor=color, edgecolor="none", alpha=0.72, boxstyle="round,pad=0.25"),
    )
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")


def draw_arrow(ax, color: str, found: bool):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    if found:
        ax.add_patch(
            FancyArrowPatch(
                (0.05, 0.5),
                (0.95, 0.5),
                transform=ax.transAxes,
                arrowstyle="->",
                mutation_scale=18,
                linewidth=2.4,
                color=color,
                alpha=0.95,
            )
        )


def draw_results(
    line1_original: Image.Image,
    line2_original: Image.Image,
    results: Sequence[PartSearchResult],
    output: str,
    title: str,
    part_width: int,
):
    n_rows = max(1, len(results))
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(9.0, max(2.4, 2.25 * n_rows)),
        gridspec_kw={"width_ratios": [1.0, 0.22, 1.0], "wspace": 0.05, "hspace": 0.28},
    )

    if n_rows == 1:
        axes = np.asarray([axes])

    fig.suptitle(title, fontsize=12, weight="bold", y=0.995)
    axes[0, 0].set_title(f"Line2 chosen part\nwidth={part_width}px", fontsize=10)
    axes[0, 2].set_title(f"Line1 matched crop\nforced width={part_width}px", fontsize=10)

    line1_height = line1_original.size[1]

    for idx, result in enumerate(results):
        color = PALETTE[idx % len(PALETTE)]
        part = result.part

        part_crop = crop_or_blank(
            line2_original,
            part.x0_original,
            part.x1_original,
            part_width,
        )
        source_label = f"part {part.part_id}\nx=[{part.x0_original},{part.x1_original}]"
        draw_masked_crop(axes[idx, 0], part_crop, color, source_label)

        draw_arrow(axes[idx, 1], color, result.found)

        if result.found:
            assert result.match_x0_original is not None and result.match_x1_original is not None
            match_crop = crop_or_blank(
                line1_original,
                result.match_x0_original,
                result.match_x1_original,
                part_width,
            )
            match_label = (
                f"match {part.part_id}\n"
                f"x=[{result.match_x0_original},{result.match_x1_original}]\n"
                f"sim={result.mean_similarity:.3f}"
            )
            draw_masked_crop(axes[idx, 2], match_crop, color, match_label)

            print(
                f"part {part.part_id}: FOUND | "
                f"line1_fixed_x=[{result.match_x0_original},{result.match_x1_original}] "
                f"width={result.match_x1_original - result.match_x0_original} | "
                f"raw_run_x=[{result.run_x0_original:.1f},{result.run_x1_original:.1f}] | "
                f"run_len={result.run_length} | "
                f"mean_sim={result.mean_similarity:.4f} | "
                f"score={result.sw_score:.4f}"
            )
        else:
            blank = Image.new("RGB", (part_width, line1_height), (255, 255, 255))
            draw_masked_crop(axes[idx, 2], blank, color, f"part {part.part_id}\nnot found")
            print(f"part {part.part_id}: NOT FOUND | score={result.sw_score:.4f} | {result.message}")

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.94))
    plt.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Crop fixed-width parts from line2 and find where they appear in full line1."
    )
    parser.add_argument("--weights", required=True, help="Path to trained model .pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.86)
    parser.add_argument("--match", type=float, default=1.0)
    parser.add_argument("--mismatch", type=float, default=-1.5)
    parser.add_argument("--gap", type=float, default=-0.15)
    parser.add_argument("--min-run-length", type=int, default=3)
    parser.add_argument("--use-flip", action="store_true", help="Use when Arabic windows were encoded right-to-left")
    parser.add_argument("--no-bilstm", action="store_true", help="Disable BiLSTM even if checkpoint config is missing")

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
    if starts is None:
        starts = default_part_starts(
            line2_original_width,
            args.part_width,
            args.num_parts,
            args.part_mode,
        )

    print(f"Part width: {args.part_width}")
    print(f"Part starts: {starts}")

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
        f"Line2 fixed-width parts searched inside line1{sample_label} | "
        f"part_width={args.part_width}, thr={args.threshold}, gap={args.gap}"
    )
    draw_results(line1_original, line2_original, results, args.output, title, args.part_width)


if __name__ == "__main__":
    main()
