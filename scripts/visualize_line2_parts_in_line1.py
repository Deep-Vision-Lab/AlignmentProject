#!/usr/bin/env python3
"""
Search selected fixed-width parts from line2 inside the full line1 image.

The script:
  1. loads the trained image encoder,
  2. takes the whole first line image,
  3. crops N parts from the second line image, each with --part-width pixels,
  4. aligns each part against the whole first line using Smith-Waterman,
  5. masks the matched region in line1,
  6. masks the selected source part in line2,
  7. connects each match with an arrow.

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
    x0_display: int
    x1_display: int


@dataclass
class PartSearchResult:
    part: SourcePart
    found: bool
    match_x0: Optional[int] = None
    match_x1: Optional[int] = None
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


def clamp_part_start(start: int, line_width: int, part_width: int) -> int:
    if line_width <= part_width:
        return 0
    return max(0, min(int(start), line_width - part_width))


def make_source_parts(
    line2_path: str,
    starts: Sequence[int],
    part_width: int,
    display_scale: float,
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
                x0_display=int(round(x0 * display_scale)),
                x1_display=int(round(x1 * display_scale)),
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

    match_x0, match_x1 = window_range_to_pixels(
        run.line1_start,
        run.line1_end,
        line1_num_windows,
        line1_display_width,
        args.window_size,
        args.stride,
        args.use_flip,
    )

    return PartSearchResult(
        part=part,
        found=True,
        match_x0=match_x0,
        match_x1=match_x1,
        run_length=int(run.length),
        mean_similarity=float(run.mean_similarity),
        sw_score=float(run.sw_score),
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def draw_results(
    img1: Image.Image,
    img2: Image.Image,
    results: Sequence[PartSearchResult],
    output: str,
    title: str,
):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    h1, w1 = arr1.shape[:2]
    h2, w2 = arr2.shape[:2]

    canvas_w = max(w1, w2)
    gap = 110
    top = 30
    y1_top = top
    y1_bottom = y1_top + h1
    y2_top = y1_bottom + gap
    y2_bottom = y2_top + h2

    fig_w = max(11.0, canvas_w / 100.0)
    fig_h = max(4.8, (y2_bottom + 40) / 100.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    ax.imshow(arr1, extent=(0, w1, y1_bottom, y1_top))
    ax.imshow(arr2, extent=(0, w2, y2_bottom, y2_top))

    ax.text(0, y1_top - 8, "Line 1: searched full line", fontsize=11, weight="bold")
    ax.text(0, y2_top - 8, "Line 2: selected fixed-width parts", fontsize=11, weight="bold")

    for idx, result in enumerate(results):
        color = PALETTE[idx % len(PALETTE)]
        part = result.part

        # Mask the selected source crop on line2.
        ax.add_patch(
            Rectangle(
                (part.x0_display, y2_top),
                max(1, part.x1_display - part.x0_display),
                h2,
                facecolor=color,
                edgecolor=color,
                linewidth=2,
                alpha=0.28,
            )
        )

        source_label = f"part {part.part_id}"
        if not result.found:
            source_label += "\nnot found"
        ax.text(
            0.5 * (part.x0_display + part.x1_display),
            y2_top + 0.5 * h2,
            source_label,
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            weight="bold",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.72, boxstyle="round,pad=0.25"),
        )

        if not result.found:
            print(f"part {part.part_id}: NOT FOUND | score={result.sw_score:.4f} | {result.message}")
            continue

        assert result.match_x0 is not None and result.match_x1 is not None

        # Mask the discovered match on line1.
        ax.add_patch(
            Rectangle(
                (result.match_x0, y1_top),
                max(1, result.match_x1 - result.match_x0),
                h1,
                facecolor=color,
                edgecolor=color,
                linewidth=2,
                alpha=0.30,
            )
        )

        ax.text(
            0.5 * (result.match_x0 + result.match_x1),
            y1_top + 0.5 * h1,
            f"part {part.part_id}\nsim={result.mean_similarity:.3f}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            weight="bold",
            bbox=dict(facecolor=color, edgecolor="none", alpha=0.72, boxstyle="round,pad=0.25"),
        )

        cx1 = 0.5 * (result.match_x0 + result.match_x1)
        cx2 = 0.5 * (part.x0_display + part.x1_display)
        ax.add_patch(
            FancyArrowPatch(
                (cx1, y1_bottom + 4),
                (cx2, y2_top - 4),
                arrowstyle="->",
                mutation_scale=18,
                linewidth=2.4,
                color=color,
                alpha=0.95,
            )
        )

        print(
            f"part {part.part_id}: FOUND | "
            f"line1_x=[{result.match_x0},{result.match_x1}] | "
            f"run_len={result.run_length} | "
            f"mean_sim={result.mean_similarity:.4f} | "
            f"score={result.sw_score:.4f}"
        )

    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(y2_bottom + 28, 0)
    ax.axis("off")

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

    img1, tensor1 = preprocess_line_image(line1_path, target_height=args.height)
    img2, _tensor2 = preprocess_line_image(line2_path, target_height=args.height)
    emb_line1 = image_embeddings(model, tensor1, args.device)

    line2_original = Image.open(line2_path).convert("RGB")
    line2_original_width, _ = line2_original.size
    display_scale = img2.size[0] / float(line2_original_width)

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
            display_scale,
            tmp_dir,
        )

        for part in parts:
            result = search_one_part(
                model,
                emb_line1,
                part,
                line1_num_windows=int(emb_line1.shape[0]),
                line1_display_width=int(img1.size[0]),
                args=args,
            )
            results.append(result)

    sample_label = f" sample {args.index}" if args.index is not None else ""
    title = (
        f"Line2 fixed-width parts searched inside line1{sample_label} | "
        f"part_width={args.part_width}, thr={args.threshold}, gap={args.gap}"
    )
    draw_results(img1, img2, results, args.output, title)


if __name__ == "__main__":
    main()
