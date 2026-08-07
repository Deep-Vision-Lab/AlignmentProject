"""Needleman-Wunsch global scoring, local-region extraction, and visualization."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from Evaluation._eval_utils import NWResult, NWStep, patch_range_to_pixels
from Evaluation.sw_core import ImagePair, dense_alignment_region


@dataclass(frozen=True)
class NWTrace:
    """Global traceback plus the best positive-scoring region on that path."""

    global_path: list[tuple[int, int]]
    global_traceback: np.ndarray
    region_path: list[tuple[int, int]]
    region_traceback: np.ndarray
    region_score: float
    gap_in_line1: int
    gap_in_line2: int


def _forward_boundaries(steps: list[NWStep]) -> list[tuple[float, float]]:
    """Return DP boundaries as (column, row), from origin to terminal."""
    row = col = 0
    boundaries = [(0.0, 0.0)]
    for step in steps:
        if step.index1 is not None:
            row += 1
        if step.index2 is not None:
            col += 1
        boundaries.append((float(col), float(row)))
    return boundaries


def _best_positive_segment(
    steps: list[NWStep],
    match_scores: np.ndarray,
    gap_penalty: float,
) -> tuple[int, int, float]:
    """Maximum-sum contiguous segment of the fixed global NW traceback.

    Needleman-Wunsch consumes the complete lines. Synthetic masks, however,
    annotate the shared local phrase. The best positive segment preserves the
    global path while isolating the part supported by positive match evidence.
    """
    if not steps:
        return 0, 0, 0.0

    rewards = []
    for step in steps:
        if step.index1 is not None and step.index2 is not None:
            rewards.append(float(match_scores[int(step.index1), int(step.index2)]))
        else:
            rewards.append(float(gap_penalty))

    best_score = 0.0
    best_start = best_end = 0
    current_score = 0.0
    current_start = 0
    for index, reward in enumerate(rewards):
        if current_score <= 0.0:
            current_score = reward
            current_start = index
        else:
            current_score += reward
        if current_score > best_score:
            best_score = current_score
            best_start = current_start
            best_end = index + 1
    return best_start, best_end, float(best_score)


def trace_alignment(
    result: NWResult,
    match_scores: np.ndarray,
    gap_penalty: float,
) -> NWTrace:
    """Convert NW steps into plotting paths and a mask-comparable local region."""
    steps = list(result.steps)
    boundaries = _forward_boundaries(steps)
    global_traceback = np.asarray(list(reversed(boundaries)), dtype=np.float32)
    global_path = [
        (int(step.index1), int(step.index2))
        for step in steps
        if step.index1 is not None and step.index2 is not None
    ]

    start, end, region_score = _best_positive_segment(
        steps, np.asarray(match_scores, dtype=np.float32), gap_penalty
    )
    region_steps = steps[start:end]
    region_path = [
        (int(step.index1), int(step.index2))
        for step in region_steps
        if step.index1 is not None and step.index2 is not None
    ]
    region_boundaries = boundaries[start : end + 1]
    region_traceback = np.asarray(
        list(reversed(region_boundaries)), dtype=np.float32
    )
    if not region_path:
        region_traceback = np.empty((0, 2), dtype=np.float32)

    return NWTrace(
        global_path=global_path,
        global_traceback=global_traceback,
        region_path=region_path,
        region_traceback=region_traceback,
        region_score=region_score,
        gap_in_line1=sum(step.operation == "gap_in_line1" for step in steps),
        gap_in_line2=sum(step.operation == "gap_in_line2" for step in steps),
    )


def select_heatmap_matrix(
    raw_similarity: np.ndarray,
    match_scores: np.ndarray,
    score_matrix: np.ndarray,
    source: str,
    score_mode: str,
) -> tuple[np.ndarray, str]:
    value = str(source).strip().lower()
    if value == "cosine":
        return np.asarray(raw_similarity, dtype=np.float32), "raw cosine similarity"
    if value == "match-score":
        return (
            np.asarray(match_scores, dtype=np.float32),
            f"NW diagonal reward ({score_mode} score - threshold)",
        )
    if value == "dp-score":
        return (
            np.asarray(score_matrix[1:, 1:], dtype=np.float32),
            "accumulated global Needleman-Wunsch DP score",
        )
    raise ValueError("heatmap source must be cosine, match-score, or dp-score")


def format_heatmap_value(value: float, decimals: int) -> str:
    if not np.isfinite(value):
        return ""
    return f"{float(value):.{max(0, int(decimals))}f}"


def annotate_heatmap_values(ax, matrix, image, decimals=2, fontsize=5.0):
    """Use the contrast-safe, high-z-order labels from the revised SW view."""
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            if not np.isfinite(value):
                continue
            rgba = image.cmap(image.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            dark_cell = luminance <= 0.58
            ax.text(
                col,
                row,
                format_heatmap_value(value, decimals),
                ha="center",
                va="center",
                color="white" if dark_cell else "black",
                fontsize=float(fontsize),
                fontweight="medium",
                zorder=20,
                clip_on=True,
                bbox={
                    "boxstyle": "round,pad=0.08",
                    "facecolor": "black" if dark_cell else "white",
                    "edgecolor": "none",
                    "alpha": 0.58,
                },
            )


def _draw_traceback_arrows(ax, traceback):
    """Draw sparse arrows without obscuring the heatmap values."""
    if len(traceback) < 2:
        return
    xs = traceback[:, 0] - 0.5
    ys = traceback[:, 1] - 0.5
    stride = max(1, (len(traceback) - 1) // 8)
    for index in range(0, len(traceback) - 1, stride):
        next_index = min(index + 1, len(traceback) - 1)
        ax.annotate(
            "",
            xy=(xs[next_index], ys[next_index]),
            xytext=(xs[index], ys[index]),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "black",
                "lw": 0.8,
                "alpha": 0.45,
                "mutation_scale": 7,
                "shrinkA": 4,
                "shrinkB": 4,
            },
            zorder=4,
        )


def _draw_dense_span(ax, array, start, end, n_windows, use_flip):
    if start < 0 or end < start:
        return
    x0, x1 = patch_range_to_pixels(
        start, end + 1, n_windows, array.shape[1], use_flip
    )
    ax.add_patch(
        Rectangle(
            (x0, 1),
            max(2.0, x1 - x0),
            max(2.0, array.shape[0] - 2),
            facecolor="red",
            edgecolor="red",
            alpha=0.28,
            linewidth=1.5,
        )
    )


def save_visualization(
    arr1,
    arr2,
    features1,
    features2,
    trace: NWTrace,
    heatmap_matrix,
    heatmap_label,
    score,
    normalized_score,
    output,
    use_flip,
    binarized,
    pair: ImagePair,
    score_mode: str,
    show_heatmap=True,
    annotate_values=True,
    value_decimals=2,
    annotation_fontsize=5.0,
):
    """Render global NW traceback while highlighting its best local region."""
    n1, n2 = heatmap_matrix.shape
    if show_heatmap:
        heatmap_height = max(8.0, min(24.0, 0.30 * n1))
        figure_width = max(18.0, min(30.0, 0.34 * n2))
        rows, ratios = 3, [2.2, 2.2, heatmap_height]
        figure_height = 5.0 + heatmap_height
    else:
        figure_width, figure_height = 18.0, 5.5
        rows, ratios = 2, [2.2, 2.2]

    fig = plt.figure(figsize=(figure_width, figure_height))
    grid = fig.add_gridspec(rows, 1, height_ratios=ratios, hspace=0.16)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    axes[0].imshow(arr1, aspect="auto")
    axes[1].imshow(arr2, aspect="auto")
    suffix = " (binarized)" if binarized else ""
    axes[0].set_ylabel(f"line A{suffix}", rotation=0, labelpad=50, va="center")
    axes[1].set_ylabel(f"line B{suffix}", rotation=0, labelpad=50, va="center")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    region = dense_alignment_region(trace.region_path, trace.region_traceback)
    if not region.empty:
        _draw_dense_span(
            axes[0], arr1, region.line1_start, region.line1_end,
            len(features1.contextual), use_flip,
        )
        _draw_dense_span(
            axes[1], arr2, region.line2_start, region.line2_end,
            len(features2.contextual), use_flip,
        )
        axes[0].set_title(
            f"best positive NW region: windows {region.line1_start}–{region.line1_end}",
            fontsize=9,
        )
        axes[1].set_title(
            f"best positive NW region: windows {region.line2_start}–{region.line2_end}",
            fontsize=9,
        )

    metadata = f" | pair_id={pair.pair_id} | label={pair.label_type}" if pair.pair_id else ""
    input_label = "binarized real input" if binarized else "synthetic input"
    total_gaps = trace.gap_in_line1 + trace.gap_in_line2
    fig.suptitle(
        f"Needleman-Wunsch global image alignment | score={score:.4f} | "
        f"normalized={normalized_score:.4f} | score_mode={score_mode} | "
        f"gaps={total_gaps} | red=best positive path region | {input_label}{metadata}",
        fontsize=11,
        fontweight="bold",
    )

    if show_heatmap:
        ax = fig.add_subplot(grid[2])
        finite = heatmap_matrix[np.isfinite(heatmap_matrix)]
        if finite.size and float(finite.min()) >= 0.0:
            upper = max(1e-6, float(np.percentile(finite, 98)))
            image = ax.imshow(
                heatmap_matrix, aspect="equal", origin="upper", vmin=0.0,
                vmax=upper, cmap="viridis", interpolation="nearest"
            )
        else:
            absolute = np.abs(finite)
            limit = max(0.5, float(np.percentile(absolute, 98)) if absolute.size else 1.0)
            image = ax.imshow(
                heatmap_matrix, aspect="equal", origin="upper", vmin=-limit,
                vmax=limit, cmap="coolwarm", interpolation="nearest"
            )

        if annotate_values:
            annotate_heatmap_values(
                ax, heatmap_matrix, image, decimals=value_decimals,
                fontsize=annotation_fontsize,
            )

        traceback = trace.global_traceback
        if traceback.size:
            xs, ys = traceback[:, 0] - 0.5, traceback[:, 1] - 0.5
            ax.plot(
                xs, ys, color="black", linewidth=1.7, marker=".",
                markersize=2.8, label="global traceback: terminal → origin", zorder=7,
            )
            _draw_traceback_arrows(ax, traceback)
            ax.scatter(
                [n2 - 1], [n1 - 1], marker="*", s=150,
                facecolors="lime", edgecolors="black", linewidths=1.0,
                label="terminal global DP score", zorder=10,
            )
            ax.scatter(
                [-0.5], [-0.5], marker="s", s=45, facecolors="white",
                edgecolors="black", linewidths=1.0, label="global origin", zorder=10,
            )

        if trace.global_path:
            ax.scatter(
                [col for _, col in trace.global_path],
                [row for row, _ in trace.global_path],
                s=30, facecolors="none", edgecolors="yellow", linewidths=1.2,
                label="global diagonal correspondences", zorder=9,
            )
        if trace.region_path:
            ax.scatter(
                [col for _, col in trace.region_path],
                [row for row, _ in trace.region_path],
                s=18, facecolors="cyan", edgecolors="black", linewidths=0.5,
                label="best positive region", zorder=11,
            )

        ax.set_xlim(-0.5, n2 - 0.5)
        ax.set_ylim(n1 - 0.5, -0.5)
        ax.set_title(
            f"{heatmap_label}: full global path retained; red masks use its best positive segment",
            fontsize=10,
        )
        logical = "logical windows (0 = rightmost)" if use_flip else "logical windows"
        ax.set_xlabel(f"line B {logical}")
        ax.set_ylabel(f"line A {logical}")
        ax.legend(loc="upper left", fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015, label=heatmap_label)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
