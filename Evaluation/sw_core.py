"""Smith-Waterman scoring and visualization helpers for image-window alignment."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

from Evaluation._eval_utils import patch_range_to_pixels
from Evaluation.window_alignment import alignment_score_matrix


@dataclass(frozen=True)
class ImagePair:
    index: int
    image1: Path
    image2: Path
    pair_id: str = ""
    label_type: str = ""
    text_score: float = 0.0
    manifest_position: int = -1
    split: str = ""


def smith_waterman(
    similarity: np.ndarray,
    threshold: float = 0.45,
    gap_penalty: float = -0.30,
    return_traceback: bool = False,
    match_scores: np.ndarray | None = None,
):
    """Run local SW from the maximum accumulated DP cell.

    The returned traceback is ordered exactly as backtracking occurs: the first
    coordinate is the maximum DP boundary and the last is the zero-score local
    boundary. ``path`` remains in forward reading order for reporting.
    """
    raw = np.asarray(similarity, dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError(f"Expected a 2-D similarity matrix, got {raw.shape}")
    diagonal_scores = (
        raw - float(threshold)
        if match_scores is None
        else np.asarray(match_scores, dtype=np.float32)
    )
    if diagonal_scores.shape != raw.shape:
        raise ValueError(
            "match_scores must have the same shape as similarity: "
            f"{diagonal_scores.shape} != {raw.shape}"
        )

    n, m = raw.shape
    score = np.zeros((n + 1, m + 1), dtype=np.float32)
    trace = np.zeros((n + 1, m + 1), dtype=np.uint8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + float(diagonal_scores[i - 1, j - 1])
            up = score[i - 1, j] + float(gap_penalty)
            left = score[i, j - 1] + float(gap_penalty)
            values = (0.0, diag, up, left)
            best = int(np.argmax(values))
            score[i, j] = values[best]
            trace[i, j] = best

    max_row, max_col = map(int, np.unravel_index(np.argmax(score), score.shape))
    best_score = float(score[max_row, max_col])
    if best_score <= 0.0:
        empty = np.empty((0, 2), dtype=np.float32)
        return ([], best_score, score, empty) if return_traceback else ([], best_score, score)

    i, j = max_row, max_col
    matched_backwards: list[tuple[int, int]] = []
    traceback_max_to_zero = [(float(j), float(i))]
    while i > 0 and j > 0 and score[i, j] > 0.0:
        code = int(trace[i, j])
        if code == 1:
            matched_backwards.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif code == 2:
            i -= 1
        elif code == 3:
            j -= 1
        else:
            break
        traceback_max_to_zero.append((float(j), float(i)))

    path = list(reversed(matched_backwards))
    traceback = np.asarray(traceback_max_to_zero, dtype=np.float32)
    return (path, best_score, score, traceback) if return_traceback else (path, best_score, score)


def resolve_score_mode(score_mode: str, dataset_type: str) -> str:
    value = str(score_mode).strip().lower().replace("_", "-")
    if value == "auto":
        return "mutual-z" if str(dataset_type).lower() == "real" else "raw"
    if value not in {"raw", "centered", "mutual-z"}:
        raise ValueError("score mode must be auto, raw, centered, or mutual-z")
    return value


def build_match_scores(
    raw_similarity: np.ndarray,
    score_mode: str,
    score_clip: float,
    threshold: float,
) -> np.ndarray:
    normalized = alignment_score_matrix(
        raw_similarity,
        mode=str(score_mode).replace("-", "_"),
        clip=float(score_clip),
    )
    return normalized.detach().cpu().numpy().astype(np.float32) - float(threshold)


def format_heatmap_value(value: float, decimals: int) -> str:
    if not np.isfinite(value):
        return ""
    return f"{float(value):.{max(0, int(decimals))}f}"


def annotate_heatmap_values(ax, matrix, image, decimals=2, fontsize=5.0):
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(matrix[row, col])
            if not np.isfinite(value):
                continue
            rgba = image.cmap(image.norm(value))
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(
                col,
                row,
                format_heatmap_value(value, decimals),
                ha="center",
                va="center",
                color="black" if luminance > 0.58 else "white",
                fontsize=float(fontsize),
                zorder=3,
                clip_on=True,
            )


def select_heatmap_matrix(
    similarity: np.ndarray,
    threshold: float,
    dp_score: np.ndarray,
    source: str,
    match_scores: np.ndarray | None = None,
    score_mode: str = "raw",
):
    source = str(source).lower()
    if source == "cosine":
        return np.asarray(similarity, dtype=np.float32), "raw cosine similarity"
    if source == "match-score":
        matrix = (
            np.asarray(similarity, dtype=np.float32) - float(threshold)
            if match_scores is None
            else np.asarray(match_scores, dtype=np.float32)
        )
        return matrix, f"SW diagonal reward ({score_mode} score - threshold)"
    if source == "dp-score":
        return np.asarray(dp_score[1:, 1:], dtype=np.float32), "accumulated local SW DP score"
    raise ValueError("heatmap source must be cosine, match-score, or dp-score")


def contiguous_runs(indices) -> list[tuple[int, int]]:
    """Return half-open runs so intervening gap windows are not painted."""
    values = sorted({int(index) for index in indices})
    if not values:
        return []
    runs = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            runs.append((start, previous + 1))
            start = value
        previous = value
    runs.append((start, previous + 1))
    return runs


def _draw_matched_runs(ax, array, indices, n_windows, use_flip):
    for start, end in contiguous_runs(indices):
        x0, x1 = patch_range_to_pixels(
            start, end, n_windows, array.shape[1], use_flip
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


def _draw_traceback_arrows(ax, traceback: np.ndarray):
    if len(traceback) < 2:
        return
    xs = traceback[:, 0] - 0.5
    ys = traceback[:, 1] - 0.5
    stride = max(1, (len(traceback) - 1) // 12)
    for index in range(0, len(traceback) - 1, stride):
        next_index = min(index + 1, len(traceback) - 1)
        ax.annotate(
            "",
            xy=(xs[next_index], ys[next_index]),
            xytext=(xs[index], ys[index]),
            arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
            zorder=8,
        )


def save_visualization(
    arr1,
    arr2,
    features1,
    features2,
    path,
    traceback,
    heatmap_matrix,
    heatmap_label,
    score,
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

    if path:
        _draw_matched_runs(
            axes[0], arr1, [row for row, _ in path], len(features1.contextual), use_flip
        )
        _draw_matched_runs(
            axes[1], arr2, [col for _, col in path], len(features2.contextual), use_flip
        )

    metadata = f" | pair_id={pair.pair_id} | label={pair.label_type}" if pair.pair_id else ""
    input_label = "binarized real input" if binarized else "synthetic input"
    fig.suptitle(
        f"Smith-Waterman local image alignment | score={score:.4f} | "
        f"score_mode={score_mode} | {input_label}{metadata}",
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
                fontsize=annotation_fontsize
            )

        if traceback.size:
            xs, ys = traceback[:, 0] - 0.5, traceback[:, 1] - 0.5
            ax.plot(
                xs, ys, color="black", linewidth=1.7, marker=".",
                markersize=2.8, label="traceback: DP maximum → zero", zorder=7
            )
            _draw_traceback_arrows(ax, traceback)
            max_col, max_row = int(traceback[0, 0]), int(traceback[0, 1])
            if max_row > 0 and max_col > 0:
                ax.scatter(
                    [max_col - 1], [max_row - 1], marker="*", s=150,
                    facecolors="lime", edgecolors="black", linewidths=1.0,
                    label="maximum accumulated DP score", zorder=10
                )
            ax.scatter(
                [traceback[-1, 0] - 0.5], [traceback[-1, 1] - 0.5],
                marker="s", s=45, facecolors="white", edgecolors="black",
                linewidths=1.0, label="zero-score local boundary", zorder=10
            )

        if path:
            ax.scatter(
                [col for _, col in path], [row for row, _ in path], s=30,
                facecolors="none", edgecolors="yellow", linewidths=1.2,
                label="matched diagonal cells", zorder=9
            )

        ax.set_xlim(-0.5, n2 - 0.5)
        ax.set_ylim(n1 - 0.5, -0.5)
        ax.set_title(f"{heatmap_label}: values + directed local traceback", fontsize=10)
        logical = "logical windows (0 = rightmost)" if use_flip else "logical windows"
        ax.set_xlabel(f"line B {logical}")
        ax.set_ylabel(f"line A {logical}")
        ax.legend(loc="upper left", fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015, label=heatmap_label)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# Backward-compatible private names used by existing tests and callers.
_resolve_score_mode = resolve_score_mode
_build_match_scores = build_match_scores
_format_heatmap_value = format_heatmap_value
_annotate_heatmap_values = annotate_heatmap_values
_heatmap_matrix = select_heatmap_matrix
_contiguous_runs = contiguous_runs
