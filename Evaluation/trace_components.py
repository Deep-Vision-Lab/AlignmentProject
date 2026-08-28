"""Shared traceback/component logic for ViT SW and NW evaluation.

The dynamic-programming algorithms are intentionally left untouched:

* Smith-Waterman traces from the maximum accumulated local-DP cell back to the
  zero-score boundary.
* Needleman-Wunsch traces from the terminal DP boundary (N, M) back to (0, 0).

This module only interprets the resulting trace.  Supported diagonal matches are
split into separate aligned components.  Up to ``TRACE_COMPONENT_MAX_BRIDGE_STEPS``
unsupported traceback steps (default: 2) may be bridged inside one component;
a longer unsupported valley opens a real hole in the predicted line mask.

Arabic encoders reverse the logical window sequence.  Heatmaps and line masks
therefore have to be mapped back to physical left-to-right image coordinates
before they are rendered.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


class ComponentPath(list):
    """Flat supported match path plus explicit disconnected component runs."""

    def __init__(
        self,
        values=(),
        *,
        runs=(),
        full_match_steps=0,
        full_traceback_steps=0,
        bridged_trace_steps=0,
        bridge_limit=2,
        support_floor=0.0,
    ):
        super().__init__(values)
        self.runs = tuple(tuple(run) for run in runs if run)
        self.full_match_steps = int(full_match_steps)
        self.full_traceback_steps = int(full_traceback_steps)
        self.bridged_trace_steps = int(bridged_trace_steps)
        self.bridge_limit = int(bridge_limit)
        self.support_floor = float(support_floor)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def component_settings() -> dict:
    return {
        "support_floor": _env_float("TRACE_COMPONENT_SUPPORT_FLOOR", 0.0),
        "max_bridge_steps": max(0, _env_int("TRACE_COMPONENT_MAX_BRIDGE_STEPS", 2)),
        "max_window_gap": max(0, _env_int("TRACE_COMPONENT_MAX_WINDOW_GAP", 2)),
        "min_matches": max(1, _env_int("TRACE_COMPONENT_MIN_MATCHES", 3)),
    }


def _group_records(records: list[dict], *, full_match_steps: int, full_traceback_steps: int) -> ComponentPath:
    settings = component_settings()
    floor = float(settings["support_floor"])
    supported = [record for record in records if float(record["score"]) > floor]
    if not supported:
        return ComponentPath(
            (),
            runs=(),
            full_match_steps=full_match_steps,
            full_traceback_steps=full_traceback_steps,
            bridge_limit=settings["max_bridge_steps"],
            support_floor=floor,
        )

    groups: list[list[dict]] = []
    group = [supported[0]]
    bridged = 0
    for record in supported[1:]:
        previous = group[-1]
        path_gap = int(record["position"]) - int(previous["position"]) - 1
        row_gap = int(record["row"]) - int(previous["row"]) - 1
        col_gap = int(record["col"]) - int(previous["col"]) - 1
        can_bridge = (
            path_gap <= int(settings["max_bridge_steps"])
            and row_gap <= int(settings["max_window_gap"])
            and col_gap <= int(settings["max_window_gap"])
        )
        if can_bridge:
            bridged += max(0, path_gap)
            group.append(record)
        else:
            groups.append(group)
            group = [record]
    groups.append(group)

    groups = [group for group in groups if len(group) >= int(settings["min_matches"])]
    runs = [
        [(int(record["row"]), int(record["col"])) for record in group]
        for group in groups
    ]
    flat = [pair for run in runs for pair in run]
    return ComponentPath(
        flat,
        runs=runs,
        full_match_steps=full_match_steps,
        full_traceback_steps=full_traceback_steps,
        bridged_trace_steps=bridged,
        bridge_limit=settings["max_bridge_steps"],
        support_floor=floor,
    )


def sw_component_path(full_path, traceback: np.ndarray, match_scores: np.ndarray) -> ComponentPath:
    """Split the true SW maximum->zero traceback into supported components."""
    matrix = np.asarray(match_scores, dtype=np.float32)
    trace = np.asarray(traceback, dtype=np.float32)
    full_pairs = {(int(row), int(col)) for row, col in full_path}
    records: list[dict] = []
    if len(trace) >= 2:
        # SW returns DP boundaries in maximum->zero order.  Reverse them only for
        # component grouping so row/column indices increase monotonically.  The
        # original maximum->zero traceback is retained unchanged for rendering.
        forward = trace[::-1]
        for position in range(len(forward) - 1):
            col0, row0 = forward[position]
            col1, row1 = forward[position + 1]
            drow = int(round(float(row1 - row0)))
            dcol = int(round(float(col1 - col0)))
            if drow == 1 and dcol == 1:
                row = int(round(float(row1))) - 1
                col = int(round(float(col1))) - 1
                if (
                    0 <= row < matrix.shape[0]
                    and 0 <= col < matrix.shape[1]
                    and (row, col) in full_pairs
                ):
                    records.append(
                        {
                            "position": int(position),
                            "row": row,
                            "col": col,
                            "score": float(matrix[row, col]),
                        }
                    )
    return _group_records(
        records,
        full_match_steps=len(full_path),
        full_traceback_steps=max(0, len(trace) - 1),
    )


def nw_traceback_boundaries(result) -> np.ndarray:
    """Return the true NW DP-boundary traceback in terminal->origin order."""
    row = col = 0
    boundaries = [(0.0, 0.0)]  # stored as (column, row)
    for step in result.steps:
        if step.index1 is not None:
            row += 1
        if step.index2 is not None:
            col += 1
        boundaries.append((float(col), float(row)))

    n = int(result.score_matrix.shape[0] - 1)
    m = int(result.score_matrix.shape[1] - 1)
    if (row, col) != (n, m):
        raise RuntimeError(
            f"NW traceback did not terminate at (N,M): got {(row, col)} expected {(n, m)}"
        )
    return np.asarray(list(reversed(boundaries)), dtype=np.float32)


def nw_component_path(result, match_scores: np.ndarray) -> ComponentPath:
    """Split supported diagonals on the full origin->terminal NW path."""
    matrix = np.asarray(match_scores, dtype=np.float32)
    records = []
    full_matches = 0
    for position, step in enumerate(result.steps):
        if step.index1 is None or step.index2 is None:
            continue
        row, col = int(step.index1), int(step.index2)
        full_matches += 1
        records.append(
            {
                "position": int(position),
                "row": row,
                "col": col,
                "score": float(matrix[row, col]),
            }
        )
    return _group_records(
        records,
        full_match_steps=full_matches,
        full_traceback_steps=len(result.steps),
    )


def path_runs(path) -> list[list[tuple[int, int]]]:
    explicit = getattr(path, "runs", None)
    if explicit is not None:
        return [list(run) for run in explicit if run]
    return [list(path)] if path else []


def _run_ranges(path, axis: int) -> list[list[int]]:
    ranges = []
    for run in path_runs(path):
        values = [int(pair[axis]) for pair in run]
        if values:
            ranges.append([min(values), max(values)])
    return ranges


def component_metrics(path, similarity_shape) -> dict:
    n1, n2 = map(int, similarity_shape)
    runs = path_runs(path)
    rows = {int(row) for row, _ in path}
    cols = {int(col) for _, col in path}
    ranges1 = _run_ranges(path, 0)
    ranges2 = _run_ranges(path, 1)
    span1 = sum(end - start + 1 for start, end in ranges1)
    span2 = sum(end - start + 1 for start, end in ranges2)
    full_matches = int(getattr(path, "full_match_steps", len(path)))
    traceback_steps = int(getattr(path, "full_traceback_steps", len(path)))
    return {
        "path_steps": int(len(path)),
        "traceback_steps": traceback_steps,
        "warp_steps": max(0, traceback_steps - full_matches),
        "component_count": int(len(runs)),
        "full_match_steps": full_matches,
        "bridged_trace_steps": int(getattr(path, "bridged_trace_steps", 0)),
        "component_bridge_limit": int(getattr(path, "bridge_limit", 0)),
        "component_support_floor": float(getattr(path, "support_floor", 0.0)),
        "line1_path_windows": len(rows),
        "line2_path_windows": len(cols),
        "line1_span_windows": int(span1),
        "line2_span_windows": int(span2),
        "line1_path_fraction": len(rows) / max(1, n1),
        "line2_path_fraction": len(cols) / max(1, n2),
        "line1_matched_fraction": min(1.0, span1 / max(1, n1)),
        "line2_matched_fraction": min(1.0, span2 / max(1, n2)),
        "line1_path_start": min(rows) if rows else -1,
        "line1_path_end": max(rows) if rows else -1,
        "line2_path_start": min(cols) if cols else -1,
        "line2_path_end": max(cols) if cols else -1,
        "line1_component_ranges": json.dumps(ranges1, separators=(",", ":")),
        "line2_component_ranges": json.dumps(ranges2, separators=(",", ":")),
    }


def window_range_to_pixels(
    start: int,
    end: int,
    n_windows: int,
    image_width: int,
    flipped: bool,
    *,
    window_size: int | None = None,
    stride: int | None = None,
) -> tuple[float, float]:
    """Map a half-open logical heatmap-window range to physical image pixels."""
    n_windows = int(n_windows)
    image_width = int(image_width)
    if n_windows <= 0:
        return 0.0, float(image_width)
    start = max(0, min(int(start), n_windows - 1))
    end = max(start + 1, min(int(end), n_windows))

    if flipped:
        physical_first = n_windows - end
        physical_last = n_windows - start - 1
    else:
        physical_first = start
        physical_last = end - 1

    try:
        window_size = int(window_size) if window_size is not None else 0
        stride = int(stride) if stride is not None else 0
    except (TypeError, ValueError):
        window_size = stride = 0

    if window_size > 0 and stride > 0:
        first_center = physical_first * stride + 0.5 * window_size
        last_center = physical_last * stride + 0.5 * window_size
        x0 = first_center - 0.5 * stride
        x1 = last_center + 0.5 * stride
        if physical_first == 0:
            x0 = 0.0
        if physical_last == n_windows - 1:
            x1 = float(image_width)
    else:
        # Heatmap-cell fallback: every logical feature owns one equal physical cell.
        x0 = physical_first / n_windows * image_width
        x1 = (physical_last + 1) / n_windows * image_width

    x0 = max(0.0, min(float(image_width), float(x0)))
    x1 = max(0.0, min(float(image_width), float(x1)))
    return min(x0, x1), max(x0, x1)


def component_intervals_px(
    path,
    axis: int,
    n_windows: int,
    image_width: int,
    flipped: bool,
    *,
    window_size: int | None = None,
    stride: int | None = None,
) -> list[list[float]]:
    intervals = []
    for run in path_runs(path):
        values = [int(pair[axis]) for pair in run]
        if not values:
            continue
        left, right = window_range_to_pixels(
            min(values),
            max(values) + 1,
            n_windows,
            image_width,
            flipped,
            window_size=window_size,
            stride=stride,
        )
        intervals.append([float(left), float(right)])
    return intervals


def physical_matrix(matrix: np.ndarray, flipped: bool) -> np.ndarray:
    value = np.asarray(matrix)
    return value[::-1, ::-1] if flipped else value


def physical_pair(row: int, col: int, n1: int, n2: int, flipped: bool):
    if not flipped:
        return int(row), int(col)
    return int(n1 - 1 - row), int(n2 - 1 - col)


def physical_traceback(traceback: np.ndarray, n1: int, n2: int, flipped: bool) -> np.ndarray:
    value = np.asarray(traceback, dtype=np.float32).copy()
    if not value.size or not flipped:
        return value
    # DP boundaries are stored as (column, row), including 0 and N/M.
    value[:, 0] = float(n2) - value[:, 0]
    value[:, 1] = float(n1) - value[:, 1]
    return value


def _annotate_values(ax, matrix, image, decimals: int, fontsize: float) -> None:
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
                f"{value:.{max(0, int(decimals))}f}",
                ha="center",
                va="center",
                color="black" if luminance > 0.58 else "white",
                fontsize=float(fontsize),
                zorder=15,
                clip_on=True,
            )


def save_alignment_visualization(
    *,
    arr1,
    arr2,
    features1,
    features2,
    full_path,
    component_path,
    traceback,
    heatmap_matrix,
    heatmap_label,
    score: float,
    output,
    use_flip: bool,
    pair,
    score_mode: str,
    algorithm: str,
    traceback_label: str,
    traceback_start_label: str,
    traceback_end_label: str,
    normalized_score: float | None = None,
    binarized: bool = True,
    annotate_values: bool = False,
    value_decimals: int = 2,
    annotation_fontsize: float = 5.0,
    window_size: int | None = None,
    stride: int | None = None,
) -> None:
    """Render physical heatmap, true traceback, and disconnected line masks."""
    matrix = np.asarray(heatmap_matrix, dtype=np.float32)
    n1, n2 = matrix.shape
    heatmap_height = max(8.0, min(24.0, 0.30 * n1))
    figure_width = max(18.0, min(30.0, 0.34 * n2))
    fig = plt.figure(figsize=(figure_width, 5.0 + heatmap_height))
    grid = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.2, heatmap_height], hspace=0.16)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    axes[0].imshow(arr1, aspect="auto")
    axes[1].imshow(arr2, aspect="auto")
    suffix = " (binarized)" if binarized else ""
    axes[0].set_ylabel(f"line A{suffix}", rotation=0, labelpad=50, va="center")
    axes[1].set_ylabel(f"line B{suffix}", rotation=0, labelpad=50, va="center")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    intervals1 = component_intervals_px(
        component_path, 0, len(features1.contextual), arr1.shape[1], use_flip,
        window_size=window_size, stride=stride,
    )
    intervals2 = component_intervals_px(
        component_path, 1, len(features2.contextual), arr2.shape[1], use_flip,
        window_size=window_size, stride=stride,
    )
    for axis, array, intervals in ((axes[0], arr1, intervals1), (axes[1], arr2, intervals2)):
        for left, right in intervals:
            axis.add_patch(
                Rectangle(
                    (left, 1),
                    max(2.0, right - left),
                    max(2.0, array.shape[0] - 2),
                    facecolor="red",
                    edgecolor="red",
                    alpha=0.28,
                    linewidth=1.5,
                )
            )

    ranges1 = _run_ranges(component_path, 0)
    ranges2 = _run_ranges(component_path, 1)
    if ranges1:
        axes[0].set_title(f"supported components ({len(ranges1)}), logical windows: {ranges1}", fontsize=9)
        axes[1].set_title(f"supported components ({len(ranges2)}), logical windows: {ranges2}", fontsize=9)
    else:
        axes[0].set_title("no supported component", fontsize=9)
        axes[1].set_title("no supported component", fontsize=9)

    ax = fig.add_subplot(grid[2])
    shown = physical_matrix(matrix, bool(use_flip))
    finite = shown[np.isfinite(shown)]
    if finite.size and float(finite.min()) >= 0.0:
        upper = max(1e-6, float(np.percentile(finite, 98)))
        image = ax.imshow(
            shown, aspect="equal", origin="upper", vmin=0.0, vmax=upper,
            cmap="viridis", interpolation="nearest"
        )
    else:
        limit = max(
            0.5,
            float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0,
        )
        image = ax.imshow(
            shown, aspect="equal", origin="upper", vmin=-limit, vmax=limit,
            cmap="coolwarm", interpolation="nearest"
        )
    if annotate_values:
        _annotate_values(ax, shown, image, value_decimals, annotation_fontsize)

    shown_trace = physical_traceback(traceback, n1, n2, bool(use_flip))
    if len(shown_trace) >= 2:
        xs = shown_trace[:, 0] - 0.5
        ys = shown_trace[:, 1] - 0.5
        ax.plot(
            xs, ys, color="black", linewidth=1.7, marker=".", markersize=2.8,
            label=traceback_label, zorder=7,
        )
        arrow_stride = max(1, (len(shown_trace) - 1) // 10)
        for index in range(0, len(shown_trace) - 1, arrow_stride):
            nxt = min(index + 1, len(shown_trace) - 1)
            ax.annotate(
                "", xy=(xs[nxt], ys[nxt]), xytext=(xs[index], ys[index]),
                arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.0},
                zorder=8,
            )
        ax.scatter(
            [xs[0]], [ys[0]], marker="*", s=150, facecolors="lime",
            edgecolors="black", linewidths=1.0, label=traceback_start_label, zorder=10,
        )
        ax.scatter(
            [xs[-1]], [ys[-1]], marker="s", s=50, facecolors="white",
            edgecolors="black", linewidths=1.0, label=traceback_end_label, zorder=10,
        )

    if full_path:
        displayed = [physical_pair(row, col, n1, n2, bool(use_flip)) for row, col in full_path]
        ax.scatter(
            [col for row, col in displayed], [row for row, col in displayed],
            s=29, facecolors="none", edgecolors="yellow", linewidths=1.1,
            label="all diagonal traceback matches", zorder=9,
        )
    if component_path:
        displayed = [
            physical_pair(row, col, n1, n2, bool(use_flip))
            for row, col in component_path
        ]
        ax.scatter(
            [col for row, col in displayed], [row for row, col in displayed],
            s=20, facecolors="cyan", edgecolors="black", linewidths=0.45,
            label="accepted component matches", zorder=11,
        )

    ax.set_xlim(-0.5, n2 - 0.5)
    ax.set_ylim(n1 - 0.5, -0.5)
    ax.set_xlabel("line B physical windows: left → right")
    ax.set_ylabel("line A physical windows: left → right, top → bottom")
    ax.set_title(
        f"{heatmap_label} | physical coordinates | bridge≤{getattr(component_path, 'bridge_limit', 0)}"
    )
    ax.legend(loc="upper left", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015, label=heatmap_label)

    metadata = f" | pair_id={pair.pair_id} | label={pair.label_type}" if getattr(pair, "pair_id", "") else ""
    normalized = "" if normalized_score is None else f" | normalized={normalized_score:.4f}"
    fig.suptitle(
        f"{algorithm} image alignment | score={score:.4f}{normalized} | "
        f"score_mode={score_mode} | red=component masks; holes are preserved{metadata}",
        fontsize=11,
        fontweight="bold",
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_numeric_evidence(
    output,
    *,
    algorithm: str,
    raw_similarity: np.ndarray,
    match_scores: np.ndarray,
    component_path,
    full_path,
    traceback: np.ndarray,
    intervals1=None,
    intervals2=None,
) -> dict:
    """Persist text-readable matrices/evidence so pushed results are inspectable."""
    output = Path(output)
    result = {"matrix_csv_cosine": "", "matrix_csv_match": "", "evidence_json": ""}
    if not _env_flag("SAVE_HEATMAP_CSV", True):
        return result

    matrix_dir = output.parent / "matrices"
    evidence_dir = output.parent / "evidence"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    cosine_path = matrix_dir / f"{output.stem}_cosine.csv"
    match_path = matrix_dir / f"{output.stem}_match_score.csv"
    evidence_path = evidence_dir / f"{output.stem}_trace.json"
    np.savetxt(cosine_path, np.asarray(raw_similarity, dtype=np.float32), delimiter=",", fmt="%.6f")
    np.savetxt(match_path, np.asarray(match_scores, dtype=np.float32), delimiter=",", fmt="%.6f")

    runs = path_runs(component_path)
    run_records = []
    for index, run in enumerate(runs, start=1):
        raw_values = [float(raw_similarity[row, col]) for row, col in run]
        match_values = [float(match_scores[row, col]) for row, col in run]
        run_records.append(
            {
                "component": index,
                "pairs": [[int(row), int(col)] for row, col in run],
                "line1_range": _run_ranges(ComponentPath(run, runs=[run]), 0)[0],
                "line2_range": _run_ranges(ComponentPath(run, runs=[run]), 1)[0],
                "mean_cosine": float(np.mean(raw_values)) if raw_values else None,
                "min_cosine": float(np.min(raw_values)) if raw_values else None,
                "mean_match_score": float(np.mean(match_values)) if match_values else None,
                "min_match_score": float(np.min(match_values)) if match_values else None,
            }
        )
    payload = {
        "algorithm": algorithm,
        "trace_direction": (
            "maximum_dp_to_zero" if algorithm.lower().startswith("smith")
            else "terminal_nm_to_origin"
        ),
        "full_diagonal_matches": [[int(row), int(col)] for row, col in full_path],
        "traceback_boundaries_col_row": np.asarray(traceback, dtype=np.float32).tolist(),
        "component_settings": component_settings(),
        "components": run_records,
        "line1_component_intervals_px": intervals1 or [],
        "line2_component_intervals_px": intervals2 or [],
    }
    evidence_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    result.update(
        {
            "matrix_csv_cosine": str(cosine_path),
            "matrix_csv_match": str(match_path),
            "evidence_json": str(evidence_path),
        }
    )
    return result
