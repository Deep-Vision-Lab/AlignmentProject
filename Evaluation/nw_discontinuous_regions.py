"""Discontinuous supported-region extraction for global Needleman-Wunsch evaluation.

The global NW traceback is unchanged. Only the predicted aligned region derived
from that path is split so sustained unsupported valleys remain holes.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from Evaluation import nw_core
from Evaluation import sw_core
from Evaluation._eval_utils import patch_range_to_pixels


class DiscontinuousPath(list):
    def __init__(self, values=(), *, runs=(), run_step_counts=()):
        super().__init__(values)
        self.runs = tuple(tuple(run) for run in runs)
        self.run_step_counts = tuple(int(value) for value in run_step_counts)


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


def _step_reward(step, match_scores: np.ndarray, gap_penalty: float) -> float:
    if step.index1 is not None and step.index2 is not None:
        return float(match_scores[int(step.index1), int(step.index2)])
    return float(gap_penalty)


def _supported_runs(steps, match_scores: np.ndarray, gap_penalty: float):
    """Split the old best positive envelope at sustained mismatch valleys."""
    if not steps:
        return [], 0.0

    matrix = np.asarray(match_scores, dtype=np.float32)
    outer_start, outer_end, _ = nw_core._best_positive_segment(
        list(steps), matrix, float(gap_penalty)
    )
    if outer_end <= outer_start:
        return [], 0.0

    support_floor = _env_float("NW_REGION_SUPPORT_FLOOR", 0.0)
    max_bridge = max(0, _env_int("NW_REGION_MAX_BRIDGE_STEPS", 1))
    min_matches = max(1, _env_int("NW_REGION_MIN_MATCH_STEPS", 2))

    supported_positions = []
    for position in range(outer_start, outer_end):
        step = steps[position]
        if step.index1 is None or step.index2 is None:
            continue
        if float(matrix[int(step.index1), int(step.index2)]) > support_floor:
            supported_positions.append(position)

    if not supported_positions:
        return [], 0.0

    groups = []
    group = [supported_positions[0]]
    for position in supported_positions[1:]:
        unsupported_between = position - group[-1] - 1
        if unsupported_between <= max_bridge:
            group.append(position)
        else:
            groups.append(group)
            group = [position]
    groups.append(group)

    kept = [group for group in groups if len(group) >= min_matches]
    if not kept:
        strongest = max(
            supported_positions,
            key=lambda position: _step_reward(
                steps[position], matrix, float(gap_penalty)
            ),
        )
        kept = [[strongest]]

    runs = []
    total_score = 0.0
    for group in kept:
        step_start = int(group[0])
        step_end = int(group[-1]) + 1
        run_pairs = [
            (int(steps[position].index1), int(steps[position].index2))
            for position in group
        ]
        run_score = sum(
            _step_reward(steps[position], matrix, float(gap_penalty))
            for position in range(step_start, step_end)
        )
        runs.append(
            {
                "path": run_pairs,
                "traceback_steps": step_end - step_start,
                "score": float(run_score),
            }
        )
        total_score += float(run_score)

    return runs, float(total_score)


def trace_alignment(result, match_scores: np.ndarray, gap_penalty: float):
    steps = list(result.steps)
    boundaries = nw_core._forward_boundaries(steps)
    global_traceback = np.asarray(list(reversed(boundaries)), dtype=np.float32)
    global_path = [
        (int(step.index1), int(step.index2))
        for step in steps
        if step.index1 is not None and step.index2 is not None
    ]

    records, region_score = _supported_runs(
        steps, np.asarray(match_scores, dtype=np.float32), float(gap_penalty)
    )
    runs = [record["path"] for record in records]
    flat = [pair for run in runs for pair in run]
    region_path = DiscontinuousPath(
        flat,
        runs=runs,
        run_step_counts=[record["traceback_steps"] for record in records],
    )

    return nw_core.NWTrace(
        global_path=global_path,
        global_traceback=global_traceback,
        region_path=region_path,
        region_traceback=np.empty((0, 2), dtype=np.float32),
        region_score=float(region_score),
        gap_in_line1=sum(step.operation == "gap_in_line1" for step in steps),
        gap_in_line2=sum(step.operation == "gap_in_line2" for step in steps),
    )


def _path_runs(path):
    explicit = getattr(path, "runs", None)
    if explicit is not None:
        return [list(run) for run in explicit if run]
    return [list(path)] if path else []


def alignment_region_metrics(path, traceback, similarity_shape) -> dict:
    """Keep SW-compatible keys but measure the union of supported NW runs."""
    if len(similarity_shape) != 2:
        raise ValueError(
            f"Expected a two-dimensional similarity shape, got {similarity_shape}"
        )
    n1, n2 = map(int, similarity_shape)
    runs = _path_runs(path)
    sparse_rows = {int(row) for row, _ in path}
    sparse_cols = {int(col) for _, col in path}
    regions = [sw_core.dense_alignment_region(run, None) for run in runs if run]
    line1_span_windows = sum(region.line1_span_windows for region in regions)
    line2_span_windows = sum(region.line2_span_windows for region in regions)
    run_step_counts = getattr(path, "run_step_counts", None)
    traceback_steps = (
        sum(int(value) for value in run_step_counts)
        if run_step_counts is not None
        else len(path)
    )
    rows = sorted(sparse_rows)
    cols = sorted(sparse_cols)

    return {
        "path_steps": int(len(path)),
        "traceback_steps": int(traceback_steps),
        "warp_steps": max(0, int(traceback_steps) - int(len(path))),
        "line1_path_windows": len(sparse_rows),
        "line2_path_windows": len(sparse_cols),
        "line1_span_windows": int(line1_span_windows),
        "line2_span_windows": int(line2_span_windows),
        "line1_path_fraction": len(sparse_rows) / max(1, n1),
        "line2_path_fraction": len(sparse_cols) / max(1, n2),
        "line1_matched_fraction": min(1.0, line1_span_windows / max(1, n1)),
        "line2_matched_fraction": min(1.0, line2_span_windows / max(1, n2)),
        "line1_path_start": int(rows[0]) if rows else -1,
        "line1_path_end": int(rows[-1]) if rows else -1,
        "line2_path_start": int(cols[0]) if cols else -1,
        "line2_path_end": int(cols[-1]) if cols else -1,
    }


def _merge_intervals(intervals):
    ordered = sorted(
        (float(min(left, right)), float(max(left, right)))
        for left, right in intervals
        if float(right) != float(left)
    )
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for left, right in ordered[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return [(float(left), float(right)) for left, right in merged]


def _union_iou(intervals, gt):
    intervals = _merge_intervals(intervals)
    if not intervals or gt is None:
        return None
    gt_left, gt_right = map(float, gt)
    intersection = sum(
        max(0.0, min(right, gt_right) - max(left, gt_left))
        for left, right in intervals
    )
    pred_length = sum(right - left for left, right in intervals)
    gt_length = max(0.0, gt_right - gt_left)
    union = pred_length + gt_length - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def synthetic_mask_region_metrics(
    path,
    traceback,
    similarity_shape,
    image1,
    image2,
    image_width1: int,
    image_width2: int,
    use_flip: bool,
) -> dict:
    """Compute IoU from the union of supported runs, preserving internal holes."""
    keys = {}
    for prefix in ("line1", "line2"):
        keys.update(
            {
                f"{prefix}_pred_start_px": None,
                f"{prefix}_pred_end_px": None,
                f"{prefix}_gt_start_px": None,
                f"{prefix}_gt_end_px": None,
                f"{prefix}_region_iou": None,
                f"{prefix}_start_error_px": None,
                f"{prefix}_end_error_px": None,
            }
        )
    keys["mean_region_iou"] = None

    runs = _path_runs(path)
    if not runs:
        return keys

    n1, n2 = map(int, similarity_shape)
    specifications = (
        ("line1", 0, n1, int(image_width1), Path(image1)),
        ("line2", 1, n2, int(image_width2), Path(image2)),
    )
    ious = []
    for prefix, axis, n_windows, width, image_path in specifications:
        intervals = []
        for run in runs:
            values = [int(pair[axis]) for pair in run]
            if not values:
                continue
            left, right = patch_range_to_pixels(
                min(values), max(values) + 1, n_windows, width, use_flip
            )
            intervals.append((min(left, right), max(left, right)))

        merged = _merge_intervals(intervals)
        if not merged:
            continue

        pred_start = min(left for left, _ in merged)
        pred_end = max(right for _, right in merged)
        gt = sw_core._mask_interval(sw_core._synthetic_mask_path(image_path))
        keys[f"{prefix}_pred_start_px"] = float(pred_start)
        keys[f"{prefix}_pred_end_px"] = float(pred_end)
        if gt is None:
            continue

        iou = _union_iou(merged, gt)
        keys[f"{prefix}_gt_start_px"] = int(gt[0])
        keys[f"{prefix}_gt_end_px"] = int(gt[1])
        keys[f"{prefix}_region_iou"] = iou
        keys[f"{prefix}_start_error_px"] = abs(float(pred_start) - float(gt[0]))
        keys[f"{prefix}_end_error_px"] = abs(float(pred_end) - float(gt[1]))
        if iou is not None:
            ious.append(float(iou))

    if ious:
        keys["mean_region_iou"] = float(np.mean(ious))
    return keys


def _ranges_text(runs, axis):
    values = []
    for run in runs:
        indices = [int(pair[axis]) for pair in run]
        if indices:
            values.append(f"{min(indices)}–{max(indices)}")
    return ", ".join(values)


def save_visualization(
    arr1,
    arr2,
    features1,
    features2,
    trace,
    heatmap_matrix,
    heatmap_label,
    score,
    normalized_score,
    output,
    use_flip,
    binarized,
    pair,
    score_mode: str,
    show_heatmap=True,
    annotate_values=True,
    value_decimals=2,
    annotation_fontsize=5.0,
):
    """Draw separate red rectangles for supported runs, leaving mismatch holes."""
    n1, n2 = heatmap_matrix.shape
    if show_heatmap:
        heatmap_height = max(8.0, min(24.0, 0.30 * n1))
        figure_width = max(18.0, min(30.0, 0.34 * n2))
        rows, ratios = 3, [2.2, 2.2, heatmap_height]
        figure_height = 5.0 + heatmap_height
    else:
        figure_width, figure_height = 18.0, 5.5
        rows, ratios = 2, [2.2, 2.2]

    fig = nw_core.plt.figure(figsize=(figure_width, figure_height))
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

    runs = _path_runs(trace.region_path)
    for run in runs:
        region = sw_core.dense_alignment_region(run, None)
        if region.empty:
            continue
        nw_core._draw_dense_span(
            axes[0], arr1, region.line1_start, region.line1_end,
            len(features1.contextual), use_flip,
        )
        nw_core._draw_dense_span(
            axes[1], arr2, region.line2_start, region.line2_end,
            len(features2.contextual), use_flip,
        )

    if runs:
        axes[0].set_title(
            f"supported NW runs ({len(runs)}): {_ranges_text(runs, 0)}", fontsize=9
        )
        axes[1].set_title(
            f"supported NW runs ({len(runs)}): {_ranges_text(runs, 1)}", fontsize=9
        )
    else:
        axes[0].set_title("no supported NW region", fontsize=9)
        axes[1].set_title("no supported NW region", fontsize=9)

    metadata = (
        f" | pair_id={pair.pair_id} | label={pair.label_type}"
        if pair.pair_id else ""
    )
    input_label = "binarized real input" if binarized else "synthetic input"
    total_gaps = trace.gap_in_line1 + trace.gap_in_line2
    fig.suptitle(
        f"Needleman-Wunsch global image alignment | score={score:.4f} | "
        f"normalized={normalized_score:.4f} | score_mode={score_mode} | "
        f"gaps={total_gaps} | red=supported runs; holes=unsupported | "
        f"{input_label}{metadata}",
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
            limit = max(
                0.5, float(np.percentile(absolute, 98)) if absolute.size else 1.0
            )
            image = ax.imshow(
                heatmap_matrix, aspect="equal", origin="upper", vmin=-limit,
                vmax=limit, cmap="coolwarm", interpolation="nearest"
            )

        if annotate_values:
            nw_core.annotate_heatmap_values(
                ax, heatmap_matrix, image, decimals=value_decimals,
                fontsize=annotation_fontsize
            )

        traceback_global = trace.global_traceback
        if traceback_global.size:
            xs = traceback_global[:, 0] - 0.5
            ys = traceback_global[:, 1] - 0.5
            ax.plot(
                xs, ys, color="black", linewidth=1.7, marker=".",
                markersize=2.8, label="global traceback: terminal → origin", zorder=7
            )
            nw_core._draw_traceback_arrows(ax, traceback_global)
            ax.scatter(
                [n2 - 1], [n1 - 1], marker="*", s=150,
                facecolors="lime", edgecolors="black", linewidths=1.0,
                label="terminal global DP score", zorder=10
            )
            ax.scatter(
                [-0.5], [-0.5], marker="s", s=45, facecolors="white",
                edgecolors="black", linewidths=1.0, label="global origin", zorder=10
            )

        if trace.global_path:
            ax.scatter(
                [col for _, col in trace.global_path],
                [row for row, _ in trace.global_path],
                s=30, facecolors="none", edgecolors="yellow", linewidths=1.2,
                label="global diagonal correspondences", zorder=9
            )
        if trace.region_path:
            ax.scatter(
                [col for _, col in trace.region_path],
                [row for row, _ in trace.region_path],
                s=20, facecolors="cyan", edgecolors="black", linewidths=0.5,
                label="supported positive runs", zorder=11
            )

        ax.set_xlim(-0.5, n2 - 0.5)
        ax.set_ylim(n1 - 0.5, -0.5)
        ax.set_title(
            f"{heatmap_label}: global path retained; sustained unsupported "
            "valleys split the red regions",
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
    nw_core.plt.close(fig)


def install(runner) -> None:
    """Patch region interpretation only; the global NW DP remains unchanged."""
    runner.trace_alignment = trace_alignment
    runner.alignment_region_metrics = alignment_region_metrics
    runner.synthetic_mask_region_metrics = synthetic_mask_region_metrics
    runner.save_visualization = save_visualization
    nw_core.trace_alignment = trace_alignment
    nw_core.save_visualization = save_visualization
