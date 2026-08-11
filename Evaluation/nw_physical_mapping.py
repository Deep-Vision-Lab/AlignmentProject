"""Physical-coordinate visualization fixes for Needleman-Wunsch evaluation.

The Arabic visual encoder reverses its sliding-window sequence so logical window
0 is the rightmost physical window.  NW operates correctly in that logical
sequence, but visualizations must map those logical indices back to the physical
left-to-right line coordinates.  This module keeps the NW DP/traceback itself
unchanged and fixes only score-mode defaults, plotting orientation, and
window-to-pixel conversion.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from Evaluation import _eval_utils
from Evaluation import nw_core
from Evaluation import nw_discontinuous_regions as regions
from Evaluation import sw_core


_ORIGINAL_PATCH_RANGE_TO_PIXELS = _eval_utils.patch_range_to_pixels


def _int_env(name: str, default: int = 0) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def patch_range_to_pixels(
    window_start: int,
    window_end: int,
    n_windows: int,
    image_width: int,
    flipped: bool,
) -> tuple[float, float]:
    """Map a logical window interval to the exact covered input pixels.

    ``window_end`` is exclusive.  When Arabic flipping is enabled, logical
    window 0 corresponds to physical window ``n_windows - 1``.
    """
    n_windows = int(n_windows)
    image_width = int(image_width)
    if n_windows <= 0:
        return 0.0, float(image_width)

    start = max(0, min(int(window_start), n_windows - 1))
    end = max(start + 1, min(int(window_end), n_windows))
    window_size = _int_env("NW_VIS_WINDOW_SIZE", 0)
    stride = _int_env("NW_VIS_WINDOW_STRIDE", 0)
    if window_size <= 0 or stride <= 0:
        return _ORIGINAL_PATCH_RANGE_TO_PIXELS(
            start, end, n_windows, image_width, bool(flipped)
        )

    if flipped:
        physical_first = n_windows - end
        physical_last = n_windows - start - 1
    else:
        physical_first = start
        physical_last = end - 1

    x0 = float(physical_first * stride)
    x1 = float(physical_last * stride + window_size)
    x0 = max(0.0, min(float(image_width), x0))
    x1 = max(0.0, min(float(image_width), x1))
    return min(x0, x1), max(x0, x1)


def _draw_dense_span(ax, array, start, end, n_windows, use_flip):
    if start < 0 or end < start:
        return
    x0, x1 = patch_range_to_pixels(
        start, end + 1, n_windows, array.shape[1], use_flip
    )
    ax.add_patch(
        nw_core.Rectangle(
            (x0, 1),
            max(2.0, x1 - x0),
            max(2.0, array.shape[0] - 2),
            facecolor="red",
            edgecolor="red",
            alpha=0.28,
            linewidth=1.5,
        )
    )


def _physical_matrix(matrix: np.ndarray, use_flip: bool) -> np.ndarray:
    value = np.asarray(matrix)
    if not use_flip:
        return value
    # Both logical axes run right-to-left for Arabic.  Reverse rows and columns
    # so displayed index 0 is the physical leftmost window on each line.
    return value[::-1, ::-1]


def _physical_pair(row: int, col: int, n1: int, n2: int, use_flip: bool):
    if not use_flip:
        return int(row), int(col)
    return int(n1 - 1 - row), int(n2 - 1 - col)


def _physical_traceback(traceback, n1: int, n2: int, use_flip: bool):
    value = np.asarray(traceback, dtype=np.float32).copy()
    if not value.size or not use_flip:
        return value
    # Traceback stores DP boundaries as (column, row), including 0 and n.
    value[:, 0] = float(n2) - value[:, 0]
    value[:, 1] = float(n1) - value[:, 1]
    return value


def _window_center_px(display_index: int) -> float | None:
    window_size = _int_env("NW_VIS_WINDOW_SIZE", 0)
    stride = _int_env("NW_VIS_WINDOW_STRIDE", 0)
    if window_size <= 0 or stride <= 0:
        return None
    return float(display_index * stride + 0.5 * window_size)


def _set_physical_ticks(ax, n1: int, n2: int) -> None:
    """Label heatmap axes with model-input x coordinates for direct comparison."""
    if _window_center_px(0) is None:
        return

    def positions(count: int):
        if count <= 1:
            return [0]
        return sorted(set(int(round(value)) for value in np.linspace(0, count - 1, 6)))

    xticks = positions(n2)
    yticks = positions(n1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{_window_center_px(value):.0f}" for value in xticks])
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{_window_center_px(value):.0f}" for value in yticks])


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
    """Draw NW results in physical line coordinates rather than logical RTL order."""
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

    runs = regions._path_runs(trace.region_path)
    for run in runs:
        region = sw_core.dense_alignment_region(run, None)
        if region.empty:
            continue
        _draw_dense_span(
            axes[0], arr1, region.line1_start, region.line1_end,
            len(features1.contextual), use_flip,
        )
        _draw_dense_span(
            axes[1], arr2, region.line2_start, region.line2_end,
            len(features2.contextual), use_flip,
        )

    if runs:
        axes[0].set_title(
            f"supported NW runs ({len(runs)}): {regions._ranges_text(runs, 0)}", fontsize=9
        )
        axes[1].set_title(
            f"supported NW runs ({len(runs)}): {regions._ranges_text(runs, 1)}", fontsize=9
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
    window_size = _int_env("NW_VIS_WINDOW_SIZE", 0)
    stride = _int_env("NW_VIS_WINDOW_STRIDE", 0)
    geometry = f" | window={window_size} stride={stride}" if window_size and stride else ""
    fig.suptitle(
        f"Needleman-Wunsch global image alignment | score={score:.4f} | "
        f"normalized={normalized_score:.4f} | score_mode={score_mode} | "
        f"gaps={total_gaps} | red=supported runs; holes=unsupported | "
        f"{input_label}{metadata}{geometry}",
        fontsize=11,
        fontweight="bold",
    )

    if show_heatmap:
        ax = fig.add_subplot(grid[2])
        shown = _physical_matrix(heatmap_matrix, bool(use_flip))
        finite = shown[np.isfinite(shown)]
        if finite.size and float(finite.min()) >= 0.0:
            upper = max(1e-6, float(np.percentile(finite, 98)))
            image = ax.imshow(
                shown, aspect="equal", origin="upper", vmin=0.0,
                vmax=upper, cmap="viridis", interpolation="nearest"
            )
        else:
            absolute = np.abs(finite)
            limit = max(
                0.5, float(np.percentile(absolute, 98)) if absolute.size else 1.0
            )
            image = ax.imshow(
                shown, aspect="equal", origin="upper", vmin=-limit,
                vmax=limit, cmap="coolwarm", interpolation="nearest"
            )

        if annotate_values:
            nw_core.annotate_heatmap_values(
                ax, shown, image, decimals=value_decimals,
                fontsize=annotation_fontsize
            )

        traceback_global = _physical_traceback(
            trace.global_traceback, n1, n2, bool(use_flip)
        )
        if traceback_global.size:
            xs = traceback_global[:, 0] - 0.5
            ys = traceback_global[:, 1] - 0.5
            ax.plot(
                xs, ys, color="black", linewidth=1.7, marker=".",
                markersize=2.8, label="global traceback: terminal → origin", zorder=7
            )
            nw_core._draw_traceback_arrows(ax, traceback_global)

            terminal_row, terminal_col = _physical_pair(
                n1 - 1, n2 - 1, n1, n2, bool(use_flip)
            )
            ax.scatter(
                [terminal_col], [terminal_row], marker="*", s=150,
                facecolors="lime", edgecolors="black", linewidths=1.0,
                label="terminal global DP score", zorder=10
            )
            if use_flip:
                origin_x, origin_y = n2 - 0.5, n1 - 0.5
            else:
                origin_x, origin_y = -0.5, -0.5
            ax.scatter(
                [origin_x], [origin_y], marker="s", s=45, facecolors="white",
                edgecolors="black", linewidths=1.0, label="global origin", zorder=10
            )

        if trace.global_path:
            displayed = [
                _physical_pair(row, col, n1, n2, bool(use_flip))
                for row, col in trace.global_path
            ]
            ax.scatter(
                [col for row, col in displayed],
                [row for row, col in displayed],
                s=30, facecolors="none", edgecolors="yellow", linewidths=1.2,
                label="global diagonal correspondences", zorder=9
            )
        if trace.region_path:
            displayed = [
                _physical_pair(row, col, n1, n2, bool(use_flip))
                for row, col in trace.region_path
            ]
            ax.scatter(
                [col for row, col in displayed],
                [row for row, col in displayed],
                s=20, facecolors="cyan", edgecolors="black", linewidths=0.5,
                label="supported positive runs", zorder=11
            )

        ax.set_xlim(-0.5, n2 - 0.5)
        ax.set_ylim(n1 - 0.5, -0.5)
        ax.set_title(
            f"{heatmap_label}: physical line coordinates; global path retained",
            fontsize=10,
        )
        if use_flip:
            ax.set_xlabel("line B physical x (px): left → right")
            ax.set_ylabel("line A physical x (px): left → right, top → bottom")
        else:
            ax.set_xlabel("line B physical x (px): left → right")
            ax.set_ylabel("line A physical x (px): left → right, top → bottom")
        _set_physical_ticks(ax, n1, n2)
        ax.legend(loc="upper left", fontsize=7)
        fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015, label=heatmap_label)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    nw_core.plt.close(fig)


def install(runner) -> None:
    """Install physical plotting/mapping without changing the global NW DP."""
    if getattr(runner, "_nw_physical_mapping_installed", False):
        return

    original_evaluate_sample = runner.evaluate_sample
    original_resolve_score_mode = runner.resolve_score_mode

    def resolve_score_mode(score_mode: str, dataset_type: str) -> str:
        # The synthetic NW reference path resolves auto -> raw.  Make real NW use
        # that same default; explicit centered/mutual-z requests are preserved.
        value = str(score_mode).strip().lower().replace("_", "-")
        if value == "auto" and str(dataset_type).lower() == "real":
            return "raw"
        return original_resolve_score_mode(score_mode, dataset_type)

    def evaluate_sample(*args, **kwargs):
        models = args[0] if args else kwargs.get("models")
        if models is not None:
            model = models.image_model
            os.environ["NW_VIS_WINDOW_SIZE"] = str(int(model.window_size))
            os.environ["NW_VIS_WINDOW_STRIDE"] = str(int(model.stride))

        dataset_type = kwargs.get("dataset_type")
        if dataset_type is None and len(args) >= 3:
            dataset_type = args[2]
        if str(dataset_type).lower() == "real":
            # For localization inspection, show the actual NW diagonal reward,
            # not the accumulated DP table. Set NW_REAL_KEEP_DP_HEATMAP=1 to
            # explicitly retain a requested dp-score heatmap.
            current = kwargs.get("heatmap_source")
            if current is None and len(args) >= 13:
                current = args[12]
            if (
                str(current).lower() == "dp-score"
                and os.environ.get("NW_REAL_KEEP_DP_HEATMAP", "0").strip().lower()
                not in {"1", "true", "yes", "on"}
            ):
                kwargs["heatmap_source"] = "match-score"

        return original_evaluate_sample(*args, **kwargs)

    _eval_utils.patch_range_to_pixels = patch_range_to_pixels
    nw_core.patch_range_to_pixels = patch_range_to_pixels
    nw_core._draw_dense_span = _draw_dense_span
    sw_core.patch_range_to_pixels = patch_range_to_pixels
    regions.patch_range_to_pixels = patch_range_to_pixels

    runner.resolve_score_mode = resolve_score_mode
    runner.evaluate_sample = evaluate_sample
    runner.save_visualization = save_visualization
    regions.save_visualization = save_visualization
    runner._nw_physical_mapping_installed = True
