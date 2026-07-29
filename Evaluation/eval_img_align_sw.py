#!/usr/bin/env python3
"""Checkpoint-compatible Smith-Waterman local image alignment."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

# When this file is executed directly, Python adds Evaluation/ rather than the
# repository root to sys.path. Add the project root before importing the
# Evaluation package so both of these forms work:
#   python Evaluation/eval_img_align_sw.py ...
#   python -m Evaluation.eval_img_align_sw ...
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Install exactly the same canvas and measured-ink-height normalization used by
# synthetic and real training before evaluation preprocessing modules import it.
from unified_line_geometry import install_evaluation_geometry

_EVALUATION_GEOMETRY = install_evaluation_geometry()

# Reconstruct either the legacy CNN+BiLSTM or the new ViT from checkpoint config.
# This must be installed before sw_runner imports load_evaluation_models.
from Evaluation.vit_evaluation import install_vit_evaluation_loader

install_vit_evaluation_loader()

# Install shared real-image preprocessing and balanced ArabicDataset
# split/sampling before sw_runner imports functions from Evaluation.sw_dataset.
from Evaluation.zero_shot_sw import install_dataset_patches

install_dataset_patches()

from Evaluation import sw_core as _sw_core
from Evaluation import sw_dataset as _sw_dataset
from Evaluation.balanced_sampling import balanced_group_split_pairs as _balanced_split


def _readable_heatmap_values(ax, matrix, image, decimals=2, fontsize=5.0):
    """Render every value above traceback graphics with a contrast-safe label."""
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
                _sw_core.format_heatmap_value(value, decimals),
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


def _readable_traceback_arrows(ax, traceback):
    """Draw sparse direction arrows without covering the heatmap values."""
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


# sw_core.save_visualization resolves these helpers from its module globals at
# draw time. Values therefore stay above the traceback line, arrows, matched
# circles, and maximum marker without changing the alignment itself.
_sw_core.annotate_heatmap_values = _readable_heatmap_values
_sw_core._annotate_heatmap_values = _readable_heatmap_values
_sw_core._draw_traceback_arrows = _readable_traceback_arrows


def _diverse_group_split(pairs, seed):
    return _balanced_split(pairs, seed, _sw_dataset.random_split_pairs)


_sw_dataset.group_split_pairs = _diverse_group_split
_sw_dataset._group_split_pairs = _diverse_group_split

from Evaluation import sw_runner as _implementation
from Evaluation.zero_shot_sw import install_runner_patches

install_runner_patches(_implementation)

# Real images still need to be binarized before feature extraction, but those
# intermediate files are now created only inside a temporary directory and are
# deleted immediately after each sample. No binarization folder is written into
# the evaluation output directory.
_patched_evaluate_sample = _implementation.evaluate_sample


def _evaluate_without_saved_binarization(*args, **kwargs):
    kwargs["save_binary"] = False
    return _patched_evaluate_sample(*args, **kwargs)


_implementation.evaluate_sample = _evaluate_without_saved_binarization
_implementation._evaluate_sample = _evaluate_without_saved_binarization

# Re-export public and private helpers for backward-compatible imports/tests.
globals().update(
    {
        name: getattr(_implementation, name)
        for name in dir(_implementation)
        if not name.startswith("__")
    }
)


if __name__ == "__main__":
    _implementation.main()
