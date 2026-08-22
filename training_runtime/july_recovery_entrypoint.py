#!/usr/bin/env python3
"""July-2026 recovery training entrypoint.

This keeps the branch-selected visual architecture intact while restoring the
training-time behavior that distinguished the stronger July runs. The public
launcher sets the historical window/stride and loss weights; this entrypoint is
kept separate from the canonical runtime so recovery experiments cannot silently
change ordinary jobs.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _install_recovery_span_capacity() -> None:
    """Synchronize feasibility filtering with the active visual lattice."""
    line_width = int(os.environ.get("LINE_WIDTH", "1024"))
    window_size = int(os.environ.get("WINDOW_SIZE", "32"))
    stride_ratio = float(os.environ.get("STRIDE_RATIO", "0.5"))
    if line_width <= 0 or window_size <= 0 or stride_ratio <= 0:
        raise RuntimeError(
            "Recovery geometry requires positive LINE_WIDTH, WINDOW_SIZE, and STRIDE_RATIO"
        )
    if window_size > line_width:
        raise RuntimeError(
            f"Recovery WINDOW_SIZE={window_size} exceeds LINE_WIDTH={line_width}"
        )
    stride = max(1, int(window_size * stride_ratio))
    windows = ((line_width - window_size) // stride) + 1
    os.environ["REAL_MAX_ALIGNMENT_WINDOWS"] = str(windows)
    os.environ["RECOVERY_STRIDE_PIXELS"] = str(stride)


_install_recovery_span_capacity()

from training_runtime import entrypoint as branch_runtime
from zero_shot_preprocessing import (
    configure_domain_robust_normalization,
    install_embedding_profile,
)

optimized = branch_runtime.optimized
install_embedding_profile(optimized.base)

_NORMALIZATION_CONFIG: dict[str, object] = {}
_original_prepare_raw_model = optimized.prepare_raw_model


def _prepare_raw_model_with_july_normalization(model):
    _original_prepare_raw_model(model)
    _NORMALIZATION_CONFIG.clear()
    _NORMALIZATION_CONFIG.update(configure_domain_robust_normalization(model))
    if optimized.base.CTX.is_main:
        print(
            "july_recovery_normalization "
            f"mode={_NORMALIZATION_CONFIG.get('zero_shot_norm_mode', 'unknown')} "
            f"layers={_NORMALIZATION_CONFIG.get('zero_shot_norm_layers', 0)}",
            flush=True,
        )


optimized.prepare_raw_model = _prepare_raw_model_with_july_normalization

_original_model_config = optimized.base.model_config


def _july_model_config(stride, args):
    config = dict(_original_model_config(stride, args))
    config.update(
        {
            "july_recovery_profile": True,
            "july_recovery_reference": "late-july-2026-cnn-bilstm-behavior",
            "july_grouped_blend": float(
                os.environ.get("ZERO_SHOT_GROUPED_BLEND", "0.50")
            ),
            "july_norm_mode": os.environ.get("ZERO_SHOT_NORM_MODE", "frozen-bn"),
            "july_local_hard_negative_weight": float(
                os.environ.get("LOCAL_HARD_NEGATIVE_WEIGHT", "0.35")
            ),
            "july_image_pair_loss_weight": float(
                os.environ.get("IMAGE_PAIR_LOSS_WEIGHT", "0.40")
            ),
            "july_num_negatives": int(os.environ.get("NUM_NEGATIVES", "4")),
            "july_whole_line_sequence_ranking": os.environ.get(
                "USE_SEQUENCE_ALIGNMENT_RANKING", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
            "july_bridge_cross_text_weight": float(
                os.environ.get("BRIDGE_CROSS_TEXT_WEIGHT", "0.0")
            ),
            "july_alignment_capacity_windows": int(
                os.environ.get("REAL_MAX_ALIGNMENT_WINDOWS", "0")
            ),
            "july_stride_pixels": int(os.environ.get("RECOVERY_STRIDE_PIXELS", "0")),
        }
    )
    if _NORMALIZATION_CONFIG:
        config["july_norm_layers"] = int(
            _NORMALIZATION_CONFIG.get("zero_shot_norm_layers", 0)
        )
    return config


optimized.base.model_config = _july_model_config


if __name__ == "__main__":
    optimized.main()
