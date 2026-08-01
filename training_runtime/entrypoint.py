#!/usr/bin/env python3
"""Internal branch-aware entrypoint for the optimized trainer.

Users should run ``scripts/train/run_connected_subword_finetune.sh`` on the
connected-subword experiment branches. ``model_backend.py`` remains the only
active difference between the CNN+BiLSTM and ViT branches.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.train import train_optimized as optimized

import model_backend
from connected_subword_mode import (
    install_connected_subword_mode,
    install_connected_subword_training,
)
from distributed_runtime_guard import install_distributed_runtime_guard
from unified_line_geometry import install_training_geometry


_GEOMETRY_CONFIG = install_training_geometry()
install_connected_subword_mode()
install_connected_subword_training(optimized.base)

model_backend.install_training_backend(optimized.base)
optimized.prepare_raw_model = model_backend.prepare_visual_model
install_distributed_runtime_guard(optimized.base)

_original_model_config = optimized.base.model_config


def _branch_model_config(stride, args):
    config = _original_model_config(stride, args)
    config.update(_GEOMETRY_CONFIG)
    config.update(model_backend.visual_model_config())
    config.update(
        {
            "span_dtw_batch_bucket_mode": os.environ.get(
                "SPAN_DTW_BATCH_BUCKET_MODE", "power2"
            ),
            "span_dtw_text_bucket_size": int(
                os.environ.get("SPAN_DTW_TEXT_BUCKET_SIZE", "32")
            ),
            "jax_compilation_cache_dir": os.environ.get(
                "JAX_COMPILATION_CACHE_DIR", ""
            ),
            "distributed_timeout_seconds": int(
                os.environ.get("DIST_TIMEOUT_SECONDS", "7200")
            ),
            "span_tokenization_mode": os.environ.get(
                "SPAN_TOKENIZATION_MODE", "character_span"
            ),
            "span_subword_boundary_token": "<SUBWORD_BOUNDARY>",
            "span_use_blank_transitions": os.environ.get(
                "SPAN_USE_BLANK_TRANSITIONS", "1"
            ).lower()
            in {"1", "true", "yes", "on"},
            "span_connected_windows_per_char": int(
                os.environ.get("SPAN_CONNECTED_WINDOWS_PER_CHAR", "3")
            ),
            "span_connected_extra_windows": int(
                os.environ.get("SPAN_CONNECTED_EXTRA_WINDOWS", "1")
            ),
            "span_subword_boundary_max_windows": int(
                os.environ.get("SPAN_SUBWORD_BOUNDARY_MAX_WINDOWS", "2")
            ),
            "span_space_max_windows": int(
                os.environ.get("SPAN_SPACE_MAX_WINDOWS", "3")
            ),
        }
    )
    return config


optimized.base.model_config = _branch_model_config


if __name__ == "__main__":
    optimized.main()
