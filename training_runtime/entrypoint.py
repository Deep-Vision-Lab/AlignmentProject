#!/usr/bin/env python3
"""Internal branch-aware entrypoint for optimized stroke-aware training."""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

_tokenization_mode = os.environ.get(
    "SPAN_TOKENIZATION_MODE", "character_span"
).strip().lower()
if _tokenization_mode in {
    "connected_subword",
    "connected-subword",
    "joining_run",
    "joining-run",
} and int(os.environ.get("MAX_WINDOWS_PER_SPAN", "3")) > 3:
    os.environ["ALLOW_UNSAFE_SPAN_CONFIG"] = "1"

# This branch is stroke-aware even when the synthetic job uses Span-DTW rather
# than renderer-box direct supervision.
os.environ.setdefault("DIRECT_SUBWORD_STROKE_INPUT", "1")

from scripts.train import train_optimized as optimized

import model_backend
from connected_subword_mode import (
    install_connected_subword_mode,
    install_connected_subword_training,
)
from direct_subword_supervision import config as direct_subword_config
from direct_subword_supervision import install as install_direct_subword_supervision
from distributed_runtime_guard import install_distributed_runtime_guard
from stroke_aware_preprocessing import install_training_preprocessing
from unified_line_geometry import install_training_geometry


_GEOMETRY_CONFIG = install_training_geometry()
install_connected_subword_mode()
install_connected_subword_training(optimized.base)
install_training_preprocessing()

model_backend.install_training_backend(optimized.base)
optimized.prepare_raw_model = model_backend.prepare_visual_model
install_distributed_runtime_guard(optimized.base)
_DIRECT_SUBWORD_CONFIG = install_direct_subword_supervision(optimized.base)

_original_model_config = optimized.base.model_config


def _branch_model_config(stride, args):
    config = _original_model_config(stride, args)
    config.update(_GEOMETRY_CONFIG)
    config.update(model_backend.visual_model_config())
    config.update(_DIRECT_SUBWORD_CONFIG or direct_subword_config())
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
            "span_connected_max_units_per_span": int(
                os.environ.get(
                    "SPAN_CONNECTED_MAX_UNITS_PER_SPAN",
                    os.environ.get("MAX_TEXT_SPAN_CHARS", "3"),
                )
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
