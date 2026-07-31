#!/usr/bin/env python3
"""Canonical optimized trainer for the active branch model.

Both canonical branches execute this file. The shared optimized trainer owns
CUDA-rank isolation, DDP, data loading, losses, AMP, validation, checkpointing,
resume, profiling, and W&B. ``model_backend.py`` is the only branch-specific
active model boundary.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Import the optimized runtime first. It isolates each torchrun rank's CUDA
# device before importing branch model implementations.
from scripts.train import train_optimized as optimized

import model_backend
from distributed_runtime_guard import install_distributed_runtime_guard
from unified_line_geometry import install_training_geometry


# Install one deterministic canvas/ink geometry for synthetic and real data
# before the optimized trainer constructs any train/validation/test loaders.
_GEOMETRY_CONFIG = install_training_geometry()

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
                os.environ.get("SPAN_DTW_TEXT_BUCKET_SIZE", "64")
            ),
            "jax_compilation_cache_dir": os.environ.get(
                "JAX_COMPILATION_CACHE_DIR", ""
            ),
            "distributed_timeout_seconds": int(
                os.environ.get("DIST_TIMEOUT_SECONDS", "7200")
            ),
        }
    )
    return config


optimized.base.model_config = _branch_model_config


if __name__ == "__main__":
    optimized.main()
