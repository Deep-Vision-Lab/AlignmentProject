#!/usr/bin/env python3
"""Internal branch-aware entrypoint for the optimized trainer.

Users should run ``scripts/train/run_real_finetune.sh`` rather than invoking this
module directly. ``model_backend.py`` remains the only active difference between
the CNN+BiLSTM and ViT branches.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Import the optimized runtime first. It isolates each torchrun rank's CUDA
# device before importing branch model implementations.
from scripts.train import train_optimized as optimized

import model_backend
from distributed_runtime_guard import install_distributed_runtime_guard
from training_stability import install_training_stability
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
            "explicit_real_split_manifests": os.environ.get(
                "REAL_USE_EXPLICIT_SPLIT_MANIFESTS", "0"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
        }
    )

    # Install the numerical guard only after the final branch-aware config is
    # known. The optimized train() closure resolves train_one_epoch dynamically,
    # so this replacement is active before the first optimizer step.
    install_training_stability(optimized.base, config, args.job_id)
    return config


optimized.base.model_config = _branch_model_config


if __name__ == "__main__":
    optimized.main()
