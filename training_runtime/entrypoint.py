#!/usr/bin/env python3
"""Internal branch-aware entrypoint for the optimized trainer.

Users should run ``scripts/train/run_real_finetune.sh`` rather than invoking this
module directly. ``model_backend.py`` remains the only active model difference between
the CNN+BiLSTM and ViT branches.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _git_value(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_DIR), *args],
        text=True,
    ).strip()


def _validate_pinned_checkout() -> None:
    expected_branch = os.environ.get("TRAIN_EXPECTED_BRANCH", "").strip()
    expected_commit = os.environ.get("TRAIN_EXPECTED_COMMIT", "").strip()
    if not expected_branch and not expected_commit:
        return

    current_branch = _git_value("branch", "--show-current")
    current_commit = _git_value("rev-parse", "HEAD")
    problems = []
    if expected_branch and current_branch != expected_branch:
        problems.append(
            f"branch changed: expected={expected_branch} current={current_branch}"
        )
    if expected_commit and current_commit != expected_commit:
        problems.append(
            f"commit changed: expected={expected_commit} current={current_commit}"
        )
    if problems:
        raise RuntimeError(
            "Training checkout changed after submission; refusing to mix model "
            "architectures. " + "; ".join(problems) + ". Keep the shared checkout "
            "on the submitted branch/commit until the Slurm job finishes."
        )


_validate_pinned_checkout()

# Import the optimized runtime only after verifying that the checkout is still
# the exact branch/commit captured by the public launcher.
from scripts.train import train_optimized as optimized

import model_backend
from distributed_runtime_guard import install_distributed_runtime_guard
from epoch_subset_sampling import install_epoch_subset_sampling
from training_stability import install_training_stability
from unified_line_geometry import install_training_geometry


def _validate_backend_identity() -> None:
    expected = os.environ.get("TRAIN_EXPECTED_BACKEND", "").strip().lower()
    actual = str(model_backend.MODEL_NAME).strip().lower()
    if expected and actual != expected:
        raise RuntimeError(
            "Training backend changed after submission: "
            f"expected={expected} current={actual}. Refusing to load the checkpoint."
        )


_validate_backend_identity()

# Install one deterministic canvas/ink geometry for synthetic and real data
# before the optimized trainer constructs any train/validation/test loaders.
_GEOMETRY_CONFIG = install_training_geometry()

model_backend.install_training_backend(optimized.base)
optimized.prepare_raw_model = model_backend.prepare_visual_model
install_distributed_runtime_guard(optimized.base)
install_epoch_subset_sampling(optimized.base)

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
    install_training_stability(optimized.base, config, args.job_id)
    return config


optimized.base.model_config = _branch_model_config


if __name__ == "__main__":
    optimized.main()
