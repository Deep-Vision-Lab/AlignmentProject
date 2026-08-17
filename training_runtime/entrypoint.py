#!/usr/bin/env python3
"""Internal branch-aware entrypoint for the optimized trainer.

Users should run ``scripts/train/run_real_finetune.sh`` rather than invoking this
module directly. ``model_backend.py`` selects the visual architecture while the
training/data/loss runtime remains shared.
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
from extra_real_training_v2 import install as install_extra_real_training
from real_unique_line_training import install as install_unique_real_training
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

# Do not rely only on replacing train.EmbeddingModel. The generic trainer owns a
# build_image_embedding() helper and that indirection allowed a CNN constructor
# to survive in a ViT run. Replace the builder itself with the active branch
# backend so model construction is unambiguous.
def _branch_build_image_embedding(stride):
    P = optimized.base.P
    return model_backend.build_visual_model(
        window_size=P.window_size,
        stride=stride,
        vector_size=P.vector_size,
        device=P.device,
        use_flip=(P.lang.lower() == "arabic"),
        use_bilstm=getattr(P, "use_bilstm", True),
        bilstm_layers=getattr(P, "bilstm_layers", 2),
        bilstm_hidden_dim=getattr(P, "bilstm_hidden_dim", None),
        use_local_grouping=getattr(P, "use_local_window_grouping", True),
        local_group_size=getattr(P, "local_window_group_size", 3),
    )


model_backend.install_training_backend(optimized.base)
optimized.base.build_image_embedding = _branch_build_image_embedding
optimized.prepare_raw_model = model_backend.prepare_visual_model
install_distributed_runtime_guard(optimized.base)

_unique_real = os.environ.get("REAL_UNIQUE_LINE_ADAPTATION", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
_extra_real = os.environ.get("REAL_USE_EXTRA_NO_SHARED", "0").strip().lower() in {
    "1", "true", "yes", "on"
}
if _unique_real and _extra_real:
    raise RuntimeError(
        "REAL_UNIQUE_LINE_ADAPTATION and REAL_USE_EXTRA_NO_SHARED are mutually exclusive."
    )
if _unique_real:
    install_unique_real_training(optimized.base)
elif _extra_real:
    install_extra_real_training(optimized.base)
install_epoch_subset_sampling(optimized.base)

# Validate the freshly constructed architecture before attempting to load any
# pretrained/resume checkpoint. This converts a huge load_state_dict mismatch
# into an immediate, precise backend-construction error.
_original_load_initial_states = optimized.base._load_initial_states


def _load_initial_states_checked(args, model, text_encoder):
    backend = str(model_backend.MODEL_NAME).strip().lower()
    keys = tuple(model.state_dict().keys())
    has_vit = any(key.startswith("vit_encoder.") for key in keys)
    has_cnn = any(key.startswith("cnn_encoder.") for key in keys)
    has_bilstm = any(key.startswith("sequence_encoder.bilstm.") for key in keys)
    expects_bilstm = bool(getattr(optimized.base.P, "use_bilstm", True))

    if backend == "vit" and (not has_vit or has_cnn or has_bilstm):
        raise RuntimeError(
            "ViT backend constructed the wrong visual model before checkpoint load: "
            f"has_vit={has_vit} has_cnn={has_cnn} has_bilstm={has_bilstm}."
        )
    if backend == "cnn_bilstm" and (
        not has_cnn
        or has_vit
        or (expects_bilstm and not has_bilstm)
        or (not expects_bilstm and has_bilstm)
    ):
        mode = "cnn_bilstm" if expects_bilstm else "cnn_only"
        raise RuntimeError(
            f"CNN backend constructed the wrong {mode} visual model before checkpoint load: "
            f"has_vit={has_vit} has_cnn={has_cnn} has_bilstm={has_bilstm}."
        )

    if optimized.base.CTX.is_main:
        print(
            "visual_builder "
            f"backend={backend} model_class={model.__class__.__name__} "
            f"cnn_mode={'bilstm' if expects_bilstm else 'cnn_only'} "
            f"has_vit={int(has_vit)} has_cnn={int(has_cnn)} "
            f"has_bilstm={int(has_bilstm)}",
            flush=True,
        )
    return _original_load_initial_states(args, model, text_encoder)


optimized.base._load_initial_states = _load_initial_states_checked

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
            "unique_real_line_adaptation": _unique_real,
            "use_extra_no_shared_real_lines": _extra_real,
            "extra_real_exclude_eval_pages": os.environ.get(
                "REAL_EXTRA_EXCLUDE_EVAL_PAGES", "1"
            ).strip().lower()
            in {"1", "true", "yes", "on"},
        }
    )
    install_training_stability(optimized.base, config, args.job_id)
    return config


optimized.base.model_config = _branch_model_config


if __name__ == "__main__":
    optimized.main()
