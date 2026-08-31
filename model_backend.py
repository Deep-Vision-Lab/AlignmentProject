"""Branch-selected visual model backend.

This file is intentionally the only active model difference between the two
canonical branches. Training, DDP, losses, data loading, evaluation, scripts,
and optimization code remain shared.
"""
from __future__ import annotations

import os
from pathlib import Path

MODEL_NAME = "vit"
VISUAL_ENCODER_TYPE = "vit"


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _integer(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _job_named_weight_path(path) -> Path:
    """Prefix .pth artifacts under Weights/<job_id>/ with their job_id."""
    resolved = Path(path)
    if resolved.suffix.lower() != ".pth":
        return resolved
    if resolved.parent.parent.name != "Weights":
        return resolved
    job_id = resolved.parent.name
    prefix = f"{job_id}_"
    if resolved.name.startswith(prefix):
        return resolved
    return resolved.with_name(prefix + resolved.name)


def _install_job_named_atomic_saves() -> None:
    """Apply job-id filenames to optimized and stability checkpoint writes."""
    import training_optimizations as optimizations

    if getattr(optimizations, "_job_named_weight_saves_installed", False):
        return
    original_atomic_save = optimizations.atomic_torch_save

    def atomic_torch_save(payload, path):
        return original_atomic_save(payload, _job_named_weight_path(path))

    optimizations.atomic_torch_save = atomic_torch_save
    optimizations._job_named_weight_saves_installed = True


# model_backend is imported before training_stability, so that module imports the
# already-wrapped atomic saver and rescue checkpoints follow the same convention.
_install_job_named_atomic_saves()


def build_visual_model(
    *,
    window_size,
    stride,
    vector_size,
    device,
    use_flip,
    **_ignored,
):
    from vit_embedding_model import build_vit_from_environment

    return build_vit_from_environment(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=use_flip,
    )


def _install_legacy_weight_savers(base_module) -> None:
    """Keep fallback trainer save helpers on the same job-id filename policy."""
    if getattr(base_module, "_job_named_legacy_savers_installed", False):
        return

    original_save_model_weights = base_module.save_model_weights
    original_save_checkpoint = base_module.save_checkpoint

    def save_model_weights(model, text_encoder, job_id, config):
        old_path = Path(
            original_save_model_weights(model, text_encoder, job_id, config)
        )
        new_path = _job_named_weight_path(old_path)
        if new_path != old_path and old_path.exists():
            os.replace(old_path, new_path)
        return str(new_path)

    def save_checkpoint(
        model,
        text_encoder,
        optimizer,
        scheduler,
        scaler,
        epoch,
        job_id,
        config,
    ):
        result = original_save_checkpoint(
            model,
            text_encoder,
            optimizer,
            scheduler,
            scaler,
            epoch,
            job_id,
            config,
        )
        old_path = Path(base_module.weights_dir(job_id)) / "checkpoint_latest.pth"
        new_path = _job_named_weight_path(old_path)
        if new_path != old_path and old_path.exists():
            os.replace(old_path, new_path)
        return result

    base_module.save_model_weights = save_model_weights
    base_module.save_checkpoint = save_checkpoint
    base_module._job_named_legacy_savers_installed = True


def install_training_backend(base_module) -> None:
    """Install this branch's constructor into the shared train.py module."""
    os.environ["VISUAL_ENCODER_TYPE"] = VISUAL_ENCODER_TYPE
    os.environ["USE_BILSTM"] = "0"
    os.environ["USE_LOCAL_WINDOW_GROUPING"] = "0"

    _install_legacy_weight_savers(base_module)

    def constructor(
        window_size=32,
        stride=16,
        vector_size=128,
        device="cuda",
        use_flip=False,
        **kwargs,
    ):
        return build_visual_model(
            window_size=window_size,
            stride=stride,
            vector_size=vector_size,
            device=device,
            use_flip=use_flip,
            **kwargs,
        )

    base_module.EmbeddingModel = constructor


def prepare_visual_model(model) -> None:
    from vit_embedding_model import prepare_vit_model

    prepare_vit_model(model)


def visual_model_config() -> dict:
    return {
        "model_backend": MODEL_NAME,
        "visual_encoder_type": VISUAL_ENCODER_TYPE,
        "use_bilstm": False,
        "use_local_window_grouping": False,
        "vit_input_height": _integer("VIT_INPUT_HEIGHT", 128),
        "vit_layers": _integer("VIT_LAYERS", 4),
        "vit_heads": _integer("VIT_HEADS", 4),
        "vit_mlp_dim": _integer("VIT_MLP_DIM", 512),
        "vit_dropout": _number("VIT_DROPOUT", 0.10),
        "vit_max_tokens": _integer("VIT_MAX_TOKENS", 256),
        "vit_position_base_tokens": _integer("VIT_POSITION_BASE_TOKENS", 63),
        "torch_compile_visual": _flag("TORCH_COMPILE_VISUAL", False),
        "weight_filename_policy": "<job_id>_<artifact>.pth",
    }
