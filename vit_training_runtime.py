"""Runtime wiring for the dedicated zero-shot ViT training branch."""
from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def install_vit_training_runtime() -> bool:
    """Replace only visual model construction and visual preparation."""
    if os.environ.get("VISUAL_ENCODER_TYPE", "cnn_bilstm").strip().lower() != "vit":
        return False

    import train as base
    import training_optimizations
    from vit_embedding_model import build_vit_from_environment, prepare_vit_model

    if getattr(base, "_vit_training_runtime_installed", False):
        return True

    def build_image_embedding(stride):
        return build_vit_from_environment(
            window_size=base.P.window_size,
            stride=stride,
            vector_size=base.P.vector_size,
            device=base.P.device,
            use_flip=(base.P.lang.lower() == "arabic"),
        )

    original_model_config = base.model_config

    def model_config(stride, args):
        config = dict(original_model_config(stride, args))
        config.update(
            {
                "visual_encoder_type": "vit",
                "use_bilstm": False,
                "use_local_window_grouping": False,
                "vit_input_height": _env_int("VIT_INPUT_HEIGHT", 128),
                "vit_layers": _env_int("VIT_LAYERS", 4),
                "vit_heads": _env_int("VIT_HEADS", 4),
                "vit_mlp_dim": _env_int("VIT_MLP_DIM", 512),
                "vit_dropout": _env_float("VIT_DROPOUT", 0.10),
                "vit_max_tokens": _env_int("VIT_MAX_TOKENS", 256),
                "vit_patch_projection": "full-height-overlapping-window",
                "vit_pretrained": False,
            }
        )
        return config

    base.build_image_embedding = build_image_embedding
    base.model_config = model_config
    training_optimizations.prepare_raw_model = prepare_vit_model
    base._vit_training_runtime_installed = True
    return True
