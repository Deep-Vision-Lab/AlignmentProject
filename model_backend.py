"""Branch-selected visual model backend.

This file is intentionally the only active model difference between the two
canonical branches. Training, DDP, losses, data loading, evaluation, scripts,
and optimization code remain shared.
"""
from __future__ import annotations

import os

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


def install_training_backend(base_module) -> None:
    """Install this branch's constructor into the shared train.py module."""
    os.environ["VISUAL_ENCODER_TYPE"] = VISUAL_ENCODER_TYPE
    os.environ["USE_BILSTM"] = "0"
    os.environ["USE_LOCAL_WINDOW_GROUPING"] = "0"

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
        "torch_compile_visual": _flag("TORCH_COMPILE_VISUAL", False),
    }
