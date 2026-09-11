"""Branch-selected backend for CFM-inspired spatial language alignment.

Each manuscript line independently produces spatially indexed vectors in a
frozen Arabic language space. The two images meet only through the existing
image-image contrastive/order losses and final cosine/NW alignment.
"""
from __future__ import annotations

import os

import Parameters as P
from vlm_spatial_language_alignment import (
    apply_branch_config,
    attach_spatial_language_stages,
    install_training_objective,
    model_config as spatial_model_config,
)

apply_branch_config(P)
P.export_environment()

os.environ["ALLOW_UNSAFE_SPAN_CONFIG"] = "1"
os.environ["SPAN_MAX_CORE_CHARS_CAP"] = "3"

MODEL_NAME = "vit_cfm_spatial_language"
VISUAL_ENCODER_TYPE = "vit"


def _flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _integer(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _number(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def build_visual_model(
    *, window_size, stride, vector_size, device, use_flip, **_ignored
):
    from embeddingModel import build_vit_from_environment

    model = build_vit_from_environment(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=use_flip,
    )
    return attach_spatial_language_stages(model, P)


def install_training_backend(base_module):
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
    install_training_objective(base_module)


def prepare_visual_model(model):
    from embeddingModel import prepare_vit_model
    prepare_vit_model(model)


def visual_model_config():
    config = {
        "model_backend": MODEL_NAME,
        "visual_encoder_type": VISUAL_ENCODER_TYPE,
        "use_bilstm": False,
        "use_local_window_grouping": False,
        "vit_input_height": _integer("VIT_INPUT_HEIGHT", 128),
        "vit_layers": int(P.vit_layers),
        "vit_heads": _integer("VIT_HEADS", 4),
        "vit_mlp_dim": _integer("VIT_MLP_DIM", 512),
        "vit_dropout": _number("VIT_DROPOUT", 0.10),
        "vit_max_tokens": int(P.vit_max_tokens),
        "vit_position_base_tokens": _integer(
            "VIT_POSITION_BASE_TOKENS", 63
        ),
        "vit_binarize_input": bool(P.vit_binarize_input),
        "vit_binarize_method": (
            "otsu" if bool(P.vit_binarize_input) else "none"
        ),
        "torch_compile_visual": _flag("TORCH_COMPILE_VISUAL", False),
    }
    config.update(spatial_model_config(P))
    return config
