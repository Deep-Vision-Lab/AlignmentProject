"""Branch backend for ResNet-18 window encoding + ViT-Tiny contextual DTW.

Each 128x32 manuscript window is encoded locally by a shared ResNet-18.
The resulting local tokens are contextualized by a ViT-Tiny-style Transformer
(192-D, 12 blocks, 3 heads by default), fused with their local vectors, and
trained against a frozen Arabic character codebook with positive and negative
letter-DTW. There is no restoration decoder or reconstruction objective.
"""
from __future__ import annotations

import os

import Parameters as P
from vlm_resnet18_tinyvit_positive_dtw import (
    apply_branch_config,
    attach_resnet18_tinyvit_stages,
    install_training_objective,
    model_config as resnet_tinyvit_model_config,
)

apply_branch_config(P)
P.export_environment()

MODEL_NAME = "resnet18_tinyvit_positive_dtw"
# Keep the shared ViT runtime path because the context container is LineWindowViT.
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
    return attach_resnet18_tinyvit_stages(model, P)


def install_training_backend(base_module):
    os.environ["VISUAL_ENCODER_TYPE"] = VISUAL_ENCODER_TYPE
    os.environ["USE_BILSTM"] = "0"
    os.environ["USE_LOCAL_WINDOW_GROUPING"] = "0"

    def constructor(
        window_size=32,
        stride=16,
        vector_size=192,
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
        "local_encoder": "resnet18",
        "use_bilstm": False,
        "use_local_window_grouping": False,
        "vit_input_height": int(P.vit_input_height),
        "vit_layers": int(P.vit_layers),
        "vit_heads": int(P.vit_heads),
        "vit_mlp_dim": int(P.vit_mlp_dim),
        "vit_dropout": float(P.vit_dropout),
        "vit_max_tokens": int(P.vit_max_tokens),
        "vit_position_base_tokens": int(P.vit_position_base_tokens),
        "vit_binarize_input": bool(P.vit_binarize_input),
        "vit_binarize_method": "none",
        "torch_compile_visual": _flag("TORCH_COMPILE_VISUAL", False),
    }
    config.update(resnet_tinyvit_model_config(P))
    return config
