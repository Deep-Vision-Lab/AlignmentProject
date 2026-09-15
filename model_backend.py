"""Branch backend for ResNet-18 + ViT-Tiny contextual letter-DTW.

Each manuscript line is processed independently. Overlapping 128x32 RGB windows
are encoded by one shared ResNet-18, projected to 192-D local tokens, then
contextualized by a canonical ViT-Tiny transformer core (12 layers, 3 heads,
768-D MLP). Local and contextual vectors are fused for DTW. There is no decoder
or restoration loss.
"""
from __future__ import annotations

import os

import Parameters as P
from vlm_restoration_positive_dtw import (
    apply_branch_config,
    attach_restoration_dtw_stages,
    install_training_objective,
    model_config as restoration_model_config,
)

apply_branch_config(P)
P.export_environment()

MODEL_NAME = "resnet18_tinyvit_positive_dtw"
VISUAL_ENCODER_TYPE = "vit"


def _flag(name, default):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
    return attach_restoration_dtw_stages(model, P)


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
        "use_bilstm": False,
        "use_local_window_grouping": False,
        "vit_input_height": 128,
        "vit_variant": "vit_tiny",
        "vit_layers": int(P.vit_layers),
        "vit_heads": int(P.vit_heads),
        "vit_mlp_dim": int(P.vit_mlp_dim),
        "vit_dropout": float(P.vit_dropout),
        "vit_max_tokens": int(P.vit_max_tokens),
        "vit_position_base_tokens": int(getattr(P, "vit_position_base_tokens", 63)),
        "vit_binarize_input": False,
        "vit_binarize_method": "none",
        "local_encoder_type": "resnet18",
        "resnet18_pretrained": bool(P.resnet18_pretrained),
        "resnet18_pretrained_source": "torchvision/ResNet18_Weights.DEFAULT",
        "tiny_vit_pretrained": bool(P.tiny_vit_pretrained),
        "tiny_vit_pretrained_model": str(P.tiny_vit_pretrained_model),
        "pretrained_local_only": bool(P.pretrained_local_only),
        "torch_compile_visual": _flag("TORCH_COMPILE_VISUAL", False),
    }
    config.update(restoration_model_config(P))
    return config
