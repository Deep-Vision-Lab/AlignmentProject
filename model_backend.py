"""Branch backend for ResNet local + direct physical-window ViT context DTW.

Each manuscript line is processed with two aligned visual paths using exactly
same 128x32 RGB windows at stride 16:

* local path: window -> pretrained ResNet-18 -> 192-D local token L_t
* context path: SAME window pixels -> direct linear ViT token -> pretrained
  ViT-Tiny transformer sequence -> contextual token C_t

L_t and C_t are fused for positive letter-DTW. The contextual path never
consumes ResNet vectors and never subdivides the physical window into smaller
custom patches.
"""
from __future__ import annotations

import os

import Parameters as P
from physical_window_vit_branch import attach_physical_window_vit_stages
from vlm_restoration_positive_dtw import (
    apply_branch_config,
    install_training_objective,
    model_config as restoration_model_config,
)

apply_branch_config(P)
P.experiment_name = "resnet18_physical_window_tinyvit_positive_dtw"
P.restoration_context_input = "direct_physical_128x32_rgb_window"
P.export_environment()

MODEL_NAME = "resnet18_physical_window_tinyvit_positive_dtw"
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
    return attach_physical_window_vit_stages(model, P)


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
        "window_height": 128,
        "window_width": 32,
        "window_stride": int(round(P.window_size * P.stride_ratio)),
        "local_encoder_type": "resnet18",
        "local_input": "physical_rgb_128x32_window",
        "resnet18_pretrained": bool(P.resnet18_pretrained),
        "resnet18_pretrained_source": "torchvision/ResNet18_Weights.DEFAULT",
        "context_input": "same_physical_rgb_128x32_window",
        "context_window_subdivision": "none",
        "context_patch_projection": "flatten(3x128x32)->linear(192)+layernorm",
        "vit_variant": "vit_tiny",
        "vit_layers": int(P.vit_layers),
        "vit_heads": int(P.vit_heads),
        "vit_mlp_dim": int(P.vit_mlp_dim),
        "vit_dropout": float(P.vit_dropout),
        "vit_max_tokens": int(P.vit_max_tokens),
        "vit_position_base_tokens": int(getattr(P, "vit_position_base_tokens", 63)),
        "vit_binarize_input": False,
        "vit_binarize_method": "none",
        "tiny_vit_pretrained": bool(P.tiny_vit_pretrained),
        "tiny_vit_pretrained_model": str(P.tiny_vit_pretrained_model),
        "tiny_vit_pretrained_scope": "transformer+position; physical-window projection is new",
        "pretrained_local_only": bool(P.pretrained_local_only),
        "fusion": "concat_projection_norm",
        "torch_compile_visual": _flag("TORCH_COMPILE_VISUAL", False),
    }
    config.update(restoration_model_config(P))
    return config
