"""Selectable ResNet18 + ViT-Tiny backend for restoration-positive-DTW.

Default (resnet_token):
    128x32 window -> ResNet18 -> 192-D local token -> ViT-Tiny context.

Optional (physical_window):
    the same physical RGB window feeds both the ResNet local path and the direct
    physical-window TinyViT projection used by the Point-2 ablation.
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

_VARIANT = os.environ.get("MODEL_BACKEND_VARIANT", "resnet_token").strip().lower()
if _VARIANT in {"old", "resnet", "resnet-token", "resnet_token", "token"}:
    _VARIANT = "resnet_token"
elif _VARIANT in {"physical", "physical-window", "physical_window"}:
    _VARIANT = "physical_window"
else:
    raise ValueError(
        "MODEL_BACKEND_VARIANT must be resnet_token or physical_window; "
        f"got {_VARIANT!r}"
    )

apply_branch_config(P)
from architecture_experiment import is_compact, metadata as architecture_metadata
if is_compact(P) and _VARIANT != "resnet_token":
    raise ValueError("The compact architecture requires MODEL_BACKEND_VARIANT=resnet_token")
if _VARIANT == "physical_window":
    P.experiment_name = "resnet18_physical_window_tinyvit_positive_dtw"
    P.restoration_context_input = "direct_physical_128x32_rgb_window"
else:
    P.experiment_name = "resnet18_tinyvit_positive_dtw"
    P.restoration_context_input = "resnet18_window_token"
P.export_environment()
if is_compact(P):
    P.experiment_name = P.architecture_variant

MODEL_NAME = (
    "resnet18_physical_window_tinyvit_positive_dtw"
    if _VARIANT == "physical_window"
    else "resnet18_tinyvit_positive_dtw"
)
VISUAL_ENCODER_TYPE = "vit"
if is_compact(P):
    MODEL_NAME = P.architecture_variant


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
    if _VARIANT == "physical_window":
        from physical_window_vit_branch import attach_physical_window_vit_stages

        return attach_physical_window_vit_stages(model, P)
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
        "model_backend_variant": _VARIANT,
        "visual_encoder_type": VISUAL_ENCODER_TYPE,
        "use_bilstm": False,
        "use_local_window_grouping": False,
        "window_height": 128,
        "window_width": 32,
        "window_stride": int(round(P.window_size * P.stride_ratio)),
        "local_encoder_type": "resnet18",
        "local_input": "physical_gray_128x32_window" if P.visual_input_channels == 1 else "physical_rgb_128x32_window",
        "resnet18_pretrained": bool(P.resnet18_pretrained),
        "resnet18_pretrained_source": "torchvision/ResNet18_Weights.DEFAULT",
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
        "pretrained_local_only": bool(P.pretrained_local_only),
        "fusion": "concat_projection_norm",
        "fusion_dropout": float(os.environ.get("FUSION_DROPOUT", "0.0")),
        "torch_compile_visual": _flag("TORCH_COMPILE_VISUAL", False),
    }
    if _VARIANT == "physical_window":
        config.update(
            {
                "context_input": "same_physical_rgb_128x32_window",
                "context_window_subdivision": "none",
                "context_patch_projection": "flatten(3x128x32)->linear(192)+layernorm",
                "tiny_vit_pretrained_scope": (
                    "transformer+position; physical-window projection is new"
                ),
            }
        )
    else:
        config.update(
            {
                "context_input": "resnet18_projected_window_token_sequence",
                "context_window_subdivision": "none",
                "context_patch_projection": "resnet18(512)->linear(192)",
                "tiny_vit_pretrained_scope": "transformer+position",
            }
        )
    config.update(restoration_model_config(P))
    config.update(architecture_metadata(P))
    return config
