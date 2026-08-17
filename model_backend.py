"""Branch-selected visual model backend.

This branch supports the two CNN modes used by the project:
- CNN only (``USE_BILSTM=0``)
- CNN + BiLSTM (``USE_BILSTM=1``)

New CNN experiments default to ResNet-18.  Historical ResNet-34 checkpoints remain
supported through ``CNN_BACKBONE=resnet34``; checkpoint-aware launchers/evaluation
resolve that legacy setting automatically.
"""
from __future__ import annotations

import os

MODEL_NAME = "cnn_bilstm"
VISUAL_ENCODER_TYPE = "cnn_bilstm"


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


def build_visual_model(
    *,
    window_size,
    stride,
    vector_size,
    device,
    use_flip,
    use_bilstm=True,
    bilstm_layers=2,
    bilstm_hidden_dim=None,
    use_local_grouping=True,
    local_group_size=3,
    **_ignored,
):
    # Configure before EmbeddingModel.__init__, which looks up its CNN constructor
    # at runtime.  This keeps all sequence/local-grouping code shared.
    from cnn_backbone_runtime import configure_embedding_model_backbone

    backbone = configure_embedding_model_backbone(default="resnet18")
    from embeddingModel import EmbeddingModel

    model = EmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=use_flip,
        use_bilstm=use_bilstm,
        bilstm_layers=bilstm_layers,
        bilstm_hidden_dim=bilstm_hidden_dim,
        use_local_grouping=use_local_grouping,
        local_group_size=local_group_size,
    )
    model.cnn_backbone_name = backbone
    return model


def install_training_backend(base_module) -> None:
    """Install this branch's constructor into the shared train.py module."""
    os.environ["VISUAL_ENCODER_TYPE"] = VISUAL_ENCODER_TYPE

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
    from training_optimizations import prepare_raw_model

    prepare_raw_model(model)


def visual_model_config() -> dict:
    from cnn_backbone_runtime import selected_cnn_backbone

    return {
        "model_backend": MODEL_NAME,
        "visual_encoder_type": VISUAL_ENCODER_TYPE,
        "cnn_backbone": selected_cnn_backbone(default="resnet18"),
        "cnn_chunk_size": _integer("CNN_CHUNK_SIZE", 1024),
        "use_channels_last": _flag("USE_CHANNELS_LAST", True),
        "torch_compile_visual": _flag("TORCH_COMPILE_VISUAL", False),
    }
