"""Resolve the CNN backbone recorded by an AlignmentProject checkpoint.

Historical CNN checkpoints predate the ``cnn_backbone`` config field and are known
to use the modified ResNet-34. New checkpoints record the field explicitly. This
helper keeps evaluation and fine-tuning compatible with both generations.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch


def checkpoint_cnn_backbone(path: str | os.PathLike) -> str | None:
    checkpoint = torch.load(Path(path), map_location="cpu")
    if not isinstance(checkpoint, dict):
        return "resnet34"
    config = checkpoint.get("model_config")
    config = config if isinstance(config, dict) else {}
    backend = str(
        config.get("visual_encoder_type", config.get("model_backend", "cnn_bilstm"))
    ).lower()
    if "vit" in backend or "dinov3" in backend or "convnext" in backend:
        return None
    recorded = str(config.get("cnn_backbone", "")).strip().lower().replace("-", "")
    if recorded:
        if recorded not in {"resnet18", "resnet34"}:
            raise RuntimeError(
                f"Checkpoint records unsupported cnn_backbone={recorded!r}: {path}"
            )
        return recorded
    # All project CNN checkpoints created before this field was introduced used
    # ModifiedOCRResNet34.
    return "resnet34"


def configure_for_checkpoint(path: str | os.PathLike) -> str | None:
    backbone = checkpoint_cnn_backbone(path)
    if backbone is None:
        return None
    os.environ["CNN_BACKBONE"] = backbone
    from cnn_backbone_runtime import configure_embedding_model_backbone

    configure_embedding_model_backbone(default=backbone)
    return backbone
