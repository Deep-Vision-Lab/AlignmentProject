"""Runtime selector for the OCR CNN backbone.

New CNN experiments default to ResNet-18 to reduce visual-encoder size/compute.
Existing AlignmentProject checkpoints were produced with the historical modified
ResNet-34, so ``CNN_BACKBONE=resnet34`` remains a strict compatibility mode.

Both variants preserve the same OCR-specific asymmetric layer3/layer4 strides and
produce the same ``vector_size`` output contract.  The selector patches only the
CNN constructor used inside ``embeddingModel.EmbeddingModel``; the surrounding
local grouping, optional BiLSTM, checkpoint format, and training losses are shared.
"""
from __future__ import annotations

import os

import torch.nn as nn
import torchvision

SUPPORTED_BACKBONES = {"resnet18", "resnet34"}


def selected_cnn_backbone(default: str = "resnet18") -> str:
    name = os.environ.get("CNN_BACKBONE", default).strip().lower().replace("-", "")
    aliases = {
        "18": "resnet18",
        "r18": "resnet18",
        "resnet18": "resnet18",
        "34": "resnet34",
        "r34": "resnet34",
        "resnet34": "resnet34",
    }
    resolved = aliases.get(name, name)
    if resolved not in SUPPORTED_BACKBONES:
        raise ValueError(
            f"CNN_BACKBONE must be one of {sorted(SUPPORTED_BACKBONES)}, got {name!r}."
        )
    return resolved


def _load_resnet18():
    try:
        return torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
        )
    except Exception as exc:
        print(
            "[ModifiedOCRResNet18] ImageNet weights unavailable "
            f"({exc}); using weights=None.",
            flush=True,
        )
        return torchvision.models.resnet18(weights=None)


class ModifiedOCRResNet18(nn.Module):
    """ResNet-18 with the same OCR-preserving horizontal strides as legacy R34."""

    def __init__(self, vector_size=512):
        super().__init__()
        base_resnet = _load_resnet18()

        # Match the established OCR ResNet-34 geometry: downsample vertically in
        # the last two stages while retaining more horizontal detail for script.
        base_resnet.layer3[0].conv1.stride = (2, 1)
        base_resnet.layer3[0].downsample[0].stride = (2, 1)
        base_resnet.layer4[0].conv1.stride = (2, 1)
        base_resnet.layer4[0].downsample[0].stride = (2, 1)

        self.backbone = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool,
            base_resnet.layer1,
            base_resnet.layer2,
            base_resnet.layer3,
            base_resnet.layer4,
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_proj = nn.Linear(512, vector_size)

    def forward(self, x):
        x = self.backbone(x)
        x = self.adaptive_pool(x).flatten(1)
        return self.feature_proj(x)


def configure_embedding_model_backbone(default: str = "resnet18") -> str:
    """Patch ``EmbeddingModel`` to construct the selected CNN implementation."""
    import embeddingModel

    name = selected_cnn_backbone(default)
    if name == "resnet18":
        embeddingModel.ModifiedOCRResNet34 = ModifiedOCRResNet18
    # resnet34 deliberately leaves the historical class untouched.  A fresh
    # Python process is used for every training/evaluation job, so there is no
    # need to reverse a prior in-process patch in production.
    return name
