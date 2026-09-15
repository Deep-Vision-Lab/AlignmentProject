"""Shared ResNet-18 encoder for overlapping full-height line windows."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18WindowEncoder(nn.Module):
    """Encode each 3x128x32 physical window with one shared ResNet-18.

    Input:
        [B, 3, 128, line_width]
    Output:
        [B, D, 1, T]

    ResNet-18 produces a 512-D pooled feature per window. A learned projection
    maps 512 -> D (D=192 on the ViT-Tiny branch).
    """

    def __init__(
        self,
        *,
        input_height: int = 128,
        window_size: int = 32,
        stride: int = 16,
        embed_dim: int = 192,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if int(input_height) != 128 or int(window_size) != 32:
            raise ValueError("ResNet18WindowEncoder expects 128x32 windows")
        self.input_height = int(input_height)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.embed_dim = int(embed_dim)
        self.pretrained = bool(pretrained)

        weights = ResNet18_Weights.DEFAULT if self.pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(512, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

    def extract_windows(self, line: torch.Tensor) -> torch.Tensor:
        if line.ndim != 4 or int(line.shape[1]) != 3:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(line.shape)}")
        if int(line.shape[2]) != self.input_height:
            raise ValueError(
                f"Expected line height {self.input_height}, got {line.shape[2]}"
            )
        if int(line.shape[3]) < self.window_size:
            raise ValueError(
                f"Line width {line.shape[3]} is smaller than window size "
                f"{self.window_size}"
            )
        patches = line.unfold(
            dimension=3,
            size=self.window_size,
            step=self.stride,
        )
        return patches.permute(0, 3, 1, 2, 4).contiguous()

    def forward(self, line: torch.Tensor) -> torch.Tensor:
        patches = self.extract_windows(line)
        batch, count, channels, height, width = patches.shape
        flat = patches.reshape(batch * count, channels, height, width)
        features = self.backbone(flat)
        if features.ndim != 2 or int(features.shape[1]) != 512:
            raise RuntimeError(
                f"ResNet-18 must return [N,512], got {tuple(features.shape)}"
            )
        tokens = self.projection(features)
        tokens = tokens.reshape(batch, count, self.embed_dim)
        return tokens.transpose(1, 2).unsqueeze(2).contiguous()
