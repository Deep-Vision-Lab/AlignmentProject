"""Direct physical-window tokenization for the contextual ViT branch.

Each ViT input token corresponds to exactly one real 3x128x32 manuscript
window extracted with the same horizontal stride used by the ResNet local
branch. The window is NOT subdivided into smaller image patches and is NOT
passed through a CNN. Its pixels are flattened and projected directly into the
192-D ViT-Tiny token space.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PhysicalWindowEmbedding(nn.Module):
    """Map each exact 128x32 RGB window directly to one ViT token.

    Input:
        [B, 3, 128, line_width]
    Output:
        [B, D, 1, T]

    Windows are extracted explicitly with ``Tensor.unfold`` using window width
    32 and stride 16 (or the configured stride). Therefore token ``t`` covers
    exactly the same physical x-range as local ResNet token ``t``.
    """

    def __init__(
        self,
        *,
        input_height: int = 128,
        window_size: int = 32,
        stride: int = 16,
        embed_dim: int = 192,
    ) -> None:
        super().__init__()
        if int(input_height) != 128 or int(window_size) != 32:
            raise ValueError("PhysicalWindowEmbedding expects 128x32 windows")
        if int(stride) <= 0:
            raise ValueError("stride must be positive")

        self.input_height = int(input_height)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.embed_dim = int(embed_dim)
        self.input_dim = 3 * self.input_height * self.window_size

        # This is the ViT patch-projection analogue, except the entire physical
        # 128x32 manuscript window is one patch/token. No smaller visual patches
        # and no CNN are introduced here.
        self.projection = nn.Linear(self.input_dim, self.embed_dim)
        self.norm = nn.LayerNorm(self.embed_dim)
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

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
        # [B,3,H,T,W] -> [B,T,3,H,W]
        return patches.permute(0, 3, 1, 2, 4).contiguous()

    def forward(self, line: torch.Tensor) -> torch.Tensor:
        windows = self.extract_windows(line)
        batch, count, channels, height, width = windows.shape
        if channels != 3 or height != self.input_height or width != self.window_size:
            raise RuntimeError(
                "Unexpected physical-window geometry: "
                f"{tuple(windows.shape)}"
            )
        flat = windows.reshape(batch, count, self.input_dim)
        tokens = self.norm(self.projection(flat))
        return tokens.transpose(1, 2).unsqueeze(2).contiguous()
