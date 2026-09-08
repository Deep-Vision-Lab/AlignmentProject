"""Exact RGB window extraction followed by a shared, trainable spatial CNN.

Extraction preserves the values of the model input (including any upstream
normalization). Learning changes CNN features and weights, never input pixels.
The output layout matches the existing ViT patch projection so all downstream
depiction, RTL ordering, positional encoding and contextual losses stay intact.
"""
from __future__ import annotations

import torch
from torch import nn


def extract_rgb_windows(image: torch.Tensor, window_size: int = 32,
                        stride: int = 16) -> torch.Tensor:
    """Copy full-height windows into [B, N, 3, H, window_size], left to right.

    No weights, interpolation, averaging, activation or intensity change occurs
    here. Only complete windows are returned; a short trailing remainder is
    omitted, matching the original no-padding patch projection.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected RGB image [B,3,H,W]")
    window_size, stride = int(window_size), int(stride)
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    if image.shape[-1] < window_size:
        raise ValueError("Image width is smaller than window_size")
    # clone() also prevents aliasing when there is only one window. Gradients
    # remain connected to the input if a caller explicitly requests them.
    return image.unfold(3, window_size, stride).permute(0, 3, 1, 2, 4).clone(
        memory_format=torch.contiguous_format
    )


class SlidingWindowCNN(nn.Module):
    """Extract windows first, then apply the same spatial CNN to every window.

    For H=128 and window_size=32:
      RGB window -> 16x64x32 -> 32x32x16 -> 64x16x8
      -> adaptive average pool 64x4x2 -> flatten 512 -> Linear(512,D).
    GroupNorm uses each window independently; no batch statistics or neighboring
    windows enter its local representation. The 4x2 grid retains coarse spatial
    layout before the final learned projection.
    """
    def __init__(self, input_height: int, window_size: int, stride: int,
                 embed_dim: int) -> None:
        super().__init__()
        self.input_height = int(input_height)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.embed_dim = int(embed_dim)
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=(2, 1), padding=1, bias=False),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 2))
        self.projection = nn.Linear(64 * 4 * 2, self.embed_dim)

    def encode_windows(self, windows: torch.Tensor) -> torch.Tensor:
        """Learn one vector per already-extracted RGB window: [B,N,D]."""
        if windows.ndim != 5 or tuple(windows.shape[2:]) != (
            3, self.input_height, self.window_size
        ):
            raise ValueError("Expected windows [B,N,3,input_height,window_size]")
        batch, count = windows.shape[:2]
        maps = self.features(windows.reshape(-1, *windows.shape[2:]))
        vectors = self.projection(self.pool(maps).flatten(1))
        return vectors.reshape(batch, count, self.embed_dim)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        windows = extract_rgb_windows(image, self.window_size, self.stride)
        tokens = self.encode_windows(windows)
        # Preserve the existing patch_embedding interface [B,D,1,N]. RTL token
        # reversal is still performed by the outer encoder, exactly once.
        return tokens.transpose(1, 2).unsqueeze(2).contiguous()
