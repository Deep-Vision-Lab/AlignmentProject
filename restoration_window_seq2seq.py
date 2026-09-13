"""Restoration-pretrained local window encoder for Arabic manuscript lines.

The module treats a manuscript line as a sequence of overlapping 128x32 windows.
Each window is encoded by the SAME convolutional encoder, producing one 128-D
local token.  The decoder reconstructs the COMPLETE original RGB window.

Downsampling is intentionally anisotropic: height is reduced aggressively while
width is preserved for longer, because Arabic dots, short strokes and neighboring
letters are easily destroyed by early horizontal downsampling.  Sequence context
is added later by the branch Transformer; this module remains purely local.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, stride=1, groups=8):
        super().__init__()
        groups = max(1, min(int(groups), int(out_channels)))
        while out_channels % groups:
            groups -= 1
        stride_value = (
            tuple(int(value) for value in stride)
            if isinstance(stride, (tuple, list))
            else int(stride)
        )
        self.block = nn.Sequential(
            nn.Conv2d(
                int(in_channels),
                int(out_channels),
                kernel_size=3,
                stride=stride_value,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, int(out_channels)),
            nn.GELU(),
            nn.Conv2d(
                int(out_channels),
                int(out_channels),
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(groups, int(out_channels)),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class WindowSequenceCNNEncoder(nn.Module):
    """Encode all overlapping full-height windows with a shared local CNN.

    Input:
        [B,3,128,line_width]
    Output:
        [B,D,1,T]

    The output layout intentionally matches the historical Conv2d patch
    projection so the rest of the branch can keep the same interface.
    """

    def __init__(
        self,
        *,
        input_height=128,
        window_size=32,
        stride=16,
        embed_dim=128,
        base_channels=32,
    ):
        super().__init__()
        if int(input_height) != 128 or int(window_size) != 32:
            raise ValueError(
                "WindowSequenceCNNEncoder currently expects 128x32 windows"
            )
        self.input_height = int(input_height)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.embed_dim = int(embed_dim)
        c = max(16, int(base_channels))

        # 128x32 -> 64x32 -> 32x32 -> 16x16 -> 8x8.
        # Width is not reduced until the third block, preserving fine Arabic
        # stroke/dot geometry much longer than the historical 128x32 -> 8x2 path.
        self.encoder = nn.Sequential(
            ConvBlock(3, c, stride=1, groups=4),                 # 128x32
            ConvBlock(c, c * 2, stride=(2, 1), groups=8),        # 64x32
            ConvBlock(c * 2, c * 3, stride=(2, 1), groups=8),    # 32x32
            ConvBlock(c * 3, c * 4, stride=(2, 2), groups=8),    # 16x16
            ConvBlock(c * 4, self.embed_dim, stride=(2, 2), groups=8), # 8x8
        )
        # Never global-average the spatial feature map. Flattening keeps WHERE a
        # stroke occurred before the 128-D bottleneck.
        self.spatial_height = 8
        self.spatial_width = 8
        self.to_token = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(
                self.embed_dim * self.spatial_height * self.spatial_width,
                self.embed_dim,
            ),
            nn.LayerNorm(self.embed_dim),
        )

    def extract_windows(self, line: torch.Tensor) -> torch.Tensor:
        if line.ndim != 4 or line.shape[1] != 3:
            raise ValueError(f"Expected [B,3,H,W], got {tuple(line.shape)}")
        if int(line.shape[2]) != self.input_height:
            raise ValueError(
                f"Expected line height {self.input_height}, got {line.shape[2]}"
            )
        patches = line.unfold(
            dimension=3,
            size=self.window_size,
            step=self.stride,
        )
        # [B,C,H,T,W] -> [B,T,C,H,W]
        return patches.permute(0, 3, 1, 2, 4).contiguous()

    def _spatial_features(self, line: torch.Tensor):
        patches = self.extract_windows(line)
        batch, count, channels, height, width = patches.shape
        flat = patches.reshape(batch * count, channels, height, width)
        spatial = self.encoder(flat)
        expected = (
            batch * count,
            self.embed_dim,
            self.spatial_height,
            self.spatial_width,
        )
        if tuple(spatial.shape) != expected:
            raise RuntimeError(
                f"Unexpected local CNN feature map {tuple(spatial.shape)}; "
                f"expected {expected}"
            )
        return spatial, batch, count

    def forward(self, line: torch.Tensor) -> torch.Tensor:
        spatial, batch, count = self._spatial_features(line)
        encoded = self.to_token(spatial)
        encoded = encoded.reshape(batch, count, self.embed_dim)
        # Historical patch_embedding contract: [B,D,1,T]
        return encoded.transpose(1, 2).unsqueeze(2).contiguous()

    def spatial_vectors(self, line: torch.Tensor, vectors_per_window: int = 4):
        """Optional K spatial vectors/window for the point-11 ablation.

        This is intentionally NOT the default alignment representation. It gives
        us a controlled experiment if one 128-D vector cannot represent windows
        containing several neighboring letters.
        """
        k = max(1, int(vectors_per_window))
        spatial, batch, count = self._spatial_features(line)
        pooled = torch.nn.functional.adaptive_avg_pool2d(spatial, (1, k))
        pooled = pooled.squeeze(2).transpose(1, 2).contiguous()
        pooled = pooled.reshape(batch, count, k, self.embed_dim)
        return torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)


class WindowSequenceStrokeDecoder(nn.Module):
    """Decode each local token into the complete original 128x32 RGB window."""

    def __init__(self, dim=128, output_height=128, output_width=32, channels=128):
        super().__init__()
        if int(output_height) != 128 or int(output_width) != 32:
            raise ValueError("WindowSequenceStrokeDecoder expects 128x32 output")
        self.output_height = int(output_height)
        self.output_width = int(output_width)
        channels = max(64, int(channels))
        self.channels = channels
        self.project = nn.Linear(int(dim), channels * 8 * 2)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1), # 16x4
            nn.GroupNorm(8, channels),
            nn.GELU(),
            nn.ConvTranspose2d(channels, channels // 2, 4, stride=2, padding=1), # 32x8
            nn.GroupNorm(8, channels // 2),
            nn.GELU(),
            nn.ConvTranspose2d(channels // 2, channels // 4, 4, stride=2, padding=1), # 64x16
            nn.GroupNorm(4, channels // 4),
            nn.GELU(),
            nn.ConvTranspose2d(channels // 4, 3, 4, stride=2, padding=1), # RGB 128x32
            nn.Sigmoid(),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError("Expected primitive tokens [B,T,D]")
        batch, count, _ = tokens.shape
        x = self.project(tokens)
        x = x.reshape(batch * count, self.channels, 8, 2)
        x = self.decoder(x)
        return x.reshape(
            batch, count, 3, self.output_height, self.output_width
        )
