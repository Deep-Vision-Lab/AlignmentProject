"""Shared ResNet-18 encoder for overlapping full-height line windows."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18WindowEncoder(nn.Module):
    """Encode each grayscale/RGB 128x32 physical window with shared ResNet-18.

    Input:
        [B, C, 128, line_width], C in {1,3}
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
        local_files_only: bool = True,
        input_channels: int = 3,
    ) -> None:
        super().__init__()
        if int(input_height) != 128 or int(window_size) != 32:
            raise ValueError("ResNet18WindowEncoder expects 128x32 windows")
        self.input_height = int(input_height)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.embed_dim = int(embed_dim)
        self.pretrained = bool(pretrained)
        self.local_files_only = bool(local_files_only)
        self.input_channels = int(input_channels)
        if self.input_channels not in {1, 3}:
            raise ValueError("ResNet18WindowEncoder input_channels must be 1 or 3")

        if self.pretrained and self.local_files_only:
            weights = ResNet18_Weights.DEFAULT
            filename = Path(urlparse(weights.url).path).name
            checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / filename
            if not checkpoint.is_file():
                raise FileNotFoundError(
                    "Missing cached ResNet-18 ImageNet weights at "
                    f"{checkpoint}. Run: python scripts/cache_pretrained_models.py "
                    "before submitting the offline SLURM job."
                )
            self.backbone = resnet18(weights=None)
            state = torch.load(checkpoint, map_location="cpu")
            self.backbone.load_state_dict(state, strict=True)
        else:
            weights = ResNet18_Weights.DEFAULT if self.pretrained else None
            self.backbone = resnet18(weights=weights)
        if self.input_channels == 1:
            old_conv = self.backbone.conv1
            gray_conv = nn.Conv2d(
                1,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                # Keep ImageNet initialization by collapsing RGB filters into
                # one luminance-like channel. BatchNorm absorbs the modest
                # scale change during fine-tuning.
                gray_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            self.backbone.conv1 = gray_conv

        self.backbone.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Linear(512, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

    def extract_windows(self, line: torch.Tensor) -> torch.Tensor:
        if line.ndim != 4 or int(line.shape[1]) != self.input_channels:
            raise ValueError(
                f"Expected [B,{self.input_channels},H,W], got {tuple(line.shape)}"
            )
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

    def _encode_flat_windows(self, flat: torch.Tensor) -> torch.Tensor:
        features = self.backbone(flat)
        if features.ndim != 2 or int(features.shape[1]) != 512:
            raise RuntimeError(
                f"ResNet-18 must return [N,512], got {tuple(features.shape)}"
            )
        return self.projection(features)

    def forward_packed(
        self,
        line: torch.Tensor,
        token_valid: torch.Tensor,
        *,
        use_flip: bool = False,
    ):
        """Encode only content-overlapping windows, then pad token sequences.

        token_valid is in physical left-to-right window order. Completely
        artificial side-padding windows are never passed through ResNet-18.
        Internal blank gaps remain selected because the mask spans the complete
        content rectangle.
        """
        patches = self.extract_windows(line)
        batch, count, channels, height, width = patches.shape
        if token_valid.shape != (batch, count):
            raise ValueError(
                "token_valid shape must match extracted windows: "
                f"{tuple(token_valid.shape)} != {(batch, count)}"
            )
        token_valid = token_valid.to(device=patches.device, dtype=torch.bool)
        lengths = token_valid.sum(dim=1).to(dtype=torch.long)
        if int(lengths.max().item()) <= 0:
            raise RuntimeError("No valid content windows were detected")

        selected = patches[token_valid]
        encoded = self._encode_flat_windows(
            selected.reshape(-1, channels, height, width)
        )

        max_length = int(lengths.max().item())
        packed = encoded.new_zeros((batch, max_length, self.embed_dim))
        packed_valid = torch.zeros(
            (batch, max_length), dtype=torch.bool, device=encoded.device
        )
        cursor = 0
        for batch_index in range(batch):
            length = int(lengths[batch_index].item())
            current = encoded[cursor : cursor + length]
            cursor += length
            if use_flip:
                current = torch.flip(current, dims=[0])
            packed[batch_index, :length] = current
            packed_valid[batch_index, :length] = True

        return (
            packed.transpose(1, 2).unsqueeze(2).contiguous(),
            packed_valid,
        )

    def forward(self, line: torch.Tensor) -> torch.Tensor:
        patches = self.extract_windows(line)
        batch, count, channels, height, width = patches.shape
        flat = patches.reshape(batch * count, channels, height, width)
        tokens = self._encode_flat_windows(flat)
        tokens = tokens.reshape(batch, count, self.embed_dim)
        return tokens.transpose(1, 2).unsqueeze(2).contiguous()
