"""Patch-based Transformer visual encoder for line-image alignment.

The encoder preserves a fixed full-height patch width while allowing denser
horizontal overlap. A pretrained 63-token positional table can therefore be
interpolated to the 125-token stride-8 sequence without exposing positional
slots that never received gradients during pretraining.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from embeddingModel import sliding_window, window_ink_ratio_from_patches


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class LineWindowViT(nn.Module):
    """Convert overlapping full-height line windows into contextual tokens."""

    def __init__(
        self,
        *,
        input_height: int,
        window_size: int,
        stride: int,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
        max_tokens: int,
        position_base_tokens: int,
    ) -> None:
        super().__init__()
        input_height = int(input_height)
        window_size = int(window_size)
        stride = int(stride)
        embed_dim = int(embed_dim)
        num_layers = int(num_layers)
        num_heads = int(num_heads)
        mlp_dim = int(mlp_dim)
        max_tokens = int(max_tokens)
        position_base_tokens = int(position_base_tokens)

        if input_height <= 0 or window_size <= 0 or stride <= 0:
            raise ValueError("input_height, window_size, and stride must be positive")
        if num_layers <= 0 or num_heads <= 0 or mlp_dim <= 0:
            raise ValueError("ViT layers, heads, and MLP dimension must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"VIT_HEADS={num_heads} must divide VECTOR_SIZE={embed_dim}"
            )
        if max_tokens <= 0:
            raise ValueError("VIT_MAX_TOKENS must be positive")
        if position_base_tokens <= 0 or position_base_tokens > max_tokens:
            raise ValueError(
                "VIT_POSITION_BASE_TOKENS must be positive and no larger than "
                f"VIT_MAX_TOKENS={max_tokens}, got {position_base_tokens}"
            )

        self.input_height = input_height
        self.window_size = window_size
        self.stride = stride
        self.embed_dim = embed_dim
        self.max_tokens = max_tokens
        self.position_base_tokens = position_base_tokens

        # This is the ViT patch embedding. It is a learned linear projection of
        # every full-height 3 x H x window_size patch, not a CNN feature hierarchy.
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=embed_dim,
            kernel_size=(input_height, window_size),
            stride=(input_height, stride),
            padding=0,
            bias=True,
        )
        self.local_norm = nn.LayerNorm(embed_dim)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, max_tokens, embed_dim)
        )
        self.input_dropout = nn.Dropout(float(dropout))

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)

    def _position_tokens(self, count: int) -> torch.Tensor:
        """Resize only the positional table learned by the pretrained sequence."""
        count = int(count)
        if count <= 0:
            raise ValueError("position token count must be positive")

        # The pretrained stride-16 model optimized positions 0..62. Slots beyond
        # that range still exist in the state dict but did not receive gradients.
        # Interpolate the trained base table instead of slicing untrained slots.
        base = self.position_embedding[:, : self.position_base_tokens]
        if count == self.position_base_tokens:
            return base
        return F.interpolate(
            base.transpose(1, 2),
            size=count,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, image: torch.Tensor, *, use_flip: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(
                "ViT input must have shape [B, 3, H, W], "
                f"got {tuple(image.shape)}"
            )
        if int(image.shape[2]) != self.input_height:
            raise ValueError(
                f"ViT expects input height {self.input_height}, got {image.shape[2]}"
            )
        if int(image.shape[3]) < self.window_size:
            raise ValueError(
                f"Input width {image.shape[3]} is smaller than window size {self.window_size}"
            )

        tokens = self.patch_embedding(image)
        if tokens.shape[2] != 1:
            raise RuntimeError(
                "Full-height patch embedding should produce one vertical token row, "
                f"got shape {tuple(tokens.shape)}"
            )
        tokens = tokens.squeeze(2).transpose(1, 2).contiguous()
        if use_flip:
            tokens = torch.flip(tokens, dims=[1])

        local_tokens = self.local_norm(tokens)
        contextual = local_tokens + self._position_tokens(local_tokens.shape[1]).to(
            dtype=local_tokens.dtype,
            device=local_tokens.device,
        )
        contextual = self.encoder(self.input_dropout(contextual))
        return contextual, local_tokens


class ViTEmbeddingModel(nn.Module):
    """Drop-in replacement for EmbeddingModel using only patch projection + ViT."""

    visual_encoder_type = "vit"

    def __init__(
        self,
        window_size: int = 32,
        stride: int = 16,
        vector_size: int = 128,
        device: str | torch.device = "cuda",
        use_flip: bool = False,
        input_height: int = 128,
        vit_layers: int = 4,
        vit_heads: int = 4,
        vit_mlp_dim: int = 512,
        vit_dropout: float = 0.10,
        vit_max_tokens: int = 256,
        vit_position_base_tokens: int = 63,
    ) -> None:
        super().__init__()
        self.device = device
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.vector_size = int(vector_size)
        self.input_height = int(input_height)
        self.use_bilstm = False

        self.register_buffer(
            "_use_flip_state",
            torch.tensor(1 if use_flip else 0, dtype=torch.uint8),
        )
        # Keep the compatibility buffer expected by shared evaluation utilities.
        # Grouped features intentionally equal local tokens in this pure-ViT ablation.
        self.register_buffer(
            "_use_local_grouping_state",
            torch.tensor(0, dtype=torch.uint8),
        )

        self.vit_encoder = LineWindowViT(
            input_height=self.input_height,
            window_size=self.window_size,
            stride=self.stride,
            embed_dim=self.vector_size,
            num_layers=int(vit_layers),
            num_heads=int(vit_heads),
            mlp_dim=int(vit_mlp_dim),
            dropout=float(vit_dropout),
            max_tokens=int(vit_max_tokens),
            position_base_tokens=int(vit_position_base_tokens),
        ).to(device)
        self.vision_norm = nn.LayerNorm(self.vector_size).to(device)

        self.vit_layers = int(vit_layers)
        self.vit_heads = int(vit_heads)
        self.vit_mlp_dim = int(vit_mlp_dim)
        self.vit_dropout = float(vit_dropout)
        self.vit_max_tokens = int(vit_max_tokens)
        self.vit_position_base_tokens = int(vit_position_base_tokens)

    @property
    def use_flip(self) -> bool:
        return bool(int(self._use_flip_state.item()))

    @property
    def use_local_grouping(self) -> bool:
        return False

    def forward(
        self,
        image: torch.Tensor,
        show_dims: bool = False,
        return_local: bool = False,
        return_ink: bool = False,
        return_grouped: bool = False,
    ):
        contextual, local = self.vit_encoder(image, use_flip=self.use_flip)
        ink_ratio = None
        if return_ink:
            patches = sliding_window(image, self.window_size, self.stride)
            if self.use_flip:
                patches = torch.flip(patches, dims=[1])
            ink_ratio = window_ink_ratio_from_patches(patches)
            if ink_ratio.shape[1] != local.shape[1]:
                raise RuntimeError(
                    "ViT token count and ink-window count differ: "
                    f"{local.shape[1]} != {ink_ratio.shape[1]}"
                )

        contextual = self.vision_norm(contextual)
        local = self.vision_norm(local)
        grouped = local

        if show_dims:
            print(
                "image embeddings: "
                f"encoder=vit contextual={tuple(contextual.shape)} "
                f"local={tuple(local.shape)} flip={self.use_flip} "
                f"position_base_tokens={self.vit_position_base_tokens}",
                flush=True,
            )

        outputs = [contextual]
        if return_local:
            outputs.append(local)
        if return_grouped:
            outputs.append(grouped)
        if return_ink:
            outputs.append(ink_ratio)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def model_config(self) -> dict:
        return {
            "visual_encoder_type": "vit",
            "use_bilstm": False,
            "use_local_window_grouping": False,
            "vit_input_height": self.input_height,
            "vit_layers": self.vit_layers,
            "vit_heads": self.vit_heads,
            "vit_mlp_dim": self.vit_mlp_dim,
            "vit_dropout": self.vit_dropout,
            "vit_max_tokens": self.vit_max_tokens,
            "vit_position_base_tokens": self.vit_position_base_tokens,
        }


def build_vit_from_environment(*, window_size, stride, vector_size, device, use_flip):
    return ViTEmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=use_flip,
        input_height=_env_int("VIT_INPUT_HEIGHT", 128),
        vit_layers=_env_int("VIT_LAYERS", 4),
        vit_heads=_env_int("VIT_HEADS", 4),
        vit_mlp_dim=_env_int("VIT_MLP_DIM", 512),
        vit_dropout=_env_float("VIT_DROPOUT", 0.10),
        vit_max_tokens=_env_int("VIT_MAX_TOKENS", 256),
        vit_position_base_tokens=_env_int("VIT_POSITION_BASE_TOKENS", 63),
    )


def prepare_vit_model(model: ViTEmbeddingModel) -> ViTEmbeddingModel:
    """Install optimized ink estimation and optional torch.compile for ViT."""
    import embeddingModel as embedding_model_module
    from training_optimizations import fast_window_ink_ratio_from_patches

    embedding_model_module.window_ink_ratio_from_patches = (
        fast_window_ink_ratio_from_patches
    )
    # The function was imported into this module, so update this module global too.
    globals()["window_ink_ratio_from_patches"] = fast_window_ink_ratio_from_patches

    if _env_flag("TORCH_COMPILE_VISUAL", False):
        if not hasattr(torch, "compile"):
            print("torch.compile unavailable; continuing without it", flush=True)
        else:
            try:
                model.vit_encoder = torch.compile(
                    model.vit_encoder,
                    mode=os.environ.get("TORCH_COMPILE_MODE", "reduce-overhead"),
                    dynamic=False,
                )
                print("compiled visual ViT with torch.compile", flush=True)
            except Exception as exc:
                print(f"torch.compile visual ViT failed: {exc}", flush=True)
    return model
