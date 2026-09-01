"""Canonical patch-based Transformer visual encoder for line-image alignment.

All ViT branches import :class:`EmbeddingModel` from this module.  It replaces
the historical ResNet/BiLSTM implementation while preserving the shared model
contract used by training and evaluation:
- contextual window embeddings are returned by default;
- ``return_local=True`` exposes pre-Transformer local patch tokens;
- ``return_grouped=True`` returns local tokens (there is no separate grouping
  stage in the pure-ViT architecture);
- ``return_ink=True`` returns one foreground/ink ratio per horizontal window.

The full-height patch projection is a ViT patch embedding, not a CNN feature
hierarchy.  Existing ViT checkpoints remain state-dict compatible because the
``vit_encoder.*`` and ``vision_norm.*`` parameter names are unchanged.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


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


def sliding_window(image: torch.Tensor, window_size: int, stride: int) -> torch.Tensor:
    """Return horizontal full-height windows as ``[B, S, C, H, W]``."""
    patches = image.unfold(dimension=3, size=int(window_size), step=int(stride))
    return patches.permute(0, 3, 1, 2, 4).contiguous()


def _denormalize_imagenet_patches(patches: torch.Tensor) -> torch.Tensor:
    if patches.ndim != 5:
        raise ValueError(
            "Expected patches with shape [B, S, C, H, W], "
            f"got {tuple(patches.shape)}"
        )
    if patches.shape[2] != 3:
        raise ValueError(
            "Ink estimation expects three RGB channels, "
            f"got C={patches.shape[2]}"
        )
    mean = patches.new_tensor(_IMAGENET_MEAN).view(1, 1, 3, 1, 1)
    std = patches.new_tensor(_IMAGENET_STD).view(1, 1, 3, 1, 1)
    return (patches.float() * std + mean).clamp(0.0, 1.0)


def _patch_background_level(gray: torch.Tensor) -> torch.Tensor:
    height, width = int(gray.shape[-2]), int(gray.shape[-1])
    border_h = max(1, int(round(height * 0.05)))
    border_w = max(1, int(round(width * 0.05)))
    border = torch.cat(
        [
            gray[..., :border_h, :].flatten(start_dim=2),
            gray[..., -border_h:, :].flatten(start_dim=2),
            gray[..., :, :border_w].flatten(start_dim=2),
            gray[..., :, -border_w:].flatten(start_dim=2),
        ],
        dim=-1,
    )
    return border.median(dim=-1).values.unsqueeze(-1).unsqueeze(-1)


def window_ink_ratio_from_patches(
    patches: torch.Tensor,
    contrast_threshold: float | None = None,
) -> torch.Tensor:
    """Estimate foreground coverage for every window and either image polarity."""
    if contrast_threshold is None:
        contrast_threshold = float(os.environ.get("INK_CONTRAST_THRESHOLD", "0.15"))
    contrast_threshold = max(0.0, min(1.0, float(contrast_threshold)))
    with torch.no_grad():
        rgb = _denormalize_imagenet_patches(patches.detach())
        gray = (
            0.2989 * rgb[:, :, 0]
            + 0.5870 * rgb[:, :, 1]
            + 0.1140 * rgb[:, :, 2]
        )
        background = _patch_background_level(gray)
        ink = (gray - background).abs().ge(contrast_threshold).float()
        return ink.mean(dim=(2, 3))


class LineWindowViT(nn.Module):
    """Convert overlapping full-height line windows into contextual tokens."""

    def __init__(self, *, input_height: int, window_size: int, stride: int, embed_dim: int, num_layers: int, num_heads: int, mlp_dim: int, dropout: float, max_tokens: int, position_base_tokens: int) -> None:
        super().__init__()
        input_height, window_size, stride, embed_dim = int(input_height), int(window_size), int(stride), int(embed_dim)
        num_layers, num_heads, mlp_dim = int(num_layers), int(num_heads), int(mlp_dim)
        max_tokens, position_base_tokens = int(max_tokens), int(position_base_tokens)
        if input_height <= 0 or window_size <= 0 or stride <= 0:
            raise ValueError("input_height, window_size, and stride must be positive")
        if num_layers <= 0 or num_heads <= 0 or mlp_dim <= 0:
            raise ValueError("ViT layers, heads, and MLP dimension must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError(f"VIT_HEADS={num_heads} must divide VECTOR_SIZE={embed_dim}")
        if max_tokens <= 0:
            raise ValueError("VIT_MAX_TOKENS must be positive")
        if position_base_tokens <= 0 or position_base_tokens > max_tokens:
            raise ValueError("VIT_POSITION_BASE_TOKENS must be positive and no larger than " f"VIT_MAX_TOKENS={max_tokens}, got {position_base_tokens}")
        self.input_height, self.window_size, self.stride = input_height, window_size, stride
        self.embed_dim, self.max_tokens, self.position_base_tokens = embed_dim, max_tokens, position_base_tokens
        self.patch_embedding = nn.Conv2d(3, embed_dim, kernel_size=(input_height, window_size), stride=(input_height, stride), padding=0, bias=True)
        self.local_norm = nn.LayerNorm(embed_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        self.input_dropout = nn.Dropout(float(dropout))
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=mlp_dim, dropout=float(dropout), activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, norm=nn.LayerNorm(embed_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.xavier_uniform_(self.patch_embedding.weight)
        if self.patch_embedding.bias is not None:
            nn.init.zeros_(self.patch_embedding.bias)

    def _position_tokens(self, count: int) -> torch.Tensor:
        count = int(count)
        if count <= 0:
            raise ValueError("position token count must be positive")
        base = self.position_embedding[:, : self.position_base_tokens]
        if count == self.position_base_tokens:
            return base
        return F.interpolate(base.transpose(1, 2), size=count, mode="linear", align_corners=False).transpose(1, 2)

    def forward(self, image: torch.Tensor, *, use_flip: bool) -> tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("ViT input must have shape [B, 3, H, W], " f"got {tuple(image.shape)}")
        if int(image.shape[2]) != self.input_height:
            raise ValueError(f"ViT expects input height {self.input_height}, got {image.shape[2]}")
        if int(image.shape[3]) < self.window_size:
            raise ValueError(f"Input width {image.shape[3]} is smaller than window size {self.window_size}")
        tokens = self.patch_embedding(image)
        if tokens.shape[2] != 1:
            raise RuntimeError("Full-height patch embedding should produce one vertical token row, " f"got shape {tuple(tokens.shape)}")
        tokens = tokens.squeeze(2).transpose(1, 2).contiguous()
        if use_flip:
            tokens = torch.flip(tokens, dims=[1])
        local_tokens = self.local_norm(tokens)
        contextual = local_tokens + self._position_tokens(local_tokens.shape[1]).to(dtype=local_tokens.dtype, device=local_tokens.device)
        contextual = self.encoder(self.input_dropout(contextual))
        return contextual, local_tokens


class EmbeddingModel(nn.Module):
    """Pure-ViT image embedder used by every ViT branch."""
    visual_encoder_type = "vit"

    def __init__(self, window_size: int = 32, stride: int = 16, vector_size: int = 128, device: str | torch.device = "cuda", use_flip: bool = False, input_height: int | None = None, vit_layers: int | None = None, vit_heads: int | None = None, vit_mlp_dim: int | None = None, vit_dropout: float | None = None, vit_max_tokens: int | None = None, vit_position_base_tokens: int | None = None, *, use_bilstm: bool = False, bilstm_layers: int | None = None, bilstm_hidden_dim: int | None = None, use_local_grouping: bool | None = False, local_group_size: int | None = None, **_ignored) -> None:
        super().__init__()
        if bool(use_bilstm):
            raise ValueError("EmbeddingModel is the pure ViT encoder on ViT branches; use_bilstm=True is not supported.")
        if bool(use_local_grouping):
            raise ValueError("EmbeddingModel is the pure ViT encoder on ViT branches; use_local_grouping=True is not supported.")
        del bilstm_layers, bilstm_hidden_dim, local_group_size
        self.device, self.window_size, self.stride, self.vector_size = device, int(window_size), int(stride), int(vector_size)
        self.input_height = int(_env_int("VIT_INPUT_HEIGHT", 128) if input_height is None else input_height)
        self.use_bilstm = False
        self.vit_layers = int(_env_int("VIT_LAYERS", 4) if vit_layers is None else vit_layers)
        self.vit_heads = int(_env_int("VIT_HEADS", 4) if vit_heads is None else vit_heads)
        self.vit_mlp_dim = int(_env_int("VIT_MLP_DIM", 512) if vit_mlp_dim is None else vit_mlp_dim)
        self.vit_dropout = float(_env_float("VIT_DROPOUT", 0.10) if vit_dropout is None else vit_dropout)
        self.vit_max_tokens = int(_env_int("VIT_MAX_TOKENS", 256) if vit_max_tokens is None else vit_max_tokens)
        self.vit_position_base_tokens = int(_env_int("VIT_POSITION_BASE_TOKENS", 63) if vit_position_base_tokens is None else vit_position_base_tokens)
        self.register_buffer("_use_flip_state", torch.tensor(1 if use_flip else 0, dtype=torch.uint8))
        self.register_buffer("_use_local_grouping_state", torch.tensor(0, dtype=torch.uint8))
        self.vit_encoder = LineWindowViT(input_height=self.input_height, window_size=self.window_size, stride=self.stride, embed_dim=self.vector_size, num_layers=self.vit_layers, num_heads=self.vit_heads, mlp_dim=self.vit_mlp_dim, dropout=self.vit_dropout, max_tokens=self.vit_max_tokens, position_base_tokens=self.vit_position_base_tokens).to(device)
        self.vision_norm = nn.LayerNorm(self.vector_size).to(device)

    @property
    def use_flip(self) -> bool:
        return bool(int(self._use_flip_state.item()))

    @property
    def use_local_grouping(self) -> bool:
        return False

    def forward(self, image: torch.Tensor, show_dims: bool = False, return_local: bool = False, return_ink: bool = False, return_grouped: bool = False):
        contextual, local = self.vit_encoder(image, use_flip=self.use_flip)
        ink_ratio = None
        if return_ink:
            patches = sliding_window(image, self.window_size, self.stride)
            if self.use_flip:
                patches = torch.flip(patches, dims=[1])
            ink_ratio = window_ink_ratio_from_patches(patches)
            if ink_ratio.shape[1] != local.shape[1]:
                raise RuntimeError("ViT token count and ink-window count differ: " f"{local.shape[1]} != {ink_ratio.shape[1]}")
        contextual, local = self.vision_norm(contextual), self.vision_norm(local)
        grouped = local
        if show_dims:
            print("image embeddings: " f"encoder=vit contextual={tuple(contextual.shape)} " f"local={tuple(local.shape)} flip={self.use_flip} " f"position_base_tokens={self.vit_position_base_tokens}", flush=True)
        outputs = [contextual]
        if return_local: outputs.append(local)
        if return_grouped: outputs.append(grouped)
        if return_ink: outputs.append(ink_ratio)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def model_config(self) -> dict:
        return {"visual_encoder_type": "vit", "use_bilstm": False, "use_local_window_grouping": False, "vit_input_height": self.input_height, "vit_layers": self.vit_layers, "vit_heads": self.vit_heads, "vit_mlp_dim": self.vit_mlp_dim, "vit_dropout": self.vit_dropout, "vit_max_tokens": self.vit_max_tokens, "vit_position_base_tokens": self.vit_position_base_tokens}


ViTEmbeddingModel = EmbeddingModel


def build_vit_from_environment(*, window_size, stride, vector_size, device, use_flip):
    return EmbeddingModel(window_size=window_size, stride=stride, vector_size=vector_size, device=device, use_flip=use_flip, input_height=_env_int("VIT_INPUT_HEIGHT", 128), vit_layers=_env_int("VIT_LAYERS", 4), vit_heads=_env_int("VIT_HEADS", 4), vit_mlp_dim=_env_int("VIT_MLP_DIM", 512), vit_dropout=_env_float("VIT_DROPOUT", 0.10), vit_max_tokens=_env_int("VIT_MAX_TOKENS", 256), vit_position_base_tokens=_env_int("VIT_POSITION_BASE_TOKENS", 63))


def prepare_vit_model(model: EmbeddingModel) -> EmbeddingModel:
    global window_ink_ratio_from_patches
    from training_optimizations import fast_window_ink_ratio_from_patches
    window_ink_ratio_from_patches = fast_window_ink_ratio_from_patches
    if _env_flag("TORCH_COMPILE_VISUAL", False):
        if not hasattr(torch, "compile"):
            print("torch.compile unavailable; continuing without it", flush=True)
        else:
            try:
                model.vit_encoder = torch.compile(model.vit_encoder, mode=os.environ.get("TORCH_COMPILE_MODE", "reduce-overhead"), dynamic=False)
                print("compiled visual ViT with torch.compile", flush=True)
            except Exception as exc:
                print(f"torch.compile visual ViT failed: {exc}", flush=True)
    return model
