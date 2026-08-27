"""DINOv3 ConvNeXt window encoder compatible with AlignmentProject training.

The official Meta DINOv3 ConvNeXt-Tiny backbone is applied independently to the
same overlapping full-height line windows used by the fixed63 recipe. Its global
feature for each window is projected to ``vector_size``. Sequence context can then
be supplied by one of three explicit modes:

- ``bilstm``: historical DINO branch, preserving old checkpoint compatibility;
- ``transformer``: global self-attention across the full 63-window line;
- ``none``: projected DINO windows only.

The transformer mode deliberately bypasses three-window fusion so it tests whether
global attention improves real/synthetic alignment instead of reusing the CNN+
BiLSTM sequence head under a different backbone.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from embeddingModel import (
    BiLSTMEncoder,
    LocalWindowGrouping,
    sliding_window,
    window_ink_ratio_from_patches,
)


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


def _number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _normalize_sequence_mode(value: str | None, use_bilstm: bool) -> str:
    raw = "" if value is None else str(value).strip().lower().replace("-", "_")
    if raw in {"", "auto", "legacy"}:
        return "bilstm" if bool(use_bilstm) else "none"
    aliases = {
        "lstm": "bilstm",
        "bi_lstm": "bilstm",
        "attention": "transformer",
        "self_attention": "transformer",
        "off": "none",
        "identity": "none",
    }
    raw = aliases.get(raw, raw)
    if raw not in {"bilstm", "transformer", "none"}:
        raise ValueError(
            "DINOV3_SEQUENCE_MODE must be bilstm, transformer, or none; "
            f"got {value!r}."
        )
    return raw


def _load_dinov3_convnext_tiny() -> nn.Module:
    repo_raw = os.environ.get("DINOV3_REPO_DIR", "").strip()
    weights_raw = os.environ.get("DINOV3_WEIGHTS", "").strip()
    if not repo_raw:
        raise RuntimeError(
            "DINOV3_REPO_DIR is required. Clone Meta's official facebookresearch/"
            "dinov3 repository locally and point DINOV3_REPO_DIR to that directory."
        )
    repo = Path(repo_raw).expanduser().resolve()
    if not (repo / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOV3_REPO_DIR has no hubconf.py: {repo}")

    weights = None
    if weights_raw:
        weights_path = Path(weights_raw).expanduser().resolve()
        if not weights_path.is_file():
            raise FileNotFoundError(f"DINOV3_WEIGHTS does not exist: {weights_path}")
        weights = str(weights_path)
    elif not _flag("DINOV3_ALLOW_RANDOM_INIT", False):
        raise RuntimeError(
            "DINOV3_WEIGHTS is required for the DINOv3 benchmark. Obtain the "
            "authorized ConvNeXt-Tiny LVD-1689M checkpoint from Meta and store it locally. "
            "Set DINOV3_ALLOW_RANDOM_INIT=1 only when a full project checkpoint will "
            "immediately overwrite the constructor state or for smoke tests."
        )

    model = torch.hub.load(
        str(repo),
        "dinov3_convnext_tiny",
        source="local",
        weights=weights,
    )
    if not hasattr(model, "embed_dim"):
        raise RuntimeError("Loaded DINOv3 ConvNeXt model has no embed_dim attribute.")
    return model


class WindowTransformerContext(nn.Module):
    """Self-attention over projected DINO window tokens with fixed63 positions."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_dim: int = 512,
        dropout: float = 0.10,
        max_tokens: int = 256,
        position_base_tokens: int = 63,
    ) -> None:
        super().__init__()
        embed_dim = int(embed_dim)
        num_layers = int(num_layers)
        num_heads = int(num_heads)
        mlp_dim = int(mlp_dim)
        max_tokens = int(max_tokens)
        position_base_tokens = int(position_base_tokens)
        if num_layers <= 0 or num_heads <= 0 or mlp_dim <= 0:
            raise ValueError("DINO transformer layers/heads/MLP must be positive")
        if embed_dim % num_heads:
            raise ValueError(
                f"DINOV3_TRANSFORMER_HEADS={num_heads} must divide VECTOR_SIZE={embed_dim}"
            )
        if position_base_tokens <= 0 or position_base_tokens > max_tokens:
            raise ValueError(
                "DINOV3_TRANSFORMER_POSITION_BASE_TOKENS must be positive and <= "
                f"max_tokens={max_tokens}; got {position_base_tokens}."
            )

        self.embed_dim = embed_dim
        self.max_tokens = max_tokens
        self.position_base_tokens = position_base_tokens
        self.input_norm = nn.LayerNorm(embed_dim)
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
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def _position_tokens(self, count: int) -> torch.Tensor:
        count = int(count)
        if count <= 0 or count > self.max_tokens:
            raise ValueError(
                f"DINO transformer token count must be in [1,{self.max_tokens}], got {count}"
            )
        base = self.position_embedding[:, : self.position_base_tokens]
        if count == self.position_base_tokens:
            return base
        return F.interpolate(
            base.transpose(1, 2),
            size=count,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        normalized = self.input_norm(windows)
        positions = self._position_tokens(normalized.shape[1]).to(
            dtype=normalized.dtype,
            device=normalized.device,
        )
        return self.encoder(self.input_dropout(normalized + positions))


class DINOv3ConvNeXtEmbeddingModel(nn.Module):
    visual_encoder_type = "dinov3_convnext"
    CNN_CHUNK_SIZE = 256

    def __init__(
        self,
        *,
        window_size: int = 32,
        stride: int = 16,
        vector_size: int = 128,
        device: str | torch.device = "cuda",
        use_flip: bool = False,
        use_bilstm: bool = True,
        bilstm_layers: int = 2,
        bilstm_hidden_dim: int | None = None,
        use_local_grouping: bool = True,
        local_group_size: int = 3,
        sequence_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.vector_size = int(vector_size)
        self.sequence_mode = _normalize_sequence_mode(sequence_mode, use_bilstm)
        self.use_bilstm = self.sequence_mode == "bilstm"
        self.freeze_backbone = _flag("DINOV3_FREEZE_BACKBONE", True)
        self.CNN_CHUNK_SIZE = _integer("DINOV3_WINDOW_CHUNK_SIZE", 256)

        self.register_buffer(
            "_use_flip_state",
            torch.tensor(1 if use_flip else 0, dtype=torch.uint8),
        )
        self.register_buffer(
            "_use_local_grouping_state",
            torch.tensor(1 if use_local_grouping else 0, dtype=torch.uint8),
        )

        self.dinov3_encoder = _load_dinov3_convnext_tiny().to(device)
        feature_dim = int(self.dinov3_encoder.embed_dim)
        self.feature_proj = nn.Linear(feature_dim, self.vector_size).to(device)

        # Keep this module constructed in every mode for strict compatibility with
        # historical DINO checkpoints. Freeze it when disabled so DDP does not see
        # trainable parameters that never participate in forward().
        self.local_group_encoder = LocalWindowGrouping(
            embed_dim=self.vector_size,
            group_size=int(local_group_size),
        ).to(device)
        if not self.use_local_grouping:
            self.local_group_encoder.requires_grad_(False)

        self.sequence_encoder = None
        if self.sequence_mode == "bilstm":
            self.sequence_encoder = BiLSTMEncoder(
                embed_dim=self.vector_size,
                hidden_dim=bilstm_hidden_dim,
                lstm_layers=int(bilstm_layers),
            ).to(device)
        elif self.sequence_mode == "transformer":
            if self.use_local_grouping:
                raise ValueError(
                    "DINOV3_SEQUENCE_MODE=transformer must use "
                    "USE_LOCAL_WINDOW_GROUPING=0. Global attention should receive the "
                    "projected windows directly, without three-window fusion."
                )
            self.sequence_encoder = WindowTransformerContext(
                self.vector_size,
                num_layers=_integer("DINOV3_TRANSFORMER_LAYERS", 4),
                num_heads=_integer("DINOV3_TRANSFORMER_HEADS", 4),
                mlp_dim=_integer("DINOV3_TRANSFORMER_MLP_DIM", 512),
                dropout=_number("DINOV3_TRANSFORMER_DROPOUT", 0.10),
                max_tokens=_integer("DINOV3_TRANSFORMER_MAX_TOKENS", 256),
                position_base_tokens=_integer(
                    "DINOV3_TRANSFORMER_POSITION_BASE_TOKENS", 63
                ),
            ).to(device)

        self.vision_norm = nn.LayerNorm(self.vector_size).to(device)

        if self.freeze_backbone:
            for parameter in self.dinov3_encoder.parameters():
                parameter.requires_grad_(False)
            self.dinov3_encoder.eval()

    @property
    def use_flip(self) -> bool:
        return bool(int(self._use_flip_state.item()))

    @property
    def use_local_grouping(self) -> bool:
        return bool(int(self._use_local_grouping_state.item()))

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.dinov3_encoder.eval()
        return self

    def _encode_chunk(self, chunk: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                features = self.dinov3_encoder(chunk)
        else:
            features = self.dinov3_encoder(chunk)
        if features.ndim != 2:
            raise RuntimeError(
                "DINOv3 ConvNeXt-Tiny should return [N, D] global features, "
                f"got {tuple(features.shape)}"
            )
        return self.feature_proj(features.float())

    def _process_patches(self, patches: torch.Tensor) -> torch.Tensor:
        batch_size, windows_num, channels, height, width = patches.shape
        flat = patches.reshape(batch_size * windows_num, channels, height, width)
        chunks = []
        for start in range(0, flat.shape[0], self.CNN_CHUNK_SIZE):
            chunks.append(self._encode_chunk(flat[start : start + self.CNN_CHUNK_SIZE]))
        encoded = torch.cat(chunks, dim=0)
        return encoded.view(batch_size, windows_num, self.vector_size)

    def forward(
        self,
        image: torch.Tensor,
        show_dims: bool = False,
        return_local: bool = False,
        return_ink: bool = False,
        return_grouped: bool = False,
    ):
        patches = sliding_window(image, self.window_size, self.stride)
        if self.use_flip:
            patches = torch.flip(patches, dims=[1])
        ink_ratio = window_ink_ratio_from_patches(patches) if return_ink else None

        # ``local`` is always the direct per-window DINO representation. The
        # local hard-negative loss therefore keeps the same semantics in all
        # sequence modes.
        local = self._process_patches(patches)
        grouped = local
        if self.use_local_grouping:
            grouped = self.local_group_encoder(grouped)

        contextual = grouped
        if self.sequence_encoder is not None:
            contextual = self.sequence_encoder(contextual)

        contextual = self.vision_norm(contextual)
        local = self.vision_norm(local)
        grouped = self.vision_norm(grouped)

        if show_dims:
            print(
                "image embeddings: encoder=dinov3_convnext_tiny "
                f"contextual={tuple(contextual.shape)} local={tuple(local.shape)} "
                f"freeze_backbone={self.freeze_backbone} "
                f"sequence_mode={self.sequence_mode} grouping={self.use_local_grouping}",
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


def build_dinov3_from_environment(
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
    sequence_mode=None,
):
    return DINOv3ConvNeXtEmbeddingModel(
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
        sequence_mode=sequence_mode,
    )


def prepare_dinov3_model(model: DINOv3ConvNeXtEmbeddingModel):
    """Reuse the optimized foreground estimator; compilation stays opt-in."""
    import embeddingModel as embedding_model_module
    from training_optimizations import fast_window_ink_ratio_from_patches

    embedding_model_module.window_ink_ratio_from_patches = fast_window_ink_ratio_from_patches
    globals()["window_ink_ratio_from_patches"] = fast_window_ink_ratio_from_patches
    return model
