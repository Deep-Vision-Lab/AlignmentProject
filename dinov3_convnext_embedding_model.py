"""DINOv3 ConvNeXt window encoder compatible with AlignmentProject training.

The official Meta DINOv3 ConvNeXt-Tiny backbone is applied to the same overlapping
full-height line windows used by the CNN baseline. Its 768-D global feature for each
window is projected into the project's ``vector_size`` embedding space, then the
shared optional local-grouping/BiLSTM sequence layers may be applied.

HPC jobs are intentionally offline: clone the official DINOv3 repository once and
store the authorized ConvNeXt-Tiny weights locally, then set ``DINOV3_REPO_DIR`` and
``DINOV3_WEIGHTS``. No network access occurs inside training.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn

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
            "Set DINOV3_ALLOW_RANDOM_INIT=1 only for constructor/smoke tests."
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
    ) -> None:
        super().__init__()
        self.device = device
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.vector_size = int(vector_size)
        self.use_bilstm = bool(use_bilstm)
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
        self.local_group_encoder = LocalWindowGrouping(
            embed_dim=self.vector_size,
            group_size=int(local_group_size),
        ).to(device)
        self.sequence_encoder = None
        if self.use_bilstm:
            self.sequence_encoder = BiLSTMEncoder(
                embed_dim=self.vector_size,
                hidden_dim=bilstm_hidden_dim,
                lstm_layers=int(bilstm_layers),
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
                f"freeze_backbone={self.freeze_backbone} bilstm={self.use_bilstm}",
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
    )


def prepare_dinov3_model(model: DINOv3ConvNeXtEmbeddingModel):
    """Reuse the optimized foreground estimator; compilation stays opt-in."""
    import embeddingModel as embedding_model_module
    from training_optimizations import fast_window_ink_ratio_from_patches

    embedding_model_module.window_ink_ratio_from_patches = fast_window_ink_ratio_from_patches
    globals()["window_ink_ratio_from_patches"] = fast_window_ink_ratio_from_patches
    return model
