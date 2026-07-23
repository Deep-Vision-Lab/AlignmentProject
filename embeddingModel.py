import os

import torch
import torch.nn as nn
import torchvision


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def sliding_window(image, window_size, stride):
    patches = image.unfold(dimension=3, size=window_size, step=stride)
    return patches.permute(0, 3, 1, 2, 4).contiguous()


def _denormalize_imagenet_patches(patches):
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


def _patch_background_level(gray):
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


def window_ink_ratio_from_patches(patches, contrast_threshold=None):
    """Estimate foreground coverage for either image polarity."""
    if contrast_threshold is None:
        contrast_threshold = float(
            os.environ.get("INK_CONTRAST_THRESHOLD", "0.15")
        )
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


class LocalWindowGrouping(nn.Module):
    """Fuse left/current/right windows through a conservative learned gate.

    The residual Conv1d starts at zero. A per-window/per-feature gate starts near
    zero as well, so neighboring characters cannot immediately leak into the
    current window representation. Training opens the gate only where the
    neighboring evidence is useful.
    """

    def __init__(self, embed_dim, group_size=3):
        super().__init__()
        group_size = int(group_size)
        if group_size != 3:
            raise ValueError(
                "LOCAL_GROUP_SIZE currently supports exactly 3 windows"
            )
        self.group_size = group_size
        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=group_size,
            padding=group_size // 2,
            bias=False,
        )
        self.gate = nn.Linear(embed_dim * 2, embed_dim)
        self.mix_logit = nn.Parameter(torch.tensor(0.0))
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -3.0)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # A checkpoint produced before the per-window gate already learned its
        # ungated Conv1d. Preserve that behavior by opening the new gate on load.
        legacy_grouping = (
            prefix + "conv.weight" in state_dict
            and prefix + "gate.weight" not in state_dict
        )
        if prefix + "gate.weight" not in state_dict:
            state_dict[prefix + "gate.weight"] = self.gate.weight.detach().clone()
        if prefix + "gate.bias" not in state_dict:
            bias = self.gate.bias.detach().clone()
            if legacy_grouping:
                bias.fill_(8.0)
            state_dict[prefix + "gate.bias"] = bias
        if prefix + "mix_logit" not in state_dict:
            state_dict[prefix + "mix_logit"] = self.mix_logit.detach().clone()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        neighbor_delta = self.conv(x.transpose(1, 2)).transpose(1, 2)
        gate = torch.sigmoid(self.gate(torch.cat([x, neighbor_delta], dim=-1)))
        global_mix = torch.sigmoid(self.mix_logit)
        return x + global_mix * gate * neighbor_delta


class BiLSTMEncoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, lstm_layers=1):
        super().__init__()
        hidden_dim = hidden_dim or embed_dim // 2
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        residual = x
        self.bilstm.flatten_parameters()
        x, _ = self.bilstm(x)
        x = self.proj(x)
        return self.norm(x + residual)


class ModifiedOCRResNet34(nn.Module):
    def __init__(self, vector_size=512):
        super().__init__()
        try:
            base_resnet = torchvision.models.resnet34(
                weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            )
        except Exception as exc:
            print(
                "[ModifiedOCRResNet34] ImageNet weights unavailable "
                f"({exc}); using weights=None.",
                flush=True,
            )
            base_resnet = torchvision.models.resnet34(weights=None)

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


class EmbeddingModel(nn.Module):
    CNN_CHUNK_SIZE = 512

    def __init__(
        self,
        window_size=128,
        stride=64,
        vector_size=512,
        device="cuda",
        use_flip=False,
        use_bilstm=True,
        bilstm_layers=1,
        bilstm_hidden_dim=None,
        use_local_grouping=None,
        local_group_size=3,
    ):
        super().__init__()
        self.device = device
        self.window_size = window_size
        self.stride = stride
        self.vector_size = vector_size
        self.use_bilstm = use_bilstm

        # Persist sequence direction in the checkpoint. Previously the evaluator
        # had to guess this independently from the training configuration.
        self.register_buffer(
            "_use_flip_state",
            torch.tensor(1 if use_flip else 0, dtype=torch.uint8),
        )

        if use_local_grouping is None:
            use_local_grouping = _env_flag("USE_LOCAL_WINDOW_GROUPING", True)
        self.register_buffer(
            "_use_local_grouping_state",
            torch.tensor(1 if use_local_grouping else 0, dtype=torch.uint8),
        )
        self.local_group_encoder = LocalWindowGrouping(
            embed_dim=vector_size,
            group_size=local_group_size,
        ).to(device)

        self.cnn_encoder = ModifiedOCRResNet34(vector_size=vector_size).to(device)
        self.sequence_encoder = None
        if use_bilstm:
            self.sequence_encoder = BiLSTMEncoder(
                embed_dim=vector_size,
                hidden_dim=bilstm_hidden_dim,
                lstm_layers=bilstm_layers,
            ).to(device)
        self.vision_norm = nn.LayerNorm(vector_size).to(device)

    @property
    def use_flip(self):
        return bool(int(self._use_flip_state.item()))

    @property
    def use_local_grouping(self):
        return bool(int(self._use_local_grouping_state.item()))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        flip_key = prefix + "_use_flip_state"
        if flip_key not in state_dict:
            state_dict[flip_key] = self._use_flip_state.detach().clone()

        grouping_key = prefix + "_use_local_grouping_state"
        force_grouping = _env_flag("FORCE_LOCAL_WINDOW_GROUPING", False)
        if grouping_key not in state_dict:
            self._use_local_grouping_state.fill_(1 if force_grouping else 0)
            state_dict[grouping_key] = self._use_local_grouping_state.detach().clone()

        # Conv/gate compatibility is also handled by LocalWindowGrouping, but
        # strict parent loading expects every submodule key to be present.
        for name, value in self.local_group_encoder.state_dict().items():
            key = prefix + "local_group_encoder." + name
            if key not in state_dict:
                if (
                    name == "gate.bias"
                    and prefix + "local_group_encoder.conv.weight" in state_dict
                ):
                    value = value.detach().clone().fill_(8.0)
                state_dict[key] = value.detach().clone()

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _process_patches(self, patches):
        batch_size, windows_num, channels, height, width = patches.shape
        total_patches = batch_size * windows_num
        patches = patches.reshape(total_patches, channels, height, width)
        chunks = []
        for start in range(0, total_patches, self.CNN_CHUNK_SIZE):
            end = min(start + self.CNN_CHUNK_SIZE, total_patches)
            chunks.append(self.cnn_encoder(patches[start:end]))
        encoded = torch.cat(chunks, dim=0)
        return encoded.view(batch_size, windows_num, self.vector_size)

    def forward(
        self,
        image,
        show_dims=False,
        return_local=False,
        return_ink=False,
        return_grouped=False,
    ):
        patches = sliding_window(image, self.window_size, self.stride)
        if self.use_flip:
            patches = torch.flip(patches, dims=[1])
        ink_ratio = (
            window_ink_ratio_from_patches(patches) if return_ink else None
        )

        local_features_raw = self._process_patches(patches)
        grouped_features = local_features_raw
        if self.use_local_grouping:
            grouped_features = self.local_group_encoder(grouped_features)

        contextual_features = grouped_features
        if self.sequence_encoder is not None:
            contextual_features = self.sequence_encoder(contextual_features)
        contextual_features = self.vision_norm(contextual_features)

        if show_dims:
            print(
                "image embeddings: "
                f"contextual={tuple(contextual_features.shape)} "
                f"flip={self.use_flip} "
                f"local_grouping={self.use_local_grouping}",
                flush=True,
            )

        outputs = [contextual_features]
        if return_local:
            outputs.append(self.vision_norm(local_features_raw))
        if return_grouped:
            outputs.append(self.vision_norm(grouped_features))
        if return_ink:
            outputs.append(ink_ratio)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)
