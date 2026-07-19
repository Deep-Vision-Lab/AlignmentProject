import os

import torch
import torch.nn as nn
import torchvision


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def sliding_window(image, window_size, stride):
    patches = image.unfold(dimension=3, size=window_size, step=stride)
    return patches.permute(0, 3, 1, 2, 4).contiguous()


def _denormalize_imagenet_patches(patches):
    """Convert ImageNet-normalized RGB patches back to the [0, 1] range."""
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
    """Estimate each patch background from a robust median over its border."""
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
    """Estimate foreground-stroke coverage independent of image polarity.

    The model receives ImageNet-normalized RGB tensors.  Earlier code used
    ``1 - gray`` and therefore treated dark pixels as ink.  That reverses the
    mask for the project's black-background/white-text images and also operates
    on normalized values rather than true pixel intensities.

    This implementation first restores RGB values to [0, 1], estimates the
    local background from each patch border, and counts pixels whose contrast
    from that background is large enough.  Consequently both black-on-white and
    white-on-black lines produce the same ink ratio.

    Args:
        patches: Tensor [B, S, 3, H, W] after ImageNet normalization.
        contrast_threshold: Minimum absolute foreground/background contrast.
            Defaults to INK_CONTRAST_THRESHOLD or 0.15.

    Returns:
        Tensor [B, S], where higher values mean more visible strokes.
    """
    if contrast_threshold is None:
        contrast_threshold = float(os.environ.get("INK_CONTRAST_THRESHOLD", "0.15"))
    contrast_threshold = max(0.0, min(1.0, float(contrast_threshold)))

    # The ratio is used as a mask/weight, not as a differentiable image path.
    with torch.no_grad():
        rgb = _denormalize_imagenet_patches(patches.detach())
        gray = (
            0.2989 * rgb[:, :, 0]
            + 0.5870 * rgb[:, :, 1]
            + 0.1140 * rgb[:, :, 2]
        )
        background = _patch_background_level(gray)
        foreground_contrast = (gray - background).abs()
        ink = foreground_contrast.ge(contrast_threshold).float()
        return ink.mean(dim=(2, 3))


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
                f"[ModifiedOCRResNet34] ImageNet weights unavailable ({exc}); using weights=None.",
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
    ):
        super().__init__()
        self.device = device
        self.window_size = window_size
        self.stride = stride
        self.vector_size = vector_size
        self.use_flip = use_flip
        self.use_bilstm = use_bilstm

        self.cnn_encoder = ModifiedOCRResNet34(vector_size=vector_size).to(device)
        self.sequence_encoder = None
        if use_bilstm:
            self.sequence_encoder = BiLSTMEncoder(
                embed_dim=vector_size,
                hidden_dim=bilstm_hidden_dim,
                lstm_layers=bilstm_layers,
            ).to(device)
        self.vision_norm = nn.LayerNorm(vector_size).to(device)

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

    def forward(self, image, show_dims=False, return_local=False, return_ink=False):
        patches = sliding_window(image, self.window_size, self.stride)
        if self.use_flip:
            patches = torch.flip(patches, dims=[1])

        ink_ratio = window_ink_ratio_from_patches(patches) if return_ink else None

        # Raw CNN window features are kept as the local visual representation.
        # They are returned before the BiLSTM so local contrastive losses can
        # learn stroke/window discrimination without sequence-context smoothing.
        local_features_raw = self._process_patches(patches)

        contextual_features = local_features_raw
        if self.sequence_encoder is not None:
            contextual_features = self.sequence_encoder(contextual_features)
        contextual_features = self.vision_norm(contextual_features)

        if show_dims:
            print(f"image embeddings: {contextual_features.shape}", flush=True)

        outputs = [contextual_features]
        if return_local:
            outputs.append(self.vision_norm(local_features_raw))
        if return_ink:
            outputs.append(ink_ratio)

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)
