import torch
import torch.nn as nn
import torchvision


def sliding_window(image, window_size, stride):
    patches = image.unfold(dimension=3, size=window_size, step=stride)
    return patches.permute(0, 3, 1, 2, 4).contiguous()


def window_ink_ratio_from_patches(patches):
    """Estimate how much dark ink exists in each sliding-window patch.

    patches: [B, S, C, H, W], usually in [0, 1] with white background near 1.
    Returns: [B, S], where higher values mean more ink/strokes.
    """
    gray = patches.float().mean(dim=2)
    ink = (1.0 - gray).clamp(min=0.0, max=1.0)
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
