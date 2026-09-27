"""Shared CNN applied independently to explicit full-height image windows."""
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

SIMPLE_CNN_CHANNELS = (32, 64, 128, 256, 512)


def simple_cnn_channels(num_layers):
    """Channel layout for the configurable simple CNN, capped at 512 channels."""
    if num_layers < 1:
        raise ValueError(f'num_layers must be a positive integer, got {num_layers}')
    return [SIMPLE_CNN_CHANNELS[min(index, len(SIMPLE_CNN_CHANNELS) - 1)]
            for index in range(num_layers)]


class CNNEncoder(nn.Module):
    def __init__(self, encoder_type='resnet18', embedding_dim=128, pretrained=True,
                 input_channels=1, local_dropout=.10, num_layers=3):
        super().__init__()
        if input_channels not in (1, 3):
            raise ValueError('input_channels must be 1 or 3')
        self.encoder_type = encoder_type
        self.initialization = 'random'
        self.num_layers = None
        self.channels = None
        if encoder_type == 'simple':
            # num_layers applies only to the simple CNN; the final block always
            # pools to 1x1 so the output is [N_windows, embedding_dim] at any depth.
            self.num_layers = num_layers
            self.channels = simple_cnn_channels(num_layers)
            blocks = []
            in_channels = input_channels
            for index, out_channels in enumerate(self.channels):
                blocks += [nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GELU(),
                           nn.AdaptiveAvgPool2d(1 if index == num_layers - 1 else 2)]
                in_channels = out_channels
            blocks.append(nn.Flatten())
            self.backbone = nn.Sequential(*blocks)
            features = self.channels[-1]
        elif encoder_type == 'resnet18':
            self.backbone = resnet18(weights=None)
            if pretrained:
                # Offline by construction: never download weights implicitly.
                filename = Path(urlparse(ResNet18_Weights.DEFAULT.url).path).name
                candidates = [Path(__file__).parent / 'Pretrained/torch/checkpoints' / filename,
                              Path(torch.hub.get_dir()) / 'checkpoints' / filename]
                checkpoint = next((p for p in candidates if p.is_file()), None)
                if checkpoint is None:
                    raise FileNotFoundError(f'Cached ImageNet ResNet18 weights required: {candidates}; '
                                            'set pretrained=False for an explicitly fresh backbone')
                self.backbone.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
                self.initialization = str(checkpoint)
            if input_channels == 1:
                old = self.backbone.conv1
                conv = nn.Conv2d(1, old.out_channels, old.kernel_size, old.stride, old.padding, bias=False)
                with torch.no_grad():
                    conv.weight.copy_(old.weight.mean(dim=1, keepdim=True))
                self.backbone.conv1 = conv
            self.backbone.fc = nn.Identity()
            features = 512
        else:
            raise ValueError('encoder_type must be simple or resnet18')
        self.projection = nn.Sequential(nn.Linear(features, embedding_dim), nn.Dropout(local_dropout))

    def forward(self, windows):
        return self.projection(self.backbone(windows))
