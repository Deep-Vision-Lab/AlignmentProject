"""Shared CNN applied independently to explicit full-height image windows."""
from pathlib import Path
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class CNNEncoder(nn.Module):
    def __init__(self, encoder_type='resnet18', embedding_dim=128, pretrained=True,
                 input_channels=1, local_dropout=.10):
        super().__init__()
        if input_channels not in (1, 3):
            raise ValueError('input_channels must be 1 or 3')
        self.encoder_type = encoder_type
        self.initialization = 'random'
        if encoder_type == 'simple':
            self.backbone = nn.Sequential(
                nn.Conv2d(input_channels, 32, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
            features = 128
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
