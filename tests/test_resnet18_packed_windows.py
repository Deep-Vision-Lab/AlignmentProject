import torch
import torch.nn as nn

from resnet18_window_encoder import ResNet18WindowEncoder


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(3, 512, bias=False)

    def forward(self, x):
        x = self.pool(x).flatten(1)
        return self.proj(x)


def test_forward_packed_skips_invalid_outer_windows():
    encoder = ResNet18WindowEncoder(
        input_height=128,
        window_size=32,
        stride=16,
        embed_dim=8,
        pretrained=False,
    )
    encoder.backbone = TinyBackbone()
    encoder.projection = nn.Sequential(
        nn.Linear(512, 8, bias=False),
        nn.LayerNorm(8),
    )

    # Width 64 -> starts at 0,16,32 -> 3 physical windows.
    line = torch.randn(2, 3, 128, 64)
    token_valid = torch.tensor(
        [
            [False, True, True],
            [True, False, False],
        ],
        dtype=torch.bool,
    )

    packed, packed_valid = encoder.forward_packed(
        line,
        token_valid,
        use_flip=False,
    )

    # Longest sample has only 2 valid windows, so no 3-token blank sequence.
    assert packed.shape == (2, 8, 1, 2)
    assert packed_valid.tolist() == [[True, True], [True, False]]

    # Padding added only for rectangular batching is a zero feature and invalid.
    assert torch.count_nonzero(packed[1, :, 0, 1]) == 0


def test_forward_packed_reverses_only_real_tokens_for_arabic_order():
    encoder = ResNet18WindowEncoder(
        input_height=128,
        window_size=32,
        stride=16,
        embed_dim=4,
        pretrained=False,
    )
    encoder.backbone = TinyBackbone()
    encoder.projection = nn.Sequential(
        nn.Linear(512, 4, bias=False),
        nn.LayerNorm(4),
    )

    line = torch.randn(1, 3, 128, 64)
    token_valid = torch.tensor([[True, True, False]], dtype=torch.bool)

    forward, valid_forward = encoder.forward_packed(
        line, token_valid, use_flip=False
    )
    reverse, valid_reverse = encoder.forward_packed(
        line, token_valid, use_flip=True
    )

    assert valid_forward.tolist() == [[True, True]]
    assert valid_reverse.tolist() == [[True, True]]
    assert torch.allclose(
        reverse[:, :, 0, :],
        torch.flip(forward[:, :, 0, :], dims=[2]),
        atol=1e-6,
    )
