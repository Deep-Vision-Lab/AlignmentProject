from types import SimpleNamespace

import torch
from torch import nn

import direct_subword_loss as direct


class Encoding:
    def __init__(self, vector):
        blank = torch.zeros_like(vector)
        self.embeddings = torch.stack([vector, blank])
        self.unit_kinds = ["subword", "blank"]
        self.texts = ["token", "<BLANK>"]


class FakeTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    def encode_many(self, texts, use_cache=None):
        del use_cache
        return [Encoding(self.table[0 if text == "a" else 1]) for text in texts]


class FakeImageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.window_size = 32
        self.stride = 16
        self.use_flip = False


class P:
    device = torch.device("cpu")
    window_size = 32
    stride_ratio = 0.5
    window_overlap_mode = "custom"
    lang = "Arabic"
    image_variance_target_std = 0.05
    image_variance_loss_weight = 0.01


def compute_embeddings(model, images):
    batch = images.shape[0]
    base = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    raw = model.scale * base.unsqueeze(0).expand(batch, -1, -1)
    norm = torch.nn.functional.normalize(raw, dim=-1)
    ink = torch.ones(batch, 4)
    return norm, norm, ink, raw


train = SimpleNamespace(
    P=P,
    compute_embeddings=compute_embeddings,
    _unwrap_model=lambda model: model,
    compute_stride=lambda window, ratio, mode: int(window * ratio),
    image_embedding_variance_loss=lambda raw, target: raw.var(dim=(0, 1)).mean(),
)


def test_direct_batch_loss_backpropagates_to_image_and_text(monkeypatch):
    monkeypatch.setenv("DIRECT_SUBWORD_SUPERVISION", "1")
    model = FakeImageModel()
    text = FakeTextEncoder()
    batch = {
        "images1": torch.zeros(1, 3, 128, 80),
        "images2": torch.zeros(1, 3, 128, 80),
        "subwords1": [[
            {"text": "a", "x0": 0.0, "x1": 48.0},
            {"text": "b", "x0": 48.0, "x1": 80.0},
        ]],
        "subwords2": [[
            {"text": "a", "x0": 0.0, "x1": 48.0},
            {"text": "b", "x0": 48.0, "x1": 80.0},
        ]],
    }
    loss, stats = direct.make_batch_loss(train)(model, text, None, batch)
    assert torch.isfinite(loss)
    assert stats["direct_subword_regions"] == 4.0
    loss.backward()
    assert model.scale.grad is not None
    assert text.table.grad is not None
