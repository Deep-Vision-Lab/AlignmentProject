from collections import OrderedDict

import torch

from arabic_span_text_encoder import ArabicSpanTextEncoder


class DummySpanEncoder(ArabicSpanTextEncoder):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.max_span_chars = 2
        self.freeze_backbone = True
        self.device = torch.device("cpu")
        self.strip_text_edges = True
        self.cache_size = 64
        self.cache_dtype = "float16"
        self.space_token = "<SPACE>"
        self.register_buffer(
            "_boundary_context_chars_state", torch.tensor(1, dtype=torch.int16)
        )
        self.register_buffer(
            "_include_space_context_state", torch.tensor(0, dtype=torch.uint8)
        )
        self.register_buffer(
            "_boundary_context_max_core_chars_state",
            torch.tensor(1, dtype=torch.int16),
        )
        self.register_buffer(
            "_allow_character_space_surfaces_state",
            torch.tensor(0, dtype=torch.uint8),
        )
        self.projection = torch.nn.Linear(4, 3, bias=False)
        self.norm = torch.nn.LayerNorm(3)
        self.space_embedding = torch.nn.Parameter(torch.randn(3))
        self.blank_embedding = torch.nn.Parameter(torch.randn(3))
        self._surface_feature_cache = OrderedDict()
        self._surface_cache_hits = 0
        self._surface_cache_misses = 0
        self.backbone_calls = 0

    def _encode_missing_visible(self, visible_texts):
        # Production returns before invoking AraBERT when every surface was a
        # cache hit. Keep the test double's call counter semantically identical.
        if not visible_texts:
            return {}
        self.backbone_calls += 1
        result = {}
        for text in visible_texts:
            values = [float((ord(character) % 17) + 1) for character in text]
            vector = torch.tensor(
                [
                    sum(values),
                    float(len(values)),
                    max(values),
                    min(values),
                ],
                dtype=self.projection.weight.dtype,
            )
            result[text] = vector
            self._cache_put_surface(text, vector)
        return result


def test_encode_many_reuses_unique_surfaces_and_keeps_projection_gradients():
    encoder = DummySpanEncoder()
    first = encoder.encode_many(["با", "باب"])
    calls_after_first = encoder.backbone_calls
    second = encoder.encode_many(["با", "باب"])

    assert calls_after_first == 1
    assert encoder.backbone_calls == calls_after_first
    assert encoder.cache_stats()["surface_cache_hit_rate"] > 0.0
    for left, right in zip(first, second):
        torch.testing.assert_close(left.embeddings, right.embeddings)
        torch.testing.assert_close(left.context_embeddings, right.context_embeddings)

    loss = sum(item.embeddings.sum() for item in second)
    loss.backward()
    assert encoder.projection.weight.grad is not None
    assert torch.isfinite(encoder.projection.weight.grad).all()
    assert encoder.blank_embedding.grad is not None


def test_epoch_clear_does_not_delete_frozen_surface_cache(monkeypatch):
    encoder = DummySpanEncoder()
    encoder.encode_many(["با"])
    size = encoder.cache_size_current()
    encoder.clear_cache()
    assert encoder.cache_size_current() == size

    monkeypatch.setenv("CLEAR_FROZEN_SPAN_CACHE", "1")
    encoder.clear_cache()
    assert encoder.cache_size_current() == 0
