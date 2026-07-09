from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from arabic_span_text_encoder import ArabicSpanTextEncoder, SpanEncoding
from span_alignment_loss import SpanContrastiveSoftDTW, _transition_cost, hard_span_dtw_path


def make_encoding():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.2, 0.2],
            [0.0, 1.0],
        ]
    )
    return SpanEncoding(
        embeddings=F.normalize(embeddings, p=2, dim=-1),
        starts=[0, 0, 1],
        lengths=[1, 2, 1],
        texts=["a", "ab", "b"],
        text_length=2,
        max_span_chars=2,
    )


def test_two_character_span_for_one_window():
    encoding = make_encoding()
    image_embeddings = F.normalize(torch.tensor([[0.2, 0.2]]), p=2, dim=-1)
    path = hard_span_dtw_path(encoding, image_embeddings, max_windows=2)
    assert [segment["text"] for segment in path] == ["ab"]


def test_single_char_spans_for_two_windows():
    encoding = make_encoding()
    image_embeddings = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        p=2,
        dim=-1,
    )
    path = hard_span_dtw_path(encoding, image_embeddings, max_windows=2)
    assert [segment["text"] for segment in path] == ["a", "b"]


def test_transition_cost_prefers_similar_window():
    span = F.normalize(torch.tensor([1.0, 0.0, 0.0]), dim=0)
    good_window = F.normalize(torch.tensor([[1.0, 0.0, 0.0]]), dim=1)
    bad_window = F.normalize(torch.tensor([[-1.0, 0.0, 0.0]]), dim=1)

    good_cost = _transition_cost(
        span, good_window, temperature=1.0, window_count_penalty=0.0
    )
    bad_cost = _transition_cost(
        span, bad_window, temperature=1.0, window_count_penalty=0.0
    )

    assert good_cost.item() >= 0.0
    assert bad_cost.item() >= 0.0
    assert good_cost.item() < bad_cost.item()


class DummyArabicSpanTextEncoder(ArabicSpanTextEncoder):
    def __init__(self, cache_size=2, cache_dtype="float16"):
        nn.Module.__init__(self)
        self.model_name = "dummy"
        self.max_span_chars = 1
        self.freeze_backbone = True
        self.device = torch.device("cpu")
        self.strip_text_edges = True
        self.cache_size = cache_size
        self.cache_dtype = cache_dtype
        self.backbone = nn.Identity()
        self.projection = nn.Linear(2, 2)
        self.norm = nn.LayerNorm(2)
        self._span_feature_cache = OrderedDict()

    def enumerate_spans(self, text):
        starts = list(range(len(text)))
        lengths = [1] * len(text)
        spans = list(text)
        return starts, lengths, spans

    def _get_frozen_span_features(self, text, use_cache=None):
        text = self._prepare_text(text)
        use_cache = self._should_use_cache(use_cache)
        cached = self._cache_get(text, use_cache=use_cache)
        if cached is not None:
            return cached

        starts, lengths, spans = self.enumerate_spans(text)
        pooled = torch.stack(
            [
                torch.tensor([float((ord(span) % 7) + 1), 1.0], dtype=torch.float32)
                for span in spans
            ]
        )
        self._cache_put(text, starts, lengths, spans, pooled, use_cache=use_cache)
        return starts, lengths, spans, pooled


def test_span_feature_cache_disabled_by_default_in_train_and_bounded_in_eval():
    encoder = DummyArabicSpanTextEncoder(cache_size=2)
    encoder.train()
    encoder("a")
    encoder("b")
    encoder("c")
    assert encoder.cache_size_current() == 0

    encoder.eval()
    encoder("a")
    encoder("b")
    encoder("c")
    assert encoder.cache_size_current() == 2
    assert all(cached[-1].device.type == "cpu" for cached in encoder._span_feature_cache.values())
    assert all(cached[-1].dtype == torch.float16 for cached in encoder._span_feature_cache.values())


def test_negative_text_encoding_disables_cache():
    class RecordingTextEncoder(DummyArabicSpanTextEncoder):
        def __init__(self):
            super().__init__(cache_size=8)
            self.calls = []

        def forward(self, text, use_cache=None):
            self.calls.append((text, use_cache))
            return super().forward(text, use_cache=use_cache)

    text_encoder = RecordingTextEncoder()
    text_encoder.train()
    criterion = SpanContrastiveSoftDTW(
        gamma=0.1,
        margin=1.0,
        temperature=1.0,
        max_windows_per_span=1,
        window_count_penalty=0.0,
        negative_grad_mode="hardest",
        backend="torch",
    )
    norm_img = F.normalize(torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]), p=2, dim=-1)

    criterion.forward_varlen(text_encoder, norm_img, ["ab"], [["ba", "aa"]])

    neg_calls = [use_cache for text, use_cache in text_encoder.calls if text in {"ba", "aa"}]
    assert neg_calls
    assert all(use_cache is False for use_cache in neg_calls)


if __name__ == "__main__":
    test_transition_cost_prefers_similar_window()
    test_two_character_span_for_one_window()
    test_single_char_spans_for_two_windows()
    test_span_feature_cache_disabled_by_default_in_train_and_bounded_in_eval()
    test_negative_text_encoding_disables_cache()
    print("span alignment checks passed")
