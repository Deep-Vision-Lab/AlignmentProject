import torch
import torch.nn.functional as F

from arabic_span_text_encoder import SpanEncoding
from span_alignment_loss import _transition_cost, hard_span_dtw_path


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


if __name__ == "__main__":
    test_transition_cost_prefers_similar_window()
    test_two_character_span_for_one_window()
    test_single_char_spans_for_two_windows()
    print("span alignment checks passed")
