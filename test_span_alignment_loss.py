import torch
import torch.nn.functional as F

from arabic_span_text_encoder import SpanEncoding
from span_alignment_loss import hard_span_dtw_path


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


if __name__ == "__main__":
    test_two_character_span_for_one_window()
    test_single_char_spans_for_two_windows()
    print("span alignment checks passed")
