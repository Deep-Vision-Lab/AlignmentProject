import torch

from arabic_span_text_encoder import ArabicSpanTextEncoder, SpanEncoding
from span_alignment_loss import SpanContrastiveSoftDTW, _min_required_spans


def make_encoding(text, max_span_chars):
    starts = []
    lengths = []
    spans = []
    for start, char in enumerate(text):
        if char.isspace():
            starts.append(start)
            lengths.append(1)
            spans.append(char)
            continue
        max_end = min(len(text), start + max_span_chars)
        for end in range(start + 1, max_end + 1):
            span = text[start:end]
            if any(ch.isspace() for ch in span):
                break
            starts.append(start)
            lengths.append(end - start)
            spans.append(span)

    return SpanEncoding(
        embeddings=torch.empty(len(starts), 1),
        starts=starts,
        lengths=lengths,
        texts=spans,
        text_length=len(text),
        max_span_chars=max_span_chars,
    )


def test_impossible_65_chars_32_windows_max_span_2():
    encoding = make_encoding("a" * 65, max_span_chars=2)
    min_required = _min_required_spans(encoding)
    assert min_required >= 33
    assert 32 < min_required


def test_possible_65_chars_32_windows_max_span_3():
    encoding = make_encoding("a" * 65, max_span_chars=3)
    min_required = _min_required_spans(encoding)
    assert min_required <= 32
    assert 32 >= min_required


def test_strip_span_text_edges_keeps_internal_spaces():
    encoder = object.__new__(ArabicSpanTextEncoder)
    encoder.strip_text_edges = True
    assert encoder._prepare_text(" abc ") == "abc"
    assert encoder._prepare_text("ab cd") == "ab cd"


def test_infeasible_path_error_message_is_helpful():
    encoding = make_encoding("a" * 65, max_span_chars=2)
    image_embeddings = torch.randn(32, 1)
    criterion = SpanContrastiveSoftDTW(max_windows_per_span=2)
    try:
        criterion._span_dtw_cost(encoding, image_embeddings)
    except ValueError as exc:
        message = str(exc)
        assert "too few image windows" in message
        assert "text_length=65" in message
        assert "image_windows=32" in message
        assert "minimum_required_spans=33" in message
        assert "MAX_TEXT_SPAN_CHARS" in message
        assert "Do not fix this by increasing MAX_WINDOWS_PER_SPAN" in message
    else:
        raise AssertionError("Expected infeasible span-DTW path to raise ValueError")


if __name__ == "__main__":
    test_impossible_65_chars_32_windows_max_span_2()
    test_possible_65_chars_32_windows_max_span_3()
    test_strip_span_text_edges_keeps_internal_spaces()
    test_infeasible_path_error_message_is_helpful()
    print("span path feasibility checks passed")
