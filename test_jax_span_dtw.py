import torch
import torch.nn.functional as F

from arabic_span_text_encoder import SpanEncoding
import span_alignment_loss as sal
from span_alignment_loss import SpanContrastiveSoftDTW, _dense_transition_costs

try:
    from jax_span_dtw import JaxSpanDTWFunction, is_jax_available
except ImportError as exc:
    JAX_IMPORT_ERROR = exc
    JaxSpanDTWFunction = None
    is_jax_available = None
else:
    JAX_IMPORT_ERROR = None


def skip_if_no_jax():
    if JAX_IMPORT_ERROR is not None or not is_jax_available():
        reason = JAX_IMPORT_ERROR or "JAX is not installed in this Python environment."
        print(f"Skipping JAX span-DTW checks: {reason}")
        return True
    return False


def make_dense_transition_costs():
    data = torch.full((3, 3, 5, 4), float("inf"), dtype=torch.float32)
    data[1, 1, 0, 0] = 0.1
    data[1, 1, 1, 1] = 0.2
    data[1, 1, 2, 2] = 0.3
    data[2, 1, 0, 0] = 0.5
    data[1, 2, 2, 1] = 0.4
    return data.requires_grad_(True)


def make_encoding():
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.2, 0.2],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return SpanEncoding(
        embeddings=F.normalize(embeddings, p=2, dim=-1),
        starts=[0, 0, 1],
        lengths=[1, 2, 1],
        texts=["a", "ab", "b"],
        text_length=2,
        max_span_chars=2,
    )


def make_long_encoding(text_length):
    embeddings = torch.eye(2, dtype=torch.float32).repeat((text_length + 1) // 2, 1)[:text_length]
    return SpanEncoding(
        embeddings=F.normalize(embeddings, p=2, dim=-1),
        starts=list(range(text_length)),
        lengths=[1] * text_length,
        texts=["a"] * text_length,
        text_length=text_length,
        max_span_chars=1,
    )


def test_jax_function_backward():
    if skip_if_no_jax():
        return

    transition_costs_dense = make_dense_transition_costs()
    cost = JaxSpanDTWFunction.apply(transition_costs_dense, 3, 3, 0.1)
    cost.backward()

    assert torch.isfinite(cost)
    assert transition_costs_dense.grad is not None
    assert torch.isfinite(transition_costs_dense.grad).all()
    assert transition_costs_dense.grad.abs().sum().item() > 0.0


def test_jax_function_uses_actual_length_not_padded_endpoint():
    if skip_if_no_jax():
        return

    transition_costs_dense = make_dense_transition_costs()
    cost = JaxSpanDTWFunction.apply(transition_costs_dense, 3, 3, 0.1)
    padded_endpoint_cost = JaxSpanDTWFunction.apply(transition_costs_dense, 4, 3, 0.1)

    assert torch.isfinite(cost)
    assert torch.isinf(padded_endpoint_cost)


def test_jax_no_grad_call_does_not_populate_transition_grad():
    if skip_if_no_jax():
        return

    transition_costs_dense = make_dense_transition_costs()
    with torch.no_grad():
        cost = JaxSpanDTWFunction.apply(transition_costs_dense, 3, 3, 0.1, False)

    assert torch.isfinite(cost)
    assert transition_costs_dense.grad is None


def test_bucketed_dense_shapes_share_text_dimension():
    image_embeddings = F.normalize(torch.tensor([[1.0, 0.0], [0.0, 1.0]]), p=2, dim=-1)
    dense_57 = _dense_transition_costs(
        make_long_encoding(57),
        image_embeddings,
        temperature=1.0,
        max_windows_per_span=1,
        window_count_penalty=0.0,
        text_steps_padded=sal._bucket_length(57, 16, 256, enabled=True),
    )
    dense_58 = _dense_transition_costs(
        make_long_encoding(58),
        image_embeddings,
        temperature=1.0,
        max_windows_per_span=1,
        window_count_penalty=0.0,
        text_steps_padded=sal._bucket_length(58, 16, 256, enabled=True),
    )

    assert dense_57.shape[2] == 65
    assert dense_58.shape[2] == 65
    assert dense_57.shape == dense_58.shape
    assert torch.isinf(dense_57[:, :, 58:, :]).all()


def test_jax_backend_matches_torch_backend():
    if skip_if_no_jax():
        return

    encoding = make_encoding()
    image_embeddings = F.normalize(
        torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        p=2,
        dim=-1,
    )
    torch_criterion = SpanContrastiveSoftDTW(
        gamma=0.1,
        temperature=1.0,
        max_windows_per_span=2,
        window_count_penalty=0.01,
        backend="torch",
    )
    jax_criterion = SpanContrastiveSoftDTW(
        gamma=0.1,
        temperature=1.0,
        max_windows_per_span=2,
        window_count_penalty=0.01,
        backend="jax",
    )

    torch_cost = torch_criterion._span_dtw_cost_torch(encoding, image_embeddings)
    jax_cost = jax_criterion._span_dtw_cost_jax(encoding, image_embeddings)

    assert torch.allclose(jax_cost, torch_cost, atol=1e-4, rtol=1e-4)


def test_jax_backend_matches_torch_with_bucketing_disabled_and_enabled():
    if skip_if_no_jax():
        return

    previous_bucket_enabled = sal.span_dtw_bucket_text_lengths
    previous_bucket_size = sal.span_dtw_text_bucket_size
    previous_max_bucket = sal.span_dtw_max_text_bucket
    try:
        encoding = make_encoding()
        image_embeddings = F.normalize(
            torch.tensor(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                ],
                dtype=torch.float32,
            ),
            p=2,
            dim=-1,
        )
        torch_criterion = SpanContrastiveSoftDTW(
            gamma=0.1,
            temperature=1.0,
            max_windows_per_span=2,
            window_count_penalty=0.01,
            backend="torch",
        )
        jax_criterion = SpanContrastiveSoftDTW(
            gamma=0.1,
            temperature=1.0,
            max_windows_per_span=2,
            window_count_penalty=0.01,
            backend="jax",
        )
        torch_cost = torch_criterion._span_dtw_cost_torch(encoding, image_embeddings)

        sal.span_dtw_bucket_text_lengths = False
        jax_exact = jax_criterion._span_dtw_cost_jax(encoding, image_embeddings)
        sal.span_dtw_bucket_text_lengths = True
        sal.span_dtw_text_bucket_size = 16
        sal.span_dtw_max_text_bucket = 256
        jax_bucketed = jax_criterion._span_dtw_cost_jax(encoding, image_embeddings)

        assert torch.allclose(jax_exact, torch_cost, atol=1e-4, rtol=1e-4)
        assert torch.allclose(jax_bucketed, torch_cost, atol=1e-4, rtol=1e-4)
    finally:
        sal.span_dtw_bucket_text_lengths = previous_bucket_enabled
        sal.span_dtw_text_bucket_size = previous_bucket_size
        sal.span_dtw_max_text_bucket = previous_max_bucket


if __name__ == "__main__":
    test_jax_function_backward()
    test_jax_function_uses_actual_length_not_padded_endpoint()
    test_jax_no_grad_call_does_not_populate_transition_grad()
    test_bucketed_dense_shapes_share_text_dimension()
    test_jax_backend_matches_torch_backend()
    test_jax_backend_matches_torch_with_bucketing_disabled_and_enabled()
    if JAX_IMPORT_ERROR is None and is_jax_available():
        print("jax span-DTW checks passed")
