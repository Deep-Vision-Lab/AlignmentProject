import torch
import torch.nn.functional as F

from arabic_span_text_encoder import SpanEncoding
from span_alignment_loss import SpanContrastiveSoftDTW

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
    data = torch.full((3, 3, 4, 4), 1e6, dtype=torch.float32)
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


def test_jax_function_backward():
    if skip_if_no_jax():
        return

    transition_costs_dense = make_dense_transition_costs()
    cost = JaxSpanDTWFunction.apply(transition_costs_dense, 0.1)
    cost.backward()

    assert torch.isfinite(cost)
    assert transition_costs_dense.grad is not None
    assert torch.isfinite(transition_costs_dense.grad).all()
    assert transition_costs_dense.grad.abs().sum().item() > 0.0


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


if __name__ == "__main__":
    test_jax_function_backward()
    test_jax_backend_matches_torch_backend()
    if JAX_IMPORT_ERROR is None and is_jax_available():
        print("jax span-DTW checks passed")
