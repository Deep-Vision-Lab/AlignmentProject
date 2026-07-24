import os
import warnings

import torch


_warned_dlpack_fallback = False
_BRIDGE_STATS = {
    "torch_to_jax_calls": 0,
    "jax_to_torch_calls": 0,
    "dlpack_fallbacks": 0,
    "single_calls": 0,
    "batched_calls": 0,
    "batched_items": 0,
}


def _import_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        return None, None, exc
    return jax, jnp, None


jax, jnp, _JAX_IMPORT_ERROR = _import_jax()


def is_jax_available():
    return _JAX_IMPORT_ERROR is None


def _require_jax():
    if _JAX_IMPORT_ERROR is not None:
        raise RuntimeError(
            "SPAN_DTW_BACKEND=jax requires JAX to be installed. "
            "Use SPAN_DTW_BACKEND=torch or install the JAX packages from requirements.txt."
        ) from _JAX_IMPORT_ERROR
    return jax, jnp


def _configure_compilation_cache():
    if not is_jax_available():
        return
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "").strip()
    if not cache_dir:
        return
    try:
        os.makedirs(cache_dir, exist_ok=True)
        from jax.experimental.compilation_cache import compilation_cache

        compilation_cache.set_cache_dir(cache_dir)
    except Exception as exc:
        warnings.warn(
            f"Could not enable persistent JAX compilation cache at {cache_dir}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )


_configure_compilation_cache()


def bridge_stats(reset=False):
    result = dict(_BRIDGE_STATS)
    if reset:
        for key in _BRIDGE_STATS:
            _BRIDGE_STATS[key] = 0
    return result


def _masked_softmin(candidates, gamma):
    """Stable soft-min over valid candidates only."""
    _require_jax()
    finite = jnp.isfinite(candidates)
    has_candidate = jnp.any(finite)
    safe_candidates = jnp.where(finite, candidates, jnp.inf)
    min_value = jnp.min(safe_candidates)
    min_value = jnp.where(
        has_candidate,
        min_value,
        jnp.array(0.0, dtype=candidates.dtype),
    )
    gamma = jnp.asarray(gamma, dtype=candidates.dtype)
    shifted = jnp.where(
        finite,
        jnp.exp(-(safe_candidates - min_value) / gamma),
        jnp.array(0.0, dtype=candidates.dtype),
    )
    denom = jnp.sum(shifted)
    tiny = jnp.asarray(jnp.finfo(candidates.dtype).tiny, dtype=candidates.dtype)
    value = min_value - gamma * jnp.log(jnp.maximum(denom, tiny))
    return jnp.where(has_candidate, value, jnp.inf)


def jax_soft_span_dtw_cost(
    transition_costs_dense,
    actual_text_length,
    actual_image_steps,
    gamma,
):
    """Soft Span-DTW with zero-text, one-window blank transitions."""
    _require_jax()
    max_span_len = transition_costs_dense.shape[0] - 1
    max_window_count = transition_costs_dense.shape[1] - 1
    text_steps = transition_costs_dense.shape[2] - 1
    image_steps = transition_costs_dense.shape[3] - 1
    dtype = transition_costs_dense.dtype
    inf = jnp.array(jnp.inf, dtype=dtype)

    dp = jnp.full((text_steps + 1, image_steps + 1), inf, dtype=dtype)
    dp = dp.at[0, 0].set(jnp.array(0.0, dtype=dtype))

    def fill_row(i, dp_in):
        def fill_column(j, dp_state):
            candidates = []
            for span_len in range(1, max_span_len + 1):
                for window_count in range(1, max_window_count + 1):
                    prev_i = i - span_len
                    prev_j = j - window_count
                    valid_index = (
                        (i > 0)
                        & (j > 0)
                        & (prev_i >= 0)
                        & (prev_j >= 0)
                    )
                    safe_prev_i = jnp.maximum(prev_i, 0)
                    safe_prev_j = jnp.maximum(prev_j, 0)
                    prev_value = dp_state[safe_prev_i, safe_prev_j]
                    transition = transition_costs_dense[
                        span_len,
                        window_count,
                        safe_prev_i,
                        safe_prev_j,
                    ]
                    value = prev_value + transition
                    valid_value = (
                        valid_index
                        & jnp.isfinite(prev_value)
                        & jnp.isfinite(transition)
                    )
                    candidates.append(jnp.where(valid_value, value, inf))

            prev_j = j - 1
            safe_prev_j = jnp.maximum(prev_j, 0)
            blank_prev = dp_state[i, safe_prev_j]
            blank_transition = transition_costs_dense[0, 1, i, safe_prev_j]
            blank_valid = (
                (j > 0)
                & jnp.isfinite(blank_prev)
                & jnp.isfinite(blank_transition)
            )
            candidates.append(
                jnp.where(
                    blank_valid,
                    blank_prev + blank_transition,
                    inf,
                )
            )

            value = _masked_softmin(jnp.stack(candidates), gamma)
            value = jnp.where(
                (i == 0) & (j == 0),
                jnp.array(0.0, dtype=dtype),
                value,
            )
            return dp_state.at[i, j].set(value)

        return jax.lax.fori_loop(0, image_steps + 1, fill_column, dp_in)

    dp = jax.lax.fori_loop(0, text_steps + 1, fill_row, dp)
    actual_text_length = jnp.asarray(actual_text_length, dtype=jnp.int32)
    actual_image_steps = jnp.asarray(actual_image_steps, dtype=jnp.int32)
    return jax.lax.dynamic_slice(
        dp,
        (actual_text_length, actual_image_steps),
        (1, 1),
    )[0, 0]


def jax_batched_soft_span_dtw_cost(
    transition_costs_dense,
    actual_text_lengths,
    actual_image_steps,
    gamma,
):
    """Vectorize equal-shaped Span-DTW problems in one JAX execution."""
    _require_jax()
    return jax.vmap(
        jax_soft_span_dtw_cost,
        in_axes=(0, 0, 0, None),
        out_axes=0,
    )(
        transition_costs_dense,
        actual_text_lengths,
        actual_image_steps,
        gamma,
    )


def _batched_cost_sum_with_aux(dense, text_lengths, image_steps, gamma):
    costs = jax_batched_soft_span_dtw_cost(
        dense, text_lengths, image_steps, gamma
    )
    return costs.sum(), costs


if is_jax_available():
    jax_soft_span_dtw_cost = jax.jit(jax_soft_span_dtw_cost)
    _jax_value_and_grad = jax.jit(
        jax.value_and_grad(jax_soft_span_dtw_cost, argnums=0)
    )
    jax_batched_soft_span_dtw_cost = jax.jit(
        jax_batched_soft_span_dtw_cost
    )
    _jax_batched_value_and_grad = jax.jit(
        jax.value_and_grad(
            _batched_cost_sum_with_aux,
            argnums=0,
            has_aux=True,
        )
    )
else:
    _jax_value_and_grad = None
    _jax_batched_value_and_grad = None


def _warn_dlpack_fallback(reason):
    global _warned_dlpack_fallback
    _BRIDGE_STATS["dlpack_fallbacks"] += 1
    if not _warned_dlpack_fallback:
        warnings.warn(
            "Falling back to CPU/NumPy copies for JAX span-DTW bridge; this may be slow. "
            f"Reason: {reason}",
            RuntimeWarning,
            stacklevel=2,
        )
        _warned_dlpack_fallback = True


def _torch_to_jax(tensor):
    _require_jax()
    _BRIDGE_STATS["torch_to_jax_calls"] += 1
    try:
        from torch.utils import dlpack as torch_dlpack

        return jax.dlpack.from_dlpack(
            torch_dlpack.to_dlpack(tensor.detach().contiguous())
        )
    except Exception as exc:
        _warn_dlpack_fallback(exc)
        return jnp.asarray(tensor.detach().cpu().numpy())


def _jax_to_torch(array, device, dtype):
    _require_jax()
    _BRIDGE_STATS["jax_to_torch_calls"] += 1
    try:
        from torch.utils import dlpack as torch_dlpack

        tensor = torch_dlpack.from_dlpack(jax.dlpack.to_dlpack(array))
        return tensor.to(device=device, dtype=dtype)
    except Exception as exc:
        _warn_dlpack_fallback(exc)
        return torch.as_tensor(
            jax.device_get(array), device=device, dtype=dtype
        )


class JaxSpanDTWFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        transition_costs_dense,
        actual_text_length,
        actual_image_steps,
        gamma,
        needs_gradient=True,
    ):
        _require_jax()
        _BRIDGE_STATS["single_calls"] += 1
        jax_transition_costs = _torch_to_jax(transition_costs_dense)
        jax_actual_text_length = jnp.asarray(
            int(actual_text_length), dtype=jnp.int32
        )
        jax_actual_image_steps = jnp.asarray(
            int(actual_image_steps), dtype=jnp.int32
        )

        if bool(needs_gradient):
            cost, grad = _jax_value_and_grad(
                jax_transition_costs,
                jax_actual_text_length,
                jax_actual_image_steps,
                float(gamma),
            )
            grad_torch = _jax_to_torch(
                grad,
                device=transition_costs_dense.device,
                dtype=transition_costs_dense.dtype,
            )
        else:
            cost = jax_soft_span_dtw_cost(
                jax_transition_costs,
                jax_actual_text_length,
                jax_actual_image_steps,
                float(gamma),
            )
            grad_torch = None

        cost_torch = _jax_to_torch(
            cost,
            device=transition_costs_dense.device,
            dtype=transition_costs_dense.dtype,
        )
        ctx.needs_gradient = bool(needs_gradient)
        if grad_torch is not None:
            ctx.save_for_backward(grad_torch)
        return cost_torch.reshape(())

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_gradient:
            return None, None, None, None, None
        (grad_transition_costs,) = ctx.saved_tensors
        return (
            grad_output.to(grad_transition_costs.dtype)
            * grad_transition_costs,
            None,
            None,
            None,
            None,
        )


class JaxBatchedSpanDTWFunction(torch.autograd.Function):
    """One Torch↔JAX bridge call for a bucket of equal-shaped DP tensors."""

    @staticmethod
    def forward(
        ctx,
        transition_costs_dense,
        actual_text_lengths,
        actual_image_steps,
        gamma,
        needs_gradient=True,
    ):
        _require_jax()
        batch_size = int(transition_costs_dense.shape[0])
        _BRIDGE_STATS["batched_calls"] += 1
        _BRIDGE_STATS["batched_items"] += batch_size

        jax_dense = _torch_to_jax(transition_costs_dense)
        jax_text_lengths = _torch_to_jax(
            actual_text_lengths.to(dtype=torch.int32)
        )
        jax_image_steps = _torch_to_jax(
            actual_image_steps.to(dtype=torch.int32)
        )

        if bool(needs_gradient):
            (_sum_cost, costs), grad = _jax_batched_value_and_grad(
                jax_dense,
                jax_text_lengths,
                jax_image_steps,
                float(gamma),
            )
            grad_torch = _jax_to_torch(
                grad,
                device=transition_costs_dense.device,
                dtype=transition_costs_dense.dtype,
            )
        else:
            costs = jax_batched_soft_span_dtw_cost(
                jax_dense,
                jax_text_lengths,
                jax_image_steps,
                float(gamma),
            )
            grad_torch = None

        costs_torch = _jax_to_torch(
            costs,
            device=transition_costs_dense.device,
            dtype=transition_costs_dense.dtype,
        ).reshape(batch_size)
        ctx.needs_gradient = bool(needs_gradient)
        if grad_torch is not None:
            ctx.save_for_backward(grad_torch)
        return costs_torch

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.needs_gradient:
            return None, None, None, None, None
        (grad_dense,) = ctx.saved_tensors
        scale = grad_output.to(grad_dense.dtype).view(
            grad_output.shape[0], *([1] * (grad_dense.ndim - 1))
        )
        return scale * grad_dense, None, None, None, None
