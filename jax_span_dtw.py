import warnings

import torch


_warned_dlpack_fallback = False


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


def _masked_softmin(candidates, gamma):
    """Stable soft-min over valid candidates only.

    Invalid transitions are ignored completely. This avoids fake invalid paths
    and prevents the impossible huge negative costs seen in the offline log.
    """
    _require_jax()
    finite = jnp.isfinite(candidates)
    has_candidate = jnp.any(finite)
    safe_candidates = jnp.where(finite, candidates, jnp.inf)
    min_value = jnp.min(safe_candidates)
    min_value = jnp.where(has_candidate, min_value, jnp.array(0.0, dtype=candidates.dtype))
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
        # Every transition consumes at least one text character, so cells in the
        # same row only depend on previous rows. That lets JAX evaluate a whole
        # row's image positions in parallel.
        def cell_value(j):
            def candidate(span_len, window_count):
                prev_i = i - span_len
                prev_j = j - window_count
                valid_index = (i > 0) & (j > 0) & (prev_i >= 0) & (prev_j >= 0)
                safe_prev_i = jnp.maximum(prev_i, 0)
                safe_prev_j = jnp.maximum(prev_j, 0)
                prev_value = dp_in[safe_prev_i, safe_prev_j]
                transition = transition_costs_dense[
                    span_len,
                    window_count,
                    safe_prev_i,
                    safe_prev_j,
                ]
                value = prev_value + transition
                valid_transition = jnp.isfinite(transition)
                valid_value = valid_index & jnp.isfinite(prev_value) & valid_transition
                return jnp.where(valid_value, value, inf)

            candidates = []
            for span_len in range(1, max_span_len + 1):
                for window_count in range(1, max_window_count + 1):
                    candidates.append(candidate(span_len, window_count))
            value = _masked_softmin(jnp.stack(candidates), gamma)
            return jnp.where((i == 0) & (j == 0), dp_in[0, 0], value)

        columns = jnp.arange(image_steps + 1)
        row_values = jax.vmap(cell_value)(columns)
        return dp_in.at[i, :].set(row_values)

    dp = jax.lax.fori_loop(0, text_steps + 1, fill_row, dp)
    actual_text_length = jnp.asarray(actual_text_length, dtype=jnp.int32)
    actual_image_steps = jnp.asarray(actual_image_steps, dtype=jnp.int32)
    return jax.lax.dynamic_slice(
        dp,
        (actual_text_length, actual_image_steps),
        (1, 1),
    )[0, 0]


if is_jax_available():
    # This is equivalent to @jax.jit, but keeps importing this module safe when
    # JAX is not installed and the torch backend is being used.
    jax_soft_span_dtw_cost = jax.jit(jax_soft_span_dtw_cost)
    _jax_value_and_grad = jax.jit(jax.value_and_grad(jax_soft_span_dtw_cost, argnums=0))
else:
    _jax_value_and_grad = None


def _warn_dlpack_fallback(reason):
    global _warned_dlpack_fallback
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
    try:
        from torch.utils import dlpack as torch_dlpack

        return jax.dlpack.from_dlpack(torch_dlpack.to_dlpack(tensor.detach().contiguous()))
    except Exception as exc:
        _warn_dlpack_fallback(exc)
        return jnp.asarray(tensor.detach().cpu().numpy())


def _jax_to_torch(array, device, dtype):
    _require_jax()
    try:
        from torch.utils import dlpack as torch_dlpack

        tensor = torch_dlpack.from_dlpack(jax.dlpack.to_dlpack(array))
        return tensor.to(device=device, dtype=dtype)
    except Exception as exc:
        _warn_dlpack_fallback(exc)
        return torch.as_tensor(jax.device_get(array), device=device, dtype=dtype)


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
        jax_transition_costs = _torch_to_jax(transition_costs_dense)
        jax_actual_text_length = jnp.asarray(int(actual_text_length), dtype=jnp.int32)
        jax_actual_image_steps = jnp.asarray(int(actual_image_steps), dtype=jnp.int32)

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
            grad_output.to(grad_transition_costs.dtype) * grad_transition_costs,
            None,
            None,
            None,
            None,
        )
