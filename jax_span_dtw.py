import warnings

import torch


INF = 1e6
_warned_dlpack_fallback = False


def _import_jax():
    try:
        import jax
        import jax.numpy as jnp
        import jax.scipy.special as jsp_special
    except ImportError as exc:
        return None, None, None, exc
    return jax, jnp, jsp_special, None


jax, jnp, jsp_special, _JAX_IMPORT_ERROR = _import_jax()


def is_jax_available():
    return _JAX_IMPORT_ERROR is None


def _require_jax():
    if _JAX_IMPORT_ERROR is not None:
        raise RuntimeError(
            "SPAN_DTW_BACKEND=jax requires JAX to be installed. "
            "Use SPAN_DTW_BACKEND=torch or install the JAX packages from requirements.txt."
        ) from _JAX_IMPORT_ERROR
    return jax, jnp, jsp_special


def _jax_softmin(candidates, gamma):
    _require_jax()
    return -gamma * jsp_special.logsumexp(-candidates / gamma)


def jax_soft_span_dtw_cost(transition_costs_dense, gamma):
    _require_jax()
    max_span_len = transition_costs_dense.shape[0] - 1
    max_window_count = transition_costs_dense.shape[1] - 1
    text_steps = transition_costs_dense.shape[2] - 1
    image_steps = transition_costs_dense.shape[3] - 1
    dtype = transition_costs_dense.dtype
    inf = jnp.array(INF, dtype=dtype)

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
                valid = (i > 0) & (j > 0) & (prev_i >= 0) & (prev_j >= 0)
                safe_prev_i = jnp.maximum(prev_i, 0)
                safe_prev_j = jnp.maximum(prev_j, 0)
                value = (
                    dp_in[safe_prev_i, safe_prev_j]
                    + transition_costs_dense[span_len, window_count, safe_prev_i, safe_prev_j]
                )
                return jnp.where(valid, value, inf)

            candidates = []
            for span_len in range(1, max_span_len + 1):
                for window_count in range(1, max_window_count + 1):
                    candidates.append(candidate(span_len, window_count))
            value = _jax_softmin(jnp.stack(candidates), gamma)
            return jnp.where((i == 0) & (j == 0), dp_in[0, 0], value)

        columns = jnp.arange(image_steps + 1)
        row_values = jax.vmap(cell_value)(columns)
        return dp_in.at[i, :].set(row_values)

    dp = jax.lax.fori_loop(0, text_steps + 1, fill_row, dp)
    return dp[text_steps, image_steps]


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
    def forward(ctx, transition_costs_dense, gamma, needs_gradient):
        _require_jax()
        jax_transition_costs = _torch_to_jax(transition_costs_dense)
        if needs_gradient:
            cost, grad = _jax_value_and_grad(jax_transition_costs, float(gamma))
        else:
            cost = jax_soft_span_dtw_cost(jax_transition_costs, float(gamma))
            grad = None

        cost_torch = _jax_to_torch(
            cost,
            device=transition_costs_dense.device,
            dtype=transition_costs_dense.dtype,
        )
        ctx.has_gradient = bool(needs_gradient)
        if ctx.has_gradient:
            grad_torch = _jax_to_torch(
                grad,
                device=transition_costs_dense.device,
                dtype=transition_costs_dense.dtype,
            )
            ctx.save_for_backward(grad_torch)
        return cost_torch.reshape(())

    @staticmethod
    def backward(ctx, grad_output):
        if not ctx.has_gradient:
            return None, None, None
        (grad_transition_costs,) = ctx.saved_tensors
        return grad_output.to(grad_transition_costs.dtype) * grad_transition_costs, None, None
