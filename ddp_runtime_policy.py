"""Runtime policy for safe DDP settings.

This module intentionally imports neither Torch nor JAX so the policy can be
resolved and tested without initializing CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import MutableMapping


def _flag(env: MutableMapping[str, str], name: str, default: bool = False) -> bool:
    value = env.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _integer(env: MutableMapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _number(env: MutableMapping[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, default))
    except (TypeError, ValueError):
        return float(default)


@dataclass(frozen=True)
class DDPStaticGraphDecision:
    requested: bool
    enabled: bool
    reasons: tuple[str, ...]
    forced: bool

    @property
    def description(self) -> str:
        if self.enabled:
            return "enabled by explicit request" if self.requested else "disabled"
        if not self.requested:
            return "disabled by default"
        if self.reasons:
            return "disabled for safety: " + "; ".join(self.reasons)
        return "disabled"


def resolve_ddp_static_graph(
    env: MutableMapping[str, str] | None = None,
) -> DDPStaticGraphDecision:
    """Resolve whether DDP ``static_graph`` is safe for this training run.

    The optimized objective has data- and iteration-dependent auxiliary paths.
    PyTorch's static graph contract requires iteration control flow to remain
    unchanged.  PyTorch 2.0 can fail inside the C++ autograd engine instead of
    producing a useful Python exception when that contract is violated.

    ``FORCE_DDP_STATIC_GRAPH=1`` remains available only for controlled ablations.
    """

    if env is None:
        env = os.environ

    requested = _flag(env, "DDP_STATIC_GRAPH", False)
    forced = _flag(env, "FORCE_DDP_STATIC_GRAPH", False)
    reasons: list[str] = []

    accumulation = max(1, _integer(env, "GRADIENT_ACCUMULATION_STEPS", 1))
    if accumulation > 1:
        reasons.append(f"gradient accumulation uses no_sync across {accumulation} microbatches")

    local_enabled = _flag(env, "USE_LOCAL_HARD_NEGATIVES", True) and _number(
        env, "LOCAL_HARD_NEGATIVE_WEIGHT", 0.25
    ) > 0
    if local_enabled:
        local_every = max(1, _integer(env, "LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES", 2))
        reasons.append(
            "local hard-negative control flow is data/iteration dependent"
            + (f" (every {local_every} batches)" if local_every > 1 else "")
        )

    pair_enabled = _flag(env, "USE_IMAGE_PAIR_CONTRASTIVE", True) and _number(
        env, "IMAGE_PAIR_LOSS_WEIGHT", 0.40
    ) > 0
    if pair_enabled:
        pair_every = max(1, _integer(env, "IMAGE_PAIR_EVERY_N_BATCHES", 1))
        reasons.append(
            "image-pair/order regions are data dependent"
            + (f" (every {pair_every} batches)" if pair_every > 1 else "")
        )

    if str(env.get("SPAN_DTW_BACKEND", "jax")).strip().lower() == "jax":
        reasons.append("the loss contains a custom Torch-to-JAX autograd bridge")

    enabled = bool(requested and (forced or not reasons))
    # Keep downstream code, checkpoint metadata, and the second defensive DDP
    # setup in training_optimizations.py consistent with the effective decision.
    env["DDP_STATIC_GRAPH"] = "1" if enabled else "0"
    env["DDP_STATIC_GRAPH_EFFECTIVE"] = "1" if enabled else "0"

    return DDPStaticGraphDecision(
        requested=requested,
        enabled=enabled,
        reasons=tuple(reasons),
        forced=forced,
    )
