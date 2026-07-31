"""Configurable DDP initialization for long per-rank JAX compilations."""
from __future__ import annotations

from datetime import timedelta
import os

import torch
import torch.distributed as dist


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        value = int(default)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def install_distributed_runtime_guard(base_module) -> None:
    """Install a timeout-aware replacement for ``DistributedContext.initialize``.

    Dense Span-DTW uses JAX/XLA. A rank may compile a previously unseen shape while
    another rank is already waiting at a PyTorch collective. The default collective
    timeout is too short for that first cold-cache compile on the cluster. This
    replacement keeps the existing one-GPU-per-rank contract and makes the timeout
    explicit and visible in the log.
    """

    context = base_module.CTX

    def initialize() -> None:
        if context.enabled and not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU DDP training requires CUDA GPUs.")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        timeout_seconds = _positive_int("DIST_TIMEOUT_SECONDS", 7200)
        if context.enabled and not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(seconds=timeout_seconds),
            )
        base_module.P.device = context.device
        print(
            f"distributed_runtime rank={context.rank} "
            f"world={context.world_size} timeout_seconds={timeout_seconds}",
            flush=True,
        )

    context.initialize = initialize
