"""Early CUDA visibility setup for torchrun processes.

This module intentionally imports only the Python standard library. Call
``isolate_local_rank_cuda_device`` before importing Torch, JAX, Transformers, or
any project module that imports them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RankDeviceSelection:
    local_rank: int
    world_size: int
    original_visible_devices: str
    selected_device: str


def isolate_local_rank_cuda_device() -> RankDeviceSelection:
    """Expose exactly one physical CUDA device to each local torchrun rank."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    original = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    selected = original

    if world_size > 1:
        if original:
            devices = [token.strip() for token in original.split(",") if token.strip()]
            if len(devices) > 1:
                if local_rank >= len(devices):
                    raise RuntimeError(
                        f"LOCAL_RANK={local_rank}, but CUDA_VISIBLE_DEVICES={original!r}."
                    )
                selected = devices[local_rank]
                os.environ["CUDA_VISIBLE_DEVICES"] = selected
            elif len(devices) == 1:
                # This is valid only when another early-isolation layer already
                # reduced visibility for this process.
                selected = devices[0]
            else:
                raise RuntimeError("CUDA_VISIBLE_DEVICES contains no usable devices.")
        else:
            selected = str(local_rank)
            os.environ["CUDA_VISIBLE_DEVICES"] = selected

    os.environ["ALIGNMENT_ORIGINAL_CUDA_VISIBLE_DEVICES"] = original
    os.environ["ALIGNMENT_SELECTED_CUDA_DEVICE"] = selected
    return RankDeviceSelection(
        local_rank=local_rank,
        world_size=world_size,
        original_visible_devices=original,
        selected_device=selected,
    )
