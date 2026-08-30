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
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    original = os.environ.get("ORIGINAL_CUDA_VISIBLE_DEVICES", visible).strip()
    selected = visible

    if world_size > 1:
        # GeForce RTX 4090 nodes can expose two CUDA devices while CUDA peer
        # access between them is unavailable. NCCL otherwise attempts a P2P
        # transport and aborts during the first DDP collective with
        # "peer access is not supported between these two devices". Disable
        # NCCL P2P by default for multi-rank runs so NCCL falls back to shared
        # host memory. Users can explicitly override this before launch on
        # hardware where P2P is known to work.
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")

        if visible:
            devices = [token.strip() for token in visible.split(",") if token.strip()]
            if len(devices) > 1:
                if local_rank >= len(devices):
                    raise RuntimeError(
                        f"LOCAL_RANK={local_rank}, but CUDA_VISIBLE_DEVICES={visible!r}."
                    )
                selected = devices[local_rank]
                os.environ["CUDA_VISIBLE_DEVICES"] = selected
            elif len(devices) == 1:
                # A one-device list is valid for multi-rank torchrun only when the
                # shell rank wrapper selected that device independently per rank.
                wrapper_selected = os.environ.get("RANK_SELECTED_CUDA_DEVICE", "").strip()
                wrapper_isolated = os.environ.get("RANK_WRAPPER_ISOLATED", "0").strip().lower()
                if wrapper_isolated not in {"1", "true", "yes", "on"} or wrapper_selected != devices[0]:
                    raise RuntimeError(
                        "WORLD_SIZE is greater than one, but this process sees only "
                        f"CUDA_VISIBLE_DEVICES={visible!r} without verified per-rank isolation. "
                        "Launch through scripts/train/run_model_full_quality.sh with bash, "
                        "not by calling sbatch on that wrapper directly."
                    )
                selected = devices[0]
            else:
                raise RuntimeError("CUDA_VISIBLE_DEVICES contains no usable devices.")
        else:
            raise RuntimeError(
                "WORLD_SIZE is greater than one, but CUDA_VISIBLE_DEVICES is unset. "
                "The per-rank GPU wrapper did not run."
            )

    os.environ["ALIGNMENT_ORIGINAL_CUDA_VISIBLE_DEVICES"] = original
    os.environ["ALIGNMENT_SELECTED_CUDA_DEVICE"] = selected
    return RankDeviceSelection(
        local_rank=local_rank,
        world_size=world_size,
        original_visible_devices=original,
        selected_device=selected,
    )
