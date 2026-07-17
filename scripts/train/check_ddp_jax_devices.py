#!/usr/bin/env python3
"""Verify one Torch/JAX runtime per torchrun rank before long DDP training."""
import os


local_rank = int(os.environ.get("LOCAL_RANK", "0"))
world_size = int(os.environ.get("WORLD_SIZE", "1"))
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if world_size > 1:
    if visible:
        devices = [item.strip() for item in visible.split(",") if item.strip()]
        if len(devices) > 1:
            os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import torch
import torch.distributed as dist
from torch.utils import dlpack as torch_dlpack


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    torch.cuda.set_device(0)
    if world_size > 1:
        dist.init_process_group("nccl", init_method="env://")

    source = torch.tensor([float(local_rank + 1)], device="cuda:0")
    jax_value = jax.dlpack.from_dlpack(torch_dlpack.to_dlpack(source))
    jax_result = jax_value * 2.0
    roundtrip = torch_dlpack.from_dlpack(jax.dlpack.to_dlpack(jax_result))

    reduced = source.clone()
    if world_size > 1:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

    report = {
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": local_rank,
        "visible": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_device": torch.cuda.get_device_name(0),
        "jax_devices": [str(device) for device in jax.devices()],
        "dlpack_value": float(roundtrip.item()),
        "allreduce_sum": float(reduced.item()),
    }

    reports = [report]
    if world_size > 1:
        reports = [None for _ in range(world_size)]
        dist.all_gather_object(reports, report)

    if int(os.environ.get("RANK", "0")) == 0:
        for item in sorted(reports, key=lambda value: value["rank"]):
            print(item, flush=True)
        expected = world_size * (world_size + 1) / 2.0
        if any(abs(item["allreduce_sum"] - expected) > 1e-5 for item in reports):
            raise RuntimeError(
                f"NCCL all-reduce failed: expected {expected}, reports={reports}"
            )
        print("DDP/JAX preflight passed.", flush=True)

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
