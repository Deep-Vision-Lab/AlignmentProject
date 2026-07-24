#!/usr/bin/env python3
"""Optimized full-quality training entrypoint.

This imports the proven ``train.py`` implementation, installs isolated hot-path
replacements, prepares the visual model before DDP construction, and then runs
with the same datasets/checkpoint format.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Allow execution as ``python scripts/train/train_optimized.py``.
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import train as base
from training_optimizations import install, prepare_raw_model


install(base)


def main():
    base.CTX.initialize()
    try:
        base._seed_everything(base._env_int("TRAIN_SEED", 42), base.CTX.rank)
        base._strip_torchrun_rank_arguments()
        args = base.parse_args()
        base.apply_overrides(args)
        if args.resume and args.pretrained_weights:
            raise SystemExit("Use either --resume or --pretrained_weights, not both.")

        stride = base.compute_stride(
            base.P.window_size,
            base.P.stride_ratio,
            base.P.window_overlap_mode,
        )
        train_loader, valid_loader, _test_loader, train_sampler = (
            base.select_dataloaders(args)
        )
        text_encoder = base.build_text_encoder()
        raw_model = base.build_image_embedding(stride).to(base.P.device)
        resume_payload = base._load_initial_states(
            args, raw_model, text_encoder
        )
        prepare_raw_model(raw_model)
        base._broadcast_trainable_text_parameters(text_encoder)

        model: nn.Module = raw_model
        if base.CTX.enabled:
            ddp_kwargs = {
                "device_ids": [0],
                "output_device": 0,
                "broadcast_buffers": False,
                "find_unused_parameters": False,
                "gradient_as_bucket_view": True,
            }
            # PyTorch 2.0 accepts static_graph in DDP. Fall back cleanly on older
            # installations rather than preventing a run.
            if os.environ.get("DDP_STATIC_GRAPH", "1").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }:
                ddp_kwargs["static_graph"] = True
            try:
                model = DDP(raw_model, **ddp_kwargs)
            except TypeError:
                ddp_kwargs.pop("static_graph", None)
                model = DDP(raw_model, **ddp_kwargs)

        criterion = base.build_criterion()
        config = base.model_config(stride, args)
        if base.CTX.is_main:
            print("resolved optimized configuration:", flush=True)
            for key in sorted(config):
                print(f"  {key}={config[key]}", flush=True)
            print(
                f"train_batches_per_rank={len(train_loader)} "
                f"valid_batches_rank0={len(valid_loader)} "
                f"epochs={args.epochs} lr={args.learning_rate}",
                flush=True,
            )

        base.CTX.barrier()
        base.train(
            model,
            text_encoder,
            criterion,
            train_loader,
            valid_loader,
            train_sampler,
            args,
            config,
            resume_payload=resume_payload,
        )
    finally:
        base.CTX.close()


if __name__ == "__main__":
    main()
