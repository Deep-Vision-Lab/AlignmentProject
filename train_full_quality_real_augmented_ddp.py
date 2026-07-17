#!/usr/bin/env python3
"""Multi-GPU DDP entry point for full-quality augmented real-data training.

Launch with torchrun, one process per GPU. Each process owns one CUDA device and
one independent JAX runtime, so per-sample Span-DTW work is split across GPUs.
The image encoder is synchronized by DistributedDataParallel. The Arabic text
encoder is invoked outside the image-model forward, so its small trainable
projection/norm gradients are synchronized explicitly after AMP unscaling.
"""
from __future__ import annotations

import math
import os
import random
import sys
import time
from typing import Any, Dict, Iterator


def _isolate_rank_cuda_device() -> tuple[int, int]:
    """Expose only this torchrun process's GPU before importing Torch/JAX."""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return local_rank, world_size

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        devices = [token.strip() for token in visible.split(",") if token.strip()]
        if len(devices) > 1:
            if local_rank >= len(devices):
                raise RuntimeError(
                    f"LOCAL_RANK={local_rank} but CUDA_VISIBLE_DEVICES={visible!r}."
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    return local_rank, world_size


_ORIGINAL_LOCAL_RANK, _ENV_WORLD_SIZE = _isolate_rank_cuda_device()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler

# Importing this profile installs the compositional pair loss, differentiable
# contextual-order loss, and augmented real-data loader on train.py.
import train_full_quality_real_augmented as profile
import train as base
import Parameters as P
from LossFunctionWithHelpers import ContrastiveSoftDTW
from span_alignment_loss import SpanContrastiveSoftDTW


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation samples without padding or duplication."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.world_size - 1) // self.world_size


class DistributedContext:
    def __init__(self):
        self.enabled = int(os.environ.get("WORLD_SIZE", "1")) > 1
        self.rank = int(os.environ.get("RANK", "0"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def initialize(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("The DDP training entry point requires CUDA GPUs.")
        torch.cuda.set_device(0)
        if self.enabled and not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        # All imported modules reference the same Parameters module object.
        P.device = self.device

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


CTX = DistributedContext()


def _strip_torchrun_rank_arguments() -> None:
    """Remove launcher rank flags before delegating to train.py argparse."""
    cleaned = [sys.argv[0]]
    skip_next = False
    for argument in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if argument in {"--local-rank", "--local_rank"}:
            skip_next = True
            continue
        if argument.startswith("--local-rank=") or argument.startswith("--local_rank="):
            continue
        cleaned.append(argument)
    sys.argv = cleaned


def _seed_everything(seed: int, rank: int) -> None:
    rank_seed = int(seed) + int(rank)
    random.seed(rank_seed)
    torch.manual_seed(rank_seed)
    torch.cuda.manual_seed_all(rank_seed)
    try:
        import numpy as np

        np.random.seed(rank_seed)
    except ImportError:
        pass


def _rebuild_loader(loader: DataLoader, sampler: Sampler[int]) -> DataLoader:
    kwargs: Dict[str, Any] = {
        "dataset": loader.dataset,
        "batch_size": loader.batch_size,
        "sampler": sampler,
        "shuffle": False,
        "num_workers": loader.num_workers,
        "collate_fn": loader.collate_fn,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "timeout": loader.timeout,
        "worker_init_fn": loader.worker_init_fn,
    }
    if loader.num_workers > 0:
        kwargs["persistent_workers"] = loader.persistent_workers
        kwargs["prefetch_factor"] = loader.prefetch_factor
    return DataLoader(**kwargs)


def build_distributed_loaders(data_dir):
    train_loader, valid_loader, test_loader = profile.build_dataloaders(data_dir)
    if not CTX.enabled:
        return train_loader, valid_loader, test_loader, None

    split_seed = int(os.environ.get("DATASET_SPLIT_SEED", "42"))
    train_sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas=CTX.world_size,
        rank=CTX.rank,
        shuffle=True,
        seed=split_seed,
        drop_last=False,
    )
    valid_sampler = DistributedEvalSampler(
        valid_loader.dataset,
        rank=CTX.rank,
        world_size=CTX.world_size,
    )
    test_sampler = DistributedEvalSampler(
        test_loader.dataset,
        rank=CTX.rank,
        world_size=CTX.world_size,
    )
    return (
        _rebuild_loader(train_loader, train_sampler),
        _rebuild_loader(valid_loader, valid_sampler),
        _rebuild_loader(test_loader, test_sampler),
        train_sampler,
    )


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _broadcast_trainable_text_parameters(text_encoder: nn.Module) -> None:
    if not CTX.enabled:
        return
    for parameter in text_encoder.parameters():
        if parameter.requires_grad:
            dist.broadcast(parameter.data, src=0)


def _allreduce_text_gradients(text_encoder: nn.Module) -> None:
    """Average trainable text-projection gradients across ranks."""
    if not CTX.enabled:
        return
    for parameter in text_encoder.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
        parameter.grad.div_(CTX.world_size)


def _batch_size(batch) -> int:
    if isinstance(batch, dict):
        return len(batch.get("texts1", []))
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return len(batch[0])
    return 1


def _merge_epoch_payload(local_payload: dict) -> tuple[float, dict]:
    payloads = [local_payload]
    if CTX.enabled:
        payloads = [None for _ in range(CTX.world_size)]
        dist.all_gather_object(payloads, local_payload)

    total_weight = sum(float(item.get("weight", 0.0)) for item in payloads)
    if total_weight <= 0:
        return float("nan"), {}

    loss = sum(float(item.get("loss_sum", 0.0)) for item in payloads) / total_weight
    keys = set()
    for item in payloads:
        keys.update((item.get("stats_sum") or {}).keys())

    merged = {}
    for key in keys:
        merged[key] = sum(
            float((item.get("stats_sum") or {}).get(key, 0.0))
            for item in payloads
        ) / total_weight
    return loss, merged


def _accumulate_stats(stats_sum: dict, stats: dict, weight: int) -> None:
    for key, value in stats.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        stats_sum[key] = stats_sum.get(key, 0.0) + numeric * weight


def _format_rank_memory(text_encoder: nn.Module) -> str:
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    cache = ""
    if hasattr(text_encoder, "cache_size_current"):
        cache = f" span_cache={text_encoder.cache_size_current()}"
    return f"gpu={allocated:.2f}/{reserved:.2f}GB{cache}"


def train_one_epoch_ddp(model, text_encoder, criterion, optimizer, scaler, loader):
    model.train()
    text_encoder.train() if base.has_trainable_parameters(text_encoder) else text_encoder.eval()

    loss_sum = 0.0
    total_weight = 0
    stats_sum: Dict[str, float] = {}

    for batch_idx, batch in enumerate(loader):
        batch_started = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss, stats = base.compute_batch_loss(model, text_encoder, criterion, batch)
        forward_elapsed = time.time() - batch_started

        backward_started = time.time()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        _allreduce_text_gradients(text_encoder)
        torch.nn.utils.clip_grad_norm_(
            [parameter for group in optimizer.param_groups for parameter in group["params"]],
            max_norm=1.0,
        )
        scaler.step(optimizer)
        scaler.update()
        backward_elapsed = time.time() - backward_started

        weight = max(1, _batch_size(batch))
        loss_sum += float(loss.detach().item()) * weight
        total_weight += weight
        _accumulate_stats(stats_sum, stats, weight)

        if CTX.is_main:
            memory = ""
            if P.log_memory_every_n_batches > 0 and (
                batch_idx == 0
                or (batch_idx + 1) % P.log_memory_every_n_batches == 0
            ):
                memory = " " + _format_rank_memory(text_encoder)
            print(
                f"rank=0 batch={batch_idx + 1}/{len(loader)} "
                f"local_batch={weight} global_batch={weight * CTX.world_size} "
                f"loss={loss.item():.4f} "
                f"norm_pos={stats.get('norm_pos', float('nan')):.4f} "
                f"norm_neg={stats.get('norm_neg', float('nan')):.4f} "
                f"gap={stats.get('gap', float('nan')):.4f} "
                f"pos_prob={stats.get('pos_prob', float('nan')):.4f} "
                f"local_hard={stats.get('local_hard_neg', 0.0):.4f} "
                f"pair={stats.get('image_pair_loss', 0.0):.4f} "
                f"order={stats.get('order_loss', 0.0):.4f} "
                f"pair_terms={stats.get('pair_terms', 0.0):.1f} "
                f"forward={forward_elapsed:.1f}s "
                f"backward={backward_elapsed:.1f}s "
                f"time={time.time() - batch_started:.1f}s{memory}",
                flush=True,
            )

    return _merge_epoch_payload(
        {"loss_sum": loss_sum, "stats_sum": stats_sum, "weight": total_weight}
    )


@torch.no_grad()
def validate_ddp(model, text_encoder, criterion, loader, max_batches=0):
    model.eval()
    text_encoder.eval()
    loss_sum = 0.0
    total_weight = 0
    stats_sum: Dict[str, float] = {}

    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        loss, stats = base.compute_batch_loss(model, text_encoder, criterion, batch)
        weight = max(1, _batch_size(batch))
        loss_sum += float(loss.detach().item()) * weight
        total_weight += weight
        _accumulate_stats(stats_sum, stats, weight)

    return _merge_epoch_payload(
        {"loss_sum": loss_sum, "stats_sum": stats_sum, "weight": total_weight}
    )


def _save_model_weights_ddp(model, text_encoder, job_id, config) -> str:
    raw_model = _unwrap_model(model)
    path = os.path.join(base.weights_dir(job_id), "model_latest.pth")
    torch.save(
        {
            "model_state_dict": raw_model.state_dict(),
            "image_model_state_dict": raw_model.state_dict(),
            "text_encoder_state_dict": text_encoder.state_dict(),
            "text_embedder_state_dict": text_encoder.state_dict(),
            "text_encoder_class": text_encoder.__class__.__name__,
            "text_embedding_class": text_encoder.__class__.__name__,
            "text_encoder_type": P.text_encoder_type,
            "model_config": config,
        },
        path,
    )
    return path


def _save_checkpoint_ddp(
    model,
    text_encoder,
    optimizer,
    scheduler,
    scaler,
    epoch,
    job_id,
    config,
) -> None:
    raw_model = _unwrap_model(model)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "image_model_state_dict": raw_model.state_dict(),
            "text_encoder_state_dict": text_encoder.state_dict(),
            "text_embedder_state_dict": text_encoder.state_dict(),
            "text_encoder_class": text_encoder.__class__.__name__,
            "text_embedding_class": text_encoder.__class__.__name__,
            "text_encoder_type": P.text_encoder_type,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "model_config": config,
        },
        os.path.join(base.weights_dir(job_id), "checkpoint_latest.pth"),
    )


def train_ddp(
    model,
    text_encoder,
    criterion,
    train_loader,
    valid_loader,
    train_sampler,
    args,
    config,
    resume_payload=None,
):
    trainable_params = list(model.parameters()) + [
        parameter for parameter in text_encoder.parameters() if parameter.requires_grad
    ]
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01,
    )
    scaler = GradScaler(enabled=base.USE_AMP)
    start_epoch = 0

    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if resume_payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume_payload["scaler_state_dict"])
        start_epoch = int(resume_payload["epoch"]) + 1

    run = base.init_wandb(args, config) if CTX.is_main else None
    history = []

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        CTX.barrier()
        started = time.time()
        if CTX.is_main:
            print(f"epoch={epoch + 1}/{args.epochs} world_size={CTX.world_size}", flush=True)

        train_loss, train_stats = train_one_epoch_ddp(
            model,
            text_encoder,
            criterion,
            optimizer,
            scaler,
            train_loader,
        )

        should_validate = (
            ((epoch + 1) % P.valid_every_n_epochs == 0)
            or ((epoch + 1) == args.epochs)
        )
        if should_validate:
            base.clear_text_encoder_cache(text_encoder)
            val_loss, _val_stats = validate_ddp(
                model,
                text_encoder,
                criterion,
                valid_loader,
                max_batches=P.valid_max_batches,
            )
            base.clear_text_encoder_cache(text_encoder)
        else:
            val_loss = float("nan")

        scheduler.step()
        if CTX.is_main:
            base.wandb_log_epoch_metrics(
                run,
                epoch + 1,
                train_loss,
                val_loss,
                train_stats,
            )
            _save_checkpoint_ddp(
                model,
                text_encoder,
                optimizer,
                scheduler,
                scaler,
                epoch,
                args.job_id,
                config,
            )
            _save_model_weights_ddp(model, text_encoder, args.job_id, config)

        if P.clear_span_cache_each_epoch:
            base.clear_text_encoder_cache(text_encoder)
        history.append(train_loss)
        CTX.barrier()

        if CTX.is_main:
            print(
                f"epoch={epoch + 1} train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} elapsed={time.time() - started:.1f}s",
                flush=True,
            )

    if run is not None:
        run.finish()
    return history


def _load_initial_states(args, model, text_encoder):
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=CTX.device)
        model.load_state_dict(base.extract_model_state(resume_payload))
        if "text_encoder_state_dict" in resume_payload:
            text_encoder.load_state_dict(
                resume_payload["text_encoder_state_dict"], strict=False
            )
        elif "text_embedder_state_dict" in resume_payload:
            text_encoder.load_state_dict(
                resume_payload["text_embedder_state_dict"], strict=False
            )
    elif args.pretrained_weights:
        loaded = torch.load(args.pretrained_weights, map_location=CTX.device)
        model.load_state_dict(base.extract_model_state(loaded))
        if isinstance(loaded, dict) and "text_encoder_state_dict" in loaded:
            text_encoder.load_state_dict(loaded["text_encoder_state_dict"], strict=False)
        elif isinstance(loaded, dict) and "text_embedder_state_dict" in loaded:
            text_encoder.load_state_dict(loaded["text_embedder_state_dict"], strict=False)
    return resume_payload


def _build_criterion():
    if P.text_encoder_type == "arabic_span":
        return SpanContrastiveSoftDTW(
            gamma=P.contrastive_soft_dtw_gamma,
            margin=P.contrastive_margin,
            temperature=P.contrastive_temperature,
            max_windows_per_span=P.max_windows_per_span,
            negative_grad_mode=P.span_negative_grad_mode,
            backend=P.span_dtw_backend,
        )
    return ContrastiveSoftDTW(
        gamma=P.contrastive_soft_dtw_gamma,
        use_cuda=True,
        margin=P.contrastive_margin,
        temperature=P.contrastive_temperature,
    )


def main() -> None:
    CTX.initialize()
    try:
        _seed_everything(int(os.environ.get("TRAIN_SEED", "42")), CTX.rank)
        _strip_torchrun_rank_arguments()
        args = base.parse_args()
        base.apply_overrides(args)
        if args.resume and args.pretrained_weights:
            raise SystemExit("Use either --resume or --pretrained_weights, not both.")

        stride = base.compute_stride(
            P.window_size,
            P.stride_ratio,
            P.window_overlap_mode,
        )
        train_loader, valid_loader, _test_loader, train_sampler = (
            build_distributed_loaders(args.data_dir)
        )

        text_encoder = base.build_text_encoder()
        raw_model = base.build_image_embedding(stride).to(CTX.device)
        resume_payload = _load_initial_states(args, raw_model, text_encoder)
        _broadcast_trainable_text_parameters(text_encoder)

        model: nn.Module = raw_model
        if CTX.enabled:
            model = DDP(
                raw_model,
                device_ids=[0],
                output_device=0,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
            )

        criterion = _build_criterion()
        config = base.model_config(stride)
        config.update(
            {
                "distributed": CTX.enabled,
                "distributed_backend": "nccl" if CTX.enabled else "none",
                "world_size": CTX.world_size,
                "per_gpu_batch_size": P.batch_size,
                "global_batch_size": P.batch_size * CTX.world_size,
                "text_gradient_sync": "manual_allreduce",
                "jax_gpu_isolation": True,
            }
        )

        if CTX.is_main:
            print(
                f"job_id={args.job_id} world_size={CTX.world_size} "
                f"per_gpu_batch={P.batch_size} "
                f"global_batch={P.batch_size * CTX.world_size} "
                f"device_per_rank=cuda:0 physical_visibility_isolated=1",
                flush=True,
            )
            print(
                f"train_batches_per_rank={len(train_loader)} "
                f"valid_batches_rank0={len(valid_loader)} "
                f"span_backend={P.span_dtw_backend} "
                f"max_span_chars={P.max_text_span_chars} "
                f"max_windows_per_span={P.max_windows_per_span}",
                flush=True,
            )

        CTX.barrier()
        train_ddp(
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
        CTX.close()


if __name__ == "__main__":
    main()
