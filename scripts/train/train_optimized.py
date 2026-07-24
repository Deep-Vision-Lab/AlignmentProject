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


def _contains_cached_model(cache_root: Path, model_name: str) -> bool:
    slug = "models--" + model_name.replace("/", "--")
    for layout in (cache_root, cache_root / "hub"):
        snapshots = layout / slug / "snapshots"
        if not snapshots.is_dir():
            continue
        for snapshot in snapshots.iterdir():
            if not snapshot.is_dir() or not (snapshot / "config.json").is_file():
                continue
            if any(snapshot.glob("model*.safetensors")) or any(
                snapshot.glob("pytorch_model*.bin")
            ):
                return True
    return False


def _resolve_hf_home() -> None:
    """Resolve the offline AraBERT cache before importing Transformers modules."""
    explicit = os.environ.get("HF_HOME", "").strip()
    model_name = os.environ.get(
        "ARABIC_TEXT_MODEL_NAME", "aubmindlab/bert-base-arabertv02"
    )
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            PROJECT_DIR / ".hf_cache",
            Path(str(PROJECT_DIR) + "_clone") / ".hf_cache",
            Path.home() / ".cache" / "huggingface",
        ]
    )
    transformers_cache = os.environ.get("TRANSFORMERS_CACHE", "").strip()
    if transformers_cache:
        candidates.append(Path(transformers_cache).expanduser())

    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve() if candidate.exists() else candidate
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if candidate.is_dir() and _contains_cached_model(candidate, model_name):
            os.environ["HF_HOME"] = str(candidate)
            os.environ.pop("TRANSFORMERS_CACHE", None)
            return

    if explicit:
        os.environ["HF_HOME"] = explicit
        return
    raise RuntimeError(
        f"Could not find an offline cache for {model_name}. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


_resolve_hf_home()

import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import span_alignment_loss
import train as base
from fast_hard_alignment import hard_span_dtw_path_fast
from jax_batch_bucketing import install_jax_batch_padding
from training_optimizations import install, prepare_raw_model


# Patch the discrete path decoder and stabilize JAX batch dimensions before any
# criterion is built.
span_alignment_loss.hard_span_dtw_path = hard_span_dtw_path_fast
base.hard_span_dtw_path = hard_span_dtw_path_fast
install_jax_batch_padding()
install(base)


def _spawn_rebuild_single_gpu_loaders(train_loader, valid_loader, test_loader):
    """Use the same spawn/prefetch loader settings outside DDP as inside DDP."""
    if base.CTX.enabled:
        return train_loader, valid_loader, test_loader
    if train_loader.num_workers <= 0:
        return train_loader, valid_loader, test_loader
    return (
        base._rebuild_loader(train_loader, train_loader.sampler),
        base._rebuild_loader(valid_loader, valid_loader.sampler),
        base._rebuild_loader(test_loader, test_loader.sampler),
    )


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
        train_loader, valid_loader, test_loader, train_sampler = (
            base.select_dataloaders(args)
        )
        train_loader, valid_loader, test_loader = (
            _spawn_rebuild_single_gpu_loaders(
                train_loader, valid_loader, test_loader
            )
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
        config["hf_home"] = os.environ.get("HF_HOME", "")
        config["span_dtw_batch_bucket_size"] = int(
            os.environ.get("SPAN_DTW_BATCH_BUCKET_SIZE", "8")
        )
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
