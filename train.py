#!/usr/bin/env python3
"""Single user-facing trainer for AlignmentProject.

Usage:
    # Train from scratch
    python train.py --dataset DataSet/ArabicDataset

    # Fine-tune from pretrained weights
    python train.py --dataset DataSet/ArabicDataset \
        --weights Weights/vit_synthetic/model_latest.pth

All architecture, loss, optimization, augmentation, and runtime settings live in
Parameters.py. Supplying --weights is the only switch that turns the run into
fine-tuning.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from runtime_device_setup import isolate_local_rank_cuda_device

RANK_DEVICE = isolate_local_rank_cuda_device()

import Parameters as P

P.export_environment()


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
    explicit = os.environ.get("HF_HOME", "").strip()
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
    for candidate in candidates:
        if candidate.is_dir() and _contains_cached_model(
            candidate, P.arabic_text_model_name
        ):
            os.environ["HF_HOME"] = str(candidate)
            os.environ.pop("TRANSFORMERS_CACHE", None)
            return
    if explicit:
        os.environ["HF_HOME"] = explicit
        return
    raise RuntimeError(
        f"Could not find an offline cache for {P.arabic_text_model_name}. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


_resolve_hf_home()

import trainer_core as base

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import model_backend
import span_alignment_loss
from ddp_runtime_policy import resolve_ddp_static_graph
from distributed_runtime_guard import install_distributed_runtime_guard
from epoch_subset_sampling import install_epoch_subset_sampling
from fast_hard_alignment import hard_span_dtw_path_fast
from jax_batch_bucketing import install_jax_batch_padding
from job_id_runtime import resolve_training_job_id
from training_optimizations import install as install_optimizations
from training_stability import install_training_stability
from unified_line_geometry import install_training_geometry
from vit_checkpoint_migration import install as install_vit_checkpoint_migration

span_alignment_loss.hard_span_dtw_path = hard_span_dtw_path_fast
base.hard_span_dtw_path = hard_span_dtw_path_fast
install_jax_batch_padding()

# Install the optimized trainer. Its optimized_compute_batch_loss keeps line1
# and line2 in two independent visual-model forward calls.
install_optimizations(base)

# Keep the historical baseline loss untouched. This helper only allows an old
# 4-layer ViT checkpoint to initialize the new 1-layer ViT by loading layer 0.
install_vit_checkpoint_migration(base)
install_distributed_runtime_guard(base)
install_epoch_subset_sampling(base)
_GEOMETRY_CONFIG = install_training_geometry()

model_backend.install_training_backend(base)


def _branch_build_image_embedding(stride):
    return model_backend.build_visual_model(
        window_size=P.window_size,
        stride=stride,
        vector_size=P.vector_size,
        device=P.device,
        use_flip=(P.lang.lower() == "arabic"),
        use_bilstm=P.use_bilstm,
        bilstm_layers=P.bilstm_layers,
        bilstm_hidden_dim=P.bilstm_hidden_dim,
        use_local_grouping=P.use_local_window_grouping,
        local_group_size=P.local_group_size,
    )


base.build_image_embedding = _branch_build_image_embedding

_original_model_config = base.model_config


def _model_config(stride, args):
    config = _original_model_config(stride, args)
    config.update(_GEOMETRY_CONFIG)
    config.update(model_backend.visual_model_config())
    config.update(
        {
            "experiment_name": P.experiment_name,
            "configuration_source": "Parameters.py",
            "initialization": "pretrained" if args.pretrained_weights else "scratch",
            "dataset_type": args.dataset_type,
            "dataset_path": args.data_dir,
            "paired_visual_forward": "separate",
        }
    )
    install_training_stability(base, config, args.job_id)
    return config


base.model_config = _model_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset directory. Dataset type is resolved from Parameters.py/manifest.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        help="Optional pretrained model. If supplied, the run is fine-tuning.",
    )
    return parser.parse_args()


def _resolve_dataset_type(dataset: Path) -> str:
    configured = str(P.dataset_type).strip().lower()
    if configured in {"real", "synthetic"}:
        return configured
    if configured != "auto":
        raise ValueError("Parameters.dataset_type must be auto, real, or synthetic")
    return "real" if (dataset / P.real_manifest_name).is_file() else "synthetic"


def _training_args(cli: argparse.Namespace) -> SimpleNamespace:
    dataset = Path(cli.dataset).expanduser().resolve()
    if not dataset.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset}")

    weights = None
    if cli.weights:
        weights_path = Path(cli.weights).expanduser().resolve()
        if not weights_path.is_file():
            raise SystemExit(f"Pretrained weights do not exist: {weights_path}")
        weights = str(weights_path)

    finetune = weights is not None
    resolved_dataset_type = _resolve_dataset_type(dataset)
    job_id = resolve_training_job_id(P.experiment_name, finetune=finetune)
    os.environ["DATASET_TYPE"] = resolved_dataset_type

    return SimpleNamespace(
        job_id=job_id,
        data_dir=str(dataset),
        dataset_type=resolved_dataset_type,
        augment=P.real_augment if resolved_dataset_type == "real" else False,
        train_samples_per_epoch=(
            P.real_train_samples_per_epoch if resolved_dataset_type == "real" else None
        ),
        num_samples=P.num_samples,
        pretrained_weights=weights,
        resume=None,
        finetune=finetune,
        window_size=None,
        stride_ratio=None,
        window_overlap_mode=None,
        negative_mode=None,
        epochs=P.finetune_epochs if finetune else P.epochs,
        learning_rate=P.finetune_learning_rate if finetune else P.learning_rate,
        num_negatives=None,
        use_bilstm=None,
        use_local_hard_negatives=None,
        local_hard_negative_weight=None,
        image_variance_loss_weight=None,
        use_image_pair_contrastive=None,
        image_pair_loss_weight=None,
    )


def _validate_constructed_backend(model: nn.Module) -> None:
    backend = str(model_backend.MODEL_NAME).strip().lower()
    keys = tuple(model.state_dict().keys())
    has_vit = any(key.startswith("vit_encoder.") for key in keys)
    has_cnn = any(key.startswith("cnn_encoder.") for key in keys)
    has_bilstm = any(key.startswith("sequence_encoder.bilstm.") for key in keys)
    if backend == "vit" and (not has_vit or has_cnn or has_bilstm):
        raise RuntimeError(
            "ViT branch built the wrong model: "
            f"has_vit={has_vit} has_cnn={has_cnn} has_bilstm={has_bilstm}"
        )


def main() -> None:
    base.CTX.initialize()
    try:
        base._strip_torchrun_rank_arguments()
        cli = parse_args()
        args = _training_args(cli)
        base._seed_everything(P.train_seed, base.CTX.rank)

        stride = base.compute_stride(
            P.window_size, P.stride_ratio, P.window_overlap_mode
        )
        train_loader, valid_loader, _test_loader, train_sampler = (
            base.select_dataloaders(args)
        )
        text_encoder = base.build_text_encoder()
        raw_model = base.build_image_embedding(stride).to(P.device)
        _validate_constructed_backend(raw_model)
        resume_payload = base._load_initial_states(args, raw_model, text_encoder)
        model_backend.prepare_visual_model(raw_model)
        base._broadcast_trainable_text_parameters(text_encoder)

        model: nn.Module = raw_model
        static_graph = resolve_ddp_static_graph()
        if base.CTX.enabled:
            ddp_kwargs = {
                "device_ids": [0],
                "output_device": 0,
                "broadcast_buffers": False,
                "find_unused_parameters": False,
                "gradient_as_bucket_view": True,
            }
            if static_graph.enabled:
                ddp_kwargs["static_graph"] = True
            try:
                model = DDP(raw_model, **ddp_kwargs)
            except TypeError:
                ddp_kwargs.pop("static_graph", None)
                model = DDP(raw_model, **ddp_kwargs)

        criterion = base.build_criterion()
        config = base.model_config(stride, args)
        config.update(
            {
                "hf_home": os.environ.get("HF_HOME", ""),
                "original_cuda_visible_devices": RANK_DEVICE.original_visible_devices,
                "selected_cuda_device": RANK_DEVICE.selected_device,
                "ddp_static_graph": static_graph.enabled,
                "ddp_static_graph_reason": static_graph.description,
            }
        )

        if base.CTX.is_main:
            mode = "fine-tune" if args.pretrained_weights else "scratch"
            print("AlignmentProject global trainer", flush=True)
            print(f"  backend      = {model_backend.MODEL_NAME}", flush=True)
            print(f"  mode         = {mode}", flush=True)
            print(f"  dataset      = {args.data_dir}", flush=True)
            print(f"  dataset_type = {args.dataset_type}", flush=True)
            print(f"  weights      = {args.pretrained_weights or '<none>'}", flush=True)
            print(f"  output       = Weights/{args.job_id}", flush=True)
            print(f"  epochs/lr    = {args.epochs}/{args.learning_rate}", flush=True)
            print(f"  window/stride= {P.window_size}/{stride}", flush=True)
            print(
                f"  vit          = layers={P.vit_layers} heads={P.vit_heads} "
                f"binarize_rgb={P.vit_binarize_input} method=otsu",
                flush=True,
            )
            print("  visual_fwd   = separate line1 / line2", flush=True)
            print(
                f"  variance     = weight={P.image_variance_loss_weight} "
                f"target_std={P.image_variance_target_std}",
                flush=True,
            )
            print(
                f"  batch        = {P.batch_size} per GPU x {base.CTX.world_size} GPUs "
                f"x {P.gradient_accumulation_steps} accumulation",
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
