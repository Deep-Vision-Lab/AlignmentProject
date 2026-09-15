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
import model_backend

# Import the branch backend before exporting environment/config or constructing
# dataloaders. This branch changes the text encoder to a frozen char codebook,
# enables transcript negatives, and keeps evaluation image-only.
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
    # Character-codebook branches require no HuggingFace model at all.
    if str(P.text_encoder_type).strip().lower() == "char":
        return

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

from ddp_runtime_policy import resolve_ddp_static_graph
from distributed_runtime_guard import install_distributed_runtime_guard
from epoch_subset_sampling import install_epoch_subset_sampling
from job_id_runtime import resolve_training_job_id
from training_optimizations import install as install_optimizations
from training_stability import install_training_stability
from unified_line_geometry import install_training_geometry
from vit_checkpoint_migration import install as install_vit_checkpoint_migration

# Install shared optimization/runtime helpers first. The branch backend then
# replaces compute_batch_loss with positive/negative letter-DTW only.
install_optimizations(base)

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
            "configuration_source": "Parameters.py + branch backend",
            "initialization": "pretrained" if args.pretrained_weights else "scratch",
            "dataset_type": args.dataset_type,
            "dataset_path": args.data_dir,
            "paired_visual_forward": "independent-lines-only",
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
        help="Optional compatible visual initialization. Mismatched shapes are skipped.",
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
    has_resnet18 = any(
        key.startswith("vit_encoder.patch_embedding.backbone.layer4.")
        for key in keys
    )
    has_decoder = any("stroke_decoder" in key for key in keys)
    visual_type = str(
        getattr(model_backend, "VISUAL_ENCODER_TYPE", backend)
    ).strip().lower()
    if visual_type == "vit" and (not has_vit or has_cnn or has_bilstm):
        raise RuntimeError(
            "ViT branch built the wrong model: "
            f"backend={backend} has_vit={has_vit} has_cnn={has_cnn} "
            f"has_bilstm={has_bilstm}"
        )
    if not has_resnet18:
        raise RuntimeError("ResNet18/ViT-Tiny backend is missing ResNet-18 parameters")
    if has_decoder:
        raise RuntimeError("Decoder parameters found even though restoration was removed")
    if not any("fusion_head" in key for key in keys):
        raise RuntimeError("Backend is missing local/context fusion parameters")
    vit = model.vit_encoder
    if int(vit.embed_dim) != 192:
        raise RuntimeError(f"ViT-Tiny embed_dim must be 192, got {vit.embed_dim}")
    if len(vit.encoder.layers) != 12:
        raise RuntimeError("ViT-Tiny must contain 12 transformer layers")


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
        train_loader, valid_loader, test_loader, train_sampler = (
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
            train_size = len(train_loader.dataset)
            valid_size = len(valid_loader.dataset)
            test_size = len(test_loader.dataset)
            dataset_size = train_size + valid_size + test_size
            print(
                f"DATASET path={args.data_dir} type={args.dataset_type} "
                f"size={dataset_size} train={train_size} "
                f"valid={valid_size} test={test_size}",
                flush=True,
            )
            print(
                "MODELS "
                "encoder=ResNet-18[ImageNet1K-pretrained](512->192) "
                "vit=ViT-Tiny[facebook/deit-tiny-patch16-224-pretrained]"
                "(dim=192,layers=12,heads=3,mlp=768)",
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
