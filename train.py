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
import hashlib
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
from architecture_experiment import is_compact
if is_compact(P):
    os.environ.setdefault("EPOCH_MONITORING", "1")
    if os.environ["EPOCH_MONITORING"] != "1":
        raise ValueError("The compact experiment requires end-of-epoch train/validation monitoring")

# Import the branch backend before exporting environment/config or constructing
# dataloaders. This branch changes the text encoder to a frozen char codebook,
# uses positive transcript DTW by default, and keeps evaluation image-only.
P.export_environment()
if os.environ.get("EPOCH_MONITORING", "0") == "1":
    # Explicit smoke controls, not optimizer or objective changes.
    P.batch_size = int(os.environ.get("EXPERIMENT_BATCH_SIZE", P.batch_size))
    P.profile_max_batches = int(os.environ.get("EXPERIMENT_SMOKE_BATCHES", "0"))
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
# replaces compute_batch_loss with positive letter-DTW by default.
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
            "initialization": model_backend.visual_model_config().get("initialization", "pretrained-resnet18+pretrained-deit-tiny"),
            "dataset_type": args.dataset_type,
            "dataset_path": args.data_dir,
            "training_sample_view": (
                "all-page-lines-own-transcript"
                if os.environ.get("REAL_ALL_PAGE_LINES", "0").strip().lower()
                in {"1", "true", "yes", "on"}
                else (
                    "independent-line-transcript"
                    if os.environ.get("REAL_INDEPENDENT_LINES", "0").strip().lower()
                    in {"1", "true", "yes", "on"}
                    else "paired-lines-single-ddp-forward"
                )
            ),
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
    from architecture_experiment import is_compact
    if is_compact(P) and (PROJECT_DIR / "Weights" / job_id).exists():
        raise SystemExit(f"New experiment requires a new JOB_NAME; refusing to overwrite Weights/{job_id}")
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
    from architecture_experiment import is_compact
    if is_compact(P):
        if (vit.embed_dim, len(vit.encoder.layers)) != (128, 5):
            raise RuntimeError("Compact backend dimension/layer mismatch")
        return
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
        # Paired manuscript lines are consolidated into one model forward.
        # Every active visual parameter participates in every batch, so this
        # branch uses DDP static-graph mode across the two GPUs.
        os.environ["DDP_STATIC_GRAPH"] = "1"
        os.environ["DDP_STATIC_GRAPH_EFFECTIVE"] = "1"
        static_graph = resolve_ddp_static_graph()
        if base.CTX.enabled:
            model = DDP(
                raw_model,
                device_ids=[0],
                output_device=0,
                broadcast_buffers=False,
                find_unused_parameters=False,
                gradient_as_bucket_view=True,
                static_graph=bool(static_graph.enabled),
            )

        criterion = base.build_criterion()
        config = base.model_config(stride, args)
        config["parameter_counts"] = {
            "visual_total": sum(p.numel() for p in raw_model.parameters()),
            "visual_trainable": sum(p.numel() for p in raw_model.parameters() if p.requires_grad),
            "text_total": sum(p.numel() for p in text_encoder.parameters()),
            "text_trainable": sum(p.numel() for p in text_encoder.parameters() if p.requires_grad),
            "components": {name: sum(p.numel() for p in module.parameters()) for name, module in raw_model.vit_encoder.named_children()},
        }
        config["codebook_state_sha256"] = hashlib.sha256(b"".join(
            k.encode() + v.detach().cpu().contiguous().numpy().tobytes()
            for k, v in text_encoder.state_dict().items())).hexdigest()
        initialization_path = getattr(raw_model.vit_encoder.patch_embedding, "initialization_source_path", None)
        config["initialization_provenance"] = {
            "resnet_cached_weights": initialization_path,
            "resnet_cached_sha256": hashlib.sha256(Path(initialization_path).read_bytes()).hexdigest() if initialization_path else None,
            "grayscale_conv_policy": "mean over RGB input-channel filters; existing policy",
            "full_model_initialization": args.pretrained_weights,
            "optimizer_resumed": resume_payload is not None,
            "transformer_pretrained": bool(P.tiny_vit_pretrained),
            "fresh_modules": ["local_projection", "transformer", "fusion"] if not P.tiny_vit_pretrained else ["fusion"],
        }
        config["dataset_split_seed"] = P.dataset_split_seed
        config["real_text_key"] = os.environ.get("REAL_TEXT_KEY", "text_original_path")
        config["normalization_mean"] = [0.449] if P.visual_input_channels == 1 else [0.485, 0.456, 0.406]
        config["normalization_std"] = [0.226] if P.visual_input_channels == 1 else [0.229, 0.224, 0.225]
        config.update(
            {
                "hf_home": os.environ.get("HF_HOME", ""),
                "original_cuda_visible_devices": RANK_DEVICE.original_visible_devices,
                "selected_cuda_device": RANK_DEVICE.selected_device,
                "ddp_static_graph": bool(static_graph.enabled),
                "ddp_static_graph_reason": static_graph.description,
                "ddp_find_unused_parameters": False,
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
                f"variant={P.architecture_variant} dim={P.vector_size} layers={P.vit_layers} "
                f"heads={P.vit_heads} mlp={P.vit_mlp_dim} deit_pretrained={P.tiny_vit_pretrained} "
                f"parameters={config['parameter_counts']}",
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
