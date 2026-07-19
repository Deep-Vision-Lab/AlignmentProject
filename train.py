#!/usr/bin/env python3
"""Generic full-quality trainer for synthetic or real manuscript alignment data.

This is the only Python training entry point. It supports:
- synthetic and real manifest datasets;
- optional real-data augmentation and per-epoch oversampling;
- single GPU and torchrun DistributedDataParallel;
- JAX Span-DTW with one isolated CUDA device per process;
- full image-text supervision on one or both paired lines;
- local hard negatives, compositional image-image matching, contextual order loss,
  and anti-collapse variance regularization;
- pretrained initialization and full checkpoint resume.
"""
from __future__ import annotations

import os


def _isolate_rank_cuda_device() -> tuple[int, int]:
    """Give every torchrun process one physical GPU before Torch/JAX imports."""
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
                    f"LOCAL_RANK={local_rank}, but CUDA_VISIBLE_DEVICES={visible!r}."
                )
            os.environ["CUDA_VISIBLE_DEVICES"] = devices[local_rank]
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    return local_rank, world_size


_ORIGINAL_LOCAL_RANK, _ENV_WORLD_SIZE = _isolate_rank_cuda_device()
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import math
import random
import sys
import time
from typing import Any, Dict, Iterable, Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Sampler

import Parameters as P
from alignment_visualization import save_d3tw_visualization
from arabic_span_text_encoder import ArabicSpanTextEncoder
from arabic_token_text_encoder import ArabicTokenTextEncoder
from embeddingModel import EmbeddingModel
from LossFunctionWithHelpers import ContrastiveSoftDTW
from span_alignment_loss import SpanContrastiveSoftDTW, hard_span_dtw_path
from textEmbedding import TextEmbedding

try:
    import psutil
except ImportError:
    psutil = None

try:
    import wandb
except ImportError:
    wandb = None


USE_AMP = torch.cuda.is_available() and os.environ.get("USE_AMP", "1") == "1"
AMP_DTYPE = torch.float16
_PROCESS = psutil.Process(os.getpid()) if psutil is not None else None
_BATCH_COUNTER = 0


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


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
        if self.enabled and not torch.cuda.is_available():
            raise RuntimeError("Multi-GPU DDP training requires CUDA GPUs.")
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        if self.enabled and not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        P.device = self.device

    def barrier(self) -> None:
        if self.enabled:
            dist.barrier()

    def close(self) -> None:
        if self.enabled and dist.is_initialized():
            dist.destroy_process_group()


CTX = DistributedContext()


class DistributedEvalSampler(Sampler[int]):
    """Shard validation/test samples without padding or duplication."""

    def __init__(self, dataset, rank: int, world_size: int):
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return 0 if remaining <= 0 else (remaining + self.world_size - 1) // self.world_size


def _strip_torchrun_rank_arguments() -> None:
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(rank_seed)
    try:
        import numpy as np

        np.random.seed(rank_seed)
    except ImportError:
        pass


def compute_stride(window_size, stride_ratio, window_overlap_mode):
    if window_overlap_mode == "no_overlap":
        return window_size
    if window_overlap_mode == "light_overlap":
        return max(1, window_size // 2)
    if window_overlap_mode == "dense_overlap":
        return max(1, window_size // 4)
    if window_overlap_mode == "custom":
        return max(1, int(window_size * stride_ratio))
    raise ValueError(f"Unknown window_overlap_mode: {window_overlap_mode!r}")


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def weights_dir(job_id):
    path = os.path.join(os.path.dirname(__file__), "Weights", job_id)
    os.makedirs(path, exist_ok=True)
    return path


def model_config(stride, args):
    return {
        "training_profile": "generic_full_quality",
        "dataset_type": args.dataset_type,
        "data_dir": args.data_dir,
        "real_augmentation": _env_flag("REAL_AUGMENT", False),
        "train_samples_per_epoch": _env_int("REAL_TRAIN_SAMPLES_PER_EPOCH", 0),
        "distributed": CTX.enabled,
        "distributed_backend": "nccl" if CTX.enabled else "none",
        "world_size": CTX.world_size,
        "per_gpu_batch_size": P.batch_size,
        "global_batch_size": P.batch_size * CTX.world_size,
        "window_size": P.window_size,
        "stride": stride,
        "stride_ratio": P.stride_ratio,
        "window_overlap_mode": P.window_overlap_mode,
        "vector_size": P.vector_size,
        "lang": P.lang,
        "negative_mode": P.negative_mode,
        "num_negatives": P.num_negatives,
        "active_negatives": _env_int(
            "SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE",
            getattr(P, "span_dtw_active_negatives_per_sample", 0),
        ),
        "use_bilstm": P.use_bilstm,
        "bilstm_layers": P.bilstm_layers,
        "bilstm_hidden_dim": P.bilstm_hidden_dim,
        "contrastive_soft_dtw_gamma": P.contrastive_soft_dtw_gamma,
        "contrastive_margin": P.contrastive_margin,
        "contrastive_temperature": P.contrastive_temperature,
        "text_encoder_type": P.text_encoder_type,
        "arabic_text_model_name": P.arabic_text_model_name,
        "max_text_token_chars": P.max_text_token_chars,
        "max_text_span_chars": P.max_text_span_chars,
        "max_windows_per_span": P.max_windows_per_span,
        "strip_span_text_edges": P.strip_span_text_edges,
        "span_feature_cache_size": P.span_feature_cache_size,
        "span_feature_cache_dtype": P.span_feature_cache_dtype,
        "clear_span_cache_each_epoch": P.clear_span_cache_each_epoch,
        "span_negative_grad_mode": P.span_negative_grad_mode,
        "span_dtw_backend": P.span_dtw_backend,
        "span_dtw_bucket_text_lengths": P.span_dtw_bucket_text_lengths,
        "span_dtw_text_bucket_size": P.span_dtw_text_bucket_size,
        "span_dtw_max_text_bucket": P.span_dtw_max_text_bucket,
        "image_text_loss_on_both_lines": getattr(P, "image_text_loss_on_both_lines", True),
        "use_local_hard_negatives": P.use_local_hard_negatives,
        "local_hard_negative_weight": P.local_hard_negative_weight,
        "local_hard_negative_margin": P.local_hard_negative_margin,
        "local_hard_negative_top_k": P.local_hard_negative_top_k,
        "local_hard_negative_exclude_radius": P.local_hard_negative_exclude_radius,
        "local_hard_negative_min_ink": P.local_hard_negative_min_ink,
        "local_hard_negative_every_n_batches": getattr(P, "local_hard_negative_every_n_batches", 1),
        "local_hard_negative_max_samples_per_batch": getattr(P, "local_hard_negative_max_samples_per_batch", 0),
        "use_image_pair_contrastive": P.use_image_pair_contrastive,
        "image_pair_loss_weight": P.image_pair_loss_weight,
        "image_pair_margin": P.image_pair_margin,
        "image_pair_top_k": P.image_pair_top_k,
        "image_pair_every_n_batches": getattr(P, "image_pair_every_n_batches", 1),
        "image_pair_max_samples_per_batch": getattr(P, "image_pair_max_samples_per_batch", 0),
        "pair_composition_max_regions": _env_int("PAIR_COMPOSITION_MAX_REGIONS", 2),
        "pair_composition_max_chars": _env_int("PAIR_COMPOSITION_MAX_CHARS", 3),
        "sequence_consistency_loss_weight": P.sequence_consistency_loss_weight,
        "order_temperature": _env_float("ORDER_TEMPERATURE", 0.07),
        "order_monotonic_margin": _env_float("ORDER_MONOTONIC_MARGIN", 0.02),
        "order_position_component_weight": _env_float("ORDER_POSITION_COMPONENT_WEIGHT", 1.0),
        "order_monotonic_component_weight": _env_float("ORDER_MONOTONIC_COMPONENT_WEIGHT", 1.0),
        "image_variance_loss_weight": P.image_variance_loss_weight,
        "image_variance_target_std": P.image_variance_target_std,
        "ink_contrast_threshold": _env_float("INK_CONTRAST_THRESHOLD", 0.15),
        "valid_every_n_epochs": P.valid_every_n_epochs,
        "valid_max_batches": P.valid_max_batches,
        "log_memory_every_n_batches": P.log_memory_every_n_batches,
        "jax_gpu_isolation": CTX.enabled,
        "text_gradient_sync": "manual_allreduce" if CTX.enabled else "none",
    }


def save_model_weights(model, text_encoder, job_id, config):
    raw_model = _unwrap_model(model)
    path = os.path.join(weights_dir(job_id), "model_latest.pth")
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


def save_checkpoint(model, text_encoder, optimizer, scheduler, scaler, epoch, job_id, config):
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
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "model_config": config,
        },
        os.path.join(weights_dir(job_id), "checkpoint_latest.pth"),
    )


def extract_model_state(loaded):
    if isinstance(loaded, dict) and "image_model_state_dict" in loaded:
        return loaded["image_model_state_dict"]
    if isinstance(loaded, dict) and "model_state_dict" in loaded:
        return loaded["model_state_dict"]
    return loaded


def build_text_encoder():
    if P.text_encoder_type == "arabic_span":
        text_encoder = ArabicSpanTextEncoder(
            model_name=P.arabic_text_model_name,
            output_dim=P.vector_size,
            max_span_chars=P.max_text_span_chars,
            freeze_backbone=True,
            device=P.device,
            strip_text_edges=P.strip_span_text_edges,
            cache_size=P.span_feature_cache_size,
            cache_dtype=P.span_feature_cache_dtype,
        )
    elif P.text_encoder_type == "arabic_token":
        text_encoder = ArabicTokenTextEncoder(
            model_name=P.arabic_text_model_name,
            output_dim=P.vector_size,
            max_token_chars=P.max_text_token_chars,
            freeze_backbone=True,
            device=P.device,
        )
    elif P.text_encoder_type == "char":
        text_encoder = TextEmbedding(embedding_dim=P.vector_size)
        for parameter in text_encoder.parameters():
            parameter.requires_grad_(False)
    else:
        raise ValueError(f"Unknown text_encoder_type: {P.text_encoder_type}")
    return text_encoder.to(P.device)


def build_image_embedding(stride):
    return EmbeddingModel(
        window_size=P.window_size,
        stride=stride,
        vector_size=P.vector_size,
        device=P.device,
        use_flip=(P.lang.lower() == "arabic"),
        use_bilstm=P.use_bilstm,
        bilstm_layers=P.bilstm_layers,
        bilstm_hidden_dim=P.bilstm_hidden_dim,
    )


def has_trainable_parameters(module):
    return any(parameter.requires_grad for parameter in module.parameters())


def embed_single_text(text_encoder, text):
    if has_trainable_parameters(text_encoder):
        embedding = text_encoder(text)
    else:
        with torch.no_grad():
            embedding = text_encoder(text)
    return F.normalize(embedding.float(), p=2, dim=-1)


def compute_similarity_lists(text_encoder, norm_img, pos_texts, neg_texts):
    sim_pos_list = []
    sim_neg_list = []
    for sample_idx, pos_text in enumerate(pos_texts):
        norm_pos_text = embed_single_text(text_encoder, pos_text)
        sim_pos_list.append(torch.einsum("sv,tv->st", norm_pos_text, norm_img[sample_idx]))
        sample_neg_sims = []
        for neg_text in neg_texts[sample_idx]:
            norm_neg_text = embed_single_text(text_encoder, neg_text)
            sample_neg_sims.append(
                torch.einsum("tv,sv->ts", norm_neg_text, norm_img[sample_idx])
            )
        sim_neg_list.append(sample_neg_sims)
    return sim_pos_list, sim_neg_list


def image_embedding_variance_loss(img_emb, target_std=0.05):
    if target_std <= 0:
        return img_emb.new_tensor(0.0)
    z = img_emb.reshape(-1, img_emb.shape[-1]).float()
    if z.shape[0] <= 1:
        return z.new_tensor(0.0)
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-6)
    return torch.relu(float(target_std) - std).mean()


def ink_weighted_mean(local_windows, ink=None, eps=1e-6):
    if ink is None:
        return local_windows.mean(dim=0)
    weights = ink.to(local_windows.device).float().clamp_min(0.0)
    if weights.numel() != local_windows.shape[0] or weights.sum().item() <= eps:
        return local_windows.mean(dim=0)
    weights = weights / weights.sum().clamp_min(eps)
    return (local_windows * weights.unsqueeze(-1)).sum(dim=0)


def _zero_local_stats(tensor):
    return tensor.new_tensor(0.0), {
        "local_hard_neg": 0.0,
        "local_pos_sim": 0.0,
        "local_neg_sim": 0.0,
        "local_terms": 0.0,
    }


def _local_hard_negative_loss_for_one_sample(
    span_encoding,
    local_image_embeddings,
    path,
    ink_ratio=None,
    margin=0.25,
    top_k=8,
    exclude_radius=2,
    min_ink=0.02,
):
    losses = []
    pos_values = []
    neg_values = []
    span_embeddings = F.normalize(span_encoding.embeddings.float(), p=2, dim=-1)
    local_image_embeddings = F.normalize(local_image_embeddings.float(), p=2, dim=-1)
    num_windows = int(local_image_embeddings.shape[0])
    ink_ratio = (
        ink_ratio.to(local_image_embeddings.device).float()
        if ink_ratio is not None
        else None
    )
    valid_ink = ink_ratio >= float(min_ink) if ink_ratio is not None else None

    for step in path:
        span_idx = int(step["span_idx"])
        w0 = int(step["window_start"])
        w1 = int(step["window_end"])
        if w1 <= w0 or w0 < 0 or w1 > num_windows:
            continue
        if valid_ink is not None and not valid_ink[w0:w1].any().item():
            continue
        span_vec = span_embeddings[span_idx]
        region_ink = ink_ratio[w0:w1] if ink_ratio is not None else None
        pos_img = F.normalize(
            ink_weighted_mean(local_image_embeddings[w0:w1], region_ink),
            p=2,
            dim=-1,
        )
        sim_pos = torch.matmul(pos_img, span_vec)

        neg_mask = torch.ones(
            num_windows, dtype=torch.bool, device=local_image_embeddings.device
        )
        left = max(0, w0 - int(exclude_radius))
        right = min(num_windows, w1 + int(exclude_radius))
        neg_mask[left:right] = False
        if valid_ink is not None:
            neg_mask &= valid_ink
        if neg_mask.sum().item() == 0:
            continue

        sim_negs = torch.matmul(local_image_embeddings[neg_mask], span_vec)
        k = min(max(1, int(top_k)), int(sim_negs.numel()))
        hard_negs = torch.topk(sim_negs, k=k).values
        losses.append(torch.relu(float(margin) - sim_pos + hard_negs).mean())
        pos_values.append(sim_pos.detach())
        neg_values.append(hard_negs.mean().detach())

    if not losses:
        return _zero_local_stats(local_image_embeddings)
    loss = torch.stack(losses).mean()
    return loss, {
        "local_hard_neg": float(loss.detach().item()),
        "local_pos_sim": float(torch.stack(pos_values).mean().item()),
        "local_neg_sim": float(torch.stack(neg_values).mean().item()),
        "local_terms": float(len(losses)),
    }


def _select_cyclic_indices(batch_size: int, max_samples: int, device):
    if max_samples <= 0 or max_samples >= batch_size:
        return None
    start = ((_BATCH_COUNTER - 1) * max_samples) % batch_size
    indices = [(start + offset) % batch_size for offset in range(max_samples)]
    return torch.as_tensor(indices, dtype=torch.long, device=device)


def local_hard_negative_loss_for_batch(
    text_encoder,
    criterion,
    norm_context_img,
    norm_local_img,
    pos_texts,
    ink_ratios=None,
):
    if (
        not P.use_local_hard_negatives
        or P.local_hard_negative_weight <= 0
        or not torch.is_grad_enabled()
    ):
        return _zero_local_stats(norm_local_img)

    max_samples = _env_int(
        "LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH",
        getattr(P, "local_hard_negative_max_samples_per_batch", 0),
    )
    indices = _select_cyclic_indices(
        int(norm_context_img.shape[0]), max_samples, norm_context_img.device
    )
    if indices is not None:
        cpu_indices = [int(index) for index in indices.detach().cpu().tolist()]
        pos_texts = [pos_texts[index] for index in cpu_indices]
        norm_context_img = norm_context_img.index_select(0, indices)
        norm_local_img = norm_local_img.index_select(0, indices)
        if torch.is_tensor(ink_ratios):
            ink_ratios = ink_ratios.index_select(0, indices)

    losses = []
    stats_list = []
    for sample_idx, pos_text in enumerate(pos_texts):
        pos_encoding = text_encoder(
            pos_text, use_cache=False if text_encoder.training else None
        )
        try:
            with torch.no_grad():
                path = hard_span_dtw_path(
                    pos_encoding,
                    norm_context_img[sample_idx],
                    temperature=criterion.temperature,
                    max_windows=criterion.max_windows_per_span,
                    window_count_penalty=criterion.window_count_penalty,
                )
        except ValueError:
            continue
        ink_ratio = ink_ratios[sample_idx] if ink_ratios is not None else None
        sample_loss, sample_stats = _local_hard_negative_loss_for_one_sample(
            pos_encoding,
            norm_local_img[sample_idx],
            path,
            ink_ratio=ink_ratio,
            margin=P.local_hard_negative_margin,
            top_k=P.local_hard_negative_top_k,
            exclude_radius=P.local_hard_negative_exclude_radius,
            min_ink=P.local_hard_negative_min_ink,
        )
        losses.append(sample_loss)
        stats_list.append(sample_stats)

    if not losses:
        return _zero_local_stats(norm_local_img)
    loss = torch.stack(losses).mean()
    return loss, {
        "local_hard_neg": float(loss.detach().item()),
        "local_pos_sim": sum(item["local_pos_sim"] for item in stats_list)
        / len(stats_list),
        "local_neg_sim": sum(item["local_neg_sim"] for item in stats_list)
        / len(stats_list),
        "local_terms": sum(item["local_terms"] for item in stats_list)
        / len(stats_list),
    }


def extract_aligned_span_regions(
    text_encoder,
    criterion,
    text,
    norm_context_img,
    norm_local_img,
    ink_ratio=None,
):
    encoding = text_encoder(text, use_cache=False if text_encoder.training else None)
    with torch.no_grad():
        path = hard_span_dtw_path(
            encoding,
            norm_context_img,
            temperature=criterion.temperature,
            max_windows=criterion.max_windows_per_span,
            window_count_penalty=criterion.window_count_penalty,
        )

    regions = []
    ink_ratio = (
        ink_ratio.to(norm_local_img.device).float()
        if ink_ratio is not None
        else None
    )
    for step in path:
        span_idx = int(step["span_idx"])
        w0 = int(step["window_start"])
        w1 = int(step["window_end"])
        if w1 <= w0 or w0 < 0 or w1 > norm_local_img.shape[0]:
            continue
        span_text = str(encoding.texts[span_idx])
        if not span_text.strip():
            continue
        region_ink = ink_ratio[w0:w1] if ink_ratio is not None else None
        if (
            region_ink is not None
            and region_ink.max().item() < P.local_hard_negative_min_ink
        ):
            continue
        regions.append(
            {
                "span_text": span_text,
                "span_idx": span_idx,
                "window_start": w0,
                "window_end": w1,
                "center": norm_local_img.new_tensor(0.5 * (w0 + w1 - 1)),
                "vec": F.normalize(
                    ink_weighted_mean(norm_local_img[w0:w1], region_ink),
                    p=2,
                    dim=-1,
                ),
                "context_vec": F.normalize(
                    ink_weighted_mean(norm_context_img[w0:w1], region_ink),
                    p=2,
                    dim=-1,
                ),
            }
        )
    return regions


def _normalised_mean(vectors: Iterable[torch.Tensor]) -> torch.Tensor:
    return F.normalize(torch.stack(list(vectors), dim=0).mean(dim=0), p=2, dim=-1)


def _build_composition_groups(regions):
    max_regions = max(1, _env_int("PAIR_COMPOSITION_MAX_REGIONS", 2))
    max_chars = max(1, _env_int("PAIR_COMPOSITION_MAX_CHARS", 3))
    groups = []
    for start in range(len(regions)):
        for count in range(1, max_regions + 1):
            end = start + count
            if end > len(regions):
                break
            members = regions[start:end]
            texts = [str(region["span_text"]) for region in members]
            if any(
                (not text.strip()) or any(character.isspace() for character in text)
                for text in texts
            ):
                break
            text = "".join(texts)
            if len(text) > max_chars:
                break
            groups.append(
                {
                    "text": text,
                    "center": torch.stack(
                        [member["center"] for member in members]
                    ).mean(),
                    "vec": _normalised_mean(member["vec"] for member in members),
                    "context_vec": _normalised_mean(
                        member.get("context_vec", member["vec"])
                        for member in members
                    ),
                }
            )
    return groups


def _directional_group_contrastive(groups_a, groups_b, margin, top_k):
    if not groups_a or not groups_b:
        return [], []
    vectors_a = torch.stack([group["vec"] for group in groups_a], dim=0)
    vectors_b = torch.stack([group["vec"] for group in groups_b], dim=0)
    similarities = torch.matmul(vectors_a, vectors_b.T)
    losses = []
    matched = []
    for index_a, group_a in enumerate(groups_a):
        positive_indices = [
            index_b
            for index_b, group_b in enumerate(groups_b)
            if group_b["text"] == group_a["text"]
        ]
        if not positive_indices:
            continue
        positive_index = max(
            positive_indices,
            key=lambda index_b: float(similarities[index_a, index_b].detach()),
        )
        negative_indices = [
            index_b
            for index_b, group_b in enumerate(groups_b)
            if group_b["text"] != group_a["text"]
        ]
        if not negative_indices:
            continue
        negative_values = similarities[index_a, negative_indices]
        k = min(max(1, int(top_k)), int(negative_values.numel()))
        hard_negatives = torch.topk(negative_values, k=k).values
        losses.append(
            torch.relu(
                float(margin)
                - similarities[index_a, positive_index]
                + hard_negatives
            ).mean()
        )
        matched.append((group_a, groups_b[positive_index]))
    return losses, matched


def image_image_span_contrastive_loss(regions1, regions2, margin=0.35, top_k=8):
    groups1 = _build_composition_groups(regions1)
    groups2 = _build_composition_groups(regions2)
    if not groups1 or not groups2:
        return None, []
    losses12, matched12 = _directional_group_contrastive(
        groups1, groups2, margin, top_k
    )
    losses21, matched21_reverse = _directional_group_contrastive(
        groups2, groups1, margin, top_k
    )
    matched21 = [(group1, group2) for group2, group1 in matched21_reverse]
    losses = losses12 + losses21
    matched = matched12 + matched21
    if not losses:
        return groups1[0]["vec"].new_tensor(0.0), matched
    return torch.stack(losses).mean(), matched


def _normalise_centers(centers: torch.Tensor) -> torch.Tensor:
    minimum = centers.min()
    maximum = centers.max()
    return (centers - minimum) / (maximum - minimum).clamp_min(1.0)


def _soft_contextual_position_loss(anchor_regions, candidate_regions):
    if len(anchor_regions) < 2 or len(candidate_regions) < 2:
        if anchor_regions:
            device = anchor_regions[0]["vec"].device
        elif candidate_regions:
            device = candidate_regions[0]["vec"].device
        else:
            device = P.device
        return torch.tensor(0.0, device=device)

    anchor_vectors = torch.stack(
        [region.get("context_vec", region["vec"]) for region in anchor_regions],
        dim=0,
    )
    candidate_vectors = torch.stack(
        [region.get("context_vec", region["vec"]) for region in candidate_regions],
        dim=0,
    )
    anchor_centers = _normalise_centers(
        torch.stack([region["center"] for region in anchor_regions]).to(
            anchor_vectors.device
        )
    )
    candidate_centers = _normalise_centers(
        torch.stack([region["center"] for region in candidate_regions]).to(
            anchor_vectors.device
        )
    )
    temperature = max(1e-4, _env_float("ORDER_TEMPERATURE", 0.07))
    probabilities = torch.softmax(
        torch.matmul(anchor_vectors, candidate_vectors.T) / temperature, dim=1
    )
    expected_positions = torch.matmul(probabilities, candidate_centers)
    position_loss = F.smooth_l1_loss(expected_positions, anchor_centers)
    order = torch.argsort(anchor_centers)
    ordered_expected = expected_positions.index_select(0, order)
    differences = ordered_expected[1:] - ordered_expected[:-1]
    monotonic_loss = torch.relu(
        _env_float("ORDER_MONOTONIC_MARGIN", 0.02) - differences
    ).mean()
    return (
        _env_float("ORDER_POSITION_COMPONENT_WEIGHT", 1.0) * position_loss
        + _env_float("ORDER_MONOTONIC_COMPONENT_WEIGHT", 1.0)
        * monotonic_loss
    )


def image_image_order_consistency_loss(regions1, regions2, matched_pairs):
    del matched_pairs
    if not regions1 or not regions2:
        if regions1:
            device = regions1[0]["vec"].device
        elif regions2:
            device = regions2[0]["vec"].device
        else:
            device = P.device
        return torch.tensor(0.0, device=device)
    return 0.5 * (
        _soft_contextual_position_loss(regions1, regions2)
        + _soft_contextual_position_loss(regions2, regions1)
    )


def compute_embeddings(image_embedder, images):
    with autocast(dtype=AMP_DTYPE, enabled=USE_AMP):
        img_emb, local_img_emb, ink_ratio = image_embedder(
            images, return_local=True, return_ink=True
        )
    return (
        F.normalize(img_emb.float(), p=2, dim=-1),
        F.normalize(local_img_emb.float(), p=2, dim=-1),
        ink_ratio,
        local_img_emb,
    )


def _select_active_negatives(neg_texts, active_per_sample: int):
    if active_per_sample <= 0:
        return neg_texts
    selected = []
    for sample_idx, sample_negs in enumerate(neg_texts):
        sample_negs = list(sample_negs)
        count = len(sample_negs)
        if count == 0 or active_per_sample >= count:
            selected.append(sample_negs)
            continue
        start = (_BATCH_COUNTER + sample_idx) % count
        selected.append(
            [
                sample_negs[(start + offset) % count]
                for offset in range(active_per_sample)
            ]
        )
    return selected


def compute_single_image_text_loss(
    image_embedder,
    text_encoder,
    criterion,
    images,
    pos_texts,
    neg_texts,
    embeddings=None,
    local_enabled=True,
):
    if P.text_encoder_type == "arabic_span":
        if embeddings is None:
            norm_img, norm_local_img, ink_ratio, local_img_emb = compute_embeddings(
                image_embedder, images
            )
        else:
            norm_img, norm_local_img, ink_ratio, local_img_emb = embeddings

        active_negatives = (
            _env_int(
                "SPAN_DTW_ACTIVE_NEGATIVES_PER_SAMPLE",
                getattr(P, "span_dtw_active_negatives_per_sample", 0),
            )
            if torch.is_grad_enabled()
            else 0
        )
        neg_texts_for_dtw = _select_active_negatives(neg_texts, active_negatives)
        loss, stats = criterion.forward_varlen(
            text_encoder, norm_img, pos_texts, neg_texts_for_dtw
        )
        stats["active_negatives"] = float(
            len(neg_texts_for_dtw[0]) if neg_texts_for_dtw else 0
        )

        if local_enabled:
            local_loss, local_stats = local_hard_negative_loss_for_batch(
                text_encoder,
                criterion,
                norm_img,
                norm_local_img,
                pos_texts,
                ink_ratios=ink_ratio,
            )
        else:
            local_loss, local_stats = _zero_local_stats(norm_local_img)
        if (
            local_enabled
            and P.use_local_hard_negatives
            and P.local_hard_negative_weight > 0
            and torch.is_grad_enabled()
        ):
            loss = loss + P.local_hard_negative_weight * local_loss
        stats.update(local_stats)

        variance_loss = image_embedding_variance_loss(
            local_img_emb, P.image_variance_target_std
        )
        if P.image_variance_loss_weight > 0 and torch.is_grad_enabled():
            loss = loss + P.image_variance_loss_weight * variance_loss
        stats["img_var_loss"] = float(variance_loss.detach().item())
        stats["total"] = float(loss.detach().item())
        return loss, stats, (norm_img, norm_local_img, ink_ratio, local_img_emb)

    with autocast(dtype=AMP_DTYPE, enabled=USE_AMP):
        img_emb = image_embedder(images)
    norm_img = F.normalize(img_emb.float(), p=2, dim=-1)
    sim_pos_list, sim_neg_list = compute_similarity_lists(
        text_encoder, norm_img, pos_texts, neg_texts
    )
    loss, stats = criterion.forward_varlen(sim_pos_list, sim_neg_list)
    variance_loss = image_embedding_variance_loss(
        img_emb, P.image_variance_target_std
    )
    if P.image_variance_loss_weight > 0 and torch.is_grad_enabled():
        loss = loss + P.image_variance_loss_weight * variance_loss
    stats["img_var_loss"] = float(variance_loss.detach().item())
    stats["total"] = float(loss.detach().item())
    return loss, stats, None


def _slice_embeddings(embeddings, max_samples: int):
    if embeddings is None or max_samples <= 0:
        return embeddings
    return tuple(
        value[:max_samples]
        if torch.is_tensor(value) and value.shape[0] >= max_samples
        else value
        for value in embeddings
    )


def compute_image_pair_loss(text_encoder, criterion, texts1, texts2, emb1, emb2):
    if (
        not P.use_image_pair_contrastive
        or P.image_pair_loss_weight <= 0
        or P.text_encoder_type != "arabic_span"
        or not torch.is_grad_enabled()
    ):
        zero = emb1[0].new_tensor(0.0)
        return zero, zero, {
            "image_pair_loss": 0.0,
            "order_loss": 0.0,
            "pair_terms": 0.0,
        }

    max_samples = _env_int(
        "IMAGE_PAIR_MAX_SAMPLES_PER_BATCH",
        getattr(P, "image_pair_max_samples_per_batch", 0),
    )
    if max_samples > 0:
        max_samples = min(
            max_samples,
            len(texts1),
            int(emb1[0].shape[0]),
            int(emb2[0].shape[0]),
        )
        texts1 = list(texts1[:max_samples])
        texts2 = list(texts2[:max_samples])
        emb1 = _slice_embeddings(emb1, max_samples)
        emb2 = _slice_embeddings(emb2, max_samples)

    norm_ctx1, norm_loc1, ink1, _raw1 = emb1
    norm_ctx2, norm_loc2, ink2, _raw2 = emb2
    pair_losses = []
    order_losses = []
    terms = 0
    for batch_index in range(norm_ctx1.shape[0]):
        try:
            regions1 = extract_aligned_span_regions(
                text_encoder,
                criterion,
                texts1[batch_index],
                norm_ctx1[batch_index],
                norm_loc1[batch_index],
                ink1[batch_index],
            )
            regions2 = extract_aligned_span_regions(
                text_encoder,
                criterion,
                texts2[batch_index],
                norm_ctx2[batch_index],
                norm_loc2[batch_index],
                ink2[batch_index],
            )
        except ValueError:
            continue
        pair_loss, matched_pairs = image_image_span_contrastive_loss(
            regions1,
            regions2,
            margin=P.image_pair_margin,
            top_k=P.image_pair_top_k,
        )
        if pair_loss is None:
            continue
        pair_losses.append(pair_loss)
        terms += len(matched_pairs)
        if P.sequence_consistency_loss_weight > 0:
            order_losses.append(
                image_image_order_consistency_loss(
                    regions1, regions2, matched_pairs
                )
            )

    if not pair_losses:
        zero = norm_ctx1.new_tensor(0.0)
        return zero, zero, {
            "image_pair_loss": 0.0,
            "order_loss": 0.0,
            "pair_terms": 0.0,
        }
    pair_loss = torch.stack(pair_losses).mean()
    order_loss = (
        torch.stack(order_losses).mean()
        if order_losses
        else pair_loss.new_tensor(0.0)
    )
    return pair_loss, order_loss, {
        "image_pair_loss": float(pair_loss.detach().item()),
        "order_loss": float(order_loss.detach().item()),
        "pair_terms": float(terms) / max(1, len(pair_losses)),
    }


def average_stats(stats_list):
    if not stats_list:
        return {}
    keys = set()
    for stats in stats_list:
        keys.update(stats.keys())
    return {
        key: sum(stats.get(key, 0.0) for stats in stats_list) / len(stats_list)
        for key in keys
    }


def compute_batch_loss(image_embedder, text_encoder, criterion, batch):
    global _BATCH_COUNTER
    if torch.is_grad_enabled():
        _BATCH_COUNTER += 1

    local_every = max(
        1,
        _env_int(
            "LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES",
            getattr(P, "local_hard_negative_every_n_batches", 1),
        ),
    )
    local_enabled = torch.is_grad_enabled() and (_BATCH_COUNTER % local_every == 0)

    if not isinstance(batch, dict):
        images, pos_texts, neg_texts = batch
        images = images.to(P.device, non_blocking=True)
        loss, stats, _embeddings = compute_single_image_text_loss(
            image_embedder,
            text_encoder,
            criterion,
            images,
            pos_texts,
            neg_texts,
            local_enabled=local_enabled,
        )
        return loss, stats

    images1 = batch["images1"].to(P.device, non_blocking=True)
    images2 = batch["images2"].to(P.device, non_blocking=True)
    texts1 = batch["texts1"]
    texts2 = batch["texts2"]
    neg_texts1 = batch["neg_texts1"]
    neg_texts2 = batch["neg_texts2"]

    if P.text_encoder_type == "arabic_span":
        emb1 = compute_embeddings(image_embedder, images1)
        emb2 = compute_embeddings(image_embedder, images2)
    else:
        emb1 = None
        emb2 = None

    loss1, stats1, emb1 = compute_single_image_text_loss(
        image_embedder,
        text_encoder,
        criterion,
        images1,
        texts1,
        neg_texts1,
        emb1,
        local_enabled=local_enabled,
    )

    if bool(getattr(P, "image_text_loss_on_both_lines", True)):
        loss2, stats2, emb2 = compute_single_image_text_loss(
            image_embedder,
            text_encoder,
            criterion,
            images2,
            texts2,
            neg_texts2,
            emb2,
            local_enabled=local_enabled,
        )
        loss = 0.5 * (loss1 + loss2)
        stats = average_stats([stats1, stats2])
    else:
        loss = loss1
        stats = dict(stats1)

    pair_every = max(
        1,
        _env_int(
            "IMAGE_PAIR_EVERY_N_BATCHES",
            getattr(P, "image_pair_every_n_batches", 1),
        ),
    )
    pair_enabled = torch.is_grad_enabled() and (_BATCH_COUNTER % pair_every == 0)
    if pair_enabled and emb1 is not None and emb2 is not None:
        pair_loss, order_loss, pair_stats = compute_image_pair_loss(
            text_encoder, criterion, texts1, texts2, emb1, emb2
        )
        loss = loss + P.image_pair_loss_weight * pair_loss
        if P.sequence_consistency_loss_weight > 0:
            loss = loss + P.sequence_consistency_loss_weight * order_loss
        stats.update(pair_stats)
    else:
        stats.update(
            {"image_pair_loss": 0.0, "order_loss": 0.0, "pair_terms": 0.0}
        )
    stats["total"] = float(loss.detach().item())
    return loss, stats


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


def select_dataloaders(args):
    os.environ["DATASET_TYPE"] = args.dataset_type
    if args.dataset_type == "real":
        from AugmentedRealDataLoader import build_dataloaders
    else:
        from DataLoader import build_dataloaders

    train_loader, valid_loader, test_loader = build_dataloaders(args.data_dir)
    if not CTX.enabled:
        return train_loader, valid_loader, test_loader, None

    split_seed = _env_int("DATASET_SPLIT_SEED", 42)
    train_sampler = DistributedSampler(
        train_loader.dataset,
        num_replicas=CTX.world_size,
        rank=CTX.rank,
        shuffle=True,
        seed=split_seed,
        drop_last=False,
    )
    valid_sampler = DistributedEvalSampler(
        valid_loader.dataset, rank=CTX.rank, world_size=CTX.world_size
    )
    test_sampler = DistributedEvalSampler(
        test_loader.dataset, rank=CTX.rank, world_size=CTX.world_size
    )
    return (
        _rebuild_loader(train_loader, train_sampler),
        _rebuild_loader(valid_loader, valid_sampler),
        _rebuild_loader(test_loader, test_sampler),
        train_sampler,
    )


def _broadcast_trainable_text_parameters(text_encoder: nn.Module) -> None:
    if not CTX.enabled:
        return
    for parameter in text_encoder.parameters():
        if parameter.requires_grad:
            dist.broadcast(parameter.data, src=0)


def _allreduce_text_gradients(text_encoder: nn.Module) -> None:
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
    if isinstance(batch, (tuple, list)) and batch:
        first = batch[0]
        return int(first.shape[0]) if torch.is_tensor(first) else len(first)
    return 1


def _accumulate_stats(stats_sum: dict, stats: dict, weight: int) -> None:
    for key, value in stats.items():
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric):
            continue
        stats_sum[key] = stats_sum.get(key, 0.0) + numeric * weight


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
    merged = {
        key: sum(
            float((item.get("stats_sum") or {}).get(key, 0.0))
            for item in payloads
        )
        / total_weight
        for key in keys
    }
    return loss, merged


def _format_memory(text_encoder):
    parts = []
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        parts.append(f"gpu={allocated:.2f}/{reserved:.2f}GB")
    if _PROCESS is not None:
        parts.append(f"rss={_PROCESS.memory_info().rss / (1024**3):.2f}GB")
    if hasattr(text_encoder, "cache_size_current"):
        parts.append(f"span_cache={text_encoder.cache_size_current()}")
    return " ".join(parts)


def clear_text_encoder_cache(text_encoder):
    if hasattr(text_encoder, "clear_cache"):
        text_encoder.clear_cache()


def train_one_epoch(model, text_encoder, criterion, optimizer, scaler, loader):
    model.train()
    text_encoder.train() if has_trainable_parameters(text_encoder) else text_encoder.eval()
    loss_sum = 0.0
    total_weight = 0
    stats_sum: Dict[str, float] = {}

    for batch_idx, batch in enumerate(loader):
        batch_started = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss, stats = compute_batch_loss(model, text_encoder, criterion, batch)
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
                memory = " " + _format_memory(text_encoder)
            print(
                f"rank=0 batch={batch_idx + 1}/{len(loader)} "
                f"local_batch={weight} global_batch={weight * CTX.world_size} "
                f"loss={loss.item():.4f} "
                f"norm_pos={stats.get('norm_pos', float('nan')):.4f} "
                f"norm_neg={stats.get('norm_neg', float('nan')):.4f} "
                f"gap={stats.get('gap', float('nan')):.4f} "
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
def validate(model, text_encoder, criterion, loader, max_batches=0):
    model.eval()
    text_encoder.eval()
    loss_sum = 0.0
    total_weight = 0
    stats_sum: Dict[str, float] = {}
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        loss, stats = compute_batch_loss(model, text_encoder, criterion, batch)
        weight = max(1, _batch_size(batch))
        loss_sum += float(loss.detach().item()) * weight
        total_weight += weight
        _accumulate_stats(stats_sum, stats, weight)
    return _merge_epoch_payload(
        {"loss_sum": loss_sum, "stats_sum": stats_sum, "weight": total_weight}
    )


def init_wandb(args, config):
    if not CTX.is_main or os.environ.get("USE_WANDB", "1") == "0":
        return None
    if wandb is None:
        print("wandb is not installed; continuing without W&B.", flush=True)
        return None
    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "alignment-project"),
        name=args.job_id,
        config=config,
    )


def wandb_log_epoch_metrics(run, epoch, train_loss, val_loss, train_stats):
    if run is None:
        return
    wandb.log(
        {
            "loss": float(train_loss),
            "validation_loss": float(val_loss),
            "pos": float(train_stats.get("norm_pos", float("nan"))),
            "negative": float(train_stats.get("norm_neg", float("nan"))),
            "raw_pos_cost": float(train_stats.get("cost_pos", float("nan"))),
            "raw_negative_cost": float(train_stats.get("cost_neg", float("nan"))),
            "gap": float(train_stats.get("gap", float("nan"))),
            "pos_prob": float(train_stats.get("pos_prob", float("nan"))),
            "local_hard_neg": float(train_stats.get("local_hard_neg", 0.0)),
            "local_pos_sim": float(train_stats.get("local_pos_sim", 0.0)),
            "local_neg_sim": float(train_stats.get("local_neg_sim", 0.0)),
            "image_pair_loss": float(train_stats.get("image_pair_loss", 0.0)),
            "order_loss": float(train_stats.get("order_loss", 0.0)),
            "pair_terms": float(train_stats.get("pair_terms", 0.0)),
            "img_var_loss": float(train_stats.get("img_var_loss", 0.0)),
        },
        step=int(epoch),
        commit=True,
    )


def _load_initial_states(args, model, text_encoder):
    resume_payload = None
    if args.resume:
        resume_payload = torch.load(args.resume, map_location=P.device)
        model.load_state_dict(extract_model_state(resume_payload))
        if "text_encoder_state_dict" in resume_payload:
            text_encoder.load_state_dict(
                resume_payload["text_encoder_state_dict"], strict=False
            )
        elif "text_embedder_state_dict" in resume_payload:
            text_encoder.load_state_dict(
                resume_payload["text_embedder_state_dict"], strict=False
            )
    elif args.pretrained_weights:
        loaded = torch.load(args.pretrained_weights, map_location=P.device)
        model.load_state_dict(extract_model_state(loaded))
        if isinstance(loaded, dict) and "text_encoder_state_dict" in loaded:
            text_encoder.load_state_dict(
                loaded["text_encoder_state_dict"], strict=False
            )
        elif isinstance(loaded, dict) and "text_embedder_state_dict" in loaded:
            text_encoder.load_state_dict(
                loaded["text_embedder_state_dict"], strict=False
            )
    return resume_payload


def build_criterion():
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
        use_cuda=torch.cuda.is_available(),
        margin=P.contrastive_margin,
        temperature=P.contrastive_temperature,
    )


def train(
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
    scaler = GradScaler(enabled=USE_AMP)
    start_epoch = 0
    if resume_payload is not None:
        optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
        scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
        if resume_payload.get("scaler_state_dict") is not None:
            scaler.load_state_dict(resume_payload["scaler_state_dict"])
        start_epoch = int(resume_payload["epoch"]) + 1

    run = init_wandb(args, config)
    history = []
    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        CTX.barrier()
        started = time.time()
        if CTX.is_main:
            print(
                f"epoch={epoch + 1}/{args.epochs} world_size={CTX.world_size}",
                flush=True,
            )

        train_loss, train_stats = train_one_epoch(
            model, text_encoder, criterion, optimizer, scaler, train_loader
        )
        should_validate = (
            ((epoch + 1) % P.valid_every_n_epochs == 0)
            or ((epoch + 1) == args.epochs)
        )
        if should_validate:
            clear_text_encoder_cache(text_encoder)
            val_loss, _val_stats = validate(
                model,
                text_encoder,
                criterion,
                valid_loader,
                max_batches=P.valid_max_batches,
            )
            clear_text_encoder_cache(text_encoder)
        else:
            val_loss = float("nan")

        scheduler.step()
        if CTX.is_main:
            wandb_log_epoch_metrics(
                run, epoch + 1, train_loss, val_loss, train_stats
            )
            save_checkpoint(
                model,
                text_encoder,
                optimizer,
                scheduler,
                scaler,
                epoch,
                args.job_id,
                config,
            )
            save_model_weights(model, text_encoder, args.job_id, config)

            should_visualize = (
                (((epoch + 1) % 10 == 0) or ((epoch + 1) == args.epochs))
                and not P.use_image_pair_contrastive
            )
            if should_visualize:
                save_d3tw_visualization(
                    _unwrap_model(model),
                    text_encoder,
                    valid_loader,
                    criterion,
                    epoch + 1,
                    args.job_id,
                    P.device,
                )
                clear_text_encoder_cache(text_encoder)

        if P.clear_span_cache_each_epoch:
            clear_text_encoder_cache(text_encoder)
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generic full-quality alignment trainer"
    )
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument(
        "--dataset_type", choices=["synthetic", "real"], required=True
    )
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--train_samples_per_epoch", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--pretrained_weights", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride_ratio", type=float, default=None)
    parser.add_argument(
        "--window_overlap_mode",
        choices=["no_overlap", "light_overlap", "dense_overlap", "custom"],
        default=None,
    )
    parser.add_argument("--negative_mode", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_negatives", type=int, default=None)
    parser.add_argument(
        "--use_bilstm", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--use_local_hard_negatives",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--local_hard_negative_weight", type=float, default=None)
    parser.add_argument("--image_variance_loss_weight", type=float, default=None)
    parser.add_argument(
        "--use_image_pair_contrastive",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--image_pair_loss_weight", type=float, default=None)
    return parser.parse_args()


def apply_overrides(args):
    os.environ["DATASET_TYPE"] = args.dataset_type
    if args.augment is not None:
        os.environ["REAL_AUGMENT"] = "1" if args.augment else "0"
    if args.train_samples_per_epoch is not None:
        os.environ["REAL_TRAIN_SAMPLES_PER_EPOCH"] = str(
            max(0, args.train_samples_per_epoch)
        )
    if args.num_samples is not None:
        P.num_samples = int(args.num_samples)
        os.environ["NUM_SAMPLES"] = str(args.num_samples)
    if args.window_size is not None:
        P.window_size = args.window_size
    if args.stride_ratio is not None:
        P.stride_ratio = args.stride_ratio
    if args.window_overlap_mode is not None:
        P.window_overlap_mode = args.window_overlap_mode
    if args.negative_mode is not None:
        P.negative_mode = args.negative_mode.lower()
    if args.num_negatives is not None:
        P.num_negatives = args.num_negatives
    if args.use_bilstm is not None:
        P.use_bilstm = args.use_bilstm
    if args.use_local_hard_negatives is not None:
        P.use_local_hard_negatives = args.use_local_hard_negatives
    if args.local_hard_negative_weight is not None:
        P.local_hard_negative_weight = args.local_hard_negative_weight
    if args.image_variance_loss_weight is not None:
        P.image_variance_loss_weight = args.image_variance_loss_weight
    if args.use_image_pair_contrastive is not None:
        P.use_image_pair_contrastive = args.use_image_pair_contrastive
    if args.image_pair_loss_weight is not None:
        P.image_pair_loss_weight = args.image_pair_loss_weight

    if args.finetune:
        args.learning_rate = (
            P.finetune_learning_rate
            if args.learning_rate is None
            else args.learning_rate
        )
        args.epochs = P.finetune_epochs if args.epochs is None else args.epochs
    else:
        args.learning_rate = (
            P.learning_rate if args.learning_rate is None else args.learning_rate
        )
        args.epochs = P.epochs if args.epochs is None else args.epochs


def main():
    CTX.initialize()
    try:
        _seed_everything(_env_int("TRAIN_SEED", 42), CTX.rank)
        _strip_torchrun_rank_arguments()
        args = parse_args()
        apply_overrides(args)
        if args.resume and args.pretrained_weights:
            raise SystemExit("Use either --resume or --pretrained_weights, not both.")

        stride = compute_stride(
            P.window_size, P.stride_ratio, P.window_overlap_mode
        )
        train_loader, valid_loader, _test_loader, train_sampler = select_dataloaders(
            args
        )
        text_encoder = build_text_encoder()
        raw_model = build_image_embedding(stride).to(P.device)
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

        criterion = build_criterion()
        config = model_config(stride, args)
        if CTX.is_main:
            print(
                f"job_id={args.job_id} dataset={args.dataset_type} "
                f"data_dir={args.data_dir} augment={_env_flag('REAL_AUGMENT', False)} "
                f"world_size={CTX.world_size} per_gpu_batch={P.batch_size} "
                f"global_batch={P.batch_size * CTX.world_size} device=cuda:0",
                flush=True,
            )
            print(
                f"train_batches_per_rank={len(train_loader)} "
                f"valid_batches_rank0={len(valid_loader)} "
                f"epochs={args.epochs} lr={args.learning_rate} "
                f"window_size={P.window_size} stride={stride} "
                f"span_backend={P.span_dtw_backend} negatives={P.num_negatives} "
                f"active_negatives={config['active_negatives']} "
                f"both_lines={config['image_text_loss_on_both_lines']} "
                f"local={P.use_local_hard_negatives} pair={P.use_image_pair_contrastive}",
                flush=True,
            )

        CTX.barrier()
        train(
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
