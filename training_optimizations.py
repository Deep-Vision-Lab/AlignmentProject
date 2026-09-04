from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import time
from types import MethodType
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


def _flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _integer(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _number(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class ComponentProfiler:
    def __init__(self):
        self.enabled = _flag("PROFILE_TRAINING", False)
        self.nvtx = _flag("ENABLE_NVTX", self.enabled)
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    @contextmanager
    def section(self, name: str):
        if self.nvtx and torch.cuda.is_available():
            torch.cuda.nvtx.range_push(name)
        if not self.enabled:
            try:
                yield
            finally:
                if self.nvtx and torch.cuda.is_available():
                    torch.cuda.nvtx.range_pop()
            return

        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                end.record()
                end.synchronize()
                elapsed = start.elapsed_time(end) / 1000.0
                self.totals[name] += elapsed
                self.counts[name] += 1
                if self.nvtx:
                    torch.cuda.nvtx.range_pop()
        else:
            started = time.perf_counter()
            try:
                yield
            finally:
                self.totals[name] += time.perf_counter() - started
                self.counts[name] += 1

    def add(self, name: str, seconds: float):
        if not self.enabled:
            return
        self.totals[name] += float(seconds)
        self.counts[name] += 1

    def summary(self):
        return {
            name: {
                "seconds": self.totals[name],
                "count": self.counts[name],
                "mean_seconds": self.totals[name] / max(1, self.counts[name]),
            }
            for name in sorted(self.totals)
        }

    def reset(self):
        self.totals.clear()
        self.counts.clear()


PROFILER = ComponentProfiler()


@dataclass
class AlignmentItem:
    encoding: Any
    path: list[dict]


def configure_runtime():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = _flag("CUDNN_BENCHMARK", True)
        torch.backends.cuda.matmul.allow_tf32 = _flag("ALLOW_TF32", True)
        torch.backends.cudnn.allow_tf32 = _flag("ALLOW_TF32", True)
        try:
            torch.set_float32_matmul_precision(
                os.environ.get("FLOAT32_MATMUL_PRECISION", "high")
            )
        except (AttributeError, ValueError):
            pass


def validate_resolved_configuration(P):
    errors = []
    if int(P.max_text_span_chars) > 2:
        errors.append("MAX_TEXT_SPAN_CHARS must be <=2 for truthful visible cores")
    if int(P.max_text_token_chars) > 2:
        errors.append("MAX_TEXT_TOKEN_CHARS must be <=2")
    if int(P.max_windows_per_span) > 3:
        errors.append("MAX_WINDOWS_PER_SPAN must be <=3")
    if bool(P.span_include_space_context):
        errors.append("SPAN_INCLUDE_SPACE_CONTEXT must be disabled")
    if bool(getattr(P, "span_allow_character_space_surfaces", False)):
        errors.append("SPAN_ALLOW_CHARACTER_SPACE_SURFACES must be disabled")
    if errors and not _flag("ALLOW_UNSAFE_SPAN_CONFIG", False):
        raise RuntimeError(
            "Unsafe/inconsistent optimized span configuration:\n- "
            + "\n- ".join(errors)
            + "\nSet ALLOW_UNSAFE_SPAN_CONFIG=1 only for an explicit ablation."
        )


def fast_window_ink_ratio_from_patches(patches, contrast_threshold=None):
    """Compute ink from normalized patches without materializing denormalized RGB."""
    if contrast_threshold is None:
        contrast_threshold = _number("INK_CONTRAST_THRESHOLD", 0.15)
    threshold = max(0.0, min(1.0, float(contrast_threshold)))
    with torch.no_grad():
        work = patches.detach().float()
        # gray = sum(channel_weight * (normalized * std + mean))
        coeff = work.new_tensor(
            [0.2989 * 0.229, 0.5870 * 0.224, 0.1140 * 0.225]
        ).view(1, 1, 3, 1, 1)
        offset = float(
            0.2989 * 0.485 + 0.5870 * 0.456 + 0.1140 * 0.406
        )
        gray = (work * coeff).sum(dim=2) + offset
        height, width = int(gray.shape[-2]), int(gray.shape[-1])
        border_h = max(1, int(round(height * 0.05)))
        border_w = max(1, int(round(width * 0.05)))
        border = torch.cat(
            [
                gray[..., :border_h, :].flatten(start_dim=2),
                gray[..., -border_h:, :].flatten(start_dim=2),
                gray[..., :, :border_w].flatten(start_dim=2),
                gray[..., :, -border_w:].flatten(start_dim=2),
            ],
            dim=-1,
        )
        background = border.median(dim=-1).values.unsqueeze(-1).unsqueeze(-1)
        return (gray - background).abs().ge(threshold).float().mean(dim=(2, 3))


def prepare_raw_model(raw_model):
    import embeddingModel as embedding_model_module

    embedding_model_module.window_ink_ratio_from_patches = (
        fast_window_ink_ratio_from_patches
    )
    chunk_size = max(1, _integer("CNN_CHUNK_SIZE", 1024))
    raw_model.CNN_CHUNK_SIZE = chunk_size

    original_process = raw_model._process_patches
    use_channels_last = _flag("USE_CHANNELS_LAST", True) and torch.cuda.is_available()
    if use_channels_last:
        raw_model.cnn_encoder.to(memory_format=torch.channels_last)

        def _process_patches_channels_last(self, patches):
            batch_size, windows_num, channels, height, width = patches.shape
            total = batch_size * windows_num
            flat = patches.reshape(total, channels, height, width).contiguous(
                memory_format=torch.channels_last
            )
            chunks = []
            for start in range(0, total, self.CNN_CHUNK_SIZE):
                end = min(start + self.CNN_CHUNK_SIZE, total)
                chunks.append(self.cnn_encoder(flat[start:end]))
            return torch.cat(chunks, dim=0).view(
                batch_size, windows_num, self.vector_size
            )

        raw_model._process_patches = MethodType(
            _process_patches_channels_last, raw_model
        )
    else:
        raw_model._process_patches = original_process

    if _flag("TORCH_COMPILE_VISUAL", False):
        if not hasattr(torch, "compile"):
            print("torch.compile unavailable; continuing without it", flush=True)
        else:
            try:
                raw_model.cnn_encoder = torch.compile(
                    raw_model.cnn_encoder,
                    mode=os.environ.get("TORCH_COMPILE_MODE", "reduce-overhead"),
                    dynamic=False,
                )
                print("compiled visual CNN with torch.compile", flush=True)
            except Exception as exc:
                print(f"torch.compile visual CNN failed: {exc}", flush=True)
    return raw_model


def rotating_indices(batch_size: int, maximum: int, counter: int, device):
    if maximum <= 0 or maximum >= batch_size:
        return list(range(batch_size))
    start = ((counter - 1) * maximum) % batch_size
    return [(start + offset) % batch_size for offset in range(maximum)]


def build_alignment_cache(
    train_module,
    text_encoder,
    criterion,
    texts,
    contextual,
    indices,
):
    if not indices:
        return {}
    selected_texts = [texts[index] for index in indices]
    if hasattr(text_encoder, "encode_many"):
        encodings = text_encoder.encode_many(selected_texts, use_cache=True)
    else:
        encodings = [text_encoder(text) for text in selected_texts]
    result = {}
    with torch.no_grad(), PROFILER.section("hard_path"):
        for local_index, batch_index in enumerate(indices):
            try:
                path = train_module.hard_span_dtw_path(
                    encodings[local_index],
                    contextual[batch_index],
                    temperature=criterion.temperature,
                    max_windows=criterion.max_windows_per_span,
                    window_count_penalty=criterion.window_count_penalty,
                )
            except ValueError:
                continue
            result[batch_index] = AlignmentItem(encodings[local_index], path)
    return result


def local_loss_from_cache(
    train_module,
    cache,
    local_embeddings,
    ink,
    indices,
):
    losses = []
    stats = []
    for index in indices:
        item = cache.get(index)
        if item is None:
            continue
        sample_loss, sample_stats = (
            train_module._local_hard_negative_loss_for_one_sample(
                item.encoding,
                local_embeddings[index],
                item.path,
                ink_ratio=ink[index] if ink is not None else None,
                margin=train_module.P.local_hard_negative_margin,
                top_k=train_module.P.local_hard_negative_top_k,
                exclude_radius=train_module.P.local_hard_negative_exclude_radius,
                min_ink=train_module.P.local_hard_negative_min_ink,
            )
        )
        losses.append(sample_loss)
        stats.append(sample_stats)
    if not losses:
        return train_module._zero_local_stats(local_embeddings)
    loss = torch.stack(losses).mean()
    return loss, {
        "local_hard_neg": float(loss.detach().item()),
        "local_pos_sim": sum(item["local_pos_sim"] for item in stats) / len(stats),
        "local_neg_sim": sum(item["local_neg_sim"] for item in stats) / len(stats),
        "local_terms": sum(item["local_terms"] for item in stats) / len(stats),
    }


def regions_from_alignment(
    train_module,
    item,
    contextual,
    local,
    ink,
):
    result = []
    span_encoding = item.encoding
    for step in item.path:
        if step.get("is_blank", False):
            continue
        span_index = int(step["span_idx"])
        w0, w1 = int(step["window_start"]), int(step["window_end"])
        if w1 <= w0 or w0 < 0 or w1 > int(local.shape[0]):
            continue
        text = str(span_encoding.texts[span_index])
        if not text.strip() or text == "<BLANK>":
            continue
        region_ink = ink[w0:w1] if ink is not None else None
        if (
            region_ink is not None
            and region_ink.max().item()
            < train_module.P.local_hard_negative_min_ink
        ):
            continue
        result.append(
            {
                "span_text": text,
                "span_idx": span_index,
                "window_start": w0,
                "window_end": w1,
                "center": local.new_tensor(0.5 * (w0 + w1 - 1)),
                "vec": F.normalize(
                    train_module.ink_weighted_mean(local[w0:w1], region_ink),
                    p=2,
                    dim=-1,
                ),
                "context_vec": F.normalize(
                    train_module.ink_weighted_mean(
                        contextual[w0:w1], region_ink
                    ),
                    p=2,
                    dim=-1,
                ),
            }
        )
    return result


def optimized_compute_batch_loss(train_module):
    original = train_module.compute_batch_loss

    def compute_batch_loss(image_embedder, text_encoder, criterion, batch):
        if not isinstance(batch, dict) or train_module.P.text_encoder_type != "arabic_span":
            return original(image_embedder, text_encoder, criterion, batch)

        train_module._BATCH_COUNTER += int(torch.is_grad_enabled())
        counter = max(1, train_module._BATCH_COUNTER)
        P = train_module.P
        images1 = batch["images1"].to(P.device, non_blocking=True)
        images2 = batch["images2"].to(P.device, non_blocking=True)
        if _flag("USE_CHANNELS_LAST", True) and torch.cuda.is_available():
            images1 = images1.contiguous(memory_format=torch.channels_last)
            images2 = images2.contiguous(memory_format=torch.channels_last)
        texts1, texts2 = batch["texts1"], batch["texts2"]
        neg1, neg2 = batch["neg_texts1"], batch["neg_texts2"]
        batch_size = int(images1.shape[0])

        # The two manuscript lines are always encoded independently. They share
        # the same ViT weights, but no image tensor is concatenated before the
        # visual encoder and no visual token from one line can affect the other.
        with PROFILER.section("visual_forward_line1"):
            emb1 = train_module.compute_embeddings(image_embedder, images1)
        with PROFILER.section("visual_forward_line2"):
            emb2 = train_module.compute_embeddings(image_embedder, images2)

        with PROFILER.section("image_text_line1"):
            loss1, stats1, emb1 = train_module.compute_single_image_text_loss(
                image_embedder,
                text_encoder,
                criterion,
                images1,
                texts1,
                neg1,
                emb1,
                local_enabled=False,
            )
        if bool(getattr(P, "image_text_loss_on_both_lines", True)):
            with PROFILER.section("image_text_line2"):
                loss2, stats2, emb2 = train_module.compute_single_image_text_loss(
                    image_embedder,
                    text_encoder,
                    criterion,
                    images2,
                    texts2,
                    neg2,
                    emb2,
                    local_enabled=False,
                )
            loss = 0.5 * (loss1 + loss2)
            stats = train_module.average_stats([stats1, stats2])
        else:
            loss = loss1
            stats = dict(stats1)

        local_every = max(
            1,
            _integer(
                "LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES",
                getattr(P, "local_hard_negative_every_n_batches", 1),
            ),
        )
        pair_every = max(
            1,
            _integer(
                "IMAGE_PAIR_EVERY_N_BATCHES",
                getattr(P, "image_pair_every_n_batches", 1),
            ),
        )
        local_enabled = (
            torch.is_grad_enabled()
            and P.use_local_hard_negatives
            and P.local_hard_negative_weight > 0
            and counter % local_every == 0
        )
        pair_enabled = (
            torch.is_grad_enabled()
            and P.use_image_pair_contrastive
            and P.image_pair_loss_weight > 0
            and counter % pair_every == 0
        )
        local_max = _integer(
            "LOCAL_HARD_NEGATIVE_MAX_SAMPLES_PER_BATCH",
            getattr(P, "local_hard_negative_max_samples_per_batch", 0),
        )
        pair_max = _integer(
            "IMAGE_PAIR_MAX_SAMPLES_PER_BATCH",
            getattr(P, "image_pair_max_samples_per_batch", 0),
        )
        local_indices = (
            rotating_indices(batch_size, local_max, counter, images1.device)
            if local_enabled
            else []
        )
        pair_indices = (
            rotating_indices(batch_size, pair_max, counter, images1.device)
            if pair_enabled
            else []
        )
        union_indices = sorted(set(local_indices) | set(pair_indices))

        ctx1, loc1, ink1, _raw1 = emb1
        ctx2, loc2, ink2, _raw2 = emb2
        cache1 = build_alignment_cache(
            train_module, text_encoder, criterion, texts1, ctx1, union_indices
        )
        cache2 = build_alignment_cache(
            train_module, text_encoder, criterion, texts2, ctx2, union_indices
        )

        if local_enabled:
            with PROFILER.section("local_hard_negative"):
                local1, local_stats1 = local_loss_from_cache(
                    train_module, cache1, loc1, ink1, local_indices
                )
                local2, local_stats2 = local_loss_from_cache(
                    train_module, cache2, loc2, ink2, local_indices
                )
                local_loss = 0.5 * (local1 + local2)
                local_stats = train_module.average_stats(
                    [local_stats1, local_stats2]
                )
                local_scale = (
                    local_every
                    if os.environ.get("OPTIMIZATION_MODE", "quality").lower()
                    == "fast"
                    else 1
                )
                loss = loss + P.local_hard_negative_weight * local_scale * local_loss
                stats.update(local_stats)
        else:
            stats.update(train_module._zero_local_stats(loc1)[1])

        if pair_enabled:
            with PROFILER.section("image_pair_and_order"):
                pair_losses, order_losses = [], []
                pair_terms = 0
                for index in pair_indices:
                    if index not in cache1 or index not in cache2:
                        continue
                    regions1 = regions_from_alignment(
                        train_module, cache1[index], ctx1[index], loc1[index], ink1[index]
                    )
                    regions2 = regions_from_alignment(
                        train_module, cache2[index], ctx2[index], loc2[index], ink2[index]
                    )
                    pair_loss, matched = train_module.image_image_span_contrastive_loss(
                        regions1,
                        regions2,
                        margin=P.image_pair_margin,
                        top_k=P.image_pair_top_k,
                    )
                    if pair_loss is None:
                        continue
                    pair_losses.append(pair_loss)
                    pair_terms += len(matched)
                    if P.sequence_consistency_loss_weight > 0:
                        order_losses.append(
                            train_module.image_image_order_consistency_loss(
                                regions1, regions2, matched
                            )
                        )
                if pair_losses:
                    pair_loss = torch.stack(pair_losses).mean()
                    order_loss = (
                        torch.stack(order_losses).mean()
                        if order_losses
                        else pair_loss.new_tensor(0.0)
                    )
                    pair_scale = (
                        pair_every
                        if os.environ.get("OPTIMIZATION_MODE", "quality").lower()
                        == "fast"
                        else 1
                    )
                    loss = loss + P.image_pair_loss_weight * pair_scale * pair_loss
                    if P.sequence_consistency_loss_weight > 0:
                        loss = (
                            loss
                            + P.sequence_consistency_loss_weight
                            * pair_scale
                            * order_loss
                        )
                    stats.update(
                        {
                            "image_pair_loss": float(pair_loss.detach().item()),
                            "order_loss": float(order_loss.detach().item()),
                            "pair_terms": float(pair_terms) / max(1, len(pair_losses)),
                        }
                    )
                else:
                    stats.update(
                        {"image_pair_loss": 0.0, "order_loss": 0.0, "pair_terms": 0.0}
                    )
        else:
            stats.update(
                {"image_pair_loss": 0.0, "order_loss": 0.0, "pair_terms": 0.0}
            )

        if hasattr(text_encoder, "cache_stats"):
            stats.update(text_encoder.cache_stats())
        try:
            from jax_span_dtw import bridge_stats

            stats.update(
                {f"jax_{key}": float(value) for key, value in bridge_stats().items()}
            )
        except Exception:
            pass
        stats["total"] = float(loss.detach().item())
        return loss, stats

    return compute_batch_loss


def coalesced_text_allreduce(text_encoder, world_size):
    if world_size <= 1 or not dist.is_initialized():
        return
    parameters = [p for p in text_encoder.parameters() if p.requires_grad]
    if not parameters:
        return
    grads = []
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        grads.append(parameter.grad)
    flat = torch._utils._flatten_dense_tensors(grads)
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat.div_(world_size)
    synced = torch._utils._unflatten_dense_tensors(flat, grads)
    for target, source in zip(grads, synced):
        target.copy_(source)


def optimized_train_one_epoch(train_module):
    def train_one_epoch(model, text_encoder, criterion, optimizer, scaler, loader):
        model.train()
        text_encoder.train() if train_module.has_trainable_parameters(text_encoder) else text_encoder.eval()
        accumulation = max(1, _integer("GRADIENT_ACCUMULATION_STEPS", 1))
        max_batches = max(0, _integer("PROFILE_MAX_BATCHES", 0))
        effective_batches = min(len(loader), max_batches) if max_batches else len(loader)
        clip_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        loss_sum = 0.0
        total_weight = 0
        stats_sum = {}
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.perf_counter()
        previous_end = epoch_started

        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            batch_started = time.perf_counter()
            PROFILER.add("data_wait", batch_started - previous_end)
            is_boundary = (
                ((batch_idx + 1) % accumulation == 0)
                or ((batch_idx + 1) == effective_batches)
            )
            sync_context = nullcontext()
            if isinstance(model, DDP) and not is_boundary:
                sync_context = model.no_sync()

            with sync_context:
                with PROFILER.section("forward_total"):
                    loss, stats = train_module.compute_batch_loss(
                        model, text_encoder, criterion, batch
                    )
                with PROFILER.section("backward"):
                    scaler.scale(loss / accumulation).backward()

            if is_boundary:
                with PROFILER.section("optimizer_and_sync"):
                    scaler.unscale_(optimizer)
                    coalesced_text_allreduce(
                        text_encoder, train_module.CTX.world_size
                    )
                    torch.nn.utils.clip_grad_norm_(clip_parameters, max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            weight = max(1, train_module._batch_size(batch))
            loss_sum += float(loss.detach().item()) * weight
            total_weight += weight
            train_module._accumulate_stats(stats_sum, stats, weight)
            elapsed = time.perf_counter() - batch_started
            samples_per_second = (
                weight * train_module.CTX.world_size / max(elapsed, 1e-9)
            )
            train_module._accumulate_stats(
                stats_sum,
                {"samples_per_second": samples_per_second},
                weight,
            )

            if train_module.CTX.is_main:
                memory = ""
                if train_module.P.log_memory_every_n_batches > 0 and (
                    batch_idx == 0
                    or (batch_idx + 1)
                    % train_module.P.log_memory_every_n_batches
                    == 0
                ):
                    memory = " " + train_module._format_memory(text_encoder)
                print(
                    f"rank=0 batch={batch_idx + 1}/{effective_batches} "
                    f"microbatch={weight} "
                    f"effective_global_batch={weight * train_module.CTX.world_size * accumulation} "
                    f"loss={loss.item():.4f} "
                    f"norm_pos={stats.get('norm_pos', float('nan')):.4f} "
                    f"norm_neg={stats.get('norm_neg', float('nan')):.4f} "
                    f"gap={stats.get('gap', float('nan')):.4f} "
                    f"throughput={samples_per_second:.2f} samples/s "
                    f"time={elapsed:.2f}s{memory}",
                    flush=True,
                )
            previous_end = time.perf_counter()

        PROFILER.add("epoch_wall", time.perf_counter() - epoch_started)
        return train_module._merge_epoch_payload(
            {"loss_sum": loss_sum, "stats_sum": stats_sum, "weight": total_weight}
        )

    return train_one_epoch


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def model_payload(train_module, model, text_encoder, config):
    raw = train_module._unwrap_model(model)
    state = raw.state_dict()
    text_state = text_encoder.state_dict()
    return {
        "model_state_dict": state,
        "image_model_state_dict": state,
        "text_encoder_state_dict": text_state,
        "text_embedder_state_dict": text_state,
        "text_encoder_class": text_encoder.__class__.__name__,
        "text_embedding_class": text_encoder.__class__.__name__,
        "text_encoder_type": train_module.P.text_encoder_type,
        "model_config": config,
    }


def optimized_train(train_module):
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
        parameters = list(model.parameters()) + [
            parameter
            for parameter in text_encoder.parameters()
            if parameter.requires_grad
        ]
        optimizer_kwargs = {"lr": args.learning_rate}
        fused_requested = _flag("USE_FUSED_ADAM", True) and torch.cuda.is_available()
        if fused_requested:
            optimizer_kwargs["fused"] = True
        try:
            optimizer = torch.optim.Adam(parameters, **optimizer_kwargs)
        except (TypeError, RuntimeError):
            optimizer_kwargs.pop("fused", None)
            optimizer = torch.optim.Adam(parameters, **optimizer_kwargs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.learning_rate * 0.01,
        )
        scaler = GradScaler(enabled=train_module.USE_AMP)
        start_epoch = 0
        if resume_payload is not None:
            optimizer.load_state_dict(resume_payload["optimizer_state_dict"])
            scheduler.load_state_dict(resume_payload["scheduler_state_dict"])
            if resume_payload.get("scaler_state_dict") is not None:
                scaler.load_state_dict(resume_payload["scaler_state_dict"])
            start_epoch = int(resume_payload["epoch"]) + 1

        if isinstance(model, DDP) and _flag("DDP_STATIC_GRAPH", True):
            try:
                model._set_static_graph()
            except Exception as exc:
                if train_module.CTX.is_main:
                    print(f"DDP static graph unavailable: {exc}", flush=True)

        run = train_module.init_wandb(args, config)
        history = []
        best_validation = float("inf")
        full_every = max(1, _integer("FULL_CHECKPOINT_EVERY_N_EPOCHS", 5))
        weights_every = max(1, _integer("MODEL_WEIGHTS_EVERY_N_EPOCHS", 2))

        for epoch in range(start_epoch, args.epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_module.CTX.barrier()
            started = time.perf_counter()
            PROFILER.reset()
            if train_module.CTX.is_main:
                print(
                    f"epoch={epoch + 1}/{args.epochs} world_size={train_module.CTX.world_size}",
                    flush=True,
                )

            train_loss, train_stats = train_module.train_one_epoch(
                model,
                text_encoder,
                criterion,
                optimizer,
                scaler,
                train_loader,
            )
            should_validate = (
                ((epoch + 1) % train_module.P.valid_every_n_epochs == 0)
                or ((epoch + 1) == args.epochs)
            )
            if should_validate:
                val_loss, val_stats = train_module.validate(
                    model,
                    text_encoder,
                    criterion,
                    valid_loader,
                    max_batches=train_module.P.valid_max_batches,
                )
            else:
                val_loss, val_stats = float("nan"), {}
            scheduler.step()

            elapsed = time.perf_counter() - started
            profile_summary = PROFILER.summary()
            train_stats["epoch_seconds"] = elapsed
            if train_module.CTX.is_main:
                train_module.wandb_log_epoch_metrics(
                    run, epoch + 1, train_loss, val_loss, train_stats
                )
                if run is not None:
                    extra = {
                        "optimization/epoch_seconds": elapsed,
                        "optimization/learning_rate": scheduler.get_last_lr()[0],
                    }
                    for name, value in profile_summary.items():
                        extra[f"timing/{name}_seconds"] = value["seconds"]
                        extra[f"timing/{name}_mean_seconds"] = value["mean_seconds"]
                    extra.update(
                        {
                            f"validation/{key}": float(value)
                            for key, value in val_stats.items()
                            if isinstance(value, (int, float))
                        }
                    )
                    train_module.wandb.log(extra, step=epoch + 1, commit=True)

                final_epoch = (epoch + 1) == args.epochs
                improved = should_validate and math.isfinite(val_loss) and val_loss < best_validation
                if improved:
                    best_validation = val_loss
                directory = Path(train_module.weights_dir(args.job_id))
                base_payload = model_payload(
                    train_module, model, text_encoder, config
                )
                if final_epoch or improved or ((epoch + 1) % weights_every == 0):
                    atomic_torch_save(base_payload, directory / "model_latest.pth")
                    if improved:
                        atomic_torch_save(base_payload, directory / "model_best.pth")
                if final_epoch or ((epoch + 1) % full_every == 0):
                    checkpoint = dict(base_payload)
                    checkpoint.update(
                        {
                            "epoch": epoch,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                        }
                    )
                    atomic_torch_save(
                        checkpoint, directory / "checkpoint_latest.pth"
                    )

                report_dir = Path("logs") / "performance"
                report_dir.mkdir(parents=True, exist_ok=True)
                report = {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "validation_loss": val_loss,
                    "epoch_seconds": elapsed,
                    "train_stats": train_stats,
                    "validation_stats": val_stats,
                    "profile": profile_summary,
                    "config": config,
                }
                (report_dir / f"{args.job_id}_epoch_{epoch + 1:03d}.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            history.append(train_loss)
            train_module.CTX.barrier()
            if train_module.CTX.is_main:
                print(
                    f"epoch={epoch + 1} train_loss={train_loss:.4f} "
                    f"val_loss={val_loss:.4f} elapsed={elapsed:.1f}s",
                    flush=True,
                )

        if run is not None:
            run.finish()
        return history

    return train


def optimized_rebuild_loader(train_module):
    def rebuild(loader, sampler):
        kwargs = {
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
            kwargs.update(
                {
                    "persistent_workers": True,
                    "prefetch_factor": loader.prefetch_factor or 2,
                    "multiprocessing_context": os.environ.get(
                        "DATALOADER_MP_CONTEXT", "spawn"
                    ),
                }
            )
        return DataLoader(**kwargs)

    return rebuild


def wrap_model_config(train_module):
    original = train_module.model_config

    def model_config(stride, args):
        config = original(stride, args)
        config.update(
            {
                "optimization_entrypoint": True,
                "optimization_mode": os.environ.get("OPTIMIZATION_MODE", "quality"),
                "gradient_accumulation_steps": _integer(
                    "GRADIENT_ACCUMULATION_STEPS", 1
                ),
                "effective_global_batch_size": (
                    train_module.P.batch_size
                    * train_module.CTX.world_size
                    * max(1, _integer("GRADIENT_ACCUMULATION_STEPS", 1))
                ),
                "cnn_chunk_size": _integer("CNN_CHUNK_SIZE", 1024),
                "channels_last": _flag("USE_CHANNELS_LAST", True),
                "allow_tf32": _flag("ALLOW_TF32", True),
                "cudnn_benchmark": _flag("CUDNN_BENCHMARK", True),
                "fused_adam": _flag("USE_FUSED_ADAM", True),
                "ddp_static_graph": _flag("DDP_STATIC_GRAPH", True),
                "profile_training": _flag("PROFILE_TRAINING", False),
                "span_use_blank_transitions": _flag(
                    "SPAN_USE_BLANK_TRANSITIONS", True
                ),
                "span_blank_penalty": _number("SPAN_BLANK_PENALTY", 0.35),
                "span_backbone_batch_size": _integer(
                    "SPAN_BACKBONE_BATCH_SIZE", 512
                ),
            }
        )
        return config

    return model_config


def install(train_module):
    configure_runtime()
    validate_resolved_configuration(train_module.P)
    train_module.compute_batch_loss = optimized_compute_batch_loss(train_module)
    train_module.train_one_epoch = optimized_train_one_epoch(train_module)
    train_module.train = optimized_train(train_module)
    train_module._rebuild_loader = optimized_rebuild_loader(train_module)
    train_module.model_config = wrap_model_config(train_module)
    return train_module
