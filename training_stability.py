"""Numerical-safety guard for optimized DDP/AMP training.

The optimized trainer manually synchronizes the trainable text-encoder gradients.
Those gradients must be synchronized while they are still scaled, *before*
GradScaler.unscale_ performs its non-finite check. Otherwise an Inf introduced
by manual synchronization can bypass GradScaler and be turned into NaN by
gradient clipping.

This module replaces only ``train_one_epoch`` after ``training_optimizations``
has been installed. It also writes a last-known-finite rescue checkpoint before
an unsafe optimizer step and skips transient non-finite updates.
"""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from training_optimizations import (
    PROFILER,
    atomic_torch_save,
    coalesced_text_allreduce,
    model_payload,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _valid_token_rows(tensor: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    value = tensor.detach().float()
    if mask is None:
        return value.reshape(-1, value.shape[-1])
    valid = mask.detach().bool()
    if valid.shape != value.shape[:2]:
        return value.reshape(-1, value.shape[-1])
    selected = value[valid]
    return selected if selected.numel() else value.reshape(-1, value.shape[-1])


def _point2_representation_diagnostics(vit, record: dict) -> dict[str, float]:
    """Observation-only local/context/fusion diagnostics; never enters the loss."""
    local = record.get("after_resnet18")
    contextual = record.get("after_vit_tiny")
    fused = record.get("after_fusion")
    mask = record.get("token_valid")
    if local is None or contextual is None or fused is None:
        return {}

    with torch.no_grad():
        local_d = local.detach()
        context_d = contextual.detach()
        fused_d = fused.detach()

        local_rows = _valid_token_rows(local_d, mask)
        context_rows = _valid_token_rows(context_d, mask)
        fused_rows = _valid_token_rows(fused_d, mask)

        local_norm = local_rows.norm(dim=-1).mean()
        context_norm = context_rows.norm(dim=-1).mean()
        local_context_cos = F.cosine_similarity(
            local_rows, context_rows, dim=-1
        ).mean()
        context_delta_rel = (
            (context_rows - local_rows).norm(dim=-1)
            / local_rows.norm(dim=-1).clamp_min(1e-8)
        ).mean()

        zeros_local = torch.zeros_like(local_d)
        zeros_context = torch.zeros_like(context_d)
        no_context = vit.fusion_head(local_d, zeros_context)
        no_local = vit.fusion_head(zeros_local, context_d)

        # Deterministically misalign context with its local window without using
        # RNG (and therefore without perturbing training randomness). Permute
        # only valid line tokens so artificial padding cannot drive the result.
        permuted_context = context_d.clone()
        for sample_index in range(int(context_d.shape[0])):
            if mask is None:
                valid_indices = torch.arange(
                    context_d.shape[1], device=context_d.device
                )
            else:
                valid_indices = torch.where(mask[sample_index].detach().bool())[0]
            count = int(valid_indices.numel())
            if count <= 1:
                continue
            shift = max(1, count // 2)
            source_indices = torch.roll(valid_indices, shifts=shift, dims=0)
            permuted_context[sample_index, valid_indices] = context_d[
                sample_index, source_indices
            ]
        permuted = vit.fusion_head(local_d, permuted_context)

        no_context_rows = _valid_token_rows(no_context, mask)
        no_local_rows = _valid_token_rows(no_local, mask)
        permuted_rows = _valid_token_rows(permuted, mask)

        context_sensitivity = (
            1.0
            - F.cosine_similarity(fused_rows, no_context_rows, dim=-1).mean()
        )
        local_sensitivity = (
            1.0
            - F.cosine_similarity(fused_rows, no_local_rows, dim=-1).mean()
        )
        permuted_context_sensitivity = (
            1.0
            - F.cosine_similarity(fused_rows, permuted_rows, dim=-1).mean()
        )

    return {
        "local_norm": float(local_norm.item()),
        "context_norm": float(context_norm.item()),
        "local_context_cos": float(local_context_cos.item()),
        "context_delta_rel": float(context_delta_rel.item()),
        "context_sensitivity": float(context_sensitivity.item()),
        "local_sensitivity": float(local_sensitivity.item()),
        "permuted_context_sensitivity": float(
            permuted_context_sensitivity.item()
        ),
    }


def _global_any(train_module, value: bool) -> bool:
    flag = torch.tensor(
        1 if value else 0,
        dtype=torch.int32,
        device=train_module.P.device,
    )
    if train_module.CTX.enabled:
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(int(flag.item()))


def _all_gradients_finite(parameters) -> bool:
    for parameter in parameters:
        gradient = parameter.grad
        if gradient is not None and not bool(torch.isfinite(gradient).all().item()):
            return False
    return True


def _nonfinite_gradient_names(named_parameters, limit: int = 8):
    bad = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        finite = torch.isfinite(gradient)
        if bool(finite.all().item()):
            continue
        finite_values = gradient.detach().float()[finite]
        max_abs = (
            float(finite_values.abs().max().item())
            if finite_values.numel()
            else float("nan")
        )
        bad.append(
            {
                "name": str(name),
                "shape": tuple(gradient.shape),
                "finite_fraction": float(finite.float().mean().item()),
                "finite_max_abs": max_abs,
            }
        )
        if len(bad) >= int(limit):
            break
    return bad


def _all_parameters_finite(parameters) -> bool:
    for parameter in parameters:
        if not bool(torch.isfinite(parameter.detach()).all().item()):
            return False
    return True


def _trainable_named_parameters(model, text_encoder):
    raw_model = model.module if isinstance(model, DDP) else model
    items = []
    for name, parameter in raw_model.named_parameters():
        if parameter.requires_grad:
            items.append(("model." + name, parameter))
    for name, parameter in text_encoder.named_parameters():
        if parameter.requires_grad:
            items.append(("text." + name, parameter))
    return items


def _snapshot_trainable(named_parameters):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_parameters
    }


def _restore_trainable(named_parameters, snapshot) -> None:
    by_name = dict(named_parameters)
    with torch.no_grad():
        for name, value in snapshot.items():
            parameter = by_name.get(name)
            if parameter is None:
                continue
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def _save_last_finite(
    train_module,
    model,
    text_encoder,
    optimizer,
    scaler,
    config,
    job_id: str,
    *,
    batch_index: int,
    reason: str,
    optimizer_state_valid: bool,
) -> None:
    if train_module.CTX.is_main:
        directory = Path(train_module.weights_dir(job_id))
        payload = model_payload(train_module, model, text_encoder, config)
        payload["stability_rescue"] = {
            "reason": str(reason),
            "batch_index": int(batch_index),
            "optimizer_state_valid": bool(optimizer_state_valid),
            "created_unix_time": float(time.time()),
        }
        atomic_torch_save(payload, directory / "model_last_finite.pth")

        if optimizer_state_valid:
            checkpoint = dict(payload)
            checkpoint.update(
                {
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                }
            )
            atomic_torch_save(
                checkpoint, directory / "checkpoint_last_finite.pth"
            )

        print(
            "stability_guard saved last finite weights "
            f"reason={reason} batch={batch_index} "
            f"path={directory / 'model_last_finite.pth'} "
            f"optimizer_state_valid={int(optimizer_state_valid)}",
            flush=True,
        )
    train_module.CTX.barrier()


def install_training_stability(train_module, config: dict, job_id: str) -> None:
    if getattr(train_module, "_training_stability_installed", False):
        return

    max_nonfinite_skips = max(1, _env_int("NONFINITE_MAX_CONSECUTIVE_SKIPS", 3))
    clip_norm = float(os.environ.get("GRADIENT_CLIP_NORM", "1.0"))
    post_step_guard = _env_flag("NONFINITE_POST_STEP_GUARD", True)

    config.update(
        {
            "stability_guard": True,
            "stability_sync_scaled_text_gradients_before_unscale": True,
            "stability_nonfinite_max_consecutive_skips": max_nonfinite_skips,
            "stability_gradient_clip_norm": clip_norm,
            "stability_post_step_guard": post_step_guard,
            "stability_rescue_weights": "model_last_finite.pth",
        }
    )

    def train_one_epoch(model, text_encoder, criterion, optimizer, scaler, loader):
        """Diagnostic loop: exactly one backward/update per batch, no accumulation."""
        model.train()
        if train_module.has_trainable_parameters(text_encoder):
            text_encoder.train()
        else:
            text_encoder.eval()

        max_batches = max(0, _env_int("PROFILE_MAX_BATCHES", 0))
        effective_batches = min(len(loader), max_batches) if max_batches else len(loader)
        clip_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        named_trainable = _trainable_named_parameters(model, text_encoder)
        trainable_parameters = [parameter for _, parameter in named_trainable]

        def unwrap(module):
            return module.module if isinstance(module, DDP) else module

        def parameter_grad_norm(parameters):
            total = None
            for parameter in parameters:
                if parameter.grad is None:
                    continue
                value = parameter.grad.detach().float().pow(2).sum()
                total = value if total is None else total + value
            if total is None:
                return float("nan")
            return float(total.sqrt().item())

        def activation_grad_norm(tensor, backward_scale):
            if tensor is None or tensor.grad is None:
                return float("nan")
            value = float(tensor.grad.detach().float().norm().item())
            return value / max(float(backward_scale), 1.0)

        epoch_number = int(getattr(train_module, "_diagnostic_epoch", 0))
        loss_sum = 0.0
        total_weight = 0
        stats_sum = {}
        epoch_started = time.perf_counter()
        consecutive_nonfinite = 0

        parameters_bad = _global_any(
            train_module, not _all_parameters_finite(trainable_parameters)
        )
        if parameters_bad:
            raise FloatingPointError(
                "stability_guard: model already contains non-finite parameters "
                "before the epoch."
            )

        for batch_idx, batch in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break

            # One optimizer update per batch. No gradient accumulation.
            batch_weight = max(1, train_module._batch_size(batch))
            optimizer.zero_grad(set_to_none=True)
            raw_model = unwrap(model)
            raw_model._gradient_probe_records = []

            with PROFILER.section("forward_total"):
                loss, stats = train_module.compute_batch_loss(
                    model, text_encoder, criterion, batch
                )

            bad_loss = _global_any(
                train_module,
                not bool(torch.isfinite(loss.detach()).all().item()),
            )
            if bad_loss:
                consecutive_nonfinite += 1
                if consecutive_nonfinite >= max_nonfinite_skips:
                    raise FloatingPointError(
                        "stability_guard: repeated non-finite batch losses."
                    )
                continue

            # Save the AMP scale used for this backward. Parameter gradients are
            # unscaled by GradScaler below; retained activation gradients are not,
            # so their reported norms are divided by this scale explicitly.
            backward_scale = float(scaler.get_scale()) if scaler.is_enabled() else 1.0
            with PROFILER.section("backward"):
                scaler.scale(loss).backward()

            with PROFILER.section("optimizer_and_sync"):
                coalesced_text_allreduce(
                    text_encoder, train_module.CTX.world_size
                )
                scaler.unscale_(optimizer)

                gradients_bad = _global_any(
                    train_module,
                    not _all_gradients_finite(clip_parameters),
                )
                if gradients_bad:
                    bad_gradients = _nonfinite_gradient_names(named_trainable)
                    if train_module.CTX.is_main:
                        print(
                            "NONFINITE_GRAD "
                            f"epoch={epoch_number} batch={batch_idx + 1}/{effective_batches} "
                            f"loss={float(loss.detach().item()):.8f} "
                            f"amp_dtype={getattr(train_module, 'AMP_DTYPE', None)} "
                            f"scaler_enabled={int(scaler.is_enabled())} "
                            f"scale={float(scaler.get_scale()) if scaler.is_enabled() else 1.0:.1f} "
                            f"dtw={float(stats.get('positive_letter_dtw', stats.get('norm_pos', float('nan')))):.8f} "
                            f"sigreg={float(stats.get('sigreg_loss', 0.0)):.8f} "
                            f"sigreg_weighted={float(stats.get('sigreg_weighted', 0.0)):.8f} "
                            f"bad={bad_gradients}",
                            flush=True,
                        )
                    optimizer.zero_grad(set_to_none=True)
                    scaler.update()
                    consecutive_nonfinite += 1
                    if consecutive_nonfinite >= max_nonfinite_skips:
                        raise FloatingPointError(
                            "stability_guard: repeated non-finite gradients."
                        )
                    continue

                vit = raw_model.vit_encoder
                resnet_param_grad = parameter_grad_norm(
                    list(vit.patch_embedding.parameters())
                    + list(vit.local_norm.parameters())
                )
                vit_param_grad = parameter_grad_norm(
                    list(vit.encoder.parameters()) + [vit.position_embedding]
                )
                fusion_param_grad = parameter_grad_norm(
                    list(vit.fusion_head.parameters())
                    + list(raw_model.vision_norm.parameters())
                )

                if train_module.CTX.is_main:
                    print(
                        f"BATCH_LOSS epoch={epoch_number} "
                        f"batch={batch_idx + 1}/{effective_batches} "
                        f"local_batch={batch_weight} "
                        f"global_batch={batch_weight * train_module.CTX.world_size} "
                        f"loss={float(loss.detach().item()):.8f}",
                        flush=True,
                    )
                    records = list(
                        getattr(raw_model, "_gradient_probe_records", [])
                    )
                    point2_enabled = _env_flag("POINT2_DIAGNOSTICS", True)
                    point2_every = max(1, _env_int("POINT2_DIAGNOSTIC_EVERY", 1))
                    for pass_idx, record in enumerate(records, start=1):
                        if point2_enabled and (batch_idx + 1) % point2_every == 0:
                            diag = _point2_representation_diagnostics(vit, record)
                            if diag:
                                print(
                                    f"POINT2 epoch={epoch_number} batch={batch_idx + 1} "
                                    f"pass={pass_idx} "
                                    f"local_norm={diag['local_norm']:.8e} "
                                    f"context_norm={diag['context_norm']:.8e} "
                                    f"local_context_cos={diag['local_context_cos']:.8f} "
                                    f"context_delta_rel={diag['context_delta_rel']:.8f} "
                                    f"context_sensitivity={diag['context_sensitivity']:.8f} "
                                    f"local_sensitivity={diag['local_sensitivity']:.8f} "
                                    f"permuted_context_sensitivity={diag['permuted_context_sensitivity']:.8f}",
                                    flush=True,
                                )
                        print(
                            f"GRAD epoch={epoch_number} batch={batch_idx + 1} pass={pass_idx} "
                            "after=ResNet18 "
                            f"activation_norm={activation_grad_norm(record.get('after_resnet18'), backward_scale):.8e}",
                            flush=True,
                        )
                        print(
                            f"GRAD epoch={epoch_number} batch={batch_idx + 1} pass={pass_idx} "
                            "after=ViT-Tiny "
                            f"activation_norm={activation_grad_norm(record.get('after_vit_tiny'), backward_scale):.8e}",
                            flush=True,
                        )
                        print(
                            f"GRAD epoch={epoch_number} batch={batch_idx + 1} pass={pass_idx} "
                            "after=Fusion "
                            f"activation_norm={activation_grad_norm(record.get('after_fusion'), backward_scale):.8e}",
                            flush=True,
                        )
                        print(
                            f"GRAD epoch={epoch_number} batch={batch_idx + 1} pass={pass_idx} "
                            "after=DTW-input "
                            f"activation_norm={activation_grad_norm(record.get('final_fused'), backward_scale):.8e}",
                            flush=True,
                        )
                    print(
                        f"GRAD epoch={epoch_number} batch={batch_idx + 1} part=ResNet18 "
                        f"parameter_norm={resnet_param_grad:.8e}",
                        flush=True,
                    )
                    print(
                        f"GRAD epoch={epoch_number} batch={batch_idx + 1} part=ViT-Tiny "
                        f"parameter_norm={vit_param_grad:.8e}",
                        flush=True,
                    )
                    print(
                        f"GRAD epoch={epoch_number} batch={batch_idx + 1} part=Fusion "
                        f"parameter_norm={fusion_param_grad:.8e}",
                        flush=True,
                    )

                # Do not keep the last batch graph/tensors alive after diagnostics.
                raw_model._gradient_probe_records = []

                pre_step_snapshot = (
                    _snapshot_trainable(named_trainable)
                    if post_step_guard
                    else None
                )
                total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    clip_parameters,
                    max_norm=clip_norm,
                    error_if_nonfinite=True,
                )
                stats["gradient_norm"] = float(total_grad_norm.detach().item())

                scaler.step(optimizer)
                scaler.update()

                parameters_bad = _global_any(
                    train_module,
                    not _all_parameters_finite(trainable_parameters),
                )
                if parameters_bad:
                    if pre_step_snapshot is not None:
                        _restore_trainable(named_trainable, pre_step_snapshot)
                    raise FloatingPointError(
                        "stability_guard: optimizer produced non-finite parameters."
                    )

                consecutive_nonfinite = 0

            weight = batch_weight
            loss_sum += float(loss.detach().item()) * weight
            total_weight += weight
            train_module._accumulate_stats(stats_sum, stats, weight)

        PROFILER.add("epoch_wall", time.perf_counter() - epoch_started)
        return train_module._merge_epoch_payload(
            {"loss_sum": loss_sum, "stats_sum": stats_sum, "weight": total_weight}
        )

    train_module.train_one_epoch = train_one_epoch
    train_module._training_stability_installed = True
