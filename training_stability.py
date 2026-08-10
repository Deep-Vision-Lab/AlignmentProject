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
        model.train()
        if train_module.has_trainable_parameters(text_encoder):
            text_encoder.train()
        else:
            text_encoder.eval()

        accumulation = max(1, _env_int("GRADIENT_ACCUMULATION_STEPS", 1))
        max_batches = max(0, _env_int("PROFILE_MAX_BATCHES", 0))
        effective_batches = min(len(loader), max_batches) if max_batches else len(loader)
        clip_parameters = [
            parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        named_trainable = _trainable_named_parameters(model, text_encoder)
        trainable_parameters = [parameter for _, parameter in named_trainable]

        loss_sum = 0.0
        total_weight = 0
        stats_sum = {}
        optimizer.zero_grad(set_to_none=True)
        epoch_started = time.perf_counter()
        previous_end = epoch_started
        consecutive_nonfinite = 0

        parameters_bad = _global_any(
            train_module, not _all_parameters_finite(trainable_parameters)
        )
        if parameters_bad:
            raise FloatingPointError(
                "stability_guard: model already contains non-finite parameters "
                "before the epoch; use model_last_finite.pth or an earlier checkpoint."
            )

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

                bad_loss = _global_any(
                    train_module,
                    not bool(torch.isfinite(loss.detach()).all().item()),
                )
                if bad_loss:
                    optimizer.zero_grad(set_to_none=True)
                    consecutive_nonfinite += 1
                    _save_last_finite(
                        train_module,
                        model,
                        text_encoder,
                        optimizer,
                        scaler,
                        config,
                        job_id,
                        batch_index=batch_idx + 1,
                        reason="nonfinite_loss_before_backward",
                        optimizer_state_valid=True,
                    )
                    if consecutive_nonfinite >= max_nonfinite_skips:
                        raise FloatingPointError(
                            "stability_guard: repeated non-finite losses; rescue "
                            "weights were saved to model_last_finite.pth"
                        )
                    if train_module.CTX.is_main:
                        print(
                            "stability_guard skipped batch due to non-finite loss "
                            f"batch={batch_idx + 1} consecutive={consecutive_nonfinite}/"
                            f"{max_nonfinite_skips}",
                            flush=True,
                        )
                    previous_end = time.perf_counter()
                    continue

                with PROFILER.section("backward"):
                    scaler.scale(loss / accumulation).backward()

            if is_boundary:
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
                        _save_last_finite(
                            train_module,
                            model,
                            text_encoder,
                            optimizer,
                            scaler,
                            config,
                            job_id,
                            batch_index=batch_idx + 1,
                            reason="nonfinite_gradient_before_optimizer_step",
                            optimizer_state_valid=True,
                        )
                        optimizer.zero_grad(set_to_none=True)
                        scaler.update()
                        consecutive_nonfinite += 1
                        if train_module.CTX.is_main:
                            print(
                                "stability_guard skipped unsafe optimizer step "
                                f"batch={batch_idx + 1} consecutive={consecutive_nonfinite}/"
                                f"{max_nonfinite_skips}",
                                flush=True,
                            )
                        if consecutive_nonfinite >= max_nonfinite_skips:
                            raise FloatingPointError(
                                "stability_guard: repeated non-finite gradients; "
                                "rescue weights were saved to model_last_finite.pth"
                            )
                        weight = max(1, train_module._batch_size(batch))
                        loss_sum += float(loss.detach().item()) * weight
                        total_weight += weight
                        train_module._accumulate_stats(stats_sum, stats, weight)
                        previous_end = time.perf_counter()
                        continue

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
                    if train_module.CTX.is_main:
                        stats["gradient_norm"] = float(total_grad_norm.detach().item())

                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                    parameters_bad = _global_any(
                        train_module,
                        not _all_parameters_finite(trainable_parameters),
                    )
                    if parameters_bad:
                        if pre_step_snapshot is not None:
                            _restore_trainable(named_trainable, pre_step_snapshot)
                        _save_last_finite(
                            train_module,
                            model,
                            text_encoder,
                            optimizer,
                            scaler,
                            config,
                            job_id,
                            batch_index=batch_idx + 1,
                            reason="nonfinite_parameter_after_optimizer_step_rolled_back",
                            optimizer_state_valid=False,
                        )
                        raise FloatingPointError(
                            "stability_guard: optimizer produced non-finite parameters; "
                            "the pre-step weights were restored and saved to "
                            "model_last_finite.pth"
                        )

                    consecutive_nonfinite = 0

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
                grad_text = ""
                if "gradient_norm" in stats:
                    grad_text = f" grad_norm={stats['gradient_norm']:.4f}"
                print(
                    f"rank=0 batch={batch_idx + 1}/{effective_batches} "
                    f"microbatch={weight} "
                    f"effective_global_batch={weight * train_module.CTX.world_size * accumulation} "
                    f"loss={loss.item():.4f} "
                    f"norm_pos={stats.get('norm_pos', float('nan')):.4f} "
                    f"norm_neg={stats.get('norm_neg', float('nan')):.4f} "
                    f"gap={stats.get('gap', float('nan')):.4f}"
                    f"{grad_text} "
                    f"throughput={samples_per_second:.2f} samples/s "
                    f"time={elapsed:.2f}s{memory}",
                    flush=True,
                )
            previous_end = time.perf_counter()

        PROFILER.add("epoch_wall", time.perf_counter() - epoch_started)
        return train_module._merge_epoch_payload(
            {"loss_sum": loss_sum, "stats_sum": stats_sum, "weight": total_weight}
        )

    train_module.train_one_epoch = train_one_epoch
    train_module._training_stability_installed = True
