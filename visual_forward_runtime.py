"""Control whether line images are encoded one-by-one or as a batched tensor.

The training optimizer may concatenate line1 and line2 along the *batch* axis for
throughput.  This runtime keeps that optimization optional without changing any
losses or pair semantics.

Modes
-----
per_line (default)
    Every image line is passed through the visual model in its own forward call.
    Returned embeddings are concatenated back along the batch dimension before
    the existing losses run.

batched
    Preserve the previous fast path: all images supplied to ``compute_embeddings``
    are passed through the visual model in one forward call.  For paired training,
    ``training_optimizations.py`` may first batch-concatenate line1 and line2.

Set with ``VISUAL_FORWARD_MODE=per_line`` or ``VISUAL_FORWARD_MODE=batched``.
"""
from __future__ import annotations

import os

import torch


_VALID_MODES = {"per_line", "batched"}


def resolve_visual_forward_mode() -> str:
    value = os.environ.get("VISUAL_FORWARD_MODE", "per_line").strip().lower()
    aliases = {
        "single": "per_line",
        "separate": "per_line",
        "individual": "per_line",
        "batch": "batched",
        "concat": "batched",
        "batch_concat": "batched",
    }
    value = aliases.get(value, value)
    if value not in _VALID_MODES:
        raise ValueError(
            "VISUAL_FORWARD_MODE must be 'per_line' or 'batched', "
            f"got {value!r}"
        )
    return value


def _concat_outputs(outputs):
    first = outputs[0]
    if torch.is_tensor(first):
        return torch.cat(outputs, dim=0)
    if isinstance(first, tuple):
        return tuple(
            torch.cat([item[index] for item in outputs], dim=0)
            if torch.is_tensor(first[index])
            else first[index]
            for index in range(len(first))
        )
    raise TypeError(
        "compute_embeddings returned an unsupported type for per-line execution: "
        f"{type(first).__name__}"
    )


def install(train_module) -> str:
    """Wrap ``trainer_core.compute_embeddings`` according to visual forward mode."""
    if getattr(train_module, "_visual_forward_runtime_installed", False):
        return resolve_visual_forward_mode()

    mode = resolve_visual_forward_mode()
    original_compute_embeddings = train_module.compute_embeddings

    def compute_embeddings(image_embedder, images):
        if mode == "batched" or not torch.is_tensor(images) or images.shape[0] <= 1:
            return original_compute_embeddings(image_embedder, images)

        # Intentionally preserve a batch dimension of one for every line.
        # The same shared model and gradients are used for all calls.
        outputs = [
            original_compute_embeddings(image_embedder, images[index : index + 1])
            for index in range(int(images.shape[0]))
        ]
        return _concat_outputs(outputs)

    train_module.compute_embeddings = compute_embeddings
    train_module._visual_forward_runtime_installed = True
    train_module._visual_forward_mode = mode
    return mode
