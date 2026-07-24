from __future__ import annotations

from collections import defaultdict
import os

import torch

import span_alignment_loss as span_loss
import span_alignment_loss_legacy as legacy
from jax_span_dtw import JaxBatchedSpanDTWFunction


def _integer(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _padded_batch_size(size: int) -> int:
    multiple = max(1, _integer("SPAN_DTW_BATCH_BUCKET_SIZE", 8))
    return ((int(size) + multiple - 1) // multiple) * multiple


def _optimized_span_dtw_costs_jax(self, encodings, image_embeddings):
    """Bucket by DP dimensions and pad batch size for JAX compile reuse."""
    if not encodings:
        return image_embeddings.new_empty((0,))

    buckets = defaultdict(list)
    for index, (encoding, image) in enumerate(zip(encodings, image_embeddings)):
        self._check_path_feasible(encoding, int(image.shape[0]))
        padded_text = legacy._bucket_length(
            int(encoding.text_length),
            span_loss.span_dtw_text_bucket_size,
            span_loss.span_dtw_max_text_bucket,
            enabled=span_loss.span_dtw_bucket_text_lengths,
        )
        key = (
            int(getattr(encoding, "max_span_chars", 0)) + 1,
            int(self.max_windows_per_span) + 1,
            int(padded_text) + 1,
            int(image.shape[0]) + 1,
        )
        buckets[key].append(index)

    outputs = [None] * len(encodings)
    for _key, indices in buckets.items():
        dense_items = []
        text_lengths = []
        image_steps = []
        for index in indices:
            encoding = encodings[index]
            image = image_embeddings[index]
            padded_text = legacy._bucket_length(
                int(encoding.text_length),
                span_loss.span_dtw_text_bucket_size,
                span_loss.span_dtw_max_text_bucket,
                enabled=span_loss.span_dtw_bucket_text_lengths,
            )
            dense_items.append(
                span_loss._dense_transition_costs(
                    encoding,
                    image,
                    self.temperature,
                    self.max_windows_per_span,
                    self.window_count_penalty,
                    text_steps_padded=padded_text,
                )
            )
            text_lengths.append(int(encoding.text_length))
            image_steps.append(int(image.shape[0]))

        real_size = len(dense_items)
        padded_size = _padded_batch_size(real_size)
        if padded_size > real_size:
            # Detached padding preserves one fixed JAX shape without adding
            # gradient contributions to any real item. Slicing the returned
            # costs causes PyTorch to send zero grad_output for padded entries.
            template = dense_items[-1].detach()
            padding = padded_size - real_size
            dense_items.extend(template.clone() for _ in range(padding))
            text_lengths.extend([text_lengths[-1]] * padding)
            image_steps.extend([image_steps[-1]] * padding)

        dense_batch = torch.stack(dense_items, dim=0)
        text_tensor = torch.as_tensor(
            text_lengths, device=dense_batch.device, dtype=torch.int32
        )
        image_tensor = torch.as_tensor(
            image_steps, device=dense_batch.device, dtype=torch.int32
        )
        needs_gradient = torch.is_grad_enabled() and dense_batch.requires_grad
        padded_costs = JaxBatchedSpanDTWFunction.apply(
            dense_batch,
            text_tensor,
            image_tensor,
            self.gamma,
            needs_gradient,
        )
        costs = padded_costs[:real_size]
        for local_index, original_index in enumerate(indices):
            outputs[original_index] = costs[local_index]

    return torch.stack(outputs)


def install_jax_batch_padding():
    span_loss.SpanContrastiveSoftDTW._span_dtw_costs_jax = (
        _optimized_span_dtw_costs_jax
    )
