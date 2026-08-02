"""No-DTW local-region contrastive and interval-localization objectives."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

from direct_subword_data import flag, integer, number, window_overlap_weights

_SPECIAL = {"<BLANK>", "<SPACE>", "<SUBWORD_BOUNDARY>"}


def _semantic_index(encoding) -> int:
    for index, kind in enumerate(getattr(encoding, "unit_kinds", [])):
        if str(kind) == "subword":
            return index
    for index, text in enumerate(getattr(encoding, "texts", [])):
        if str(text).strip() and str(text) not in _SPECIAL:
            return index
    raise ValueError("Text encoder returned no connected-subword state")


def encode_texts(text_encoder, texts: Sequence[str]) -> torch.Tensor:
    """Encode each unique subword once, then restore the requested order."""
    labels = [str(text).strip() for text in texts]
    unique = list(dict.fromkeys(labels))
    encodings = (
        text_encoder.encode_many(
            unique, use_cache=False if text_encoder.training else None
        )
        if hasattr(text_encoder, "encode_many")
        else [text_encoder(text) for text in unique]
    )
    vectors = {
        text: encoding.embeddings[_semantic_index(encoding)]
        for text, encoding in zip(unique, encodings)
    }
    return F.normalize(
        torch.stack([vectors[text] for text in labels]).float(), p=2, dim=-1
    )


def _class_prototypes(
    text: torch.Tensor, labels: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    unique_labels = list(dict.fromkeys(str(label) for label in labels))
    class_lookup = {label: index for index, label in enumerate(unique_labels)}
    class_ids = torch.as_tensor(
        [class_lookup[str(label)] for label in labels],
        dtype=torch.long,
        device=text.device,
    )
    prototypes = []
    for class_index in range(len(unique_labels)):
        prototype = text[class_ids == class_index].mean(dim=0)
        prototypes.append(F.normalize(prototype.float(), dim=-1))
    return torch.stack(prototypes), class_ids, unique_labels


def multi_positive_info_nce(
    visual: torch.Tensor,
    text: torch.Tensor,
    labels: Sequence[str],
    temperature: float,
) -> tuple[torch.Tensor, dict]:
    """Duplicate-neutral bidirectional supervised contrastive loss.

    Repeated subword strings are collapsed into one text prototype for the
    visual-to-text direction. The reverse direction rewards total probability
    mass assigned to every visual occurrence of that text, so frequent labels
    do not become artificially easy merely because they appear more often.
    """
    if visual.ndim != 2 or visual.shape != text.shape or not len(labels):
        raise ValueError("Expected matching non-empty visual/text [N,D] tensors")
    visual = F.normalize(visual.float(), dim=-1)
    text = F.normalize(text.float(), dim=-1)
    prototypes, class_ids, unique_labels = _class_prototypes(text, labels)
    temperature = max(1e-4, float(temperature))

    visual_to_class = visual @ prototypes.T / temperature
    visual_loss = F.cross_entropy(visual_to_class, class_ids)

    class_to_visual = prototypes @ visual.T / temperature
    class_mask = (
        torch.arange(len(unique_labels), device=visual.device)[:, None]
        == class_ids[None, :]
    )
    numerator = torch.logsumexp(
        class_to_visual.masked_fill(~class_mask, -torch.inf), dim=1
    )
    denominator = torch.logsumexp(class_to_visual, dim=1)
    reverse_loss = (denominator - numerator).mean()
    loss = 0.5 * (visual_loss + reverse_loss)

    pair_mask = class_ids[:, None] == class_ids[None, :]
    cosine = visual @ text.T
    positives = cosine.masked_select(pair_mask)
    negatives = cosine.masked_select(~pair_mask)
    return loss, {
        "positive_similarity": positives.mean(),
        "negative_similarity": (
            negatives.mean() if negatives.numel() else positives.new_tensor(0.0)
        ),
        "positive_mask": pair_mask,
        "num_unique_labels": len(unique_labels),
    }


def interval_localization_loss(windows, text_vector, overlap, temperature):
    """Attention-style auxiliary loss retained for backward-compatible tests."""
    target = overlap.to(windows.device).float().clamp_min(0.0)
    target = target / target.sum().clamp_min(1e-8)
    logits = F.normalize(windows.float(), dim=-1) @ F.normalize(
        text_vector.float(), dim=-1
    )
    return -(target * F.log_softmax(logits / max(1e-4, temperature), dim=0)).sum()


def soft_interval_bce_loss(
    windows,
    text_vector,
    overlap,
    *,
    window_size: int,
    temperature: float,
    similarity_threshold: float,
    focal_gamma: float,
    positive_boost: float,
):
    """Multi-label soft interval supervision for every overlapping window."""
    target = (
        overlap.to(windows.device).float().clamp_min(0.0)
        / max(1.0, float(window_size))
    ).clamp(0.0, 1.0)
    similarities = F.normalize(windows.float(), dim=-1) @ F.normalize(
        text_vector.float(), dim=-1
    )
    logits = (similarities - float(similarity_threshold)) / max(
        1e-4, float(temperature)
    )
    elementwise = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    gamma = max(0.0, float(focal_gamma))
    if gamma > 0:
        probability = torch.sigmoid(logits)
        probability_t = target * probability + (1.0 - target) * (
            1.0 - probability
        )
        elementwise = elementwise * (1.0 - probability_t).pow(gamma)
    weights = 1.0 + max(0.0, float(positive_boost)) * target
    return (elementwise * weights).sum() / weights.sum().clamp_min(1e-8)


def _pool(windows, overlap, ink=None):
    weights = overlap.to(windows.device, windows.dtype).clamp_min(0.0)
    if ink is not None and flag("DIRECT_SUBWORD_USE_INK_WEIGHTING", True):
        ink_floor = max(0.0, number("DIRECT_SUBWORD_INK_FLOOR", 0.05))
        weights = weights * (
            ink.to(windows.device, windows.dtype).clamp(0.0, 1.0) + ink_floor
        )
    if weights.sum().item() <= 0:
        weights = overlap.to(windows.device, windows.dtype).clamp_min(0.0)
    weights = weights / weights.sum().clamp_min(1e-8)
    return F.normalize((windows * weights[:, None]).sum(0).float(), dim=-1)


def _outside_loss(windows, text_vector, positive, excluded, ink=None):
    outside = excluded.to(windows.device) <= 0
    if ink is not None:
        min_ink = max(0.0, number("DIRECT_SUBWORD_OUTSIDE_MIN_INK", 0.01))
        ink_mask = ink.to(windows.device) >= min_ink
        if (outside & ink_mask).any():
            outside = outside & ink_mask
    if not outside.any():
        return positive.new_tensor(0.0), None
    similarities = F.normalize(windows[outside].float(), dim=-1) @ text_vector
    k = min(max(1, integer("DIRECT_SUBWORD_OUTSIDE_TOP_K", 8)), similarities.numel())
    hard = torch.topk(similarities, k=int(k)).values
    margin = number("DIRECT_SUBWORD_OUTSIDE_MARGIN", 0.25)
    return torch.relu(margin - positive + hard).mean(), hard.mean()


def _interval_metrics(windows, text_vector, overlap, stride):
    similarities = F.normalize(windows.float(), dim=-1) @ F.normalize(
        text_vector.float(), dim=-1
    )
    target_mask = overlap.to(windows.device) > 0
    count = int(target_mask.sum().item())
    if count <= 0:
        zero = similarities.new_tensor(0.0)
        return zero, zero, zero, zero, zero
    k = min(count, similarities.numel())
    predicted_indices = torch.topk(similarities, k=k).indices
    predicted_mask = torch.zeros_like(target_mask)
    predicted_mask[predicted_indices] = True
    intersection = (predicted_mask & target_mask).sum().float()
    union = (predicted_mask | target_mask).sum().float().clamp_min(1.0)
    iou = intersection / union

    indices = torch.arange(similarities.numel(), device=similarities.device).float()
    target_weights = overlap.to(similarities.device).float().clamp_min(0.0)
    target_center = (indices * target_weights).sum() / target_weights.sum().clamp_min(1e-8)
    prediction_weights = F.softmax(
        similarities / max(1e-4, number("DIRECT_SUBWORD_METRIC_TEMPERATURE", 0.07)),
        dim=0,
    )
    predicted_center = (indices * prediction_weights).sum()
    center_error = (predicted_center - target_center).abs() * float(stride)

    target_indices = torch.nonzero(target_mask, as_tuple=False).flatten()
    predicted_indices = predicted_indices.sort().values
    start_error = (
        predicted_indices[0].float() - target_indices[0].float()
    ).abs() * float(stride)
    end_error = (
        predicted_indices[-1].float() - target_indices[-1].float()
    ).abs() * float(stride)
    boundary_error = 0.5 * (start_error + end_error)
    return iou, center_error, boundary_error, start_error, end_error


def _line_regions(
    local,
    contextual,
    ink,
    sidecars,
    text_encoder,
    window_size,
    stride,
    use_flip,
):
    labels = [str(region["text"]).strip() for sample in sidecars for region in sample]
    if not labels:
        raise ValueError("Direct-subword batch contains no labeled intervals")
    text_vectors = encode_texts(text_encoder, labels)
    examples = []
    values = {
        "localization": [],
        "context_localization": [],
        "attention": [],
        "outside": [],
        "window_iou": [],
        "center_error": [],
        "boundary_error": [],
        "start_error": [],
        "end_error": [],
    }
    offset = 0
    for image_index, regions in enumerate(sidecars):
        local_windows = local[image_index]
        contextual_windows = contextual[image_index]
        ink_values = ink[image_index] if ink is not None else None
        overlaps, same_text = [], {}
        for region in regions:
            overlap = window_overlap_weights(
                region["x0"],
                region["x1"],
                num_windows=local_windows.shape[0],
                window_size=window_size,
                stride=stride,
                use_flip=use_flip,
                device=local_windows.device,
            )
            if overlap.sum().item() <= 0:
                raise ValueError(f"Interval for {region['text']!r} misses all windows")
            overlaps.append(overlap)
            key = str(region["text"]).strip()
            same_text[key] = torch.maximum(
                same_text.get(key, torch.zeros_like(overlap)), overlap
            )

        for local_index, (region, overlap) in enumerate(zip(regions, overlaps)):
            key = str(region["text"]).strip()
            text_vector = text_vectors[offset + local_index]
            local_visual = _pool(local_windows, overlap, ink_values)
            contextual_visual = _pool(contextual_windows, overlap, ink_values)
            positive = local_visual @ text_vector

            common = {
                "window_size": window_size,
                "temperature": number("DIRECT_SUBWORD_BCE_TEMPERATURE", 0.10),
                "similarity_threshold": number(
                    "DIRECT_SUBWORD_SIMILARITY_THRESHOLD", 0.20
                ),
                "focal_gamma": number("DIRECT_SUBWORD_FOCAL_GAMMA", 1.5),
                "positive_boost": number("DIRECT_SUBWORD_POSITIVE_BOOST", 2.0),
            }
            values["localization"].append(
                soft_interval_bce_loss(
                    local_windows, text_vector, overlap, **common
                )
            )
            values["context_localization"].append(
                soft_interval_bce_loss(
                    contextual_windows, text_vector, overlap, **common
                )
            )
            values["attention"].append(
                interval_localization_loss(
                    contextual_windows,
                    text_vector,
                    overlap,
                    number("DIRECT_SUBWORD_TEMPERATURE", 0.07),
                )
            )
            out_loss, hard = _outside_loss(
                local_windows,
                text_vector,
                positive,
                same_text[key],
                ink_values,
            )
            values["outside"].append(out_loss)
            metrics = _interval_metrics(
                local_windows, text_vector, overlap, stride
            )
            for name, metric in zip(
                (
                    "window_iou",
                    "center_error",
                    "boundary_error",
                    "start_error",
                    "end_error",
                ),
                metrics,
            ):
                values[name].append(metric)
            examples.append(
                (
                    key,
                    local_visual,
                    contextual_visual,
                    text_vector,
                    positive,
                    hard,
                )
            )
        offset += len(regions)

    reduced = {
        key: torch.stack(items).mean()
        for key, items in values.items()
        if items
    }
    return examples, reduced


def make_batch_loss(train_module):
    """Create the synthetic direct-supervision batch loss."""

    def compute_batch_loss(image_model, text_encoder, criterion, batch):
        del criterion
        if not isinstance(batch, dict) or "subwords1" not in batch:
            raise ValueError("Direct mode requires paired synthetic subword sidecars")
        P = train_module.P
        images1 = batch["images1"].to(P.device, non_blocking=True)
        images2 = batch["images2"].to(P.device, non_blocking=True)
        if flag("USE_CHANNELS_LAST", True) and torch.cuda.is_available():
            images1 = images1.contiguous(memory_format=torch.channels_last)
            images2 = images2.contiguous(memory_format=torch.channels_last)

        combined = torch.cat([images1, images2])
        contextual, local, ink, raw_local = train_module.compute_embeddings(
            image_model, combined
        )
        if local is None:
            local = contextual
        batch_size = images1.shape[0]
        model = train_module._unwrap_model(image_model)
        window_size = int(getattr(model, "window_size", P.window_size))
        stride = int(
            getattr(
                model,
                "stride",
                train_module.compute_stride(
                    P.window_size, P.stride_ratio, P.window_overlap_mode
                ),
            )
        )
        use_flip = bool(getattr(model, "use_flip", P.lang.lower() == "arabic"))

        examples = []
        reduced_values = {}
        for contextual_windows, local_windows, ink_values, sidecars in (
            (
                contextual[:batch_size],
                local[:batch_size],
                ink[:batch_size] if ink is not None else None,
                batch["subwords1"],
            ),
            (
                contextual[batch_size:],
                local[batch_size:],
                ink[batch_size:] if ink is not None else None,
                batch["subwords2"],
            ),
        ):
            line_examples, line_values = _line_regions(
                local_windows,
                contextual_windows,
                ink_values,
                sidecars,
                text_encoder,
                window_size,
                stride,
                use_flip,
            )
            examples.extend(line_examples)
            for key, value in line_values.items():
                reduced_values.setdefault(key, []).append(value)

        reduced = {
            key: torch.stack(values).mean()
            for key, values in reduced_values.items()
        }
        labels = [item[0] for item in examples]
        local_visual = torch.stack([item[1] for item in examples])
        contextual_visual = torch.stack([item[2] for item in examples])
        text = torch.stack([item[3] for item in examples])

        region_loss, local_values = multi_positive_info_nce(
            local_visual,
            text,
            labels,
            number("DIRECT_SUBWORD_TEMPERATURE", 0.07),
        )
        context_region_loss, _context_values = multi_positive_info_nce(
            contextual_visual,
            text,
            labels,
            number("DIRECT_SUBWORD_TEMPERATURE", 0.07),
        )
        localization_loss = reduced["localization"]
        context_localization_loss = reduced["context_localization"]
        attention_loss = reduced["attention"]
        outside_loss = reduced["outside"]

        loss = (
            number("DIRECT_SUBWORD_REGION_WEIGHT", 1.0) * region_loss
            + number("DIRECT_SUBWORD_CONTEXT_REGION_WEIGHT", 0.15)
            * context_region_loss
            + number("DIRECT_SUBWORD_LOCALIZATION_WEIGHT", 1.0)
            * localization_loss
            + number("DIRECT_SUBWORD_CONTEXT_LOCALIZATION_WEIGHT", 0.25)
            * context_localization_loss
            + number("DIRECT_SUBWORD_ATTENTION_WEIGHT", 0.10)
            * attention_loss
            + number("DIRECT_SUBWORD_OUTSIDE_WEIGHT", 0.25) * outside_loss
        )
        variance = train_module.image_embedding_variance_loss(
            raw_local, P.image_variance_target_std
        )
        if P.image_variance_loss_weight > 0 and torch.is_grad_enabled():
            loss = loss + P.image_variance_loss_weight * variance

        pos = local_values["positive_similarity"]
        neg = local_values["negative_similarity"]
        pos_cost, neg_cost = 1.0 - pos, 1.0 - neg
        stats = {
            "norm_pos": float(pos_cost.detach()),
            "norm_neg": float(neg_cost.detach()),
            "cost_pos": float(pos_cost.detach()),
            "cost_neg": float(neg_cost.detach()),
            "gap": float((pos_cost - neg_cost).detach()),
            "pos_prob": float(torch.sigmoid(pos).detach()),
            "direct_region_loss": float(region_loss.detach()),
            "direct_context_region_loss": float(context_region_loss.detach()),
            "direct_localization_loss": float(localization_loss.detach()),
            "direct_context_localization_loss": float(
                context_localization_loss.detach()
            ),
            "direct_attention_loss": float(attention_loss.detach()),
            "direct_outside_loss": float(outside_loss.detach()),
            "direct_positive_similarity": float(pos.detach()),
            "direct_negative_similarity": float(neg.detach()),
            "direct_subword_regions": float(len(examples)),
            "direct_unique_subwords": float(local_values["num_unique_labels"]),
            "direct_window_iou": float(reduced["window_iou"].detach()),
            "direct_center_error_px": float(reduced["center_error"].detach()),
            "direct_boundary_error_px": float(reduced["boundary_error"].detach()),
            "direct_start_error_px": float(reduced["start_error"].detach()),
            "direct_end_error_px": float(reduced["end_error"].detach()),
            "img_var_loss": float(variance.detach()),
            "local_hard_neg": float(outside_loss.detach()),
            "local_pos_sim": float(pos.detach()),
            "local_neg_sim": float(neg.detach()),
            "image_pair_loss": 0.0,
            "order_loss": float(context_localization_loss.detach()),
            "pair_terms": float(len(examples)),
            "active_negatives": 0.0,
            "total": float(loss.detach()),
        }
        return loss, stats

    return compute_batch_loss
