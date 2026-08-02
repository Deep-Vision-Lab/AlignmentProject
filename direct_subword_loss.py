"""No-DTW region contrastive and interval localization objectives."""
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
    encodings = (
        text_encoder.encode_many(
            list(texts), use_cache=False if text_encoder.training else None
        )
        if hasattr(text_encoder, "encode_many")
        else [text_encoder(text) for text in texts]
    )
    vectors = [encoding.embeddings[_semantic_index(encoding)] for encoding in encodings]
    return F.normalize(torch.stack(vectors).float(), p=2, dim=-1)


def multi_positive_info_nce(
    visual: torch.Tensor,
    text: torch.Tensor,
    labels: Sequence[str],
    temperature: float,
) -> tuple[torch.Tensor, dict]:
    if visual.ndim != 2 or visual.shape != text.shape or not len(labels):
        raise ValueError("Expected matching non-empty visual/text [N,D] tensors")
    logits = visual @ text.T / max(1e-4, float(temperature))
    mask = torch.as_tensor(
        [[left == right for right in labels] for left in labels],
        dtype=torch.bool,
        device=logits.device,
    )

    def direction(values, positives):
        numerator = torch.logsumexp(values.masked_fill(~positives, -torch.inf), 1)
        return (torch.logsumexp(values, 1) - numerator).mean()

    loss = 0.5 * (direction(logits, mask) + direction(logits.T, mask.T))
    cosine = visual @ text.T
    positives = cosine.masked_select(mask)
    negatives = cosine.masked_select(~mask)
    return loss, {
        "positive_similarity": positives.mean(),
        "negative_similarity": (
            negatives.mean() if negatives.numel() else positives.new_tensor(0.0)
        ),
        "positive_mask": mask,
    }


def interval_localization_loss(windows, text_vector, overlap, temperature):
    target = overlap.to(windows.device).float().clamp_min(0.0)
    target = target / target.sum().clamp_min(1e-8)
    logits = F.normalize(windows.float(), dim=-1) @ F.normalize(
        text_vector.float(), dim=-1
    )
    return -(target * F.log_softmax(logits / max(1e-4, temperature), dim=0)).sum()


def _pool(windows, overlap):
    weights = overlap.to(windows.device, windows.dtype)
    weights = weights / weights.sum().clamp_min(1e-8)
    return F.normalize((windows * weights[:, None]).sum(0).float(), dim=-1)


def _outside_loss(windows, text_vector, positive, excluded):
    outside = excluded.to(windows.device) <= 0
    if not outside.any():
        return positive.new_tensor(0.0), None
    similarities = F.normalize(windows[outside].float(), dim=-1) @ text_vector
    k = min(max(1, integer("DIRECT_SUBWORD_OUTSIDE_TOP_K", 8)), similarities.numel())
    hard = torch.topk(similarities, k=int(k)).values
    margin = number("DIRECT_SUBWORD_OUTSIDE_MARGIN", 0.25)
    return torch.relu(margin - positive + hard).mean(), hard.mean()


def _line_regions(contextual, sidecars, text_encoder, window_size, stride, use_flip):
    labels = [str(region["text"]).strip() for sample in sidecars for region in sample]
    if not labels:
        raise ValueError("Direct-subword batch contains no labeled intervals")
    text_vectors = encode_texts(text_encoder, labels)
    examples, localization, outside = [], [], []
    offset = 0
    for image_index, regions in enumerate(sidecars):
        windows = contextual[image_index]
        overlaps, same_text = [], {}
        for region in regions:
            overlap = window_overlap_weights(
                region["x0"], region["x1"],
                num_windows=windows.shape[0], window_size=window_size,
                stride=stride, use_flip=use_flip, device=windows.device,
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
            visual = _pool(windows, overlap)
            positive = visual @ text_vector
            localization.append(
                interval_localization_loss(
                    windows, text_vector, overlap,
                    number("DIRECT_SUBWORD_TEMPERATURE", 0.07),
                )
            )
            out_loss, hard = _outside_loss(
                windows, text_vector, positive, same_text[key]
            )
            outside.append(out_loss)
            examples.append((key, visual, text_vector, positive, hard))
        offset += len(regions)
    return examples, torch.stack(localization).mean(), torch.stack(outside).mean()


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
        contextual, _local, _ink, raw_local = train_module.compute_embeddings(
            image_model, combined
        )
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
        examples, localization, outside = [], [], []
        for windows, sidecars in (
            (contextual[:batch_size], batch["subwords1"]),
            (contextual[batch_size:], batch["subwords2"]),
        ):
            line_examples, line_localization, line_outside = _line_regions(
                windows, sidecars, text_encoder, window_size, stride, use_flip
            )
            examples.extend(line_examples)
            localization.append(line_localization)
            outside.append(line_outside)
        labels = [item[0] for item in examples]
        visual = torch.stack([item[1] for item in examples])
        text = torch.stack([item[2] for item in examples])
        region_loss, values = multi_positive_info_nce(
            visual, text, labels, number("DIRECT_SUBWORD_TEMPERATURE", 0.07)
        )
        localization_loss = torch.stack(localization).mean()
        outside_loss = torch.stack(outside).mean()
        loss = (
            number("DIRECT_SUBWORD_REGION_WEIGHT", 1.0) * region_loss
            + number("DIRECT_SUBWORD_LOCALIZATION_WEIGHT", 1.0) * localization_loss
            + number("DIRECT_SUBWORD_OUTSIDE_WEIGHT", 0.25) * outside_loss
        )
        variance = train_module.image_embedding_variance_loss(
            raw_local, P.image_variance_target_std
        )
        if P.image_variance_loss_weight > 0 and torch.is_grad_enabled():
            loss = loss + P.image_variance_loss_weight * variance
        pos, neg = values["positive_similarity"], values["negative_similarity"]
        pos_cost, neg_cost = 1.0 - pos, 1.0 - neg
        stats = {
            "norm_pos": float(pos_cost.detach()),
            "norm_neg": float(neg_cost.detach()),
            "cost_pos": float(pos_cost.detach()),
            "cost_neg": float(neg_cost.detach()),
            "gap": float((pos_cost - neg_cost).detach()),
            "pos_prob": float(torch.sigmoid(pos).detach()),
            "direct_region_loss": float(region_loss.detach()),
            "direct_localization_loss": float(localization_loss.detach()),
            "direct_outside_loss": float(outside_loss.detach()),
            "direct_positive_similarity": float(pos.detach()),
            "direct_negative_similarity": float(neg.detach()),
            "direct_subword_regions": float(len(examples)),
            "img_var_loss": float(variance.detach()),
            "local_hard_neg": 0.0,
            "local_pos_sim": float(pos.detach()),
            "local_neg_sim": float(neg.detach()),
            "image_pair_loss": 0.0,
            "order_loss": 0.0,
            "pair_terms": float(len(examples)),
            "active_negatives": 0.0,
            "total": float(loss.detach()),
        }
        return loss, stats
    return compute_batch_loss
