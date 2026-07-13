import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

import Parameters as P
from alignment_visualization import save_d3tw_visualization
from arabic_span_text_encoder import ArabicSpanTextEncoder
from arabic_token_text_encoder import ArabicTokenTextEncoder
from LossFunctionWithHelpers import ContrastiveSoftDTW
from embeddingModel import EmbeddingModel
from DataLoader import build_dataloaders
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


def weights_dir(job_id):
    path = os.path.join(os.path.dirname(__file__), "Weights", job_id)
    os.makedirs(path, exist_ok=True)
    return path


def model_config(stride):
    return {
        "window_size": P.window_size,
        "stride": stride,
        "stride_ratio": P.stride_ratio,
        "window_overlap_mode": P.window_overlap_mode,
        "vector_size": P.vector_size,
        "lang": P.lang,
        "negative_mode": P.negative_mode,
        "num_negatives": P.num_negatives,
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
        "use_local_hard_negatives": P.use_local_hard_negatives,
        "local_hard_negative_weight": P.local_hard_negative_weight,
        "local_hard_negative_margin": P.local_hard_negative_margin,
        "local_hard_negative_top_k": P.local_hard_negative_top_k,
        "local_hard_negative_exclude_radius": P.local_hard_negative_exclude_radius,
        "local_hard_negative_min_ink": P.local_hard_negative_min_ink,
        "use_image_pair_contrastive": P.use_image_pair_contrastive,
        "image_pair_loss_weight": P.image_pair_loss_weight,
        "image_pair_margin": P.image_pair_margin,
        "image_pair_top_k": P.image_pair_top_k,
        "sequence_consistency_loss_weight": P.sequence_consistency_loss_weight,
        "image_variance_loss_weight": P.image_variance_loss_weight,
        "image_variance_target_std": P.image_variance_target_std,
        "valid_every_n_epochs": P.valid_every_n_epochs,
        "valid_max_batches": P.valid_max_batches,
        "log_memory_every_n_batches": P.log_memory_every_n_batches,
    }


def save_model_weights(model, text_encoder, job_id, config):
    path = os.path.join(weights_dir(job_id), "model_latest.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "image_model_state_dict": model.state_dict(),
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
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "image_model_state_dict": model.state_dict(),
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
        sim_pos = torch.einsum("sv,tv->st", norm_pos_text, norm_img[sample_idx])
        sim_pos_list.append(sim_pos)
        sample_neg_sims = []
        for neg_text in neg_texts[sample_idx]:
            norm_neg_text = embed_single_text(text_encoder, neg_text)
            sample_neg_sims.append(torch.einsum("tv,sv->ts", norm_neg_text, norm_img[sample_idx]))
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
    ink_ratio = ink_ratio.to(local_image_embeddings.device).float() if ink_ratio is not None else None
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
        pos_img = F.normalize(ink_weighted_mean(local_image_embeddings[w0:w1], region_ink), p=2, dim=-1)
        sim_pos = torch.matmul(pos_img, span_vec)

        neg_mask = torch.ones(num_windows, dtype=torch.bool, device=local_image_embeddings.device)
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


def local_hard_negative_loss_for_batch(text_encoder, criterion, norm_context_img, norm_local_img, pos_texts, ink_ratios=None):
    if not P.use_local_hard_negatives or P.local_hard_negative_weight <= 0 or not torch.is_grad_enabled():
        return _zero_local_stats(norm_local_img)
    losses = []
    stats_list = []
    for sample_idx, pos_text in enumerate(pos_texts):
        pos_encoding = text_encoder(pos_text, use_cache=False if text_encoder.training else None)
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
        "local_pos_sim": sum(s["local_pos_sim"] for s in stats_list) / len(stats_list),
        "local_neg_sim": sum(s["local_neg_sim"] for s in stats_list) / len(stats_list),
        "local_terms": sum(s["local_terms"] for s in stats_list) / len(stats_list),
    }


def extract_aligned_span_regions(text_encoder, criterion, text, norm_context_img, norm_local_img, ink_ratio=None):
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
    ink_ratio = ink_ratio.to(norm_local_img.device).float() if ink_ratio is not None else None
    for step in path:
        span_idx = int(step["span_idx"])
        w0 = int(step["window_start"])
        w1 = int(step["window_end"])
        if w1 <= w0 or w0 < 0 or w1 > norm_local_img.shape[0]:
            continue
        span_text = encoding.texts[span_idx]
        if not span_text.strip():
            continue
        region_ink = ink_ratio[w0:w1] if ink_ratio is not None else None
        if region_ink is not None and region_ink.max().item() < P.local_hard_negative_min_ink:
            continue
        region_vec = F.normalize(ink_weighted_mean(norm_local_img[w0:w1], region_ink), p=2, dim=-1)
        regions.append({
            "span_text": span_text,
            "span_idx": span_idx,
            "window_start": w0,
            "window_end": w1,
            "center": norm_local_img.new_tensor(0.5 * (w0 + w1 - 1)),
            "vec": region_vec,
        })
    return regions


def image_image_span_contrastive_loss(regions1, regions2, margin=0.35, top_k=8):
    if not regions1 or not regions2:
        return None, []
    vecs1 = torch.stack([r["vec"] for r in regions1], dim=0)
    vecs2 = torch.stack([r["vec"] for r in regions2], dim=0)
    sim = torch.matmul(vecs1, vecs2.T)
    losses = []
    matched_pairs = []
    for i, r1 in enumerate(regions1):
        candidates = [j for j, r2 in enumerate(regions2) if r2["span_text"] == r1["span_text"]]
        if not candidates:
            continue
        pos_j = max(candidates, key=lambda j: float(sim[i, j].detach()))
        sim_pos = sim[i, pos_j]
        neg_mask = torch.ones(sim.shape[1], dtype=torch.bool, device=sim.device)
        neg_mask[pos_j] = False
        if neg_mask.sum().item() == 0:
            continue
        sim_negs = sim[i, neg_mask]
        k = min(max(1, int(top_k)), int(sim_negs.numel()))
        hard_negs = torch.topk(sim_negs, k=k).values
        losses.append(torch.relu(float(margin) - sim_pos + hard_negs).mean())
        matched_pairs.append((i, pos_j))
    if not losses:
        return vecs1.new_tensor(0.0), matched_pairs
    return torch.stack(losses).mean(), matched_pairs


def image_image_order_consistency_loss(regions1, regions2, matched_pairs):
    if len(matched_pairs) < 2 or not regions1:
        device = regions1[0]["vec"].device if regions1 else (regions2[0]["vec"].device if regions2 else P.device)
        return torch.tensor(0.0, device=device)
    penalties = []
    for (i1, j1), (i2, j2) in zip(matched_pairs[:-1], matched_pairs[1:]):
        d1 = regions1[i2]["center"] - regions1[i1]["center"]
        d2 = regions2[j2]["center"] - regions2[j1]["center"]
        penalties.append(torch.relu(-(d1 * d2)))
    return torch.stack(penalties).mean() if penalties else regions1[0]["vec"].new_tensor(0.0)


def compute_embeddings(image_embedder, images):
    with autocast(dtype=AMP_DTYPE, enabled=USE_AMP):
        img_emb, local_img_emb, ink_ratio = image_embedder(images, return_local=True, return_ink=True)
    return (
        F.normalize(img_emb.float(), p=2, dim=-1),
        F.normalize(local_img_emb.float(), p=2, dim=-1),
        ink_ratio,
        local_img_emb,
    )


def compute_single_image_text_loss(image_embedder, text_encoder, criterion, images, pos_texts, neg_texts, embeddings=None):
    if P.text_encoder_type == "arabic_span":
        if embeddings is None:
            norm_img, norm_local_img, ink_ratio, local_img_emb = compute_embeddings(image_embedder, images)
        else:
            norm_img, norm_local_img, ink_ratio, local_img_emb = embeddings
        loss, stats = criterion.forward_varlen(text_encoder, norm_img, pos_texts, neg_texts)
        local_loss, local_stats = local_hard_negative_loss_for_batch(
            text_encoder, criterion, norm_img, norm_local_img, pos_texts, ink_ratios=ink_ratio
        )
        if P.use_local_hard_negatives and P.local_hard_negative_weight > 0 and torch.is_grad_enabled():
            loss = loss + P.local_hard_negative_weight * local_loss
        stats.update(local_stats)
        var_loss = image_embedding_variance_loss(local_img_emb, P.image_variance_target_std)
        if P.image_variance_loss_weight > 0 and torch.is_grad_enabled():
            loss = loss + P.image_variance_loss_weight * var_loss
        stats["img_var_loss"] = float(var_loss.detach().item())
        stats["total"] = float(loss.detach().item())
        return loss, stats, (norm_img, norm_local_img, ink_ratio, local_img_emb)

    with autocast(dtype=AMP_DTYPE, enabled=USE_AMP):
        img_emb = image_embedder(images)
    norm_img = F.normalize(img_emb.float(), p=2, dim=-1)
    sim_pos_list, sim_neg_list = compute_similarity_lists(text_encoder, norm_img, pos_texts, neg_texts)
    loss, stats = criterion.forward_varlen(sim_pos_list, sim_neg_list)
    var_loss = image_embedding_variance_loss(img_emb, P.image_variance_target_std)
    if P.image_variance_loss_weight > 0 and torch.is_grad_enabled():
        loss = loss + P.image_variance_loss_weight * var_loss
    stats["img_var_loss"] = float(var_loss.detach().item())
    stats["total"] = float(loss.detach().item())
    return loss, stats, None


def compute_image_pair_loss(text_encoder, criterion, texts1, texts2, emb1, emb2):
    if not P.use_image_pair_contrastive or P.image_pair_loss_weight <= 0 or P.text_encoder_type != "arabic_span" or not torch.is_grad_enabled():
        base = emb1[0].new_tensor(0.0)
        return base, base, {"image_pair_loss": 0.0, "order_loss": 0.0, "pair_terms": 0.0}
    norm_ctx1, norm_loc1, ink1, _raw1 = emb1
    norm_ctx2, norm_loc2, ink2, _raw2 = emb2
    pair_losses = []
    order_losses = []
    terms = 0
    for b in range(norm_ctx1.shape[0]):
        try:
            regions1 = extract_aligned_span_regions(text_encoder, criterion, texts1[b], norm_ctx1[b], norm_loc1[b], ink1[b])
            regions2 = extract_aligned_span_regions(text_encoder, criterion, texts2[b], norm_ctx2[b], norm_loc2[b], ink2[b])
        except ValueError:
            continue
        pair_loss, matched_pairs = image_image_span_contrastive_loss(
            regions1, regions2, margin=P.image_pair_margin, top_k=P.image_pair_top_k
        )
        if pair_loss is None:
            continue
        pair_losses.append(pair_loss)
        order_losses.append(image_image_order_consistency_loss(regions1, regions2, matched_pairs))
        terms += len(matched_pairs)
    if not pair_losses:
        base = norm_ctx1.new_tensor(0.0)
        return base, base, {"image_pair_loss": 0.0, "order_loss": 0.0, "pair_terms": 0.0}
    pair_loss = torch.stack(pair_losses).mean()
    order_loss = torch.stack(order_losses).mean() if order_losses else pair_loss.new_tensor(0.0)
    return pair_loss, order_loss, {
        "image_pair_loss": float(pair_loss.detach().item()),
        "order_loss": float(order_loss.detach().item()),
        "pair_terms": float(terms) / max(1, len(pair_losses)),
    }


def compute_batch_loss(image_embedder, text_encoder, criterion, batch):
    if isinstance(batch, dict):
        images1 = batch["images1"].to(P.device, non_blocking=True)
        images2 = batch["images2"].to(P.device, non_blocking=True)
        texts1 = batch["texts1"]
        texts2 = batch["texts2"]
        neg_texts1 = batch["neg_texts1"]
        neg_texts2 = batch["neg_texts2"]
        emb1 = compute_embeddings(image_embedder, images1) if P.text_encoder_type == "arabic_span" else None
        emb2 = compute_embeddings(image_embedder, images2) if P.text_encoder_type == "arabic_span" else None
        loss1, stats1, emb1 = compute_single_image_text_loss(image_embedder, text_encoder, criterion, images1, texts1, neg_texts1, emb1)
        loss2, stats2, emb2 = compute_single_image_text_loss(image_embedder, text_encoder, criterion, images2, texts2, neg_texts2, emb2)
        loss = 0.5 * (loss1 + loss2)
        stats = average_stats([stats1, stats2])
        if emb1 is not None and emb2 is not None:
            pair_loss, order_loss, pair_stats = compute_image_pair_loss(text_encoder, criterion, texts1, texts2, emb1, emb2)
            loss = loss + P.image_pair_loss_weight * pair_loss
            if P.sequence_consistency_loss_weight > 0 and torch.is_grad_enabled():
                loss = loss + P.sequence_consistency_loss_weight * order_loss
            stats.update(pair_stats)
        stats["total"] = float(loss.detach().item())
        return loss, stats

    images, pos_texts, neg_texts = batch
    images = images.to(P.device, non_blocking=True)
    loss, stats, _emb = compute_single_image_text_loss(image_embedder, text_encoder, criterion, images, pos_texts, neg_texts)
    return loss, stats


def average_stats(stats_list):
    if not stats_list:
        return {}
    keys = set()
    for stats in stats_list:
        keys.update(stats.keys())
    return {key: sum(stats.get(key, 0.0) for stats in stats_list) / len(stats_list) for key in keys}


def _format_memory(text_encoder):
    parts = []
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        parts.append(f"gpu={allocated:.2f}/{reserved:.2f}GB")
    if _PROCESS is not None:
        rss = _PROCESS.memory_info().rss / (1024 ** 3)
        parts.append(f"rss={rss:.2f}GB")
    if hasattr(text_encoder, "cache_size_current"):
        parts.append(f"span_cache={text_encoder.cache_size_current()}")
    return " ".join(parts)


def clear_text_encoder_cache(text_encoder):
    if hasattr(text_encoder, "clear_cache"):
        text_encoder.clear_cache()


def train_one_epoch(model, text_encoder, criterion, optimizer, scaler, loader):
    model.train()
    text_encoder.train() if has_trainable_parameters(text_encoder) else text_encoder.eval()
    total = 0.0
    stats_list = []
    for batch_idx, batch in enumerate(loader):
        batch_started = time.time()
        optimizer.zero_grad(set_to_none=True)
        loss, stats = compute_batch_loss(model, text_encoder, criterion, batch)
        forward_elapsed = time.time() - batch_started
        backward_started = time.time()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for g in optimizer.param_groups for p in g["params"]], max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        backward_elapsed = time.time() - backward_started
        elapsed = time.time() - batch_started
        total += loss.item()
        stats_list.append(stats)
        mem_suffix = ""
        if P.log_memory_every_n_batches > 0 and (batch_idx == 0 or (batch_idx + 1) % P.log_memory_every_n_batches == 0):
            mem_suffix = " " + _format_memory(text_encoder)
        print(
            f"batch={batch_idx + 1}/{len(loader)} "
            f"loss={loss.item():.4f} "
            f"norm_pos={stats.get('norm_pos', float('nan')):.4f} "
            f"norm_neg={stats.get('norm_neg', float('nan')):.4f} "
            f"gap={stats.get('gap', float('nan')):.4f} "
            f"pos_prob={stats.get('pos_prob', float('nan')):.4f} "
            f"local_hard={stats.get('local_hard_neg', 0.0):.4f} "
            f"local_pos={stats.get('local_pos_sim', 0.0):.4f} "
            f"local_neg={stats.get('local_neg_sim', 0.0):.4f} "
            f"pair={stats.get('image_pair_loss', 0.0):.4f} "
            f"order={stats.get('order_loss', 0.0):.4f} "
            f"pair_terms={stats.get('pair_terms', 0.0):.1f} "
            f"img_var={stats.get('img_var_loss', 0.0):.4f} "
            f"cost_pos={stats.get('cost_pos', float('nan')):.2f} "
            f"cost_neg={stats.get('cost_neg', float('nan')):.2f} "
            f"forward={forward_elapsed:.1f}s backward={backward_elapsed:.1f}s time={elapsed:.1f}s"
            f"{mem_suffix}",
            flush=True,
        )
    return total / max(len(loader), 1), average_stats(stats_list)


@torch.no_grad()
def validate(model, text_encoder, criterion, loader, max_batches=0):
    model.eval()
    text_encoder.eval()
    total = 0.0
    stats_list = []
    for batch_idx, batch in enumerate(loader):
        if max_batches and batch_idx >= max_batches:
            break
        loss, stats = compute_batch_loss(model, text_encoder, criterion, batch)
        total += loss.item()
        stats_list.append(stats)
    return total / max(len(stats_list), 1), average_stats(stats_list)


def init_wandb(args, config):
    if os.environ.get("USE_WANDB", "1") == "0":
        return None
    if wandb is None:
        print("wandb is not installed; continuing without W&B.", flush=True)
        return None
    return wandb.init(project=os.environ.get("WANDB_PROJECT", "alignment-project"), name=args.job_id, config=config)


def wandb_log_epoch_metrics(run, epoch, train_loss, val_loss, train_stats):
    if run is not None:
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


def train(model, text_encoder, criterion, train_loader, valid_loader, args, config):
    trainable_params = list(model.parameters()) + [p for p in text_encoder.parameters() if p.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.learning_rate * 0.01)
    scaler = GradScaler(enabled=USE_AMP)
    run = init_wandb(args, config)
    start_epoch = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=P.device)
        model.load_state_dict(extract_model_state(checkpoint))
        if "text_encoder_state_dict" in checkpoint:
            text_encoder.load_state_dict(checkpoint["text_encoder_state_dict"], strict=False)
        elif "text_embedder_state_dict" in checkpoint:
            text_encoder.load_state_dict(checkpoint["text_embedder_state_dict"], strict=False)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if checkpoint.get("scaler_state_dict") is not None:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1

    history = []
    for epoch in range(start_epoch, args.epochs):
        started = time.time()
        print(f"epoch={epoch + 1}/{args.epochs}", flush=True)
        train_loss, train_stats = train_one_epoch(model, text_encoder, criterion, optimizer, scaler, train_loader)
        should_validate = ((epoch + 1) % P.valid_every_n_epochs == 0) or ((epoch + 1) == args.epochs)
        if should_validate:
            clear_text_encoder_cache(text_encoder)
            val_loss, _val_stats = validate(model, text_encoder, criterion, valid_loader, max_batches=P.valid_max_batches)
            clear_text_encoder_cache(text_encoder)
        else:
            val_loss = float("nan")
        scheduler.step()
        wandb_log_epoch_metrics(run, epoch + 1, train_loss, val_loss, train_stats)
        save_checkpoint(model, text_encoder, optimizer, scheduler, scaler, epoch, args.job_id, config)
        save_model_weights(model, text_encoder, args.job_id, config)

        should_visualize = (((epoch + 1) % 10 == 0) or ((epoch + 1) == args.epochs)) and not P.use_image_pair_contrastive
        if should_visualize:
            save_d3tw_visualization(model, text_encoder, valid_loader, criterion, epoch + 1, args.job_id, P.device)
            clear_text_encoder_cache(text_encoder)
        if P.clear_span_cache_each_epoch:
            clear_text_encoder_cache(text_encoder)
        history.append(train_loss)
        elapsed = time.time() - started
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} elapsed={elapsed:.1f}s", flush=True)

    if run is not None:
        run.finish()
    return history


def parse_args():
    parser = argparse.ArgumentParser(description="Train the alignment model")
    parser.add_argument("--job_id", required=True)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--pretrained_weights", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--window_size", type=int, default=None)
    parser.add_argument("--stride_ratio", type=float, default=None)
    parser.add_argument("--window_overlap_mode", choices=["no_overlap", "light_overlap", "dense_overlap", "custom"], default=None)
    parser.add_argument("--negative_mode", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--num_negatives", type=int, default=None)
    parser.add_argument("--use_bilstm", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_local_hard_negatives", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--local_hard_negative_weight", type=float, default=None)
    parser.add_argument("--image_variance_loss_weight", type=float, default=None)
    parser.add_argument("--use_image_pair_contrastive", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--image_pair_loss_weight", type=float, default=None)
    return parser.parse_args()


def apply_overrides(args):
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
        import DataLoader
        DataLoader.num_negatives = args.num_negatives
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
        args.learning_rate = P.finetune_learning_rate if args.learning_rate is None else args.learning_rate
        args.epochs = P.finetune_epochs if args.epochs is None else args.epochs
    else:
        args.learning_rate = P.learning_rate if args.learning_rate is None else args.learning_rate
        args.epochs = P.epochs if args.epochs is None else args.epochs


def select_dataloaders(args):
    data_dir = args.data_dir
    if data_dir is None and args.finetune:
        data_dir = P.finetune_data_dir
    return build_dataloaders(data_dir)


def main():
    args = parse_args()
    apply_overrides(args)
    if args.resume and args.pretrained_weights:
        raise SystemExit("Use either --resume or --pretrained_weights, not both.")
    stride = compute_stride(P.window_size, P.stride_ratio, P.window_overlap_mode)
    train_loader, valid_loader, _test_loader = select_dataloaders(args)
    text_encoder = build_text_encoder()
    model = build_image_embedding(stride).to(P.device)

    if args.pretrained_weights:
        loaded = torch.load(args.pretrained_weights, map_location=P.device)
        model.load_state_dict(extract_model_state(loaded))
        if isinstance(loaded, dict) and "text_encoder_state_dict" in loaded:
            text_encoder.load_state_dict(loaded["text_encoder_state_dict"], strict=False)
        elif isinstance(loaded, dict) and "text_embedder_state_dict" in loaded:
            text_encoder.load_state_dict(loaded["text_embedder_state_dict"], strict=False)

    if P.text_encoder_type == "arabic_span":
        criterion = SpanContrastiveSoftDTW(
            gamma=P.contrastive_soft_dtw_gamma,
            margin=P.contrastive_margin,
            temperature=P.contrastive_temperature,
            max_windows_per_span=P.max_windows_per_span,
            negative_grad_mode=P.span_negative_grad_mode,
            backend=P.span_dtw_backend,
        )
    else:
        criterion = ContrastiveSoftDTW(
            gamma=P.contrastive_soft_dtw_gamma,
            use_cuda=torch.cuda.is_available(),
            margin=P.contrastive_margin,
            temperature=P.contrastive_temperature,
        )
    config = model_config(stride)
    print(
        f"job_id={args.job_id} device={P.device} epochs={args.epochs} lr={args.learning_rate} "
        f"window_size={P.window_size} stride={stride} negatives={P.num_negatives}",
        flush=True,
    )
    if P.text_encoder_type == "arabic_span":
        print(
            f"text_encoder=ArabicSpanTextEncoder model={P.arabic_text_model_name} "
            f"max_span_chars={P.max_text_span_chars} max_windows_per_span={P.max_windows_per_span} "
            f"negative_grad_mode={P.span_negative_grad_mode} span_dtw_backend={P.span_dtw_backend} "
            f"local_hard_negatives={P.use_local_hard_negatives} local_weight={P.local_hard_negative_weight} "
            f"local_margin={P.local_hard_negative_margin} local_top_k={P.local_hard_negative_top_k} "
            f"min_ink={P.local_hard_negative_min_ink} "
            f"image_pair={P.use_image_pair_contrastive} pair_weight={P.image_pair_loss_weight} "
            f"pair_margin={P.image_pair_margin} pair_top_k={P.image_pair_top_k} "
            f"order_weight={P.sequence_consistency_loss_weight} freeze_backbone=True",
            flush=True,
        )
    else:
        print(f"text_encoder={P.text_encoder_type}", flush=True)
    print(
        f"use_bilstm={P.use_bilstm} soft_dtw_gamma={P.contrastive_soft_dtw_gamma} "
        f"img_var_weight={P.image_variance_loss_weight} img_var_target={P.image_variance_target_std}",
        flush=True,
    )

    if os.environ.get("DEBUG_IMAGE_SHAPE", "0") == "1":
        batch = next(iter(train_loader))
        images = batch["images1"] if isinstance(batch, dict) else batch[0]
        images = images.to(P.device, non_blocking=True)
        with torch.no_grad():
            img_emb = model(images[:1], show_dims=True)
        print(f"DEBUG image embedding shape: {tuple(img_emb.shape)}", flush=True)

    train(model, text_encoder, criterion, train_loader, valid_loader, args, config)


if __name__ == "__main__":
    main()
