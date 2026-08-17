"""Balanced training on an offline real-conditioned synthetic bridge manifest.

The manifest contains one positive synthetic span and several guaranteed-negative
synthetic spans for every real anchor. This loader keeps complete anchor ``pair_id``
groups together, uses only positive rows for internal validation/test, and balances
positive vs no-shared rows 50/50 during training regardless of how many negatives
were generated per anchor.

Bridge v1 uses four complementary signals:
- real A <-> its genuine transcript (the established image-text objective);
- synthetic B <-> its exactly known transcript (the same image-text objective);
- positive/negative real A <-> synthetic B image sequence discrimination; and
- DIRECT real-image <-> synthetic-text sequence ranking: a positive synthetic text
  must form a stronger/longer local image-text path in the real anchor than a
  guaranteed-negative synthetic text.

The direct term intentionally operates on *sequence alignment*, rather than pushing
every individual real window away from every character in a negative line. Arabic
negative lines may legitimately contain isolated repeated characters; what is
forbidden by the offline builder is a meaningful shared multi-character sequence.
"""
from __future__ import annotations

import math
import os
import shutil

import torch
from torch.utils.data import Dataset, Subset

from bridge_group_split import group_split
import extra_real_training as legacy
import extra_real_training_v4 as sequence


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


class BalancedOfflineBridgeMix(Dataset):
    """Alternate positive and negative slots, then let the DataLoader shuffle them."""

    def __init__(self, positive: Dataset, negative: Dataset, target_length: int):
        if len(positive) <= 0 or len(negative) <= 0:
            raise ValueError("Bridge training requires both positive and negative rows.")
        self.positive = positive
        self.negative = negative
        self.target_length = max(2, int(target_length))
        if self.target_length % 2:
            self.target_length += 1

    def __len__(self):
        return self.target_length

    def __getitem__(self, index):
        index = int(index)
        occurrence = index // 2
        if index % 2 == 0:
            return self.positive[occurrence % len(self.positive)]
        return self.negative[occurrence % len(self.negative)]


def _pair_ids(dataset, subset: Subset) -> set[str]:
    return {
        str(dataset.samples[int(index)].get("pair_id", index))
        for index in subset.indices
    }


def build_bridge_dataloaders(data_dir):
    positive_dataset = legacy._manifest_dataset(data_dir, legacy.POSITIVE_LABELS)
    train_raw, valid_raw, test_raw = group_split(positive_dataset)

    train_positive, train_stats = legacy._filter_feasible(
        positive_dataset, train_raw, "bridge_train_positive"
    )
    valid_positive, valid_stats = legacy._filter_feasible(
        positive_dataset, valid_raw, "bridge_valid_positive"
    )
    test_positive, test_stats = legacy._filter_feasible(
        positive_dataset, test_raw, "bridge_test_positive"
    )

    train_pair_ids = _pair_ids(positive_dataset, train_raw)
    negative_dataset = legacy._manifest_dataset(data_dir, (legacy.EXTRA_LABEL,))
    negative_indices = [
        index
        for index, sample in enumerate(negative_dataset.samples)
        if str(sample.get("pair_id", index)) in train_pair_ids
    ]
    if not negative_indices:
        raise RuntimeError(
            "Offline bridge manifest has no no_shared_content rows belonging to "
            "the internal training anchors."
        )
    negative_raw = Subset(negative_dataset, negative_indices)
    train_negative, negative_stats = legacy._filter_feasible(
        negative_dataset, negative_raw, "bridge_train_negative"
    )

    # Expose every generated negative once per epoch by default and repeat
    # positives as needed to preserve exact 50/50 class balance.
    natural_target = 2 * len(train_negative)
    requested_target = _env_int("BRIDGE_TRAIN_SAMPLES_PER_EPOCH", 0)
    target_length = requested_target if requested_target > 0 else natural_target
    train_dataset = BalancedOfflineBridgeMix(
        train_positive,
        train_negative,
        target_length=target_length,
    )

    print(
        "Offline real-synthetic bridge dataset: "
        f"source={data_dir} "
        f"positive_train={len(train_positive)} "
        f"negative_train={len(train_negative)} "
        f"train_per_epoch={len(train_dataset)} "
        f"positive_ratio=0.500 negative_ratio=0.500 "
        f"valid_positive={len(valid_positive)} test_positive={len(test_positive)} "
        "online_rendering=False online_augmentation=False",
        flush=True,
    )
    print(
        "Bridge feasibility: "
        f"positive_removed={train_stats.removed} "
        f"negative_removed={negative_stats.removed} "
        f"valid_removed={valid_stats.removed} test_removed={test_stats.removed}",
        flush=True,
    )

    return (
        legacy._make_loader(train_dataset, shuffle=True),
        legacy._make_loader(valid_positive, shuffle=False),
        legacy._make_loader(test_positive, shuffle=False),
    )


def _direct_cross_text_loss(base, text_encoder, texts2, emb1, labels):
    """Rank positive synthetic text above guaranteed-negative text for REAL image A.

    ``emb1`` contains normalized local embeddings of the real anchor. ``texts2`` is
    the transcript of the rendered synthetic side. The frozen/detached text vectors
    therefore act as a fixed target space; gradients from this bridge term flow into
    the visual representation, not into the text target.
    """
    reference = emb1[0]
    zero = reference.new_tensor(0.0)
    empty_stats = {
        "bridge_cross_text_loss": 0.0,
        "bridge_text_path_rank_loss": 0.0,
        "bridge_text_score_rank_loss": 0.0,
        "bridge_text_positive_floor_loss": 0.0,
        "bridge_text_negative_ceiling_loss": 0.0,
        "bridge_text_pos_fraction": 0.0,
        "bridge_text_neg_fraction": 0.0,
        "bridge_text_fraction_gap": 0.0,
        "bridge_text_pos_score": 0.0,
        "bridge_text_neg_score": 0.0,
        "bridge_text_active": 0.0,
    }
    if not torch.is_grad_enabled() or _env_float("BRIDGE_CROSS_TEXT_WEIGHT", 0.10) <= 0:
        return zero, empty_stats

    positive_indices = [
        i for i, label in enumerate(labels) if str(label) in legacy.POSITIVE_LABELS
    ]
    negative_indices = [
        i for i, label in enumerate(labels) if str(label) == legacy.EXTRA_LABEL
    ]
    max_pos = _env_int("BRIDGE_CROSS_TEXT_MAX_POS_PER_BATCH", 4)
    max_neg = _env_int("BRIDGE_CROSS_TEXT_MAX_NEG_PER_BATCH", 4)
    if max_pos > 0:
        positive_indices = positive_indices[:max_pos]
    if max_neg > 0:
        negative_indices = negative_indices[:max_neg]
    if not positive_indices or not negative_indices:
        return zero, empty_stats

    _context1, norm_local1, _ink1, _raw1 = emb1

    def metrics(index: int):
        # Detached text embeddings guarantee that this bridge-specific objective
        # cannot move the semantic target space even if the encoder has a trainable
        # projection layer elsewhere in the project.
        text_embedding = base.embed_single_text(text_encoder, texts2[index]).detach()
        if text_embedding.ndim == 1:
            text_embedding = text_embedding.unsqueeze(0)
        similarity = torch.matmul(norm_local1[index], text_embedding.T)
        return sequence.soft_local_alignment_metrics(
            similarity,
            _env_float("BRIDGE_CROSS_TEXT_THRESHOLD", 0.50),
            _env_float("BRIDGE_CROSS_TEXT_GAP", -0.30),
            _env_float("BRIDGE_CROSS_TEXT_TEMPERATURE", 0.03),
        )

    positive = [metrics(index) for index in positive_indices]
    negative = [metrics(index) for index in negative_indices]
    pos_scores = torch.stack([item[0] for item in positive])
    pos_fractions = torch.stack([item[1] for item in positive])
    neg_scores = torch.stack([item[0] for item in negative])
    neg_fractions = torch.stack([item[1] for item in negative])

    # Pairwise ranking attacks AUROC/distribution overlap. The additional absolute
    # floor/ceiling prevents a trivial solution where both positive and negative
    # paths move together while preserving only a small relative margin.
    path_rank_loss = torch.relu(
        _env_float("BRIDGE_CROSS_TEXT_PATH_MARGIN", 0.10)
        - pos_fractions[:, None]
        + neg_fractions[None, :]
    ).mean()
    score_rank_loss = torch.relu(
        _env_float("BRIDGE_CROSS_TEXT_SCORE_MARGIN", 0.10)
        - pos_scores[:, None]
        + neg_scores[None, :]
    ).mean()
    positive_floor_loss = torch.relu(
        _env_float("BRIDGE_CROSS_TEXT_POSITIVE_FLOOR", 0.20) - pos_fractions
    ).mean()
    negative_ceiling_loss = torch.relu(
        neg_fractions - _env_float("BRIDGE_CROSS_TEXT_NEGATIVE_CEILING", 0.15)
    ).mean()

    loss = (
        path_rank_loss
        + _env_float("BRIDGE_CROSS_TEXT_SCORE_COMPONENT_WEIGHT", 0.20)
        * score_rank_loss
        + _env_float("BRIDGE_CROSS_TEXT_POSITIVE_FLOOR_WEIGHT", 0.50)
        * positive_floor_loss
        + _env_float("BRIDGE_CROSS_TEXT_NEGATIVE_CEILING_WEIGHT", 0.75)
        * negative_ceiling_loss
    )

    stats = {
        "bridge_cross_text_loss": float(loss.detach().item()),
        "bridge_text_path_rank_loss": float(path_rank_loss.detach().item()),
        "bridge_text_score_rank_loss": float(score_rank_loss.detach().item()),
        "bridge_text_positive_floor_loss": float(positive_floor_loss.detach().item()),
        "bridge_text_negative_ceiling_loss": float(negative_ceiling_loss.detach().item()),
        "bridge_text_pos_fraction": float(pos_fractions.mean().detach().item()),
        "bridge_text_neg_fraction": float(neg_fractions.mean().detach().item()),
        "bridge_text_fraction_gap": float(
            (pos_fractions.mean() - neg_fractions.mean()).detach().item()
        ),
        "bridge_text_pos_score": float(pos_scores.mean().detach().item()),
        "bridge_text_neg_score": float(neg_scores.mean().detach().item()),
        "bridge_text_active": 1.0,
    }
    return loss, stats


def _install_direct_cross_text(base) -> None:
    """Attach the direct bridge term at the existing pair-loss hook point."""
    original_pair_loss = sequence.shared._positive_pair_loss

    def bridge_pair_loss(
        base_arg,
        text_encoder,
        criterion,
        texts1,
        texts2,
        emb1,
        emb2,
        labels,
    ):
        pair_loss, order_loss, stats = original_pair_loss(
            base_arg,
            text_encoder,
            criterion,
            texts1,
            texts2,
            emb1,
            emb2,
            labels,
        )
        direct_loss, direct_stats = _direct_cross_text_loss(
            base_arg, text_encoder, texts2, emb1, labels
        )
        requested_weight = _env_float("BRIDGE_CROSS_TEXT_WEIGHT", 0.10)
        outer_weight = float(getattr(base_arg.P, "image_pair_loss_weight", 0.0))
        if requested_weight > 0:
            if outer_weight <= 0:
                raise RuntimeError(
                    "BRIDGE_CROSS_TEXT_WEIGHT requires IMAGE_PAIR_LOSS_WEIGHT > 0 "
                    "because the bridge hook shares the existing pair-loss call site."
                )
            # extra_real_training_v4 multiplies the returned pair loss by the
            # image-pair weight. Rescale here so BRIDGE_CROSS_TEXT_WEIGHT is the
            # actual requested coefficient in the final total loss.
            pair_loss = pair_loss + (requested_weight / outer_weight) * direct_loss
        stats.update(direct_stats)
        return pair_loss, order_loss, stats

    sequence.shared._positive_pair_loss = bridge_pair_loss


def _install_best_validation_checkpoint(base) -> None:
    """Keep the best validated epoch so a longer max run cannot erase it."""
    state = {"last_val": float("nan"), "best_val": float("inf"), "best_epoch": 0}
    previous_log = base.wandb_log_epoch_metrics
    previous_save = base.save_checkpoint

    def log_epoch(run, epoch, train_loss, val_loss, train_stats):
        state["last_val"] = float(val_loss)
        state["epoch"] = int(epoch)
        return previous_log(run, epoch, train_loss, val_loss, train_stats)

    def save_checkpoint(
        model,
        text_encoder,
        optimizer,
        scheduler,
        scaler,
        epoch,
        job_id,
        config,
    ):
        result = previous_save(
            model,
            text_encoder,
            optimizer,
            scheduler,
            scaler,
            epoch,
            job_id,
            config,
        )
        val_loss = float(state.get("last_val", float("nan")))
        if math.isfinite(val_loss) and val_loss < float(state["best_val"]):
            state["best_val"] = val_loss
            state["best_epoch"] = int(epoch) + 1
            source = os.path.join(base.weights_dir(job_id), "checkpoint_latest.pth")
            target = os.path.join(base.weights_dir(job_id), "checkpoint_best_val.pth")
            shutil.copy2(source, target)
            print(
                "bridge_best_checkpoint "
                f"epoch={state['best_epoch']} val_loss={val_loss:.6f} path={target}",
                flush=True,
            )
        return result

    base.wandb_log_epoch_metrics = log_epoch
    base.save_checkpoint = save_checkpoint


def install(base) -> None:
    # v4 calls legacy.install(), and legacy.install() consults this module-global
    # builder at runtime. Replace it first so no online rendering/augmentation runs.
    legacy.build_dataloaders = build_bridge_dataloaders
    sequence.install(base)
    _install_direct_cross_text(base)
    _install_best_validation_checkpoint(base)

    previous_model_config = base.model_config

    def model_config(stride, args):
        config = dict(previous_model_config(stride, args))
        config.update(
            {
                "real_synthetic_bridge": True,
                "bridge_offline_rendering": True,
                "bridge_class_balance_positive": 0.5,
                "bridge_class_balance_negative": 0.5,
                "bridge_direct_cross_text_loss": True,
                "bridge_cross_text_weight": _env_float(
                    "BRIDGE_CROSS_TEXT_WEIGHT", 0.10
                ),
                "bridge_cross_text_threshold": _env_float(
                    "BRIDGE_CROSS_TEXT_THRESHOLD", 0.50
                ),
                "bridge_cross_text_path_margin": _env_float(
                    "BRIDGE_CROSS_TEXT_PATH_MARGIN", 0.10
                ),
                "bridge_cross_text_positive_floor": _env_float(
                    "BRIDGE_CROSS_TEXT_POSITIVE_FLOOR", 0.20
                ),
                "bridge_cross_text_negative_ceiling": _env_float(
                    "BRIDGE_CROSS_TEXT_NEGATIVE_CEILING", 0.15
                ),
                "bridge_best_validation_checkpoint": True,
            }
        )
        return config

    base.model_config = model_config
    if getattr(base.CTX, "is_main", True):
        print(
            "Real-conditioned synthetic bridge installed: offline 50/50 positive/"
            "negative pairs + image-text supervision + image-sequence ranking + "
            "DIRECT real-image/synthetic-text sequence ranking; text targets are "
            "detached; best validation checkpoint is preserved.",
            flush=True,
        )
