"""Use no-shared-content rows as extra real image-text supervision safely.

The canonical real dataset contains high/medium-match positive A/B pairs and
``no_shared_content`` A/B pairs.  The latter must never be treated as positive
image-image alignments, but each individual side still has a genuine real line
image and its own transcript.  This module keeps the canonical positive
train/valid/test split, adds only training-side no-shared rows, and masks those
rows out of the image-image/order losses while retaining image-text losses on
both sides.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler, Subset

import DataLoader as base_loader
import AugmentedRealDataLoader as augmented_loader
from RealDataAugmentation import AugmentedRealSubset, RealLinePairAugmentor
from RealDataSet import ArabicManifestLinePairDataset
from real_span_feasibility import filter_subset_by_span_feasibility


POSITIVE_LABELS = {"high_match", "medium_match"}
EXTRA_LABEL = "no_shared_content"


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _manifest_dataset(data_dir, allowed_labels):
    return ArabicManifestLinePairDataset(
        manifest_path=base_loader._real_manifest_path(data_dir),
        transform=base_loader.real_transform,
        text_key=os.environ.get("REAL_TEXT_KEY", "text_original_path"),
        allowed_labels=tuple(allowed_labels),
        max_samples=None,
        paired=True,
        min_text_score=0.0,
        validate_paths=_flag("REAL_VALIDATE_PATHS", False),
    )


def _sample_pair_ids(dataset, subset) -> set[str]:
    return {
        str(dataset.samples[int(index)].get("pair_id", index))
        for index in subset.indices
    }


def _sample_page_ids(dataset, subsets: Iterable[Subset]) -> set[str]:
    page_ids: set[str] = set()
    for subset in subsets:
        for index in subset.indices:
            sample = dataset.samples[int(index)]
            for key in ("A_page_id", "B_page_id"):
                value = sample.get(key)
                if value is not None:
                    page_ids.add(str(value))
    return page_ids


def _line_paths(dataset, subset) -> set[str]:
    paths: set[str] = set()
    for index in subset.indices:
        sample = dataset.samples[int(index)]
        for side_name in ("A", "B"):
            value = (sample.get(side_name) or {}).get("line_image_path")
            if value:
                paths.add(str(value))
    return paths


def _filter_feasible(dataset, subset, split_name):
    return filter_subset_by_span_feasibility(
        dataset,
        subset,
        split_name=split_name,
        max_image_windows=int(os.environ.get("REAL_MAX_ALIGNMENT_WINDOWS", "125")),
        max_span_chars=int(os.environ.get("MAX_TEXT_SPAN_CHARS", "2")),
    )


def _collate_with_pair_mask(batch):
    output = base_loader.custom_collate_fn(batch)
    if not isinstance(output, dict):
        return output
    labels = [str(sample.get("label_type", "")) for sample in batch]
    output["pair_labels"] = labels
    output["pair_positive_mask"] = [label in POSITIVE_LABELS for label in labels]
    return output


def _make_loader(dataset, shuffle):
    kwargs = dict(
        batch_size=base_loader.batch_size,
        shuffle=shuffle,
        collate_fn=_collate_with_pair_mask,
        num_workers=base_loader._num_workers,
        pin_memory=base_loader._pin_memory,
        persistent_workers=base_loader._persistent_workers,
        drop_last=False,
    )
    if base_loader._prefetch_factor is not None:
        kwargs["prefetch_factor"] = base_loader._prefetch_factor
    return DataLoader(dataset, **kwargs)


def build_dataloaders(data_dir):
    """Build positive eval splits plus an expanded genuine-real train split."""
    positive_dataset = _manifest_dataset(data_dir, POSITIVE_LABELS)
    train_raw, valid_raw, test_raw = base_loader._group_split_real_dataset(
        positive_dataset
    )

    # Preserve the exact canonical Stage-2 group assignment before feasibility
    # filtering, so adding extra rows cannot move any positive sample between
    # train/validation/test.
    train_pair_ids = _sample_pair_ids(positive_dataset, train_raw)
    eval_page_ids = _sample_page_ids(positive_dataset, (valid_raw, test_raw))

    train_positive, train_stats = _filter_feasible(
        positive_dataset, train_raw, "train_positive"
    )
    valid_positive, valid_stats = _filter_feasible(
        positive_dataset, valid_raw, "valid"
    )
    test_positive, test_stats = _filter_feasible(
        positive_dataset, test_raw, "test"
    )

    extra_dataset = _manifest_dataset(data_dir, (EXTRA_LABEL,))
    strict_eval_page_exclusion = _flag("REAL_EXTRA_EXCLUDE_EVAL_PAGES", True)
    extra_indices = []
    excluded_pair = 0
    excluded_eval_page = 0
    for index, sample in enumerate(extra_dataset.samples):
        pair_id = str(sample.get("pair_id", index))
        if pair_id not in train_pair_ids:
            excluded_pair += 1
            continue
        if strict_eval_page_exclusion:
            sample_pages = {
                str(value)
                for value in (sample.get("A_page_id"), sample.get("B_page_id"))
                if value is not None
            }
            if sample_pages & eval_page_ids:
                excluded_eval_page += 1
                continue
        extra_indices.append(index)

    if not extra_indices:
        raise RuntimeError(
            "REAL_USE_EXTRA_NO_SHARED=1 found no safe no_shared_content rows "
            "for the canonical training split."
        )

    extra_raw = Subset(extra_dataset, extra_indices)
    extra_train, extra_stats = _filter_feasible(
        extra_dataset, extra_raw, "train_no_shared"
    )

    augmentor = RealLinePairAugmentor.from_env()
    if augmentor.enabled:
        # The main expanded-real recipe uses appearance perturbations only.
        # Stitching can be enabled explicitly, but it is intentionally disabled
        # by the public launcher because these rows are used for genuine line
        # coverage rather than content synthesis.
        train_transform = augmented_loader._train_real_transform()
        train_positive_dataset = AugmentedRealSubset(
            positive_dataset,
            train_positive.indices,
            train_transform,
            augmentor,
        )
        extra_train_dataset = AugmentedRealSubset(
            extra_dataset,
            extra_train.indices,
            train_transform,
            augmentor,
        )
    else:
        train_positive_dataset = train_positive
        extra_train_dataset = extra_train

    combined_train = ConcatDataset(
        [train_positive_dataset, extra_train_dataset]
    )
    natural_train_size = len(combined_train)
    target_train_size = int(os.environ.get("REAL_TRAIN_SAMPLES_PER_EPOCH", "0"))
    if target_train_size > natural_train_size:
        combined_train = augmented_loader.RepeatToLengthDataset(
            combined_train, target_train_size
        )

    positive_train_paths = _line_paths(positive_dataset, train_positive)
    extra_train_paths = _line_paths(extra_dataset, extra_train)
    genuinely_new_paths = extra_train_paths - positive_train_paths

    print(
        "Expanded genuine-real training dataset: "
        f"positive_train_rows={len(train_positive)} "
        f"extra_no_shared_rows={len(extra_train)} "
        f"natural_train_rows={natural_train_size} "
        f"train_per_epoch={len(combined_train)} "
        f"positive_unique_lines={len(positive_train_paths)} "
        f"extra_unique_lines={len(extra_train_paths)} "
        f"new_unique_lines={len(genuinely_new_paths)} "
        f"valid={len(valid_positive)} test={len(test_positive)} "
        f"exclude_eval_pages={strict_eval_page_exclusion} "
        f"excluded_nontrain_pair_rows={excluded_pair} "
        f"excluded_eval_page_rows={excluded_eval_page} "
        f"augment={augmentor.enabled} stitch_prob={augmentor.stitch_probability:.3f}",
        flush=True,
    )
    print(
        "Expanded-real feasibility: "
        f"positive_removed={train_stats.removed} "
        f"extra_removed={extra_stats.removed} "
        f"valid_removed={valid_stats.removed} "
        f"test_removed={test_stats.removed}",
        flush=True,
    )

    return (
        _make_loader(combined_train, shuffle=True),
        _make_loader(valid_positive, shuffle=False),
        _make_loader(test_positive, shuffle=False),
    )


def install(base) -> None:
    """Install expanded-real loading and pair-positive masking into train.py."""
    original_select_dataloaders = base.select_dataloaders
    original_pair_loss = base.compute_image_pair_loss

    def select_dataloaders(args):
        if args.dataset_type != "real" or not _flag(
            "REAL_USE_EXTRA_NO_SHARED", False
        ):
            return original_select_dataloaders(args)

        os.environ["DATASET_TYPE"] = "real"
        train_loader, valid_loader, test_loader = build_dataloaders(args.data_dir)
        if not base.CTX.enabled:
            return train_loader, valid_loader, test_loader, None

        split_seed = base._env_int("DATASET_SPLIT_SEED", 42)
        train_sampler = DistributedSampler(
            train_loader.dataset,
            num_replicas=base.CTX.world_size,
            rank=base.CTX.rank,
            shuffle=True,
            seed=split_seed,
            drop_last=False,
        )
        valid_sampler = base.DistributedEvalSampler(
            valid_loader.dataset,
            rank=base.CTX.rank,
            world_size=base.CTX.world_size,
        )
        test_sampler = base.DistributedEvalSampler(
            test_loader.dataset,
            rank=base.CTX.rank,
            world_size=base.CTX.world_size,
        )
        return (
            base._rebuild_loader(train_loader, train_sampler),
            base._rebuild_loader(valid_loader, valid_sampler),
            base._rebuild_loader(test_loader, test_sampler),
            train_sampler,
        )

    def masked_pair_loss(text_encoder, criterion, texts1, texts2, emb1, emb2, mask):
        if mask is None:
            return original_pair_loss(
                text_encoder, criterion, texts1, texts2, emb1, emb2
            )
        keep = [index for index, enabled in enumerate(mask) if bool(enabled)]
        if not keep:
            zero = emb1[0].new_tensor(0.0)
            return zero, zero, {
                "image_pair_loss": 0.0,
                "order_loss": 0.0,
                "pair_terms": 0.0,
                "pair_positive_samples": 0.0,
            }
        index_tensor = torch.as_tensor(
            keep, dtype=torch.long, device=emb1[0].device
        )

        def select_embeddings(values):
            return tuple(
                value.index_select(0, index_tensor)
                if torch.is_tensor(value) and value.ndim > 0
                else value
                for value in values
            )

        result = original_pair_loss(
            text_encoder,
            criterion,
            [texts1[index] for index in keep],
            [texts2[index] for index in keep],
            select_embeddings(emb1),
            select_embeddings(emb2),
        )
        result[2]["pair_positive_samples"] = float(len(keep))
        return result

    def compute_batch_loss(image_embedder, text_encoder, criterion, batch):
        if not isinstance(batch, dict):
            return base._ORIGINAL_COMPUTE_BATCH_LOSS(
                image_embedder, text_encoder, criterion, batch
            )

        if torch.is_grad_enabled():
            base._BATCH_COUNTER += 1
        local_every = max(
            1,
            base._env_int(
                "LOCAL_HARD_NEGATIVE_EVERY_N_BATCHES",
                getattr(base.P, "local_hard_negative_every_n_batches", 1),
            ),
        )
        local_enabled = torch.is_grad_enabled() and (
            base._BATCH_COUNTER % local_every == 0
        )

        images1 = batch["images1"].to(base.P.device, non_blocking=True)
        images2 = batch["images2"].to(base.P.device, non_blocking=True)
        texts1 = batch["texts1"]
        texts2 = batch["texts2"]
        neg_texts1 = batch["neg_texts1"]
        neg_texts2 = batch["neg_texts2"]
        pair_mask = batch.get("pair_positive_mask")

        if base.P.text_encoder_type == "arabic_span":
            emb1 = base.compute_embeddings(image_embedder, images1)
            emb2 = base.compute_embeddings(image_embedder, images2)
        else:
            emb1 = None
            emb2 = None

        loss1, stats1, emb1 = base.compute_single_image_text_loss(
            image_embedder,
            text_encoder,
            criterion,
            images1,
            texts1,
            neg_texts1,
            emb1,
            local_enabled=local_enabled,
        )
        if bool(getattr(base.P, "image_text_loss_on_both_lines", True)):
            loss2, stats2, emb2 = base.compute_single_image_text_loss(
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
            stats = base.average_stats([stats1, stats2])
        else:
            loss = loss1
            stats = dict(stats1)

        pair_every = max(
            1,
            base._env_int(
                "IMAGE_PAIR_EVERY_N_BATCHES",
                getattr(base.P, "image_pair_every_n_batches", 1),
            ),
        )
        pair_enabled = torch.is_grad_enabled() and (
            base._BATCH_COUNTER % pair_every == 0
        )
        if pair_enabled and emb1 is not None and emb2 is not None:
            pair_loss, order_loss, pair_stats = masked_pair_loss(
                text_encoder,
                criterion,
                texts1,
                texts2,
                emb1,
                emb2,
                pair_mask,
            )
            loss = loss + base.P.image_pair_loss_weight * pair_loss
            if base.P.sequence_consistency_loss_weight > 0:
                loss = loss + base.P.sequence_consistency_loss_weight * order_loss
            stats.update(pair_stats)
        else:
            stats.update(
                {
                    "image_pair_loss": 0.0,
                    "order_loss": 0.0,
                    "pair_terms": 0.0,
                    "pair_positive_samples": 0.0,
                }
            )
        stats["total"] = float(loss.detach().item())
        return loss, stats

    # Keep a stable handle for the non-dict fallback in the patched function.
    if not hasattr(base, "_ORIGINAL_COMPUTE_BATCH_LOSS"):
        base._ORIGINAL_COMPUTE_BATCH_LOSS = base.compute_batch_loss
    base.select_dataloaders = select_dataloaders
    base.compute_batch_loss = compute_batch_loss
