"""Clean real training plus one-sided synthetic partners for no-shared lines.

This is the corrected Stage-2-style recipe:

* canonical high/medium real pairs are used exactly as stored, with no augmentation;
* canonical no_shared_content A/B pairs are also kept exactly as stored and remain
  true negative pairs for direct sequence ranking;
* each safe no-shared training row additionally contributes one generated positive
  pair in which one original side is an untouched anchor and the opposite side is
  a synthetic partner containing 1--3 bbox-exact aligned islands;
* validation and test remain untouched canonical positive real pairs.

The synthetic-partner manifest is built offline by
``scripts/data/build_no_shared_synthetic_partners.py`` from training-only lines.
"""
from __future__ import annotations

import os
from pathlib import Path

from torch.utils.data import ConcatDataset, Subset

import DataLoader as base_loader
import extra_real_training as legacy
from RealDataSet import ArabicManifestLinePairDataset


def _synthetic_manifest_path(data_dir) -> Path:
    configured = os.environ.get("REAL_SYNTHETIC_PARTNER_MANIFEST", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (
        Path(data_dir).expanduser().resolve().parent
        / "ArabicDatasetSyntheticPartners"
        / "dataset_manifest.jsonl"
    )


def _resolved_paths(dataset, indices) -> set[str]:
    paths: set[str] = set()
    for index in indices:
        sample = dataset.samples[int(index)]
        for side_name in ("A", "B"):
            value = (sample.get(side_name) or {}).get("line_image_path")
            if value:
                paths.add(str(dataset._resolve(value)))
    return paths


def _build_partial_overlap_dataloaders(data_dir):
    positive_dataset = legacy._manifest_dataset(data_dir, legacy.POSITIVE_LABELS)
    train_raw, valid_raw, test_raw = base_loader._group_split_real_dataset(
        positive_dataset
    )

    train_pair_ids = legacy._sample_pair_ids(positive_dataset, train_raw)
    eval_page_ids = legacy._sample_page_ids(positive_dataset, (valid_raw, test_raw))

    train_positive, train_stats = legacy._filter_feasible(
        positive_dataset, train_raw, "train_positive"
    )
    valid_positive, valid_stats = legacy._filter_feasible(
        positive_dataset, valid_raw, "valid"
    )
    test_positive, test_stats = legacy._filter_feasible(
        positive_dataset, test_raw, "test"
    )

    extra_dataset = legacy._manifest_dataset(data_dir, (legacy.EXTRA_LABEL,))
    strict_eval_page_exclusion = legacy._flag("REAL_EXTRA_EXCLUDE_EVAL_PAGES", True)
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
            "Synthetic-partner training found no leakage-safe no_shared_content rows"
        )

    extra_train, extra_stats = legacy._filter_feasible(
        extra_dataset,
        Subset(extra_dataset, extra_indices),
        "train_no_shared",
    )
    if not train_positive.indices or not extra_train.indices:
        raise RuntimeError(
            "Synthetic-partner training requires both clean positive and no-shared rows"
        )

    synthetic_manifest = _synthetic_manifest_path(data_dir)
    if not synthetic_manifest.is_file():
        raise RuntimeError(
            "Synthetic-partner manifest is missing: "
            f"{synthetic_manifest}. Build it first with "
            "scripts/data/build_no_shared_synthetic_partners.py"
        )
    synthetic_dataset = ArabicManifestLinePairDataset(
        manifest_path=synthetic_manifest,
        transform=base_loader.real_transform,
        text_key=os.environ.get("REAL_TEXT_KEY", "text_original_path"),
        allowed_labels=("medium_match",),
        max_samples=None,
        paired=True,
        min_text_score=0.0,
        validate_paths=legacy._flag("REAL_VALIDATE_PATHS", False),
    )

    safe_source_pair_ids = {
        str(extra_dataset.samples[int(index)].get("pair_id", index))
        for index in extra_train.indices
    }
    invalid_source_ids = []
    malformed = []
    for sample in synthetic_dataset.samples:
        if sample.get("sample_type") != "synthetic_partner_partial_overlap":
            malformed.append(str(sample.get("pair_id", "<missing>")))
        source_pair_id = str(sample.get("source_pair_id", ""))
        if source_pair_id not in safe_source_pair_ids:
            invalid_source_ids.append(source_pair_id or "<missing>")
    if malformed:
        raise RuntimeError(
            "Synthetic-partner manifest contains unexpected sample types: "
            f"{malformed[:5]}"
        )
    if invalid_source_ids:
        raise RuntimeError(
            "Synthetic-partner manifest contains non-training source pair IDs: "
            f"{invalid_source_ids[:5]}"
        )

    # Path-level leakage checks cover canonical anchors/mates and all donor images
    # recorded in the generated partner metadata.
    eval_paths = _resolved_paths(
        positive_dataset,
        list(valid_positive.indices) + list(test_positive.indices),
    )
    train_positive_paths = _resolved_paths(positive_dataset, train_positive.indices)
    no_shared_paths = _resolved_paths(extra_dataset, extra_train.indices)
    if (train_positive_paths | no_shared_paths) & eval_paths:
        raise RuntimeError("Canonical training/evaluation path leakage detected")

    synthetic_source_paths: set[str] = set()
    for sample in synthetic_dataset.samples:
        for side_name in ("A", "B"):
            value = (sample.get(side_name) or {}).get("line_image_path")
            if value:
                synthetic_source_paths.add(str(synthetic_dataset._resolve(value)))
        partner_meta = sample.get("synthetic_partner") or {}
        base_mate = partner_meta.get("base_unrelated_mate")
        anchor_image = partner_meta.get("anchor_image")
        if base_mate:
            synthetic_source_paths.add(str(Path(base_mate).expanduser().resolve()))
        if anchor_image:
            synthetic_source_paths.add(str(Path(anchor_image).expanduser().resolve()))
        for detail in partner_meta.get("region_details", []) or []:
            donor = detail.get("donor_image")
            if donor:
                synthetic_source_paths.add(str(Path(donor).expanduser().resolve()))
    leaked_synthetic = synthetic_source_paths & eval_paths
    if leaked_synthetic:
        raise RuntimeError(
            "Synthetic-partner donor/anchor leakage detected against validation/test: "
            f"{sorted(leaked_synthetic)[:5]}"
        )

    # Natural clean-data composition is the default.  Optional oversampling only
    # repeats this complete mixture; it never switches to an augmented dataset.
    combined_train = ConcatDataset(
        [train_positive, synthetic_dataset, extra_train]
    )
    natural_total = len(combined_train)
    requested_total = int(os.environ.get("REAL_TRAIN_SAMPLES_PER_EPOCH", "0"))
    if requested_total > natural_total:
        combined_train = base_loader.RepeatToLengthDataset(
            combined_train, requested_total
        ) if hasattr(base_loader, "RepeatToLengthDataset") else combined_train
        if len(combined_train) == natural_total:
            import AugmentedRealDataLoader as augmented_loader
            combined_train = augmented_loader.RepeatToLengthDataset(
                combined_train, requested_total
            )
    elif 0 < requested_total < natural_total:
        raise RuntimeError(
            "REAL_TRAIN_SAMPLES_PER_EPOCH smaller than the natural clean mixture "
            f"({requested_total} < {natural_total}) would drop real samples. Use 0 or >= {natural_total}."
        )

    component_total = len(train_positive) + len(synthetic_dataset) + len(extra_train)
    print(
        "Clean + synthetic-partner training mixture: "
        f"clean_positive={len(train_positive)} ({len(train_positive) / component_total:.1%}) "
        f"synthetic_partner_positive={len(synthetic_dataset)} ({len(synthetic_dataset) / component_total:.1%}) "
        f"clean_no_shared_negative={len(extra_train)} ({len(extra_train) / component_total:.1%}) "
        f"natural_total={component_total} train_per_epoch={len(combined_train)} "
        f"valid={len(valid_positive)} test={len(test_positive)} "
        f"augment=0 synthetic_manifest={synthetic_manifest} "
        f"exclude_eval_pages={strict_eval_page_exclusion} "
        f"excluded_nontrain_pair_rows={excluded_pair} "
        f"excluded_eval_page_rows={excluded_eval_page}",
        flush=True,
    )
    print(
        "Clean synthetic-partner feasibility: "
        f"positive_removed={train_stats.removed} "
        f"extra_removed={extra_stats.removed} "
        f"valid_removed={valid_stats.removed} "
        f"test_removed={test_stats.removed}",
        flush=True,
    )

    return (
        legacy._make_loader(combined_train, shuffle=True),
        legacy._make_loader(valid_positive, shuffle=False),
        legacy._make_loader(test_positive, shuffle=False),
    )


def install(base) -> None:
    # Keep the latest direct sequence-ranking objective.  Clean no-shared A/B
    # pairs remain its negatives, while generated anchor/partner pairs become
    # partial-overlap positives.
    legacy.build_dataloaders = _build_partial_overlap_dataloaders

    from extra_real_training_v4 import install as install_sequence_ranking

    install_sequence_ranking(base)
