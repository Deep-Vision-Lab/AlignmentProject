#!/usr/bin/env python3
"""Build train-only synthetic partners for clean no_shared_content real lines.

For every leakage-safe no-shared training row, one original side is kept completely
unchanged as an anchor. The opposite original side becomes the base distractor
canvas, and 1--3 bbox-exact complete-subword strips are replaced by different-
handwriting donor strips whose canonical text occurs in the anchor.

No canonical high/medium image is augmented. Validation and test are never used
as anchors, mates, or donors. Matching is by canonical Arabic text; anchor and
donor are allowed to use different bbox segmentation counts for the same text.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import random
import shutil
import sys

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.data.augment_real_bbox_strip_injection import (
    SourceLine,
    _load_line,
    _resolve,
    _save_side,
)
from SyntheticPartnerRealAugmentation import (
    PartnerSynthesisConfig,
    build_training_donor_index,
    donor_index_diagnostics,
    synthesize_partner,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_DIR / "DataSet/ArabicDataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "DataSet/ArabicDatasetSyntheticPartners",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--min-regions", type=int, default=1)
    parser.add_argument("--max-regions", type=int, default=3)
    parser.add_argument("--max-run-boxes", type=int, default=3)
    parser.add_argument("--min-chars", type=int, default=3)
    parser.add_argument("--max-chars", type=int, default=28)
    parser.add_argument("--width-ratio-min", type=float, default=0.40)
    parser.add_argument("--width-ratio-max", type=float, default=2.50)
    parser.add_argument("--multi-region-prob", type=float, default=0.65)
    parser.add_argument("--three-region-prob", type=float, default=0.15)
    parser.add_argument("--max-attempts", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _choose_region_count(rng: random.Random, args) -> int:
    if args.max_regions <= 1 or rng.random() > float(args.multi_region_prob):
        return 1
    if args.max_regions >= 3 and rng.random() < float(args.three_region_prob):
        return 3
    return 2


def _safe_extra_indices(legacy, positive_dataset, train_raw, valid_raw, test_raw, extra_dataset):
    train_pair_ids = legacy._sample_pair_ids(positive_dataset, train_raw)
    eval_page_ids = legacy._sample_page_ids(positive_dataset, (valid_raw, test_raw))
    strict_eval_page_exclusion = legacy._flag("REAL_EXTRA_EXCLUDE_EVAL_PAGES", True)
    indices = []
    excluded_pair = 0
    excluded_eval_page = 0
    for index, sample in enumerate(extra_dataset.samples):
        pair_id = str(sample.get("pair_id", index))
        if pair_id not in train_pair_ids:
            excluded_pair += 1
            continue
        if strict_eval_page_exclusion:
            pages = {
                str(value)
                for value in (sample.get("A_page_id"), sample.get("B_page_id"))
                if value is not None
            }
            if pages & eval_page_ids:
                excluded_eval_page += 1
                continue
        indices.append(index)
    return indices, excluded_pair, excluded_eval_page


def _load_unique_line(
    cache: dict[Path, SourceLine],
    rejected: Counter,
    dataset_root: Path,
    pair_id: str,
    side_name: str,
    side: dict,
    height: int,
):
    try:
        image_path = _resolve(dataset_root, side["line_image_path"])
        if image_path not in cache:
            cache[image_path] = _load_line(dataset_root, pair_id, side_name, side, height)
        return cache[image_path]
    except Exception as exc:
        rejected[f"{type(exc).__name__}: {exc}"] += 1
        return None


def _absolute_original_side(dataset_root: Path, side: dict) -> dict:
    output = dict(side)
    output["line_image_path"] = str(_resolve(dataset_root, side["line_image_path"]))
    output["text_original_path"] = str(_resolve(dataset_root, side["text_original_path"]))
    if side.get("text_tashkeel_path"):
        output["text_tashkeel_path"] = str(_resolve(dataset_root, side["text_tashkeel_path"]))
    return output


def main():
    args = parse_args()
    if not (1 <= int(args.min_regions) <= int(args.max_regions) <= 3):
        raise SystemExit("ERROR: require 1 <= min-regions <= max-regions <= 3")

    dataset_root = args.data_dir.expanduser().resolve()
    manifest = dataset_root / os.environ.get("REAL_MANIFEST_NAME", "dataset_manifest.jsonl")
    if not manifest.is_file():
        raise SystemExit(f"ERROR: missing manifest: {manifest}")

    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"ERROR: output is not empty: {output_root}; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    os.environ["REAL_MANIFEST_NAME"] = manifest.name
    os.environ["REAL_AUGMENT"] = "0"
    os.environ.setdefault("REAL_EXTRA_EXCLUDE_EVAL_PAGES", "1")

    import DataLoader as base_loader
    import extra_real_training as legacy
    from torch.utils.data import Subset

    positive_dataset = legacy._manifest_dataset(dataset_root, legacy.POSITIVE_LABELS)
    train_raw, valid_raw, test_raw = base_loader._group_split_real_dataset(positive_dataset)
    train_positive, train_stats = legacy._filter_feasible(
        positive_dataset, train_raw, "synthetic_partner_train_positive"
    )

    extra_dataset = legacy._manifest_dataset(dataset_root, (legacy.EXTRA_LABEL,))
    extra_indices, excluded_pair, excluded_eval_page = _safe_extra_indices(
        legacy, positive_dataset, train_raw, valid_raw, test_raw, extra_dataset
    )
    if not extra_indices:
        raise RuntimeError("No leakage-safe no_shared_content rows are available")
    extra_train, extra_stats = legacy._filter_feasible(
        extra_dataset,
        Subset(extra_dataset, extra_indices),
        "synthetic_partner_train_no_shared",
    )

    line_cache: dict[Path, SourceLine] = {}
    rejected: Counter[str] = Counter()
    for dataset, indices in (
        (positive_dataset, train_positive.indices),
        (extra_dataset, extra_train.indices),
    ):
        for index in indices:
            record = dataset.samples[int(index)]
            pair_id = str(record.get("pair_id", index))
            for side_name in ("A", "B"):
                _load_unique_line(
                    line_cache,
                    rejected,
                    dataset_root,
                    pair_id,
                    side_name,
                    record[side_name],
                    int(args.height),
                )

    config = PartnerSynthesisConfig(
        height=int(args.height),
        min_regions=int(args.min_regions),
        max_regions=int(args.max_regions),
        max_run_boxes=int(args.max_run_boxes),
        min_chars=int(args.min_chars),
        max_chars=int(args.max_chars),
        width_ratio_min=float(args.width_ratio_min),
        width_ratio_max=float(args.width_ratio_max),
        max_attempts=int(args.max_attempts),
    )
    donor_index = build_training_donor_index(line_cache.values(), config)
    donor_diag = donor_index_diagnostics(donor_index)
    preflight = {
        "clean_positive_train_rows": len(train_positive),
        "safe_no_shared_train_rows": len(extra_train),
        "bbox_valid_training_lines": len(line_cache),
        "bbox_rejected_line_loads": int(sum(rejected.values())),
        "donor_index": donor_diag,
        "min_chars": int(args.min_chars),
        "max_chars": int(args.max_chars),
        "max_run_boxes": int(args.max_run_boxes),
    }
    print("=== SYNTHETIC PARTNER DONOR PREFLIGHT ===")
    print(json.dumps(preflight, ensure_ascii=False, indent=2), flush=True)
    (output_root / "donor_preflight.json").write_text(
        json.dumps({**preflight, "bbox_rejections": dict(rejected)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if not donor_index or donor_diag["multi_image_text_keys"] <= 0:
        top_rejections = dict(rejected.most_common(10))
        raise RuntimeError(
            "No canonical Arabic text span is available in two distinct bbox-valid "
            "training images. See donor_preflight.json. Top bbox load rejections: "
            f"{top_rejections}"
        )

    rng = random.Random(int(args.seed))
    output_manifest = output_root / "dataset_manifest.jsonl"
    region_histogram: Counter[int] = Counter()
    orientation_histogram: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    generated = 0

    with output_manifest.open("w", encoding="utf-8") as handle:
        for ordinal, index in enumerate(extra_train.indices):
            record = extra_dataset.samples[int(index)]
            source_pair_id = str(record.get("pair_id", index))

            loaded = {}
            for side_name in ("A", "B"):
                image_path = _resolve(dataset_root, record[side_name]["line_image_path"])
                loaded[side_name] = line_cache.get(image_path)
            if loaded["A"] is None or loaded["B"] is None:
                failures["missing_bbox_side"] += 1
                continue

            preferred_anchor = "A" if ordinal % 2 == 0 else "B"
            orientations = [preferred_anchor, "B" if preferred_anchor == "A" else "A"]
            requested = _choose_region_count(rng, args)
            result = None
            chosen_anchor = None
            chosen_mate = None
            used_regions = None

            for anchor_side_name in orientations:
                mate_side = "B" if anchor_side_name == "A" else "A"
                for regions in range(requested, int(args.min_regions) - 1, -1):
                    result = synthesize_partner(
                        rng,
                        loaded[anchor_side_name],
                        loaded[mate_side],
                        donor_index,
                        regions,
                        config,
                    )
                    if result is not None:
                        chosen_anchor = anchor_side_name
                        chosen_mate = mate_side
                        used_regions = regions
                        break
                if result is not None:
                    break

            if result is None or chosen_anchor is None or chosen_mate is None:
                failures["no_feasible_synthetic_partner"] += 1
                continue

            partner_state, metadata = result
            output_index = generated + 1
            pair_root = output_root / "pairs" / f"partner_{output_index:06d}"
            saved_partner = _save_side(pair_root, "synthetic_partner", partner_state)
            clean_anchor_side = _absolute_original_side(dataset_root, record[chosen_anchor])

            synthetic_record = {
                "pair_id": f"{source_pair_id}__synthetic_partner_{output_index:06d}",
                "label_type": "medium_match",
                "sample_type": "synthetic_partner_partial_overlap",
                "source_label_type": "no_shared_content",
                "source_pair_id": source_pair_id,
                "anchor_source_side": chosen_anchor,
                "base_mate_source_side": chosen_mate,
                "synthetic_partner": metadata,
                "scores": {
                    "text_score": 0.5,
                    "avg_sim": 0.5,
                    "coverage_A": 0.5,
                    "coverage_B": 0.5,
                },
                "A": clean_anchor_side,
                "B": {
                    **saved_partner,
                    "line_idx": int(record[chosen_mate].get("line_idx", -1)),
                    "source_line_image_path": str(loaded[chosen_mate].image_path),
                    "source_bbox_json": loaded[chosen_mate].annotation_path,
                },
            }
            handle.write(json.dumps(synthetic_record, ensure_ascii=False) + "\n")
            generated += 1
            region_histogram[int(used_regions)] += 1
            orientation_histogram[chosen_anchor] += 1

    summary = {
        "source_manifest": str(manifest),
        "output_manifest": str(output_manifest),
        "clean_positive_train_rows": len(train_positive),
        "safe_no_shared_train_rows": len(extra_train),
        "generated_synthetic_partner_rows": generated,
        "generation_coverage": generated / max(1, len(extra_train)),
        "bbox_valid_training_lines": len(line_cache),
        "donor_index": donor_diag,
        "region_histogram": {str(k): v for k, v in sorted(region_histogram.items())},
        "anchor_orientation_histogram": dict(orientation_histogram),
        "excluded_nontrain_pair_rows": excluded_pair,
        "excluded_eval_page_rows": excluded_eval_page,
        "positive_feasibility_removed": train_stats.removed,
        "no_shared_feasibility_removed": extra_stats.removed,
        "bbox_rejections": dict(rejected),
        "generation_failures": dict(failures),
        "anchor_policy": "original canonical image/text, never modified",
        "partner_policy": "original unrelated mate with 1-3 bbox-exact matching donor strips",
        "donor_policy": "same canonical Arabic text, different training-only real image; bbox counts may differ",
        "online_augmentation": False,
    }
    (output_root / "synthetic_partner_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=== SYNTHETIC PARTNER BUILD SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if generated == 0:
        raise SystemExit(
            "ERROR: generated zero synthetic partners. Inspect generation_failures and donor_preflight.json"
        )


if __name__ == "__main__":
    main()
