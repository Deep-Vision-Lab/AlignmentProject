#!/usr/bin/env python3
"""CPU smoke test for clean real + no-shared synthetic-partner training.

The test exercises the exact production loader.  It verifies that canonical real
rows are loaded without augmentation, generated partners come from the offline
training-only manifest, synthetic anchors are declared unmodified, 1--3 aligned
regions are present, tensors have the expected shape, and the loader's leakage
checks pass.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_DIR / "DataSet/ArabicDataset",
    )
    parser.add_argument(
        "--synthetic-manifest",
        type=Path,
        default=PROJECT_DIR / "DataSet/ArabicDatasetSyntheticPartners/dataset_manifest.jsonl",
    )
    parser.add_argument("--samples", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    manifest = root / os.environ.get("REAL_MANIFEST_NAME", "dataset_manifest.jsonl")
    synthetic_manifest = args.synthetic_manifest.expanduser().resolve()
    if not manifest.is_file():
        raise SystemExit(f"ERROR: missing canonical manifest: {manifest}")
    if not synthetic_manifest.is_file():
        raise SystemExit(
            "ERROR: missing synthetic-partner manifest: "
            f"{synthetic_manifest}\n"
            "Run scripts/data/build_no_shared_synthetic_partners.py first."
        )

    os.environ["REAL_MANIFEST_NAME"] = manifest.name
    os.environ["REAL_SYNTHETIC_PARTNER_MANIFEST"] = str(synthetic_manifest)
    os.environ["REAL_AUGMENT"] = "0"
    os.environ["AUGMENT"] = "0"
    os.environ["REAL_TRAIN_SAMPLES_PER_EPOCH"] = "0"
    os.environ.setdefault("DATASET_SPLIT_SEED", "42")
    os.environ.setdefault("REAL_EXTRA_EXCLUDE_EVAL_PAGES", "1")
    os.environ.setdefault("REAL_VALIDATE_PATHS", "0")

    from torch.utils.data import ConcatDataset

    from RealDataSet import ArabicManifestLinePairDataset
    from extra_real_training_partial_overlap import _build_partial_overlap_dataloaders

    train_loader, valid_loader, test_loader = _build_partial_overlap_dataloaders(root)
    train_dataset = train_loader.dataset
    if not isinstance(train_dataset, ConcatDataset) or len(train_dataset.datasets) != 3:
        raise RuntimeError(
            "Expected natural ConcatDataset(clean_positive, synthetic_partner, clean_no_shared)"
        )

    clean_positive, synthetic_partner, clean_no_shared = train_dataset.datasets
    if not isinstance(synthetic_partner, ArabicManifestLinePairDataset):
        raise RuntimeError("Synthetic-partner component is not a manifest dataset")

    counts = (len(clean_positive), len(synthetic_partner), len(clean_no_shared))
    if min(counts) <= 0:
        raise RuntimeError(f"Empty training component: counts={counts}")

    print("=== CLEAN SYNTHETIC-PARTNER PRODUCTION SMOKE TEST ===")
    print(f"canonical_manifest={manifest}")
    print(f"synthetic_manifest={synthetic_manifest}")
    print(
        "train_components="
        f"clean_positive:{counts[0]} synthetic_partner:{counts[1]} "
        f"clean_no_shared_negative:{counts[2]} natural_total:{sum(counts)}"
    )
    print("generic_real_augmentation=OFF")
    print(f"valid_rows={len(valid_loader.dataset)} test_rows={len(test_loader.dataset)}")

    histogram = {1: 0, 2: 0, 3: 0}
    inspect_count = min(max(1, int(args.samples)), len(synthetic_partner))
    for index in range(inspect_count):
        raw = synthetic_partner.samples[index]
        sample = synthetic_partner[index]
        if raw.get("sample_type") != "synthetic_partner_partial_overlap":
            raise RuntimeError(f"Bad synthetic sample_type at row {index}")
        metadata = raw.get("synthetic_partner") or {}
        if metadata.get("anchor_modified") is not False:
            raise RuntimeError(f"Anchor was not declared clean/unmodified at row {index}")
        regions = int(metadata.get("regions", 0))
        if regions not in {1, 2, 3}:
            raise RuntimeError(f"Bad aligned-region count at row {index}: {regions}")
        if len(metadata.get("region_details", []) or []) != regions:
            raise RuntimeError(f"Region metadata mismatch at row {index}")

        shape1 = tuple(sample["image1"].shape)
        shape2 = tuple(sample["image2"].shape)
        if shape1 != (3, 128, 1024) or shape2 != (3, 128, 1024):
            raise RuntimeError(f"Bad tensor shape at row {index}: {shape1} / {shape2}")
        histogram[regions] = histogram.get(regions, 0) + 1
        shared_texts = [
            detail.get("shared_text", "")
            for detail in metadata.get("region_details", []) or []
        ]
        print(
            f"sample={index:02d} source_pair={raw.get('source_pair_id')} "
            f"anchor_side={raw.get('anchor_source_side')} aligned_regions={regions} "
            f"shared_texts={shared_texts} shape={shape1}"
        )

    print(f"aligned_region_histogram={histogram}")
    print("ANCHOR_UNMODIFIED_CHECK=PASS")
    print("LEAKAGE_GUARD=PASS")
    print("SMOKE_TEST=PASS")


if __name__ == "__main__":
    main()
