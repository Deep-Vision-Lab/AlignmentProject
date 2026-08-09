#!/usr/bin/env python3
"""Build a leakage-safe real Arabic training dataset augmented to a target size.

The builder keeps the original positive real pairs, reproduces the current
pair_id-safe 60/20/20 real-data split with seed 42 by default, and augments ONLY
the training split with bbox-exact full-height aligned strip injections.

Outputs
-------
train_manifest.jsonl
    Original real train pairs + generated aligned bbox-strip pairs. Exactly
    --target-train-pairs rows unless the original train split is already larger.
valid_manifest.jsonl
    Untouched original validation pairs.
test_manifest.jsonl
    Untouched original test pairs.
dataset_manifest.jsonl
    Concatenation of the three manifests with an explicit ``split`` field.
no_shared_content_manifest.jsonl
    Untouched negative real pairs, kept separate so they are never loaded as
    positive training pairs accidentally.
dataset_summary.json
    Counts, split seed, paths, provenance, and augmentation configuration.

Original files are referenced by absolute path rather than copied. Generated
samples live under ``augmented/`` inside the output dataset directory.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import copy
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
from typing import Iterable


POSITIVE_LABELS = {"high_match", "medium_match"}
NEGATIVE_LABEL = "no_shared_content"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _resolve_source_path(dataset_root: Path, value) -> str:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    candidates = (
        dataset_root / path,
        dataset_root.parent / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    # Keep a deterministic absolute path even when a non-essential optional
    # path is currently missing; mandatory image/text paths are validated later.
    return str((dataset_root / path).resolve())


def _absolutize_original_record(dataset_root: Path, row: dict, split: str) -> dict:
    record = copy.deepcopy(row)
    record["split"] = split
    record["sample_origin"] = "original_real"
    record["source_dataset_root"] = str(dataset_root)
    for side_name in ("A", "B"):
        side = record.get(side_name)
        if not isinstance(side, dict):
            raise ValueError(f"record {record.get('pair_id')} is missing side {side_name}")
        for key in (
            "line_image_path",
            "text_original_path",
            "text_tashkeel_path",
            "text_raw_path",
            "bbox_path",
        ):
            if side.get(key):
                side[key] = _resolve_source_path(dataset_root, side[key])
        for required in ("line_image_path", "text_original_path"):
            value = side.get(required)
            if not value or not Path(value).is_file():
                raise FileNotFoundError(
                    f"original record {record.get('pair_id')} side {side_name} "
                    f"has unresolved {required}: {value}"
                )
    return record


def _random_split(rows: list[dict], seed: int):
    indices = list(range(len(rows)))
    random.Random(int(seed)).shuffle(indices)
    train_size = int(0.6 * len(indices))
    valid_size = int(0.2 * len(indices))
    train = [rows[i] for i in indices[:train_size]]
    valid = [rows[i] for i in indices[train_size : train_size + valid_size]]
    test = [rows[i] for i in indices[train_size + valid_size :]]
    return train, valid, test


def _diverse_group_split(rows: list[dict], seed: int):
    """Mirror Evaluation/eval_img_align_sw.py's current real split policy."""
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for position, row in enumerate(rows):
        group_id = str(row.get("pair_id") or f"sample_{position}")
        groups.setdefault(group_id, []).append(row)
    if len(groups) < 3:
        return _random_split(rows, seed)

    rng = random.Random(int(seed))
    items = list(groups.items())
    rng.shuffle(items)
    # Stable sort means the prior shuffle breaks ties among equally-sized groups,
    # exactly like the evaluator.
    items.sort(key=lambda item: len(item[1]))

    assigned = {"train": [], "valid": [], "test": []}
    minimum_eval_groups = 2 if len(items) >= 6 else 1

    for _ in range(minimum_eval_groups):
        _group_id, members = items.pop(0)
        assigned["test"].extend(members)
    for _ in range(minimum_eval_groups):
        _group_id, members = items.pop(0)
        assigned["valid"].extend(members)
    if items:
        _group_id, members = items.pop()
        assigned["train"].extend(members)

    targets = {
        "train": 0.60 * len(rows),
        "valid": 0.20 * len(rows),
        "test": 0.20 * len(rows),
    }
    for _group_id, members in sorted(
        items, key=lambda item: len(item[1]), reverse=True
    ):
        destination = max(
            ("train", "valid", "test"),
            key=lambda split: targets[split] - len(assigned[split]),
        )
        assigned[destination].extend(members)

    return assigned["train"], assigned["valid"], assigned["test"]


def _pair_ids(rows: Iterable[dict]) -> set[str]:
    return {str(row.get("pair_id", "")) for row in rows}


def _validate_disjoint_splits(train: list[dict], valid: list[dict], test: list[dict]) -> None:
    ids = {"train": _pair_ids(train), "valid": _pair_ids(valid), "test": _pair_ids(test)}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = ids[left] & ids[right]
        if overlap:
            raise RuntimeError(
                f"pair_id leakage between {left} and {right}: {sorted(overlap)[:10]}"
            )


def _prepare_output(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output is not empty: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _run_augmentation(
    repo_root: Path,
    python_bin: str,
    dataset_root: Path,
    source_train_manifest: Path,
    augmented_root: Path,
    count: int,
    args,
) -> None:
    command = [
        "bash",
        str(repo_root / "scripts/data/run_real_bbox_strip_injection.sh"),
        "--data-dir", str(dataset_root),
        "--manifest", str(source_train_manifest),
        "--output-dir", str(augmented_root),
        "--num-pairs", str(count),
        "--seed", str(args.seed),
        "--height", str(args.height),
        "--min-regions", str(args.min_regions),
        "--max-regions", str(args.max_regions),
        "--max-run-boxes", str(args.max_run_boxes),
        "--injection-min-chars", str(args.injection_min_chars),
        "--injection-max-chars", str(args.injection_max_chars),
        "--injection-width-ratio-min", str(args.injection_width_ratio_min),
        "--injection-width-ratio-max", str(args.injection_width_ratio_max),
        "--max-attempts-per-output", str(args.max_attempts_per_output),
        "--max-attempts-per-region", str(args.max_attempts_per_region),
        "--overwrite",
    ]
    env = os.environ.copy()
    env["CONDA_ENV_PYTHON"] = python_bin
    env["PYTHONPATH"] = str(repo_root) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    print("Generating augmented training pairs:", flush=True)
    print("  " + " ".join(command), flush=True)
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def _decorate_augmented(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        record = copy.deepcopy(row)
        record["split"] = "train"
        record["sample_origin"] = "bbox_strip_augmented"
        record["training_positive"] = True
        # The source text_score no longer describes the generated pair.
        record.pop("scores", None)
        output.append(record)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("DataSet/ArabicDataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("DataSet/ArabicDatasetRealAug10K"))
    parser.add_argument("--target-train-pairs", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--min-regions", type=int, default=1)
    parser.add_argument("--max-regions", type=int, default=3)
    parser.add_argument("--max-run-boxes", type=int, default=3)
    parser.add_argument("--injection-min-chars", type=int, default=4)
    parser.add_argument("--injection-max-chars", type=int, default=28)
    parser.add_argument("--injection-width-ratio-min", type=float, default=0.50)
    parser.add_argument("--injection-width-ratio-max", type=float, default=2.00)
    parser.add_argument("--max-attempts-per-output", type=int, default=240)
    parser.add_argument("--max-attempts-per-region", type=int, default=120)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_train_pairs <= 0:
        raise ValueError("--target-train-pairs must be positive")
    if not (1 <= args.min_regions <= args.max_regions <= 3):
        raise ValueError("require 1 <= --min-regions <= --max-regions <= 3")

    repo_root = Path(__file__).resolve().parents[2]
    dataset_root = args.data_dir.expanduser().resolve()
    source_manifest = dataset_root / "dataset_manifest.jsonl"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Source manifest not found: {source_manifest}")

    output_root = args.output_dir.expanduser().resolve()
    _prepare_output(output_root, args.overwrite)

    all_source = _read_jsonl(source_manifest)
    positives = [
        row for row in all_source
        if str(row.get("label_type", "")) in POSITIVE_LABELS
    ]
    negatives = [
        row for row in all_source
        if str(row.get("label_type", "")) == NEGATIVE_LABEL
    ]
    other_labels = Counter(
        str(row.get("label_type", ""))
        for row in all_source
        if str(row.get("label_type", "")) not in POSITIVE_LABELS | {NEGATIVE_LABEL}
    )
    if not positives:
        raise RuntimeError("No high_match/medium_match real positive pairs were found")

    train_raw, valid_raw, test_raw = _diverse_group_split(positives, args.seed)
    _validate_disjoint_splits(train_raw, valid_raw, test_raw)

    original_train = [
        _absolutize_original_record(dataset_root, row, "train") for row in train_raw
    ]
    original_valid = [
        _absolutize_original_record(dataset_root, row, "valid") for row in valid_raw
    ]
    original_test = [
        _absolutize_original_record(dataset_root, row, "test") for row in test_raw
    ]
    original_negatives = [
        _absolutize_original_record(dataset_root, row, "negative") for row in negatives
    ]

    original_train_count = len(original_train)
    if original_train_count > args.target_train_pairs:
        raise RuntimeError(
            f"Original train split already has {original_train_count} pairs, larger than "
            f"target {args.target_train_pairs}; refusing to discard real pairs"
        )
    augment_count = args.target_train_pairs - original_train_count

    # Keep a source-only train manifest for provenance and as the only donor pool.
    source_train_manifest = output_root / "source_train_manifest.jsonl"
    _write_jsonl(source_train_manifest, original_train)

    augmented_rows: list[dict] = []
    augmented_root = output_root / "augmented"
    if augment_count > 0:
        _run_augmentation(
            repo_root=repo_root,
            python_bin=sys.executable,
            dataset_root=dataset_root,
            source_train_manifest=source_train_manifest,
            augmented_root=augmented_root,
            count=augment_count,
            args=args,
        )
        generated_manifest = augmented_root / "dataset_manifest.jsonl"
        if not generated_manifest.is_file():
            raise RuntimeError(f"Augmentation did not create {generated_manifest}")
        augmented_rows = _decorate_augmented(_read_jsonl(generated_manifest))
        if len(augmented_rows) != augment_count:
            raise RuntimeError(
                f"Expected {augment_count} augmented pairs, got {len(augmented_rows)}"
            )

    final_train = original_train + augmented_rows
    if len(final_train) != args.target_train_pairs:
        raise RuntimeError(
            f"Train manifest size mismatch: {len(final_train)} != {args.target_train_pairs}"
        )

    train_manifest = output_root / "train_manifest.jsonl"
    valid_manifest = output_root / "valid_manifest.jsonl"
    test_manifest = output_root / "test_manifest.jsonl"
    negative_manifest = output_root / "no_shared_content_manifest.jsonl"
    combined_manifest = output_root / "dataset_manifest.jsonl"

    _write_jsonl(train_manifest, final_train)
    _write_jsonl(valid_manifest, original_valid)
    _write_jsonl(test_manifest, original_test)
    _write_jsonl(negative_manifest, original_negatives)
    _write_jsonl(combined_manifest, [*final_train, *original_valid, *original_test])

    summary = {
        "source_dataset": str(dataset_root),
        "source_manifest": str(source_manifest),
        "output_dataset": str(output_root),
        "split_policy": "Evaluation/eval_img_align_sw.py pair_id-safe diverse 60/20/20 split",
        "split_seed": int(args.seed),
        "positive_labels": sorted(POSITIVE_LABELS),
        "negative_label_kept_separate": NEGATIVE_LABEL,
        "source_total_rows": len(all_source),
        "source_positive_rows": len(positives),
        "source_negative_rows": len(negatives),
        "source_other_labels": dict(other_labels),
        "original_train_pairs": original_train_count,
        "augmented_train_pairs": len(augmented_rows),
        "final_train_pairs": len(final_train),
        "valid_pairs": len(original_valid),
        "test_pairs": len(original_test),
        "negative_pairs": len(original_negatives),
        "target_train_pairs": int(args.target_train_pairs),
        "augmentation": {
            "type": "aligned bbox-exact full-height strip injection",
            "height_px": int(args.height),
            "regions_min": int(args.min_regions),
            "regions_max": int(args.max_regions),
            "max_run_boxes": int(args.max_run_boxes),
            "min_chars": int(args.injection_min_chars),
            "max_chars": int(args.injection_max_chars),
            "width_ratio_min": float(args.injection_width_ratio_min),
            "width_ratio_max": float(args.injection_width_ratio_max),
            "donor_pool": "train split only",
            "validation_test_used_as_donors": False,
            "online_stochastic_augmentation": False,
        },
        "manifests": {
            "train": str(train_manifest),
            "valid": str(valid_manifest),
            "test": str(test_manifest),
            "combined": str(combined_manifest),
            "negative": str(negative_manifest),
            "augmentation_source_train": str(source_train_manifest),
        },
    }
    summary_path = output_root / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Dataset ready: {output_root}", flush=True)
    print(f"Training manifest: {train_manifest}", flush=True)


if __name__ == "__main__":
    main()
