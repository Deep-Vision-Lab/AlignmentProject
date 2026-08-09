#!/usr/bin/env python3
"""Build a standalone real+augmented Arabic dataset using the real dataset skeleton.

The output is a physical dataset, not a manifest of references.  It copies the
original ArabicDataset tree and then materializes bbox-exact aligned injection
pairs under the same ``DatasetPairs/page_pairs/pair_XXXXXX/A|B`` hierarchy.

Training leakage policy
-----------------------
* Positive original rows (high_match / medium_match) are split by pair_id with
  the same deterministic diverse 60/20/20 policy used by the current evaluator.
* Only original TRAIN rows are used as augmentation targets and donor sources.
* Validation and test remain original-only.
* The final training manifest is expanded to ``--target-train-pairs`` rows.
* ``no_shared_content`` rows are copied with the original dataset and are kept
  out of the positive training manifest.

Augmented real-pair skeleton
----------------------------
Each generated sample is converted from the bbox-strip generator output into::

    DatasetPairs/page_pairs/pair_XXXXXX/
      A/
        linesImages/line_01.png
        text/final/original/line_01.txt
        text/final/tashkeel/line_01.txt
        text/raw/line_01.txt
        bbox.json
        debug/bboxes.json
        debug/augmentation.json
      B/
        ... same skeleton ...

The generated ``debug/bboxes.json`` uses the same flat text-bearing bbox schema
observed in the original real dataset (x1,y1,x2,y2,w,h,cx,cy,text).  The
line-local ``bbox.json`` is retained as well for unambiguous downstream use.

Main manifests
--------------
``dataset_manifest.jsonl`` contains every original source row plus all generated
rows and an explicit split field.  ``train_manifest.jsonl`` contains original
train positives plus generated positives.  Validation/test manifests contain
only untouched original positives.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import copy
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from typing import Iterable


POSITIVE_LABELS = {"high_match", "medium_match"}
NEGATIVE_LABEL = "no_shared_content"
_PAIR_DIR_RE = re.compile(r"pair_(\d+)$", re.IGNORECASE)
_PATH_KEYS = (
    "line_image_path",
    "text_original_path",
    "text_tashkeel_path",
    "text_raw_path",
    "bbox_path",
)


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


def _random_split(rows: list[dict], seed: int):
    indices = list(range(len(rows)))
    random.Random(int(seed)).shuffle(indices)
    train_size = int(0.60 * len(indices))
    valid_size = int(0.20 * len(indices))
    train = [rows[i] for i in indices[:train_size]]
    valid = [rows[i] for i in indices[train_size : train_size + valid_size]]
    test = [rows[i] for i in indices[train_size + valid_size :]]
    return train, valid, test


def _diverse_group_split(rows: list[dict], seed: int):
    """Mirror Evaluation/eval_img_align_sw.py's pair_id-safe split."""
    groups: OrderedDict[str, list[dict]] = OrderedDict()
    for position, row in enumerate(rows):
        group_id = str(row.get("pair_id") or f"sample_{position}")
        groups.setdefault(group_id, []).append(row)
    if len(groups) < 3:
        return _random_split(rows, seed)

    rng = random.Random(int(seed))
    items = list(groups.items())
    rng.shuffle(items)
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
    ids = {
        "train": _pair_ids(train),
        "valid": _pair_ids(valid),
        "test": _pair_ids(test),
    }
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        overlap = ids[left] & ids[right]
        if overlap:
            raise RuntimeError(
                f"pair_id leakage between {left} and {right}: {sorted(overlap)[:10]}"
            )


def _prepare_output(output_root: Path, source_root: Path, overwrite: bool) -> None:
    try:
        output_root.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            f"Output dataset must not be inside source dataset: {output_root}"
        )

    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output is not empty: {output_root}; set OVERWRITE=1 or pass --overwrite"
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def _resolve_source_path(dataset_root: Path, value) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path.resolve()
    for candidate in (
        dataset_root / path,
        dataset_root.parent / path,
        Path.cwd() / path,
    ):
        if candidate.exists():
            return candidate.resolve()
    return (dataset_root / path).resolve()


def _source_relative_path(dataset_root: Path, value) -> str:
    resolved = _resolve_source_path(dataset_root, value)
    try:
        return resolved.relative_to(dataset_root).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Source manifest path is outside ArabicDataset and cannot be made standalone: {resolved}"
        ) from exc


def _portable_original_record(
    dataset_root: Path,
    row: dict,
    split: str,
) -> dict:
    """Keep original metadata but make file paths relative to the copied dataset."""
    record = copy.deepcopy(row)
    record["split"] = split
    record["sample_origin"] = "original_real"
    record["training_positive"] = str(record.get("label_type", "")) in POSITIVE_LABELS
    for side_name in ("A", "B"):
        side = record.get(side_name)
        if not isinstance(side, dict):
            raise ValueError(f"record {record.get('pair_id')} missing side {side_name}")
        for key in _PATH_KEYS:
            if side.get(key):
                side[key] = _source_relative_path(dataset_root, side[key])
        for required in ("line_image_path", "text_original_path"):
            if not side.get(required):
                raise ValueError(
                    f"record {record.get('pair_id')} side {side_name} missing {required}"
                )
            if not (dataset_root / side[required]).is_file():
                raise FileNotFoundError(
                    f"record {record.get('pair_id')} side {side_name} unresolved {required}: "
                    f"{dataset_root / side[required]}"
                )
    return record


def _copy_source_dataset(source_root: Path, output_root: Path) -> None:
    """Physically copy the source dataset while preserving a temporary build dir."""
    print(f"Copying original real dataset: {source_root} -> {output_root}", flush=True)

    def ignore(directory: str, names: list[str]):
        # The temporary build directory belongs only to the destination, but be
        # defensive if a source dataset ever contains one with the same name.
        ignored = []
        if Path(directory).resolve() == source_root.resolve() and ".real_aug_build" in names:
            ignored.append(".real_aug_build")
        return ignored

    shutil.copytree(
        source_root,
        output_root,
        dirs_exist_ok=True,
        copy_function=shutil.copy2,
        ignore=ignore,
    )
    print("Original dataset copy complete.", flush=True)


def _run_augmentation(
    repo_root: Path,
    python_bin: str,
    source_root: Path,
    source_train_manifest: Path,
    generated_root: Path,
    count: int,
    args,
) -> None:
    if count <= 0:
        return
    command = [
        "bash",
        str(repo_root / "scripts/data/run_real_bbox_strip_injection.sh"),
        "--data-dir", str(source_root),
        "--manifest", str(source_train_manifest),
        "--output-dir", str(generated_root),
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
    print(f"Generating {count} bbox-injected train pairs...", flush=True)
    subprocess.run(command, cwd=repo_root, env=env, check=True)


def _next_pair_number(page_pairs_root: Path) -> int:
    maximum = 0
    if page_pairs_root.is_dir():
        for child in page_pairs_root.iterdir():
            if not child.is_dir():
                continue
            match = _PAIR_DIR_RE.fullmatch(child.name)
            if match:
                maximum = max(maximum, int(match.group(1)))
    return maximum + 1


def _read_generated_bbox(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    boxes = payload.get("boxes") if isinstance(payload, dict) else None
    if not isinstance(boxes, list) or not boxes:
        raise ValueError(f"Generated bbox payload has no boxes: {path}")
    return payload


def _flat_debug_boxes(payload: dict) -> list[dict]:
    output = []
    for box in payload["boxes"]:
        text = str(box.get("text", "")).strip()
        x0 = float(box["x0"])
        y0 = float(box["y0"])
        x1 = float(box["x1"])
        y1 = float(box["y1"])
        if not text or x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid generated subword bbox: {box}")
        output.append(
            {
                "x1": round(x0, 3),
                "y1": round(y0, 3),
                "x2": round(x1, 3),
                "y2": round(y1, 3),
                "w": round(x1 - x0, 3),
                "h": round(y1 - y0, 3),
                "cx": round((x0 + x1) / 2.0, 3),
                "cy": round((y0 + y1) / 2.0, 3),
                "text": text,
            }
        )
    return output


def _copy_generated_side(
    generated_side: Path,
    destination_side: Path,
    augmentation_metadata: dict,
) -> dict:
    source_image = generated_side / "line.png"
    source_text = generated_side / "text_original.txt"
    source_bbox = generated_side / "bbox.json"
    for path in (source_image, source_text, source_bbox):
        if not path.is_file():
            raise FileNotFoundError(f"Missing generated augmentation file: {path}")

    image_rel = Path("linesImages/line_01.png")
    original_rel = Path("text/final/original/line_01.txt")
    tashkeel_rel = Path("text/final/tashkeel/line_01.txt")
    raw_rel = Path("text/raw/line_01.txt")
    bbox_rel = Path("bbox.json")
    debug_bbox_rel = Path("debug/bboxes.json")
    augmentation_rel = Path("debug/augmentation.json")

    for relative in (
        image_rel,
        original_rel,
        tashkeel_rel,
        raw_rel,
        bbox_rel,
        debug_bbox_rel,
        augmentation_rel,
    ):
        (destination_side / relative).parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_image, destination_side / image_rel)
    text = source_text.read_text(encoding="utf-8").strip()
    for relative in (original_rel, tashkeel_rel, raw_rel):
        (destination_side / relative).write_text(text + "\n", encoding="utf-8")

    bbox_payload = _read_generated_bbox(source_bbox)
    bbox_payload = copy.deepcopy(bbox_payload)
    bbox_payload["image"] = "line_01.png"
    (destination_side / bbox_rel).write_text(
        json.dumps(bbox_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (destination_side / debug_bbox_rel).write_text(
        json.dumps(_flat_debug_boxes(bbox_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (destination_side / augmentation_rel).write_text(
        json.dumps(augmentation_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "line_idx": 1,
        "line_image_path": image_rel.as_posix(),
        "text_original_path": original_rel.as_posix(),
        "text_tashkeel_path": tashkeel_rel.as_posix(),
        "text_raw_path": raw_rel.as_posix(),
        "bbox_path": bbox_rel.as_posix(),
        "text_tashkeel_is_original_fallback": True,
    }


def _materialize_generated_pairs(
    generated_rows: list[dict],
    generated_root: Path,
    output_root: Path,
) -> list[dict]:
    page_pairs_root = output_root / "DatasetPairs" / "page_pairs"
    page_pairs_root.mkdir(parents=True, exist_ok=True)
    next_number = _next_pair_number(page_pairs_root)
    output_rows: list[dict] = []

    for index, generated in enumerate(generated_rows):
        pair_number = next_number + index
        pair_name = f"pair_{pair_number:06d}"
        pair_root = page_pairs_root / pair_name
        if pair_root.exists():
            raise FileExistsError(f"Generated pair destination already exists: {pair_root}")

        source_label = str(generated.get("source_label_type", ""))
        label_type = source_label if source_label in POSITIVE_LABELS else "high_match"
        augmentation = copy.deepcopy(generated.get("augmentation") or {})
        augmentation.update(
            {
                "dataset_builder": "build_real_augmented_same_skeleton.py",
                "generated_pair_index": index + 1,
                "source_pair_id": generated.get("source_pair_id"),
            }
        )

        generated_pair_root = generated_root / "pairs" / f"aug_{index + 1:06d}"
        side_a = _copy_generated_side(
            generated_pair_root / "A", pair_root / "A", augmentation
        )
        side_b = _copy_generated_side(
            generated_pair_root / "B", pair_root / "B", augmentation
        )

        prefix = Path("DatasetPairs") / "page_pairs" / pair_name
        for side_name, side in (("A", side_a), ("B", side_b)):
            for key in _PATH_KEYS:
                if side.get(key):
                    side[key] = (prefix / side_name / side[key]).as_posix()

        output_rows.append(
            {
                "pair_id": pair_name,
                "label_type": label_type,
                "source_label_type": source_label,
                "source_pair_id": generated.get("source_pair_id"),
                "split": "train",
                "sample_origin": "bbox_strip_augmented",
                "training_positive": True,
                "augmentation": augmentation,
                "A": side_a,
                "B": side_b,
            }
        )
    return output_rows


def _split_name_for_original(
    row: dict,
    train_ids: set[str],
    valid_ids: set[str],
    test_ids: set[str],
) -> str:
    pair_id = str(row.get("pair_id", ""))
    label = str(row.get("label_type", ""))
    if label in POSITIVE_LABELS:
        if pair_id in train_ids:
            return "train"
        if pair_id in valid_ids:
            return "valid"
        if pair_id in test_ids:
            return "test"
        raise RuntimeError(f"Positive original pair has no split: {pair_id}")
    if label == NEGATIVE_LABEL:
        return "negative"
    return "other"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("DataSet/ArabicDataset"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DataSet/ArabicDatasetRealAug10K"),
    )
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
    parser.add_argument("--keep-build-artifacts", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.target_train_pairs <= 0:
        raise ValueError("--target-train-pairs must be positive")
    if not (1 <= args.min_regions <= args.max_regions <= 3):
        raise ValueError("require 1 <= --min-regions <= --max-regions <= 3")

    repo_root = Path(__file__).resolve().parents[2]
    source_root = args.data_dir.expanduser().resolve()
    source_manifest = source_root / "dataset_manifest.jsonl"
    if not source_manifest.is_file():
        raise FileNotFoundError(f"Source manifest not found: {source_manifest}")
    if not (source_root / "DatasetPairs" / "page_pairs").is_dir():
        raise FileNotFoundError(
            f"Expected real dataset skeleton missing: {source_root / 'DatasetPairs/page_pairs'}"
        )

    output_root = args.output_dir.expanduser().resolve()
    _prepare_output(output_root, source_root, args.overwrite)
    build_root = output_root / ".real_aug_build"
    build_root.mkdir(parents=True, exist_ok=True)

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
        raise RuntimeError("No high_match/medium_match positive rows found")

    train_raw, valid_raw, test_raw = _diverse_group_split(positives, args.seed)
    _validate_disjoint_splits(train_raw, valid_raw, test_raw)
    if len(train_raw) > args.target_train_pairs:
        raise RuntimeError(
            f"Original real train split already has {len(train_raw)} rows, greater than "
            f"target {args.target_train_pairs}; refusing to discard original real data"
        )

    augment_count = args.target_train_pairs - len(train_raw)
    source_train_manifest = build_root / "source_train_manifest.jsonl"
    # Use the original source rows here so the bbox generator resolves paths
    # against the original ArabicDataset exactly as before.
    _write_jsonl(source_train_manifest, train_raw)

    generated_root = build_root / "generated_bbox_injection"
    if augment_count:
        _run_augmentation(
            repo_root=repo_root,
            python_bin=sys.executable,
            source_root=source_root,
            source_train_manifest=source_train_manifest,
            generated_root=generated_root,
            count=augment_count,
            args=args,
        )
        generated_manifest = generated_root / "dataset_manifest.jsonl"
        if not generated_manifest.is_file():
            raise RuntimeError(f"Augmentation did not create: {generated_manifest}")
        generated_rows = _read_jsonl(generated_manifest)
        if len(generated_rows) != augment_count:
            raise RuntimeError(
                f"Expected {augment_count} generated rows, got {len(generated_rows)}"
            )
    else:
        generated_rows = []

    # Copy only after augmentation preflight/generation succeeds so a bbox error
    # does not waste time duplicating the whole source dataset first.
    _copy_source_dataset(source_root, output_root)

    train_ids = _pair_ids(train_raw)
    valid_ids = _pair_ids(valid_raw)
    test_ids = _pair_ids(test_raw)

    portable_originals: list[dict] = []
    for row in all_source:
        split = _split_name_for_original(row, train_ids, valid_ids, test_ids)
        portable_originals.append(_portable_original_record(source_root, row, split))

    augmented_rows = _materialize_generated_pairs(
        generated_rows=generated_rows,
        generated_root=generated_root,
        output_root=output_root,
    )

    original_train = [
        row for row in portable_originals
        if row.get("split") == "train" and row.get("training_positive")
    ]
    original_valid = [
        row for row in portable_originals
        if row.get("split") == "valid" and row.get("training_positive")
    ]
    original_test = [
        row for row in portable_originals
        if row.get("split") == "test" and row.get("training_positive")
    ]
    original_negatives = [
        row for row in portable_originals if row.get("split") == "negative"
    ]

    final_train = original_train + augmented_rows
    if len(final_train) != args.target_train_pairs:
        raise RuntimeError(
            f"Final train count mismatch: {len(final_train)} != {args.target_train_pairs}"
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
    _write_jsonl(combined_manifest, [*portable_originals, *augmented_rows])

    summary = {
        "dataset_type": "standalone_real_plus_bbox_injection",
        "source_dataset": str(source_root),
        "output_dataset": str(output_root),
        "architecture": "ArabicDataset/DatasetPairs/page_pairs/pair_XXXXXX/A|B",
        "original_dataset_physically_copied": True,
        "source_total_rows": len(all_source),
        "source_positive_rows": len(positives),
        "source_negative_rows": len(negatives),
        "source_other_labels": dict(other_labels),
        "split_seed": int(args.seed),
        "split_policy": "pair_id-safe diverse 60/20/20; augmentation train-only",
        "original_train_positive_rows": len(original_train),
        "augmented_train_positive_rows": len(augmented_rows),
        "final_train_positive_rows": len(final_train),
        "original_valid_positive_rows": len(original_valid),
        "original_test_positive_rows": len(original_test),
        "original_negative_rows": len(original_negatives),
        "combined_manifest_rows": len(portable_originals) + len(augmented_rows),
        "target_train_pairs": int(args.target_train_pairs),
        "augmentation": {
            "type": "aligned bbox-exact full-height strip injection",
            "height_px": int(args.height),
            "regions_min": int(args.min_regions),
            "regions_max": int(args.max_regions),
            "max_run_boxes": int(args.max_run_boxes),
            "donor_pool": "original train split only",
            "validation_test_used_as_donors": False,
            "augmented_label_policy": "inherit source high_match/medium_match label",
            "online_stochastic_augmentation": False,
        },
        "manifests": {
            "combined": "dataset_manifest.jsonl",
            "train": "train_manifest.jsonl",
            "valid": "valid_manifest.jsonl",
            "test": "test_manifest.jsonl",
            "negative": "no_shared_content_manifest.jsonl",
        },
    }
    (output_root / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if not args.keep_build_artifacts:
        shutil.rmtree(build_root, ignore_errors=True)

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Standalone dataset ready: {output_root}", flush=True)
    print(f"Training manifest: {train_manifest}", flush=True)


if __name__ == "__main__":
    main()
