#!/usr/bin/env python3
"""Sanity-check unaligned evaluation with aligned control pairs.

Every sample, negative or aligned control, is passed through the exact same
Evaluation.evaluate_unaligned_pairs._evaluate_one function.  The controls exist
only to demonstrate that the negative evaluator is capable of producing masks.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
from PIL import Image, ImageDraw

from Evaluation import evaluate_unaligned_pairs as base


def _mask_nonempty(root: Path, role: int, index: int) -> bool:
    candidates = [
        root / "masks" / f"mask{role}_{index}.png",
        root / f"mask{role}_{index}.png",
    ]
    path = next((item for item in candidates if item.is_file()), None)
    if path is None:
        return False
    with Image.open(path) as opened:
        return bool(np.any(np.asarray(opened.convert("L")) > 0))


def _synthetic_controls(args, count: int) -> list[dict]:
    root = Path(args.data_dir).expanduser().resolve()
    indices = base._fixed63_test_indices(args.num_samples, args.split_seed)
    rng = random.Random(args.seed + 9173)
    rng.shuffle(indices)
    controls = []
    for index in indices:
        if not (_mask_nonempty(root, 1, index) and _mask_nonempty(root, 2, index)):
            continue
        image1, text1_path = base._synthetic_paths(root, 1, index)
        image2, text2_path = base._synthetic_paths(root, 2, index)
        controls.append(
            {
                "image1": image1,
                "image2": image2,
                "text1": text1_path.read_text(encoding="utf-8").strip(),
                "text2": text2_path.read_text(encoding="utf-8").strip(),
                "source_index1": index,
                "source_index2": index,
                "pair_id": f"synthetic_aligned_control_{index}",
                "label_type": "aligned_control_same_sample",
                "selection_diagnostics": {"gt_masks_nonempty": True},
                "sample_kind": "ALIGNED_CONTROL",
                "expected_mask": True,
            }
        )
        if len(controls) >= count:
            break
    if len(controls) < count:
        raise RuntimeError(
            f"Found only {len(controls)} synthetic aligned controls with non-empty GT masks; "
            f"requested {count}."
        )
    return controls


def _real_controls(args, count: int) -> list[dict]:
    root = Path(args.data_dir).expanduser().resolve()
    manifest = Path(args.arabic_manifest or root / "dataset_manifest.jsonl")
    namespace = SimpleNamespace(
        arabic_manifest=str(manifest),
        data_dir=str(root),
        real_text_key=args.real_text_key,
        real_labels="high_match,medium_match",
        real_min_text_score=0.0,
        real_validate_paths=True,
        split_seed=args.split_seed,
        real_split=args.real_split,
    )
    pairs = list(base.load_arabic_dataset_pairs(namespace))
    rng = random.Random(args.seed + 9173)
    rng.shuffle(pairs)
    controls = []
    for pair in pairs[:count]:
        controls.append(
            {
                "image1": Path(pair.image1),
                "image2": Path(pair.image2),
                "text1": "",
                "text2": "",
                "source_index1": pair.manifest_position,
                "source_index2": pair.manifest_position,
                "pair_id": pair.pair_id,
                "label_type": f"aligned_control_{pair.label_type}",
                "selection_diagnostics": {"source_label": pair.label_type},
                "sample_kind": "ALIGNED_CONTROL",
                "expected_mask": True,
            }
        )
    if len(controls) < count:
        raise RuntimeError(
            f"Found only {len(controls)} real aligned controls in split={args.real_split}; "
            f"requested {count}."
        )
    return controls


def _negative_records(args) -> list[dict]:
    records = (
        base._select_synthetic_negatives(args)
        if args.dataset_type == "synthetic"
        else base._select_real_negatives(args)
    )
    for record in records:
        record["sample_kind"] = "UNALIGNED"
        record["expected_mask"] = False
    return records


def _add_control_banner(path: Path, kind: str, expected_mask: bool, detected_mask: bool) -> None:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
    banner_height = 52
    canvas = Image.new("RGB", (image.width, image.height + banner_height), "white")
    canvas.paste(image, (0, banner_height))
    draw = ImageDraw.Draw(canvas)
    expectation = "MASK EXPECTED" if expected_mask else "NO MASK EXPECTED"
    result = "MASK DETECTED" if detected_mask else "NO MASK DETECTED"
    draw.text((14, 8), f"{kind} | {expectation} | {result}", fill="black")
    draw.text(
        (14, 27),
        "Same _evaluate_one() prediction function is used for aligned controls and unaligned negatives.",
        fill="black",
    )
    canvas.save(path)


def _summary(rows: list[dict], dataset_type: str) -> dict:
    negatives = [row for row in rows if row["sample_kind"] == "UNALIGNED"]
    controls = [row for row in rows if row["sample_kind"] == "ALIGNED_CONTROL"]
    negative_zero = sum(bool(row["zero_mask"]) for row in negatives)
    control_detected = sum(not bool(row["zero_mask"]) for row in controls)
    return {
        "dataset_type": dataset_type,
        "same_prediction_function": "Evaluation.evaluate_unaligned_pairs._evaluate_one",
        "unaligned_samples": len(negatives),
        "unaligned_zero_mask_pairs": negative_zero,
        "unaligned_zero_mask_rate": negative_zero / len(negatives) if negatives else 0.0,
        "unaligned_false_positive_pairs": len(negatives) - negative_zero,
        "unaligned_false_positive_rate": (len(negatives) - negative_zero) / len(negatives) if negatives else 0.0,
        "aligned_control_samples": len(controls),
        "aligned_control_mask_detected_pairs": control_detected,
        "aligned_control_mask_detection_rate": control_detected / len(controls) if controls else 0.0,
        "aligned_control_missed_pairs": len(controls) - control_detected,
        "sanity_check_demonstrates_mask_capability": bool(control_detected > 0),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-type", choices=("synthetic", "real"), required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--arabic-manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=20, help="Number of unaligned negatives")
    parser.add_argument("--n-aligned-controls", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=27000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-split", choices=("train", "valid", "test", "all"), default="test")
    parser.add_argument("--real-text-key", default="text_original_path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature", choices=("contextual", "local", "grouped"), default="contextual")
    parser.add_argument("--score-mode", choices=("auto", "raw", "centered", "mutual-z"), default="auto")
    parser.add_argument("--score-clip", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--gap", type=float, default=-0.30)
    parser.add_argument("--min-shared-word-chars", type=int, default=4)
    parser.add_argument("--min-common-compact-chars", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_samples <= 0 or args.n_aligned_controls <= 0:
        raise SystemExit("Both --n-samples and --n-aligned-controls must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identical v3-local policy used by the existing synthetic negative evaluator.
    base.os.environ.setdefault("NW_COMPONENT_WEAK_GLOBAL_SCORE", "-1000000.0")
    base.os.environ.setdefault("NW_COMPONENT_MIN_MATCHES", "7")
    base.os.environ.setdefault("NW_COMPONENT_MIN_SPAN_WINDOWS", "7")
    base.os.environ.setdefault("NW_COMPONENT_MIN_SPAN_FRACTION", "0.13")

    negatives = _negative_records(args)
    controls = (
        _synthetic_controls(args, args.n_aligned_controls)
        if args.dataset_type == "synthetic"
        else _real_controls(args, args.n_aligned_controls)
    )
    records = negatives + controls
    random.Random(args.seed + 314159).shuffle(records)

    serializable = []
    for record in records:
        item = dict(record)
        item["image1"] = str(item["image1"])
        item["image2"] = str(item["image2"])
        serializable.append(item)
    (output_dir / "selected_mixed_pairs.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    models = base.load_evaluation_models(args.weights, args.device, load_text_model=False)
    rows = []
    for position, record in enumerate(records, start=1):
        kind = record["sample_kind"]
        output = output_dir / f"pair_{position:03d}_{kind.lower()}.png"
        # Critical sanity-check property: BOTH classes call the exact same predictor.
        row = base._evaluate_one(models, record, args.dataset_type, args, output)
        detected_mask = not bool(row["zero_mask"])
        expected_mask = bool(record["expected_mask"])
        row["index"] = position
        row["sample_kind"] = kind
        row["expected_mask"] = expected_mask
        row["detected_mask"] = detected_mask
        row["prediction_matches_expectation"] = detected_mask == expected_mask
        _add_control_banner(output, kind, expected_mask, detected_mask)
        rows.append(row)
        print(
            f"[{position}/{len(records)}] {kind} pair_id={row['pair_id']} "
            f"expected_mask={expected_mask} detected_mask={detected_mask} "
            f"components={row['predicted_components']}",
            flush=True,
        )

    fieldnames = [
        "index", "sample_kind", "expected_mask", "detected_mask",
        "prediction_matches_expectation", "pair_id", "label_type",
        "source_index1", "source_index2", "algorithm", "score",
        "normalized_score", "predicted_components", "zero_mask",
        "false_positive_mask", "false_mask_fraction_line1",
        "false_mask_fraction_line2", "mean_false_mask_fraction",
        "image1", "image2", "output",
    ]
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = _summary(rows, args.dataset_type)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
