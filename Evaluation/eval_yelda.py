#!/usr/bin/env python3
"""Image-only NW evaluation of the original hierarchy and cross-attention depiction runs."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict, deque
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def local_weight_value(value):
    weight = float(value)
    if not math.isfinite(weight) or not 0.0 <= weight <= 1.0:
        raise argparse.ArgumentTypeError("local-weight must be finite and between 0 and 1")
    return weight


def evaluation_similarity(first, second, args, output_dir):
    from Evaluation._eval_utils import compute_similarity
    if args.representation != "joint":
        return compute_similarity(first.select(args.feature), second.select(args.feature))
    import numpy as np
    from Evaluation.joint_similarity import joint_components
    local, contextual, joint = joint_components(
        first.local, second.local, first.contextual, second.contextual, args.local_weight
    )
    for name, scores in (("local_cosine_similarity", local),
                         ("contextual_cosine_similarity", contextual),
                         ("joint_similarity", joint)):
        array = scores.detach().cpu().numpy().astype(np.float32)
        np.save(output_dir / f"{name}.npy", array)
        np.savetxt(output_dir / f"{name}.csv", array, delimiter=",", fmt="%.8f")
    return joint


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--branch",
        choices=("auto", "hierarchy", "cross", "spatial", "restoration"),
        default="auto",
    )
    parser.add_argument("--image-preprocessing", choices=("original", "training"), default="original",
                        help="original: full RGB image resized to 1024x128, without cropping, padding or binarization")
    parser.add_argument("--representation", choices=("joint", "primary", "local", "independent"), default="joint")
    parser.add_argument("--local-weight", type=local_weight_value, default=0.5,
                        help="Joint score: weight of local cosine; contextual weight is 1 minus this")
    parser.add_argument("--split", choices=("test", "valid", "train", "all"), default="test")
    parser.add_argument("--training-samples", type=int, default=6000,
                        help="Synthetic population used by training before its 60/20/20 split")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-samples", type=int, default=100, help="0 means the entire selected split")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-mode", choices=("raw", "centered", "mutual-z"), default="raw")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--gap", type=float, default=-0.30)
    parser.add_argument("--score-clip", type=float, default=4.0)
    parser.add_argument("--min-ink", type=float, default=0.02)
    return parser.parse_args(argv)


def synthetic_split(pairs, split, population, seed):
    import torch
    if split == "all":
        return list(pairs)
    if population <= 0:
        raise ValueError("--training-samples must be positive")
    # Training uses records 1..min(num_samples, detected), then torch.random_split.
    count = min(population, len(pairs))
    by_id = {pair.manifest_position: pair for pair in pairs}
    missing = set(range(1, count + 1)) - set(by_id)
    if missing:
        raise ValueError(f"Cannot reproduce training split: missing synthetic IDs {sorted(missing)[:10]}")
    indices = torch.randperm(count, generator=torch.Generator().manual_seed(seed)).tolist()
    train, valid = int(0.6 * count), int(0.2 * count)
    slices = {"train": indices[:train], "valid": indices[train:train + valid], "test": indices[train + valid:]}
    return [replace(by_id[index + 1], split=split) for index in slices[split]]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def balanced_pairs(pairs):
    """Round-robin page-pair groups within an already selected real split."""
    groups = defaultdict(deque)
    for pair in pairs:
        groups[pair.pair_id or str(pair.index)].append(pair)
    ordered = []
    while groups:
        for key in list(groups):
            ordered.append(groups[key].popleft())
            if not groups[key]:
                del groups[key]
    return ordered


def configure_geometry(config, image_preprocessing="original"):
    if int(config.get("line_height", 128)) != 128 or int(config.get("line_width", 1024)) != 1024:
        raise ValueError("This evaluator currently supports the trained 128x1024 line canvas")
    if image_preprocessing == "original":
        return {"line_height": 128, "line_width": 1024,
                "line_geometry_mode": "full-image-resize", "color_mode": "RGB",
                "binarize": False, "crop_foreground": False, "padding": False,
                "preserve_aspect": False, "autocontrast": False, "auto_invert": False}
    if image_preprocessing != "training":
        raise ValueError(f"Unknown image preprocessing: {image_preprocessing}")
    # Current runs record geometry; older preprocessing flags fall back to the
    # branch's Parameters.py. The resolved values are included in the report.
    mapping = {
        "line_height": "LINE_HEIGHT", "line_width": "LINE_WIDTH",
        "target_ink_height_ratio": "TARGET_INK_HEIGHT_RATIO",
        "zero_shot_preprocess": "ZERO_SHOT_PREPROCESS",
        "zero_shot_preserve_aspect": "ZERO_SHOT_PRESERVE_ASPECT",
        "zero_shot_foreground_crop": "ZERO_SHOT_FOREGROUND_CROP",
        "real_binarize": "REAL_BINARIZE", "synthetic_binarize": "SYNTHETIC_BINARIZE",
    }
    for key, env in mapping.items():
        if key in config:
            value = config[key]
            os.environ[env] = str(int(value)) if isinstance(value, bool) else str(value)
    if config.get("line_geometry_mode", "source-compatible-height") != "source-compatible-height":
        raise ValueError("Unsupported checkpoint geometry")
    os.environ["ZERO_SHOT_SOURCE_GEOMETRY"] = "1"
    from unified_line_geometry import install_evaluation_geometry
    return install_evaluation_geometry()


def evaluate_pair(base, models, pair, args, destination):
    from PIL import Image
    from Evaluation.yelda_geometry import prepare_line, source_intervals
    destination.mkdir(parents=True)
    prepared, geometry = [], []
    for role, path in ((1, pair.image1), (2, pair.image2)):
        image, mapping = prepare_line(path, pair.preprocess_domain(role), args.image_preprocessing)
        output = destination / f"line{role}_model_input.png"
        image.save(output)
        prepared.append(output)
        geometry.append(mapping)
    transformed = replace(pair, image1=prepared[0], image2=prepared[1],
                          side1_preprocess="synthetic", side2_preprocess="synthetic",
                          gt_mask1=None, gt_mask2=None)
    row = base.evaluate(models, transformed, args, destination)
    # Score region masks in ORIGINAL source coordinates, using the inverse
    # crop/scale/pad transform. Do not stretch raw GT to the normalized canvas.
    for role, mapping, gt_path in ((1, geometry[0], pair.gt_mask1), (2, geometry[1], pair.gt_mask2)):
        intervals = source_intervals(row[f"line{role}_intervals_px"], mapping)
        shape = (mapping["source_height"], mapping["source_width"])
        pred = base._predicted_mask(shape, intervals)
        Image.fromarray(pred).save(destination / f"line{role}_source_pred_mask.png")
        if gt_path is not None:
            with Image.open(gt_path) as gt_image:
                if gt_image.size != (shape[1], shape[0]):
                    raise ValueError(f"Ground-truth mask must match source image size: {gt_path}")
        gt = base._load_gt_mask(gt_path, shape)
        if gt is not None:
            Image.fromarray(gt).save(destination / f"line{role}_source_gt_mask.png")
        row.update(base._mask_metrics(pred, gt, f"line{role}"))
        row[f"line{role}_source_intervals_px"] = intervals
    row["mean_mask_iou"] = base._mean([{"iou": row[f"line{role}_mask_iou"]} for role in (1, 2)], "iou")
    row.update(image1=str(pair.image1), image2=str(pair.image2),
               side1_preprocess=pair.preprocess_domain(1), side2_preprocess=pair.preprocess_domain(2),
               gt_mask1=str(pair.gt_mask1 or ""), gt_mask2=str(pair.gt_mask2 or ""),
               feature="joint" if args.representation == "joint" else args.feature, representation=args.representation, local_weight=args.local_weight if args.representation == "joint" else None, geometry=geometry, mask_coordinate_system="source_image")
    write_json(destination / "summary.json", row)
    return row


def main(argv=None):
    args = parse_args(argv)
    if args.n_samples < 0 or args.start_index < 1:
        raise SystemExit("n-samples must be nonnegative; start-index must be positive")
    dataset, weights = Path(args.dataset).expanduser().resolve(), Path(args.weights).expanduser().resolve()
    destination = Path(args.output_dir).expanduser().resolve()
    if not dataset.exists() or not weights.is_file():
        raise SystemExit("Dataset and checkpoint must exist before evaluation")
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"Output directory is not empty; choose a new run directory: {destination}")

    from Evaluation import eval_img_align_nw_diagnostic as base
    from Evaluation.yelda_runtime import read_checkpoint, load_visual_models, pair_features, configure_image_preprocessing
    from Evaluation.trace_components import component_settings
    checkpoint = read_checkpoint(weights)
    models = load_visual_models(checkpoint, args.device, args.branch)
    geometry = configure_geometry(models.config, args.image_preprocessing)
    input_settings = configure_image_preprocessing(models, args.image_preprocessing)
    base.P.dataset_split_seed = args.split_seed
    os.environ["DISCRETE_ALIGNMENT_SCORES"] = "0"
    os.environ["SW_INK_AWARE"] = "1"
    os.environ["SW_MIN_INK"] = str(args.min_ink)
    os.environ["SW_BLANK_BLANK_SCORE"] = "-0.20"
    os.environ["SW_BLANK_INK_SCORE"] = "-0.50"
    args.feature = "local" if args.representation == "local" else "contextual"
    layout, pairs = base.load_pairs(dataset, args.split)
    if layout == "synthetic":
        pairs = synthetic_split(pairs, args.split, args.training_samples, args.split_seed)
    elif args.split != "all" and any(pair.split != args.split for pair in pairs):
        raise ValueError("Dataset has no reproducible requested split; use an explicit split manifest or --split all")
    if layout != "synthetic":
        pairs = balanced_pairs(pairs)
    start = args.start_index - 1
    selected = pairs[start:] if args.n_samples == 0 else pairs[start:start + args.n_samples]
    if not selected:
        raise ValueError("No pairs selected")
    destination.mkdir(parents=True, exist_ok=True)
    family = str(models.config.get("architecture_family", ""))
    branch = (
        "restoration"
        if family == "restoration-positive-dtw-window-encoder"
        else (
            "spatial"
            if family == "cfm-inspired-spatial-language-alignment"
            else ("cross" if models.pair_cross_attention is not None else "hierarchy")
        )
    )
    if branch == "restoration":
        stage = (
            "joint_primitive_semantic"
            if args.representation == "joint"
            else (
                "primitive_stroke"
                if args.representation == "local"
                else "semantic_letter_aligned"
            )
        )
    else:
        stage = "fused_contextual" if branch == "cross" and args.representation in {"primary", "joint"} else args.feature
    if args.representation == "joint" and branch != "restoration":
        stage = "joint_local_" + stage
    selection = [{"pair_id": p.pair_id, "index": p.index, "split": p.split,
                  "image1": str(p.image1), "image2": str(p.image2)} for p in selected]
    write_json(destination / "selected_pairs.json", selection)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    metadata = {"algorithm": "needleman_wunsch", "branch": branch, "feature_stage": stage,
                "text_encoder_loaded": False, "checkpoint_sha256": checkpoint["_evaluation_sha256"],
                "source_checkpoint": os.environ.get("EVALUATION_SOURCE_WEIGHTS", str(weights)),
                "checkpoint_epoch": checkpoint.get("epoch"), "model_config": models.config,
                "evaluation_git_commit": revision, "arguments": vars(args), "geometry": geometry,
                "image_input": input_settings,
                "trace_components": component_settings(),
                "split_population_source": "explicit_evaluation_argument_not_checkpoint",
                "split_policy": "torch_60_20_20" if layout == "synthetic" and args.split != "all" else (
                    "balanced_page_pair_groups" if layout == "real" else "manifest_or_all"),
                "mask_metrics_are_character_alignment_accuracy": False}
    write_json(destination / "run.json", metadata)
    print(f"Yelda evaluation: branch={branch} stage={stage} split={args.split} pairs={len(selected)} text_encoder=none", flush=True)
    print(f"Image input: {args.image_preprocessing}; geometry={geometry['line_geometry_mode']}; "
          f"model_binarize={input_settings['effective_vit_binarize_input']}", flush=True)
    if args.representation == "joint":
        print(f"Joint scores: local={args.local_weight:.3f}, contextual={1-args.local_weight:.3f}; one NW alignment", flush=True)
    rows = []
    original_similarity_hook = base.compute_pair_similarity
    base.compute_pair_similarity = evaluation_similarity
    original_pair_hook = base.get_pair_image_features
    base.get_pair_image_features = lambda m, a, b: pair_features(m, a, b, args.representation)
    try:
        for pair in selected:
            output = destination / f"pair_{pair.index:05d}"
            try:
                row = evaluate_pair(base, models, pair, args, output)
                row.update(status="ok", output=str(output))
                print(f"[{pair.index}] NW={row['normalized_nw_score']:.4f} mask_IoU={row['mean_mask_iou']}", flush=True)
            except Exception as exc:
                row = {"index": pair.index, "pair_id": pair.pair_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                print(row["error"], file=sys.stderr, flush=True)
            rows.append(row)
    finally:
        base.get_pair_image_features = original_pair_hook
        base.compute_pair_similarity = original_similarity_hook
    with (destination / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items()} for row in rows)
    ok = [row for row in rows if row["status"] == "ok"]
    summary = dict(metadata, selected=len(rows), successful=len(ok), failed=len(rows)-len(ok))
    for metric in ("normalized_nw_score", "mean_path_cosine", "mean_mask_iou", "component_count"):
        summary["mean_" + metric.removeprefix("mean_")] = base._mean(ok, metric)
    for role in (1, 2):
        summary[f"line{role}_gt_count"] = sum(row.get(f"line{role}_mask_iou") is not None for row in ok)
        for metric in ("iou", "dice", "precision", "recall"):
            summary[f"mean_line{role}_mask_{metric}"] = base._mean(ok, f"line{role}_mask_{metric}")
    write_json(destination / "summary.json", summary)
    print(f"Saved {destination / 'summary.json'}", flush=True)
    return 0 if len(ok) == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
