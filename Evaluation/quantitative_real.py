#!/usr/bin/env python3
"""Quantitative evaluation for real manuscript alignment without dense masks.

The benchmark provides external targets in three ways:

1. Real-derived crop localization: crop a known ink region from a preprocessed
   real line, apply a controlled degradation, and localize it back in the full
   line with Smith-Waterman.
2. Real pair retrieval/discrimination: rank each true partner against lines
   from other pair IDs and evaluate pair scores with calibration/test separation.
3. Optional sparse intervals: evaluate manually annotated horizontal intervals
   stored in CSV/JSON/JSONL on the 1024-pixel evaluation canvas.

Outputs are CSV files, summary.json, and report.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


CANVAS_WIDTH = 1024
CANVAS_HEIGHT = 128


def _install_runtime():
    """Install checkpoint/backend and real-preprocessing patches before imports."""
    from unified_line_geometry import install_evaluation_geometry
    install_evaluation_geometry()

    from Evaluation.zero_shot_sw import install_dataset_patches

    install_dataset_patches()

    from Evaluation import _eval_utils, sw_core, sw_dataset, zero_shot_sw

    return SimpleNamespace(
        utils=_eval_utils,
        core=sw_core,
        dataset=sw_dataset,
        zero=zero_shot_sw,
    )


def _csv_values(value: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(value).split(",") if item.strip()]


def _json_ready(value):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _mean(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.mean(clean)) if clean else None


def _median(values: Iterable[float]) -> float | None:
    clean = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(clean)) if clean else None


def _interval_iou(pred: tuple[float, float], truth: tuple[float, float]) -> float:
    intersection = max(0.0, min(pred[1], truth[1]) - max(pred[0], truth[0]))
    union = max(pred[1], truth[1]) - min(pred[0], truth[0])
    return float(intersection / union) if union > 0.0 else 0.0


def _ink_mask(array: np.ndarray) -> np.ndarray:
    gray = np.asarray(Image.fromarray(array).convert("L"), dtype=np.float32)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    background = float(np.median(border))
    return np.abs(gray - background) >= 32.0


def _degrade_crop(crop: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    image = Image.fromarray(crop).convert("RGB")
    mode = mode.strip().lower()
    if mode in {"", "none", "identity"}:
        return np.asarray(image)
    if mode == "blur":
        return np.asarray(image.filter(ImageFilter.GaussianBlur(radius=1.2)))
    if mode == "contrast":
        return np.asarray(ImageEnhance.Contrast(image).enhance(0.65))
    if mode == "brightness":
        return np.asarray(ImageEnhance.Brightness(image).enhance(0.78))
    if mode == "morphology":
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
        border = np.median(np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1])))
        working = Image.fromarray(255 - gray if border > 127 else gray)
        working = working.filter(ImageFilter.MaxFilter(size=3))
        result = np.asarray(working, dtype=np.uint8)
        if border > 127:
            result = 255 - result
        return np.repeat(result[:, :, None], 3, axis=2)
    if mode == "noise":
        values = np.asarray(image, dtype=np.float32)
        noise = rng.normal(0.0, 10.0, size=values.shape)
        return np.clip(values + noise, 0, 255).astype(np.uint8)
    raise ValueError(
        f"Unsupported degradation {mode!r}; use none,blur,contrast,brightness,noise,morphology"
    )


def _crop_start(ink: np.ndarray, crop_width: int, rng: np.random.Generator) -> int:
    width = int(ink.shape[1])
    crop_width = max(8, min(int(crop_width), width))
    column_ink = ink.sum(axis=0).astype(np.float64)
    rolling = np.convolve(column_ink, np.ones(crop_width, dtype=np.float64), mode="valid")
    if not rolling.size:
        return 0
    cutoff = float(np.quantile(rolling, 0.70))
    candidates = np.flatnonzero(rolling >= max(cutoff, 1.0))
    if not candidates.size:
        return int(np.argmax(rolling))
    return int(rng.choice(candidates))


def _padded_query(crop: np.ndarray, background: int = 255) -> np.ndarray:
    height, width = crop.shape[:2]
    canvas = np.full((CANVAS_HEIGHT, CANVAS_WIDTH, 3), int(background), dtype=np.uint8)
    if height != CANVAS_HEIGHT:
        crop = np.asarray(Image.fromarray(crop).resize((width, CANVAS_HEIGHT), Image.BILINEAR))
    width = min(width, CANVAS_WIDTH)
    x0 = (CANVAS_WIDTH - width) // 2
    canvas[:, x0 : x0 + width] = crop[:, :width, :3]
    return canvas


def _load_pairs(runtime, args, split: str | None = None) -> list:
    loader_args = SimpleNamespace(
        data_dir=args.real_data_dir,
        arabic_manifest=args.arabic_manifest,
        real_text_key=args.real_text_key,
        real_labels=args.labels,
        real_min_text_score=args.real_min_text_score,
        real_validate_paths=True,
        split_seed=args.split_seed,
        real_split=split or args.real_split,
    )
    pairs = runtime.dataset.load_arabic_dataset_pairs(loader_args)
    if not pairs:
        raise RuntimeError("No real pairs matched the requested split and labels")
    return pairs


def _alignment(runtime, left, right, args) -> dict:
    similarity = runtime.utils.compute_similarity(
        left.select(args.feature), right.select(args.feature)
    ).detach().cpu().numpy()
    score_mode = runtime.core.resolve_score_mode(args.score_mode, "real")
    match_scores = runtime.core.build_match_scores(
        similarity, score_mode, args.score_clip, args.threshold
    )
    match_scores = runtime.zero.ink_aware_match_scores(
        match_scores,
        left.ink.detach().cpu().numpy(),
        right.ink.detach().cpu().numpy(),
    )
    path, score, dp_score, traceback = runtime.core.smith_waterman(
        similarity,
        threshold=args.threshold,
        gap_penalty=args.gap,
        return_traceback=True,
        match_scores=match_scores,
    )
    region = runtime.core.dense_alignment_region(path, traceback)
    path_cosines = [float(similarity[i, j]) for i, j in path]
    traceback_steps = max(0, len(traceback) - 1)
    normalized_score = float(score) / max(1, traceback_steps)
    mean_cosine = float(np.mean(path_cosines)) if path_cosines else -1.0
    return {
        "path": path,
        "traceback": traceback,
        "region": region,
        "score": float(score),
        "normalized_score": normalized_score,
        "mean_path_cosine": mean_cosine,
        "hybrid_score": normalized_score + mean_cosine,
        "line1_windows": int(similarity.shape[0]),
        "line2_windows": int(similarity.shape[1]),
        "score_mode": score_mode + "+ink",
    }


def _ranking_score(alignment: dict, mode: str) -> float:
    mode = mode.lower()
    if mode == "normalized_sw":
        return float(alignment["normalized_score"])
    if mode == "mean_cosine":
        return float(alignment["mean_path_cosine"])
    if mode == "hybrid":
        return float(alignment["hybrid_score"])
    raise ValueError("ranking_score must be normalized_sw, mean_cosine, or hybrid")


def _feature_cache(runtime, models, dataset_type: str, temp_root: Path | None = None):
    cache: dict[tuple[str, str], object] = {}
    prepared: dict[str, Path] = {}

    def get(path: str | Path):
        resolved = str(Path(path).resolve())
        key = (resolved, dataset_type)
        if key not in cache:
            feature_path = Path(path)
            feature_dataset_type = dataset_type
            if dataset_type == "real":
                if temp_root is None:
                    raise ValueError("temp_root is required for real feature caching")
                if resolved not in prepared:
                    array = runtime.dataset.display_image(feature_path, "real")
                    prepared_path = temp_root / f"real_{len(prepared):06d}.png"
                    Image.fromarray(array).save(prepared_path)
                    prepared[resolved] = prepared_path
                feature_path = prepared[resolved]
                feature_dataset_type = "synthetic"
            cache[key] = runtime.utils.get_image_features(
                models, feature_path, feature_dataset_type
            )
        return cache[key]

    return get


def run_crop_localization(runtime, models, pairs, args, output: Path) -> tuple[list[dict], dict]:
    rng = np.random.default_rng(args.seed)
    line_paths = []
    seen = set()
    for pair in pairs:
        for path in (pair.image1, pair.image2):
            resolved = str(Path(path).resolve())
            if resolved not in seen:
                seen.add(resolved)
                line_paths.append(Path(path))
    random.Random(args.seed).shuffle(line_paths)
    line_paths = line_paths[: args.crop_lines]
    fractions = _csv_values(args.crop_fractions, float)
    degradations = _csv_values(args.degradations, str)
    if not fractions or not degradations:
        raise ValueError("crop_fractions and degradations must not be empty")

    stride = int(models.config.get("stride", 16))
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="real_crop_quant_") as temp_dir:
        temp_root = Path(temp_dir)
        real_features = _feature_cache(runtime, models, "real", temp_root)
        example_id = 0
        crop_id = 0
        for line_index, line_path in enumerate(line_paths, start=1):
            array = runtime.dataset.display_image(line_path, "real")
            _height, width = array.shape[:2]
            ink = _ink_mask(array)
            border_values = np.concatenate(
                (array[0].reshape(-1), array[-1].reshape(-1))
            )
            background = int(np.median(border_values))
            target_features = real_features(line_path)
            for local_index in range(args.crops_per_line):
                crop_id += 1
                fraction = fractions[local_index % len(fractions)]
                crop_width = max(16, int(round(width * fraction)))
                gt_start = _crop_start(ink, crop_width, rng)
                gt_end = min(width, gt_start + crop_width)
                clean_crop = array[:, gt_start:gt_end]
                for degradation in degradations:
                    example_id += 1
                    crop = _degrade_crop(clean_crop, degradation, rng)
                    query = _padded_query(crop, background=background)
                    query_path = temp_root / f"query_{example_id:06d}.png"
                    Image.fromarray(query).save(query_path)
                    query_features = runtime.utils.get_image_features(
                        models, query_path, "synthetic"
                    )
                    aligned = _alignment(
                        runtime, query_features, target_features, args
                    )
                    region = aligned["region"]
                    if region.empty:
                        pred_start = pred_end = 0.0
                    else:
                        pred_start, pred_end = runtime.utils.patch_range_to_pixels(
                            region.line2_start,
                            region.line2_end + 1,
                            aligned["line2_windows"],
                            width,
                            bool(models.image_model.use_flip),
                        )
                    iou = _interval_iou(
                        (pred_start, pred_end), (gt_start, gt_end)
                    )
                    boundary_error = (
                        abs(pred_start - gt_start) + abs(pred_end - gt_end)
                    ) / 2.0
                    center_error = abs(
                        (pred_start + pred_end - gt_start - gt_end) / 2.0
                    )
                    rows.append(
                        {
                            "example_id": example_id,
                            "crop_id": crop_id,
                            "line_index": line_index,
                            "image": str(line_path),
                            "crop_fraction": float(fraction),
                            "degradation": degradation,
                            "gt_start_px": float(gt_start),
                            "gt_end_px": float(gt_end),
                            "pred_start_px": float(pred_start),
                            "pred_end_px": float(pred_end),
                            "interval_iou": iou,
                            "boundary_mae_px": float(boundary_error),
                            "boundary_mae_windows": float(boundary_error / max(1, stride)),
                            "boundary_mae_normalized": float(boundary_error / max(1, width)),
                            "center_error_px": float(center_error),
                            "center_error_windows": float(center_error / max(1, stride)),
                            "center_error_normalized": float(center_error / max(1, width)),
                            "sw_score": aligned["score"],
                            "normalized_sw_score": aligned["normalized_score"],
                            "mean_path_cosine": aligned["mean_path_cosine"],
                            "path_steps": int(region.path_steps),
                            "target_windows": aligned["line2_windows"],
                            "score_mode": aligned["score_mode"],
                        }
                    )
                    print(
                        f"crop {example_id}: iou={iou:.3f} "
                        f"boundary={boundary_error:.1f}px "
                        f"fraction={fraction:.2f} degradation={degradation}",
                        flush=True,
                    )

    def summarize(items):
        return {
            "examples": len(items),
            "unique_crops": len({row["crop_id"] for row in items}),
            "mean_iou": _mean(row["interval_iou"] for row in items),
            "median_iou": _median(row["interval_iou"] for row in items),
            "success_iou_030": _mean(row["interval_iou"] >= 0.30 for row in items),
            "success_iou_050": _mean(row["interval_iou"] >= 0.50 for row in items),
            "success_iou_070": _mean(row["interval_iou"] >= 0.70 for row in items),
            "mean_boundary_mae_px": _mean(row["boundary_mae_px"] for row in items),
            "mean_boundary_mae_windows": _mean(
                row["boundary_mae_windows"] for row in items
            ),
            "mean_center_error_px": _mean(row["center_error_px"] for row in items),
            "mean_center_error_windows": _mean(
                row["center_error_windows"] for row in items
            ),
        }

    summary = summarize(rows)
    summary["benchmark_type"] = "controlled_same_line_crop_relocalization"
    summary["cross_manuscript_localization"] = False
    summary["window_stride_px"] = stride
    summary["by_degradation"] = {
        mode: summarize([row for row in rows if row["degradation"] == mode])
        for mode in degradations
    }
    summary["by_crop_fraction"] = {
        str(fraction): summarize(
            [row for row in rows if abs(row["crop_fraction"] - fraction) < 1e-9]
        )
        for fraction in fractions
    }
    _write_csv(output / "crop_localization.csv", rows)
    return rows, summary

def _roc_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = [score for label, score in zip(labels, scores) if int(label) == 1]
    negatives = [score for label, score in zip(labels, scores) if int(label) == 0]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += 1.0 if positive > negative else 0.5 if positive == negative else 0.0
    return float(wins / (len(positives) * len(negatives)))


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    positives = sum(int(label) == 1 for label in labels)
    if positives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    hit = 0
    total = 0.0
    for rank, index in enumerate(order, start=1):
        if int(labels[index]) == 1:
            hit += 1
            total += hit / rank
    return float(total / positives)


def _classification_at_threshold(labels, scores, threshold: float) -> dict:
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        predicted = float(score) >= float(threshold)
        if int(label) == 1 and predicted:
            tp += 1
        elif int(label) == 0 and predicted:
            fp += 1
        elif int(label) == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / max(1, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def _best_f1_threshold(labels, scores) -> float:
    if not scores:
        return 0.0
    unique = sorted(set(float(value) for value in scores))
    candidates = [unique[0] - 1e-6, *unique, unique[-1] + 1e-6]
    best = max(candidates, key=lambda threshold: _classification_at_threshold(labels, scores, threshold)["f1"])
    return float(best)


def _equal_error_rate(labels: Sequence[int], scores: Sequence[float]) -> tuple[float | None, float | None]:
    if not labels or len(set(int(v) for v in labels)) < 2:
        return None, None
    unique = sorted(set(float(value) for value in scores))
    candidates = [unique[0] - 1e-6, *unique, unique[-1] + 1e-6]
    best = None
    for threshold in candidates:
        stats = _classification_at_threshold(labels, scores, threshold)
        fp, tn = stats["fp"], stats["tn"]
        fn, tp = stats["fn"], stats["tp"]
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)
        candidate = (abs(fpr - fnr), 0.5 * (fpr + fnr), float(threshold))
        if best is None or candidate < best:
            best = candidate
    return float(best[1]), float(best[2])


def _score_candidate_pools(
    runtime,
    models,
    pairs,
    args,
    *,
    seed_offset: int,
    query_limit: int,
    label: str,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(args.seed + seed_offset)
    selected = list(pairs)
    rng.shuffle(selected)
    selected = selected[: min(query_limit, len(selected))]
    retrieval_rows: list[dict] = []
    score_rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix=f"real_{label}_quant_") as temp_dir:
        get_features = _feature_cache(runtime, models, "real", Path(temp_dir))
        for query_order, pair in enumerate(selected, start=1):
            negatives = [
                item
                for item in pairs
                if item.pair_id != pair.pair_id
                and Path(item.image2).resolve() != Path(pair.image2).resolve()
            ]
            rng.shuffle(negatives)
            pool = [pair, *negatives[: max(0, args.retrieval_pool_size - 1)]]
            rng.shuffle(pool)
            query_features = get_features(pair.image1)
            candidates = []
            for candidate in pool:
                target_features = get_features(candidate.image2)
                aligned = _alignment(runtime, query_features, target_features, args)
                rank_score = _ranking_score(aligned, args.ranking_score)
                is_positive = int(candidate.pair_id == pair.pair_id)
                candidates.append((rank_score, is_positive, candidate, aligned))
                score_rows.append(
                    {
                        "split_role": label,
                        "query_order": query_order,
                        "query_pair_id": pair.pair_id,
                        "query_label_type": pair.label_type,
                        "candidate_pair_id": candidate.pair_id,
                        "is_positive": is_positive,
                        "ranking_score": rank_score,
                        "normalized_sw_score": aligned["normalized_score"],
                        "mean_path_cosine": aligned["mean_path_cosine"],
                        "hybrid_score": aligned["hybrid_score"],
                        "path_steps": int(aligned["region"].path_steps),
                        "query_image": str(pair.image1),
                        "candidate_image": str(candidate.image2),
                    }
                )
            candidates.sort(key=lambda item: item[0], reverse=True)
            positive_rank = next(
                (rank for rank, item in enumerate(candidates, start=1) if item[1] == 1),
                len(candidates) + 1,
            )
            retrieval_rows.append(
                {
                    "split_role": label,
                    "query_order": query_order,
                    "query_pair_id": pair.pair_id,
                    "label_type": pair.label_type,
                    "pool_size": len(candidates),
                    "positive_rank": positive_rank,
                    "reciprocal_rank": 1.0 / positive_rank,
                    "recall_at_1": int(positive_rank <= 1),
                    "recall_at_5": int(positive_rank <= 5),
                    "recall_at_10": int(positive_rank <= 10),
                    "average_precision": 1.0 / positive_rank,
                    "query_image": str(pair.image1),
                    "true_candidate": str(pair.image2),
                }
            )
            print(
                f"{label} retrieval {query_order}: rank={positive_rank}/{len(candidates)} "
                f"pair_id={pair.pair_id}",
                flush=True,
            )
    return retrieval_rows, score_rows


def _retrieval_summary(rows: Sequence[dict]) -> dict:
    return {
        "queries": len(rows),
        "recall_at_1": _mean(row["recall_at_1"] for row in rows),
        "recall_at_5": _mean(row["recall_at_5"] for row in rows),
        "recall_at_10": _mean(row["recall_at_10"] for row in rows),
        "mrr": _mean(row["reciprocal_rank"] for row in rows),
        "map": _mean(row["average_precision"] for row in rows),
        "mean_pool_size": _mean(row["pool_size"] for row in rows),
    }


def run_retrieval(
    runtime,
    models,
    pairs,
    calibration_pairs,
    calibration_split: str,
    args,
    output: Path,
) -> tuple[list[dict], list[dict], dict]:
    retrieval_rows, score_rows = _score_candidate_pools(
        runtime,
        models,
        pairs,
        args,
        seed_offset=17,
        query_limit=args.retrieval_queries,
        label="test",
    )
    calibration_query_limit = min(args.calibration_queries, len(calibration_pairs))
    _calibration_retrieval, calibration_scores_rows = _score_candidate_pools(
        runtime,
        models,
        calibration_pairs,
        args,
        seed_offset=1017,
        query_limit=calibration_query_limit,
        label=f"calibration_{calibration_split}",
    )

    test_labels = [int(row["is_positive"]) for row in score_rows]
    test_scores = [float(row["ranking_score"]) for row in score_rows]
    calibration_labels = [int(row["is_positive"]) for row in calibration_scores_rows]
    calibration_scores = [float(row["ranking_score"]) for row in calibration_scores_rows]
    if len(set(calibration_labels)) < 2:
        raise RuntimeError(
            "Validation calibration must contain both positive and negative comparisons"
        )

    threshold = _best_f1_threshold(calibration_labels, calibration_scores)
    eer, eer_threshold = _equal_error_rate(test_labels, test_scores)
    discrimination = {
        "threshold_source": f"{calibration_split}_split",
        "calibration_split": calibration_split,
        "calibration_queries": calibration_query_limit,
        "calibration_comparisons": len(calibration_labels),
        "test_comparisons": len(test_labels),
        "auroc": _roc_auc(test_labels, test_scores),
        "auprc": _average_precision(test_labels, test_scores),
        "average_precision": _average_precision(test_labels, test_scores),
        "equal_error_rate": eer,
        "equal_error_rate_threshold": eer_threshold,
        **_classification_at_threshold(test_labels, test_scores, threshold),
    }

    by_label = {}
    for label_type in sorted({str(row.get("label_type", "")) for row in retrieval_rows}):
        if not label_type:
            continue
        label_retrieval = [row for row in retrieval_rows if str(row.get("label_type")) == label_type]
        label_scores_rows = [
            row for row in score_rows if str(row.get("query_label_type")) == label_type
        ]
        label_labels = [int(row["is_positive"]) for row in label_scores_rows]
        label_scores = [float(row["ranking_score"]) for row in label_scores_rows]
        label_summary = _retrieval_summary(label_retrieval)
        label_summary["pair_discrimination"] = {
            "comparisons": len(label_labels),
            "auroc": _roc_auc(label_labels, label_scores),
            "auprc": _average_precision(label_labels, label_scores),
            **_classification_at_threshold(label_labels, label_scores, threshold),
        }
        by_label[label_type] = label_summary

    summary = {
        **_retrieval_summary(retrieval_rows),
        "requested_pool_size": args.retrieval_pool_size,
        "ranking_score": args.ranking_score,
        "pair_discrimination": discrimination,
        "by_label": by_label,
    }
    _write_csv(output / "retrieval_results.csv", retrieval_rows)
    _write_csv(output / "pair_classification.csv", score_rows)
    _write_csv(output / "validation_pair_classification.csv", calibration_scores_rows)
    return retrieval_rows, score_rows, summary

def _manifest_records(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("pairs", payload.get("annotations", payload))
    if not isinstance(payload, list):
        raise ValueError("Interval manifest must contain a list of records")
    return payload


def _resolve_annotation_path(value: str, manifest: Path, data_dir: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    for candidate in (manifest.parent / path, data_dir / path, data_dir / "images" / path):
        if candidate.is_file():
            return candidate
    return manifest.parent / path


def run_sparse_intervals(runtime, models, args, output: Path) -> tuple[list[dict], dict | None]:
    if not args.interval_manifest:
        return [], None
    manifest = Path(args.interval_manifest)
    records = _manifest_records(manifest)
    rows = []
    stride = int(models.config.get("stride", 16))
    with tempfile.TemporaryDirectory(prefix="real_sparse_quant_") as temp_dir:
        get_features = _feature_cache(runtime, models, "real", Path(temp_dir))
        for index, record in enumerate(records, start=1):
            image1 = _resolve_annotation_path(
                str(record["image1"]), manifest, Path(args.real_data_dir)
            )
            image2 = _resolve_annotation_path(
                str(record["image2"]), manifest, Path(args.real_data_dir)
            )
            aligned = _alignment(
                runtime, get_features(image1), get_features(image2), args
            )
            region = aligned["region"]
            if region.empty:
                pred1 = pred2 = (0.0, 0.0)
            else:
                pred1 = runtime.utils.patch_range_to_pixels(
                    region.line1_start,
                    region.line1_end + 1,
                    aligned["line1_windows"],
                    CANVAS_WIDTH,
                    bool(models.image_model.use_flip),
                )
                pred2 = runtime.utils.patch_range_to_pixels(
                    region.line2_start,
                    region.line2_end + 1,
                    aligned["line2_windows"],
                    CANVAS_WIDTH,
                    bool(models.image_model.use_flip),
                )
            gt1 = (
                float(record["line1_start_px"]),
                float(record["line1_end_px"]),
            )
            gt2 = (
                float(record["line2_start_px"]),
                float(record["line2_end_px"]),
            )
            iou1, iou2 = _interval_iou(pred1, gt1), _interval_iou(pred2, gt2)
            pair_mean_iou = 0.5 * (iou1 + iou2)
            joint = math.sqrt(max(0.0, iou1 * iou2))
            boundary_errors = [
                abs(pred1[0] - gt1[0]),
                abs(pred1[1] - gt1[1]),
                abs(pred2[0] - gt2[0]),
                abs(pred2[1] - gt2[1]),
            ]
            boundary_mae_px = float(np.mean(boundary_errors))
            center1 = abs(0.5 * sum(pred1) - 0.5 * sum(gt1))
            center2 = abs(0.5 * sum(pred2) - 0.5 * sum(gt2))
            center_error_px = 0.5 * (center1 + center2)
            rows.append(
                {
                    "index": index,
                    "pair_id": record.get("pair_id", index),
                    "image1": str(image1),
                    "image2": str(image2),
                    "line1_gt_start_px": gt1[0],
                    "line1_gt_end_px": gt1[1],
                    "line1_pred_start_px": pred1[0],
                    "line1_pred_end_px": pred1[1],
                    "line1_iou": iou1,
                    "line2_gt_start_px": gt2[0],
                    "line2_gt_end_px": gt2[1],
                    "line2_pred_start_px": pred2[0],
                    "line2_pred_end_px": pred2[1],
                    "line2_iou": iou2,
                    "mean_two_line_iou": pair_mean_iou,
                    "joint_iou_geometric": joint,
                    "boundary_mae_px": boundary_mae_px,
                    "boundary_mae_windows": boundary_mae_px / max(1, stride),
                    "center_error_px": center_error_px,
                    "center_error_windows": center_error_px / max(1, stride),
                    "both_iou_030": int(iou1 >= 0.30 and iou2 >= 0.30),
                    "both_iou_050": int(iou1 >= 0.50 and iou2 >= 0.50),
                    "both_iou_075": int(iou1 >= 0.75 and iou2 >= 0.75),
                    "normalized_sw_score": aligned["normalized_score"],
                    "mean_path_cosine": aligned["mean_path_cosine"],
                }
            )
    summary = {
        "benchmark_type": "sparse_manual_cross_line_intervals",
        "annotations": len(rows),
        "window_stride_px": stride,
        "mean_line1_iou": _mean(row["line1_iou"] for row in rows),
        "mean_line2_iou": _mean(row["line2_iou"] for row in rows),
        "mean_two_line_iou": _mean(row["mean_two_line_iou"] for row in rows),
        "mean_joint_iou_geometric": _mean(
            row["joint_iou_geometric"] for row in rows
        ),
        "mean_boundary_mae_px": _mean(row["boundary_mae_px"] for row in rows),
        "mean_boundary_mae_windows": _mean(
            row["boundary_mae_windows"] for row in rows
        ),
        "mean_center_error_px": _mean(row["center_error_px"] for row in rows),
        "both_success_iou_030": _mean(row["both_iou_030"] for row in rows),
        "both_success_iou_050": _mean(row["both_iou_050"] for row in rows),
        "both_success_iou_075": _mean(row["both_iou_075"] for row in rows),
        "coordinate_system": "preprocessed 1024x128 evaluation canvas",
    }
    _write_csv(output / "sparse_intervals.csv", rows)
    return rows, summary

def _format_metric(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(summary: dict, output: Path) -> None:
    crop = summary.get("crop_localization", {})
    retrieval = summary.get("retrieval", {})
    discrimination = retrieval.get("pair_discrimination", {})
    sparse = summary.get("sparse_intervals")
    cycle = summary.get("cycle_consistency", {})
    robustness = summary.get("robustness", {})
    lines = [
        "# Real quantitative alignment evaluation",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Backend: `{summary['model_backend']}`",
        f"- Split: `{summary['real_split']}`",
        f"- Labels: `{summary['labels']}`",
        "",
        "## Automatic crop localization",
        "",
        f"- Examples: {_format_metric(crop.get('examples'))}",
        f"- Mean interval IoU: {_format_metric(crop.get('mean_iou'))}",
        f"- Success@IoU 0.50: {_format_metric(crop.get('success_iou_050'))}",
        f"- Mean boundary MAE: {_format_metric(crop.get('mean_boundary_mae_px'))} px",
        "",
        "## Retrieval and pair discrimination",
        "",
        f"- Queries: {_format_metric(retrieval.get('queries'))}",
        f"- Recall@1: {_format_metric(retrieval.get('recall_at_1'))}",
        f"- Recall@5: {_format_metric(retrieval.get('recall_at_5'))}",
        f"- MRR: {_format_metric(retrieval.get('mrr'))}",
        f"- Pair AUROC: {_format_metric(discrimination.get('auroc'))}",
        f"- Pair average precision: {_format_metric(discrimination.get('average_precision'))}",
        f"- Thresholded F1: {_format_metric(discrimination.get('f1'))}",
        f"- Threshold source: {_format_metric(discrimination.get('threshold_source'))}",
        f"- Equal error rate: {_format_metric(discrimination.get('equal_error_rate'))}",
    ]
    if sparse:
        lines.extend(
            [
                "",
                "## Sparse real intervals",
                "",
                f"- Annotations: {_format_metric(sparse.get('annotations'))}",
                f"- Mean two-line IoU: {_format_metric(sparse.get('mean_two_line_iou'))}",
                f"- Geometric JointIoU: {_format_metric(sparse.get('mean_joint_iou_geometric'))}",
                f"- Both lines Success@0.50: {_format_metric(sparse.get('both_success_iou_050'))}",
            ]
        )
    lines.extend(
        [
            "",
            "## Cycle consistency (diagnostic only)",
            "",
            f"- Mean cycle error (windows): {_format_metric(cycle.get('mean_cycle_error_windows'))}",
            f"- Mean normalized cycle error: {_format_metric(cycle.get('mean_cycle_error_normalized'))}",
            f"- Return within 1/2/4 windows: "
            f"{_format_metric(cycle.get('within_1_window'))} / "
            f"{_format_metric(cycle.get('within_2_windows'))} / "
            f"{_format_metric(cycle.get('within_4_windows'))}",
            "",
            "## Perturbation stability (diagnostic only)",
            "",
            f"- Mean endpoint drift: {_format_metric(robustness.get('mean_endpoint_drift'))}",
            f"- Mean prediction interval IoU vs baseline: "
            f"{_format_metric(robustness.get('mean_prediction_interval_iou'))}",
            f"- SW-score coefficient of variation: "
            f"{_format_metric(robustness.get('mean_sw_score_coefficient_of_variation'))}",
            f"- Path-length coefficient of variation: "
            f"{_format_metric(robustness.get('mean_path_length_coefficient_of_variation'))}",
            "",
            "## Interpretation",
            "",
            "Controlled crop localization has exact coordinates but is same-line "
            "re-localization, not independent cross-manuscript localization. "
            "Retrieval/discrimination tests real-to-real pair recognition. Sparse "
            "manual intervals, when provided, are the strongest direct cross-line "
            "localization measure. Cycle consistency and perturbation stability are "
            "diagnostics only. Internal SW/NW scores and path cosine are not "
            "localization accuracy.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--real-data-dir", default="DataSet/ArabicDataset")
    parser.add_argument("--arabic-manifest", default="DataSet/ArabicDataset/dataset_manifest.jsonl")
    parser.add_argument("--real-split", choices=("train", "valid", "test", "all"), default="test")
    parser.add_argument("--labels", default="high_match,medium_match")
    parser.add_argument("--real-text-key", default="text_original_path")
    parser.add_argument("--real-min-text-score", type=float, default=0.0)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature", choices=("contextual", "local", "grouped"), default="contextual")
    parser.add_argument("--score-mode", default="auto")
    parser.add_argument("--score-clip", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--gap", type=float, default=-0.30)
    parser.add_argument("--crop-lines", type=int, default=80)
    parser.add_argument("--crops-per-line", type=int, default=5)
    parser.add_argument("--crop-fractions", default="0.10,0.20,0.30,0.40,0.50")
    parser.add_argument("--degradations", default="none,blur,contrast,noise,morphology")
    parser.add_argument("--retrieval-queries", type=int, default=80)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--calibration-queries", type=int, default=40,
                        help="Validation-split queries used only to choose the classification threshold")
    parser.add_argument(
        "--ranking-score",
        choices=("normalized_sw", "mean_cosine", "hybrid"),
        default="normalized_sw",
    )
    parser.add_argument("--interval-manifest", default="")
    parser.add_argument("--cycle-pairs", type=int, default=80)
    parser.add_argument("--robustness-pairs", type=int, default=30)
    parser.add_argument("--robustness-modes", default="blur,contrast,brightness,noise,horizontal_scale,vertical_shift,erosion,dilation")
    parser.add_argument("--min-ink", type=float, default=0.02)
    args = parser.parse_args()
    for name in ("crop_lines", "crops_per_line", "retrieval_queries", "retrieval_pool_size", "calibration_queries", "cycle_pairs", "robustness_pairs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runtime = _install_runtime()
    models = runtime.utils.load_evaluation_models(
        args.weights, device=args.device, load_text_model=False
    )
    pairs = _load_pairs(runtime, args)
    if args.real_split == "all":
        raise ValueError(
            "Quantitative pair discrimination requires a held-out split; "
            "use --real-split test (recommended), valid, or train instead of all."
        )
    calibration_split = "train" if args.real_split == "valid" else "valid"
    calibration_pairs = _load_pairs(runtime, args, split=calibration_split)
    print(
        f"quantitative_real backend={models.config.get('model_backend', 'cnn_bilstm')} "
        f"pairs={len(pairs)} split={args.real_split} labels={args.labels}",
        flush=True,
    )

    _crop_rows, crop_summary = run_crop_localization(runtime, models, pairs, args, output)
    _retrieval_rows, _score_rows, retrieval_summary = run_retrieval(
        runtime, models, pairs, calibration_pairs, calibration_split, args, output
    )
    _sparse_rows, sparse_summary = run_sparse_intervals(runtime, models, args, output)

    from Evaluation.quantitative_diagnostics import (
        run_cycle_consistency,
        run_robustness,
    )
    cycle_rows, cycle_summary = run_cycle_consistency(
        runtime, models, pairs, args, output
    )
    robustness_rows, robustness_summary = run_robustness(
        runtime, models, pairs, args, output
    )

    per_sample = []
    per_sample.extend({"benchmark": "controlled_crop", **row} for row in _crop_rows)
    per_sample.extend({"benchmark": "retrieval", **row} for row in _retrieval_rows)
    per_sample.extend({"benchmark": "sparse_interval", **row} for row in _sparse_rows)
    per_sample.extend({"benchmark": "cycle_consistency", **row} for row in cycle_rows)
    per_sample.extend({"benchmark": "robustness", **row} for row in robustness_rows)
    _write_csv(output / "per_sample.csv", per_sample)

    summary = {
        "checkpoint": str(Path(args.weights).resolve()),
        "model_backend": str(models.config.get("model_backend", "cnn_bilstm")),
        "real_split": args.real_split,
        "labels": args.labels,
        "available_pairs": len(pairs),
        "calibration_split": calibration_split,
        "calibration_pairs": len(calibration_pairs),
        "feature": args.feature,
        "score_mode": args.score_mode,
        "threshold": args.threshold,
        "gap": args.gap,
        "seed": args.seed,
        "input_policy": {
            "zero_shot_preprocess": str(os.environ.get("ZERO_SHOT_PREPROCESS", "")),
            "real_binarize": str(os.environ.get("REAL_BINARIZE", "")),
            "preserve_aspect": str(os.environ.get("ZERO_SHOT_PRESERVE_ASPECT", "")),
            "foreground_crop": str(os.environ.get("ZERO_SHOT_FOREGROUND_CROP", "")),
            "line_height": int(os.environ.get("LINE_HEIGHT", "128")),
            "line_width": int(os.environ.get("LINE_WIDTH", "1024")),
        },
        "crop_localization": crop_summary,
        "retrieval": retrieval_summary,
        "sparse_intervals": sparse_summary,
        "cycle_consistency": cycle_summary,
        "robustness": robustness_summary,
        "evaluation_claims": {
            "localization_accuracy_sources": ["controlled_crop", "sparse_intervals"],
            "pair_recognition_sources": ["retrieval", "pair_discrimination"],
            "diagnostic_only": ["cycle_consistency", "robustness", "mean_path_cosine", "normalized_sw_score"],
        },
    }
    (output / "summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(summary, output)
    print(f"Quantitative evaluation complete: {output}", flush=True)


if __name__ == "__main__":
    main()
