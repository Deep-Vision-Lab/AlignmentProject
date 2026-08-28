#!/usr/bin/env python3
"""Needleman-Wunsch evaluation for real-style ViT line-pair datasets.

The global NW algorithm is unchanged.  Its traceback always starts at the
terminal DP boundary (N, M) and continues to (0, 0).  Supported diagonal matches
on that full global route are then split into disconnected mask components; up
to two unsupported trace/window steps are bridged by default.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_line_geometry import install_evaluation_geometry
install_evaluation_geometry()

from Evaluation.vit_evaluation import install_vit_evaluation_loader
install_vit_evaluation_loader()

from Evaluation.zero_shot_sw import install_dataset_patches, ink_aware_match_scores
install_dataset_patches()

from Evaluation._eval_utils import (
    compute_similarity,
    get_image_features,
    load_evaluation_models,
    needleman_wunsch,
)
from Evaluation.sw_core import build_match_scores, resolve_score_mode
from Evaluation.sw_dataset import (
    arabic_manifest_path,
    batch_pairs,
    display_image,
    load_arabic_dataset_pairs,
    load_pair_manifest,
    pair_for_index,
)
from Evaluation.trace_components import (
    component_intervals_px,
    component_metrics,
    nw_component_path,
    nw_traceback_boundaries,
    save_alignment_visualization,
    save_numeric_evidence,
)


def _matrix_stats(matrix: np.ndarray, prefix: str) -> dict:
    values = np.asarray(matrix, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if not finite.size:
        return {
            f"{prefix}_min": None,
            f"{prefix}_max": None,
            f"{prefix}_mean": None,
            f"{prefix}_std": None,
        }
    return {
        f"{prefix}_min": float(finite.min()),
        f"{prefix}_max": float(finite.max()),
        f"{prefix}_mean": float(finite.mean()),
        f"{prefix}_std": float(finite.std()),
    }


def _heatmap_matrix(raw_similarity, match_scores, result, source: str):
    value = str(source).strip().lower()
    if value == "cosine":
        return np.asarray(raw_similarity, dtype=np.float32), "raw cosine similarity"
    if value == "match-score":
        return np.asarray(match_scores, dtype=np.float32), "NW diagonal match score"
    if value == "dp-score":
        return (
            np.asarray(result.score_matrix[1:, 1:], dtype=np.float32),
            "accumulated global NW DP score",
        )
    raise ValueError("heatmap source must be cosine, match-score, or dp-score")


def evaluate_sample(
    models,
    pair,
    *,
    dataset_type: str,
    feature: str,
    threshold: float,
    gap: float,
    score_mode: str,
    score_clip: float,
    heatmap_source: str,
    output: Path,
):
    image1, image2 = Path(pair.image1), Path(pair.image2)
    if not image1.is_file() or not image2.is_file():
        missing = [str(path) for path in (image1, image2) if not path.is_file()]
        raise FileNotFoundError("Missing image pair: " + ", ".join(missing))

    temporary_directory = None
    try:
        if str(dataset_type).lower() == "real":
            arr1 = display_image(image1, "real")
            arr2 = display_image(image2, "real")
            temporary_directory = tempfile.TemporaryDirectory(prefix="nw_real_binary_")
            root = Path(temporary_directory.name)
            model_image1, model_image2 = root / "line1.png", root / "line2.png"
            Image.fromarray(arr1).save(model_image1)
            Image.fromarray(arr2).save(model_image2)
            feature_dataset_type = "synthetic"
        else:
            with Image.open(image1) as opened:
                arr1 = np.asarray(opened.convert("RGB"))
            with Image.open(image2) as opened:
                arr2 = np.asarray(opened.convert("RGB"))
            model_image1, model_image2 = image1, image2
            feature_dataset_type = "synthetic"

        features1 = get_image_features(models, model_image1, feature_dataset_type)
        features2 = get_image_features(models, model_image2, feature_dataset_type)
        raw_similarity = compute_similarity(
            features1.select(feature), features2.select(feature)
        ).cpu().numpy()

        resolved_mode = resolve_score_mode(score_mode, dataset_type)
        match_scores = build_match_scores(
            raw_similarity, resolved_mode, score_clip, threshold
        )
        match_scores = ink_aware_match_scores(
            match_scores,
            features1.ink.detach().cpu().numpy(),
            features2.ink.detach().cpu().numpy(),
        )

        # Unchanged GLOBAL Needleman-Wunsch.  Unlike SW, the final score and
        # traceback are anchored at the terminal DP state (N,M), not argmax(DP).
        result = needleman_wunsch(
            match_scores,
            gap_penalty=float(gap),
            similarity_offset=0.0,
        )
        full_path = list(result.pairs)
        global_traceback = nw_traceback_boundaries(result)
        component_path = nw_component_path(result, match_scores)

        heatmap, heatmap_label = _heatmap_matrix(
            raw_similarity, match_scores, result, heatmap_source
        )
        window_size = getattr(models.image_model, "window_size", None)
        stride = getattr(models.image_model, "stride", None)
        intervals1 = component_intervals_px(
            component_path, 0, raw_similarity.shape[0], arr1.shape[1],
            bool(models.image_model.use_flip), window_size=window_size, stride=stride,
        )
        intervals2 = component_intervals_px(
            component_path, 1, raw_similarity.shape[1], arr2.shape[1],
            bool(models.image_model.use_flip), window_size=window_size, stride=stride,
        )

        save_alignment_visualization(
            arr1=arr1,
            arr2=arr2,
            features1=features1,
            features2=features2,
            full_path=full_path,
            component_path=component_path,
            traceback=global_traceback,
            heatmap_matrix=heatmap,
            heatmap_label=heatmap_label,
            score=float(result.score),
            normalized_score=float(result.normalized_score),
            output=output,
            use_flip=bool(models.image_model.use_flip),
            pair=pair,
            score_mode=resolved_mode + "+ink",
            algorithm="Needleman-Wunsch",
            traceback_label="NW traceback: terminal (N,M) → origin (0,0)",
            traceback_start_label="terminal DP boundary (N,M)",
            traceback_end_label="global origin (0,0)",
            binarized=str(dataset_type).lower() == "real",
            annotate_values=os.environ.get("ANNOTATE_HEATMAP_VALUES", "0").lower()
            in {"1", "true", "yes", "on"},
            window_size=window_size,
            stride=stride,
        )

        evidence_files = save_numeric_evidence(
            output,
            algorithm="Needleman-Wunsch",
            raw_similarity=raw_similarity,
            match_scores=match_scores,
            component_path=component_path,
            full_path=full_path,
            traceback=global_traceback,
            intervals1=intervals1,
            intervals2=intervals2,
        )

        component_path_cosines = [
            float(raw_similarity[i, j]) for i, j in component_path
        ]
        full_path_cosines = [float(raw_similarity[i, j]) for i, j in full_path]
        gap_steps = sum(
            1 for step in result.steps
            if step.index1 is None or step.index2 is None
        )
        component_score = float(
            sum(max(0.0, float(match_scores[i, j])) for i, j in component_path)
        )
        metrics = component_metrics(component_path, raw_similarity.shape)
        row = {
            "index": int(pair.index),
            "manifest_position": int(pair.manifest_position),
            "pair_id": pair.pair_id,
            "label_type": pair.label_type,
            "text_score": float(pair.text_score),
            "split": pair.split,
            "status": "ok",
            "nw_score": float(result.score),
            "normalized_nw_score": float(result.normalized_score),
            **metrics,
            "match_steps": int(len(full_path)),
            "gap_steps": int(gap_steps),
            "component_score": component_score,
            "mean_path_cosine": (
                float(np.mean(component_path_cosines)) if component_path_cosines else 0.0
            ),
            "mean_full_path_cosine": (
                float(np.mean(full_path_cosines)) if full_path_cosines else 0.0
            ),
            "positive_match_fraction": float(np.mean(match_scores > 0.0)),
            "line1_windows": int(raw_similarity.shape[0]),
            "line2_windows": int(raw_similarity.shape[1]),
            "line1_component_intervals_px": json.dumps(intervals1, separators=(",", ":")),
            "line2_component_intervals_px": json.dumps(intervals2, separators=(",", ":")),
            **_matrix_stats(raw_similarity, "cosine"),
            **_matrix_stats(match_scores, "match"),
            **evidence_files,
            "feature": str(feature),
            "score_mode": resolved_mode,
            "score_clip": float(score_clip),
            "threshold": float(threshold),
            "gap": float(gap),
            "heatmap_source": str(heatmap_source),
            "dataset_type": str(dataset_type),
            "binarized": str(dataset_type).lower() == "real",
            "binarization": (
                os.environ.get("REAL_BINARIZE_METHOD", "otsu").lower()
                if str(dataset_type).lower() == "real" else "none"
            ),
            "flipped": bool(models.image_model.use_flip),
            "traceback_start": "terminal_nm",
            "traceback_end": "origin_00",
            "image1": str(image1),
            "image2": str(image2),
            "output": str(output),
            "error": "",
        }
        print(
            f"[{pair.index}] pair_id={pair.pair_id or '-'} "
            f"NW={result.normalized_score:.4f} trace=(N,M)->(0,0) "
            f"components={row['component_count']} "
            f"supported={row['path_steps']}/{row['full_match_steps']} "
            f"gaps={gap_steps} bridge<={row['component_bridge_limit']} "
            f"cos={row['mean_path_cosine']:.4f} saved={output}",
            flush=True,
        )
        return row
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--pair-manifest")
    parser.add_argument("--arabic-manifest")
    parser.add_argument("--real-labels", default=None)
    parser.add_argument("--real-min-text-score", type=float, default=0.0)
    parser.add_argument(
        "--real-text-key",
        default=os.environ.get("REAL_TEXT_KEY", "text_original_path"),
    )
    parser.add_argument(
        "--real-split", choices=("all", "train", "valid", "test"), default="test"
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=int(os.environ.get("DATASET_SPLIT_SEED", "42")),
    )
    parser.add_argument("--real-validate-paths", action="store_true")
    parser.add_argument("--image1-pattern", default="img1_{index}.png")
    parser.add_argument("--image2-pattern", default="img2_{index}.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-type", choices=("synthetic", "real"), default="real")
    parser.add_argument(
        "--feature", choices=("contextual", "local", "grouped"), default="contextual"
    )
    parser.add_argument(
        "--score-mode", choices=("auto", "raw", "centered", "mutual-z"), default="auto"
    )
    parser.add_argument("--score-clip", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--gap", type=float, default=-0.30)
    parser.add_argument(
        "--heatmap-source",
        choices=("dp-score", "match-score", "cosine"),
        default="match-score",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be greater than zero")

    manifest_pairs = []
    manifest = arabic_manifest_path(args)
    if args.dataset_type == "real" and not args.pair_manifest and manifest.is_file():
        manifest_pairs = load_arabic_dataset_pairs(args)
        print(
            f"Loaded real-style manifest: {manifest} split={args.real_split} "
            f"samples={len(manifest_pairs)}",
            flush=True,
        )
    elif args.pair_manifest:
        manifest_pairs = load_pair_manifest(args.pair_manifest, args.data_dir)

    models = load_evaluation_models(args.weights, args.device, load_text_model=False)
    selected = batch_pairs(args, manifest_pairs) if args.batch else [
        pair_for_index(args, args.start_index, manifest_pairs)
    ]
    if not selected:
        raise SystemExit("No image pairs were selected for evaluation")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in selected:
        output = output_dir / f"pair_{pair.index}.png"
        try:
            row = evaluate_sample(
                models,
                pair,
                dataset_type=args.dataset_type,
                feature=args.feature,
                threshold=args.threshold,
                gap=args.gap,
                score_mode=args.score_mode,
                score_clip=args.score_clip,
                heatmap_source=args.heatmap_source,
                output=output,
            )
        except Exception as exc:
            row = {
                "index": int(pair.index),
                "manifest_position": int(pair.manifest_position),
                "pair_id": pair.pair_id,
                "label_type": pair.label_type,
                "text_score": float(pair.text_score),
                "split": pair.split,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "image1": str(pair.image1),
                "image2": str(pair.image2),
                "output": str(output),
            }
            print(f"[{pair.index}] failed: {row['error']}", file=sys.stderr, flush=True)
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    successful = [row for row in rows if row.get("status") == "ok"]

    def mean(key):
        values = []
        for row in successful:
            try:
                value = float(row.get(key))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                values.append(value)
        return float(np.mean(values)) if values else None

    summary = {
        "algorithm": "needleman_wunsch",
        "traceback": "terminal_(N,M)_to_origin_(0,0)",
        "samples": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "mean_normalized_nw_score": mean("normalized_nw_score"),
        "mean_component_count": mean("component_count"),
        "mean_supported_component_matches": mean("path_steps"),
        "mean_full_nw_matches": mean("full_match_steps"),
        "mean_component_span_fraction_line1": mean("line1_matched_fraction"),
        "mean_component_span_fraction_line2": mean("line2_matched_fraction"),
        "mean_path_cosine": mean("mean_path_cosine"),
        "mean_full_path_cosine": mean("mean_full_path_cosine"),
        "mean_positive_match_fraction": mean("positive_match_fraction"),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
