#!/usr/bin/env python3
"""Checkpoint-compatible Needleman-Wunsch evaluation for real-style line-pair datasets.

This is deliberately the global-alignment counterpart of ``eval_img_align_sw``.
It reuses the same checkpoint reconstruction, real-image preprocessing, manifest
splitting, feature extraction, score normalization, and ink-aware suppression so
SW and NW can be compared on exactly the same visual evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
from Evaluation.sw_core import ImagePair, build_match_scores, resolve_score_mode
from Evaluation.sw_dataset import (
    arabic_manifest_path,
    batch_pairs,
    display_image,
    load_arabic_dataset_pairs,
    load_pair_manifest,
    pair_for_index,
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
        return np.asarray(match_scores, dtype=np.float32), "NW match score"
    if value == "dp-score":
        return np.asarray(result.score_matrix[1:, 1:], dtype=np.float32), "accumulated global NW DP score"
    raise ValueError("heatmap source must be cosine, match-score, or dp-score")


def _save_visualization(
    arr1,
    arr2,
    heatmap,
    heatmap_label,
    result,
    output: Path,
    pair: ImagePair,
    score_mode: str,
):
    n1, n2 = heatmap.shape
    heatmap_height = max(8.0, min(24.0, 0.30 * n1))
    figure_width = max(18.0, min(30.0, 0.34 * n2))
    fig = plt.figure(figsize=(figure_width, 5.0 + heatmap_height))
    grid = fig.add_gridspec(3, 1, height_ratios=[2.2, 2.2, heatmap_height], hspace=0.16)

    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    axes[0].imshow(arr1, aspect="auto")
    axes[1].imshow(arr2, aspect="auto")
    axes[0].set_ylabel("line A", rotation=0, labelpad=35, va="center")
    axes[1].set_ylabel("line B", rotation=0, labelpad=35, va="center")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    ax = fig.add_subplot(grid[2])
    finite = heatmap[np.isfinite(heatmap)]
    if finite.size and float(finite.min()) >= 0.0:
        upper = max(1e-6, float(np.percentile(finite, 98)))
        image = ax.imshow(
            heatmap,
            aspect="equal",
            origin="upper",
            vmin=0.0,
            vmax=upper,
            cmap="viridis",
            interpolation="nearest",
        )
    else:
        limit = max(
            0.5,
            float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0,
        )
        image = ax.imshow(
            heatmap,
            aspect="equal",
            origin="upper",
            vmin=-limit,
            vmax=limit,
            cmap="coolwarm",
            interpolation="nearest",
        )

    path = result.pairs
    if path:
        ax.plot(
            [col for _, col in path],
            [row for row, _ in path],
            color="black",
            linewidth=1.4,
            marker=".",
            markersize=2.5,
            label="global NW path",
        )
        ax.legend(loc="upper left", fontsize=8)

    ax.set_xlim(-0.5, n2 - 0.5)
    ax.set_ylim(n1 - 0.5, -0.5)
    ax.set_xlabel("line B windows")
    ax.set_ylabel("line A windows")
    ax.set_title(heatmap_label)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015)

    metadata = f" | pair_id={pair.pair_id}" if pair.pair_id else ""
    fig.suptitle(
        f"Needleman-Wunsch global image alignment | score={result.score:.4f} | "
        f"normalized={result.normalized_score:.4f} | score_mode={score_mode}{metadata}",
        fontsize=11,
        fontweight="bold",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def evaluate_sample(
    models,
    pair: ImagePair,
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
        result = needleman_wunsch(
            match_scores,
            gap_penalty=float(gap),
            similarity_offset=0.0,
        )

        heatmap, heatmap_label = _heatmap_matrix(
            raw_similarity, match_scores, result, heatmap_source
        )
        _save_visualization(
            arr1,
            arr2,
            heatmap,
            heatmap_label,
            result,
            output,
            pair,
            resolved_mode + "+ink",
        )

        path = result.pairs
        path_cosines = [float(raw_similarity[i, j]) for i, j in path]
        gap_steps = sum(
            1 for step in result.steps
            if step.index1 is None or step.index2 is None
        )
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
            "match_steps": int(len(path)),
            "gap_steps": int(gap_steps),
            "mean_path_cosine": float(np.mean(path_cosines)) if path_cosines else 0.0,
            "positive_match_fraction": float(np.mean(match_scores > 0.0)),
            "line1_windows": int(raw_similarity.shape[0]),
            "line2_windows": int(raw_similarity.shape[1]),
            **_matrix_stats(raw_similarity, "cosine"),
            **_matrix_stats(match_scores, "match"),
            "feature": str(feature),
            "score_mode": resolved_mode,
            "score_clip": float(score_clip),
            "threshold": float(threshold),
            "gap": float(gap),
            "heatmap_source": str(heatmap_source),
            "dataset_type": str(dataset_type),
            "image1": str(image1),
            "image2": str(image2),
            "output": str(output),
            "error": "",
        }
        print(
            f"[{pair.index}] pair_id={pair.pair_id or '-'} "
            f"nw={result.normalized_score:.4f} matches={len(path)} gaps={gap_steps} "
            f"cosine=[{row['cosine_min']:.3f},{row['cosine_max']:.3f}] "
            f"match=[{row['match_min']:.3f},{row['match_max']:.3f}] "
            f"positive={row['positive_match_fraction']:.3f} saved={output}",
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
    summary = {
        "algorithm": "needleman_wunsch",
        "samples": len(rows),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "mean_normalized_nw_score": float(np.mean([
            row["normalized_nw_score"] for row in successful
        ])) if successful else None,
        "mean_path_cosine": float(np.mean([
            row["mean_path_cosine"] for row in successful
        ])) if successful else None,
        "mean_positive_match_fraction": float(np.mean([
            row["positive_match_fraction"] for row in successful
        ])) if successful else None,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
