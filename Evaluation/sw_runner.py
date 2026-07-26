"""CLI runner for local image-to-image Smith-Waterman evaluation."""
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

from Evaluation._eval_utils import (
    compute_similarity,
    get_image_features,
    load_evaluation_models,
)
from Evaluation.sw_core import (
    ImagePair,
    _build_match_scores,
    _contiguous_runs,
    _format_heatmap_value,
    _heatmap_matrix,
    _resolve_score_mode,
    build_match_scores,
    contiguous_runs,
    format_heatmap_value,
    resolve_score_mode,
    save_visualization,
    select_heatmap_matrix,
    smith_waterman,
)
from Evaluation.sw_dataset import (
    _display_image,
    _group_split_pairs,
    _load_arabic_dataset_pairs,
    _load_pair_manifest,
    _real_pair_paths,
    arabic_manifest_path,
    batch_pairs,
    display_image,
    group_split_pairs,
    load_arabic_dataset_pairs,
    load_pair_manifest,
    pair_for_index,
    real_pair_paths,
    save_binarized_inputs,
)


def evaluate_sample(
    models,
    pair: ImagePair,
    dataset_type,
    feature,
    threshold,
    gap,
    score_mode,
    score_clip,
    output,
    save_binary,
    show_heatmap,
    heatmap_source,
    annotate_values,
    value_decimals,
    annotation_fontsize,
):
    image1, image2 = Path(pair.image1), Path(pair.image2)
    if not image1.is_file() or not image2.is_file():
        missing = [str(path) for path in (image1, image2) if not path.is_file()]
        raise FileNotFoundError("Missing image pair: " + ", ".join(missing))

    binarized = str(dataset_type).lower() == "real"
    binary1 = binary2 = ""
    temporary_directory = None
    try:
        if binarized:
            arr1 = display_image(image1, "real")
            arr2 = display_image(image2, "real")
            if save_binary:
                model_image1, model_image2 = save_binarized_inputs(
                    arr1, arr2, output, pair.index
                )
                binary1, binary2 = str(model_image1), str(model_image2)
            else:
                temporary_directory = tempfile.TemporaryDirectory(
                    prefix="sw_real_binary_"
                )
                temp_root = Path(temporary_directory.name)
                model_image1, model_image2 = temp_root / "line1.png", temp_root / "line2.png"
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
        path, score, dp_score, traceback = smith_waterman(
            raw_similarity,
            threshold=threshold,
            gap_penalty=gap,
            return_traceback=True,
            match_scores=match_scores,
        )
        displayed_matrix, displayed_label = select_heatmap_matrix(
            raw_similarity,
            threshold,
            dp_score,
            heatmap_source,
            match_scores=match_scores,
            score_mode=resolved_mode,
        )
        save_visualization(
            arr1,
            arr2,
            features1,
            features2,
            path,
            traceback,
            displayed_matrix,
            displayed_label,
            score,
            output,
            models.image_model.use_flip,
            binarized,
            pair,
            resolved_mode,
            show_heatmap=show_heatmap,
            annotate_values=annotate_values,
            value_decimals=value_decimals,
            annotation_fontsize=annotation_fontsize,
        )
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()

    path_cosines = [float(raw_similarity[i, j]) for i, j in path]
    max_row, max_col = map(int, np.unravel_index(np.argmax(dp_score), dp_score.shape))
    matched_rows = {row for row, _ in path}
    matched_cols = {col for _, col in path}
    row = {
        "index": int(pair.index),
        "manifest_position": int(pair.manifest_position),
        "pair_id": pair.pair_id,
        "label_type": pair.label_type,
        "text_score": float(pair.text_score),
        "split": pair.split,
        "status": "ok",
        "score": float(score),
        "path_steps": len(path),
        "traceback_steps": max(0, len(traceback) - 1),
        "dp_max_row": max_row,
        "dp_max_col": max_col,
        "dp_max_is_terminal": bool(
            max_row == dp_score.shape[0] - 1 and max_col == dp_score.shape[1] - 1
        ),
        "mean_path_cosine": float(np.mean(path_cosines)) if path_cosines else 0.0,
        "line1_matched_fraction": len(matched_rows) / max(1, raw_similarity.shape[0]),
        "line2_matched_fraction": len(matched_cols) / max(1, raw_similarity.shape[1]),
        "line1_windows": int(raw_similarity.shape[0]),
        "line2_windows": int(raw_similarity.shape[1]),
        "line1_path_start": int(path[0][0]) if path else -1,
        "line1_path_end": int(path[-1][0]) if path else -1,
        "line2_path_start": int(path[0][1]) if path else -1,
        "line2_path_end": int(path[-1][1]) if path else -1,
        "feature": str(feature),
        "score_mode": resolved_mode,
        "score_clip": float(score_clip),
        "threshold": float(threshold),
        "gap": float(gap),
        "heatmap_source": str(heatmap_source),
        "dataset_type": str(dataset_type),
        "binarized": bool(binarized),
        "binarization": (
            os.environ.get("REAL_BINARIZE_METHOD", "otsu").lower()
            if binarized else "none"
        ),
        "flipped": bool(models.image_model.use_flip),
        "image1": str(image1),
        "image2": str(image2),
        "binarized_image1": binary1,
        "binarized_image2": binary2,
        "output": str(output),
        "error": "",
    }
    print(
        f"[{pair.index}] pair_id={pair.pair_id or '-'} label={pair.label_type or '-'} "
        f"score={score:.6f} score_mode={resolved_mode} "
        f"dp_max=({max_row},{max_col}) terminal={row['dp_max_is_terminal']} "
        f"path_steps={len(path)} matched_fraction="
        f"({row['line1_matched_fraction']:.3f},{row['line2_matched_fraction']:.3f}) "
        f"mean_cosine={row['mean_path_cosine']:.4f} saved={output}",
        flush=True,
    )
    return row


def aggregate(rows):
    successful = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    values = lambda key: [float(row[key]) for row in successful]
    scores, paths = values("score"), values("path_steps")
    traces, cosines = values("traceback_steps"), values("mean_path_cosine")
    fractions1, fractions2 = values("line1_matched_fraction"), values("line2_matched_fraction")
    mean = lambda items: float(np.mean(items)) if items else 0.0
    return {
        "samples": len(rows),
        "successful": len(successful),
        "failed": len(failed),
        "mean_score": mean(scores),
        "std_score": float(np.std(scores)) if scores else 0.0,
        "mean_path_steps": mean(paths),
        "mean_traceback_steps": mean(traces),
        "mean_path_cosine": mean(cosines),
        "mean_line1_matched_fraction": mean(fractions1),
        "mean_line2_matched_fraction": mean(fractions2),
        "terminal_dp_maxima": sum(bool(row.get("dp_max_is_terminal")) for row in successful),
        "binarized_samples": sum(bool(row.get("binarized")) for row in successful),
        "failed_indices": [int(row["index"]) for row in failed],
    }


def batch_fieldnames():
    return [
        "index", "manifest_position", "pair_id", "label_type", "text_score", "split",
        "status", "score", "path_steps", "traceback_steps", "dp_max_row", "dp_max_col",
        "dp_max_is_terminal", "mean_path_cosine", "line1_matched_fraction",
        "line2_matched_fraction", "line1_windows", "line2_windows", "line1_path_start",
        "line1_path_end", "line2_path_start", "line2_path_end", "feature", "score_mode",
        "score_clip", "threshold", "gap", "heatmap_source", "dataset_type", "binarized",
        "binarization", "flipped", "image1", "image2", "binarized_image1",
        "binarized_image2", "output", "error",
    ]


def error_row(args, pair, output, models, exc):
    row = {key: "" for key in batch_fieldnames()}
    row.update({
        "index": int(pair.index), "manifest_position": int(pair.manifest_position),
        "pair_id": pair.pair_id, "label_type": pair.label_type,
        "text_score": float(pair.text_score), "split": pair.split, "status": "error",
        "score": 0.0, "path_steps": 0, "traceback_steps": 0, "dp_max_row": -1,
        "dp_max_col": -1, "dp_max_is_terminal": False, "mean_path_cosine": 0.0,
        "line1_matched_fraction": 0.0, "line2_matched_fraction": 0.0,
        "line1_windows": 0, "line2_windows": 0, "line1_path_start": -1,
        "line1_path_end": -1, "line2_path_start": -1, "line2_path_end": -1,
        "feature": args.feature, "score_mode": args.score_mode,
        "score_clip": float(args.score_clip), "threshold": float(args.threshold),
        "gap": float(args.gap), "heatmap_source": args.heatmap_source,
        "dataset_type": args.dataset_type, "binarized": args.dataset_type == "real",
        "binarization": os.environ.get("REAL_BINARIZE_METHOD", "otsu").lower()
        if args.dataset_type == "real" else "none",
        "flipped": bool(models.image_model.use_flip), "image1": str(pair.image1),
        "image2": str(pair.image2), "output": str(output),
        "error": f"{type(exc).__name__}: {exc}",
    })
    return row


def write_batch_outputs(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=batch_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(aggregate(rows), ensure_ascii=False, indent=2), encoding="utf-8"
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="DataSet/Synthetic_Arabic")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--image1")
    parser.add_argument("--image2")
    parser.add_argument("--pair-manifest")
    parser.add_argument("--arabic-manifest")
    parser.add_argument("--real-labels", default=None)
    parser.add_argument("--real-min-text-score", type=float, default=float(os.environ.get("REAL_MIN_TEXT_SCORE", "0.0")))
    parser.add_argument("--real-text-key", default=os.environ.get("REAL_TEXT_KEY", "text_original_path"))
    parser.add_argument("--real-split", choices=("all", "train", "valid", "test"), default="test")
    parser.add_argument("--split-seed", type=int, default=int(os.environ.get("DATASET_SPLIT_SEED", "42")))
    parser.add_argument("--real-validate-paths", action="store_true")
    parser.add_argument("--image1-pattern", default="img1_{index}.png")
    parser.add_argument("--image2-pattern", default="img2_{index}.png")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-type", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--feature", choices=("contextual", "local", "grouped"), default="contextual")
    parser.add_argument(
        "--score-mode", choices=("auto", "raw", "centered", "mutual-z"), default="auto",
        help="auto uses mutual-z for real data and raw cosine for synthetic data",
    )
    parser.add_argument("--score-clip", type=float, default=4.0)
    parser.add_argument(
        "--threshold", type=float, default=0.45,
        help="offset subtracted from the selected score; in raw mode this is the cosine threshold",
    )
    parser.add_argument("--gap", type=float, default=-0.30)
    parser.add_argument(
        "--heatmap-source", choices=("dp-score", "match-score", "cosine"),
        default="dp-score", help="dp-score shows the table whose maximum starts traceback",
    )
    parser.add_argument("--no-heatmap", action="store_true")
    parser.add_argument("--no-annotate-heatmap-values", action="store_true")
    parser.add_argument("--heatmap-value-decimals", type=int, default=2)
    parser.add_argument("--heatmap-annotation-fontsize", type=float, default=5.0)
    parser.add_argument("--output", default="Results/Evaluation/SW/smith_waterman.png")
    parser.add_argument("--output-dir", default="Results/Evaluation/SW/windows")
    parser.add_argument("--no-save-binarized-images", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch and (args.image1 or args.image2):
        raise SystemExit("--image1/--image2 cannot be used with --batch")
    if bool(args.image1) != bool(args.image2):
        raise SystemExit("Provide both --image1 and --image2, or neither")
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be greater than zero")

    manifest_pairs = []
    manifest = arabic_manifest_path(args)
    if args.dataset_type == "real" and not args.pair_manifest and manifest.is_file():
        manifest_pairs = load_arabic_dataset_pairs(args)
        print(
            f"Loaded ArabicDataset manifest: {manifest} split={args.real_split} "
            f"samples={len(manifest_pairs)}", flush=True
        )
    elif args.pair_manifest:
        manifest_pairs = load_pair_manifest(args.pair_manifest, args.data_dir)

    models = load_evaluation_models(args.weights, args.device, load_text_model=False)
    common = dict(
        dataset_type=args.dataset_type, feature=args.feature, threshold=args.threshold,
        gap=args.gap, score_mode=args.score_mode, score_clip=args.score_clip,
        save_binary=not args.no_save_binarized_images, show_heatmap=not args.no_heatmap,
        heatmap_source=args.heatmap_source,
        annotate_values=not args.no_annotate_heatmap_values,
        value_decimals=args.heatmap_value_decimals,
        annotation_fontsize=args.heatmap_annotation_fontsize,
    )

    if not args.batch:
        pair = (
            ImagePair(args.index, Path(args.image1), Path(args.image2))
            if args.image1 else pair_for_index(args, args.index, manifest_pairs)
        )
        evaluate_sample(models, pair, output=Path(args.output), **common)
        return

    output_dir = Path(args.output_dir)
    selected = batch_pairs(args, manifest_pairs)
    if not selected:
        raise SystemExit("No image pairs were selected for evaluation")
    rows = []
    for pair in selected:
        output = output_dir / f"pair_{pair.index}.png"
        try:
            row = evaluate_sample(models, pair, output=output, **common)
        except Exception as exc:
            row = error_row(args, pair, output, models, exc)
            print(f"[{pair.index}] failed: {row['error']}", file=sys.stderr, flush=True)
        rows.append(row)
    write_batch_outputs(rows, output_dir)
    print(json.dumps(aggregate(rows), ensure_ascii=False), flush=True)


# Backward-compatible private names imported by tests.
_evaluate_sample = evaluate_sample
_aggregate = aggregate
_batch_fieldnames = batch_fieldnames
_error_row = error_row
_write_batch_outputs = write_batch_outputs

if __name__ == "__main__":
    main()
