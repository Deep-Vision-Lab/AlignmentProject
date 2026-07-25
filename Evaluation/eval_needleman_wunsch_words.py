#!/usr/bin/env python3
"""Evaluate image-only Needleman-Wunsch word alignment and mask paired words.

Word boundaries are localized independently in each line with the trained
image-text Span-DTW model. The cross-line prediction itself uses only pooled
visual word embeddings and Needleman-Wunsch. Transcript words are used after
prediction only to score the alignment against a reference.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation._eval_utils import (
    EvaluationModels,
    PairPaths,
    compute_similarity,
    evaluate_word_alignment,
    extract_word_regions,
    get_image_features,
    iter_synthetic_pairs,
    json_ready,
    load_evaluation_models,
    needleman_wunsch,
    patch_range_to_pixels,
    read_text,
    synthetic_pair_paths,
    transcript_reference_alignment,
    validate_pair_paths,
    word_similarity_matrix,
)

PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#9A6324",
    "#800000", "#808000", "#000075", "#a9a9a9", "#fabed4",
]


def evaluate_pair(
    models: EvaluationModels,
    pair: PairPaths,
    feature: str,
    word_gap: float,
    similarity_offset: float,
    dataset_type: str,
):
    validate_pair_paths(pair)
    text1 = read_text(pair.text1, boundary_spaces=False)
    text2 = read_text(pair.text2, boundary_spaces=False)
    features1 = get_image_features(models, pair.image1, dataset_type)
    features2 = get_image_features(models, pair.image2, dataset_type)
    regions1, span_path1 = extract_word_regions(models, text1, features1, feature)
    regions2, span_path2 = extract_word_regions(models, text2, features2, feature)
    word_sim = word_similarity_matrix(regions1, regions2)
    predicted = needleman_wunsch(word_sim, word_gap, similarity_offset)
    reference = transcript_reference_alignment(regions1, regions2)
    metrics = evaluate_word_alignment(predicted, reference, regions1, regions2)

    patch_sim = compute_similarity(features1.select(feature), features2.select(feature))
    patch_nw = needleman_wunsch(
        patch_sim,
        gap_penalty=word_gap,
        similarity_offset=similarity_offset,
    )
    metrics.update(
        {
            "index": pair.index,
            "line1_words": len(regions1),
            "line2_words": len(regions2),
            "patch_nw_score": patch_nw.score,
            "patch_nw_normalized_score": patch_nw.normalized_score,
            "span_steps_line1": len(span_path1),
            "span_steps_line2": len(span_path2),
        }
    )
    return {
        "pair": pair,
        "text1": text1,
        "text2": text2,
        "features1": features1,
        "features2": features2,
        "regions1": regions1,
        "regions2": regions2,
        "word_similarity": word_sim,
        "predicted": predicted,
        "reference": reference,
        "patch_nw": patch_nw,
        "metrics": metrics,
    }


def _mask_axis(
    ax,
    image,
    regions,
    n_windows,
    pairs,
    side,
    flipped,
    min_similarity,
):
    ax.imshow(image, aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    height, width = image.shape[:2]
    paired = {}
    for pair_index, step in enumerate(pairs):
        if step.similarity is None or step.similarity < min_similarity:
            continue
        region_index = step.index1 if side == 1 else step.index2
        if region_index is not None:
            paired[int(region_index)] = (pair_index, step)
    for region_index, (pair_index, step) in paired.items():
        region = regions[region_index]
        x0, x1 = patch_range_to_pixels(
            region.window_start,
            region.window_end,
            n_windows,
            width,
            flipped,
        )
        color = PALETTE[pair_index % len(PALETTE)]
        ax.add_patch(
            Rectangle(
                (x0, 1),
                max(2.0, x1 - x0),
                max(2.0, height - 2),
                facecolor=color,
                edgecolor=color,
                alpha=0.30,
                linewidth=2,
            )
        )
        ax.text(
            (x0 + x1) / 2,
            height / 2,
            f"{region.text}\n{step.similarity:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white",
            fontweight="bold",
            bbox=dict(facecolor=color, alpha=0.78, pad=1.5, boxstyle="round"),
        )
    return paired


def save_visualization(
    result: dict,
    output: Path,
    min_similarity: float,
    show_heatmap: bool,
):
    pair = result["pair"]
    regions1, regions2 = result["regions1"], result["regions2"]
    features1, features2 = result["features1"], result["features2"]
    predicted = result["predicted"]
    with Image.open(pair.image1) as opened:
        image1 = np.asarray(opened.convert("RGB"))
    with Image.open(pair.image2) as opened:
        image2 = np.asarray(opened.convert("RGB"))

    rows = 4 if show_heatmap else 3
    ratios = [2.2, 0.65, 2.2, 3.3] if show_heatmap else [2.2, 0.65, 2.2]
    fig = plt.figure(figsize=(15, 10 if show_heatmap else 6.5))
    grid = fig.add_gridspec(rows, 1, height_ratios=ratios, hspace=0.10)
    ax1 = fig.add_subplot(grid[0])
    connectors = fig.add_subplot(grid[1])
    ax2 = fig.add_subplot(grid[2])

    flipped = bool(result.get("flipped", True))
    pairs = [
        step
        for step in predicted.steps
        if step.index1 is not None and step.index2 is not None
    ]
    paired1 = _mask_axis(
        ax1,
        image1,
        regions1,
        len(features1.contextual),
        pairs,
        1,
        flipped,
        min_similarity,
    )
    paired2 = _mask_axis(
        ax2,
        image2,
        regions2,
        len(features2.contextual),
        pairs,
        2,
        flipped,
        min_similarity,
    )
    ax1.set_ylabel("line 1", rotation=0, labelpad=32, va="center")
    ax2.set_ylabel("line 2", rotation=0, labelpad=32, va="center")

    connectors.set_xlim(0, 1)
    connectors.set_ylim(0, 1)
    connectors.axis("off")
    width1, width2 = image1.shape[1], image2.shape[1]
    for pair_index, step in enumerate(pairs):
        if step.similarity is None or step.similarity < min_similarity:
            continue
        if step.index1 not in paired1 or step.index2 not in paired2:
            continue
        r1, r2 = regions1[step.index1], regions2[step.index2]
        x01, x11 = patch_range_to_pixels(
            r1.window_start,
            r1.window_end,
            len(features1.contextual),
            width1,
            flipped,
        )
        x02, x12 = patch_range_to_pixels(
            r2.window_start,
            r2.window_end,
            len(features2.contextual),
            width2,
            flipped,
        )
        color = PALETTE[pair_index % len(PALETTE)]
        connectors.add_line(
            Line2D(
                [((x01 + x11) / 2) / width1, ((x02 + x12) / 2) / width2],
                [1.0, 0.0],
                transform=connectors.transAxes,
                color=color,
                linewidth=1.8,
                alpha=0.8,
            )
        )

    metrics = result["metrics"]
    fig.suptitle(
        "Image-only Needleman–Wunsch word alignment\n"
        f"pair={pair.index}  F1={metrics['pair_f1']:.3f}  "
        f"exact={metrics['exact_word_accuracy']:.3f}  "
        f"mean cosine={metrics['mean_matched_cosine']:.3f}",
        fontsize=11,
        fontweight="bold",
    )

    if show_heatmap:
        axh = fig.add_subplot(grid[3])
        matrix = result["word_similarity"].detach().cpu().numpy()
        heatmap = axh.imshow(
            matrix,
            aspect="auto",
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
        )
        axh.set_xlabel("line 2 words")
        axh.set_ylabel("line 1 words")
        axh.set_xticks(
            range(len(regions2)),
            [region.text for region in regions2],
            rotation=45,
            ha="right",
            fontsize=7,
        )
        axh.set_yticks(
            range(len(regions1)),
            [region.text for region in regions1],
            fontsize=7,
        )
        axh.plot(
            [step.index2 for step in pairs],
            [step.index1 for step in pairs],
            color="black",
            linewidth=1.3,
            marker="o",
            markersize=3,
        )
        fig.colorbar(
            heatmap,
            ax=axh,
            fraction=0.025,
            pad=0.02,
            label="cosine similarity",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"samples": 0}
    numeric = sorted(
        key
        for key in rows[0]
        if key != "index"
        and all(isinstance(row.get(key), (int, float)) for row in rows)
    )
    summary = {"samples": len(rows)}
    for key in numeric:
        values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        summary[f"mean_{key}"] = float(values.mean())
        summary[f"std_{key}"] = float(values.std())
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="DataSet/Synthetic_Arabic")
    parser.add_argument(
        "--dataset-type",
        choices=("synthetic", "real"),
        default="synthetic",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--feature",
        choices=("contextual", "local", "grouped"),
        default="local",
    )
    parser.add_argument("--word-gap", type=float, default=-0.25)
    parser.add_argument("--similarity-offset", type=float, default=0.0)
    parser.add_argument("--min-similarity", type=float, default=-1.0)
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--image1")
    parser.add_argument("--image2")
    parser.add_argument("--text1-file")
    parser.add_argument("--text2-file")
    parser.add_argument(
        "--output",
        default="Results/Evaluation/NW/needleman_wunsch_words.png",
    )
    parser.add_argument("--output-dir", default="Results/Evaluation/NW")
    parser.add_argument("--no-heatmap", action="store_true")
    return parser.parse_args()


def explicit_pair(args) -> PairPaths | None:
    values = (args.image1, args.image2, args.text1_file, args.text2_file)
    if not any(values):
        return None
    if not all(values):
        raise SystemExit(
            "--image1, --image2, --text1-file, and --text2-file "
            "must be supplied together"
        )
    return PairPaths(
        Path(args.image1),
        Path(args.image2),
        Path(args.text1_file),
        Path(args.text2_file),
        args.index,
    )


def main():
    args = parse_args()
    models = load_evaluation_models(args.weights, args.device, load_text_model=True)
    if str(models.config.get("text_encoder_type", "arabic_span")) != "arabic_span":
        raise SystemExit(
            "Needleman-Wunsch word masking requires an arabic_span checkpoint"
        )

    if args.batch:
        pairs = list(
            iter_synthetic_pairs(args.data_dir, args.start_index, args.n_samples)
        )
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for pair in pairs:
            result = evaluate_pair(
                models,
                pair,
                args.feature,
                args.word_gap,
                args.similarity_offset,
                args.dataset_type,
            )
            result["flipped"] = models.image_model.use_flip
            rows.append(dict(result["metrics"]))
            save_visualization(
                result,
                output_dir / f"nw_words_{pair.index}.png",
                args.min_similarity,
                not args.no_heatmap,
            )
        summary = _aggregate(rows)
        (output_dir / "summary.json").write_text(
            json.dumps(
                json_ready({"summary": summary, "samples": rows}),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if rows:
            with (output_dir / "samples.csv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    pair = explicit_pair(args) or synthetic_pair_paths(args.data_dir, args.index)
    result = evaluate_pair(
        models,
        pair,
        args.feature,
        args.word_gap,
        args.similarity_offset,
        args.dataset_type,
    )
    result["flipped"] = models.image_model.use_flip
    save_visualization(
        result,
        Path(args.output),
        args.min_similarity,
        not args.no_heatmap,
    )
    print(json.dumps(json_ready(result["metrics"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
