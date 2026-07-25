#!/usr/bin/env python3
"""Run Needleman-Wunsch directly over line-image window embeddings.

The prediction is fully image-to-image: raw cosine similarities are converted to
mutual row/column-normalized match scores before global Needleman-Wunsch. This
removes the broad positive cosine background that otherwise pulls the path toward
a plain diagonal. Transcripts are used only to annotate/evaluate matched windows
with the token assigned by the trained blank-aware Span-DTW path.
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
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation._eval_utils import (
    align_text_to_windows,
    compute_similarity,
    get_image_features,
    iter_synthetic_pairs,
    json_ready,
    load_evaluation_models,
    needleman_wunsch,
    read_text,
    synthetic_pair_paths,
    validate_pair_paths,
)
from Evaluation.window_alignment import (
    alignment_score_matrix,
    attach_raw_similarities,
    consecutive_match_segments,
    window_alignment_metrics,
    window_token_labels,
)


def _window_pixels(index, n_windows, image_width, config, flipped):
    window = int(config.get("window_size", 32))
    stride = int(config.get("stride", max(1, window // 2)))
    model_width = int(config.get("input_width", 1024))
    logical = int(index)
    physical = n_windows - 1 - logical if flipped else logical
    start = max(0, physical * stride)
    end = min(model_width, start + window)
    scale = float(image_width) / max(1, model_width)
    return start * scale, end * scale


def _segment_pixels(segment, side, n_windows, image_width, config, flipped):
    indices = [
        int(step.index1 if side == 1 else step.index2)
        for step in segment
    ]
    bounds = [
        _window_pixels(index, n_windows, image_width, config, flipped)
        for index in indices
    ]
    return min(start for start, _end in bounds), max(end for _start, end in bounds)


def _labels_for_line(models, text_path, features):
    text = read_text(text_path, boundary_spaces=False)
    _prepared, _encoding, path = align_text_to_windows(models, text, features, True)
    return text, path, window_token_labels(path, len(features.contextual))


def evaluate_pair(
    models,
    pair,
    feature,
    gap,
    similarity_offset,
    score_mode,
    score_clip,
    dataset_type,
):
    validate_pair_paths(pair)
    features1 = get_image_features(models, pair.image1, dataset_type)
    features2 = get_image_features(models, pair.image2, dataset_type)
    selected1 = features1.select(feature)
    selected2 = features2.select(feature)

    raw_similarity = compute_similarity(selected1, selected2)
    match_scores = alignment_score_matrix(
        raw_similarity,
        mode=score_mode,
        clip=score_clip,
    )
    scored_alignment = needleman_wunsch(
        match_scores,
        gap_penalty=gap,
        similarity_offset=similarity_offset,
    )
    alignment = attach_raw_similarities(scored_alignment, raw_similarity)

    text1, span_path1, labels1 = _labels_for_line(models, pair.text1, features1)
    text2, span_path2, labels2 = _labels_for_line(models, pair.text2, features2)
    metrics = window_alignment_metrics(alignment, labels1, labels2)
    metrics.update(
        {
            "index": pair.index,
            "line1_windows": len(selected1),
            "line2_windows": len(selected2),
            "feature": feature,
            "score_mode": score_mode,
            "span_steps_line1": len(span_path1),
            "span_steps_line2": len(span_path2),
        }
    )
    return {
        "pair": pair,
        "features1": features1,
        "features2": features2,
        "similarity": raw_similarity,
        "alignment_scores": match_scores,
        "alignment": alignment,
        "labels1": labels1,
        "labels2": labels2,
        "text1": text1,
        "text2": text2,
        "metrics": metrics,
    }


def _visible_segments(alignment, min_similarity, max_drawn_segments):
    segments = consecutive_match_segments(
        alignment,
        min_similarity=float(min_similarity),
    )
    limit = int(max_drawn_segments)
    if limit <= 0 or len(segments) <= limit:
        return segments
    positions = np.linspace(0, len(segments) - 1, limit).round().astype(int)
    return [segments[int(position)] for position in positions]


def _full_traceback_coordinates(alignment):
    """Return the complete NW traceback in DP-boundary coordinates."""
    row = 0
    col = 0
    xs = [0.0]
    ys = [0.0]
    for step in alignment.steps:
        if step.index1 is not None:
            row += 1
        if step.index2 is not None:
            col += 1
        xs.append(float(col))
        ys.append(float(row))
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def save_visualization(
    result,
    output,
    min_similarity=-1.0,
    max_drawn_segments=0,
    show_heatmap=True,
):
    pair = result["pair"]
    with Image.open(pair.image1) as opened:
        image1 = np.asarray(opened.convert("RGB"))
    with Image.open(pair.image2) as opened:
        image2 = np.asarray(opened.convert("RGB"))

    n1 = len(result["features1"].contextual)
    n2 = len(result["features2"].contextual)
    flipped = bool(result.get("flipped", True))
    segments = _visible_segments(
        result["alignment"],
        min_similarity,
        max_drawn_segments,
    )
    cmap = plt.get_cmap("turbo", max(2, len(segments)))

    rows = 3 if show_heatmap else 2
    ratios = [2.2, 2.2, 4.2] if show_heatmap else [2.2, 2.2]
    fig = plt.figure(figsize=(18, 10 if show_heatmap else 5.5))
    grid = fig.add_gridspec(rows, 1, height_ratios=ratios, hspace=0.16)
    ax1 = fig.add_subplot(grid[0])
    ax2 = fig.add_subplot(grid[1])

    for ax, image, label in ((ax1, image1, "line 1"), (ax2, image2, "line 2")):
        ax.imshow(image, aspect="auto")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, labelpad=32, va="center")

    # One color and one continuous rectangle per strict consecutive diagonal block.
    # No arrows/connectors are drawn between the two lines.
    for segment_index, segment in enumerate(segments):
        color = cmap(segment_index)
        x01, x11 = _segment_pixels(
            segment,
            side=1,
            n_windows=n1,
            image_width=image1.shape[1],
            config=result["models_config"],
            flipped=flipped,
        )
        x02, x12 = _segment_pixels(
            segment,
            side=2,
            n_windows=n2,
            image_width=image2.shape[1],
            config=result["models_config"],
            flipped=flipped,
        )
        for ax, image, x0, x1 in (
            (ax1, image1, x01, x11),
            (ax2, image2, x02, x12),
        ):
            ax.add_patch(
                Rectangle(
                    (x0, 1),
                    max(2.0, x1 - x0),
                    max(2.0, image.shape[0] - 2),
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.22,
                    linewidth=1.4,
                )
            )

    metrics = result["metrics"]
    fig.suptitle(
        "Window-level Needleman–Wunsch alignment\n"
        f"pair={pair.index}  score_mode={metrics['score_mode']}  "
        f"matched={metrics['matched_window_pairs']}  "
        f"segments={metrics['consecutive_segments']}  "
        f"mean cosine={metrics['mean_matched_cosine']:.3f}  "
        f"token agreement={metrics['token_agreement']:.3f}",
        fontsize=11,
        fontweight="bold",
    )

    if show_heatmap:
        heat_grid = grid[2].subgridspec(1, 2, wspace=0.22)
        ax_raw = fig.add_subplot(heat_grid[0, 0])
        ax_score = fig.add_subplot(heat_grid[0, 1])

        raw = result["similarity"].detach().cpu().numpy()
        raw_image = ax_raw.imshow(
            raw,
            aspect="auto",
            origin="upper",
            vmin=-1,
            vmax=1,
            cmap="coolwarm",
            interpolation="nearest",
        )
        for segment_index, segment in enumerate(segments):
            color = cmap(segment_index)
            ax_raw.scatter(
                [int(step.index2) for step in segment],
                [int(step.index1) for step in segment],
                s=18,
                facecolors="none",
                edgecolors=[color],
                linewidths=0.9,
            )
        ax_raw.set_title(
            "Raw cosine similarity + colored consecutive match blocks",
            fontsize=9,
        )
        ax_raw.set_xlabel("line 2 logical windows (0 = rightmost)")
        ax_raw.set_ylabel("line 1 logical windows (0 = rightmost)")
        fig.colorbar(
            raw_image,
            ax=ax_raw,
            fraction=0.035,
            pad=0.02,
            label="cosine similarity",
        )

        scores = result["alignment_scores"].detach().cpu().numpy()
        finite = np.abs(scores[np.isfinite(scores)])
        score_limit = float(np.percentile(finite, 98)) if finite.size else 1.0
        score_limit = max(0.5, score_limit)
        score_image = ax_score.imshow(
            scores,
            aspect="auto",
            origin="upper",
            vmin=-score_limit,
            vmax=score_limit,
            cmap="coolwarm",
            interpolation="nearest",
        )
        trace_x, trace_y = _full_traceback_coordinates(result["alignment"])
        # DP coordinates describe window boundaries. Shift by half a cell to overlay
        # them on the [S1,S2] match-score matrix while preserving gap moves.
        ax_score.plot(
            trace_x - 0.5,
            trace_y - 0.5,
            color="black",
            linewidth=1.45,
            marker=".",
            markersize=2.2,
            label="complete NW traceback",
            zorder=4,
        )
        ax_score.set_xlim(-0.5, n2 - 0.5)
        ax_score.set_ylim(n1 - 0.5, -0.5)
        ax_score.set_title(
            f"{metrics['score_mode']} match score used by NW + traceback",
            fontsize=9,
        )
        ax_score.set_xlabel("line 2 logical windows (0 = rightmost)")
        ax_score.set_ylabel("line 1 logical windows (0 = rightmost)")
        ax_score.legend(loc="upper left", fontsize=7)
        fig.colorbar(
            score_image,
            ax=ax_score,
            fraction=0.035,
            pad=0.02,
            label="NW match score",
        )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _aggregate(rows):
    if not rows:
        return {"samples": 0}
    numeric = [
        key
        for key in rows[0]
        if key != "index" and all(isinstance(row.get(key), (int, float)) for row in rows)
    ]
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
    parser.add_argument("--dataset-type", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature", choices=("contextual", "local", "grouped"), default="contextual")
    parser.add_argument("--gap", type=float, default=-0.35)
    parser.add_argument("--similarity-offset", type=float, default=0.0)
    parser.add_argument(
        "--score-mode",
        choices=("raw", "centered", "mutual-z"),
        default="mutual-z",
        help="Matrix used by NW; mutual-z removes broad row/column cosine bias.",
    )
    parser.add_argument(
        "--score-clip",
        type=float,
        default=4.0,
        help="Absolute clipping bound for mutual-z scores; <=0 disables clipping.",
    )
    parser.add_argument("--min-similarity", type=float, default=-1.0)
    parser.add_argument(
        "--max-drawn-segments",
        type=int,
        default=0,
        help="Maximum colored consecutive blocks to draw; <=0 draws every block.",
    )
    parser.add_argument(
        "--max-drawn-pairs",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--output", default="Results/Evaluation/NW/window_nw.png")
    parser.add_argument("--output-dir", default="Results/Evaluation/NW/windows")
    parser.add_argument("--no-heatmap", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    models = load_evaluation_models(args.weights, args.device, load_text_model=True)
    if args.dataset_type != "synthetic":
        raise SystemExit(
            "Batch pair discovery currently supports synthetic layout; "
            "use a manifest adapter for real data."
        )
    pairs = (
        list(iter_synthetic_pairs(args.data_dir, args.start_index, args.n_samples))
        if args.batch
        else [synthetic_pair_paths(args.data_dir, args.index)]
    )
    rows = []
    output_dir = Path(args.output_dir)
    max_drawn_segments = (
        args.max_drawn_segments
        if args.max_drawn_pairs is None
        else args.max_drawn_pairs
    )
    for pair in pairs:
        result = evaluate_pair(
            models,
            pair,
            args.feature,
            args.gap,
            args.similarity_offset,
            args.score_mode,
            args.score_clip,
            args.dataset_type,
        )
        result["models_config"] = models.config
        result["flipped"] = bool(models.image_model.use_flip)
        rows.append(result["metrics"])
        output = output_dir / f"window_nw_{pair.index}.png" if args.batch else Path(args.output)
        save_visualization(
            result,
            output,
            min_similarity=args.min_similarity,
            max_drawn_segments=max_drawn_segments,
            show_heatmap=not args.no_heatmap,
        )
        print(json.dumps(json_ready(result["metrics"]), ensure_ascii=False), flush=True)

    if args.batch:
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]) if rows else ["index"])
            writer.writeheader()
            writer.writerows(rows)
        (output_dir / "summary.json").write_text(
            json.dumps(json_ready(_aggregate(rows)), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
