#!/usr/bin/env python3
"""Component-aware Needleman-Wunsch evaluation for fixed-63 synthetic ViT checkpoints."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
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
    patch_range_to_pixels,
)
from Evaluation.sw_core import build_match_scores, resolve_score_mode


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _mutual_stats(matrix: np.ndarray, row: int, col: int) -> tuple[float, float]:
    value = float(matrix[row, col])
    row_values = np.asarray(matrix[row], dtype=np.float32)
    col_values = np.asarray(matrix[:, col], dtype=np.float32)
    row_std = max(float(np.std(row_values)), 1e-4)
    col_std = max(float(np.std(col_values)), 1e-4)
    row_z = (value - float(np.mean(row_values))) / row_std
    col_z = (value - float(np.mean(col_values))) / col_std
    row_pct = float(np.mean(row_values <= value))
    col_pct = float(np.mean(col_values <= value))
    return float(min(row_z, col_z)), float(min(row_pct, col_pct))


def _records(result, matrix: np.ndarray):
    values = []
    for position, step in enumerate(result.steps):
        if step.index1 is None or step.index2 is None:
            continue
        row, col = int(step.index1), int(step.index2)
        z_value, percentile = _mutual_stats(matrix, row, col)
        values.append(
            {
                "position": int(position),
                "row": row,
                "col": col,
                "score": float(matrix[row, col]),
                "mutual_z": z_value,
                "mutual_pct": percentile,
            }
        )
    return values


def _can_join(left, right, max_path_gap: int, max_window_gap: int) -> bool:
    return (
        int(right["position"]) - int(left["position"]) - 1 <= max_path_gap
        and int(right["row"]) - int(left["row"]) - 1 <= max_window_gap
        and int(right["col"]) - int(left["col"]) - 1 <= max_window_gap
    )


def _component_record(group, seed_positions: set[int]) -> dict:
    positions = [int(item["position"]) for item in group]
    rows = [int(item["row"]) for item in group]
    cols = [int(item["col"]) for item in group]
    scores = np.asarray([float(item["score"]) for item in group], dtype=np.float32)
    zs = np.asarray([float(item["mutual_z"]) for item in group], dtype=np.float32)
    pcts = np.asarray([float(item["mutual_pct"]) for item in group], dtype=np.float32)
    step_span = max(1, max(positions) - min(positions) + 1)
    row_span = max(rows) - min(rows) + 1
    col_span = max(cols) - min(cols) + 1
    density = float(len(group) / step_span)
    balance = float(min(row_span, col_span) / max(1, max(row_span, col_span)))
    positive_sum = float(np.maximum(scores, 0.0).sum())
    mean_z = float(zs.mean())
    mean_pct = float(pcts.mean())
    quality = positive_sum * density * mean_pct * max(0.25, min(2.0, mean_z + 0.5))
    return {
        "items": list(group),
        "path": [(int(item["row"]), int(item["col"])) for item in group],
        "step_start": min(positions),
        "step_end": max(positions) + 1,
        "traceback_steps": step_span,
        "row_span": row_span,
        "col_span": col_span,
        "seed_count": sum(position in seed_positions for position in positions),
        "mean_match_score": float(scores.mean()),
        "mean_mutual_z": mean_z,
        "mean_mutual_pct": mean_pct,
        "support_density": density,
        "span_balance": balance,
        "score": positive_sum,
        "quality": float(quality),
    }


def _merge_components(left, right, seed_positions):
    items = sorted(
        {int(item["position"]): item for item in (*left["items"], *right["items"])}.values(),
        key=lambda item: int(item["position"]),
    )
    return _component_record(items, seed_positions)


def extract_components(result, match_scores: np.ndarray) -> list[dict]:
    """Use the same component-v2 logic as the current CNN+BiLSTM evaluation."""
    matrix = np.asarray(match_scores, dtype=np.float32)
    records = _records(result, matrix)
    if not records:
        return []

    seed_score = _env_float("NW_COMPONENT_SEED_SCORE", 0.22)
    seed_z = _env_float("NW_COMPONENT_SEED_MUTUAL_Z", 0.25)
    seed_pct = _env_float("NW_COMPONENT_SEED_PERCENTILE", 0.82)
    support_score = _env_float("NW_COMPONENT_SUPPORT_SCORE", 0.04)
    support_z = _env_float("NW_COMPONENT_SUPPORT_MUTUAL_Z", -0.10)
    support_pct = _env_float("NW_COMPONENT_SUPPORT_PERCENTILE", 0.62)
    max_path_gap = max(0, _env_int("NW_COMPONENT_MAX_PATH_GAP", 3))
    max_window_gap = max(0, _env_int("NW_COMPONENT_MAX_WINDOW_GAP", 2))
    merge_path_gap = max(0, _env_int("NW_COMPONENT_MERGE_PATH_GAP", 3))
    merge_window_gap = max(0, _env_int("NW_COMPONENT_MERGE_WINDOW_GAP", 2))

    min_matches = max(1, _env_int("NW_COMPONENT_MIN_MATCHES", 7))
    min_span_windows = max(1, _env_int("NW_COMPONENT_MIN_SPAN_WINDOWS", 7))
    min_span_fraction = max(0.0, min(1.0, _env_float("NW_COMPONENT_MIN_SPAN_FRACTION", 0.13)))
    required_row_span = max(min_span_windows, int(math.ceil(matrix.shape[0] * min_span_fraction)))
    required_col_span = max(min_span_windows, int(math.ceil(matrix.shape[1] * min_span_fraction)))
    min_seeds = max(1, _env_int("NW_COMPONENT_MIN_SEEDS", 2))
    min_mean_score = _env_float("NW_COMPONENT_MIN_MEAN_SCORE", 0.12)
    min_mean_z = _env_float("NW_COMPONENT_MIN_MEAN_MUTUAL_Z", 0.10)
    min_mean_pct = _env_float("NW_COMPONENT_MIN_MEAN_PERCENTILE", 0.72)
    min_density = _env_float("NW_COMPONENT_MIN_DENSITY", 0.50)
    min_balance = _env_float("NW_COMPONENT_MIN_SPAN_BALANCE", 0.55)
    min_quality = _env_float("NW_COMPONENT_MIN_QUALITY", 1.25)
    min_relative = max(0.0, min(1.0, _env_float("NW_COMPONENT_MIN_RELATIVE_QUALITY", 0.35)))
    max_components = max(1, _env_int("NW_COMPONENT_MAX_COMPONENTS", 3))
    weak_global_score = _env_float("NW_COMPONENT_WEAK_GLOBAL_SCORE", -0.05)
    weak_global_min_coverage = max(
        0.0, min(1.0, _env_float("NW_COMPONENT_WEAK_GLOBAL_MIN_COVERAGE", 0.16))
    )

    seeds = [
        item for item in records
        if item["score"] >= seed_score
        and item["mutual_z"] >= seed_z
        and item["mutual_pct"] >= seed_pct
    ]
    if not seeds:
        return []
    seed_positions = {int(item["position"]) for item in seeds}

    support = [
        item for item in records
        if item["score"] >= support_score
        and item["mutual_z"] >= support_z
        and item["mutual_pct"] >= support_pct
    ]
    if not support:
        return []

    groups = []
    group = [support[0]]
    for item in support[1:]:
        if _can_join(group[-1], item, max_path_gap, max_window_gap):
            group.append(item)
        else:
            groups.append(group)
            group = [item]
    groups.append(group)

    components = [_component_record(group, seed_positions) for group in groups]
    components = [
        component for component in components
        if len(component["path"]) >= min_matches
        and component["row_span"] >= required_row_span
        and component["col_span"] >= required_col_span
        and component["seed_count"] >= min_seeds
        and component["mean_match_score"] >= min_mean_score
        and component["mean_mutual_z"] >= min_mean_z
        and component["mean_mutual_pct"] >= min_mean_pct
        and component["support_density"] >= min_density
        and component["span_balance"] >= min_balance
        and component["quality"] >= min_quality
    ]
    if not components:
        return []

    components.sort(key=lambda component: component["step_start"])
    merged = [components[0]]
    for component in components[1:]:
        previous = merged[-1]
        if _can_join(previous["items"][-1], component["items"][0], merge_path_gap, merge_window_gap):
            merged[-1] = _merge_components(previous, component, seed_positions)
        else:
            merged.append(component)

    strongest = max(component["quality"] for component in merged)
    credible = [
        component for component in merged
        if component["quality"] >= max(min_quality, strongest * min_relative)
    ]
    credible.sort(key=lambda component: component["quality"], reverse=True)
    credible = credible[:max_components]
    credible.sort(key=lambda component: component["step_start"])

    rows = {row for component in credible for row, _ in component["path"]}
    cols = {col for component in credible for _, col in component["path"]}
    coverage = min(len(rows) / max(1, matrix.shape[0]), len(cols) / max(1, matrix.shape[1]))
    if float(result.normalized_score) < weak_global_score and coverage < weak_global_min_coverage:
        return []
    return credible


def _load_manifest(path: Path):
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    return records


def _mask_path(image_path: Path) -> Path:
    name = image_path.name
    if name.startswith("img1_"):
        mask_name = "mask1_" + name[len("img1_"):]
    elif name.startswith("img2_"):
        mask_name = "mask2_" + name[len("img2_"):]
    else:
        raise ValueError(f"Cannot infer synthetic mask from {image_path}")
    return image_path.parent.parent / "masks" / mask_name


def _gt_columns(mask_path: Path, width: int) -> np.ndarray:
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened.convert("L"))
    columns = np.any(mask > 0, axis=0)
    if columns.shape[0] != width:
        image = Image.fromarray((columns.astype(np.uint8) * 255)[None, :]).resize((width, 1), Image.NEAREST)
        columns = np.asarray(image)[0] > 0
    return columns.astype(bool)


def _pred_columns(components, axis: int, n_windows: int, width: int, use_flip: bool) -> np.ndarray:
    prediction = np.zeros(width, dtype=bool)
    for component in components:
        indices = [int(pair[axis]) for pair in component["path"]]
        if not indices:
            continue
        left, right = patch_range_to_pixels(
            min(indices), max(indices) + 1, n_windows, width, use_flip
        )
        lo = max(0, min(width, int(math.floor(min(left, right)))))
        hi = max(0, min(width, int(math.ceil(max(left, right)))))
        prediction[lo:hi] = True
    return prediction


def _counts(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    return {
        "tp": int(np.logical_and(pred, gt).sum()),
        "fp": int(np.logical_and(pred, ~gt).sum()),
        "fn": int(np.logical_and(~pred, gt).sum()),
        "tn": int(np.logical_and(~pred, ~gt).sum()),
    }


def _metrics(counts: dict) -> dict:
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": float(precision),
        "recall": float(recall),
        "iou": float(iou),
        "dice": float(dice),
        "f1": float(f1),
    }


def _sum_counts(items):
    return {key: int(sum(item[key] for item in items)) for key in ("tp", "fp", "fn", "tn")}


def _draw_span(ax, arr, component, axis, n_windows, use_flip):
    indices = [int(pair[axis]) for pair in component["path"]]
    if not indices:
        return
    left, right = patch_range_to_pixels(min(indices), max(indices) + 1, n_windows, arr.shape[1], use_flip)
    x0, x1 = min(left, right), max(left, right)
    ax.add_patch(Rectangle((x0, 1), max(2.0, x1 - x0), max(2.0, arr.shape[0] - 2),
                           facecolor="red", edgecolor="red", alpha=0.28, linewidth=1.5))


def _save_visualization(arr1, arr2, raw_similarity, result, components, features1, features2, output, use_flip):
    matrix = np.asarray(result.score_matrix[1:, 1:], dtype=np.float32)
    n1, n2 = raw_similarity.shape
    fig = plt.figure(figsize=(22, 16))
    grid = fig.add_gridspec(3, 1, height_ratios=[2.0, 2.0, 10.0], hspace=0.18)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    axes[0].imshow(arr1, aspect="auto")
    axes[1].imshow(arr2, aspect="auto")
    for component in components:
        _draw_span(axes[0], arr1, component, 0, len(features1.contextual), use_flip)
        _draw_span(axes[1], arr2, component, 1, len(features2.contextual), use_flip)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].set_title(f"line A | predicted components={len(components)}")
    axes[1].set_title("line B | red = component-aware aligned regions")

    ax = fig.add_subplot(grid[2])
    finite = matrix[np.isfinite(matrix)]
    if finite.size and float(finite.min()) >= 0.0:
        upper = max(1e-6, float(np.percentile(finite, 98)))
        image = ax.imshow(matrix, aspect="equal", origin="upper", vmin=0.0, vmax=upper,
                          cmap="viridis", interpolation="nearest")
    else:
        limit = max(0.5, float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0)
        image = ax.imshow(matrix, aspect="equal", origin="upper", vmin=-limit, vmax=limit,
                          cmap="coolwarm", interpolation="nearest")
    global_path = [
        (int(step.index1), int(step.index2)) for step in result.steps
        if step.index1 is not None and step.index2 is not None
    ]
    if global_path:
        ax.plot([col for _, col in global_path], [row for row, _ in global_path],
                color="black", linewidth=1.5, marker=".", markersize=3,
                label="global NW diagonal path")
    selected = [pair for component in components for pair in component["path"]]
    if selected:
        ax.scatter([col for _, col in selected], [row for row, _ in selected],
                   s=22, facecolors="cyan", edgecolors="black", linewidths=0.5,
                   label="accepted aligned components", zorder=6)
    ax.set_xlim(-0.5, n2 - 0.5)
    ax.set_ylim(n1 - 0.5, -0.5)
    ax.set_xlabel("line B windows")
    ax.set_ylabel("line A windows")
    ax.set_title(
        f"ViT Needleman-Wunsch | score={result.score:.4f} | "
        f"normalized={result.normalized_score:.4f}"
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015, label="accumulated NW DP score")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--pair-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature", default="contextual", choices=("contextual", "local", "grouped"))
    parser.add_argument("--score-mode", default="raw", choices=("raw", "centered", "mutual-z", "auto"))
    parser.add_argument("--score-clip", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--gap", type=float, default=-0.30)
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--save-predicted-masks", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = _load_manifest(Path(args.pair_manifest))
    if not records:
        raise SystemExit("No test pairs found in manifest")

    models = load_evaluation_models(args.weights, args.device, load_text_model=False)
    rows = []
    line_counts = []
    pair_counts = []
    no_prediction_pairs = 0
    macro_lines = []

    preview_dir = output_dir / "mask_previews"
    if args.save_predicted_masks:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for position, record in enumerate(records, start=1):
        image1 = Path(record["image1"])
        image2 = Path(record["image2"])
        with Image.open(image1) as opened:
            arr1 = np.asarray(opened.convert("RGB"))
        with Image.open(image2) as opened:
            arr2 = np.asarray(opened.convert("RGB"))

        features1 = get_image_features(models, image1, "synthetic")
        features2 = get_image_features(models, image2, "synthetic")
        raw_similarity = compute_similarity(
            features1.select(args.feature), features2.select(args.feature)
        ).cpu().numpy()
        resolved_mode = resolve_score_mode(args.score_mode, "synthetic")
        match_scores = build_match_scores(raw_similarity, resolved_mode, args.score_clip, args.threshold)
        match_scores = ink_aware_match_scores(
            match_scores,
            features1.ink.detach().cpu().numpy(),
            features2.ink.detach().cpu().numpy(),
        )
        result = needleman_wunsch(match_scores, gap_penalty=args.gap, similarity_offset=0.0)
        components = extract_components(result, match_scores)
        if not components:
            no_prediction_pairs += 1

        use_flip = bool(models.image_model.use_flip)
        pred1 = _pred_columns(components, 0, raw_similarity.shape[0], arr1.shape[1], use_flip)
        pred2 = _pred_columns(components, 1, raw_similarity.shape[1], arr2.shape[1], use_flip)
        gt1 = _gt_columns(_mask_path(image1), arr1.shape[1])
        gt2 = _gt_columns(_mask_path(image2), arr2.shape[1])
        counts1, counts2 = _counts(pred1, gt1), _counts(pred2, gt2)
        counts_pair = _sum_counts([counts1, counts2])
        metrics1, metrics2, metrics_pair = _metrics(counts1), _metrics(counts2), _metrics(counts_pair)
        line_counts.extend([counts1, counts2])
        pair_counts.append(counts_pair)
        macro_lines.extend([metrics1, metrics2])

        dataset_index = int(record.get("dataset_index", position))
        row = {
            "index": position,
            "dataset_index": dataset_index,
            "pair_id": record.get("pair_id", ""),
            "nw_score": float(result.score),
            "normalized_nw_score": float(result.normalized_score),
            "predicted_components": len(components),
            **{f"line1_{key}": value for key, value in {**counts1, **metrics1}.items()},
            **{f"line2_{key}": value for key, value in {**counts2, **metrics2}.items()},
            **{f"pair_{key}": value for key, value in {**counts_pair, **metrics_pair}.items()},
            "image1": str(image1),
            "image2": str(image2),
        }
        rows.append(row)

        if args.save_predicted_masks:
            Image.fromarray((pred1.astype(np.uint8) * 255)[None, :]).resize((arr1.shape[1], 32), Image.NEAREST).save(preview_dir / f"pred1_{position}.png")
            Image.fromarray((gt1.astype(np.uint8) * 255)[None, :]).resize((arr1.shape[1], 32), Image.NEAREST).save(preview_dir / f"gt1_{position}.png")
            Image.fromarray((pred2.astype(np.uint8) * 255)[None, :]).resize((arr2.shape[1], 32), Image.NEAREST).save(preview_dir / f"pred2_{position}.png")
            Image.fromarray((gt2.astype(np.uint8) * 255)[None, :]).resize((arr2.shape[1], 32), Image.NEAREST).save(preview_dir / f"gt2_{position}.png")

        if not args.metrics_only:
            _save_visualization(
                arr1, arr2, raw_similarity, result, components,
                features1, features2, output_dir / f"pair_{position}.png", use_flip,
            )
        print(
            f"[{position}/{len(records)}] dataset_index={dataset_index} "
            f"components={len(components)} pair_iou={metrics_pair['iou']:.4f} "
            f"pair_f1={metrics_pair['f1']:.4f}",
            flush=True,
        )

    micro_counts = _sum_counts(pair_counts)
    micro = _metrics(micro_counts)
    macro_pair = {
        key: float(np.mean([row[f"pair_{key}"] for row in rows]))
        for key in ("precision", "recall", "iou", "dice", "f1")
    }
    macro_line = {
        key: float(np.mean([item[key] for item in macro_lines]))
        for key in ("precision", "recall", "iou", "dice", "f1")
    }

    fieldnames = list(rows[0].keys())
    csv_name = "mask_metrics.csv" if args.metrics_only else "samples.csv"
    with (output_dir / csv_name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "metric_space": "horizontal_mask_columns",
        "algorithm": "needleman_wunsch_component_aware_v2",
        "visual_encoder": "vit",
        "evaluated_pairs": len(rows),
        "evaluated_lines": 2 * len(rows),
        "no_prediction_pairs": int(no_prediction_pairs),
        "micro_counts": micro_counts,
        "micro": micro,
        "macro_pair": macro_pair,
        "macro_line": macro_line,
        "dice_equals_f1_for_binary_masks": True,
        "feature": args.feature,
        "score_mode": resolved_mode,
        "score_clip": float(args.score_clip),
        "threshold": float(args.threshold),
        "gap": float(args.gap),
    }
    summary_name = "mask_metrics_summary.json" if args.metrics_only else "summary.json"
    (output_dir / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
