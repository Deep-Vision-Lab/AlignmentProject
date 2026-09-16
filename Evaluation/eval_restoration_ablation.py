#!/usr/bin/env python3
"""Ablation evaluation for ResNet18 + ViT-Tiny + Fusion manuscript alignment.

Compares the SAME trained checkpoint in four inference modes:
  local             ResNet-18 window features only
  contextual        ViT-Tiny contextual features only
  fused             trained Fusion(local, contextual)
  shuffled_context  Fusion(local, deterministically permuted contextual tokens)

For Synthetic63-style data, mask1_i.png/mask2_i.png are treated as the exact
shared-region ground truth.  The image-to-image monotonic path maps the GT
source region into the opposite line; we score the mapped region with IoU,
center error, source coverage, and nearest-neighbour hit rate.

The script also exports per-mode similarity heatmaps with the monotonic path and
GT shared-token bands for visual inspection.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import torch
from PIL import Image

from Evaluation._eval_utils import (
    compute_similarity,
    get_image_features,
    iter_synthetic_pairs,
    load_evaluation_models,
    needleman_wunsch,
    read_text,
)

MODES = ("local", "contextual", "fused", "shuffled_context")


def _stride(config: dict) -> int:
    if "stride" in config:
        return max(1, int(config["stride"]))
    window = int(config.get("window_size", 32))
    return max(1, int(round(window * float(config.get("stride_ratio", 0.5)))))


def load_mask_columns(path: Path, target_width: int = 1024) -> np.ndarray:
    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size[0] != int(target_width):
            mask = mask.resize((int(target_width), mask.size[1]), Image.Resampling.NEAREST)
        arr = np.asarray(mask, dtype=np.uint8)
    return np.any(arr >= 128, axis=0)


def token_interval(
    token_index: int,
    n_tokens: int,
    *,
    window_size: int,
    stride: int,
    width: int,
    flipped: bool,
) -> tuple[int, int]:
    raw_index = n_tokens - 1 - int(token_index) if flipped else int(token_index)
    x0_model = raw_index * int(stride)
    x1_model = x0_model + int(window_size)
    model_width = max(int(window_size), (n_tokens - 1) * int(stride) + int(window_size))
    scale = float(width) / float(model_width)
    x0 = max(0, min(width, int(round(x0_model * scale))))
    x1 = max(x0 + 1, min(width, int(round(x1_model * scale))))
    return x0, x1


def mask_token_indices(
    columns: np.ndarray,
    n_tokens: int,
    *,
    window_size: int,
    stride: int,
    flipped: bool,
) -> list[int]:
    width = int(columns.shape[0])
    indices = []
    for token in range(int(n_tokens)):
        x0, x1 = token_interval(
            token,
            n_tokens,
            window_size=window_size,
            stride=stride,
            width=width,
            flipped=flipped,
        )
        center = min(width - 1, max(0, (x0 + x1 - 1) // 2))
        if bool(columns[center]) or bool(columns[x0:x1].any()):
            indices.append(token)
    return indices


def predicted_columns_from_tokens(
    tokens: set[int],
    n_tokens: int,
    width: int,
    *,
    window_size: int,
    stride: int,
    flipped: bool,
) -> np.ndarray:
    result = np.zeros(int(width), dtype=bool)
    for token in sorted(tokens):
        x0, x1 = token_interval(
            token,
            n_tokens,
            window_size=window_size,
            stride=stride,
            width=width,
            flipped=flipped,
        )
        result[x0:x1] = True
    return result


def binary_iou(left: np.ndarray, right: np.ndarray) -> float:
    union = np.logical_or(left, right).sum()
    if union <= 0:
        return 0.0
    return float(np.logical_and(left, right).sum() / union)


def mask_center(columns: np.ndarray) -> float | None:
    xs = np.flatnonzero(columns)
    if xs.size == 0:
        return None
    return float((xs[0] + xs[-1]) / 2.0)


def center_error(predicted: np.ndarray, target: np.ndarray) -> float:
    p = mask_center(predicted)
    t = mask_center(target)
    if p is None or t is None:
        return float("nan")
    return abs(p - t)


def _mapping_metrics(
    pairs: list[tuple[int, int]],
    source_gt: list[int],
    target_gt: list[int],
    *,
    source_is_first: bool,
    source_n_tokens: int,
    target_n_tokens: int,
    target_mask: np.ndarray,
    window_size: int,
    stride: int,
    flipped: bool,
) -> dict:
    source_set = set(int(v) for v in source_gt)
    target_set = set(int(v) for v in target_gt)
    mapping: dict[int, set[int]] = {}
    for i, j in pairs:
        source, target = (i, j) if source_is_first else (j, i)
        mapping.setdefault(int(source), set()).add(int(target))

    mapped = set()
    covered_sources = 0
    for source in source_set:
        values = mapping.get(source, set())
        if values:
            covered_sources += 1
            mapped.update(values)

    pred_columns = predicted_columns_from_tokens(
        mapped,
        target_n_tokens,
        int(target_mask.shape[0]),
        window_size=window_size,
        stride=stride,
        flipped=flipped,
    )
    return {
        "iou": binary_iou(pred_columns, target_mask),
        "center_error_px": center_error(pred_columns, target_mask),
        "source_coverage": covered_sources / max(1, len(source_set)),
        "mapped_target_tokens": len(mapped),
        "mapped_target_gt_precision": len(mapped & target_set) / max(1, len(mapped)),
        "pred_columns": pred_columns,
    }


def nearest_neighbor_hit_rates(
    similarity: torch.Tensor,
    source_gt: list[int],
    target_gt: list[int],
    *,
    source_is_first: bool,
    top_k: int = 5,
) -> tuple[float, float]:
    matrix = similarity.detach().float().cpu()
    target_set = set(int(v) for v in target_gt)
    if not source_gt or not target_set:
        return 0.0, 0.0

    hit1 = 0
    hitk = 0
    for source in source_gt:
        scores = matrix[int(source)] if source_is_first else matrix[:, int(source)]
        k = min(max(1, int(top_k)), int(scores.numel()))
        top = torch.topk(scores, k=k).indices.tolist()
        hit1 += int(int(top[0]) in target_set)
        hitk += int(any(int(value) in target_set for value in top))
    count = max(1, len(source_gt))
    return hit1 / count, hitk / count


def _mean_path_similarity(similarity: torch.Tensor, pairs: list[tuple[int, int]]) -> float:
    if not pairs:
        return 0.0
    values = [float(similarity[i, j].item()) for i, j in pairs]
    return float(np.mean(values))


def evaluate_mode(
    mode: str,
    feat1,
    feat2,
    mask1: np.ndarray,
    mask2: np.ndarray,
    *,
    window_size: int,
    stride: int,
    flipped: bool,
    gap_penalty: float,
) -> tuple[dict, dict]:
    left = feat1.select(mode)
    right = feat2.select(mode)
    similarity = compute_similarity(left, right)
    alignment = needleman_wunsch(similarity, gap_penalty=float(gap_penalty))
    pairs = alignment.pairs

    gt1 = mask_token_indices(
        mask1, left.shape[0],
        window_size=window_size, stride=stride, flipped=flipped,
    )
    gt2 = mask_token_indices(
        mask2, right.shape[0],
        window_size=window_size, stride=stride, flipped=flipped,
    )

    map12 = _mapping_metrics(
        pairs, gt1, gt2,
        source_is_first=True,
        source_n_tokens=left.shape[0],
        target_n_tokens=right.shape[0],
        target_mask=mask2,
        window_size=window_size,
        stride=stride,
        flipped=flipped,
    )
    map21 = _mapping_metrics(
        pairs, gt2, gt1,
        source_is_first=False,
        source_n_tokens=right.shape[0],
        target_n_tokens=left.shape[0],
        target_mask=mask1,
        window_size=window_size,
        stride=stride,
        flipped=flipped,
    )

    nn12_1, nn12_5 = nearest_neighbor_hit_rates(
        similarity, gt1, gt2, source_is_first=True, top_k=5
    )
    nn21_1, nn21_5 = nearest_neighbor_hit_rates(
        similarity, gt2, gt1, source_is_first=False, top_k=5
    )

    row = {
        "mode": mode,
        "path_pairs": len(pairs),
        "path_mean_cosine": _mean_path_similarity(similarity, pairs),
        "nw_normalized_score": float(alignment.normalized_score),
        "iou_1_to_2": float(map12["iou"]),
        "iou_2_to_1": float(map21["iou"]),
        "mean_mask_iou": float(np.mean([map12["iou"], map21["iou"]])),
        "center_error_1_to_2_px": float(map12["center_error_px"]),
        "center_error_2_to_1_px": float(map21["center_error_px"]),
        "mean_center_error_px": float(np.nanmean([
            map12["center_error_px"], map21["center_error_px"]
        ])),
        "coverage_1_to_2": float(map12["source_coverage"]),
        "coverage_2_to_1": float(map21["source_coverage"]),
        "mean_source_coverage": float(np.mean([
            map12["source_coverage"], map21["source_coverage"]
        ])),
        "mapped_target_precision_1_to_2": float(map12["mapped_target_gt_precision"]),
        "mapped_target_precision_2_to_1": float(map21["mapped_target_gt_precision"]),
        "nn_top1_hit_rate": float(np.mean([nn12_1, nn21_1])),
        "nn_top5_hit_rate": float(np.mean([nn12_5, nn21_5])),
        "gt_tokens_line1": len(gt1),
        "gt_tokens_line2": len(gt2),
    }
    aux = {
        "similarity": similarity.detach().cpu().numpy(),
        "pairs": pairs,
        "gt1": gt1,
        "gt2": gt2,
        "pred1": map21["pred_columns"],
        "pred2": map12["pred_columns"],
    }
    return row, aux


def save_heatmap(path: Path, similarity: np.ndarray, pairs, gt1, gt2, title: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(similarity, aspect="auto", origin="lower")
    if pairs:
        ys = [i for i, _ in pairs]
        xs = [j for _, j in pairs]
        ax.plot(xs, ys, linewidth=1.5, label="DTW/NW path")
    if gt1:
        ax.axhspan(min(gt1) - 0.5, max(gt1) + 0.5, alpha=0.12, label="GT shared region")
    if gt2:
        ax.axvspan(min(gt2) - 0.5, max(gt2) + 0.5, alpha=0.12)
    ax.set_xlabel("Line 2 window index")
    ax.set_ylabel("Line 1 window index")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.colorbar(image, ax=ax, label="Cosine similarity")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_region_overlay(
    path: Path,
    image1: Path,
    image2: Path,
    mask1: np.ndarray,
    mask2: np.ndarray,
    pred1: np.ndarray,
    pred2: np.ndarray,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    with Image.open(image1) as im1:
        arr1 = np.asarray(im1.convert("RGB"))
    with Image.open(image2) as im2:
        arr2 = np.asarray(im2.convert("RGB"))

    fig, axes = plt.subplots(2, 1, figsize=(14, 4.8))
    for ax, arr, gt, pred, name in (
        (axes[0], arr1, mask1, pred1, "Line 1"),
        (axes[1], arr2, mask2, pred2, "Line 2"),
    ):
        ax.imshow(arr)
        gt_x = np.flatnonzero(gt)
        pred_x = np.flatnonzero(pred)
        if gt_x.size:
            ax.axvspan(gt_x[0], gt_x[-1], alpha=0.18, label="GT shared region")
        if pred_x.size:
            ax.axvspan(pred_x[0], pred_x[-1], alpha=0.18, label="Mapped prediction")
        ax.set_title(name)
        ax.set_axis_off()
        ax.legend(loc="upper right")
    fig.suptitle(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize(rows: list[dict]) -> list[dict]:
    output = []
    numeric_keys = [
        "mean_mask_iou",
        "mean_center_error_px",
        "mean_source_coverage",
        "mapped_target_precision_1_to_2",
        "mapped_target_precision_2_to_1",
        "nn_top1_hit_rate",
        "nn_top5_hit_rate",
        "path_mean_cosine",
        "nw_normalized_score",
    ]
    for mode in MODES:
        selected = [row for row in rows if row["mode"] == mode]
        if not selected:
            continue
        item = {"mode": mode, "n_pairs": len(selected)}
        for key in numeric_keys:
            values = [
                float(row[key]) for row in selected
                if math.isfinite(float(row[key]))
            ]
            item[key + "_mean"] = mean(values) if values else float("nan")
            item[key + "_std"] = pstdev(values) if len(values) > 1 else 0.0
        output.append(item)
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--gap-penalty", type=float, default=-0.25)
    parser.add_argument("--save-figures", action="store_true")
    parser.add_argument("--max-figures", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    models = load_evaluation_models(args.weights, device=args.device, load_text_model=False)
    family = str(models.config.get("architecture_family", ""))
    if family != "restoration-positive-dtw-window-encoder":
        raise SystemExit(
            "This evaluator is for restoration-positive-dtw-window-encoder checkpoints; "
            f"got architecture_family={family!r}"
        )

    window_size = int(models.config.get("window_size", 32))
    stride = _stride(models.config)
    flipped = str(models.config.get("lang", "Arabic")).lower() == "arabic"

    rows: list[dict] = []
    pairs = list(iter_synthetic_pairs(
        data_dir,
        start_index=args.start_index,
        n_samples=args.n_samples,
    ))
    if not pairs:
        raise SystemExit("No synthetic pairs found")

    for pair_number, pair in enumerate(pairs, start=1):
        mask1_path = data_dir / "masks" / f"mask1_{pair.index}.png"
        mask2_path = data_dir / "masks" / f"mask2_{pair.index}.png"
        if not mask1_path.is_file() or not mask2_path.is_file():
            raise FileNotFoundError(
                f"Ground-truth masks are required: {mask1_path}, {mask2_path}"
            )

        feat1 = get_image_features(models, pair.image1, dataset_type="synthetic")
        feat2 = get_image_features(models, pair.image2, dataset_type="synthetic")
        mask1 = load_mask_columns(mask1_path, target_width=1024)
        mask2 = load_mask_columns(mask2_path, target_width=1024)
        text1 = read_text(pair.text1, boundary_spaces=False)
        text2 = read_text(pair.text2, boundary_spaces=False)

        for mode in MODES:
            row, aux = evaluate_mode(
                mode, feat1, feat2, mask1, mask2,
                window_size=window_size,
                stride=stride,
                flipped=flipped,
                gap_penalty=args.gap_penalty,
            )
            row.update({
                "pair_index": pair.index,
                "text1": text1,
                "text2": text2,
            })
            rows.append(row)

            if args.save_figures and pair_number <= int(args.max_figures):
                prefix = output_dir / "figures" / f"pair_{pair.index:06d}_{mode}"
                save_heatmap(
                    prefix.with_name(prefix.name + "_heatmap.png"),
                    aux["similarity"], aux["pairs"], aux["gt1"], aux["gt2"],
                    f"Pair {pair.index} — {mode}",
                )
                save_region_overlay(
                    prefix.with_name(prefix.name + "_regions.png"),
                    pair.image1, pair.image2,
                    mask1, mask2, aux["pred1"], aux["pred2"],
                    f"Pair {pair.index} — {mode}",
                )

        if pair_number % 10 == 0 or pair_number == len(pairs):
            print(f"EVAL {pair_number}/{len(pairs)}", flush=True)

    summaries = summarize(rows)
    _write_csv(output_dir / "per_pair.csv", rows)
    _write_csv(output_dir / "summary.csv", summaries)

    by_mode = {item["mode"]: item for item in summaries}
    comparisons = {}
    if "fused" in by_mode and "local" in by_mode:
        comparisons["fused_minus_local_iou"] = (
            by_mode["fused"]["mean_mask_iou_mean"]
            - by_mode["local"]["mean_mask_iou_mean"]
        )
    if "fused" in by_mode and "contextual" in by_mode:
        comparisons["fused_minus_contextual_iou"] = (
            by_mode["fused"]["mean_mask_iou_mean"]
            - by_mode["contextual"]["mean_mask_iou_mean"]
        )
    if "fused" in by_mode and "shuffled_context" in by_mode:
        comparisons["fused_minus_shuffled_iou"] = (
            by_mode["fused"]["mean_mask_iou_mean"]
            - by_mode["shuffled_context"]["mean_mask_iou_mean"]
        )

    report = {
        "weights": str(Path(args.weights).resolve()),
        "data_dir": str(data_dir.resolve()),
        "window_size": window_size,
        "stride": stride,
        "flipped": flipped,
        "n_pairs": len(pairs),
        "modes": list(MODES),
        "summary": summaries,
        "comparisons": comparisons,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nABlation summary")
    print(
        "mode               IoU      center_px  coverage  NN@1     NN@5     path_cos",
        flush=True,
    )
    for item in summaries:
        print(
            f"{item['mode']:<18} "
            f"{item['mean_mask_iou_mean']:.4f}   "
            f"{item['mean_center_error_px_mean']:.2f}      "
            f"{item['mean_source_coverage_mean']:.4f}    "
            f"{item['nn_top1_hit_rate_mean']:.4f}   "
            f"{item['nn_top5_hit_rate_mean']:.4f}   "
            f"{item['path_mean_cosine_mean']:.4f}",
            flush=True,
        )
    for name, value in comparisons.items():
        print(f"{name}={value:+.4f}", flush=True)
    print(f"results={output_dir}", flush=True)


if __name__ == "__main__":
    main()
