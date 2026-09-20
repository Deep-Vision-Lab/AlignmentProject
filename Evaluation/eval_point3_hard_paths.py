#!/usr/bin/env python3
"""Point-3 diagnostic: inspect hard monotonic image-to-transcript letter paths.

For the same fixed test pairs, compare the trained fused representation from:
  1) the original ResNet-token -> TinyViT architecture, and
  2) the physical-window -> TinyViT architecture.

Point 3 is deliberately qualitative/path-structural.  It does NOT score GT
masks or word-level success; those belong to Points 4 and 5.

For each pair/architecture this script saves:
  * the two exact model-input line images,
  * raw fused cosine similarity matrix (.npy/.csv),
  * hard classic-DTW path (.csv),
  * a heatmap with the DTW path and normalized diagonal reference,
  * path-shape statistics (warp ratio, diagonal deviation, path margin/z).

Both architectures use the exact same selected pair list.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from types import SimpleNamespace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from Evaluation._eval_utils import compute_similarity
from Evaluation import eval_img_align_nw_diagnostic as pair_loader
from Evaluation.eval_yelda import synthetic_split, balanced_pairs, configure_geometry
from Evaluation.point2_runtime import load_point2_visual_models, point2_pair_features
from Evaluation.yelda_geometry import prepare_line
from Evaluation.yelda_runtime import read_checkpoint
from textEmbedding import OrthogonalCharEmbedding
from vlm_restoration_positive_dtw import _clean_letters, letter_dtw_cost_matrix


def hard_dtw_path(similarity: np.ndarray) -> list[tuple[int, int]]:
    """Classic DTW on cost=1-cosine with deterministic diagonal tie preference."""
    sim = np.asarray(similarity, dtype=np.float64)
    if sim.ndim != 2 or sim.shape[0] < 1 or sim.shape[1] < 1:
        raise ValueError(f"Expected non-empty 2-D similarity matrix, got {sim.shape}")

    n, m = sim.shape
    cost = 1.0 - sim
    dp = np.full((n, m), np.inf, dtype=np.float64)
    back = np.full((n, m), -1, dtype=np.int8)  # 0 diag, 1 up, 2 left
    dp[0, 0] = cost[0, 0]

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            candidates: list[tuple[float, int]] = []
            if i > 0 and j > 0:
                candidates.append((dp[i - 1, j - 1], 0))
            if i > 0:
                candidates.append((dp[i - 1, j], 1))
            if j > 0:
                candidates.append((dp[i, j - 1], 2))
            prev, direction = min(candidates, key=lambda item: (item[0], item[1]))
            dp[i, j] = cost[i, j] + prev
            back[i, j] = direction

    i, j = n - 1, m - 1
    path = [(i, j)]
    while i > 0 or j > 0:
        direction = int(back[i, j])
        if direction == 0:
            i -= 1
            j -= 1
        elif direction == 1:
            i -= 1
        elif direction == 2:
            j -= 1
        else:
            raise RuntimeError(f"Invalid DTW traceback at {(i, j)}: {direction}")
        path.append((i, j))
    path.reverse()
    return path


def _finite_mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def path_metrics(similarity: np.ndarray, path: list[tuple[int, int]]) -> dict:
    n, m = similarity.shape
    diag_steps = vertical_steps = horizontal_steps = 0
    for (i0, j0), (i1, j1) in zip(path, path[1:]):
        delta = (i1 - i0, j1 - j0)
        if delta == (1, 1):
            diag_steps += 1
        elif delta == (1, 0):
            vertical_steps += 1
        elif delta == (0, 1):
            horizontal_steps += 1
        else:
            raise RuntimeError(f"Non-monotonic DTW step: {delta}")

    path_values = np.asarray([similarity[i, j] for i, j in path], dtype=np.float64)
    matrix_mean = float(np.mean(similarity))
    matrix_std = float(np.std(similarity))
    path_mean = float(np.mean(path_values))
    margin = path_mean - matrix_mean
    z_score = margin / matrix_std if matrix_std > 1e-8 else None

    norm_i = np.asarray([i / max(1, n - 1) for i, _ in path], dtype=np.float64)
    norm_j = np.asarray([j / max(1, m - 1) for _, j in path], dtype=np.float64)
    deviation = np.abs(norm_i - norm_j)

    total_moves = max(1, len(path) - 1)
    warp_steps = vertical_steps + horizontal_steps
    return {
        "line1_windows": int(n),
        "line2_windows": int(m),
        "path_points": int(len(path)),
        "diagonal_steps": int(diag_steps),
        "vertical_steps": int(vertical_steps),
        "horizontal_steps": int(horizontal_steps),
        "warp_steps": int(warp_steps),
        "warp_ratio": float(warp_steps / total_moves),
        "mean_path_cosine": path_mean,
        "matrix_mean_cosine": matrix_mean,
        "matrix_std_cosine": matrix_std,
        "path_cosine_margin": float(margin),
        "path_cosine_z": float(z_score) if z_score is not None else None,
        "normalized_diagonal_mae": float(np.mean(deviation)),
        "normalized_diagonal_max_error": float(np.max(deviation)),
        "line1_path_coverage": float(len({i for i, _ in path}) / n),
        "line2_path_coverage": float(len({j for _, j in path}) / m),
    }


def save_heatmap(
    similarity: np.ndarray,
    path: list[tuple[int, int]],
    line1_path: Path,
    line2_path: Path,
    output: Path,
    title: str,
) -> None:
    from PIL import Image

    with Image.open(line1_path) as image:
        line1 = np.asarray(image.convert("RGB"))
    with Image.open(line2_path) as image:
        line2 = np.asarray(image.convert("RGB"))

    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 6.0, 1.0], hspace=0.22)

    ax_top = fig.add_subplot(grid[0])
    ax_top.imshow(line1)
    ax_top.set_title("Line 1: exact model input")
    ax_top.axis("off")

    ax = fig.add_subplot(grid[1])
    heat = ax.imshow(similarity, aspect="auto", origin="upper", vmin=-1.0, vmax=1.0)
    xs = [j for i, j in path]
    ys = [i for i, j in path]
    ax.plot(xs, ys, linewidth=2.0, label="hard DTW path")
    ax.plot(
        [0, max(0, similarity.shape[1] - 1)],
        [0, max(0, similarity.shape[0] - 1)],
        linestyle="--",
        linewidth=1.2,
        label="normalized diagonal reference",
    )
    ax.set_xlabel("Line 2 window index (model sequence order)")
    ax.set_ylabel("Line 1 window index (model sequence order)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.colorbar(heat, ax=ax, label="cosine similarity")

    ax_bottom = fig.add_subplot(grid[2])
    ax_bottom.imshow(line2)
    ax_bottom.set_title("Line 2: exact model input")
    ax_bottom.axis("off")

    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_path_csv(path_file: Path, path: list[tuple[int, int]], similarity: np.ndarray) -> None:
    with path_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["step", "line1_window", "line2_window", "cosine"]
        )
        writer.writeheader()
        for step, (i, j) in enumerate(path):
            writer.writerow(
                {
                    "step": step,
                    "line1_window": i,
                    "line2_window": j,
                    "cosine": float(similarity[i, j]),
                }
            )


def select_pairs(dataset: Path, split: str, training_samples: int, split_seed: int,
                 start_index: int, n_samples: int):
    layout, pairs = pair_loader.load_pairs(dataset, split)
    if layout == "synthetic":
        pairs = synthetic_split(pairs, split, training_samples, split_seed)
    else:
        # Match Evaluation.eval_yelda ordering so START_INDEX selects the same
        # held-out real pair in NW and hard-DTW diagnostics.
        pairs = balanced_pairs(pairs)
    start = start_index - 1
    selected = pairs[start:] if n_samples == 0 else pairs[start:start + n_samples]
    if not selected:
        raise ValueError("No pairs selected")
    return layout, selected


def evaluate_architecture(
    label: str,
    weights: Path,
    selected,
    output_root: Path,
    device: str,
    preprocessing: str,
) -> list[dict]:
    checkpoint = read_checkpoint(weights)
    models = load_point2_visual_models(checkpoint, device, "restoration")
    configure_geometry(models.config, preprocessing)
    config = dict(models.config)
    pconfig = SimpleNamespace(
        positive_letter_dtw_cost_mode=str(
            config.get("positive_letter_dtw_cost_mode", "full_alphabet_nll")
        ),
        positive_letter_dtw_competition_temperature=float(
            config.get("positive_letter_dtw_competition_temperature", 0.10)
        ),
        positive_letter_dtw_vertical_penalty=float(
            config.get("positive_letter_dtw_vertical_penalty", 0.05)
        ),
        positive_letter_dtw_horizontal_penalty=float(
            config.get("positive_letter_dtw_horizontal_penalty", 0.30)
        ),
        positive_letter_dtw_position_prior=float(
            config.get("positive_letter_dtw_position_prior", 0.15)
        ),
        positive_letter_dtw_disable_horizontal_when_feasible=bool(
            config.get("positive_letter_dtw_disable_horizontal_when_feasible", True)
        ),
    )
    text_encoder = OrthogonalCharEmbedding(
        embedding_dim=int(config.get("vit_embed_dim", 192)),
        vocab_size=int(config.get("letter_codebook_vocab_size", 4096)),
        seed=int(config.get("letter_codebook_seed", 1234)),
    ).to(models.device)
    text_encoder.eval()
    for parameter in text_encoder.parameters():
        parameter.requires_grad_(False)

    rows: list[dict] = []
    arch_root = output_root / label
    arch_root.mkdir(parents=True, exist_ok=True)

    for ordinal, pair in enumerate(selected, start=1):
        pair_dir = arch_root / f"pair_{int(pair.index):05d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        line1, geometry1 = prepare_line(
            pair.image1, pair.preprocess_domain(1), preprocessing
        )
        line2, geometry2 = prepare_line(
            pair.image2, pair.preprocess_domain(2), preprocessing
        )

        with tempfile.TemporaryDirectory(prefix="letter_dtw_") as tmp:
            tmp = Path(tmp)
            line1_file = tmp / "line1.png"
            line2_file = tmp / "line2.png"
            line1.save(line1_file)
            line2.save(line2_file)
            first, second = point2_pair_features(
                models, line1_file, line2_file, "fused"
            )

        side1 = _letter_dtw_side(first, pair.image1, text_encoder, pconfig)
        side2 = _letter_dtw_side(second, pair.image2, text_encoder, pconfig)

        save_letter_dtw_overview(
            line1,
            side1,
            line2,
            side2,
            pair_dir / "letter_dtw_overview.png",
            (
                f"{label} | pair={pair.index} | fused visual windows ↔ transcript letters | "
                "window axis displayed RTL"
            ),
        )
        _write_letter_paths(
            pair_dir / "letter_dtw_path.csv",
            [(1, side1), (2, side2)],
        )

        row = {
            "architecture": label,
            "ordinal": ordinal,
            "index": int(pair.index),
            "pair_id": pair.pair_id,
            "split": pair.split,
            "image1": str(pair.image1),
            "image2": str(pair.image2),
            "weights": str(weights),
            "image_preprocessing": preprocessing,
            "geometry1": geometry1,
            "geometry2": geometry2,
            "line1_transcript": side1["transcript_path"],
            "line2_transcript": side2["transcript_path"],
            "line1_windows": side1["windows"],
            "line2_windows": side2["windows"],
            "line1_letters": side1["letter_count"],
            "line2_letters": side2["letter_count"],
            "line1_mean_path_cost": side1["mean_path_cost"],
            "line2_mean_path_cost": side2["mean_path_cost"],
            "line1_final_dtw_cost": side1["final_dtw_cost"],
            "line2_final_dtw_cost": side2["final_dtw_cost"],
            "window_display_order": "RTL; W0 shown at right",
            "dtw_axis_definition": "x=visual windows, y=normalized Arabic transcript letters",
            "dtw_cost_mode": pconfig.positive_letter_dtw_cost_mode,
        }
        (pair_dir / "summary.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(row)
        print(
            f"[{label} {ordinal}/{len(selected)} pair={pair.index}] "
            f"letter-DTW side1={side1['final_dtw_cost']:.4f} "
            f"side2={side2['final_dtw_cost']:.4f} "
            f"windows={side1['windows']}/{side2['windows']} "
            f"letters={side1['letter_count']}/{side2['letter_count']}",
            flush=True,
        )
    return rows

def write_rows(path: Path, rows: list[dict]) -> None:
    serializable = []
    for row in rows:
        item = dict(row)
        for key in ("geometry1", "geometry2"):
            if key in item:
                item[key] = json.dumps(item[key], ensure_ascii=False, separators=(",", ":"))
        serializable.append(item)
    fields = sorted({key for row in serializable for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(serializable)


def aggregate(rows: list[dict], label: str) -> dict:
    metrics = [
        "mean_path_cosine",
        "path_cosine_margin",
        "path_cosine_z",
        "warp_ratio",
        "normalized_diagonal_mae",
        "normalized_diagonal_max_error",
    ]
    return {
        "architecture": label,
        "count": len(rows),
        **{f"mean_{metric}": _finite_mean([row.get(metric) for row in rows])
           for metric in metrics},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", default=None,
                        help="Evaluate one checkpoint. Preferred for epoch diagnostics.")
    parser.add_argument("--old-weights", default=None)
    parser.add_argument("--new-weights", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="test")
    parser.add_argument("--training-samples", type=int, default=6000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--image-preprocessing", choices=("original", "training"), default="original"
    )
    args = parser.parse_args()

    dataset = Path(args.dataset).expanduser().resolve()
    single_weights = (
        Path(args.weights).expanduser().resolve() if args.weights else None
    )
    old_weights = (
        Path(args.old_weights).expanduser().resolve() if args.old_weights else None
    )
    new_weights = (
        Path(args.new_weights).expanduser().resolve() if args.new_weights else None
    )
    output = Path(args.output_dir).expanduser().resolve()

    if single_weights is None and (old_weights is None or new_weights is None):
        raise SystemExit(
            "Use --weights for one checkpoint, or provide both --old-weights and --new-weights"
        )

    if not dataset.exists():
        raise SystemExit(f"Missing dataset: {dataset}")
    checkpoints = (
        [(single_weights, "checkpoint")]
        if single_weights is not None
        else [(old_weights, "old checkpoint"), (new_weights, "new checkpoint")]
    )
    for checkpoint_path, name in checkpoints:
        if checkpoint_path is None or not checkpoint_path.is_file():
            raise SystemExit(f"Missing {name}: {checkpoint_path}")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    layout, selected = select_pairs(
        dataset,
        args.split,
        args.training_samples,
        args.split_seed,
        args.start_index,
        args.n_samples,
    )
    selection = [
        {
            "ordinal": ordinal,
            "index": int(pair.index),
            "pair_id": pair.pair_id,
            "split": pair.split,
            "image1": str(pair.image1),
            "image2": str(pair.image2),
        }
        for ordinal, pair in enumerate(selected, start=1)
    ]
    (output / "selected_pairs.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        f"Point-3 hard-path diagnostic: layout={layout} split={args.split} "
        f"pairs={len(selected)} preprocessing={args.image_preprocessing}",
        flush=True,
    )

    if single_weights is not None:
        rows = evaluate_architecture(
            "checkpoint",
            single_weights,
            selected,
            output,
            args.device,
            args.image_preprocessing,
        )
        write_rows(output / "point3_samples.csv", rows)
        aggregates = [aggregate(rows, "checkpoint")]
    else:
        old_rows = evaluate_architecture(
            "old_resnet_token_vit",
            old_weights,
            selected,
            output,
            args.device,
            args.image_preprocessing,
        )
        new_rows = evaluate_architecture(
            "physical_window_vit",
            new_weights,
            selected,
            output,
            args.device,
            args.image_preprocessing,
        )
        all_rows = old_rows + new_rows
        write_rows(output / "point3_samples.csv", all_rows)
        aggregates = [
            aggregate(old_rows, "old_resnet_token_vit"),
            aggregate(new_rows, "physical_window_vit"),
        ]
        write_rows(output / "point3_architecture_summary.csv", aggregates)

    summary = {
        "point": 3,
        "goal": "hard-DTW path correctness / heatmap inspection",
        "representation": "trained fused image representation",
        "gt_localization_metrics_included": False,
        "word_level_metrics_included": False,
        "layout": layout,
        "split": args.split,
        "selected": len(selected),
        "image_preprocessing": args.image_preprocessing,
        "architectures": aggregates,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved Point-3 results to {output}", flush=True)


if __name__ == "__main__":
    main()
