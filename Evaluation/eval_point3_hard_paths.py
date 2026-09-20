#!/usr/bin/env python3
"""Point-3 diagnostic: inspect hard monotonic image-to-transcript letter paths.

For the same fixed test pairs, compare the trained fused representation from:
  1) the original ResNet-token -> TinyViT architecture, and
  2) the physical-window -> TinyViT architecture.

Point 3 is deliberately qualitative/path-structural.  It does NOT score GT
masks or word-level success; those belong to Points 4 and 5.

For each pair/architecture this script saves a compact diagnostic:
  * one overview containing the two grayscale model inputs and their
    fused-window × transcript-letter cost maps;
  * the hard monotonic letter-DTW path as CSV;
  * a JSON summary with transcript paths, sizes and path costs.

The heatmap window axis is displayed in physical Arabic RTL order. The path
uses the same letter cost mode and transition/position penalties as training;
it is the hard-path diagnostic counterpart of the soft training DTW.

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



def _transcript_path_for_line(image_path: Path) -> Path:
    """Resolve the transcript paired with one native ArabicDataset line image."""
    image_path = Path(image_path)
    side_dir = image_path.parent.parent
    candidates = [
        side_dir / "text" / "final" / "original" / f"{image_path.stem}.txt",
        side_dir / "text" / "final" / f"{image_path.stem}.txt",
        side_dir / "text" / f"{image_path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Transcript not found for {image_path}; tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _hard_letter_dtw_path(
    costs: np.ndarray,
    *,
    vertical_penalty: float,
    horizontal_penalty: float,
    position_prior_weight: float,
    disable_horizontal_when_feasible: bool,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """Hard-path counterpart of the training letter-DTW over [window, letter]."""
    matrix = np.asarray(costs, dtype=np.float64).copy()
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(
            f"Expected non-empty [windows, letters] cost matrix, got {matrix.shape}"
        )

    n_windows, n_letters = matrix.shape
    if position_prior_weight > 0.0 and n_windows > 1 and n_letters > 1:
        window_position = np.linspace(0.0, 1.0, n_windows)[:, None]
        letter_position = np.linspace(0.0, 1.0, n_letters)[None, :]
        matrix += float(position_prior_weight) * np.abs(
            window_position - letter_position
        )

    horizontal = float(horizontal_penalty)
    if disable_horizontal_when_feasible and n_windows >= n_letters:
        horizontal = 1e4

    dp = np.full((n_windows, n_letters), np.inf, dtype=np.float64)
    back = np.full((n_windows, n_letters), -1, dtype=np.int8)
    dp[0, 0] = matrix[0, 0]

    for i in range(n_windows):
        for j in range(n_letters):
            if i == 0 and j == 0:
                continue
            candidates: list[tuple[float, int]] = []
            if i > 0 and j > 0:
                candidates.append((dp[i - 1, j - 1], 0))
            if i > 0:
                candidates.append(
                    (dp[i - 1, j] + float(vertical_penalty), 1)
                )
            if j > 0:
                candidates.append((dp[i, j - 1] + horizontal, 2))
            previous, direction = min(
                candidates, key=lambda item: (item[0], item[1])
            )
            dp[i, j] = matrix[i, j] + previous
            back[i, j] = direction

    i, j = n_windows - 1, n_letters - 1
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
            raise RuntimeError(
                f"Invalid letter-DTW traceback at window={i}, letter={j}"
            )
        path.append((i, j))
    path.reverse()
    return path, matrix.astype(np.float32)


def _letter_dtw_side(features, image_path: Path, text_encoder, pconfig) -> dict:
    """Evaluate exactly one line against its own normalized Arabic transcript."""
    transcript_path = _transcript_path_for_line(image_path)
    text = transcript_path.read_text(encoding="utf-8").strip()
    letters = _clean_letters(text)
    if not letters:
        raise ValueError(
            f"No Arabic letters remain after NFKC/filtering: {transcript_path}"
        )

    visual = features.contextual
    physical_window_indices = torch.arange(
        visual.shape[0], device=visual.device, dtype=torch.long
    )
    valid = getattr(features, "ink", None)
    if valid is not None:
        valid = valid.to(visual.device).bool()
        if bool(valid.any()):
            physical_window_indices = physical_window_indices[valid]
            visual = visual[valid]
    if visual.shape[0] == 0:
        raise ValueError(f"No valid visual windows for {image_path}")

    with torch.inference_mode():
        costs = (
            letter_dtw_cost_matrix(pconfig, text_encoder, visual, letters)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    path, display_costs = _hard_letter_dtw_path(
        costs,
        vertical_penalty=float(pconfig.positive_letter_dtw_vertical_penalty),
        horizontal_penalty=float(pconfig.positive_letter_dtw_horizontal_penalty),
        position_prior_weight=float(pconfig.positive_letter_dtw_position_prior),
        disable_horizontal_when_feasible=bool(
            pconfig.positive_letter_dtw_disable_horizontal_when_feasible
        ),
    )

    path_values = np.asarray(
        [display_costs[i, j] for i, j in path], dtype=np.float64
    )
    return {
        "transcript_path": str(transcript_path),
        "text": text,
        "letters": letters,
        "costs": display_costs,
        "path": path,
        "physical_window_indices": (
            physical_window_indices.detach().cpu().numpy().astype(np.int64)
        ),
        "windows": int(display_costs.shape[0]),
        "letter_count": int(display_costs.shape[1]),
        "mean_path_cost": float(path_values.mean()),
        "normalized_hard_path_cost": float(
            path_values.sum()
            / max(1, display_costs.shape[0] + display_costs.shape[1])
        ),
    }


def _plot_letter_panel(ax, result: dict, title: str):
    """Plot letter-DTW with Arabic physical windows increasing right-to-left."""
    costs = result["costs"]
    letters = result["letters"]
    physical = result["physical_window_indices"]

    heat = ax.imshow(costs.T, aspect="auto", origin="upper")
    xs = [window for window, _letter in result["path"]]
    ys = [letter for _window, letter in result["path"]]
    ax.plot(xs, ys, linewidth=2.0, label="hard DTW path")
    ax.set_xlabel("Visual windows — RTL; rightmost valid window is on the right")
    ax.set_ylabel("Transcript letters in logical Arabic reading order")
    ax.set_title(title)

    # RTL window orientation is fixed explicitly below after all ticks are set.

    if len(letters) <= 80:
        letter_ticks = np.arange(len(letters))
    else:
        step = max(1, len(letters) // 60)
        letter_ticks = np.arange(0, len(letters), step)
    ax.set_yticks(letter_ticks)
    ax.set_yticklabels([letters[i] for i in letter_ticks], fontsize=8)

    n_windows = len(physical)

    # Every heatmap column is one VALID visual window.  Show every one of
    # them, not a sampled subset, and label it with the physical fixed-grid
    # window id used by the 63-position model sequence.
    window_ticks = np.arange(n_windows)
    ax.set_xticks(window_ticks)
    ax.set_xticklabels(
        [f"W{int(physical[i]):02d}" for i in window_ticks],
        rotation=90,
        fontsize=7,
        ha="center",
        va="top",
    )
    ax.tick_params(
        axis="x",
        which="major",
        bottom=True,
        labelbottom=True,
        length=4,
        pad=2,
    )

    # Draw the boundary of every window column so it is visually impossible
    # to confuse neighbouring heatmap cells.
    ax.set_xticks(np.arange(-0.5, n_windows, 1.0), minor=True)
    ax.grid(
        which="minor",
        axis="x",
        linewidth=0.35,
        alpha=0.45,
    )
    ax.tick_params(axis="x", which="minor", bottom=False)

    # Explicit RTL limits are more robust than invert_xaxis() after adding
    # major/minor ticks: physical/model window 0 stays on the RIGHT.
    ax.set_xlim(n_windows - 0.5, -0.5)

    ax.legend(loc="upper left")
    return heat


def save_letter_dtw_overview(
    line1,
    result1: dict,
    line2,
    result2: dict,
    output: Path,
    title: str,
) -> None:
    """Save one compact figure containing both lines and both letter-DTW maps."""
    max_windows = max(
        int(result1["windows"]),
        int(result2["windows"]),
    )
    figure_width = max(22.0, 0.42 * float(max_windows))
    fig = plt.figure(figsize=(figure_width, 15))
    grid = fig.add_gridspec(
        4, 1, height_ratios=[1.0, 5.0, 1.0, 5.0], hspace=0.36
    )

    ax_line1 = fig.add_subplot(grid[0])
    ax_line1.imshow(np.asarray(line1.convert("L")), cmap="gray", vmin=0, vmax=255)
    ax_line1.set_title("Line 1 — exact grayscale evaluation input")
    ax_line1.axis("off")

    ax_heat1 = fig.add_subplot(grid[1])
    heat1 = _plot_letter_panel(
        ax_heat1,
        result1,
        (
            "Line 1: visual windows × transcript letters | "
            f"mean hard-path cost={result1['mean_path_cost']:.3f}"
        ),
    )
    fig.colorbar(
        heat1,
        ax=ax_heat1,
        label="training DTW cell cost + position prior (lower is better)",
    )

    ax_line2 = fig.add_subplot(grid[2])
    ax_line2.imshow(np.asarray(line2.convert("L")), cmap="gray", vmin=0, vmax=255)
    ax_line2.set_title("Line 2 — exact grayscale evaluation input")
    ax_line2.axis("off")

    ax_heat2 = fig.add_subplot(grid[3])
    heat2 = _plot_letter_panel(
        ax_heat2,
        result2,
        (
            "Line 2: visual windows × transcript letters | "
            f"mean hard-path cost={result2['mean_path_cost']:.3f}"
        ),
    )
    fig.colorbar(
        heat2,
        ax=ax_heat2,
        label="training DTW cell cost + position prior (lower is better)",
    )

    fig.suptitle(title, fontsize=14)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_letter_paths(
    path_file: Path, side_results: list[tuple[int, dict]]
) -> None:
    with path_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "side",
                "step",
                "valid_window_index",
                "physical_window_index",
                "letter_index",
                "letter",
                "cost",
            ],
        )
        writer.writeheader()
        for side, result in side_results:
            physical = result["physical_window_indices"]
            for step, (window, letter_index) in enumerate(result["path"]):
                writer.writerow(
                    {
                        "side": int(side),
                        "step": int(step),
                        "valid_window_index": int(window),
                        "physical_window_index": int(physical[window]),
                        "letter_index": int(letter_index),
                        "letter": result["letters"][letter_index],
                        "cost": float(result["costs"][window, letter_index]),
                    }
                )


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
            "line1_normalized_hard_path_cost": side1["normalized_hard_path_cost"],
            "line2_normalized_hard_path_cost": side2["normalized_hard_path_cost"],
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
            f"letter-DTW side1={side1['normalized_hard_path_cost']:.4f} "
            f"side2={side2['normalized_hard_path_cost']:.4f} "
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
        "line1_mean_path_cost",
        "line2_mean_path_cost",
        "line1_normalized_hard_path_cost",
        "line2_normalized_hard_path_cost",
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
