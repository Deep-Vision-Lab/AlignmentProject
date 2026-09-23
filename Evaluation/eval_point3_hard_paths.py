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
import re
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
from Evaluation.point3_core import (
    hard_letter_path,
    hard_monotonic_path,
    sequence_to_physical_window,
)
from Evaluation.yelda_geometry import prepare_line
from Evaluation.yelda_runtime import read_checkpoint
from textEmbedding import OrthogonalCharEmbedding
from vlm_restoration_positive_dtw import _clean_letters, letter_dtw_cost_matrix


def hard_dtw_path(similarity: np.ndarray) -> list[tuple[int, int]]:
    """Classic DTW on cost=1-cosine with deterministic diagonal tie preference."""
    sim = np.asarray(similarity, dtype=np.float64)
    return hard_monotonic_path(
        1.0 - sim, vertical_penalty=0.0, horizontal_penalty=0.0,
        disable_horizontal_when_feasible=False,
    )[0]


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
                 start_index: int, n_samples: int, checkpoint_config=None):
    if checkpoint_config is None:
        layout, pairs = pair_loader.load_pairs(dataset, split)
    else:
        layout, pairs = pair_loader.load_checkpoint_pairs(dataset, split, checkpoint_config, split_seed)
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



def _transcript_path_for_line(
    image_path: Path,
    exact_transcript_path: Path | None = None,
) -> Path:
    """Resolve the transcript belonging to the exact displayed line image.

    For native ArabicDataset lines, the image itself is authoritative:
      .../<side>/linesImages/line_XX.png
        -> .../<side>/text/final/original/line_XX.txt

    This avoids accidentally using a transcript path inherited from a paired
    record that belongs to another line. Manifest association remains a
    fallback only for layouts without the native side/line structure.
    """
    image_path = Path(image_path)

    # Native real-data layout: force same-side, same-line transcript.
    if image_path.parent.name == "linesImages" and re.fullmatch(r"line_\d+", image_path.stem):
        side_dir = image_path.parent.parent
        native_candidates = [
            side_dir / "text" / "final" / "original" / f"{image_path.stem}.txt",
            side_dir / "text" / "final" / f"{image_path.stem}.txt",
            side_dir / "text" / f"{image_path.stem}.txt",
        ]
        for candidate in native_candidates:
            if candidate.is_file():
                # If the pair manifest points somewhere else, log it loudly;
                # never silently substitute that mismatched transcript.
                if exact_transcript_path is not None:
                    exact = Path(exact_transcript_path)
                    try:
                        same = exact.resolve() == candidate.resolve()
                    except OSError:
                        same = str(exact) == str(candidate)
                    if not same:
                        print(
                            "DTW_TRANSCRIPT_MISMATCH "
                            f"image={image_path} native={candidate} "
                            f"manifest={exact}; using native same-line transcript",
                            flush=True,
                        )
                return candidate
        raise FileNotFoundError(
            f"No same-line transcript found for native image {image_path}; "
            f"tried: {', '.join(str(path) for path in native_candidates)}"
        )

    # Non-native layouts may only carry an explicit manifest transcript.
    if exact_transcript_path is not None:
        exact = Path(exact_transcript_path)
        if not exact.is_file():
            raise FileNotFoundError(
                f"Manifest transcript for {image_path} does not exist: {exact}"
            )
        return exact

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
    """Compatibility view of the one shared Point-3 hard-path core."""
    result = hard_letter_path(
        costs, vertical_penalty=vertical_penalty,
        horizontal_penalty=horizontal_penalty,
        position_prior_weight=position_prior_weight,
        disable_horizontal_when_feasible=disable_horizontal_when_feasible,
    )
    return result.path, result.effective_costs.astype(np.float32)


def _letter_dtw_side(
    features,
    image_path: Path,
    transcript_path: Path | None,
    text_encoder,
    pconfig,
    *,
    window_size: int = 32,
    stride: int = 16,
    use_flip: bool = True,
    image_width: int | None = None,
) -> dict:
    """Evaluate exactly one line against its manifest-linked transcript."""
    transcript_path = _transcript_path_for_line(image_path, transcript_path)
    text = transcript_path.read_text(encoding="utf-8").strip()
    letters = _clean_letters(text)
    if not letters:
        raise ValueError(
            f"No Arabic letters remain after NFKC/filtering: {transcript_path}"
        )

    visual = features.contextual
    logical_sequence_indices = torch.arange(
        visual.shape[0], device=visual.device, dtype=torch.long
    )
    valid = getattr(features, "token_valid", None)
    if valid is None:
        valid = getattr(features, "ink", None)
    if valid is not None:
        valid = valid.to(visual.device).bool()
        if bool(valid.any()):
            logical_sequence_indices = logical_sequence_indices[valid]
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

    hard = hard_letter_path(
        costs,
        vertical_penalty=float(pconfig.positive_letter_dtw_vertical_penalty),
        horizontal_penalty=float(pconfig.positive_letter_dtw_horizontal_penalty),
        position_prior_weight=float(pconfig.positive_letter_dtw_position_prior),
        disable_horizontal_when_feasible=bool(
            pconfig.positive_letter_dtw_disable_horizontal_when_feasible
        ),
    )
    display_costs = hard.effective_costs.astype(np.float32)
    full_window_count = int(features.contextual.shape[0])
    width = int(image_width) if image_width is not None else int(window_size + (full_window_count - 1) * stride)
    physical_indices = getattr(features, "physical_window_indices", None)
    if physical_indices is None and 1 + (width - int(window_size)) // int(stride) != full_window_count:
        raise ValueError("Model sequence length does not match prepared-image window geometry")
    logical_indices = logical_sequence_indices.detach().cpu().numpy().astype(np.int64)
    mapping = [sequence_to_physical_window(
        index if physical_indices is None else int(physical_indices[index]),
        full_window_count if physical_indices is None else 1+(width-window_size)//stride,
        width=width, window=window_size,
        stride=stride, use_flip=use_flip if physical_indices is None else False) for index in logical_indices]
    return {
        "transcript_path": str(transcript_path),
        "text": text,
        "letters": letters,
        "costs": display_costs,
        "path": hard.path,
        "logical_sequence_indices": logical_indices,
        "physical_window_indices": np.asarray([value[0] for value in mapping], dtype=np.int64),
        "canvas_window_intervals": [(value[1], value[2]) for value in mapping],
        "windows": int(display_costs.shape[0]),
        "letter_count": int(display_costs.shape[1]),
        "mean_path_cost": hard.mean_path_cell_cost,
        "mean_path_cell_cost": hard.mean_path_cell_cost,
        "hard_objective_total": hard.hard_objective_total,
        "hard_objective_normalized": hard.hard_objective_normalized,
        "normalized_hard_path_cost": hard.hard_objective_normalized,
        "vertical_steps": hard.vertical_steps,
        "horizontal_steps": hard.horizontal_steps,
        "effective_horizontal_penalty": hard.effective_horizontal_penalty,
    }


def _plot_letter_panel(ax, result: dict, title: str):
    """Plot letter-DTW; the x-axis is the actual window-image strip below."""
    costs = result["costs"]
    letters = result["letters"]
    n_windows = int(costs.shape[0])

    heat = ax.imshow(costs.T, aspect="auto", origin="upper")
    xs = [window for window, _letter in result["path"]]
    ys = [letter for _window, letter in result["path"]]
    ax.plot(xs, ys, linewidth=2.0, label="hard DTW path")
    ax.set_ylabel("Transcript letters in logical Arabic reading order")
    ax.set_title(title)

    if len(letters) <= 80:
        letter_ticks = np.arange(len(letters))
    else:
        step = max(1, len(letters) // 60)
        letter_ticks = np.arange(0, len(letters), step)
    ax.set_yticks(letter_ticks)
    ax.set_yticklabels([letters[i] for i in letter_ticks], fontsize=8)

    # No W00/W01/... labels. The thumbnail strip below is the x-axis.
    ax.set_xticks([])
    ax.tick_params(axis="x", bottom=False, labelbottom=False)

    # Column boundaries visually connect each heatmap cell to its thumbnail.
    ax.set_xticks(np.arange(-0.5, n_windows, 1.0), minor=True)
    ax.grid(which="minor", axis="x", linewidth=0.35, alpha=0.40)
    ax.tick_params(axis="x", which="minor", bottom=False)

    # Arabic RTL: sequence column 0 is displayed on the right.
    ax.set_xlim(n_windows - 0.5, -0.5)
    ax.legend(loc="upper left")
    return heat


def _extract_window_images(
    line_image,
    sequence_window_indices,
    *,
    window_size: int,
    stride: int,
    use_flip: bool,
):
    """Extract the exact grayscale image crop represented by each model token."""
    image = line_image.convert("L")
    width, height = image.size
    total_windows = 1 + max(0, (width - int(window_size)) // int(stride))
    windows = []

    for sequence_index in sequence_window_indices:
        sequence_index = int(sequence_index)
        _physical, x0, x1 = sequence_to_physical_window(
            sequence_index, total_windows, width=width, window=window_size,
            stride=stride, use_flip=use_flip)
        left, right = int(x0), int(x1)
        windows.append(image.crop((left, 0, right, height)))

    return windows


def _plot_window_image_axis(
    ax,
    window_images,
    *,
    thumb_width_in_columns: float = 0.62,
    thumb_height: int = 72,
):
    """Show one small, separated window image centered under each DTW column."""
    n_windows = len(window_images)
    if n_windows == 0:
        ax.axis("off")
        return

    half_width = float(thumb_width_in_columns) / 2.0
    for column, crop in enumerate(window_images):
        pixels = np.asarray(crop.convert("L").resize((24, int(thumb_height))))
        ax.imshow(
            pixels,
            cmap="gray",
            vmin=0,
            vmax=255,
            aspect="auto",
            interpolation="nearest",
            extent=(column - half_width, column + half_width, 1.0, 0.0),
        )

    # Faint cell boundaries preserve exact one-to-one alignment while the
    # narrower thumbnails leave visible white space between windows.
    for boundary in np.arange(-0.5, n_windows + 0.5, 1.0):
        ax.axvline(boundary, linewidth=0.30, alpha=0.25)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("window\nimage", rotation=0, labelpad=28, va="center")
    ax.set_xlim(n_windows - 0.5, -0.5)
    ax.set_ylim(1.0, 0.0)

def save_letter_dtw_overview(
    line1,
    result1: dict,
    line2,
    result2: dict,
    output: Path,
    title: str,
) -> None:
    """Save one compact Line-1 DTW diagnostic with separated window thumbnails."""
    del line2, result2

    max_windows = int(result1["windows"])
    figure_width = max(24.0, 0.38 * float(max_windows))
    fig = plt.figure(figsize=(figure_width, 10))

    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1.0, 0.03],
        height_ratios=[0.9, 5.0, 1.6],
        hspace=0.10,
        wspace=0.04,
    )

    ax_line = fig.add_subplot(grid[0, 0])
    ax_line.imshow(np.asarray(line1.convert("L")), cmap="gray", vmin=0, vmax=255)
    ax_line.set_title(
        "Line 1 — exact grayscale evaluation input\n"
        f"Transcript: {result1['text']}"
    )
    ax_line.axis("off")

    ax_heat = fig.add_subplot(grid[1, 0])
    heat = _plot_letter_panel(
        ax_heat,
        result1,
        (
            "Line 1: visual-window images × transcript letters | "
            f"mean path cell cost={result1['mean_path_cell_cost']:.3f}; "
            f"normalized hard objective={result1['hard_objective_normalized']:.3f}"
        ),
    )

    cax = fig.add_subplot(grid[1, 1])
    fig.colorbar(
        heat,
        cax=cax,
        label="training DTW cell cost + position prior (lower is better)",
    )

    ax_windows = fig.add_subplot(grid[2, 0], sharex=ax_heat)
    _plot_window_image_axis(
        ax_windows,
        result1["window_images"],
        thumb_width_in_columns=0.62,
        thumb_height=72,
    )

    empty_top = fig.add_subplot(grid[0, 1])
    empty_top.axis("off")
    empty_bottom = fig.add_subplot(grid[2, 1])
    empty_bottom.axis("off")

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
                "logical_sequence_index",
                "physical_window_index",
                "canvas_x0",
                "canvas_x1",
                "letter_index",
                "letter",
                "cost",
            ],
        )
        writer.writeheader()
        for side, result in side_results:
            physical = result["physical_window_indices"]
            logical = result["logical_sequence_indices"]
            intervals = result["canvas_window_intervals"]
            for step, (window, letter_index) in enumerate(result["path"]):
                writer.writerow(
                    {
                        "side": int(side),
                        "step": int(step),
                        "valid_window_index": int(window),
                        "logical_sequence_index": int(logical[window]),
                        "physical_window_index": int(physical[window]),
                        "canvas_x0": int(intervals[window][0]),
                        "canvas_x1": int(intervals[window][1]),
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
    dataset: Path | None = None,
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
    from Evaluation.checkpoint_contract import evaluation_metadata
    (arch_root / "evaluation_contract.json").write_text(
        json.dumps(evaluation_metadata(models, weights, preprocessing, dataset), indent=2), encoding="utf-8")

    for ordinal, pair in enumerate(selected, start=1):
        pair_dir = arch_root / f"pair_{int(pair.index):05d}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        line1, geometry1 = prepare_line(
            pair.image1, pair.preprocess_domain(1), preprocessing, contract=models.contract
        )
        line2, geometry2 = prepare_line(
            pair.image2, pair.preprocess_domain(2), preprocessing, contract=models.contract
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

        window_size = models.contract.window_size
        stride = models.contract.stride
        use_flip = bool(models.image_model.use_flip)
        side1 = _letter_dtw_side(
            first, pair.image1, pair.text1, text_encoder, pconfig,
            window_size=window_size, stride=stride, use_flip=use_flip,
            image_width=line1.width,
        )
        side2 = _letter_dtw_side(
            second, pair.image2, pair.text2, text_encoder, pconfig,
            window_size=window_size, stride=stride, use_flip=use_flip,
            image_width=line2.width,
        )
        side1["window_images"] = _extract_window_images(
            line1,
            side1["logical_sequence_indices"],
            window_size=window_size,
            stride=stride,
            use_flip=use_flip,
        )
        side2["window_images"] = _extract_window_images(
            line2,
            side2["logical_sequence_indices"],
            window_size=window_size,
            stride=stride,
            use_flip=use_flip,
        )

        for side, image_path, result in ((1, pair.image1, side1), (2, pair.image2, side2)):
            print(
                "DTW_TRANSCRIPT_AUDIT "
                f"pair={pair.index} side={side} image={image_path} "
                f"transcript={result['transcript_path']} text={result['text']!r}",
                flush=True,
            )

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
            "pair_manifest_text1": str(pair.text1) if pair.text1 is not None else None,
            "pair_manifest_text2": str(pair.text2) if pair.text2 is not None else None,
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
            "line1_mean_path_cell_cost": side1["mean_path_cell_cost"],
            "line2_mean_path_cell_cost": side2["mean_path_cell_cost"],
            "line1_hard_objective_total": side1["hard_objective_total"],
            "line2_hard_objective_total": side2["hard_objective_total"],
            "line1_hard_objective_normalized": side1["hard_objective_normalized"],
            "line2_hard_objective_normalized": side2["hard_objective_normalized"],
            "line1_normalized_hard_path_cost": side1["normalized_hard_path_cost"],
            "line2_normalized_hard_path_cost": side2["normalized_hard_path_cost"],
            "window_display_order": "RTL; W0 shown at right",
            "window_index_definition": "logical is post-RTL model order; physical is left-to-right crop order",
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
        "line1_hard_objective_total",
        "line2_hard_objective_total",
        "line1_hard_objective_normalized",
        "line2_hard_objective_normalized",
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
        "--image-preprocessing", choices=("original", "training"), default="training"
    )
    args = parser.parse_args()
    if args.image_preprocessing != "training":
        print(
            "POINT3_PREPROCESSING_OVERRIDE: requested an ablation mode; "
            "results do not use checkpoint-faithful training geometry",
            flush=True,
        )

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

    selection_configs = [read_checkpoint(path)["model_config"] for path, _name in checkpoints]
    from Evaluation.checkpoint_contract import resolve_evaluation_contract
    selection_contracts = [resolve_evaluation_contract(config) for config in selection_configs]
    if len({(c.real_all_page_lines, c.split_seed) for c in selection_contracts}) != 1:
        raise ValueError("Point-3 checkpoint comparison requires the same training split contract")
    layout, selected = select_pairs(
        dataset,
        args.split,
        args.training_samples,
        args.split_seed,
        args.start_index,
        args.n_samples,
        checkpoint_config=selection_configs[0],
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
            dataset,
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
            dataset,
        )
        new_rows = evaluate_architecture(
            "physical_window_vit",
            new_weights,
            selected,
            output,
            args.device,
            args.image_preprocessing,
            dataset,
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
        "preprocessing_override": args.image_preprocessing != "training",
        "architectures": aggregates,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved Point-3 results to {output}", flush=True)


if __name__ == "__main__":
    main()
