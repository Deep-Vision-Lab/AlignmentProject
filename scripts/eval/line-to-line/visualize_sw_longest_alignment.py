#!/usr/bin/env python3
"""
Visualize the strongest local image-to-image alignment between manuscript line images.

This script works with the improve_neg model changes:
  - contextual embeddings: CNN + optional BiLSTM, useful for full line alignment
  - local embeddings: pre-BiLSTM CNN windows, useful for local/window matching

For full line-to-line visualization, contextual embeddings are the default. Pass
--embedding-space local to inspect the local visual embedding space.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

# This file lives in scripts/eval/line-to-line/. Add the repository root so
# imports continue to work after the scripts were moved into eval folders.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from embeddingModel import EmbeddingModel  # noqa: E402


PALETTE = [
    "#e6194b",
    "#3cb44b",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#42d4f4",
    "#f032e6",
    "#bfef45",
    "#fabed4",
    "#469990",
]


@dataclass
class ConsecutiveRun:
    line1_start: int
    line1_end: int
    line2_start: int
    line2_end: int
    length: int
    mean_similarity: float
    sw_score: float


# ---------------------------------------------------------------------------
# Model and image helpers
# ---------------------------------------------------------------------------


def _checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("image_model_state_dict", "model_state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def load_image_model(
    weights_path: str,
    device: str,
    window_size: int,
    stride: int,
    vector_size: int = 128,
    use_bilstm: Optional[bool] = None,
    use_flip: bool = False,
):
    checkpoint = torch.load(weights_path, map_location=device)
    model_config = checkpoint.get("model_config", {}) if isinstance(checkpoint, dict) else {}

    vector_size = int(model_config.get("vector_size", vector_size))
    if use_bilstm is None:
        use_bilstm = bool(model_config.get("use_bilstm", True))

    model = EmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=device,
        use_flip=use_flip,
        use_bilstm=use_bilstm,
        bilstm_layers=int(model_config.get("bilstm_layers", 2)),
        bilstm_hidden_dim=int(model_config.get("bilstm_hidden_dim", vector_size)),
    ).to(device)
    model.load_state_dict(_checkpoint_state_dict(checkpoint), strict=False)
    model.eval()
    return model


def preprocess_line_image(path: str, target_height: int = 128):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    scale = float(target_height) / float(height)
    new_width = max(1, int(round(width * scale)))
    image = image.resize((new_width, target_height), Image.BILINEAR)
    tensor = transforms.ToTensor()(image)
    return image, tensor


@torch.no_grad()
def image_embeddings(
    model,
    image_tensor: torch.Tensor,
    device: str,
    embedding_space: str = "contextual",
) -> torch.Tensor:
    """Return normalized image-window embeddings.

    embedding_space="contextual" uses the normal model output: CNN + optional
    BiLSTM. This is the old behavior and is good for line-level alignment.

    embedding_space="local" uses the improve_neg pre-BiLSTM CNN output. This is
    the recommended representation for part/window matching because it avoids
    BiLSTM smoothing that can make different local windows too similar.
    """
    batch = image_tensor.unsqueeze(0).to(device)
    embedding_space = str(embedding_space).lower()

    if embedding_space == "local":
        outputs = model(batch, return_local=True)
        if isinstance(outputs, tuple):
            embeddings = outputs[1]
        else:
            # Backward-compatible fallback for old checkpoints/code.
            embeddings = outputs
    elif embedding_space == "contextual":
        embeddings = model(batch)
    else:
        raise ValueError("embedding_space must be 'contextual' or 'local'")

    embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
    return embeddings.squeeze(0).cpu()


def cosine_similarity_matrix(emb1: torch.Tensor, emb2: torch.Tensor) -> np.ndarray:
    return (emb1 @ emb2.T).numpy()


def resolve_threshold(sim: np.ndarray, args) -> float:
    mode = str(getattr(args, "adaptive_threshold", "none")).lower()
    base = float(args.threshold)
    if mode == "none":
        return base
    if mode == "percentile":
        adaptive = float(np.percentile(sim, float(args.threshold_percentile)))
    elif mode == "mean_std":
        adaptive = float(np.mean(sim) + float(args.threshold_std_scale) * np.std(sim))
    else:
        raise ValueError(f"Unknown adaptive threshold mode: {mode}")

    if getattr(args, "adaptive_threshold_floor", True):
        return max(base, adaptive)
    return adaptive


# ---------------------------------------------------------------------------
# Smith-Waterman local alignment
# ---------------------------------------------------------------------------


def substitution_score(
    similarity: float,
    threshold: float,
    match_reward: float,
    mismatch_penalty: float,
) -> float:
    if similarity >= threshold:
        return match_reward * (similarity - threshold)
    return mismatch_penalty * (threshold - similarity)


def smith_waterman(
    sim: np.ndarray,
    threshold: float = 0.45,
    gap_penalty: float = -0.3,
    match_reward: float = 1.0,
    mismatch_penalty: float = -1.0,
):
    n, m = sim.shape
    H = np.zeros((n + 1, m + 1), dtype=np.float32)
    tb = np.zeros((n + 1, m + 1), dtype=np.int8)

    best_score = 0.0
    best_pos = (0, 0)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub = substitution_score(
                float(sim[i - 1, j - 1]),
                threshold=threshold,
                match_reward=match_reward,
                mismatch_penalty=mismatch_penalty,
            )
            diag = H[i - 1, j - 1] + sub
            up = H[i - 1, j] + gap_penalty
            left = H[i, j - 1] + gap_penalty
            best = max(0.0, diag, up, left)
            H[i, j] = best

            if best == 0.0:
                tb[i, j] = 0
            elif best == diag:
                tb[i, j] = 1
            elif best == up:
                tb[i, j] = 2
            else:
                tb[i, j] = 3

            if best > best_score:
                best_score = float(best)
                best_pos = (i, j)

    if best_score <= 0.0:
        return [], best_score, H

    path = []
    i, j = best_pos
    while i > 0 and j > 0 and H[i, j] > 0.0:
        code = int(tb[i, j])
        if code == 0:
            break
        if code == 1:
            path.append(("diag", i - 1, j - 1))
            i -= 1
            j -= 1
        elif code == 2:
            path.append(("up", i - 1, None))
            i -= 1
        else:
            path.append(("left", None, j - 1))
            j -= 1

    path.reverse()
    return path, best_score, H


def longest_consecutive_diagonal_run(
    path: Sequence[Tuple[str, Optional[int], Optional[int]]],
    sim: np.ndarray,
    sw_score: float,
    min_run_length: int = 1,
) -> Optional[ConsecutiveRun]:
    diag_pairs = [(i, j) for op, i, j in path if op == "diag" and i is not None and j is not None]
    if not diag_pairs:
        return None

    runs: List[List[Tuple[int, int]]] = []
    current = [diag_pairs[0]]
    for prev, cur in zip(diag_pairs[:-1], diag_pairs[1:]):
        if cur[0] == prev[0] + 1 and cur[1] == prev[1] + 1:
            current.append(cur)
        else:
            runs.append(current)
            current = [cur]
    runs.append(current)

    valid_runs = [run for run in runs if len(run) >= min_run_length]
    if not valid_runs:
        return None

    def run_key(run):
        sims = [float(sim[i, j]) for i, j in run]
        return (len(run), float(np.mean(sims)))

    best = max(valid_runs, key=run_key)
    sims = [float(sim[i, j]) for i, j in best]
    return ConsecutiveRun(
        line1_start=int(best[0][0]),
        line1_end=int(best[-1][0]),
        line2_start=int(best[0][1]),
        line2_end=int(best[-1][1]),
        length=len(best),
        mean_similarity=float(np.mean(sims)),
        sw_score=float(sw_score),
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def window_range_to_pixels(
    start: int,
    end: int,
    num_windows: int,
    image_width: int,
    window_size: int,
    stride: int,
    use_flip: bool,
):
    if use_flip:
        display_start = num_windows - 1 - end
        display_end = num_windows - 1 - start
    else:
        display_start = start
        display_end = end

    x0 = display_start * stride
    x1 = display_end * stride + window_size
    x0 = max(0, min(int(round(x0)), image_width - 1))
    x1 = max(x0 + 1, min(int(round(x1)), image_width))
    return x0, x1


def draw_longest_run(
    img1: Image.Image,
    img2: Image.Image,
    run: ConsecutiveRun,
    num_windows1: int,
    num_windows2: int,
    window_size: int,
    stride: int,
    output: str,
    use_flip: bool,
    title: str,
):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    h1, w1 = arr1.shape[:2]
    h2, w2 = arr2.shape[:2]

    canvas_w = max(w1, w2)
    gap = 90
    top = 25
    y1_top = top
    y1_bottom = y1_top + h1
    y2_top = y1_bottom + gap
    y2_bottom = y2_top + h2

    fig_w = max(10.0, canvas_w / 100.0)
    fig_h = max(4.0, (y2_bottom + 30) / 100.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(arr1, extent=(0, w1, y1_bottom, y1_top))
    ax.imshow(arr2, extent=(0, w2, y2_bottom, y2_top))

    color = PALETTE[0]
    x1a, x1b = window_range_to_pixels(
        run.line1_start, run.line1_end, num_windows1, w1, window_size, stride, use_flip
    )
    x2a, x2b = window_range_to_pixels(
        run.line2_start, run.line2_end, num_windows2, w2, window_size, stride, use_flip
    )

    ax.add_patch(Rectangle((x1a, y1_top), x1b - x1a, h1, facecolor=color, edgecolor=color, linewidth=2, alpha=0.30))
    ax.add_patch(Rectangle((x2a, y2_top), x2b - x2a, h2, facecolor=color, edgecolor=color, linewidth=2, alpha=0.30))

    cx1 = 0.5 * (x1a + x1b)
    cx2 = 0.5 * (x2a + x2b)
    ax.add_patch(
        FancyArrowPatch(
            (cx1, y1_bottom + 4),
            (cx2, y2_top - 4),
            arrowstyle="->",
            mutation_scale=18,
            linewidth=2.5,
            color=color,
            alpha=0.95,
        )
    )

    label = (
        f"longest consecutive SW match: {run.length} windows | "
        f"mean sim={run.mean_similarity:.3f} | SW score={run.sw_score:.3f}"
    )
    ax.text(0, y1_top - 7, "Line 1", fontsize=11, weight="bold")
    ax.text(0, y2_top - 7, "Line 2", fontsize=11, weight="bold")
    ax.text(
        canvas_w * 0.5,
        y1_bottom + gap * 0.5,
        label,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor=color, alpha=0.85, boxstyle="round,pad=0.3"),
    )

    ax.set_title(title, fontsize=13)
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(y2_bottom + 25, 0)
    ax.axis("off")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_debug_heatmap(sim: np.ndarray, H: np.ndarray, path, output: str):
    debug_output = os.path.splitext(output)[0] + "_heatmap.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(sim.T, origin="upper", aspect="auto", vmin=0, vmax=1)
    axes[0].set_title("Cosine similarity")
    axes[1].imshow(H[1:, 1:].T, origin="upper", aspect="auto")
    axes[1].set_title("Smith-Waterman score")
    diag_x = [i for op, i, j in path if op == "diag"]
    diag_y = [j for op, i, j in path if op == "diag"]
    if diag_x:
        axes[0].plot(diag_x, diag_y, color="white", linewidth=1.5)
        axes[1].plot(diag_x, diag_y, color="white", linewidth=1.5)
    for ax in axes:
        ax.set_xlabel("line1 windows")
        ax.set_ylabel("line2 windows")
    plt.tight_layout()
    plt.savefig(debug_output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved debug heatmap: {debug_output}")


# ---------------------------------------------------------------------------
# Inference runners
# ---------------------------------------------------------------------------


def infer_one_pair(
    model,
    line1: str,
    line2: str,
    output: str,
    args,
    sample_id: Optional[str] = None,
) -> Dict:
    img1, tensor1 = preprocess_line_image(line1, target_height=args.height)
    img2, tensor2 = preprocess_line_image(line2, target_height=args.height)

    emb1 = image_embeddings(model, tensor1, args.device, embedding_space=args.embedding_space)
    emb2 = image_embeddings(model, tensor2, args.device, embedding_space=args.embedding_space)
    sim = cosine_similarity_matrix(emb1, emb2)
    threshold = resolve_threshold(sim, args)

    sw_path, sw_score, H = smith_waterman(
        sim,
        threshold=threshold,
        gap_penalty=args.gap,
        match_reward=args.match,
        mismatch_penalty=args.mismatch,
    )
    run = longest_consecutive_diagonal_run(
        sw_path,
        sim,
        sw_score,
        min_run_length=args.min_run_length,
    )
    if run is None:
        raise RuntimeError(
            "Smith-Waterman found no consecutive diagonal aligned run. "
            "Try lowering --threshold, making --mismatch less negative, "
            "or lowering --min-run-length."
        )

    title_suffix = f" sample {sample_id}" if sample_id is not None else ""
    draw_longest_run(
        img1,
        img2,
        run,
        num_windows1=emb1.shape[0],
        num_windows2=emb2.shape[0],
        window_size=args.window_size,
        stride=args.stride,
        output=output,
        use_flip=args.use_flip,
        title=(
            f"Smith-Waterman longest consecutive local alignment{title_suffix} | "
            f"embedding={args.embedding_space}, thr={threshold:.3f}"
        ),
    )

    if args.heatmap:
        save_debug_heatmap(sim, H, sw_path, output)

    metadata = {
        "sample_id": sample_id,
        "line1": line1,
        "line2": line2,
        "output": output,
        "embedding_space": args.embedding_space,
        "threshold_used": float(threshold),
        "num_windows_line1": int(emb1.shape[0]),
        "num_windows_line2": int(emb2.shape[0]),
        "sw_path_length": int(len(sw_path)),
        "run_length": int(run.length),
        "mean_similarity": float(run.mean_similarity),
        "sw_score": float(run.sw_score),
    }

    print(
        f"[{sample_id or 'single'}] "
        f"embedding={args.embedding_space} threshold={threshold:.4f} "
        f"line1_windows={emb1.shape[0]} line2_windows={emb2.shape[0]} "
        f"sw_path={len(sw_path)} run_len={run.length} mean_sim={run.mean_similarity:.3f} "
        f"saved={output}",
        flush=True,
    )
    return metadata


def parse_indices(indices: Optional[str]) -> Optional[List[int]]:
    if indices is None or not indices.strip():
        return None
    values = []
    for part in indices.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start_i, end_i = int(start_s), int(end_s)
            step = 1 if end_i >= start_i else -1
            values.extend(range(start_i, end_i + step, step))
        else:
            values.append(int(part))
    return values


def discover_indices(data_dir: str) -> List[int]:
    images_dir = os.path.join(data_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    indices = []
    pattern = re.compile(r"^img1_(\d+)\.png$")
    for name in os.listdir(images_dir):
        match = pattern.match(name)
        if match:
            idx = int(match.group(1))
            if os.path.exists(os.path.join(images_dir, f"img2_{idx}.png")):
                indices.append(idx)
    return sorted(indices)


def resolve_batch_indices(args) -> List[int]:
    explicit = parse_indices(args.indices)
    if explicit is not None:
        return explicit

    all_indices = discover_indices(args.data_dir)
    if args.start_index is not None:
        all_indices = [idx for idx in all_indices if idx >= args.start_index]
    if args.n_samples is not None and args.n_samples > 0:
        all_indices = all_indices[: args.n_samples]
    return all_indices


def paths_for_index(data_dir: str, idx: int) -> Tuple[str, str]:
    images_dir = os.path.join(data_dir, "images")
    return (
        os.path.join(images_dir, f"img1_{idx}.png"),
        os.path.join(images_dir, f"img2_{idx}.png"),
    )


def run_batch(model, args):
    indices = resolve_batch_indices(args)
    if not indices:
        raise RuntimeError("No samples found for batch inference. Check --data-dir or pass --indices.")

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Running batch inference on {len(indices)} samples", flush=True)

    successes = []
    failures = []
    for pos, idx in enumerate(indices, start=1):
        line1, line2 = paths_for_index(args.data_dir, idx)
        output = os.path.join(args.output_dir, f"sw_longest_{idx}.png")
        if not os.path.exists(line1) or not os.path.exists(line2):
            message = f"missing image pair for index {idx}: {line1}, {line2}"
            failures.append({"sample_id": idx, "error": message})
            print(f"[{pos}/{len(indices)}] SKIP {message}", flush=True)
            if args.strict:
                raise FileNotFoundError(message)
            continue

        try:
            metadata = infer_one_pair(model, line1, line2, output, args, sample_id=str(idx))
            successes.append(metadata)
        except Exception as exc:
            failure = {"sample_id": idx, "line1": line1, "line2": line2, "error": str(exc)}
            failures.append(failure)
            print(f"[{pos}/{len(indices)}] FAILED sample {idx}: {exc}", flush=True)
            if args.strict:
                raise
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nBatch done")
    print(f"  requested: {len(indices)}")
    print(f"  success  : {len(successes)}")
    print(f"  failed   : {len(failures)}")
    if successes:
        run_lengths = [item["run_length"] for item in successes]
        mean_sims = [item["mean_similarity"] for item in successes]
        print(f"  mean run length: {float(np.mean(run_lengths)):.2f}")
        print(f"  mean similarity: {float(np.mean(mean_sims)):.3f}")
    if failures:
        print("  failed samples:")
        for failure in failures:
            print(f"    {failure['sample_id']}: {failure['error']}")

    return {"successes": successes, "failures": failures}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Smith-Waterman inference visualization: mask longest consecutive aligned part."
    )
    parser.add_argument("--weights", required=True, help="Path to trained model .pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--vector-size", type=int, default=128)
    parser.add_argument(
        "--embedding-space",
        choices=("contextual", "local"),
        default="contextual",
        help="contextual=CNN+BiLSTM for line alignment; local=pre-BiLSTM CNN for window discrimination.",
    )
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument(
        "--adaptive-threshold",
        choices=("none", "percentile", "mean_std"),
        default="none",
        help="Optionally derive a per-pair threshold from the similarity matrix.",
    )
    parser.add_argument("--threshold-percentile", type=float, default=90.0)
    parser.add_argument("--threshold-std-scale", type=float, default=1.0)
    parser.add_argument(
        "--no-adaptive-threshold-floor",
        dest="adaptive_threshold_floor",
        action="store_false",
        help="Use the adaptive threshold directly instead of max(fixed, adaptive).",
    )
    parser.set_defaults(adaptive_threshold_floor=True)
    parser.add_argument("--match", type=float, default=1.0, help="Reward scale for similarities above threshold")
    parser.add_argument("--mismatch", type=float, default=-1.0, help="Penalty scale for similarities below threshold; should be negative")
    parser.add_argument("--gap", type=float, default=-0.3, help="Smith-Waterman gap penalty; should be negative")
    parser.add_argument("--min-run-length", type=int, default=2)
    parser.add_argument("--use-flip", action="store_true", help="Use when Arabic windows were encoded right-to-left")
    parser.add_argument("--no-bilstm", action="store_true", help="Disable BiLSTM even if checkpoint config is missing")
    parser.add_argument("--heatmap", action="store_true", help="Also save similarity/SW debug heatmap")
    parser.add_argument("--strict", action="store_true", help="In batch mode, stop on first failed sample")

    # Single-pair options.
    parser.add_argument("--line1", default=None, help="Path to first line image")
    parser.add_argument("--line2", default=None, help="Path to second line image")
    parser.add_argument("--output", default=None, help="Output visualization path for single-pair mode")

    # Batch options.
    parser.add_argument("--batch", action="store_true", help="Run more than one sample")
    parser.add_argument("--data-dir", default="DataSet/Synthetic_Arabic")
    parser.add_argument("--indices", default=None, help="Comma/range list, e.g. 1,2,5-10")
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--output-dir", default="Results/Evaluation/SW_Longest")
    args = parser.parse_args()

    if args.gap > 0:
        raise ValueError("--gap should be negative for Smith-Waterman, for example --gap -0.3")
    if args.mismatch > 0:
        raise ValueError("--mismatch should be negative, for example --mismatch -2.0")
    if args.match <= 0:
        raise ValueError("--match should be positive, for example --match 1.0")

    model = load_image_model(
        args.weights,
        args.device,
        window_size=args.window_size,
        stride=args.stride,
        vector_size=args.vector_size,
        use_bilstm=False if args.no_bilstm else None,
        use_flip=args.use_flip,
    )

    if args.batch:
        run_batch(model, args)
        return

    if not args.line1 or not args.line2:
        raise ValueError("Single-pair mode requires --line1 and --line2. Use --batch for multiple samples.")
    output = args.output or "Results/Evaluation/sw_longest.png"
    infer_one_pair(model, args.line1, args.line2, output, args)


if __name__ == "__main__":
    main()
