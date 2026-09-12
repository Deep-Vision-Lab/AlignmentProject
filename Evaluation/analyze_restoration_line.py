#!/usr/bin/env python3
"""Window-by-window diagnostic for the restoration + positive-letter-DTW branch.

This script does NOT change training or alignment. It opens one manuscript line
and exposes the internal evidence for every 32-pixel window:

    pixels -> primitive P_i -> restoration -> semantic L_i
           -> letter similarities -> diagnostic hard DTW path

The hard DTW traceback is only an analysis view of the training objective.
Training itself used differentiable soft DTW.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default=str(ROOT / "DataSet" / "Synthetic63"))
    parser.add_argument("--index", type=int, default=132)
    parser.add_argument("--side", type=int, choices=(1, 2), default=1)
    parser.add_argument("--image", default="")
    parser.add_argument("--text", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--image-preprocessing",
        choices=("original", "training"),
        default="original",
    )
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def resolve_inputs(args):
    dataset = Path(args.dataset).expanduser().resolve()
    image = (
        Path(args.image).expanduser().resolve()
        if args.image
        else dataset / "images" / f"img{args.side}_{args.index}.png"
    )
    text = (
        Path(args.text).expanduser().resolve()
        if args.text
        else dataset / "texts" / f"text{args.side}_{args.index}.txt"
    )
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    if not text.is_file():
        raise FileNotFoundError(f"Transcript not found: {text}")
    return image, text


def deterministic_char_codebook(config: dict, device: torch.device):
    """Recreate OrthogonalCharEmbedding exactly without Parameters.device coupling."""
    dim = int(config.get("vector_size", 128))
    vocab = int(config.get("letter_codebook_vocab_size", 4096))
    seed = int(config.get("letter_codebook_seed", 1234))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.randn(vocab, dim, generator=generator)
    weight = F.normalize(weight, p=2, dim=-1)
    weight[1].zero_()
    # OrthogonalCharEmbedding applies a frozen LayerNorm with weight=1,bias=0.
    weight = F.layer_norm(weight, (dim,))
    return weight.to(device)


def char_index(character: str, vocab_size: int) -> int:
    if character == " ":
        return 0
    return (ord(character) % (vocab_size - 2)) + 2


def char_vectors(characters, codebook):
    vocab = int(codebook.shape[0])
    indices = [char_index(character, vocab) for character in characters]
    index_tensor = torch.tensor(indices, dtype=torch.long, device=codebook.device)
    return F.normalize(codebook.index_select(0, index_tensor).float(), p=2, dim=-1)


def unique_characters(text_letters):
    from vlm_restoration_positive_dtw import DEFAULT_ARABIC_LETTERS
    result = []
    seen = set()
    for character in list(text_letters) + list(DEFAULT_ARABIC_LETTERS):
        if character not in seen:
            seen.add(character)
            result.append(character)
    return result


def hard_monotonic_dtw(costs: np.ndarray, step_penalty: float):
    """Hard traceback with the same diag/vertical/horizontal transition topology."""
    cost = np.asarray(costs, dtype=np.float64)
    if cost.ndim != 2 or not cost.size:
        return [], float("nan")
    rows, cols = cost.shape
    dp = np.full((rows, cols), np.inf, dtype=np.float64)
    trace = np.full((rows, cols), -1, dtype=np.int8)
    dp[0, 0] = cost[0, 0]
    for i in range(1, rows):
        dp[i, 0] = dp[i - 1, 0] + float(step_penalty) + cost[i, 0]
        trace[i, 0] = 1  # vertical
    for j in range(1, cols):
        dp[0, j] = dp[0, j - 1] + float(step_penalty) + cost[0, j]
        trace[0, j] = 2  # horizontal
    for i in range(1, rows):
        for j in range(1, cols):
            options = (
                dp[i - 1, j - 1],
                dp[i - 1, j] + float(step_penalty),
                dp[i, j - 1] + float(step_penalty),
            )
            move = int(np.argmin(options))
            dp[i, j] = options[move] + cost[i, j]
            trace[i, j] = move

    path = []
    i, j = rows - 1, cols - 1
    while True:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        move = int(trace[i, j])
        if move == 0:
            i -= 1
            j -= 1
        elif move == 1:
            i -= 1
        elif move == 2:
            j -= 1
        else:
            raise RuntimeError(f"Invalid DTW traceback at {(i, j)}: {move}")
    path.reverse()
    return path, float(dp[-1, -1])


def offdiag_mean(matrix: np.ndarray, min_separation: int = 1):
    value = np.asarray(matrix, dtype=np.float64)
    n = value.shape[0]
    mask = np.ones_like(value, dtype=bool)
    for i in range(n):
        lo = max(0, i - min_separation + 1)
        hi = min(n, i + min_separation)
        mask[i, lo:hi] = False
    selected = value[mask]
    return float(selected.mean()) if selected.size else None


def effective_rank(vectors: np.ndarray):
    x = np.asarray(vectors, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(x, compute_uv=False)
    power = singular ** 2
    total = float(power.sum())
    if total <= 1e-12:
        return 0.0
    probs = power / total
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return float(np.exp(entropy))


def matrix_correlation(left: np.ndarray, right: np.ndarray):
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2:
        return None
    mask = ~np.eye(a.shape[0], dtype=bool) if a.shape[0] == a.shape[1] else np.ones_like(a, dtype=bool)
    av, bv = a[mask], b[mask]
    if av.size < 2 or np.std(av) <= 1e-8 or np.std(bv) <= 1e-8:
        return None
    return float(np.corrcoef(av, bv)[0, 1])


def edge_mae(pred: np.ndarray, target: np.ndarray):
    px = np.diff(pred, axis=1)
    tx = np.diff(target, axis=1)
    py = np.diff(pred, axis=0)
    ty = np.diff(target, axis=0)
    return 0.5 * (float(np.mean(np.abs(px - tx))) + float(np.mean(np.abs(py - ty))))


def save_matrix(path: Path, matrix, title, xlabel, ylabel, path_pairs=None):
    value = np.asarray(matrix, dtype=np.float32)
    width = max(10.0, min(24.0, value.shape[1] * 0.35))
    height = max(6.0, min(22.0, value.shape[0] * 0.24))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(value, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if path_pairs:
        xs = [j for i, j in path_pairs]
        ys = [i for i, j in path_pairs]
        ax.plot(xs, ys, linewidth=1.5)
        ax.scatter(xs, ys, s=8)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_metric_plot(path: Path, values, title, ylabel):
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(np.arange(len(values)), values, marker="o", markersize=2)
    ax.set_title(title)
    ax.set_xlabel("logical window index (Arabic reading order)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def to_uint8_patch(prepared_rgb: np.ndarray, logical_index, n_windows, window_size, stride, use_flip):
    physical_index = n_windows - 1 - logical_index if use_flip else logical_index
    x0 = physical_index * stride
    x1 = x0 + window_size
    return prepared_rgb[:, x0:x1].copy(), int(x0), int(x1)


def save_contact_sheet(path, prepared_rgb, n_windows, window_size, stride, use_flip, labels):
    columns = 7
    rows = int(math.ceil(n_windows / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(15, 2.2 * rows))
    axes = np.atleast_1d(axes).reshape(rows, columns)
    for index, ax in enumerate(axes.reshape(-1)):
        ax.axis("off")
        if index >= n_windows:
            continue
        patch, x0, x1 = to_uint8_patch(
            prepared_rgb, index, n_windows, window_size, stride, use_flip
        )
        ax.imshow(patch)
        ax.set_title(f"W{index}  x={x0}:{x1}\n{labels[index]}", fontsize=8)
    fig.suptitle("All model windows in logical Arabic reading order", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_window_card(
    path,
    patch,
    restoration_target,
    restoration_prediction,
    window_index,
    x0,
    x1,
    ink,
    top_letters,
    top_scores,
    dtw_letters,
    dtw_indices,
    p_norm,
    l_norm,
    p_l_cos,
    recon_mae,
    recon_edge_mae,
):
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(patch)
    axes[0].set_title(f"W{window_index} original\nx={x0}:{x1}, ink={ink:.3f}")
    axes[0].axis("off")

    axes[1].imshow(restoration_target, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title("restoration target")
    axes[1].axis("off")

    axes[2].imshow(restoration_prediction, cmap="gray", vmin=0.0, vmax=1.0)
    axes[2].set_title(
        f"reconstruction\nMAE={recon_mae:.3f}, edge={recon_edge_mae:.3f}"
    )
    axes[2].axis("off")

    labels = [f"{char}  {score:.3f}" for char, score in zip(top_letters, top_scores)]
    y = np.arange(len(labels))
    axes[3].barh(y, top_scores)
    axes[3].set_yticks(y)
    axes[3].set_yticklabels(labels)
    axes[3].invert_yaxis()
    axes[3].set_xlim(-0.2, 1.0)
    assigned = "".join(dtw_letters) if dtw_letters else "—"
    assigned_idx = ",".join(map(str, dtw_indices)) if dtw_indices else "—"
    axes[3].set_title(
        f"semantic letter scores\nDTW→ {assigned} [{assigned_idx}]\n"
        f"|Praw|={p_norm:.2f} |L|={l_norm:.2f} cos(P,L)={p_l_cos:.3f}"
    )
    axes[3].set_xlabel("cosine")
    axes[3].grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    args = parse_args()
    image_path, text_path = resolve_inputs(args)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    windows_dir = output / "windows"
    windows_dir.mkdir()

    from Evaluation.yelda_runtime import (
        configure_image_preprocessing,
        load_visual_models,
        read_checkpoint,
    )
    from Evaluation.yelda_geometry import prepare_line
    from vlm_restoration_positive_dtw import _clean_letters

    checkpoint = read_checkpoint(Path(args.weights).expanduser().resolve())
    models = load_visual_models(checkpoint, args.device, "restoration")
    configure_image_preprocessing(models, args.image_preprocessing)

    prepared, geometry = prepare_line(
        image_path,
        "synthetic",
        args.image_preprocessing,
    )
    prepared.save(output / "model_input_line.png")
    prepared_rgb = np.asarray(prepared.convert("RGB"), dtype=np.uint8)
    tensor = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )(prepared).unsqueeze(0).to(models.device)

    with torch.inference_mode():
        bundle = models.image_model(tensor, return_training_bundle=True)

    primitive_raw = bundle["primitive_raw"][0].float()
    primitive = bundle["primitive"][0].float()
    semantic = bundle["semantic"][0].float()
    ink = bundle["ink"][0].float()
    reconstruction = bundle["restoration"][0, :, 0].float()
    restoration_target = bundle["restoration_target"][0, :, 0].float()

    n_windows = int(semantic.shape[0])
    window_size = int(models.image_model.window_size)
    stride = int(models.image_model.stride)
    use_flip = bool(models.image_model.use_flip)

    transcript = text_path.read_text(encoding="utf-8").strip()
    letters = _clean_letters(transcript)
    if not letters:
        raise ValueError("Transcript has no Arabic letters after training cleaner")

    config = models.config
    codebook = deterministic_char_codebook(config, models.device)
    transcript_vectors = char_vectors(letters, codebook)
    semantic_unit = F.normalize(semantic, p=2, dim=-1)
    primitive_unit = F.normalize(primitive, p=2, dim=-1)

    transcript_similarity = semantic_unit @ transcript_vectors.T
    costs = 1.0 - transcript_similarity
    min_ink = float(config.get("positive_letter_dtw_min_ink", 0.01))
    valid_mask = ink >= min_ink
    if not bool(valid_mask.any()):
        valid_mask = torch.ones_like(ink, dtype=torch.bool)
    valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
    valid_costs = costs.index_select(0, valid_indices)
    step_penalty = float(config.get("positive_letter_dtw_step_penalty", 0.02))
    compact_path, hard_cost = hard_monotonic_dtw(
        valid_costs.detach().cpu().numpy(),
        step_penalty,
    )
    full_path = [
        (int(valid_indices[i].item()), int(j))
        for i, j in compact_path
    ]

    dtw_by_window = {index: [] for index in range(n_windows)}
    for window_index, letter_index in full_path:
        if letter_index not in dtw_by_window[window_index]:
            dtw_by_window[window_index].append(letter_index)

    candidates = unique_characters(letters)
    candidate_vectors = char_vectors(candidates, codebook)
    candidate_similarity = semantic_unit @ candidate_vectors.T
    top_k = min(max(1, int(args.top_k)), len(candidates))
    top_scores, top_indices = torch.topk(candidate_similarity, k=top_k, dim=1)

    primitive_cosine = (primitive_unit @ primitive_unit.T).detach().cpu().numpy()
    semantic_cosine = (semantic_unit @ semantic_unit.T).detach().cpu().numpy()
    transcript_similarity_np = transcript_similarity.detach().cpu().numpy()
    primitive_raw_np = primitive_raw.detach().cpu().numpy()
    primitive_np = primitive.detach().cpu().numpy()
    semantic_np = semantic.detach().cpu().numpy()

    np.savetxt(output / "primitive_raw_vectors.csv", primitive_raw_np, delimiter=",", fmt="%.8f")
    np.savetxt(output / "primitive_vectors.csv", primitive_np, delimiter=",", fmt="%.8f")
    np.savetxt(output / "semantic_vectors.csv", semantic_np, delimiter=",", fmt="%.8f")
    np.savetxt(output / "window_transcript_cosine.csv", transcript_similarity_np, delimiter=",", fmt="%.8f")

    save_matrix(
        output / "01_primitive_raw_vector_values.png",
        primitive_raw_np,
        "Primitive raw vectors P_i: every row is one window",
        "feature dimension (0..127)",
        "window index",
    )
    save_matrix(
        output / "02_semantic_vector_values.png",
        semantic_np,
        "Semantic vectors L_i: every row is one window",
        "feature dimension (0..127)",
        "window index",
    )
    save_matrix(
        output / "03_primitive_window_cosine.png",
        primitive_cosine,
        "Primitive P_i ↔ P_j cosine within this line",
        "window j",
        "window i",
    )
    save_matrix(
        output / "04_semantic_window_cosine.png",
        semantic_cosine,
        "Semantic L_i ↔ L_j cosine within this line",
        "window j",
        "window i",
    )
    save_matrix(
        output / "05_window_transcript_cosine_dtw.png",
        transcript_similarity_np,
        "Semantic window ↔ transcript-letter cosine (hard diagnostic DTW overlaid)",
        "transcript letter index",
        "window index",
        full_path,
    )

    pred_np = reconstruction.detach().cpu().numpy()
    target_np = restoration_target.detach().cpu().numpy()
    stroke_flat = target_np.reshape(n_windows, -1)
    stroke_norm = stroke_flat / np.clip(
        np.linalg.norm(stroke_flat, axis=1, keepdims=True), 1e-8, None
    )
    stroke_cosine = stroke_norm @ stroke_norm.T
    stroke_primitive_correlation = matrix_correlation(
        stroke_cosine, primitive_cosine
    )
    stroke_semantic_correlation = matrix_correlation(
        stroke_cosine, semantic_cosine
    )
    ink_np = ink.detach().cpu().numpy()
    p_raw_norm = torch.linalg.vector_norm(primitive_raw, dim=-1).detach().cpu().numpy()
    l_norm = torch.linalg.vector_norm(semantic, dim=-1).detach().cpu().numpy()
    p_l_cos = F.cosine_similarity(primitive, semantic, dim=-1).detach().cpu().numpy()
    reconstruction_mae = np.mean(np.abs(pred_np - target_np), axis=(1, 2))
    reconstruction_edge_mae = np.asarray(
        [edge_mae(pred_np[i], target_np[i]) for i in range(n_windows)],
        dtype=np.float32,
    )
    top_scores_np = top_scores.detach().cpu().numpy()
    top_indices_np = top_indices.detach().cpu().numpy()
    top_margin = (
        top_scores_np[:, 0] - top_scores_np[:, 1]
        if top_k > 1
        else top_scores_np[:, 0]
    )

    save_metric_plot(output / "06_ink_ratio.png", ink_np, "Ink ratio per window", "ink ratio")
    save_metric_plot(
        output / "07_restoration_mae.png",
        reconstruction_mae,
        "Restoration error per window",
        "mean absolute error",
    )
    save_metric_plot(
        output / "08_top_letter_cosine.png",
        top_scores_np[:, 0],
        "Best Arabic-letter cosine per window",
        "cosine",
    )
    save_metric_plot(
        output / "09_top_letter_margin.png",
        top_margin,
        "Top-1 minus Top-2 Arabic-letter cosine margin",
        "margin",
    )
    save_metric_plot(
        output / "10_primitive_semantic_cosine.png",
        p_l_cos,
        "How much semantic adapter changes each primitive",
        "cos(P_i, L_i)",
    )

    rows = []
    labels_for_sheet = []
    with (output / "window_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "window",
            "physical_x0",
            "physical_x1",
            "ink_ratio",
            "used_by_training_dtw",
            "primitive_raw_norm",
            "semantic_norm",
            "primitive_semantic_cosine",
            "restoration_mae",
            "restoration_edge_mae",
            "top1_letter",
            "top1_cosine",
            "top2_letter",
            "top2_cosine",
            "top1_margin",
            "dtw_letter_indices",
            "dtw_letters",
            "dtw_path_cosine_mean",
            "previous_primitive_cosine",
            "next_primitive_cosine",
            "previous_semantic_cosine",
            "next_semantic_cosine",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for index in range(n_windows):
            patch, x0, x1 = to_uint8_patch(
                prepared_rgb, index, n_windows, window_size, stride, use_flip
            )
            rank_indices = top_indices_np[index]
            rank_scores = top_scores_np[index]
            rank_letters = [candidates[int(value)] for value in rank_indices]
            assigned_indices = dtw_by_window.get(index, [])
            assigned_letters = [letters[value] for value in assigned_indices]
            path_cosines = [
                float(transcript_similarity_np[index, value])
                for value in assigned_indices
            ]
            row = {
                "window": index,
                "physical_x0": x0,
                "physical_x1": x1,
                "ink_ratio": float(ink_np[index]),
                "used_by_training_dtw": bool(valid_mask[index].item()),
                "primitive_raw_norm": float(p_raw_norm[index]),
                "semantic_norm": float(l_norm[index]),
                "primitive_semantic_cosine": float(p_l_cos[index]),
                "restoration_mae": float(reconstruction_mae[index]),
                "restoration_edge_mae": float(reconstruction_edge_mae[index]),
                "top1_letter": rank_letters[0],
                "top1_cosine": float(rank_scores[0]),
                "top2_letter": rank_letters[1] if top_k > 1 else "",
                "top2_cosine": float(rank_scores[1]) if top_k > 1 else "",
                "top1_margin": float(top_margin[index]),
                "dtw_letter_indices": " ".join(map(str, assigned_indices)),
                "dtw_letters": "".join(assigned_letters),
                "dtw_path_cosine_mean": (
                    float(np.mean(path_cosines)) if path_cosines else ""
                ),
                "previous_primitive_cosine": (
                    float(primitive_cosine[index, index - 1]) if index > 0 else ""
                ),
                "next_primitive_cosine": (
                    float(primitive_cosine[index, index + 1])
                    if index + 1 < n_windows
                    else ""
                ),
                "previous_semantic_cosine": (
                    float(semantic_cosine[index, index - 1]) if index > 0 else ""
                ),
                "next_semantic_cosine": (
                    float(semantic_cosine[index, index + 1])
                    if index + 1 < n_windows
                    else ""
                ),
            }
            writer.writerow(row)
            rows.append(row)

            assigned_display = "".join(assigned_letters) if assigned_letters else "—"
            labels_for_sheet.append(
                f"top={rank_letters[0]} {rank_scores[0]:.2f}\nDTW={assigned_display}"
            )
            save_window_card(
                windows_dir / f"window_{index:03d}.png",
                patch,
                target_np[index],
                pred_np[index],
                index,
                x0,
                x1,
                float(ink_np[index]),
                rank_letters,
                rank_scores,
                assigned_letters,
                assigned_indices,
                float(p_raw_norm[index]),
                float(l_norm[index]),
                float(p_l_cos[index]),
                float(reconstruction_mae[index]),
                float(reconstruction_edge_mae[index]),
            )

    save_contact_sheet(
        output / "00_all_windows_contact_sheet.png",
        prepared_rgb,
        n_windows,
        window_size,
        stride,
        use_flip,
        labels_for_sheet,
    )

    adjacent_primitive = [
        float(primitive_cosine[i, i + 1]) for i in range(n_windows - 1)
    ]
    adjacent_semantic = [
        float(semantic_cosine[i, i + 1]) for i in range(n_windows - 1)
    ]
    path_cosines = [
        float(transcript_similarity_np[i, j]) for i, j in full_path
    ]

    summary = {
        "image": str(image_path),
        "text": str(text_path),
        "transcript": transcript,
        "clean_letters": "".join(letters),
        "window_size": window_size,
        "stride": stride,
        "window_count": n_windows,
        "feature_dim": int(semantic.shape[-1]),
        "use_flip_arabic_reading_order": use_flip,
        "image_preprocessing": args.image_preprocessing,
        "geometry": geometry,
        "positive_letter_dtw_min_ink": min_ink,
        "positive_letter_dtw_step_penalty": step_penalty,
        "training_dtw_windows_used": int(valid_mask.sum().item()),
        "transcript_letter_count": len(letters),
        "diagnostic_hard_dtw_cost": hard_cost,
        "diagnostic_hard_dtw_path_steps": len(full_path),
        "mean_dtw_path_cosine": float(np.mean(path_cosines)) if path_cosines else None,
        "mean_restoration_mae": float(np.mean(reconstruction_mae)),
        "mean_restoration_edge_mae": float(np.mean(reconstruction_edge_mae)),
        "mean_top1_letter_cosine": float(np.mean(top_scores_np[:, 0])),
        "mean_top1_letter_margin": float(np.mean(top_margin)),
        "mean_primitive_semantic_cosine": float(np.mean(p_l_cos)),
        "mean_adjacent_primitive_cosine": float(np.mean(adjacent_primitive)),
        "mean_adjacent_semantic_cosine": float(np.mean(adjacent_semantic)),
        "mean_nonlocal_primitive_cosine_sep5": offdiag_mean(primitive_cosine, 5),
        "mean_nonlocal_semantic_cosine_sep5": offdiag_mean(semantic_cosine, 5),
        "primitive_effective_rank": effective_rank(primitive_np),
        "semantic_effective_rank": effective_rank(semantic_np),
        "stroke_primitive_similarity_correlation": stroke_primitive_correlation,
        "stroke_semantic_similarity_correlation": stroke_semantic_correlation,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    guide = f"""RESTORATION WINDOW DIAGNOSTIC
================================

Line:
  image: {image_path}
  text : {text_path}

Read these files in this order:

1. 00_all_windows_contact_sheet.png
   Confirms exactly what each logical window contains. For Arabic, W0 is the
   rightmost physical window when use_flip=true.

2. windows/window_XXX.png
   For every window:
   - original RGB pixels
   - restoration target
   - reconstruction from primitive P_i
   - semantic top-{top_k} Arabic letters
   - diagnostic DTW-assigned transcript letter(s)

3. 01_primitive_raw_vector_values.png
   Every row is the 128-D raw primitive that feeds the restoration decoder.

4. 03_primitive_window_cosine.png
   If almost every window is highly similar to every other window, P_i is
   collapsed / insufficiently discriminative.

5. 07_restoration_mae.png
   High error on ink-heavy windows means the problem already exists in P_i.

6. 02_semantic_vector_values.png and 04_semantic_window_cosine.png
   Show what the semantic adapter does. If primitive vectors are distinct but
   semantic vectors become nearly identical, the semantic adapter is collapsing.

7. 05_window_transcript_cosine_dtw.png
   This is closest to the training supervision. A healthy model should show a
   monotonic ridge rather than broad columns/rows of equally high similarity.

8. window_analysis.csv
   Numerical evidence for every window.

Warning signs:
- poor restoration + weak primitive rank -> primitive extractor problem
- good restoration, semantic cosine matrix nearly uniform -> semantic adapter problem
- semantic vectors distinct but top-letter margins near zero -> weak letter grounding
- training transcript matrix looks good but image-image NW is bad -> cross-image
  invariance/evaluation mismatch, not primitive extraction

The hard DTW shown here is DIAGNOSTIC ONLY. Training used differentiable soft DTW.
"""
    (output / "README.txt").write_text(guide, encoding="utf-8")

    print("=" * 72)
    print("Restoration window diagnostic")
    print(f"image                    : {image_path}")
    print(f"transcript letters       : {len(letters)}")
    print(f"windows                  : {n_windows}")
    print(f"training-DTW windows used: {int(valid_mask.sum().item())}")
    print(f"mean restoration MAE     : {summary['mean_restoration_mae']:.4f}")
    print(f"primitive effective rank : {summary['primitive_effective_rank']:.2f}")
    print(f"semantic effective rank  : {summary['semantic_effective_rank']:.2f}")
    print(
        "stroke↔primitive corr    : "
        f"{summary['stroke_primitive_similarity_correlation'] if summary['stroke_primitive_similarity_correlation'] is not None else float('nan'):.4f}"
    )
    print(f"mean top-1 letter cosine : {summary['mean_top1_letter_cosine']:.4f}")
    print(f"mean top-1 letter margin : {summary['mean_top1_letter_margin']:.4f}")
    print(f"mean DTW path cosine     : {summary['mean_dtw_path_cosine']:.4f}")
    print(f"output                   : {output}")
    print("=" * 72)


if __name__ == "__main__":
    main()
