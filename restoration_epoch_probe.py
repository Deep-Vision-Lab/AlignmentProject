"""Fixed-line epoch diagnostics for restoration + positive letter-DTW training.

The probe intentionally runs on the same first validation line after every epoch.
It measures:
- primitive reconstruction quality;
- primitive/semantic diversity;
- window->transcript-letter cosine structure and a hard diagnostic DTW path;
- change in the matrix/path from the previous epoch;
- change in patch-embedding weights;
- DTW vs restoration gradient norms/cosine on the primitive Conv2D.

It never contributes to the optimizer update.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def _clean_letters(text: str):
    from vlm_restoration_positive_dtw import _clean_letters
    return _clean_letters(text)


def _hard_dtw(
    costs: np.ndarray,
    vertical_penalty: float,
    horizontal_penalty: float | None = None,
    position_prior_weight: float = 0.0,
):
    costs = np.asarray(costs, dtype=np.float64)
    rows, cols = costs.shape
    if horizontal_penalty is None:
        horizontal_penalty = float(vertical_penalty)
    work = costs.copy()
    if position_prior_weight > 0 and rows > 1 and cols > 1:
        image_position = np.linspace(0.0, 1.0, rows)[:, None]
        text_position = np.linspace(0.0, 1.0, cols)[None, :]
        work += float(position_prior_weight) * np.abs(
            image_position - text_position
        )

    dp = np.full((rows, cols), np.inf, dtype=np.float64)
    trace = np.full((rows, cols), -1, dtype=np.int8)
    dp[0, 0] = work[0, 0]
    for i in range(1, rows):
        dp[i, 0] = (
            dp[i - 1, 0] + float(vertical_penalty) + work[i, 0]
        )
        trace[i, 0] = 1
    for j in range(1, cols):
        dp[0, j] = (
            dp[0, j - 1] + float(horizontal_penalty) + work[0, j]
        )
        trace[0, j] = 2
    for i in range(1, rows):
        for j in range(1, cols):
            previous = (
                dp[i - 1, j - 1],
                dp[i - 1, j] + float(vertical_penalty),
                dp[i, j - 1] + float(horizontal_penalty),
            )
            move = int(np.argmin(previous))
            dp[i, j] = previous[move] + work[i, j]
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
            raise RuntimeError(f"invalid DTW traceback at {(i, j)}")
    path.reverse()
    return path, float(dp[-1, -1])


def _effective_rank(vectors: torch.Tensor) -> float:
    x = vectors.detach().float().cpu()
    x = x - x.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(x)
    power = singular.square()
    total = power.sum()
    if float(total) <= 1e-12:
        return 0.0
    p = power / total
    entropy = -(p * torch.log(p + 1e-12)).sum()
    return float(torch.exp(entropy))


def _offdiag_mean(matrix: torch.Tensor, separation: int = 5) -> float:
    value = matrix.detach().float().cpu().numpy()
    rows, cols = value.shape
    mask = np.ones_like(value, dtype=bool)
    for i in range(min(rows, cols)):
        lo = max(0, i - separation + 1)
        hi = min(cols, i + separation)
        mask[i, lo:hi] = False
    selected = value[mask]
    return float(selected.mean()) if selected.size else float("nan")


def _matrix_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.detach().float().cpu().numpy()
    b = right.detach().float().cpu().numpy()
    if a.shape != b.shape or a.ndim != 2:
        return float("nan")
    mask = ~np.eye(a.shape[0], dtype=bool) if a.shape[0] == a.shape[1] else np.ones_like(a, dtype=bool)
    av = a[mask]
    bv = b[mask]
    if av.size < 2 or np.std(av) <= 1e-8 or np.std(bv) <= 1e-8:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def _save_heatmap(path: Path, matrix, title: str, xlabel: str, ylabel: str, path_pairs=None):
    value = np.asarray(matrix, dtype=np.float32)
    fig, ax = plt.subplots(
        figsize=(
            max(9.0, min(24.0, value.shape[1] * 0.32)),
            max(5.0, min(18.0, value.shape[0] * 0.22)),
        )
    )
    image = ax.imshow(value, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if path_pairs:
        ax.plot(
            [j for i, j in path_pairs],
            [i for i, j in path_pairs],
            linewidth=1.5,
        )
        ax.scatter(
            [j for i, j in path_pairs],
            [i for i, j in path_pairs],
            s=7,
        )
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _extract_probe_sample(valid_loader):
    batch = next(iter(valid_loader))
    if isinstance(batch, dict):
        return batch["images1"][:1], str(batch["texts1"][0]), "line1"
    images, texts, _ = batch
    return images[:1], str(texts[0]), "line1"


def _settings(config: dict):
    return SimpleNamespace(
        positive_letter_dtw_gamma=float(config.get("positive_letter_dtw_gamma", 0.05)),
        positive_letter_dtw_step_penalty=float(
            config.get("positive_letter_dtw_step_penalty", 0.05)
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
        positive_letter_dtw_competition_temperature=float(
            config.get("positive_letter_dtw_competition_temperature", 0.10)
        ),
        positive_letter_dtw_cost_mode=str(
            config.get("positive_letter_dtw_cost_mode", "full_alphabet_nll")
        ),
        positive_letter_dtw_min_ink=float(
            config.get("positive_letter_dtw_min_ink", 0.01)
        ),
        restoration_pixel_weight=float(config.get("restoration_pixel_weight", 1.0)),
        restoration_edge_weight=float(config.get("restoration_edge_weight", 0.5)),
        restoration_foreground_weight=float(
            config.get("restoration_foreground_weight", 2.0)
        ),
    )


def _loss_gradients(model, text_encoder, image, text, config):
    from vlm_restoration_positive_dtw import (
        positive_letter_dtw_loss,
        stroke_restoration_loss,
    )

    settings = _settings(config)
    parameter = next(model.vit_encoder.patch_embedding.parameters())
    model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        bundle = model(image, return_training_bundle=True)
        dtw, _ = positive_letter_dtw_loss(
            settings,
            text_encoder,
            bundle["semantic"],
            bundle["ink"],
            [text],
        )
        restoration, _, _ = stroke_restoration_loss(
            settings,
            bundle["restoration"],
            bundle["restoration_target"],
        )
        dtw_grad = torch.autograd.grad(
            dtw,
            parameter,
            retain_graph=True,
            allow_unused=True,
        )[0]
        restoration_grad = torch.autograd.grad(
            restoration,
            parameter,
            retain_graph=False,
            allow_unused=True,
        )[0]

    def norm(gradient):
        return float(gradient.float().norm()) if gradient is not None else 0.0

    dtw_norm = norm(dtw_grad)
    restoration_norm = norm(restoration_grad)
    gradient_cosine = None
    if (
        dtw_grad is not None
        and restoration_grad is not None
        and dtw_norm > 1e-12
        and restoration_norm > 1e-12
    ):
        gradient_cosine = float(
            F.cosine_similarity(
                dtw_grad.float().reshape(1, -1),
                restoration_grad.float().reshape(1, -1),
                dim=-1,
            ).item()
        )
    model.zero_grad(set_to_none=True)
    return {
        "probe_dtw_loss": float(dtw.detach()),
        "probe_restoration_loss": float(restoration.detach()),
        "dtw_patch_grad_norm": dtw_norm,
        "restoration_patch_grad_norm": restoration_norm,
        "dtw_restoration_grad_cosine": gradient_cosine,
    }


def _append_history(path: Path, metrics: dict):
    fieldnames = list(metrics.keys())
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(metrics)


def run_epoch_probe(
    *,
    model,
    text_encoder,
    valid_loader,
    epoch: int,
    job_id: str,
    config: dict,
    weights_root: Path,
    device,
    previous_patch_weight=None,
    previous_matrix=None,
    previous_training_cost=None,
    previous_path=None,
):
    """Run the fixed-line probe and return state needed for next epoch."""
    output_root = Path(weights_root) / job_id / "epoch_diagnostics"
    epoch_dir = output_root / f"epoch_{int(epoch):03d}"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()
    text_encoder.eval()

    images, text, side = _extract_probe_sample(valid_loader)
    image = images.to(device, non_blocking=False)
    letters = _clean_letters(text)
    if not letters:
        raise RuntimeError("Fixed epoch probe transcript has no Arabic letters")

    with torch.inference_mode():
        bundle = model(image, return_training_bundle=True)
        target = text_encoder("".join(letters)).detach().to(device)
        target = F.normalize(target.float(), p=2, dim=-1)
        semantic = F.normalize(bundle["semantic"][0].float(), p=2, dim=-1)
        primitive = F.normalize(bundle["primitive"][0].float(), p=2, dim=-1)
        ink = bundle["ink"][0].float()
        valid = ink >= float(config.get("positive_letter_dtw_min_ink", 0.01))
        if not bool(valid.any()):
            valid = torch.ones_like(ink, dtype=torch.bool)
        valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
        matrix = semantic @ target.T
        compact_matrix = matrix.index_select(0, valid_indices)
        from vlm_restoration_positive_dtw import letter_dtw_cost_matrix
        settings = _settings(config)
        training_cost = letter_dtw_cost_matrix(
            settings,
            text_encoder,
            bundle["semantic"][0].float(),
            letters,
        )
        compact_training_cost = training_cost.index_select(0, valid_indices)
        primitive_similarity = primitive @ primitive.T
        semantic_similarity = semantic @ semantic.T
        reconstruction = bundle["restoration"][0, :, 0].float()
        restoration_target = bundle["restoration_target"][0, :, 0].float()
        stroke_flat = F.normalize(
            restoration_target.flatten(start_dim=1).float(), p=2, dim=-1
        )
        stroke_similarity = stroke_flat @ stroke_flat.T

    compact_path, hard_cost = _hard_dtw(
        compact_training_cost.detach().cpu().numpy(),
        float(config.get("positive_letter_dtw_vertical_penalty", 0.05)),
        float(config.get("positive_letter_dtw_horizontal_penalty", 0.30)),
        float(config.get("positive_letter_dtw_position_prior", 0.15)),
    )
    full_path = [
        (int(valid_indices[i].item()), int(j))
        for i, j in compact_path
    ]

    path_values = [float(matrix[i, j]) for i, j in full_path]
    top_values = torch.topk(matrix, k=min(2, matrix.shape[1]), dim=1).values
    top1 = top_values[:, 0]
    margin = (
        top_values[:, 0] - top_values[:, 1]
        if top_values.shape[1] > 1
        else top_values[:, 0]
    )

    recon_mae = float((reconstruction - restoration_target).abs().mean())
    primitive_rank = _effective_rank(bundle["primitive"][0])
    semantic_rank = _effective_rank(bundle["semantic"][0])
    primitive_nonlocal = _offdiag_mean(primitive_similarity, 5)
    semantic_nonlocal = _offdiag_mean(semantic_similarity, 5)
    stroke_primitive_correlation = _matrix_correlation(
        stroke_similarity, primitive_similarity
    )
    stroke_semantic_correlation = _matrix_correlation(
        stroke_similarity, semantic_similarity
    )

    current_matrix = matrix.detach().cpu().numpy().astype(np.float32)
    current_training_cost = (
        training_cost.detach().cpu().numpy().astype(np.float32)
    )
    current_path = {(int(i), int(j)) for i, j in full_path}
    matrix_delta = None
    training_cost_delta = None
    matrix_correlation = None
    path_jaccard = None
    if previous_matrix is not None and previous_matrix.shape == current_matrix.shape:
        matrix_delta = float(np.mean(np.abs(current_matrix - previous_matrix)))
        left = previous_matrix.reshape(-1)
        right = current_matrix.reshape(-1)
        if np.std(left) > 1e-8 and np.std(right) > 1e-8:
            matrix_correlation = float(np.corrcoef(left, right)[0, 1])
    if (
        previous_training_cost is not None
        and previous_training_cost.shape == current_training_cost.shape
    ):
        training_cost_delta = float(
            np.mean(np.abs(current_training_cost - previous_training_cost))
        )
    if previous_path is not None:
        union = current_path | set(previous_path)
        path_jaccard = (
            float(len(current_path & set(previous_path)) / len(union))
            if union
            else 1.0
        )

    current_patch = (
        next(model.vit_encoder.patch_embedding.parameters())
        .detach()
        .float()
        .cpu()
        .clone()
    )
    patch_delta = None
    if previous_patch_weight is not None:
        difference = current_patch - previous_patch_weight
        patch_delta = float(difference.norm() / previous_patch_weight.norm().clamp_min(1e-12))

    gradient_metrics = _loss_gradients(
        model, text_encoder, image, text, config
    )

    np.save(epoch_dir / "window_letter_cosine.npy", current_matrix)
    np.savetxt(
        epoch_dir / "window_letter_cosine.csv",
        current_matrix,
        delimiter=",",
        fmt="%.8f",
    )
    training_cost_np = (
        training_cost.detach().cpu().numpy().astype(np.float32)
    )
    np.save(epoch_dir / "window_letter_training_cost.npy", training_cost_np)
    np.savetxt(
        epoch_dir / "window_letter_training_cost.csv",
        training_cost_np,
        delimiter=",",
        fmt="%.8f",
    )
    np.save(
        epoch_dir / "primitive_window_cosine.npy",
        primitive_similarity.detach().cpu().numpy().astype(np.float32),
    )
    np.save(
        epoch_dir / "semantic_window_cosine.npy",
        semantic_similarity.detach().cpu().numpy().astype(np.float32),
    )
    _save_heatmap(
        epoch_dir / "window_letter_dtw.png",
        current_matrix,
        f"Epoch {epoch}: semantic window -> transcript letter cosine",
        "transcript letter index",
        "window index",
        full_path,
    )
    _save_heatmap(
        epoch_dir / "window_letter_training_cost_dtw.png",
        training_cost_np,
        f"Epoch {epoch}: actual DTW training cost (hard path overlaid)",
        "transcript letter index",
        "window index",
        full_path,
    )
    _save_heatmap(
        epoch_dir / "primitive_window_cosine.png",
        primitive_similarity.detach().cpu().numpy(),
        f"Epoch {epoch}: primitive window self-similarity",
        "window j",
        "window i",
    )
    _save_heatmap(
        epoch_dir / "semantic_window_cosine.png",
        semantic_similarity.detach().cpu().numpy(),
        f"Epoch {epoch}: semantic window self-similarity",
        "window j",
        "window i",
    )

    horizontal_steps = sum(
        1
        for (i0, j0), (i1, j1) in zip(full_path[:-1], full_path[1:])
        if i1 == i0 and j1 == j0 + 1
    )
    vertical_steps = sum(
        1
        for (i0, j0), (i1, j1) in zip(full_path[:-1], full_path[1:])
        if i1 == i0 + 1 and j1 == j0
    )
    max_letters_same_window = 1
    current_run = 1
    for (i0, _j0), (i1, _j1) in zip(full_path[:-1], full_path[1:]):
        if i1 == i0:
            current_run += 1
            max_letters_same_window = max(max_letters_same_window, current_run)
        else:
            current_run = 1

    metrics = {
        "epoch": int(epoch),
        "probe_side": side,
        "semantic_adapter": str(
            config.get("restoration_semantic_adapter", "residual_mlp")
        ),
        "windows": int(matrix.shape[0]),
        "letters": int(matrix.shape[1]),
        "ink_windows": int(valid.sum().item()),
        "hard_dtw_cost": float(hard_cost),
        "hard_dtw_horizontal_steps": int(horizontal_steps),
        "hard_dtw_vertical_steps": int(vertical_steps),
        "hard_dtw_max_letters_same_window": int(max_letters_same_window),
        "mean_dtw_path_cosine": float(np.mean(path_values)) if path_values else float("nan"),
        "mean_top1_letter_cosine": float(top1.mean()),
        "mean_top1_margin": float(margin.mean()),
        "window_letter_matrix_std": float(matrix.std()),
        "primitive_effective_rank": primitive_rank,
        "semantic_effective_rank": semantic_rank,
        "primitive_nonlocal_cosine_sep5": primitive_nonlocal,
        "semantic_nonlocal_cosine_sep5": semantic_nonlocal,
        "stroke_primitive_similarity_correlation": stroke_primitive_correlation,
        "stroke_semantic_similarity_correlation": stroke_semantic_correlation,
        "restoration_mae": recon_mae,
        "matrix_mean_abs_delta_prev": matrix_delta,
        "training_cost_mean_abs_delta_prev": training_cost_delta,
        "matrix_correlation_prev": matrix_correlation,
        "dtw_path_jaccard_prev": path_jaccard,
        "patch_weight_relative_delta_prev": patch_delta,
        **gradient_metrics,
    }
    (epoch_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=True),
        encoding="utf-8",
    )
    (epoch_dir / "dtw_path.json").write_text(
        json.dumps(
            {
                "text": text,
                "clean_letters": "".join(letters),
                "pairs": [[i, j] for i, j in full_path],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _append_history(output_root / "history.csv", metrics)

    if was_training:
        model.train()

    print(
        "[restoration-probe] "
        f"epoch={epoch} adapter={metrics['semantic_adapter']} "
        f"recon={recon_mae:.4f} "
        f"rankP={primitive_rank:.2f} rankL={semantic_rank:.2f} "
        f"pathCos={metrics['mean_dtw_path_cosine']:.4f} "
        f"top1={metrics['mean_top1_letter_cosine']:.4f} "
        f"margin={metrics['mean_top1_margin']:.4f} "
        f"matrixDelta={matrix_delta if matrix_delta is not None else float('nan'):.5f} "
        f"pathJaccard={path_jaccard if path_jaccard is not None else float('nan'):.4f} "
        f"patchDelta={patch_delta if patch_delta is not None else float('nan'):.6f} "
        f"gradDTW={gradient_metrics['dtw_patch_grad_norm']:.4f} "
        f"gradRest={gradient_metrics['restoration_patch_grad_norm']:.4f} "
        f"gradCos={gradient_metrics['dtw_restoration_grad_cosine'] if gradient_metrics['dtw_restoration_grad_cosine'] is not None else float('nan'):.4f}",
        flush=True,
    )

    return {
        "patch_weight": current_patch,
        "matrix": current_matrix,
        "training_cost": current_training_cost,
        "path": current_path,
        "metrics": metrics,
    }
