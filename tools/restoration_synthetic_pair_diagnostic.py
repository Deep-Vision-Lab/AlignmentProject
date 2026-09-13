#!/usr/bin/env python3
"""Diagnose one REAL synthetic pair through every trained restoration stage.

This is intentionally a model-diagnostic, not a unit test.  It loads the exact
saved checkpoint and the actual synthetic pair, then writes visual/numeric
evidence for:

  real RGB windows -> local -> context -> fusion -> image-image similarity
                   -> restoration
                   -> positive/negative transcript DTW

The shell wrapper supplies all arguments so the user can run the suite with no
flags.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation._eval_utils import needleman_wunsch
from Evaluation.analyze_restoration_line import (
    char_vectors,
    deterministic_char_codebook,
)
from Evaluation.yelda_geometry import prepare_line
from Evaluation.yelda_runtime import (
    configure_image_preprocessing,
    load_visual_models,
    read_checkpoint,
)
from restoration_epoch_probe import _hard_dtw, _effective_rank
from vlm_restoration_positive_dtw import (
    _clean_letters,
    _soft_dtw_cost_matrix,
    letter_dtw_cost_matrix,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def image_path(dataset: Path, side: int, index: int) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        candidate = dataset / "images" / f"img{side}_{index}{suffix}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"img{side}_{index} not found in {dataset / 'images'}")


def text_path(dataset: Path, side: int, index: int) -> Path:
    candidate = dataset / "texts" / f"text{side}_{index}.txt"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def find_negative_text(dataset: Path, side: int, index: int, positive: str) -> tuple[Path, str]:
    files = sorted((dataset / "texts").glob(f"text{side}_*.txt"))
    for path in files:
        if path.name == f"text{side}_{index}.txt":
            continue
        value = path.read_text(encoding="utf-8").strip()
        if _clean_letters(value) and value != positive:
            return path, value
    raise RuntimeError("Could not find a different synthetic transcript for negative-DTW diagnostic")


def to_tensor(image: Image.Image, device: torch.device):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )(image).unsqueeze(0).to(device)


def unit(x):
    return F.normalize(x.float(), p=2, dim=-1)


def matrix_offdiag_mean(matrix: torch.Tensor) -> float:
    n, m = matrix.shape
    if n != m or n <= 1:
        return float(matrix.mean().item())
    mask = ~torch.eye(n, dtype=torch.bool, device=matrix.device)
    return float(matrix[mask].mean().item())


def save_matrix(path: Path, matrix, title, xlabel, ylabel, route=None):
    value = (
        matrix.detach().float().cpu().numpy()
        if torch.is_tensor(matrix)
        else np.asarray(matrix, dtype=np.float32)
    )
    fig, ax = plt.subplots(
        figsize=(
            max(8.0, min(24.0, value.shape[1] * 0.28)),
            max(6.0, min(20.0, value.shape[0] * 0.22)),
        )
    )
    im = ax.imshow(value, aspect="auto", interpolation="nearest")
    if route:
        ax.plot([j for i, j in route], [i for i, j in route], linewidth=1.6)
        ax.scatter([j for i, j in route], [i for i, j in route], s=8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def tensor_image(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu().clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_swap_sheet(path: Path, target, restored, swapped, indices):
    chosen = list(indices)
    patch_w, patch_h = 32 * 3, 128 * 3
    label_h = 28
    canvas = Image.new(
        "RGB",
        (len(chosen) * patch_w, 3 * (patch_h + label_h)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    rows = [("target", target), ("restored", restored), ("swap decoded", swapped)]
    for r, (name, tensor) in enumerate(rows):
        for c, index in enumerate(chosen):
            y0 = r * (patch_h + label_h)
            draw.text((c * patch_w + 2, y0 + 3), f"{name} W{index}", fill="black")
            image = tensor_image(tensor[index]).resize((patch_w, patch_h))
            canvas.paste(image, (c * patch_w, y0 + label_h))
    canvas.save(path)


class FrozenCharEncoder:
    def __init__(self, codebook):
        self.codebook = codebook

    def __call__(self, text):
        return char_vectors(list(str(text)), self.codebook)


def dtw_settings(config):
    return SimpleNamespace(
        positive_letter_dtw_cost_mode=str(
            config.get("positive_letter_dtw_cost_mode", "cosine")
        ),
        positive_letter_dtw_competition_temperature=float(
            config.get("positive_letter_dtw_competition_temperature", 0.10)
        ),
    )


def transcript_dtw(config, codebook, visual, text):
    letters = _clean_letters(text)
    if not letters:
        raise ValueError("Transcript contains no Arabic letters after cleaning")
    encoder = FrozenCharEncoder(codebook)
    costs = letter_dtw_cost_matrix(
        dtw_settings(config), encoder, visual, letters
    )
    v = float(config.get("positive_letter_dtw_vertical_penalty", 0.05))
    h = float(config.get("positive_letter_dtw_horizontal_penalty", 0.30))
    prior = float(config.get("positive_letter_dtw_position_prior", 0.15))
    disable_h = bool(
        config.get("positive_letter_dtw_disable_horizontal_when_feasible", True)
    )
    hard_h = 1e4 if disable_h and costs.shape[0] >= costs.shape[1] else h
    route, hard_cost = _hard_dtw(
        costs.detach().cpu().numpy(),
        vertical_penalty=v,
        horizontal_penalty=hard_h,
        position_prior_weight=prior,
    )
    soft_cost = _soft_dtw_cost_matrix(
        costs,
        gamma=float(config.get("positive_letter_dtw_gamma", 0.05)),
        vertical_penalty=v,
        horizontal_penalty=h,
        position_prior_weight=prior,
        disable_horizontal_when_feasible=disable_h,
    )
    return letters, costs, route, float(hard_cost), float(soft_cost.detach())


def nw_route(result):
    return [
        (step.i, step.j)
        for step in result.steps
        if step.i is not None and step.j is not None
    ]


def line_bundle(models, path: Path, output: Path, side: int):
    prepared, geometry = prepare_line(path, "synthetic", "training")
    prepared.save(output / f"00_side{side}_model_input.png")
    tensor = to_tensor(prepared, models.device)
    with torch.inference_mode():
        bundle = models.image_model(tensor, return_training_bundle=True)
        raw_fused, raw_local, raw_context, _, token_valid, _ = (
            models.image_model.vit_encoder.encode_restoration_sequence(
                tensor, use_flip=models.image_model.use_flip
            )
        )
    return prepared, geometry, tensor, bundle, raw_fused, raw_local, raw_context, token_valid


def main():
    args = parse_args()
    dataset = Path(args.dataset).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    checkpoint = read_checkpoint(weights)
    models = load_visual_models(checkpoint, args.device, "restoration")
    configure_image_preprocessing(models, "training")
    config = models.config

    img1 = image_path(dataset, 1, args.index)
    img2 = image_path(dataset, 2, args.index)
    txt1 = text_path(dataset, 1, args.index)
    txt2 = text_path(dataset, 2, args.index)
    text1 = txt1.read_text(encoding="utf-8").strip()
    text2 = txt2.read_text(encoding="utf-8").strip()
    neg_path, negative_text = find_negative_text(dataset, 1, args.index, text1)

    side1 = line_bundle(models, img1, output, 1)
    side2 = line_bundle(models, img2, output, 2)
    _, geometry1, tensor1, bundle1, _, local_raw1, context_raw1, valid1 = side1
    _, geometry2, tensor2, bundle2, _, local_raw2, context_raw2, valid2 = side2

    reps1 = {
        "local": unit(bundle1["primitive"][0]),
        "context": unit(bundle1["contextual"][0]),
        "fused": unit(bundle1["fused"][0]),
    }
    reps2 = {
        "local": unit(bundle2["primitive"][0]),
        "context": unit(bundle2["contextual"][0]),
        "fused": unit(bundle2["fused"][0]),
    }

    metrics = {
        "weights": str(weights),
        "dataset": str(dataset),
        "index": int(args.index),
        "image1": str(img1),
        "image2": str(img2),
        "text1": str(txt1),
        "text2": str(txt2),
        "negative_text": str(neg_path),
        "model_config": {
            "training_stage": config.get("training_stage"),
            "restoration_local_encoder": config.get("restoration_local_encoder"),
            "context_transformer_layers": config.get("context_transformer_layers"),
            "fusion_mode": config.get("fusion_mode"),
            "negative_transcripts": config.get("negative_transcripts"),
            "positive_letter_dtw_cost_mode": config.get("positive_letter_dtw_cost_mode"),
        },
        "geometry_side1": geometry1,
        "geometry_side2": geometry2,
    }

    # Point 2: prove the target is the actual synthetic RGB window.
    target1 = bundle1["restoration_target"][0]
    restored1 = bundle1["restoration"][0]
    target2 = bundle2["restoration_target"][0]
    restored2 = bundle2["restoration"][0]
    metrics["restoration"] = {
        "side1_mae": float((restored1 - target1).abs().mean()),
        "side2_mae": float((restored2 - target2).abs().mean()),
        "side1_output_diversity": float(restored1.std(dim=0).mean()),
        "side2_output_diversity": float(restored2.std(dim=0).mean()),
        "side1_target_diversity": float(target1.std(dim=0).mean()),
        "side2_target_diversity": float(target2.std(dim=0).mean()),
    }

    # Point 5: swap real synthetic local features and see whether reconstruction follows.
    valid_indices = torch.nonzero(valid1[0], as_tuple=False).flatten().tolist()
    if len(valid_indices) >= 2:
        first, last = valid_indices[0], valid_indices[-1]
    else:
        first, last = 0, max(0, local_raw1.shape[1] - 1)
    swapped_tokens = local_raw1.clone()
    swapped_tokens[:, first] = local_raw1[:, last]
    swapped_tokens[:, last] = local_raw1[:, first]
    with torch.inference_mode():
        swapped_restoration = models.image_model.vit_encoder.stroke_decoder(swapped_tokens)[0]
    metrics["feature_swap"] = {
        "first_window": int(first),
        "last_window": int(last),
        "mean_absolute_output_change": float(
            (restored1 - swapped_restoration).abs().mean()
        ),
    }
    save_swap_sheet(
        output / "01_real_feature_swap.png",
        target1,
        restored1,
        swapped_restoration,
        [first, last],
    )

    # Points 4/11: optional K spatial vectors on a real synthetic line.
    patch_encoder = models.image_model.vit_encoder.patch_embedding
    if hasattr(patch_encoder, "spatial_vectors"):
        with torch.inference_mode():
            spatial = patch_encoder.spatial_vectors(tensor1, vectors_per_window=4)[0]
        selected = int(first)
        spatial_sim = unit(spatial[selected]) @ unit(spatial[selected]).T
        save_matrix(
            output / "02_real_window_four_spatial_vectors.png",
            spatial_sim,
            f"K=4 spatial vectors inside real synthetic window {selected}",
            "sub-vector",
            "sub-vector",
        )
        metrics["spatial_vectors"] = {
            "available": True,
            "shape": list(spatial.shape),
            "selected_window": selected,
            "mean_offdiag_cosine": matrix_offdiag_mean(spatial_sim),
        }
    else:
        metrics["spatial_vectors"] = {"available": False}

    # Points 6/7/13: local -> context -> fusion behavior on actual paired lines.
    for name in ("local", "context", "fused"):
        within1 = reps1[name] @ reps1[name].T
        within2 = reps2[name] @ reps2[name].T
        cross = reps1[name] @ reps2[name].T
        nw = needleman_wunsch(cross, gap_penalty=-0.30)
        route = nw_route(nw)
        save_matrix(
            output / f"10_{name}_cross_line_similarity.png",
            cross,
            f"{name}: synthetic side1 ↔ side2 cosine + NW route",
            "side2 window",
            "side1 window",
            route,
        )
        save_matrix(
            output / f"11_{name}_within_side1_similarity.png",
            within1,
            f"{name}: within side1 window cosine",
            "window",
            "window",
        )
        metrics[name] = {
            "side1_effective_rank": _effective_rank(reps1[name]),
            "side2_effective_rank": _effective_rank(reps2[name]),
            "side1_offdiag_cosine": matrix_offdiag_mean(within1),
            "side2_offdiag_cosine": matrix_offdiag_mean(within2),
            "cross_mean_cosine": float(cross.mean()),
            "cross_max_cosine": float(cross.max()),
            "nw_score": float(nw.score),
            "nw_normalized_score": float(nw.normalized_score),
            "nw_match_steps": len(route),
        }

    # Point 6: observational amount of change introduced by real context.
    comparable = min(reps1["local"].shape[0], reps1["context"].shape[0])
    local_context_cos = F.cosine_similarity(
        reps1["local"][:comparable], reps1["context"][:comparable], dim=-1
    )
    metrics["context_effect"] = {
        "mean_local_context_cosine": float(local_context_cos.mean()),
        "min_local_context_cosine": float(local_context_cos.min()),
        "max_local_context_cosine": float(local_context_cos.max()),
    }
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        np.arange(comparable),
        (1.0 - local_context_cos).detach().cpu().numpy(),
        marker="o",
        markersize=2,
    )
    ax.set_xlabel("logical synthetic window")
    ax.set_ylabel("1 - cos(local, context)")
    ax.set_title("Actual context change per synthetic window")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "12_context_change_per_window.png", dpi=170)
    plt.close(fig)

    # Point 7: ablate each fusion input on the ACTUAL trained fusion head.
    fusion_head = models.image_model.vit_encoder.fusion_head
    with torch.inference_mode():
        fused_actual_raw = fusion_head(local_raw1, context_raw1)
        fused_no_context_raw = fusion_head(local_raw1, torch.zeros_like(context_raw1))
        fused_no_local_raw = fusion_head(torch.zeros_like(local_raw1), context_raw1)
        def finish(value):
            value = models.image_model.vision_norm(value)
            return unit(value)
        fused_actual = finish(fused_actual_raw)[0]
        fused_no_context = finish(fused_no_context_raw)[0]
        fused_no_local = finish(fused_no_local_raw)[0]
    no_context_cos = F.cosine_similarity(fused_actual, fused_no_context, dim=-1)
    no_local_cos = F.cosine_similarity(fused_actual, fused_no_local, dim=-1)
    metrics["fusion_ablation"] = {
        "actual_vs_no_context_mean_cosine": float(no_context_cos.mean()),
        "actual_vs_no_local_mean_cosine": float(no_local_cos.mean()),
        "context_contribution_change": float((1.0 - no_context_cos).mean()),
        "local_contribution_change": float((1.0 - no_local_cos).mean()),
    }
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot((1.0 - no_context_cos).cpu().numpy(), label="remove context")
    ax.plot((1.0 - no_local_cos).cpu().numpy(), label="remove local")
    ax.set_xlabel("logical synthetic window")
    ax.set_ylabel("change in fused vector")
    ax.set_title("Fusion ablation on actual synthetic features")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "13_fusion_ablation.png", dpi=170)
    plt.close(fig)

    # Point 8/9/10: actual positive and negative transcript DTW from current fused features.
    codebook = deterministic_char_codebook(config, models.device)
    valid_mask = bundle1["token_valid"][0].bool()
    visual = reps1["fused"][valid_mask]
    pos_letters, pos_costs, pos_route, pos_hard, pos_soft = transcript_dtw(
        config, codebook, visual, text1
    )
    neg_letters, neg_costs, neg_route, neg_hard, neg_soft = transcript_dtw(
        config, codebook, visual, negative_text
    )
    save_matrix(
        output / "20_positive_transcript_dtw.png",
        pos_costs,
        "Actual fused windows → POSITIVE transcript DTW cost",
        "positive transcript letter",
        "valid synthetic window",
        pos_route,
    )
    save_matrix(
        output / "21_negative_transcript_dtw.png",
        neg_costs,
        "Same fused windows → NEGATIVE transcript DTW cost",
        "negative transcript letter",
        "valid synthetic window",
        neg_route,
    )
    metrics["dtw"] = {
        "valid_windows": int(visual.shape[0]),
        "positive_letters": len(pos_letters),
        "negative_letters": len(neg_letters),
        "positive_hard_cost": pos_hard,
        "negative_hard_cost": neg_hard,
        "hard_negative_minus_positive": neg_hard - pos_hard,
        "positive_soft_cost": pos_soft,
        "negative_soft_cost": neg_soft,
        "soft_negative_minus_positive": neg_soft - pos_soft,
        "vertical_penalty": float(config.get("positive_letter_dtw_vertical_penalty", 0.05)),
        "horizontal_penalty": float(config.get("positive_letter_dtw_horizontal_penalty", 0.30)),
        "position_prior": float(config.get("positive_letter_dtw_position_prior", 0.15)),
    }

    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = f"""SYNTHETIC TRAINED-MODEL DIAGNOSTIC
==================================

checkpoint: {weights}
dataset   : {dataset}
pair index: {args.index}

Restoration
-----------
side1 MAE               : {metrics['restoration']['side1_mae']:.6f}
side2 MAE               : {metrics['restoration']['side2_mae']:.6f}
side1 output diversity  : {metrics['restoration']['side1_output_diversity']:.6f}
side1 target diversity  : {metrics['restoration']['side1_target_diversity']:.6f}
feature-swap output Δ   : {metrics['feature_swap']['mean_absolute_output_change']:.6f}

Representation quality
----------------------
local effective rank    : {metrics['local']['side1_effective_rank']:.3f}
context effective rank  : {metrics['context']['side1_effective_rank']:.3f}
fused effective rank    : {metrics['fused']['side1_effective_rank']:.3f}

Image-image alignment (same synthetic pair)
--------------------------------------------
local NW normalized     : {metrics['local']['nw_normalized_score']:.6f}
context NW normalized   : {metrics['context']['nw_normalized_score']:.6f}
fused NW normalized     : {metrics['fused']['nw_normalized_score']:.6f}

Context/Fusion
--------------
mean local↔context cos  : {metrics['context_effect']['mean_local_context_cosine']:.6f}
remove-context change   : {metrics['fusion_ablation']['context_contribution_change']:.6f}
remove-local change     : {metrics['fusion_ablation']['local_contribution_change']:.6f}

Text DTW competition
--------------------
positive soft DTW       : {pos_soft:.6f}
negative soft DTW       : {neg_soft:.6f}
negative - positive     : {neg_soft - pos_soft:.6f}

Interpretation:
- restoration output diversity should not collapse far below target diversity;
- feature-swap Δ must be visibly/non-trivially non-zero;
- effective rank should not collapse from local -> context -> fused;
- if fused NW is worse than local NW, context/fusion is hurting image-image invariance;
- negative DTW should cost MORE than the positive transcript;
- inspect the PNG matrices instead of relying only on these numbers.
"""
    (output / "README.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Full metrics: {output / 'metrics.json'}")
    print(f"Visuals: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
