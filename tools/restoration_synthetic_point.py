#!/usr/bin/env python3
"""Run one restoration recommendation check on ACTUAL synthetic data.

The shell wrappers in scripts/restoration_points/ call this with a fixed point
number. Dataset, pair index, device and restoration checkpoint are selected
automatically so the user does not need flags.

Primary data:
  DataSet/Synthetic63, pair 132 when available.

Checkpoint preference:
  Stage B -> Stage A -> legacy restoration checkpoint.

Points 9 and 10 use the REAL synthetic line's current DTW cost matrix, while
retaining a small controlled topology/recompute assertion because those
properties cannot be proven from one observed route alone.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
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

from embeddingModel import sliding_window
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
from restoration_recommended_components import denormalize_imagenet_windows
from restormer_pretrained_probe import find_restormer_assets, load_restormer
from vlm_restoration_positive_dtw import (
    _clean_letters,
    _soft_dtw_cost_matrix,
    letter_dtw_cost_matrix,
)
from zero_shot_preprocessing import ManuscriptLinePreprocessor

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
WINDOW_SIZE = 32
STRIDE = 16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--point", type=int, required=True, choices=range(1, 14))
    return p.parse_args()


def resolve_dataset() -> Path:
    override = os.environ.get("SYNTHETIC_DIAG_DATASET", "").strip()
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates += [
        ROOT / "DataSet" / "Synthetic63",
        ROOT / "DataSet" / "Synthetic_Arabic",
    ]
    candidates += sorted((ROOT / "DataSet").glob("Synthetic*"))
    seen = set()
    for candidate in candidates:
        if not candidate.is_absolute():
            candidate = ROOT / candidate
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if (candidate / "images").is_dir() and (candidate / "texts").is_dir():
            return candidate
    raise FileNotFoundError("No synthetic dataset with images/ and texts/ was found")


def image_candidate(dataset: Path, side: int, index: int) -> Path | None:
    for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
        p = dataset / "images" / f"img{side}_{index}{suffix}"
        if p.is_file():
            return p
    return None


def pair_complete(dataset: Path, index: int) -> bool:
    return (
        image_candidate(dataset, 1, index) is not None
        and image_candidate(dataset, 2, index) is not None
        and (dataset / "texts" / f"text1_{index}.txt").is_file()
        and (dataset / "texts" / f"text2_{index}.txt").is_file()
    )


def resolve_index(dataset: Path) -> int:
    override = os.environ.get("SYNTHETIC_DIAG_INDEX", "").strip()
    if override and pair_complete(dataset, int(override)):
        return int(override)
    if pair_complete(dataset, 132):
        return 132
    indices = []
    for p in (dataset / "images").glob("img1_*"):
        try:
            indices.append(int(p.stem.split("_")[-1]))
        except ValueError:
            continue
    for index in sorted(set(indices)):
        if pair_complete(dataset, index):
            return index
    raise FileNotFoundError("No complete synthetic img1/img2/text1/text2 pair found")


def resolve_weights() -> Path | None:
    override = os.environ.get("WEIGHTS", "").strip()
    candidates = []
    if override:
        candidates.append(Path(override).expanduser())
    candidates += [
        ROOT / "Weights" / "restore_fused_rgb_dtw_s16" / "model_best.pth",
        ROOT / "Weights" / "restore_rgb_pretrain_s16" / "model_best.pth",
        ROOT / "Weights" / "vit_restore_dtw_s16" / "model_best.pth",
    ]
    for p in candidates:
        if not p.is_absolute():
            p = ROOT / p
        if p.is_file():
            return p.resolve()
    return None


def resolve_device() -> str:
    override = os.environ.get("DEVICE", "").strip()
    if override:
        return override
    return "cuda" if torch.cuda.is_available() else "cpu"


def prepared_line(dataset: Path, side: int, index: int):
    path = image_candidate(dataset, side, index)
    if path is None:
        raise FileNotFoundError(f"img{side}_{index}")
    # Explicitly use the restoration branch's intended RGB geometry.
    source = Image.open(path).convert("RGB")
    processor = ManuscriptLinePreprocessor(
        size=(128, 1024),
        training=False,
        augment=False,
        binarize=False,
        preserve_aspect=True,
        crop_foreground=True,
        target_ink_height_ratio=0.72,
        autocontrast=False,
    )
    image, geometry = processor.preprocess_with_metadata(source)
    return path, source, image, geometry


def to_tensor(image: Image.Image, device) -> torch.Tensor:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )(image).unsqueeze(0).to(device)


def load_models(weights: Path | None, device: str):
    if weights is None:
        return None, "SKIP: no restoration checkpoint found"
    try:
        checkpoint = read_checkpoint(weights)
        models = load_visual_models(checkpoint, device, "restoration")
        configure_image_preprocessing(models, "training")
        return models, f"checkpoint={weights}"
    except Exception as exc:
        return None, f"SKIP: checkpoint could not be loaded as restoration model: {type(exc).__name__}: {exc}"


def model_bundle(models, image: Image.Image):
    tensor = to_tensor(image, models.device)
    with torch.inference_mode():
        bundle = models.image_model(tensor, return_training_bundle=True)
        raw_fused, raw_local, raw_context, _, token_valid, pixel_valid = (
            models.image_model.vit_encoder.encode_restoration_sequence(
                tensor, use_flip=models.image_model.use_flip
            )
        )
    return tensor, bundle, raw_fused, raw_local, raw_context, token_valid, pixel_valid


def tensor_image(x) -> Image.Image:
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    x = x.detach().float().cpu().clamp(0, 1)
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if x.shape[0] == 1:
        x = x.repeat(3, 1, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_matrix(path: Path, matrix, title: str, xlabel: str, ylabel: str, route=None):
    arr = matrix.detach().float().cpu().numpy() if torch.is_tensor(matrix) else np.asarray(matrix)
    fig, ax = plt.subplots(
        figsize=(max(7.0, min(24.0, arr.shape[1] * 0.28)),
                 max(5.0, min(18.0, arr.shape[0] * 0.22)))
    )
    im = ax.imshow(arr, aspect="auto", interpolation="nearest")
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


def save_windows_sheet(path: Path, windows, rows, titles):
    # windows/rows are list of tensors shaped C,H,W.
    converted = [[tensor_image(x) for x in row] for row in rows]
    cols = max(len(row) for row in converted)
    patch_w, patch_h, label_h = 96, 384, 28
    canvas = Image.new("RGB", (cols * patch_w, len(converted) * (patch_h + label_h)), "white")
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(converted):
        for c, image in enumerate(row):
            y = r * (patch_h + label_h)
            label = titles[r][c] if r < len(titles) and c < len(titles[r]) else ""
            draw.text((c * patch_w + 2, y + 3), label, fill="black")
            canvas.paste(image.resize((patch_w, patch_h)), (c * patch_w, y + label_h))
    canvas.save(path)


def select_informative_indices(target, count=6):
    # target [T,C,H,W], select windows with greatest RGB variance/ink.
    darkness = (1.0 - target.float()).abs().mean(dim=(1, 2, 3))
    count = min(count, int(target.shape[0]))
    values = torch.topk(darkness, k=count).indices.tolist()
    return sorted(int(v) for v in values)


def status(out: Path, text: str, payload=None):
    (out / "summary.txt").write_text(text.rstrip() + "\n", encoding="utf-8")
    if payload is not None:
        (out / "metrics.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(text)
    print(f"Results: {out}")


class FrozenCharEncoder:
    def __init__(self, codebook):
        self.codebook = codebook

    def __call__(self, text):
        return char_vectors(list(str(text)), self.codebook)


def negative_text(dataset: Path, side: int, index: int, positive: str):
    for p in sorted((dataset / "texts").glob(f"text{side}_*.txt")):
        if p.name == f"text{side}_{index}.txt":
            continue
        value = p.read_text(encoding="utf-8").strip()
        if value != positive and _clean_letters(value):
            return p, value
    raise RuntimeError("No negative synthetic transcript found")


def transcript_cost(models, visual, text):
    config = models.config
    letters = _clean_letters(text)
    codebook = deterministic_char_codebook(config, models.device)
    encoder = FrozenCharEncoder(codebook)
    settings = SimpleNamespace(
        positive_letter_dtw_cost_mode=str(config.get("positive_letter_dtw_cost_mode", "cosine")),
        positive_letter_dtw_competition_temperature=float(
            config.get("positive_letter_dtw_competition_temperature", 0.10)
        ),
    )
    costs = letter_dtw_cost_matrix(settings, encoder, visual, letters)
    v = float(config.get("positive_letter_dtw_vertical_penalty", 0.05))
    h = float(config.get("positive_letter_dtw_horizontal_penalty", 0.30))
    prior = float(config.get("positive_letter_dtw_position_prior", 0.15))
    disable_h = bool(config.get("positive_letter_dtw_disable_horizontal_when_feasible", True))
    hard_h = 1e4 if disable_h and costs.shape[0] >= costs.shape[1] else h
    route, hard = _hard_dtw(
        costs.detach().cpu().numpy(),
        vertical_penalty=v,
        horizontal_penalty=hard_h,
        position_prior_weight=prior,
    )
    soft = _soft_dtw_cost_matrix(
        costs,
        gamma=float(config.get("positive_letter_dtw_gamma", 0.05)),
        vertical_penalty=v,
        horizontal_penalty=h,
        position_prior_weight=prior,
        disable_horizontal_when_feasible=disable_h,
    )
    return letters, costs, route, float(hard), float(soft.detach()), v, h, prior, disable_h


def point1(dataset, index, out):
    image_path, source, prepared, geometry = prepared_line(dataset, 1, index)
    source.save(out / "01_original_synthetic_line.png")
    boxed = source.copy()
    draw = ImageDraw.Draw(boxed)
    box = (
        int(geometry["crop_left"]), int(geometry["crop_top"]),
        int(geometry["crop_right"]) - 1, int(geometry["crop_bottom"]) - 1,
    )
    draw.rectangle(box, outline="red", width=2)
    boxed.save(out / "02_crop_box.png")
    prepared.save(out / "03_processed_rgb_line.png")

    overlay = prepared.copy()
    draw = ImageDraw.Draw(overlay)
    for k, x0 in enumerate(range(0, prepared.width - WINDOW_SIZE + 1, STRIDE)):
        draw.rectangle((x0, 0, x0 + WINDOW_SIZE - 1, prepared.height - 1), outline="red", width=1)
        if k % 4 == 0:
            draw.text((x0 + 1, 2), str(k), fill="blue")
    overlay.save(out / "04_window_boundaries.png")
    status(
        out,
        f"""POINT 01 — SYNTHETIC CROP/GEOMETRY
image={image_path}
original_size={source.size}
processed_size={prepared.size}
crop_box={box}

Inspect the red crop box: outer blank margin should be removed, internal spaces
and Arabic dots/diacritics must remain. 03_processed_rgb_line.png is RGB and is
the line from which windows are sliced.""",
        {"image": str(image_path), "crop_box": box, "geometry": geometry},
    )


def point2(dataset, index, out, models, model_note):
    _, _, prepared, _ = prepared_line(dataset, 1, index)
    prepared.save(out / "01_processed_line.png")
    if models is None:
        status(out, f"POINT 02 — SYNTHETIC ORIGINAL-WINDOW TARGET\n{model_note}")
        return
    tensor, bundle, *_ = model_bundle(models, prepared)
    patches = sliding_window(tensor, models.image_model.window_size, models.image_model.stride)
    if models.image_model.use_flip:
        patches = torch.flip(patches, dims=[1])
    expected = denormalize_imagenet_windows(patches)
    actual = bundle["restoration_target"]
    max_diff = float((expected - actual).abs().max())
    indices = select_informative_indices(actual[0], 6)
    rows = [
        [expected[0, i] for i in indices],
        [actual[0, i] for i in indices],
    ]
    titles = [
        [f"exact RGB W{i}" for i in indices],
        [f"model target W{i}" for i in indices],
    ]
    save_windows_sheet(out / "02_exact_vs_model_target.png", None, rows, titles)
    status(
        out,
        f"""POINT 02 — SYNTHETIC ORIGINAL-WINDOW TARGET
{model_note}
max_abs_difference={max_diff:.10f}
selected_windows={indices}

The two rows in 02_exact_vs_model_target.png must be pixel-identical.""",
        {"max_abs_difference": max_diff, "selected_windows": indices},
    )


def point3(dataset, index, out):
    assets = find_restormer_assets(ROOT)
    if not assets.ready:
        status(out, f"POINT 03 — PRETRAINED RESTORMER ON SYNTHETIC WINDOWS\nSKIP\n{assets.message}")
        return
    _, _, prepared, _ = prepared_line(dataset, 1, index)
    rgb = transforms.ToTensor()(prepared).unsqueeze(0)
    patches = rgb.unfold(3, 32, 16).permute(0, 3, 1, 2, 4).contiguous()[0]
    indices = select_informative_indices(patches, 2)
    windows = patches[indices].to(resolve_device())
    model = load_restormer(assets, device=windows.device)
    model.eval()
    latent = {}
    handle = model.latent.register_forward_hook(
        lambda _m, _i, output: latent.__setitem__("value", output)
    )
    with torch.inference_mode():
        restored = model(windows)
    handle.remove()
    feat = latent["value"].float().mean(dim=(-2, -1))
    cosine = float(F.cosine_similarity(feat[0:1], feat[1:2], dim=-1)[0])
    l1 = float(F.l1_loss(restored, windows))
    rows = [[windows[0], windows[1]], [restored[0], restored[1]]]
    titles = [["input A", "input B"], ["Restormer A", "Restormer B"]]
    save_windows_sheet(out / "01_restormer_real_synthetic_windows.png", None, rows, titles)
    status(
        out,
        f"""POINT 03 — PRETRAINED RESTORMER ON SYNTHETIC WINDOWS
checkpoint={assets.checkpoint}
windows={indices}
identity_L1={l1:.6f}
latent_cosine_between_windows={cosine:.6f}

Judge the restoration visually in 01_restormer_real_synthetic_windows.png.""",
        {"windows": indices, "identity_l1": l1, "latent_cosine": cosine},
    )


def point4(dataset, index, out, models, model_note):
    if models is None:
        status(out, f"POINT 04 — FINE SPATIAL DETAIL ON SYNTHETIC WINDOW\n{model_note}")
        return
    _, _, prepared, _ = prepared_line(dataset, 1, index)
    tensor, bundle, *_ = model_bundle(models, prepared)
    enc = models.image_model.vit_encoder.patch_embedding
    target = bundle["restoration_target"][0]
    selected = select_informative_indices(target, 1)[0]
    if not hasattr(enc, "extract_windows") or not hasattr(enc, "encoder"):
        status(out, f"POINT 04 — FINE SPATIAL DETAIL\n{model_note}\nSKIP: checkpoint uses {type(enc).__name__}, not cnn_seq2seq.")
        return
    patch = enc.extract_windows(tensor)[0, selected].unsqueeze(0)
    shapes = [("input", int(patch.shape[-2]), int(patch.shape[-1]))]
    x = patch
    with torch.inference_mode():
        for i, block in enumerate(enc.encoder):
            x = block(x)
            shapes.append((f"block_{i}", int(x.shape[-2]), int(x.shape[-1])))
    fig, ax = plt.subplots(figsize=(7, 4))
    ids = np.arange(len(shapes))
    ax.plot(ids, [s[1] for s in shapes], marker="o", label="height")
    ax.plot(ids, [s[2] for s in shapes], marker="o", label="width")
    ax.set_xticks(ids)
    ax.set_xticklabels([s[0] for s in shapes], rotation=25)
    ax.set_ylabel("spatial pixels")
    ax.set_title(f"Real synthetic W{selected}: encoder spatial resolution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_spatial_resolution.png", dpi=170)
    plt.close(fig)
    save_matrix(out / "02_final_feature_map.png", x[0].mean(0), f"Real synthetic W{selected}: final mean feature map", "x", "y")
    status(
        out,
        "POINT 04 — FINE SPATIAL DETAIL ON SYNTHETIC WINDOW\n"
        + model_note + "\n"
        + "\n".join(f"{name}: {h}x{w}" for name, h, w in shapes),
        {"selected_window": selected, "shapes": shapes},
    )


def point5(dataset, index, out, models, model_note):
    if models is None:
        status(out, f"POINT 05 — FEATURE DEPENDENCY ON SYNTHETIC LINE\n{model_note}")
        return
    _, _, prepared, _ = prepared_line(dataset, 1, index)
    _, bundle, _, local, _, valid, _ = model_bundle(models, prepared)
    good = torch.nonzero(valid[0], as_tuple=False).flatten().tolist()
    a, b = (good[0], good[-1]) if len(good) >= 2 else (0, local.shape[1] - 1)
    swapped_local = local.clone()
    swapped_local[:, a] = local[:, b]
    swapped_local[:, b] = local[:, a]
    with torch.inference_mode():
        swapped = models.image_model.vit_encoder.stroke_decoder(swapped_local)[0]
    restored = bundle["restoration"][0]
    target = bundle["restoration_target"][0]
    delta = float((restored - swapped).abs().mean())
    rows = [
        [target[a], target[b]],
        [restored[a], restored[b]],
        [swapped[a], swapped[b]],
    ]
    titles = [
        [f"target W{a}", f"target W{b}"],
        [f"decoded W{a}", f"decoded W{b}"],
        [f"after swap W{a}", f"after swap W{b}"],
    ]
    save_windows_sheet(out / "01_real_feature_swap.png", None, rows, titles)
    status(
        out,
        f"""POINT 05 — FEATURE DEPENDENCY ON SYNTHETIC LINE
{model_note}
swapped_windows={a},{b}
mean_absolute_output_change={delta:.8f}

If the last row does not react to the feature swap, the decoder is not using the
local bottleneck correctly.""",
        {"swapped_windows": [a, b], "swap_delta": delta},
    )


def point6(dataset, index, out, models, model_note):
    if models is None:
        status(out, f"POINT 06 — SEQUENCE CONTEXT ON SYNTHETIC LINE\n{model_note}")
        return
    _, _, prepared, _ = prepared_line(dataset, 1, index)
    _, _, _, local, _, valid, _ = model_bundle(models, prepared)
    vit = models.image_model.vit_encoder
    count = int(local.shape[1])
    valid_mask = valid.bool()
    with torch.inference_mode():
        base = vit.encoder(
            vit.input_dropout(local + vit._position_tokens(count).to(local)),
            src_key_padding_mask=~valid_mask,
        )
        changed_local = local.clone()
        candidates = torch.nonzero(valid_mask[0], as_tuple=False).flatten()
        pivot = int(candidates[len(candidates) // 2]) if len(candidates) else count // 2
        changed_local[:, pivot] += 5.0
        changed = vit.encoder(
            vit.input_dropout(changed_local + vit._position_tokens(count).to(local)),
            src_key_padding_mask=~valid_mask,
        )
    delta = torch.linalg.vector_norm((changed - base).float(), dim=-1)[0].cpu().numpy()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(delta, marker="o", markersize=2)
    ax.axvline(pivot, linestyle="--")
    ax.set_xlabel("logical synthetic window")
    ax.set_ylabel("context-vector L2 change")
    ax.set_title(f"Perturb real local W{pivot}: influence across context sequence")
    fig.tight_layout()
    fig.savefig(out / "01_real_neighbor_influence.png", dpi=170)
    plt.close(fig)
    status(
        out,
        f"""POINT 06 — SEQUENCE CONTEXT ON SYNTHETIC LINE
{model_note}
perturbed_window={pivot}
mean_other_window_change={float(np.delete(delta, pivot).mean()):.8f}

Non-zero change away from W{pivot} proves neighboring synthetic windows influence
one another through the Transformer.""",
        {"pivot": pivot, "delta": delta.tolist()},
    )


def point7(dataset, index, out, models, model_note):
    if models is None:
        status(out, f"POINT 07 — LOCAL/CONTEXT FUSION ON SYNTHETIC LINE\n{model_note}")
        return
    _, _, prepared, _ = prepared_line(dataset, 1, index)
    _, bundle, _, local, context, _, _ = model_bundle(models, prepared)
    fusion = models.image_model.vit_encoder.fusion_head
    with torch.inference_mode():
        actual = fusion(local, context)
        no_context = fusion(local, torch.zeros_like(context))
        no_local = fusion(torch.zeros_like(local), context)
        actual = F.normalize(models.image_model.vision_norm(actual).float(), dim=-1)
        no_context = F.normalize(models.image_model.vision_norm(no_context).float(), dim=-1)
        no_local = F.normalize(models.image_model.vision_norm(no_local).float(), dim=-1)
    local_effect = 1.0 - F.cosine_similarity(actual, no_local, dim=-1)[0]
    context_effect = 1.0 - F.cosine_similarity(actual, no_context, dim=-1)[0]
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(local_effect.cpu().numpy(), label="remove local")
    ax.plot(context_effect.cpu().numpy(), label="remove context")
    ax.set_xlabel("logical synthetic window")
    ax.set_ylabel("1 - cosine to full fusion")
    ax.set_title("Actual trained fusion dependency on synthetic line")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "01_real_fusion_ablation.png", dpi=170)
    plt.close(fig)
    status(
        out,
        f"""POINT 07 — LOCAL/CONTEXT FUSION ON SYNTHETIC LINE
{model_note}
mean_local_contribution={float(local_effect.mean()):.8f}
mean_context_contribution={float(context_effect.mean()):.8f}

Both should be meaningfully non-zero if fusion uses both inputs.""",
        {
            "mean_local_contribution": float(local_effect.mean()),
            "mean_context_contribution": float(context_effect.mean()),
        },
    )


def point8(dataset, index, out, models, model_note):
    if models is None:
        status(out, f"POINT 08 — RECONSTRUCTION + CONTRASTIVE DTW ON SYNTHETIC LINE\n{model_note}")
        return
    _, _, prepared, _ = prepared_line(dataset, 1, index)
    _, bundle, *_ = model_bundle(models, prepared)
    text_path = dataset / "texts" / f"text1_{index}.txt"
    positive = text_path.read_text(encoding="utf-8").strip()
    neg_path, negative = negative_text(dataset, 1, index, positive)
    valid = bundle["token_valid"][0].bool()
    visual = F.normalize(bundle["fused"][0][valid].float(), dim=-1)
    _, pos_costs, pos_route, pos_hard, pos_soft, *_ = transcript_cost(models, visual, positive)
    _, neg_costs, neg_route, neg_hard, neg_soft, *_ = transcript_cost(models, visual, negative)
    restoration_mae = float((bundle["restoration"] - bundle["restoration_target"]).abs().mean())
    save_matrix(out / "01_positive_dtw.png", pos_costs, "Synthetic positive transcript DTW", "positive letter", "valid window", pos_route)
    save_matrix(out / "02_negative_dtw.png", neg_costs, "Synthetic negative transcript DTW", "negative letter", "valid window", neg_route)
    status(
        out,
        f"""POINT 08 — RECONSTRUCTION + CONTRASTIVE DTW ON SYNTHETIC LINE
{model_note}
positive_text={text_path.name}
negative_text={neg_path.name}
restoration_MAE={restoration_mae:.6f}
positive_soft_DTW={pos_soft:.6f}
negative_soft_DTW={neg_soft:.6f}
negative_minus_positive={neg_soft - pos_soft:.6f}

Desired: negative_minus_positive > 0 while reconstruction remains distinct.""",
        {
            "restoration_mae": restoration_mae,
            "positive_soft_dtw": pos_soft,
            "negative_soft_dtw": neg_soft,
            "negative_minus_positive": neg_soft - pos_soft,
            "positive_hard_dtw": pos_hard,
            "negative_hard_dtw": neg_hard,
        },
    )


def transition_counts(route):
    counts = {"diag": 0, "vertical_many_windows_one_letter": 0, "horizontal_one_window_many_letters": 0}
    for (i0, j0), (i1, j1) in zip(route, route[1:]):
        di, dj = i1 - i0, j1 - j0
        if (di, dj) == (1, 1):
            counts["diag"] += 1
        elif (di, dj) == (1, 0):
            counts["vertical_many_windows_one_letter"] += 1
        elif (di, dj) == (0, 1):
            counts["horizontal_one_window_many_letters"] += 1
    return counts


def point9(dataset, index, out, models, model_note):
    if models is None:
        status(out, f"POINT 09 — DTW TRANSITIONS ON SYNTHETIC COST MATRIX\n{model_note}")
        return
    _, _, prepared, _ = prepared_line(dataset, 1, index)
    _, bundle, *_ = model_bundle(models, prepared)
    text = (dataset / "texts" / f"text1_{index}.txt").read_text(encoding="utf-8").strip()
    valid = bundle["token_valid"][0].bool()
    visual = F.normalize(bundle["fused"][0][valid].float(), dim=-1)
    letters, costs, route, hard, soft, v, h, prior, disable_h = transcript_cost(models, visual, text)
    counts = transition_counts(route)
    save_matrix(out / "01_real_synthetic_dtw_transitions.png", costs, "Actual synthetic DTW route", "transcript letter", "valid window", route)

    # Controlled topology proof remains necessary: a single real path does not
    # necessarily exercise every transition type.
    controlled_many, _ = _hard_dtw(np.zeros((4, 1)), v, h)
    controlled_horizontal, _ = _hard_dtw(np.zeros((1, 4)), v, h)
    status(
        out,
        f"""POINT 09 — DTW TRANSITIONS ON SYNTHETIC COST MATRIX
{model_note}
windows={costs.shape[0]} letters={costs.shape[1]}
actual_transition_counts={counts}
vertical_penalty={v} horizontal_penalty={h}
disable_horizontal_when_feasible={disable_h}

The PNG shows the route on the real synthetic transcript. Controlled edge cases
also confirm 4-windows→1-letter and 1-window→4-letters remain representable.""",
        {
            "shape": list(costs.shape),
            "transition_counts": counts,
            "hard_cost": hard,
            "soft_cost": soft,
            "controlled_many_to_one_route": controlled_many,
            "controlled_one_to_many_route": controlled_horizontal,
        },
    )


def point10(dataset, index, out, models, model_note):
    if models is None:
        status(out, f"POINT 10 — DTW RECOMPUTE ON SYNTHETIC COST MATRIX\n{model_note}")
        return
    _, _, prepared, _ = prepared_line(dataset, 1, index)
    _, bundle, *_ = model_bundle(models, prepared)
    text = (dataset / "texts" / f"text1_{index}.txt").read_text(encoding="utf-8").strip()
    valid = bundle["token_valid"][0].bool()
    visual = F.normalize(bundle["fused"][0][valid].float(), dim=-1)
    _, costs, route1, hard1, _, v, h, prior, disable_h = transcript_cost(models, visual, text)
    arr = costs.detach().cpu().numpy().copy()
    changed = arr.copy()
    interior = route1[1:-1] if len(route1) > 2 else route1
    if interior:
        i, j = interior[len(interior) // 2]
        changed[i, j] += max(5.0, float(np.std(arr) * 20.0 + 1.0))
    hard_h = 1e4 if disable_h and changed.shape[0] >= changed.shape[1] else h
    route2, hard2 = _hard_dtw(changed, v, hard_h, prior)
    save_matrix(out / "01_original_synthetic_cost_route.png", arr, "Original synthetic DTW cost/route", "letter", "window", route1)
    save_matrix(out / "02_perturbed_synthetic_cost_route.png", changed, "Perturbed current cost matrix/recomputed route", "letter", "window", route2)
    status(
        out,
        f"""POINT 10 — DTW RECOMPUTE ON SYNTHETIC COST MATRIX
{model_note}
route_changed={route1 != route2}
original_hard_cost={hard1:.6f}
perturbed_hard_cost={hard2:.6f}

The second route is recomputed from a deliberately perturbed version of this
real synthetic line's own cost matrix. We do not force the route to change;
route_changed only reports what the current costs imply.""",
        {
            "route_changed": route1 != route2,
            "original_cost": hard1,
            "perturbed_cost": hard2,
            "original_route": route1,
            "perturbed_route": route2,
        },
    )


def point11(dataset, index, out, models, model_note):
    if models is None:
        status(out, f"POINT 11 — MULTIPLE SPATIAL VECTORS ON SYNTHETIC WINDOW\n{model_note}")
        return
    _, _, prepared, _ = prepared_line(dataset, 1, index)
    tensor, bundle, *_ = model_bundle(models, prepared)
    enc = models.image_model.vit_encoder.patch_embedding
    if not hasattr(enc, "spatial_vectors"):
        status(out, f"POINT 11 — MULTIPLE SPATIAL VECTORS\n{model_note}\nSKIP: encoder has no spatial_vectors()")
        return
    target = bundle["restoration_target"][0]
    selected = select_informative_indices(target, 1)[0]
    with torch.inference_mode():
        spatial = enc.spatial_vectors(tensor, vectors_per_window=4)[0, selected]
    spatial = F.normalize(spatial.float(), dim=-1)
    sim = spatial @ spatial.T
    save_matrix(out / "01_real_window_k4_cosine.png", sim, f"Real synthetic W{selected}: K=4 sub-vectors", "sub-vector", "sub-vector")
    status(
        out,
        f"""POINT 11 — MULTIPLE SPATIAL VECTORS ON SYNTHETIC WINDOW
{model_note}
selected_window={selected}
shape={tuple(spatial.shape)}
mean_offdiag_cosine={float(sim[~torch.eye(4, dtype=torch.bool, device=sim.device)].mean()):.6f}""",
        {"selected_window": selected, "shape": list(spatial.shape), "cosine": sim.cpu().tolist()},
    )


def point12(dataset, index, out):
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "restoration_synthetic_tiny_overfit.py"),
        "--dataset", str(dataset),
        "--index", str(index),
        "--output-dir", str(out),
    ]
    completed = subprocess.run(cmd, cwd=ROOT)
    if completed.returncode not in (0, 2):
        raise SystemExit(completed.returncode)
    # The delegated tool already saves the full summary and metrics. Preserve its
    # diagnostic FAIL exit as text but do not hide its artifacts.
    print(f"Point 12 synthetic tiny-overfit exit={completed.returncode}; inspect {out}")


def point13(dataset, index, out, models, model_note):
    if models is None:
        status(out, f"POINT 13 — IMAGE-ONLY SYNTHETIC PAIR EVALUATION\n{model_note}")
        return
    _, _, prepared1, _ = prepared_line(dataset, 1, index)
    _, _, prepared2, _ = prepared_line(dataset, 2, index)
    _, bundle1, *_ = model_bundle(models, prepared1)
    _, bundle2, *_ = model_bundle(models, prepared2)
    valid1 = bundle1["token_valid"][0].bool()
    valid2 = bundle2["token_valid"][0].bool()
    fused1 = F.normalize(bundle1["fused"][0][valid1].float(), dim=-1)
    fused2 = F.normalize(bundle2["fused"][0][valid2].float(), dim=-1)
    similarity = fused1 @ fused2.T
    nw = needleman_wunsch(similarity, gap_penalty=-0.30)
    route = [
        (step.i, step.j) for step in nw.steps
        if step.i is not None and step.j is not None
    ]
    prepared1.save(out / "01_synthetic_line_A.png")
    prepared2.save(out / "02_synthetic_line_B.png")
    save_matrix(out / "03_fused_image_image_similarity.png", similarity, "IMAGE ONLY: fused synthetic pair cosine + NW", "line B valid window", "line A valid window", route)
    status(
        out,
        f"""POINT 13 — IMAGE-ONLY SYNTHETIC PAIR EVALUATION
{model_note}
line_A_windows={fused1.shape[0]}
line_B_windows={fused2.shape[0]}
NW_normalized_score={float(nw.normalized_score):.6f}
mean_similarity={float(similarity.mean()):.6f}

No text embedding is used in this test.""",
        {
            "shape": list(similarity.shape),
            "nw_normalized_score": float(nw.normalized_score),
            "mean_similarity": float(similarity.mean()),
        },
    )


def main():
    args = parse_args()
    point = int(args.point)
    dataset = resolve_dataset()
    index = resolve_index(dataset)
    weights = resolve_weights()
    device = resolve_device()

    out = ROOT / "Results" / "Diagnostics" / "restoration_points" / f"point_{point:02d}_synthetic"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"RESTORATION POINT {point:02d} ON ACTUAL SYNTHETIC DATA")
    print(f"dataset={dataset}")
    print(f"pair_index={index}")
    print(f"weights={weights if weights is not None else '<none>'}")
    print(f"device={device}")
    print(f"output={out}")
    print("=" * 72)

    # Only load a trained checkpoint when this point needs learned model behavior.
    learned_points = {2, 4, 5, 6, 7, 8, 9, 10, 11, 13}
    models, model_note = (None, "not required")
    if point in learned_points:
        models, model_note = load_models(weights, device)

    if point == 1:
        point1(dataset, index, out)
    elif point == 2:
        point2(dataset, index, out, models, model_note)
    elif point == 3:
        point3(dataset, index, out)
    elif point == 4:
        point4(dataset, index, out, models, model_note)
    elif point == 5:
        point5(dataset, index, out, models, model_note)
    elif point == 6:
        point6(dataset, index, out, models, model_note)
    elif point == 7:
        point7(dataset, index, out, models, model_note)
    elif point == 8:
        point8(dataset, index, out, models, model_note)
    elif point == 9:
        point9(dataset, index, out, models, model_note)
    elif point == 10:
        point10(dataset, index, out, models, model_note)
    elif point == 11:
        point11(dataset, index, out, models, model_note)
    elif point == 12:
        point12(dataset, index, out)
    elif point == 13:
        point13(dataset, index, out, models, model_note)

    print(f"\nOpen: {out / 'summary.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
