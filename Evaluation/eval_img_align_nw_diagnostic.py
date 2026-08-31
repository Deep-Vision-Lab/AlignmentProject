#!/usr/bin/env python3
"""Unified Needleman-Wunsch image-image alignment diagnostic.

Required inputs:
    python Evaluation/eval_img_align_nw_diagnostic.py \
        --dataset <dataset-root-or-manifest> \
        --weights <Weights/job_id/model_best.pth>

All evaluation defaults come from Parameters.py. The dataset resolver supports:

* synthetic flat pairs::

      <root>/images/img1_*.png
      <root>/images/img2_*.png
      <root>/masks/mask1_*.png, mask2_*.png   # optional ground truth

* real ArabicDataset::

      <root>/dataset_manifest.jsonl

* explicit real/injection splits::

      <root>/train_manifest.jsonl
      <root>/valid_manifest.jsonl
      <root>/test_manifest.jsonl

* native RealSyntheticBridge_v3::

      <root>/images/real/<anchor_id>/real.*
      <root>/images/positive/<anchor_id>/positive.png
      <root>/masks/positive/<anchor_id>/positive_mask.png
      <root>/dataset_manifest.jsonl
      <root>/metadata.json

* generic .jsonl/.json/.csv pair manifests. A generic manifest can describe a
  real-synthetic pair by setting per-side ``dataset_type``/``domain`` fields.

For every selected pair the script computes window-to-window cosine similarity,
builds the configured NW match-score matrix, runs GLOBAL Needleman-Wunsch, and
saves:

* both lines one under the other with red predicted alignment masks;
* a value-annotated cosine matrix + NW traceback;
* a value-annotated NW diagonal-match matrix + traceback;
* a value-annotated accumulated NW DP matrix + traceback;
* binary predicted masks for both lines;
* numeric CSV/NPY matrices and JSON traceback evidence;
* synthetic-mask IoU/Dice/precision/recall when a ground-truth mask exists.

Arabic physical RTL display is preserved by the shared evaluation geometry.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Parameters as P

P.export_environment()
os.environ.setdefault("ANNOTATE_HEATMAP_VALUES", "1")
os.environ.setdefault("SAVE_HEATMAP_CSV", "1")

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
)
from Evaluation.sw_core import build_match_scores, resolve_score_mode
from Evaluation.sw_dataset import (
    display_image,
    load_arabic_dataset_pairs,
    read_manifest_records,
    resolve_manifest_image,
)
from Evaluation.trace_components import (
    component_intervals_px,
    component_metrics,
    nw_component_path,
    nw_traceback_boundaries,
    save_alignment_visualization,
    save_numeric_evidence,
)

_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
_REAL_MANIFEST = str(getattr(P, "real_manifest_name", "dataset_manifest.jsonl"))


@dataclass(frozen=True)
class Pair:
    index: int
    image1: Path
    image2: Path
    side1_type: str
    side2_type: str
    source_type: str
    pair_id: str = ""
    label_type: str = ""
    text_score: float = 0.0
    manifest_position: int = -1
    split: str = ""
    gt_mask1: Path | None = None
    gt_mask2: Path | None = None


def _domain(value, default="synthetic") -> str:
    value = str(value or default).strip().lower().replace("_", "-")
    if value in {"real", "scan", "manuscript", "handwritten"}:
        return "real"
    if value in {"synthetic", "synth", "generated", "rendered"}:
        return "synthetic"
    return str(default)


def _allowed_labels() -> set[str] | None:
    value = str(getattr(P, "real_dataset_labels", "high_match,medium_match")).strip()
    if value.lower() in {"", "all", "*", "any"}:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _resolve_path(value, manifest: Path) -> Path:
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    for candidate in (manifest.parent / path, ROOT / path):
        if candidate.is_file():
            return candidate
    return manifest.parent / path


def _first(mapping: dict, *keys):
    for key in keys:
        value = mapping.get(key)
        if value not in (None, ""):
            return value
    return None


def _real_manifest(root: Path) -> Path | None:
    candidate = root / _REAL_MANIFEST
    return candidate if candidate.is_file() else None


def _explicit_split_manifest(root: Path, split: str) -> list[tuple[str, Path]]:
    available = {
        name: root / f"{name}_manifest.jsonl"
        for name in ("train", "valid", "test")
    }
    if not all(path.is_file() for path in available.values()):
        return []
    if split == "all":
        return list(available.items())
    return [(split, available[split])]


def _bridge_v3_root(root: Path) -> bool:
    if not root.is_dir() or not (root / "dataset_manifest.jsonl").is_file():
        return False
    metadata = root / "metadata.json"
    if metadata.is_file():
        try:
            payload = json.loads(metadata.read_text(encoding="utf-8"))
            if int(payload.get("dataset_version", 0)) == 3:
                return True
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass
    return (
        (root / "images" / "real").is_dir()
        and (root / "images" / "positive").is_dir()
    )


def _is_synthetic_flat(root: Path) -> bool:
    images = root / "images"
    return images.is_dir() and any(
        path.is_file() and path.name.startswith("img1_") for path in images.iterdir()
    )


def _find_matching_image(images: Path, role: int, index: int) -> Path | None:
    for suffix in _IMAGE_SUFFIXES:
        candidate = images / f"img{role}_{index}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def _synthetic_mask(image: Path) -> Path | None:
    if image.name.startswith("img1_"):
        name = "mask1_" + image.name[len("img1_") :]
    elif image.name.startswith("img2_"):
        name = "mask2_" + image.name[len("img2_") :]
    else:
        return None
    path = image.parent.parent / "masks" / name
    return path if path.is_file() else None


def _synthetic_pairs(root: Path) -> list[Pair]:
    images = root / "images"
    pattern = re.compile(r"^img1_(\d+)\.[^.]+$", re.I)
    indices = sorted(
        {
            int(match.group(1))
            for path in images.iterdir()
            if path.is_file() and (match := pattern.match(path.name))
        }
    )
    pairs = []
    for dataset_index in indices:
        image1 = _find_matching_image(images, 1, dataset_index)
        image2 = _find_matching_image(images, 2, dataset_index)
        if image1 is None or image2 is None:
            continue
        pairs.append(
            Pair(
                index=len(pairs) + 1,
                image1=image1,
                image2=image2,
                side1_type="synthetic",
                side2_type="synthetic",
                source_type="synthetic",
                pair_id=f"synthetic_{dataset_index}",
                manifest_position=dataset_index,
                gt_mask1=_synthetic_mask(image1),
                gt_mask2=_synthetic_mask(image2),
            )
        )
    return pairs


def _real_pairs(manifest: Path, split: str) -> list[Pair]:
    loader_args = SimpleNamespace(
        arabic_manifest=str(manifest),
        data_dir=str(manifest.parent),
        real_text_key=str(getattr(P, "real_text_key", "text_original_path")),
        real_labels=str(getattr(P, "real_dataset_labels", "high_match,medium_match")),
        real_min_text_score=float(getattr(P, "real_min_text_score", 0.0)),
        real_validate_paths=bool(getattr(P, "real_validate_paths", False)),
        split_seed=int(getattr(P, "dataset_split_seed", 42)),
        real_split=split,
    )
    loaded = load_arabic_dataset_pairs(loader_args)
    return [
        Pair(
            index=position,
            image1=Path(item.image1),
            image2=Path(item.image2),
            side1_type="real",
            side2_type="real",
            source_type="real",
            pair_id=item.pair_id,
            label_type=item.label_type,
            text_score=float(item.text_score),
            manifest_position=int(item.manifest_position),
            split=item.split,
        )
        for position, item in enumerate(loaded, 1)
    ]


def _explicit_real_pairs(root: Path, split: str) -> list[Pair]:
    pairs: list[Pair] = []
    allowed = _allowed_labels()
    for split_name, manifest in _explicit_split_manifest(root, split):
        for record in read_manifest_records(manifest):
            label = str(record.get("label_type", ""))
            if allowed is not None and label and label not in allowed:
                continue
            if not isinstance(record.get("A"), dict) or not isinstance(record.get("B"), dict):
                raise ValueError(
                    f"Explicit split manifest must contain nested A/B records: {manifest}"
                )
            a, b = record["A"], record["B"]
            image1 = _resolve_path(_first(a, "line_image_path", "image_path", "image"), manifest)
            image2 = _resolve_path(_first(b, "line_image_path", "image_path", "image"), manifest)
            scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
            pairs.append(
                Pair(
                    index=len(pairs) + 1,
                    image1=image1,
                    image2=image2,
                    side1_type="real",
                    side2_type="real",
                    source_type="real-synthetic-injection",
                    pair_id=str(record.get("pair_id", len(pairs) + 1)),
                    label_type=label,
                    text_score=float(scores.get("text_score", 0.0) or 0.0),
                    manifest_position=len(pairs) + 1,
                    split=split_name,
                )
            )
    return pairs


def _bridge_mask(record: dict, manifest: Path) -> Path | None:
    b = record.get("B") if isinstance(record.get("B"), dict) else {}
    bridge = record.get("bridge") if isinstance(record.get("bridge"), dict) else {}
    value = b.get("alignment_mask_path") or bridge.get("alignment_mask_path")
    if not value:
        return None
    path = _resolve_path(value, manifest)
    return path if path.is_file() else None


def _bridge_v3_pairs(root: Path) -> list[Pair]:
    manifest = root / "dataset_manifest.jsonl"
    allowed = _allowed_labels()
    pairs = []
    for position, record in enumerate(read_manifest_records(manifest), 1):
        if not isinstance(record.get("A"), dict) or not isinstance(record.get("B"), dict):
            continue
        label = str(record.get("label_type", ""))
        if allowed is not None and label not in allowed:
            continue
        a, b = record["A"], record["B"]
        image1_value = _first(a, "line_image_path", "image_path", "image")
        image2_value = _first(b, "line_image_path", "image_path", "image")
        if image1_value is None or image2_value is None:
            raise ValueError(f"Bridge V3 row {position} is missing A/B image paths")
        scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
        pairs.append(
            Pair(
                index=len(pairs) + 1,
                image1=_resolve_path(image1_value, manifest),
                image2=_resolve_path(image2_value, manifest),
                side1_type="real",
                side2_type="synthetic",
                source_type="real-synthetic-bridge-v3",
                pair_id=str(record.get("pair_id", position)),
                label_type=label,
                text_score=float(scores.get("text_score", 0.0) or 0.0),
                manifest_position=position,
                split=str(record.get("split", "")),
                gt_mask2=_bridge_mask(record, manifest),
            )
        )
    return pairs


def _infer_side_type(side: dict, record: dict, image: Path, role: int, fallback: str) -> str:
    for mapping in (side, record):
        for key in (
            "dataset_type",
            "domain",
            "source_type",
            "image_type",
            f"side{role}_type",
            f"image{role}_type",
        ):
            if mapping.get(key) not in (None, ""):
                return _domain(mapping[key], fallback)
    lower = str(image).lower()
    if "synthetic" in lower or image.name.startswith(("img1_", "img2_")):
        return "synthetic"
    if any(token in lower for token in ("arabicdataset", "datasetpairs", "linesimages", "/real/")):
        return "real"
    return fallback


def _generic_manifest_pairs(manifest: Path) -> list[Pair]:
    pairs = []
    for position, record in enumerate(read_manifest_records(manifest), 1):
        nested = isinstance(record.get("A"), dict) and isinstance(record.get("B"), dict)
        if nested:
            a, b = record["A"], record["B"]
            value1 = _first(a, "line_image_path", "image_path", "image", "image1")
            value2 = _first(b, "line_image_path", "image_path", "image", "image2")
            if value1 is None or value2 is None:
                raise ValueError(f"Manifest row {position} is missing A/B image paths")
            image1 = _resolve_path(value1, manifest)
            image2 = _resolve_path(value2, manifest)
            fallback = "real" if manifest.name == _REAL_MANIFEST else "synthetic"
            scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
            text_score = float(scores.get("text_score", record.get("text_score", 0.0)) or 0.0)
        else:
            a = b = record
            value1 = _first(record, "image1", "image_1", "line1", "line_1", "source")
            value2 = _first(record, "image2", "image_2", "line2", "line_2", "target")
            if value1 is None or value2 is None:
                raise ValueError(f"Manifest row {position} is missing image1/image2")
            image1 = resolve_manifest_image(str(value1), manifest, manifest.parent)
            image2 = resolve_manifest_image(str(value2), manifest, manifest.parent)
            fallback = _domain(record.get("dataset_type"), "synthetic")
            text_score = float(record.get("text_score", 0.0) or 0.0)
        type1 = _infer_side_type(a, record, image1, 1, fallback)
        type2 = _infer_side_type(b, record, image2, 2, fallback)
        source_type = type1 if type1 == type2 else "real-synthetic"
        gt_mask1 = _synthetic_mask(image1) if type1 == "synthetic" else None
        gt_mask2 = _synthetic_mask(image2) if type2 == "synthetic" else None
        if nested and type2 == "synthetic" and gt_mask2 is None:
            gt_mask2 = _bridge_mask(record, manifest)
        pairs.append(
            Pair(
                index=len(pairs) + 1,
                image1=image1,
                image2=image2,
                side1_type=type1,
                side2_type=type2,
                source_type=source_type,
                pair_id=str(record.get("pair_id", record.get("id", position))),
                label_type=str(record.get("label_type", "")),
                text_score=text_score,
                manifest_position=position,
                split=str(record.get("split", "")),
                gt_mask1=gt_mask1,
                gt_mask2=gt_mask2,
            )
        )
    return pairs


def _load_one_root(root: Path, real_split: str) -> tuple[str, list[Pair]]:
    if _bridge_v3_root(root):
        return "real-synthetic-bridge-v3", _bridge_v3_pairs(root)
    if _explicit_split_manifest(root, real_split):
        return "real-synthetic-injection", _explicit_real_pairs(root, real_split)
    manifest = _real_manifest(root)
    if manifest is not None:
        return "real", _real_pairs(manifest, real_split)
    if _is_synthetic_flat(root):
        return "synthetic", _synthetic_pairs(root)
    generic = next(
        (
            root / name
            for name in ("pair_manifest.jsonl", "pairs.jsonl", "pairs.json", "pairs.csv")
            if (root / name).is_file()
        ),
        None,
    )
    if generic is not None:
        return "manifest", _generic_manifest_pairs(generic)
    return "unknown", []


def load_pairs(dataset: Path, real_split: str) -> tuple[str, list[Pair]]:
    if dataset.is_file():
        parent = dataset.parent
        if dataset.name == "dataset_manifest.jsonl" and _bridge_v3_root(parent):
            return "real-synthetic-bridge-v3", _bridge_v3_pairs(parent)
        if dataset.name in {"train_manifest.jsonl", "valid_manifest.jsonl", "test_manifest.jsonl"}:
            split_name = dataset.name.split("_", 1)[0]
            return "real-synthetic-injection", _explicit_real_pairs(parent, split_name)
        if dataset.name == _REAL_MANIFEST:
            return "real", _real_pairs(dataset, real_split)
        return "manifest", _generic_manifest_pairs(dataset)

    direct_kind, direct_pairs = _load_one_root(dataset, real_split)
    if direct_pairs:
        return direct_kind, direct_pairs

    groups: list[tuple[str, list[Pair]]] = []
    for child in sorted((path for path in dataset.iterdir() if path.is_dir()), key=lambda path: path.name):
        kind, pairs = _load_one_root(child, real_split)
        if pairs:
            groups.append((kind, pairs))
    if not groups:
        raise ValueError(
            "Unrecognized dataset layout. Expected a synthetic images/ folder, "
            "ArabicDataset manifest, explicit split manifests, Bridge V3, or pair manifest."
        )

    combined = []
    kinds = set()
    for kind, pairs in groups:
        kinds.add(kind)
        combined.extend(pairs)
    combined = [replace(pair, index=index) for index, pair in enumerate(combined, 1)]
    return (next(iter(kinds)) if len(kinds) == 1 else "real-synthetic-mixed-root"), combined


def _prepare(path: Path, domain: str, temporary_root: Path, role: int):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input image: {path}")
    if domain == "real":
        array = display_image(path, "real")
        model_path = temporary_root / f"line{role}.png"
        Image.fromarray(array).save(model_path)
        # display_image already applied the exact real binarization. Feeding the
        # temporary image as synthetic prevents a second Otsu/binarization pass.
        return array, model_path
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB")), path


def _predicted_mask(shape, intervals) -> np.ndarray:
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for left, right in intervals:
        lo = max(0, min(width, int(math.floor(min(left, right)))))
        hi = max(0, min(width, int(math.ceil(max(left, right)))))
        if hi > lo:
            mask[:, lo:hi] = 255
    return mask


def _load_gt_mask(path: Path | None, shape) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    height, width = shape[:2]
    resampling = getattr(Image, "Resampling", Image).NEAREST
    with Image.open(path) as opened:
        mask = opened.convert("L").resize((width, height), resampling)
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def _mask_metrics(pred: np.ndarray, gt: np.ndarray | None, prefix: str) -> dict:
    if gt is None:
        return {
            f"{prefix}_mask_iou": None,
            f"{prefix}_mask_dice": None,
            f"{prefix}_mask_precision": None,
            f"{prefix}_mask_recall": None,
        }
    pred_bool, gt_bool = pred > 0, gt > 0
    tp = int(np.logical_and(pred_bool, gt_bool).sum())
    fp = int(np.logical_and(pred_bool, ~gt_bool).sum())
    fn = int(np.logical_and(~pred_bool, gt_bool).sum())
    return {
        f"{prefix}_mask_precision": tp / (tp + fp) if tp + fp else 0.0,
        f"{prefix}_mask_recall": tp / (tp + fn) if tp + fn else 0.0,
        f"{prefix}_mask_iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        f"{prefix}_mask_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
    }


def _save_path_values(path: Path, result, cosine: np.ndarray, match_scores: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["trace_step", "operation", "window1", "window2", "cosine", "match_score"],
        )
        writer.writeheader()
        for position, step in enumerate(result.steps):
            i, j = step.index1, step.index2
            writer.writerow(
                {
                    "trace_step": position,
                    "operation": step.operation,
                    "window1": "" if i is None else int(i),
                    "window2": "" if j is None else int(j),
                    "cosine": "" if i is None or j is None else f"{float(cosine[i, j]):.8f}",
                    "match_score": "" if i is None or j is None else f"{float(match_scores[i, j]):.8f}",
                }
            )


def _visualize(
    models,
    pair,
    arr1,
    arr2,
    features1,
    features2,
    full_path,
    component_path,
    traceback,
    matrix,
    label,
    result,
    score_mode,
    output,
):
    save_alignment_visualization(
        arr1=arr1,
        arr2=arr2,
        features1=features1,
        features2=features2,
        full_path=full_path,
        component_path=component_path,
        traceback=traceback,
        heatmap_matrix=matrix,
        heatmap_label=label,
        score=float(result.score),
        normalized_score=float(result.normalized_score),
        output=output,
        use_flip=bool(models.image_model.use_flip),
        pair=pair,
        score_mode=score_mode + "+ink",
        algorithm="Needleman-Wunsch",
        traceback_label="NW traceback: terminal (N,M) → origin (0,0)",
        traceback_start_label="terminal DP boundary (N,M)",
        traceback_end_label="global origin (0,0)",
        binarized=pair.side1_type == "real" or pair.side2_type == "real",
        annotate_values=True,
        window_size=getattr(models.image_model, "window_size", None),
        stride=getattr(models.image_model, "stride", None),
    )


def evaluate(models, pair: Pair, args, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nw_diag_") as temporary:
        arr1, model_image1 = _prepare(pair.image1, pair.side1_type, Path(temporary), 1)
        arr2, model_image2 = _prepare(pair.image2, pair.side2_type, Path(temporary), 2)

        features1 = get_image_features(models, model_image1, "synthetic")
        features2 = get_image_features(models, model_image2, "synthetic")
        cosine = compute_similarity(
            features1.select(args.feature), features2.select(args.feature)
        ).detach().cpu().numpy().astype(np.float32)

        scoring_domain = (
            "real"
            if "real" in {pair.side1_type, pair.side2_type}
            else "synthetic"
        )
        resolved_mode = resolve_score_mode(args.score_mode, scoring_domain)
        match_scores = build_match_scores(
            cosine, resolved_mode, args.score_clip, args.threshold
        )
        match_scores = ink_aware_match_scores(
            match_scores,
            features1.ink.detach().cpu().numpy(),
            features2.ink.detach().cpu().numpy(),
        )

        result = needleman_wunsch(
            match_scores,
            gap_penalty=float(args.gap),
            similarity_offset=0.0,
        )
        full_path = list(result.pairs)
        traceback = nw_traceback_boundaries(result)
        component_path = nw_component_path(result, match_scores)

        window_size = getattr(models.image_model, "window_size", None)
        stride = getattr(models.image_model, "stride", None)
        intervals1 = component_intervals_px(
            component_path,
            0,
            cosine.shape[0],
            arr1.shape[1],
            bool(models.image_model.use_flip),
            window_size=window_size,
            stride=stride,
        )
        intervals2 = component_intervals_px(
            component_path,
            1,
            cosine.shape[1],
            arr2.shape[1],
            bool(models.image_model.use_flip),
            window_size=window_size,
            stride=stride,
        )

        _visualize(
            models,
            pair,
            arr1,
            arr2,
            features1,
            features2,
            full_path,
            component_path,
            traceback,
            cosine,
            "raw cosine similarity (every window-pair value shown)",
            result,
            resolved_mode,
            output_dir / "cosine_similarity_values.png",
        )
        _visualize(
            models,
            pair,
            arr1,
            arr2,
            features1,
            features2,
            full_path,
            component_path,
            traceback,
            match_scores,
            "NW diagonal match score (every value shown)",
            result,
            resolved_mode,
            output_dir / "nw_match_scores_values.png",
        )
        dp_scores = np.asarray(result.score_matrix[1:, 1:], dtype=np.float32)
        _visualize(
            models,
            pair,
            arr1,
            arr2,
            features1,
            features2,
            full_path,
            component_path,
            traceback,
            dp_scores,
            "accumulated global NW DP score (every value shown)",
            result,
            resolved_mode,
            output_dir / "nw_dp_trace_values.png",
        )

        evidence = save_numeric_evidence(
            output_dir / "nw_match_scores_values.png",
            algorithm="Needleman-Wunsch",
            raw_similarity=cosine,
            match_scores=match_scores,
            component_path=component_path,
            full_path=full_path,
            traceback=traceback,
            intervals1=intervals1,
            intervals2=intervals2,
        )
        np.savetxt(output_dir / "nw_dp_scores.csv", dp_scores, delimiter=",", fmt="%.8f")
        np.save(output_dir / "nw_dp_scores.npy", dp_scores)
        np.save(output_dir / "cosine_similarity.npy", cosine)
        np.save(output_dir / "nw_match_scores.npy", match_scores)
        _save_path_values(output_dir / "nw_trace_path_values.csv", result, cosine, match_scores)

        pred1 = _predicted_mask(arr1.shape, intervals1)
        pred2 = _predicted_mask(arr2.shape, intervals2)
        Image.fromarray(pred1).save(output_dir / "line1_pred_mask.png")
        Image.fromarray(pred2).save(output_dir / "line2_pred_mask.png")

        gt1 = _load_gt_mask(pair.gt_mask1, pred1.shape)
        gt2 = _load_gt_mask(pair.gt_mask2, pred2.shape)
        if gt1 is not None:
            Image.fromarray(gt1).save(output_dir / "line1_gt_mask.png")
        if gt2 is not None:
            Image.fromarray(gt2).save(output_dir / "line2_gt_mask.png")

        metrics = component_metrics(component_path, cosine.shape)
        supported_cosines = [float(cosine[i, j]) for i, j in component_path]
        full_cosines = [float(cosine[i, j]) for i, j in full_path]
        gap_steps = sum(
            1
            for step in result.steps
            if step.index1 is None or step.index2 is None
        )
        row = {
            "index": int(pair.index),
            "pair_id": pair.pair_id,
            "source_type": pair.source_type,
            "side1_type": pair.side1_type,
            "side2_type": pair.side2_type,
            "label_type": pair.label_type,
            "text_score": float(pair.text_score),
            "split": pair.split,
            "image1": str(pair.image1),
            "image2": str(pair.image2),
            "nw_score": float(result.score),
            "normalized_nw_score": float(result.normalized_score),
            "feature": args.feature,
            "score_mode": resolved_mode,
            "score_clip": float(args.score_clip),
            "threshold": float(args.threshold),
            "gap": float(args.gap),
            "line1_windows": int(cosine.shape[0]),
            "line2_windows": int(cosine.shape[1]),
            "gap_steps": int(gap_steps),
            "mean_path_cosine": float(np.mean(supported_cosines)) if supported_cosines else None,
            "mean_full_path_cosine": float(np.mean(full_cosines)) if full_cosines else None,
            "line1_intervals_px": intervals1,
            "line2_intervals_px": intervals2,
            "gt_mask1": str(pair.gt_mask1) if pair.gt_mask1 else "",
            "gt_mask2": str(pair.gt_mask2) if pair.gt_mask2 else "",
            **metrics,
            **_mask_metrics(pred1, gt1, "line1"),
            **_mask_metrics(pred2, gt2, "line2"),
            **evidence,
        }
        available_ious = [
            value
            for value in (row["line1_mask_iou"], row["line2_mask_iou"])
            if value is not None
        ]
        row["mean_mask_iou"] = (
            float(np.mean(available_ious)) if available_ious else None
        )
        (output_dir / "summary.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument(
        "--n-samples",
        type=int,
        default=int(getattr(P, "evaluation_n_samples", 100)),
    )
    parser.add_argument(
        "--real-split",
        choices=("all", "train", "valid", "test"),
        default=str(getattr(P, "evaluation_real_split", "test")),
    )
    parser.add_argument(
        "--feature",
        choices=("contextual", "local", "grouped"),
        default=str(getattr(P, "evaluation_feature", "contextual")),
    )
    parser.add_argument(
        "--score-mode",
        choices=("auto", "raw", "centered", "mutual-z"),
        default=str(getattr(P, "evaluation_score_mode", "auto")),
    )
    parser.add_argument(
        "--score-clip",
        type=float,
        default=float(getattr(P, "evaluation_score_clip", 4.0)),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(getattr(P, "evaluation_threshold", 0.0)),
    )
    parser.add_argument(
        "--gap",
        type=float,
        default=float(getattr(P, "evaluation_gap", -0.30)),
    )
    return parser.parse_args()


def _mean(rows, key):
    values = []
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            values.append(numeric)
    return float(np.mean(values)) if values else None


def main():
    args = parse_args()
    if args.start_index <= 0:
        raise SystemExit("--start-index is 1-based and must be positive")
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be positive")

    dataset = Path(args.dataset).expanduser().resolve()
    weights = Path(args.weights).expanduser().resolve()
    if not dataset.exists():
        raise SystemExit(f"Dataset does not exist: {dataset}")
    if not weights.is_file():
        raise SystemExit(f"Weights do not exist: {weights}")

    layout, pairs = load_pairs(dataset, args.real_split)
    if not pairs:
        raise SystemExit("No evaluable image pairs were found")
    start = args.start_index - 1
    selected = pairs[start : start + args.n_samples]
    if not selected:
        raise SystemExit("Selected sample range is outside the dataset")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else ROOT
        / "Results"
        / "Evaluation"
        / "NW"
        / (dataset.stem if dataset.is_file() else dataset.name)
        / weights.parent.name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(
        "NW diagnostic "
        f"dataset={dataset} detected={layout} total_pairs={len(pairs)} "
        f"selected={len(selected)} weights={weights} output={output_dir}",
        flush=True,
    )
    print(
        f"Parameters: feature={args.feature} score_mode={args.score_mode} "
        f"score_clip={args.score_clip} threshold={args.threshold} gap={args.gap} "
        f"real_split={args.real_split}",
        flush=True,
    )

    models = load_evaluation_models(weights, args.device, load_text_model=False)
    rows = []
    for pair in selected:
        pair_dir = output_dir / f"pair_{pair.index:05d}"
        try:
            row = evaluate(models, pair, args, pair_dir)
            row["status"] = "ok"
            row["output"] = str(pair_dir)
            print(
                f"[{pair.index}] type={pair.source_type} "
                f"NW={row['normalized_nw_score']:.4f} "
                f"components={row['component_count']} "
                f"supported={row['path_steps']}/{row['full_match_steps']} "
                f"mean_cos={row['mean_path_cosine']} mask_iou={row['mean_mask_iou']}",
                flush=True,
            )
        except Exception as exc:
            row = {
                "index": int(pair.index),
                "pair_id": pair.pair_id,
                "source_type": pair.source_type,
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "output": str(pair_dir),
            }
            print(f"[{pair.index}] failed: {row['error']}", file=sys.stderr, flush=True)
        rows.append(row)

    csv_rows = []
    for row in rows:
        csv_row = dict(row)
        for key in ("line1_intervals_px", "line2_intervals_px"):
            if key in csv_row:
                csv_row[key] = json.dumps(csv_row[key], separators=(",", ":"))
        csv_rows.append(csv_row)
    fieldnames = sorted({key for row in csv_rows for key in row})
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    successful = [row for row in rows if row.get("status") == "ok"]
    summary = {
        "algorithm": "needleman_wunsch",
        "traceback": "terminal_(N,M)_to_origin_(0,0)",
        "dataset": str(dataset),
        "detected_layout": layout,
        "weights": str(weights),
        "parameters": str(ROOT / "Parameters.py"),
        "selected": len(selected),
        "successful": len(successful),
        "failed": len(rows) - len(successful),
        "mean_normalized_nw_score": _mean(successful, "normalized_nw_score"),
        "mean_path_cosine": _mean(successful, "mean_path_cosine"),
        "mean_mask_iou": _mean(successful, "mean_mask_iou"),
        "feature": args.feature,
        "score_mode": args.score_mode,
        "score_clip": args.score_clip,
        "threshold": args.threshold,
        "gap": args.gap,
        "real_split": args.real_split,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved summary: {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
