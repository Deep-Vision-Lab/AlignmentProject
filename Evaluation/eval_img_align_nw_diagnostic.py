#!/usr/bin/env python3
"""Unified Needleman-Wunsch image-image alignment diagnostic.

Required inputs:
    python Evaluation/eval_img_align_nw_diagnostic.py \
        --dataset <dataset-root-or-manifest> \
        --weights <Weights/job_id/model_best.pth>

Defaults come from Parameters.py. The script auto-detects:
  * synthetic: <root>/images/img1_*.png + img2_*.png
  * real: <root>/dataset_manifest.jsonl
  * real-synthetic: a root containing both kinds, or a generic pair manifest
    whose two sides declare/indicate different domains.

For each pair it saves three value-annotated NW visualizations (cosine, diagonal
match score, accumulated DP score), the global traceback, predicted masks for
both lines, numeric matrices/trace evidence, and synthetic-mask IoU when ground
truth masks are available.
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

from unified_line_geometry import install_evaluation_geometry
install_evaluation_geometry()
from Evaluation.vit_evaluation import install_vit_evaluation_loader
install_vit_evaluation_loader()
from Evaluation.zero_shot_sw import install_dataset_patches, ink_aware_match_scores
install_dataset_patches()

from Evaluation._eval_utils import compute_similarity, get_image_features, load_evaluation_models, needleman_wunsch
from Evaluation.sw_core import build_match_scores, resolve_score_mode
from Evaluation.sw_dataset import display_image, load_arabic_dataset_pairs, read_manifest_records, resolve_manifest_image
from Evaluation.trace_components import (
    component_intervals_px,
    component_metrics,
    nw_component_path,
    nw_traceback_boundaries,
    save_alignment_visualization,
    save_numeric_evidence,
)


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


def _domain(value, default="synthetic"):
    value = str(value or default).strip().lower().replace("_", "-")
    if value in {"real", "scan", "manuscript"}:
        return "real"
    if value in {"synthetic", "synth", "generated"}:
        return "synthetic"
    return default


def _real_manifest(root: Path):
    if root.is_file() and root.suffix.lower() == ".jsonl":
        return root
    candidate = root / str(getattr(P, "real_manifest_name", "dataset_manifest.jsonl"))
    return candidate if candidate.is_file() else None


def _is_synthetic(root: Path):
    images = root / "images"
    return images.is_dir() and any(p.is_file() and p.name.startswith("img1_") for p in images.iterdir())


def _synthetic_mask(image: Path):
    if image.name.startswith("img1_"):
        name = "mask1_" + image.name[len("img1_"):]
    elif image.name.startswith("img2_"):
        name = "mask2_" + image.name[len("img2_"):]
    else:
        return None
    path = image.parent.parent / "masks" / name
    return path if path.is_file() else None


def _synthetic_pairs(root: Path):
    pattern = re.compile(r"^img1_(\d+)\.[^.]+$", re.I)
    indices = sorted({int(m.group(1)) for p in (root / "images").iterdir() if (m := pattern.match(p.name))})
    result = []
    for pos, idx in enumerate(indices, 1):
        image1 = root / "images" / f"img1_{idx}.png"
        image2 = root / "images" / f"img2_{idx}.png"
        if not image1.is_file() or not image2.is_file():
            continue
        result.append(Pair(pos, image1, image2, "synthetic", "synthetic", "synthetic",
                           pair_id=f"synthetic_{idx}", manifest_position=idx,
                           gt_mask1=_synthetic_mask(image1), gt_mask2=_synthetic_mask(image2)))
    return result


def _real_pairs(manifest: Path, split: str):
    args = SimpleNamespace(
        arabic_manifest=str(manifest), data_dir=str(manifest.parent),
        real_text_key=str(getattr(P, "real_text_key", "text_original_path")),
        real_labels=str(getattr(P, "real_dataset_labels", "high_match,medium_match")),
        real_min_text_score=float(getattr(P, "real_min_text_score", 0.0)),
        real_validate_paths=bool(getattr(P, "real_validate_paths", False)),
        split_seed=int(getattr(P, "dataset_split_seed", 42)), real_split=split,
    )
    pairs = load_arabic_dataset_pairs(args)
    return [Pair(i, Path(x.image1), Path(x.image2), "real", "real", "real",
                 pair_id=x.pair_id, label_type=x.label_type, text_score=float(x.text_score),
                 manifest_position=int(x.manifest_position), split=x.split)
            for i, x in enumerate(pairs, 1)]


def _first(mapping, *keys):
    for key in keys:
        if mapping.get(key) not in (None, ""):
            return mapping[key]
    return None


def _resolve_nested(value, manifest: Path):
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    for candidate in (manifest.parent / path, ROOT / path):
        if candidate.is_file():
            return candidate
    return manifest.parent / path


def _infer_side_type(side, record, image: Path, role, fallback):
    for mapping in (side, record):
        for key in ("dataset_type", "domain", "source_type", "image_type", f"side{role}_type", f"image{role}_type"):
            if mapping.get(key) not in (None, ""):
                return _domain(mapping[key], fallback)
    text = str(image).lower()
    if "synthetic" in text or image.name.startswith(("img1_", "img2_")):
        return "synthetic"
    if any(token in text for token in ("arabicdataset", "datasetpairs", "linesimages", "/real/")):
        return "real"
    return fallback


def _manifest_pairs(manifest: Path):
    result = []
    for pos, record in enumerate(read_manifest_records(manifest), 1):
        nested = isinstance(record.get("A"), dict) and isinstance(record.get("B"), dict)
        if nested:
            a, b = record["A"], record["B"]
            v1 = _first(a, "line_image_path", "image_path", "image", "image1")
            v2 = _first(b, "line_image_path", "image_path", "image", "image2")
            if v1 is None or v2 is None:
                raise ValueError(f"Manifest row {pos} is missing A/B image paths")
            image1, image2 = _resolve_nested(v1, manifest), _resolve_nested(v2, manifest)
            fallback = "real" if manifest.name == getattr(P, "real_manifest_name", "dataset_manifest.jsonl") else "synthetic"
            scores = record.get("scores") if isinstance(record.get("scores"), dict) else {}
            text_score = float(scores.get("text_score", record.get("text_score", 0.0)) or 0.0)
        else:
            a = b = record
            v1 = _first(record, "image1", "image_1", "line1", "line_1", "source")
            v2 = _first(record, "image2", "image_2", "line2", "line_2", "target")
            if v1 is None or v2 is None:
                raise ValueError(f"Manifest row {pos} is missing image1/image2")
            image1 = resolve_manifest_image(str(v1), manifest, manifest.parent)
            image2 = resolve_manifest_image(str(v2), manifest, manifest.parent)
            fallback = _domain(record.get("dataset_type"), "synthetic")
            text_score = float(record.get("text_score", 0.0) or 0.0)
        type1 = _infer_side_type(a, record, image1, 1, fallback)
        type2 = _infer_side_type(b, record, image2, 2, fallback)
        source = type1 if type1 == type2 else "real-synthetic"
        result.append(Pair(pos, image1, image2, type1, type2, source,
                           pair_id=str(record.get("pair_id", record.get("id", pos))),
                           label_type=str(record.get("label_type", "")), text_score=text_score,
                           manifest_position=pos, split=str(record.get("split", "")),
                           gt_mask1=_synthetic_mask(image1), gt_mask2=_synthetic_mask(image2)))
    return result


def load_pairs(dataset: Path, real_split: str):
    if dataset.is_file():
        return "manifest", _manifest_pairs(dataset)
    candidates = [dataset] + [p for p in dataset.iterdir() if p.is_dir()]
    real = [m for p in candidates if (m := _real_manifest(p)) is not None]
    synthetic = [p for p in candidates if _is_synthetic(p)]
    generic = next((dataset / name for name in ("pair_manifest.jsonl", "pairs.jsonl", "pairs.json", "pairs.csv")
                    if (dataset / name).is_file()), None)
    if generic is not None:
        return "manifest", _manifest_pairs(generic)
    pairs = []
    for manifest in real:
        pairs.extend(_real_pairs(manifest, real_split))
    for root in synthetic:
        pairs.extend(_synthetic_pairs(root))
    if not pairs:
        raise ValueError("Unrecognized dataset layout")
    kind = "real-synthetic" if real and synthetic else ("real" if real else "synthetic")
    return kind, [replace(pair, index=i) for i, pair in enumerate(pairs, 1)]


def _prepare(path: Path, domain: str, temp: Path, role: int):
    if domain == "real":
        array = display_image(path, "real")
        model_path = temp / f"line{role}.png"
        Image.fromarray(array).save(model_path)
        return array, model_path
    with Image.open(path) as opened:
        return np.asarray(opened.convert("RGB")), path


def _mask(shape, intervals):
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for left, right in intervals:
        lo, hi = int(math.floor(min(left, right))), int(math.ceil(max(left, right)))
        mask[:, max(0, lo):min(width, hi)] = 255
    return mask


def _load_gt(path, shape):
    if path is None or not path.is_file():
        return None
    height, width = shape
    with Image.open(path) as opened:
        image = opened.convert("L").resize((width, height), Image.NEAREST)
        return (np.asarray(image) > 0).astype(np.uint8) * 255


def _mask_metrics(pred, gt, prefix):
    if gt is None:
        return {f"{prefix}_mask_iou": None, f"{prefix}_mask_dice": None,
                f"{prefix}_mask_precision": None, f"{prefix}_mask_recall": None}
    p, g = pred > 0, gt > 0
    tp = int(np.logical_and(p, g).sum()); fp = int(np.logical_and(p, ~g).sum()); fn = int(np.logical_and(~p, g).sum())
    return {
        f"{prefix}_mask_precision": tp / (tp + fp) if tp + fp else 0.0,
        f"{prefix}_mask_recall": tp / (tp + fn) if tp + fn else 0.0,
        f"{prefix}_mask_iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
        f"{prefix}_mask_dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
    }


def _visual(models, pair, arr1, arr2, f1, f2, full_path, component_path, traceback,
            matrix, label, result, score_mode, output, binarized):
    save_alignment_visualization(
        arr1=arr1, arr2=arr2, features1=f1, features2=f2,
        full_path=full_path, component_path=component_path, traceback=traceback,
        heatmap_matrix=matrix, heatmap_label=label, score=float(result.score),
        normalized_score=float(result.normalized_score), output=output,
        use_flip=bool(models.image_model.use_flip), pair=pair,
        score_mode=score_mode + "+ink", algorithm="Needleman-Wunsch",
        traceback_label="NW traceback: terminal (N,M) → origin (0,0)",
        traceback_start_label="terminal DP boundary (N,M)", traceback_end_label="global origin (0,0)",
        binarized=binarized, annotate_values=True,
        window_size=getattr(models.image_model, "window_size", None),
        stride=getattr(models.image_model, "stride", None),
    )


def evaluate(models, pair: Pair, args, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nw_diag_") as tmp:
        arr1, model1 = _prepare(pair.image1, pair.side1_type, Path(tmp), 1)
        arr2, model2 = _prepare(pair.image2, pair.side2_type, Path(tmp), 2)
        f1 = get_image_features(models, model1, "synthetic")
        f2 = get_image_features(models, model2, "synthetic")
        cosine = compute_similarity(f1.select(args.feature), f2.select(args.feature)).cpu().numpy()
        scoring_domain = "real" if "real" in {pair.side1_type, pair.side2_type} else "synthetic"
        mode = resolve_score_mode(args.score_mode, scoring_domain)
        match = build_match_scores(cosine, mode, args.score_clip, args.threshold)
        match = ink_aware_match_scores(match, f1.ink.cpu().numpy(), f2.ink.cpu().numpy())
        result = needleman_wunsch(match, gap_penalty=float(args.gap), similarity_offset=0.0)
        full_path = list(result.pairs)
        traceback = nw_traceback_boundaries(result)
        components = nw_component_path(result, match)
        window_size = getattr(models.image_model, "window_size", None)
        stride = getattr(models.image_model, "stride", None)
        intervals1 = component_intervals_px(components, 0, cosine.shape[0], arr1.shape[1], bool(models.image_model.use_flip), window_size=window_size, stride=stride)
        intervals2 = component_intervals_px(components, 1, cosine.shape[1], arr2.shape[1], bool(models.image_model.use_flip), window_size=window_size, stride=stride)

        _visual(models, pair, arr1, arr2, f1, f2, full_path, components, traceback,
                cosine, "raw cosine similarity (values shown)", result, mode,
                output_dir / "cosine_similarity_values.png", pair.side1_type == "real" or pair.side2_type == "real")
        _visual(models, pair, arr1, arr2, f1, f2, full_path, components, traceback,
                match, "NW diagonal match score (values shown)", result, mode,
                output_dir / "nw_match_scores_values.png", pair.side1_type == "real" or pair.side2_type == "real")
        dp = np.asarray(result.score_matrix[1:, 1:], dtype=np.float32)
        _visual(models, pair, arr1, arr2, f1, f2, full_path, components, traceback,
                dp, "accumulated global NW DP score (values shown)", result, mode,
                output_dir / "nw_dp_trace_values.png", pair.side1_type == "real" or pair.side2_type == "real")

        evidence = save_numeric_evidence(output_dir / "nw_match_scores_values.png", algorithm="Needleman-Wunsch",
            raw_similarity=cosine, match_scores=match, component_path=components, full_path=full_path,
            traceback=traceback, intervals1=intervals1, intervals2=intervals2)
        np.savetxt(output_dir / "nw_dp_scores.csv", dp, delimiter=",", fmt="%.8f")
        np.save(output_dir / "nw_dp_scores.npy", dp)

        pred1, pred2 = _mask(arr1.shape, intervals1), _mask(arr2.shape, intervals2)
        Image.fromarray(pred1).save(output_dir / "line1_pred_mask.png")
        Image.fromarray(pred2).save(output_dir / "line2_pred_mask.png")
        gt1, gt2 = _load_gt(pair.gt_mask1, pred1.shape), _load_gt(pair.gt_mask2, pred2.shape)
        if gt1 is not None: Image.fromarray(gt1).save(output_dir / "line1_gt_mask.png")
        if gt2 is not None: Image.fromarray(gt2).save(output_dir / "line2_gt_mask.png")

        metrics = component_metrics(components, cosine.shape)
        row = {
            "index": pair.index, "pair_id": pair.pair_id, "source_type": pair.source_type,
            "side1_type": pair.side1_type, "side2_type": pair.side2_type,
            "label_type": pair.label_type, "text_score": pair.text_score, "split": pair.split,
            "image1": str(pair.image1), "image2": str(pair.image2),
            "nw_score": float(result.score), "normalized_nw_score": float(result.normalized_score),
            "feature": args.feature, "score_mode": mode, "score_clip": args.score_clip,
            "threshold": args.threshold, "gap": args.gap,
            "line1_windows": int(cosine.shape[0]), "line2_windows": int(cosine.shape[1]),
            "mean_path_cosine": float(np.mean([cosine[i,j] for i,j in components])) if components else None,
            "line1_intervals_px": intervals1, "line2_intervals_px": intervals2,
            **metrics, **_mask_metrics(pred1, gt1, "line1"), **_mask_metrics(pred2, gt2, "line2"), **evidence,
        }
        ious = [x for x in (row["line1_mask_iou"], row["line2_mask_iou"]) if x is not None]
        row["mean_mask_iou"] = float(np.mean(ious)) if ious else None
        (output_dir / "summary.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
        return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=int(getattr(P, "evaluation_n_samples", 100)))
    parser.add_argument("--real-split", choices=("all","train","valid","test"), default=str(getattr(P, "evaluation_real_split", "test")))
    parser.add_argument("--feature", choices=("contextual","local","grouped"), default=str(getattr(P, "evaluation_feature", "contextual")))
    parser.add_argument("--score-mode", choices=("auto","raw","centered","mutual-z"), default=str(getattr(P, "evaluation_score_mode", "auto")))
    parser.add_argument("--score-clip", type=float, default=float(getattr(P, "evaluation_score_clip", 4.0)))
    parser.add_argument("--threshold", type=float, default=float(getattr(P, "evaluation_threshold", 0.0)))
    parser.add_argument("--gap", type=float, default=float(getattr(P, "evaluation_gap", -0.30)))
    return parser.parse_args()


def main():
    args = parse_args()
    dataset, weights = Path(args.dataset).expanduser().resolve(), Path(args.weights).expanduser().resolve()
    if not weights.is_file(): raise SystemExit(f"Weights do not exist: {weights}")
    layout, pairs = load_pairs(dataset, args.real_split)
    if not pairs: raise SystemExit("No evaluable pairs found")
    selected = pairs[args.start_index-1:args.start_index-1+args.n_samples]
    if not selected: raise SystemExit("Selected range is outside the dataset")
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else ROOT / "Results" / "Evaluation" / "NW" / (dataset.stem if dataset.is_file() else dataset.name) / weights.parent.name
    output.mkdir(parents=True, exist_ok=True)
    print(f"NW diagnostic dataset={dataset} detected={layout} pairs={len(pairs)} selected={len(selected)} weights={weights} output={output}", flush=True)
    models = load_evaluation_models(weights, args.device, load_text_model=False)
    rows = []
    for pair in selected:
        pair_dir = output / f"pair_{pair.index:05d}"
        try:
            row = evaluate(models, pair, args, pair_dir); row["status"] = "ok"; row["output"] = str(pair_dir)
            print(f"[{pair.index}] type={pair.source_type} NW={row['normalized_nw_score']:.4f} components={row['component_count']} mask_iou={row['mean_mask_iou']}", flush=True)
        except Exception as exc:
            row = {"index": pair.index, "pair_id": pair.pair_id, "source_type": pair.source_type,
                   "status": "error", "error": f"{type(exc).__name__}: {exc}", "output": str(pair_dir)}
            print(f"[{pair.index}] failed: {row['error']}", file=sys.stderr, flush=True)
        rows.append(row)
    fields = sorted({k for row in rows for k in row if k not in {"line1_intervals_px","line2_intervals_px"}})
    with (output / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    def mean(key):
        vals = [float(r[key]) for r in ok if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None
    summary = {"algorithm":"needleman_wunsch", "dataset":str(dataset), "detected_layout":layout,
               "weights":str(weights), "parameters":str(ROOT / "Parameters.py"),
               "selected":len(selected), "successful":len(ok), "failed":len(rows)-len(ok),
               "mean_normalized_nw_score":mean("normalized_nw_score"), "mean_path_cosine":mean("mean_path_cosine"),
               "mean_mask_iou":mean("mean_mask_iou"), "feature":args.feature, "score_mode":args.score_mode,
               "threshold":args.threshold, "gap":args.gap, "real_split":args.real_split}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved summary: {output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
