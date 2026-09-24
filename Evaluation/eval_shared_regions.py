"""Image-only shared Arabic line regions, with checkpoint geometry and affine SW.

Public entry point: python -m Evaluation.eval_shared_regions --help
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime
import hashlib
import json
from pathlib import Path
import uuid

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from Evaluation._eval_utils import load_evaluation_models
from Evaluation.point2_runtime import point2_pair_features
from Evaluation.shared_regions import (RegionSettings, extract_regions, intervals_mask,
    region_source_intervals, valid_window_geometry)
from Evaluation.point3_spatial_metrics import (_binary_metrics, _runs, _load_mask,
    _region_matches, score_source_regions, source_window_intervals)


LIMITATIONS = [
    "Uncalibrated starting thresholds/penalties: calibrate on validation positives and unrelated negatives, never test.",
    "Greedy region extraction and suppression of rejected positive cells are not a globally optimal multi-region solution.",
    "Background correction can suppress broad or repeated genuine matches; raw mode can accept uniformly high background.",
    "One-to-one SW diagonals plus gaps do not model arbitrary duration/width differences; local DTW/common-subsequence alignment is a possible later comparison.",
    "Overlapping windows have coarse spatial resolution; masks union actual window footprints (which can touch across separate region IDs).",
    "Support counts describe distinct positive windows, not a universal minimum word length.",
    "Image-region overlap does not establish character-level alignment accuracy.",
    "No image-to-transcript DTW positional prior, transcripts, character predictions, or alignment GT are used to predict regions.",
]


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_pair(args, config):
    explicit = args.image1 is not None or args.image2 is not None
    if explicit:
        if not args.image1 or not args.image2 or args.manifest or args.split or args.record_indices:
            raise ValueError("Use both explicit images OR --manifest --split --record-indices")
        paths = [Path(args.image1).expanduser().resolve(), Path(args.image2).expanduser().resolve()]
        selection = dict(strategy="explicit_images", split="unverified", record_ids=None)
    else:
        if not args.manifest or not args.split or args.record_indices is None:
            raise ValueError("Saved membership requires --manifest --split train|validation --record-indices I J")
        data = Path(args.manifest).read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        expected = config.get("split_manifest_sha256")
        if expected and expected != digest:
            raise ValueError("Saved split manifest SHA256 differs from checkpoint metadata")
        manifests = json.loads(data)
        if set(manifests) != {"train_eval", "val_eval"}:
            raise ValueError("Expected saved train_eval/val_eval membership; no split regeneration/test fallback")
        from epoch_monitoring import assert_disjoint
        assert_disjoint(manifests)
        key = {"train": "train_eval", "validation": "val_eval"}[args.split]
        rows = manifests[key]
        if any(i < 0 or i >= len(rows) for i in args.record_indices):
            raise ValueError(f"Record indices must lie in [0,{len(rows)}) for {key}")
        chosen = [rows[i] for i in args.record_indices]
        if any(not r.get("line_image_path") for r in chosen):
            raise ValueError("Select independent-line records with line_image_path; paired records need explicit images")
        paths = [(Path(r["_root"]) / r["line_image_path"]).resolve() for r in chosen]
        selection = dict(strategy="saved_monitoring_membership", split=key, in_sample=key == "train_eval",
                         manifest=str(Path(args.manifest).resolve()), manifest_sha256=digest,
                         manifest_identity="verified" if expected else "unavailable_in_checkpoint",
                         record_indices=args.record_indices, record_ids=[r["record_id"] for r in chosen],
                         population=len(rows), eligibility="both records are members of the requested saved split")
    if paths[0] == paths[1]:
        raise ValueError("Choose two distinct source images")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    selection["images"] = [str(p) for p in paths]
    return paths, selection


def preprocessing_contract(contract, preprocessing):
    if preprocessing == "no-xml":
        # Explicit ablation: same checkpoint intensity/resize policy, XML disabled.
        contract = replace(contract, real_bbox_crop=False,
                           config=dict(contract.config, real_bbox_crop=False))
    elif preprocessing != "training":
        raise ValueError("preprocessing must be training or no-xml")
    return contract


def prepare_image(models, path, preprocessing):
    contract = preprocessing_contract(models.contract, preprocessing)
    try:
        image, geometry = contract.prepare_line(path, "real", "training")
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"Checkpoint preprocessing failed for {path}: {exc}. "
                         "External images without native XML require explicit --preprocessing no-xml.") from exc
    geometry["explicit_preprocessing_override"] = preprocessing != "training"
    geometry["requested_preprocessing"] = preprocessing
    return image, geometry


def score_annotations(mask, gt_path, geometry, contract, settings, pair_label):
    """Only invoked after all predictions are fixed and written."""
    pred = np.asarray(mask) > 0
    report = dict(status="unavailable", reason="no_source_size_ground_truth_mask",
                  predicted_coverage=float(pred.mean()),
                  negative_pair_false_positive_coverage=float(pred.mean()) if pair_label == "negative" else None)
    if not gt_path:
        return report
    gt = _load_mask(Path(gt_path))
    if gt.shape != pred.shape:
        raise ValueError(f"GT mask geometry {gt.shape} != source image {pred.shape}: {gt_path}; resizing is forbidden")
    if pair_label == "negative" and gt.any():
        raise ValueError("Negative-pair label conflicts with nonempty GT shared-region mask")
    intervals, truth = _runs(pred), _runs(gt)
    matches = _region_matches(intervals, truth, pred.shape[1], settings.min_windows, 0.5,
                              physical_windows=source_window_intervals(geometry, contract.window_size, contract.stride))
    boundaries = [(abs(r["pred"][0]-r["gt"][0])+abs(r["pred"][1]-r["gt"][1]))/2
                  for r in matches if r["pred"] is not None]
    report.update(status="available", reason=None, gt_path=str(Path(gt_path).resolve()),
        gt_sha256=sha256_file(gt_path), pixel_metrics=_binary_metrics(pred, gt),
        column_region_metrics=score_source_regions(intervals, gt_path, geometry,
            contract.window_size, contract.stride, settings.min_windows, 0.5),
        boundary_error_px=float(np.mean(boundaries)) if boundaries else None,
        gt_regions=len(truth), predicted_regions=len(intervals), region_matches=matches,
        region_metric_rule="IoU >= 0.5 and geometric overlap of >= min_windows consecutive physical window footprints; positive anchor counts reported separately",
        empty_mask_convention="existing metrics: both empty IoU/Dice=1, precision/recall/F1=0")
    if not gt.any():
        report["negative_pair_false_positive_coverage"] = float(pred.mean())
    return report


def _overlay(original, colored):
    rgb = np.asarray(original.convert("RGB")).copy()
    active = np.any(colored > 0, axis=2)
    rgb[active] = np.round(0.6*rgb[active] + 0.4*colored[active]).astype(np.uint8)
    return Image.fromarray(rgb)


def save_figures(output, originals, region_intervals, cosine, rewards, regions, windows, rtl):
    colors = [plt.get_cmap("tab10")(i % 10) for i in range(len(regions))]
    overlays = []
    for side in (0, 1):
        colored = np.zeros((originals[side].height, originals[side].width, 3), dtype=np.uint8)
        for color, intervals in zip(colors, region_intervals[side]):
            active = np.asarray(intervals_mask(intervals, originals[side].size)) > 0
            colored[active] = np.round(np.asarray(color[:3])*255).astype(np.uint8)
        overlay = _overlay(originals[side], colored)
        overlay.save(output / f"line{side+1}_overlay.png")
        overlays.append(overlay)
    fig, axes = plt.subplots(2, 1, figsize=(16, 6), constrained_layout=True)
    for side, ax in enumerate(axes):
        ax.imshow(overlays[side], aspect="auto")
        ax.set(xlim=(-.5, originals[side].width-.5), xlabel="Original source x (pixels)",
               ylabel=f"Line {side+1}", title=f"Line {side+1}: {originals[side].width} x {originals[side].height}; full source extent")
        for k, intervals in enumerate(region_intervals[side]):
            x = sum((min(a for a,b in intervals), max(b for a,b in intervals)))/2
            ax.text(x, 0, f"R{k+1}", color=colors[k], backgroundcolor="white", va="top", ha="center")
    fig.suptitle(f"Accepted shared regions: {len(regions)}" if regions else "No accepted shared region — both masks are empty")
    fig.savefig(output / "overview.png", dpi=150)
    plt.close(fig)
    for matrix, filename, title, bound in (
            (cosine, "cosine_heatmap.png", "Raw window cosine similarity", 1.),
            (rewards, "alignment_score_heatmap.png", "Alignment rewards (not cosines or probabilities)",
             max(1e-6, float(np.max(np.abs(rewards))) if rewards.size else 1.))):
        fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
        display = matrix if matrix.size else np.zeros((1, 1))
        im = ax.imshow(display, origin="upper", interpolation="nearest", cmap="coolwarm", vmin=-bound, vmax=bound, aspect="auto")
        for k, region in enumerate(regions):
            pairs = np.asarray(region["pairs"])
            ax.scatter(pairs[:, 1], pairs[:, 0], marker="s", s=36, facecolors="none", edgecolors=[colors[k]], label=f"R{k+1}")
        for axis, side in ((ax.yaxis, 0), (ax.xaxis, 1)):
            if windows[side]:
                ticks = np.unique(np.linspace(0, len(windows[side])-1, min(10, len(windows[side]))).astype(int))
                axis.set_ticks(ticks)
                axis.set_ticklabels([str(windows[side][i]["logical_index"]) for i in ticks])
        order = "RTL: index 0 = rightmost physical window" if rtl else "LTR model sequence order"
        ax.set(xlabel=f"Line 2 — logical window index ({order})", ylabel=f"Line 1 — logical window index ({order})",
               title=title + ("; no accepted match" if not regions else "; squares = positive accepted matches"))
        if regions:
            ax.legend(loc="upper left", bbox_to_anchor=(1.13, 1))
        fig.colorbar(im, ax=ax, label="Cosine" if filename.startswith("cosine") else "Reward")
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)


def evaluate(args):
    settings = RegionSettings(args.score_mode, args.cosine_threshold, args.contrast_margin,
        args.gap_open, args.gap_extend, args.min_windows,
        0 if args.strict_consecutive else args.max_internal_gap, args.min_region_score, args.max_candidates)
    settings.validate()
    output = Path(args.output_dir) if args.output_dir else Path("Results/Evaluation/SharedRegions") / (
        datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output}")
    weights = Path(args.weights).expanduser().resolve()
    before = weights.stat()
    checkpoint_hash = sha256_file(weights)
    models = load_evaluation_models(weights, args.device, load_text_model=False)
    after = weights.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Checkpoint changed while loading; use a stable saved epoch")
    if models.config.get("architecture_family") != "restoration-positive-dtw-window-encoder":
        raise ValueError("Shared-region representation API requires a restoration window-encoder checkpoint")
    paths, selection = select_pair(args, models.config)
    originals, prepared, geometries = [], [], []
    for path in paths:
        with Image.open(path) as image:
            originals.append(image.copy())
        image, geometry = prepare_image(models, path, args.preprocessing)
        prepared.append(image)
        geometries.append(geometry)
    output.mkdir(parents=True, exist_ok=False)
    for side in (0, 1):
        originals[side].save(output / f"line{side+1}_original.png")
        prepared[side].save(output / f"line{side+1}_model_input.png")
    models.image_model.eval()
    features = point2_pair_features(models, output / "line1_model_input.png", output / "line2_model_input.png", args.representation)
    rtl = bool(models.image_model.use_flip)
    windows = [valid_window_geometry(f, g, models.contract, rtl) for f, g in zip(features, geometries)]
    vectors = []
    for feature, entries in zip(features, windows):
        value = feature.contextual[[w["logical_index"] for w in entries]].detach().float()
        if value.numel() and bool((value.norm(dim=-1) == 0).any()):
            raise ValueError("Valid embedding has zero norm; cosine is undefined")
        vectors.append(torch.nn.functional.normalize(value, dim=-1))
    cosine = (vectors[0] @ vectors[1].T).cpu().numpy()
    result = extract_regions(cosine, *[[w["physical_index"] for w in entries] for entries in windows], settings)
    regions, rewards = result["regions"], result.pop("rewards")
    intervals = [[region_source_intervals(r, side, geometries[side], models.contract) for r in regions] for side in (0, 1)]
    masks = [intervals_mask([item for region in side_regions for item in region], original.size)
             for side_regions, original in zip(intervals, originals)]
    for side, mask in enumerate(masks, 1):
        mask.save(output / f"line{side}_mask.png")
    for name, matrix in (("cosine", cosine), ("alignment_scores", rewards)):
        np.save(output / f"{name}.npy", matrix)
        np.savetxt(output / f"{name}.csv", matrix, delimiter=",", fmt="%.17g")
    columns = ["region", "line1_logical_index", "line2_logical_index", "line1_physical_index", "line2_physical_index",
               "cosine", "alignment_score", "line1_source_x0", "line1_source_x1", "line2_source_x0", "line2_source_x1"]
    with (output / "correspondences.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for k, region in enumerate(regions, 1):
            region["region_id"] = k
            region["source_intervals"] = [intervals[s][k-1] for s in (0, 1)]
            for i, j in region["pairs"]:
                a, b = windows[0][i], windows[1][j]
                writer.writerow(dict(zip(columns, [k, a["logical_index"], b["logical_index"], a["physical_index"], b["physical_index"],
                    float(cosine[i, j]), float(rewards[i, j]), *a["source_interval"], *b["source_interval"]])))
    save_figures(output, originals, intervals, cosine, rewards, regions, windows, rtl)
    summary = dict(checkpoint=dict(path=str(weights), sha256=checkpoint_hash, epoch=models.checkpoint.get("epoch"),
                       architecture_variant=models.config.get("architecture_variant"), load="strict; image model only"),
        preprocessing=preprocessing_contract(models.contract, args.preprocessing).metadata(weights, args.preprocessing),
        checkpoint_preprocessing=models.contract.metadata(weights, "training"), geometry=geometries,
        selection=selection, image_sha256=[sha256_file(p) for p in paths], representation=args.representation,
        feature_dimensions=[list(v.shape) for v in vectors], valid_tokens=[len(w) for w in windows],
        token_count=[int(f.contextual.shape[0]) for f in features], windows=windows,
        model_input_shapes=[[1, models.contract.visual_input_channels, im.height, im.width] for im in prepared],
        eval_mode=True, augmentation=False, precision="float32", use_flip=rtl,
        cosine_statistics={k: float(fn(cosine)) if cosine.size else None for k, fn in
                           (("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std))},
        algorithm="greedy local Smith-Waterman with affine gaps; positive-window support filtering",
        matrix_indexing="pairs/steps index the valid-only matrix; windows maps each row/column to logical and physical indices",
        gap_cost="opening + (length-1)*extension", alignment=result,
        status="accepted_regions" if regions else "no_accepted_match", limitations=LIMITATIONS,
        pair_label_for_scoring_only=args.pair_label, label_source=args.label_source,
        ground_truth_read_after_prediction=True)
    try:
        summary["metrics"] = [score_annotations(mask, gt, geometry, models.contract, settings, args.pair_label)
                              for mask, gt, geometry in zip(masks, (args.gt_mask1, args.gt_mask2), geometries)]
    except (ValueError, OSError) as exc:
        summary["annotation_error"] = str(exc)
        (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
        raise
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    print(json.dumps(dict(output=str(output.resolve()), status=summary["status"], regions=len(regions),
        valid_tokens=summary["valid_tokens"], representation=args.representation, cosine=summary["cosine_statistics"],
        metrics=summary["metrics"]), indent=2))
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image1")
    parser.add_argument("--image2")
    parser.add_argument("--manifest", help="Exact saved monitoring split_manifest.json")
    parser.add_argument("--split", choices=["train", "validation"])
    parser.add_argument("--record-indices", type=int, nargs=2, metavar=("I", "J"), help="Zero-based indices within the requested saved split")
    parser.add_argument("--representation", choices=["fused", "local", "context"], default="fused")
    parser.add_argument("--preprocessing", choices=["training", "no-xml"], default="training")
    parser.add_argument("--score-mode", choices=["background", "raw"], default="background")
    parser.add_argument("--cosine-threshold", type=float, default=.60)
    parser.add_argument("--contrast-margin", type=float, default=.05)
    parser.add_argument("--gap-open", type=float, default=.20)
    parser.add_argument("--gap-extend", type=float, default=.05)
    parser.add_argument("--min-windows", type=int, default=5)
    parser.add_argument("--max-internal-gap", type=int, default=1)
    parser.add_argument("--strict-consecutive", action="store_true", help="Override internal gap allowance to zero")
    parser.add_argument("--min-region-score", type=float, default=0.)
    parser.add_argument("--max-candidates", type=int, default=128)
    parser.add_argument("--gt-mask1")
    parser.add_argument("--gt-mask2")
    parser.add_argument("--pair-label", choices=["unknown", "positive", "negative"], default="unknown", help="Independent pair label for reporting/scoring only")
    parser.add_argument("--label-source", default="user_supplied; not independently verified", help="Provenance of the optional pair label")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", help="Must not exist; default is a new timestamp/UUID directory")
    return parser.parse_args(argv)


if __name__ == "__main__":
    evaluate(parse_args())
