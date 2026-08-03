#!/usr/bin/env python3
"""Quantitative real-line evaluation using Excel subword bounding boxes."""
from __future__ import annotations

import argparse
from collections import OrderedDict
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation._eval_utils import load_image_model
from Evaluation.eval_img_align_sw import get_sim, smith_waterman, _patch_to_pixels
from Evaluation.real_subword_box_metrics import (
    aggregate,
    binary_metrics,
    lcs_pairs,
    line_metrics,
    load_annotations,
    normalize_text,
)
from Parameters import device


def _resolve(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [root / path, Path.cwd() / path, root.parent / path]
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve manifest path: {value}")


def _load_manifest(path: Path, labels: set[str]):
    root = path.parent
    pairs = []
    with path.open("r", encoding="utf-8") as handle:
        for position, raw in enumerate(handle):
            if not raw.strip():
                continue
            sample = json.loads(raw)
            label = str(sample.get("label_type", ""))
            if labels and label not in labels:
                continue
            scores = sample.get("scores") or {}
            pairs.append({
                "manifest_position": position,
                "pair_id": str(sample.get("pair_id", position)),
                "label_type": label,
                "text_score": float(scores.get("text_score", 0.0)),
                "image1": _resolve(root, sample["A"]["line_image_path"]),
                "image2": _resolve(root, sample["B"]["line_image_path"]),
            })
    return pairs


def _group_split(pairs, seed: int):
    groups = OrderedDict()
    for pair in pairs:
        groups.setdefault(pair["pair_id"], []).append(pair)
    items = list(groups.items())
    random.Random(int(seed)).shuffle(items)
    targets = {"train": 0.60 * len(pairs), "valid": 0.20 * len(pairs), "test": 0.20 * len(pairs)}
    assigned = {"train": [], "valid": [], "test": []}
    for _group, members in sorted(items, key=lambda item: len(item[1]), reverse=True):
        split = max(targets, key=lambda name: targets[name] - len(assigned[name]))
        assigned[split].extend(members)
    return assigned


def _interval(path, axis: int, windows: int, width: int):
    if not path:
        return None
    values = [int(point[axis]) for point in path]
    return tuple(sorted(_patch_to_pixels(min(values), max(values), int(windows), int(width))))


def _pair_metrics(pair, model, threshold, gap, annotation_root):
    sim, windows1, windows2 = get_sim(model, str(pair["image1"]), str(pair["image2"]))
    path, score, _matrix = smith_waterman(sim, threshold=float(threshold), gap_penalty=float(gap))
    with Image.open(pair["image1"]) as opened:
        width1 = opened.width
    with Image.open(pair["image2"]) as opened:
        width2 = opened.width
    interval1 = _interval(path, 0, windows1, width1)
    interval2 = _interval(path, 1, windows2, width2)

    ann1 = load_annotations(pair["image1"], annotation_root)
    ann2 = load_annotations(pair["image2"], annotation_root)
    text1 = [normalize_text(box.text) for box in ann1.boxes]
    text2 = [normalize_text(box.text) for box in ann2.boxes]
    matches = lcs_pairs(text1, text2) if ann1.boxes and ann2.boxes else []
    gt1 = {left for left, _ in matches}
    gt2 = {right for _, right in matches}
    line1 = line_metrics("line1", ann1, gt1, interval1)
    line2 = line_metrics("line2", ann2, gt2, interval2)

    tp = line1["line1_box_tp"] + line2["line2_box_tp"]
    fp = line1["line1_box_fp"] + line2["line2_box_fp"]
    fn = line1["line1_box_fn"] + line2["line2_box_fn"]
    tn = line1["line1_box_tn"] + line2["line2_box_tn"]
    metrics = binary_metrics(tp, fp, fn, tn)
    ious = [value for value in (line1["line1_box_interval_iou"], line2["line2_box_interval_iou"]) if value is not None]
    status = "ok"
    if not ann1.boxes or not ann2.boxes:
        status = "missing_annotations"
    elif ann1.status != "ok" or ann2.status != "ok":
        status = "annotation_warning"
    elif not matches:
        status = "no_shared_subword_boxes"

    return {
        "manifest_position": pair["manifest_position"],
        "pair_id": pair["pair_id"],
        "label_type": pair["label_type"],
        "text_score": pair["text_score"],
        "status": "ok",
        "score": float(score),
        "path_steps": len(path),
        "image1": str(pair["image1"]),
        "image2": str(pair["image2"]),
        "real_box_evaluated": bool(ann1.boxes and ann2.boxes),
        "real_box_status": status,
        "shared_subword_matches": len(matches),
        "pair_box_tp": tp,
        "pair_box_fp": fp,
        "pair_box_fn": fn,
        "pair_box_tn": tn,
        "pair_box_precision": metrics["precision"],
        "pair_box_recall": metrics["recall"],
        "pair_box_f1": metrics["f1"],
        "pair_box_specificity": metrics["specificity"],
        "pair_box_accuracy": metrics["accuracy"],
        "mean_box_interval_iou": float(np.mean(ious)) if ious else None,
        **line1,
        **line2,
        "error": "",
    }


def _write(rows, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(json.dumps(aggregate(rows), ensure_ascii=False, indent=2), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--arabic-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--labels", default="high_match,medium_match")
    parser.add_argument("--split", choices=("all", "train", "valid", "test"), default="test")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--gap", type=float, default=-0.30)
    parser.add_argument("--annotation-root", default="")
    parser.add_argument("--require-annotations", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    labels = {item.strip() for item in args.labels.split(",") if item.strip()}
    pairs = _load_manifest(Path(args.arabic_manifest).resolve(), labels)
    if args.split != "all":
        pairs = _group_split(pairs, args.split_seed)[args.split]
    start = max(0, args.start_index - 1)
    pairs = pairs[start:start + args.n_samples]
    if not pairs:
        raise SystemExit("No real pairs selected")

    model = load_image_model(args.weights, device)
    rows = []
    for order, pair in enumerate(pairs, start=1):
        try:
            row = _pair_metrics(pair, model, args.threshold, args.gap, args.annotation_root)
            if args.require_annotations and not row["real_box_evaluated"]:
                raise FileNotFoundError("Required Excel boxes were not found for both line images")
            print(
                f"[{order}/{len(pairs)}] pair={row['pair_id']} status={row['real_box_status']} "
                f"precision={row['pair_box_precision']:.4f} recall={row['pair_box_recall']:.4f} "
                f"f1={row['pair_box_f1']:.4f} iou={row['mean_box_interval_iou']}",
                flush=True,
            )
        except Exception as exc:
            row = {
                "manifest_position": pair["manifest_position"],
                "pair_id": pair["pair_id"],
                "label_type": pair["label_type"],
                "text_score": pair["text_score"],
                "status": "error",
                "real_box_evaluated": False,
                "real_box_status": "error",
                "image1": str(pair["image1"]),
                "image2": str(pair["image2"]),
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"[{order}/{len(pairs)}] failed: {row['error']}", file=sys.stderr, flush=True)
        rows.append(row)
    _write(rows, Path(args.output_dir))
    print(json.dumps(aggregate(rows), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
