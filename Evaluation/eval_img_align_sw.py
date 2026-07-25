#!/usr/bin/env python3
"""Checkpoint-compatible Smith-Waterman local image alignment.

The evaluator supports both the synthetic ``images/img1_<n>.png`` layout and
``DataSet/ArabicDataset/dataset_manifest.jsonl``. Real ArabicDataset images are
binarized with the same configurable preprocessing used by training, and those
exact binary images are both passed to the model and shown in the result figure.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Evaluation._eval_utils import (
    compute_similarity,
    get_image_features,
    load_evaluation_models,
    patch_range_to_pixels,
    synthetic_pair_paths,
)


_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class ImagePair:
    index: int
    image1: Path
    image2: Path
    pair_id: str = ""
    label_type: str = ""
    text_score: float = 0.0
    manifest_position: int = -1
    split: str = ""


def smith_waterman(similarity: np.ndarray, threshold=0.45, gap_penalty=-0.30):
    n, m = similarity.shape
    score = np.zeros((n + 1, m + 1), dtype=np.float32)
    trace = np.zeros((n + 1, m + 1), dtype=np.uint8)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + float(similarity[i - 1, j - 1]) - threshold
            up = score[i - 1, j] + gap_penalty
            left = score[i, j - 1] + gap_penalty
            values = (0.0, diag, up, left)
            best = int(np.argmax(values))
            score[i, j] = values[best]
            trace[i, j] = best
    i, j = map(int, np.unravel_index(np.argmax(score), score.shape))
    best_score = float(score[i, j])
    path = []
    while i > 0 and j > 0 and score[i, j] > 0:
        code = int(trace[i, j])
        if code == 1:
            path.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif code == 2:
            i -= 1
        elif code == 3:
            j -= 1
        else:
            break
    path.reverse()
    return path, best_score, score


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _image_root(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    images = root / "images"
    return images if images.is_dir() else root


def _candidate_patterns(role: int, requested: str) -> list[str]:
    defaults = [
        f"img{role}_{{index}}.png",
        f"image{role}_{{index}}.png",
        f"line{role}_{{index}}.png",
        f"{{index}}_{role}.png",
    ]
    values = [requested] + defaults
    return list(dict.fromkeys(value for value in values if value))


def _with_extension_fallback(path: Path) -> list[Path]:
    values = [path]
    stem = path.with_suffix("") if path.suffix else path
    for suffix in _IMAGE_EXTENSIONS:
        candidate = stem.with_suffix(suffix)
        if candidate not in values:
            values.append(candidate)
    return values


def _resolve_pattern_image(
    data_dir: str | Path,
    pattern: str,
    index: int,
    role: int,
) -> Path:
    root = Path(data_dir)
    image_root = _image_root(root)
    first_expected = None
    for candidate_pattern in _candidate_patterns(role, pattern):
        relative = Path(candidate_pattern.format(index=int(index)))
        for base in (image_root, root):
            for candidate in _with_extension_fallback(base / relative):
                if first_expected is None:
                    first_expected = candidate
                if candidate.is_file():
                    return candidate
    return first_expected or image_root / pattern.format(index=int(index))


def _real_pair_paths(args, index: int) -> ImagePair:
    return ImagePair(
        index=int(index),
        image1=_resolve_pattern_image(
            args.data_dir,
            args.image1_pattern,
            index,
            role=1,
        ),
        image2=_resolve_pattern_image(
            args.data_dir,
            args.image2_pattern,
            index,
            role=2,
        ),
    )


def _manifest_image_value(record: dict, role: int) -> str:
    aliases = (
        ("image1", "image_1", "line1", "line_1", "source")
        if role == 1
        else ("image2", "image_2", "line2", "line_2", "target")
    )
    for key in aliases:
        value = record.get(key)
        if value not in (None, ""):
            return str(value)
    raise ValueError(f"Manifest record is missing image{role}: {record}")


def _resolve_manifest_image(value: str, manifest: Path, data_dir: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [manifest.parent / path, Path(data_dir) / path, _image_root(data_dir) / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _read_manifest_records(path: str | Path) -> list[dict]:
    manifest = Path(path)
    suffix = manifest.suffix.lower()
    if suffix == ".csv":
        with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        records = payload.get("pairs", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("JSON manifest must be a list or contain a 'pairs' list")
        return records
    raise ValueError("Pair manifest must use .csv, .json, or .jsonl")


def _load_pair_manifest(path: str | Path, data_dir: str | Path) -> list[ImagePair]:
    """Load a flat generic manifest with image1/image2-style columns."""
    manifest = Path(path)
    records = _read_manifest_records(manifest)
    if records and isinstance(records[0].get("A"), dict):
        raise ValueError(
            "Nested ArabicDataset manifests must be loaded through --data-dir "
            "DataSet/ArabicDataset or --arabic-manifest."
        )

    pairs = []
    for position, record in enumerate(records, start=1):
        index = int(record.get("index", record.get("id", position)))
        pairs.append(
            ImagePair(
                index=index,
                image1=_resolve_manifest_image(
                    _manifest_image_value(record, 1), manifest, data_dir
                ),
                image2=_resolve_manifest_image(
                    _manifest_image_value(record, 2), manifest, data_dir
                ),
                pair_id=str(record.get("pair_id", "")),
                label_type=str(record.get("label_type", "")),
                manifest_position=position,
            )
        )
    return sorted(pairs, key=lambda item: item.index)


def _arabic_manifest_path(args) -> Path:
    if args.arabic_manifest:
        return Path(args.arabic_manifest)
    root = Path(args.data_dir)
    if root.suffix.lower() == ".jsonl":
        return root
    return root / os.environ.get("REAL_MANIFEST_NAME", "dataset_manifest.jsonl")


def _parse_real_labels(value: str | None):
    raw = (
        str(value).strip()
        if value is not None
        else os.environ.get("REAL_DATASET_LABELS", "high_match,medium_match").strip()
    )
    if raw.lower() in {"all", "*", "any"}:
        return None
    labels = [label.strip() for label in raw.split(",") if label.strip()]
    if not labels:
        raise ValueError("--real-labels cannot be empty; use high_match,medium_match or all")
    return labels


def _split_lengths(length: int) -> tuple[int, int, int]:
    train_size = int(0.6 * length)
    valid_size = int(0.2 * length)
    return train_size, valid_size, length - train_size - valid_size


def _random_split_pairs(pairs: list[ImagePair], seed: int):
    indices = list(range(len(pairs)))
    random.Random(int(seed)).shuffle(indices)
    train_size, valid_size, _test_size = _split_lengths(len(indices))
    train = [pairs[index] for index in indices[:train_size]]
    valid = [pairs[index] for index in indices[train_size : train_size + valid_size]]
    test = [pairs[index] for index in indices[train_size + valid_size :]]
    return train, valid, test


def _group_split_pairs(pairs: list[ImagePair], seed: int):
    """Mirror training's pair_id-safe 60/20/20 split."""
    groups: dict[str, list[ImagePair]] = {}
    for position, pair in enumerate(pairs):
        group_id = pair.pair_id or f"sample_{position}"
        groups.setdefault(group_id, []).append(pair)
    if len(groups) < 3:
        return _random_split_pairs(pairs, seed)

    group_ids = list(groups)
    random.Random(int(seed)).shuffle(group_ids)
    train_target = int(0.6 * len(pairs))
    valid_target = int(0.2 * len(pairs))
    train: list[ImagePair] = []
    valid: list[ImagePair] = []
    test: list[ImagePair] = []
    for group_id in group_ids:
        members = groups[group_id]
        if len(train) < train_target:
            train.extend(members)
        elif len(valid) < valid_target:
            valid.extend(members)
        else:
            test.extend(members)
    if not train or not valid or not test:
        return _random_split_pairs(pairs, seed)
    return train, valid, test


def _load_arabic_dataset_pairs(args) -> list[ImagePair]:
    """Load the native nested A/B ArabicDataset manifest."""
    from RealDataSet import ArabicManifestLinePairDataset

    manifest = _arabic_manifest_path(args)
    labels = _parse_real_labels(args.real_labels)
    dataset = ArabicManifestLinePairDataset(
        manifest_path=manifest,
        transform=None,
        text_key=args.real_text_key,
        allowed_labels=labels,
        max_samples=None,
        paired=True,
        min_text_score=float(args.real_min_text_score),
        validate_paths=bool(args.real_validate_paths),
    )

    pairs = []
    for manifest_position, sample in enumerate(dataset.samples, start=1):
        side_a = sample["A"]
        side_b = sample["B"]
        scores = sample.get("scores") or {}
        pairs.append(
            ImagePair(
                index=manifest_position,
                image1=dataset._resolve(side_a["line_image_path"]),
                image2=dataset._resolve(side_b["line_image_path"]),
                pair_id=str(sample.get("pair_id", manifest_position)),
                label_type=str(sample.get("label_type", "")),
                text_score=float(scores.get("text_score", 0.0)),
                manifest_position=manifest_position,
            )
        )

    train, valid, test = _group_split_pairs(pairs, args.split_seed)
    selected = {
        "all": pairs,
        "train": train,
        "valid": valid,
        "test": test,
    }[args.real_split]
    return [
        replace(pair, index=position, split=args.real_split)
        for position, pair in enumerate(selected, start=1)
    ]


def _pair_for_index(args, index: int, manifest_pairs: list[ImagePair]) -> ImagePair:
    if manifest_pairs:
        position = int(index) - 1
        if position < 0 or position >= len(manifest_pairs):
            raise IndexError(
                f"Requested index {index}, but selected manifest split contains "
                f"{len(manifest_pairs)} pairs"
            )
        return manifest_pairs[position]
    if args.dataset_type == "real":
        return _real_pair_paths(args, index)
    pair = synthetic_pair_paths(args.data_dir, index)
    return ImagePair(index=int(index), image1=pair.image1, image2=pair.image2)


def _batch_pairs(args, manifest_pairs: list[ImagePair]) -> list[ImagePair]:
    if manifest_pairs:
        start = max(0, int(args.start_index) - 1)
        return manifest_pairs[start : start + int(args.n_samples)]
    return [
        _pair_for_index(args, index, manifest_pairs=[])
        for index in range(args.start_index, args.start_index + args.n_samples)
    ]


def _real_binarizer():
    """Use the exact configurable binarizer used by the real training loader."""
    from DataLoader import ResizeAndBinarize

    return ResizeAndBinarize(
        size=(128, 1024),
        enabled=True,
        method=os.environ.get("REAL_BINARIZE_METHOD", "otsu").lower(),
        fixed_threshold=int(os.environ.get("REAL_BINARIZE_THRESHOLD", "180")),
        auto_invert=_env_flag("REAL_BINARIZE_AUTO_INVERT", True),
        autocontrast=_env_flag("REAL_BINARIZE_AUTOCONTRAST", True),
    )


def _display_image(path: str | Path, dataset_type: str) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        if str(dataset_type).lower() == "real":
            image = _real_binarizer()(image)
        return np.asarray(image.convert("RGB"))


def _save_binarized_inputs(
    array1: np.ndarray,
    array2: np.ndarray,
    output: str | Path,
    index: int,
) -> tuple[Path, Path]:
    directory = Path(output).parent / "binarized"
    directory.mkdir(parents=True, exist_ok=True)
    line1 = directory / f"pair_{int(index)}_line1.png"
    line2 = directory / f"pair_{int(index)}_line2.png"
    Image.fromarray(array1).save(line1)
    Image.fromarray(array2).save(line2)
    return line1, line2


def _save_visualization(
    arr1,
    arr2,
    features1,
    features2,
    path,
    score,
    output,
    use_flip,
    binarized,
    pair: ImagePair,
):
    fig, axes = plt.subplots(2, 1, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(arr1, aspect="auto")
    axes[1].imshow(arr2, aspect="auto")
    suffix = " (binarized)" if binarized else ""
    axes[0].set_ylabel(f"line A{suffix}", rotation=0, labelpad=50, va="center")
    axes[1].set_ylabel(f"line B{suffix}", rotation=0, labelpad=50, va="center")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    if path:
        i_values, j_values = zip(*path)
        x01, x11 = patch_range_to_pixels(
            min(i_values),
            max(i_values) + 1,
            len(features1.contextual),
            arr1.shape[1],
            use_flip,
        )
        x02, x12 = patch_range_to_pixels(
            min(j_values),
            max(j_values) + 1,
            len(features2.contextual),
            arr2.shape[1],
            use_flip,
        )
        for ax, array, x0, x1 in (
            (axes[0], arr1, x01, x11),
            (axes[1], arr2, x02, x12),
        ):
            ax.add_patch(
                Rectangle(
                    (x0, 1),
                    x1 - x0,
                    array.shape[0] - 2,
                    facecolor="red",
                    edgecolor="red",
                    alpha=0.28,
                    linewidth=2,
                )
            )

    input_label = "binarized real input" if binarized else "synthetic input"
    metadata = ""
    if pair.pair_id:
        metadata = f" | pair_id={pair.pair_id} | label={pair.label_type}"
    fig.suptitle(
        f"Smith-Waterman local image alignment | score={score:.4f} | "
        f"{input_label}{metadata}"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _evaluate_sample(
    models,
    pair: ImagePair,
    dataset_type,
    feature,
    threshold,
    gap,
    output,
    save_binarized_images,
):
    image1 = Path(pair.image1)
    image2 = Path(pair.image2)
    if not image1.is_file() or not image2.is_file():
        missing = [str(path) for path in (image1, image2) if not path.is_file()]
        raise FileNotFoundError("Missing image pair: " + ", ".join(missing))

    binarized = str(dataset_type).lower() == "real"
    binary1 = binary2 = ""
    temporary_directory = None
    try:
        if binarized:
            # Produce the exact black/white 128x1024 inputs first. The same files
            # are displayed and passed through the synthetic transform, which only
            # resizes/normalizes and therefore cannot re-threshold them differently.
            arr1 = _display_image(image1, "real")
            arr2 = _display_image(image2, "real")
            if save_binarized_images:
                model_image1, model_image2 = _save_binarized_inputs(
                    arr1, arr2, output, pair.index
                )
                binary1, binary2 = str(model_image1), str(model_image2)
            else:
                temporary_directory = tempfile.TemporaryDirectory(prefix="sw_real_binary_")
                temp_root = Path(temporary_directory.name)
                model_image1 = temp_root / "line1.png"
                model_image2 = temp_root / "line2.png"
                Image.fromarray(arr1).save(model_image1)
                Image.fromarray(arr2).save(model_image2)
            feature_dataset_type = "synthetic"
        else:
            with Image.open(image1) as opened:
                arr1 = np.asarray(opened.convert("RGB"))
            with Image.open(image2) as opened:
                arr2 = np.asarray(opened.convert("RGB"))
            model_image1, model_image2 = image1, image2
            feature_dataset_type = "synthetic"

        features1 = get_image_features(models, model_image1, feature_dataset_type)
        features2 = get_image_features(models, model_image2, feature_dataset_type)
        similarity = compute_similarity(
            features1.select(feature),
            features2.select(feature),
        ).cpu().numpy()
        path, score, _score_matrix = smith_waterman(similarity, threshold, gap)

        _save_visualization(
            arr1,
            arr2,
            features1,
            features2,
            path,
            score,
            output,
            models.image_model.use_flip,
            binarized=binarized,
            pair=pair,
        )
    finally:
        if temporary_directory is not None:
            temporary_directory.cleanup()

    path_similarities = [float(similarity[i, j]) for i, j in path]
    row = {
        "index": int(pair.index),
        "manifest_position": int(pair.manifest_position),
        "pair_id": pair.pair_id,
        "label_type": pair.label_type,
        "text_score": float(pair.text_score),
        "split": pair.split,
        "status": "ok",
        "score": float(score),
        "path_steps": len(path),
        "mean_path_cosine": float(np.mean(path_similarities)) if path_similarities else 0.0,
        "line1_windows": int(similarity.shape[0]),
        "line2_windows": int(similarity.shape[1]),
        "line1_path_start": int(path[0][0]) if path else -1,
        "line1_path_end": int(path[-1][0]) if path else -1,
        "line2_path_start": int(path[0][1]) if path else -1,
        "line2_path_end": int(path[-1][1]) if path else -1,
        "feature": str(feature),
        "threshold": float(threshold),
        "gap": float(gap),
        "dataset_type": str(dataset_type),
        "binarized": bool(binarized),
        "binarization": (
            os.environ.get("REAL_BINARIZE_METHOD", "otsu").lower()
            if binarized
            else "none"
        ),
        "flipped": bool(models.image_model.use_flip),
        "image1": str(image1),
        "image2": str(image2),
        "binarized_image1": binary1,
        "binarized_image2": binary2,
        "output": str(output),
        "error": "",
    }
    print(
        f"[{pair.index}] pair_id={pair.pair_id or '-'} label={pair.label_type or '-'} "
        f"score={score:.6f} path_steps={len(path)} "
        f"mean_cosine={row['mean_path_cosine']:.4f} "
        f"binarized={row['binarized']} saved={output}",
        flush=True,
    )
    return row


def _aggregate(rows):
    successful = [row for row in rows if row.get("status") == "ok"]
    failed = [row for row in rows if row.get("status") != "ok"]
    scores = [float(row["score"]) for row in successful]
    path_steps = [float(row["path_steps"]) for row in successful]
    path_cosines = [float(row["mean_path_cosine"]) for row in successful]
    return {
        "samples": len(rows),
        "successful": len(successful),
        "failed": len(failed),
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "std_score": float(np.std(scores)) if scores else 0.0,
        "mean_path_steps": float(np.mean(path_steps)) if path_steps else 0.0,
        "mean_path_cosine": float(np.mean(path_cosines)) if path_cosines else 0.0,
        "binarized_samples": sum(bool(row.get("binarized", False)) for row in successful),
        "failed_indices": [int(row["index"]) for row in failed],
    }


def _batch_fieldnames():
    return [
        "index",
        "manifest_position",
        "pair_id",
        "label_type",
        "text_score",
        "split",
        "status",
        "score",
        "path_steps",
        "mean_path_cosine",
        "line1_windows",
        "line2_windows",
        "line1_path_start",
        "line1_path_end",
        "line2_path_start",
        "line2_path_end",
        "feature",
        "threshold",
        "gap",
        "dataset_type",
        "binarized",
        "binarization",
        "flipped",
        "image1",
        "image2",
        "binarized_image1",
        "binarized_image2",
        "output",
        "error",
    ]


def _error_row(args, pair: ImagePair, output: Path, models, exc: Exception) -> dict:
    row = {key: "" for key in _batch_fieldnames()}
    row.update(
        {
            "index": int(pair.index),
            "manifest_position": int(pair.manifest_position),
            "pair_id": pair.pair_id,
            "label_type": pair.label_type,
            "text_score": float(pair.text_score),
            "split": pair.split,
            "status": "error",
            "score": 0.0,
            "path_steps": 0,
            "mean_path_cosine": 0.0,
            "line1_windows": 0,
            "line2_windows": 0,
            "line1_path_start": -1,
            "line1_path_end": -1,
            "line2_path_start": -1,
            "line2_path_end": -1,
            "feature": str(args.feature),
            "threshold": float(args.threshold),
            "gap": float(args.gap),
            "dataset_type": str(args.dataset_type),
            "binarized": args.dataset_type == "real",
            "binarization": (
                os.environ.get("REAL_BINARIZE_METHOD", "otsu").lower()
                if args.dataset_type == "real"
                else "none"
            ),
            "flipped": bool(models.image_model.use_flip),
            "image1": str(pair.image1),
            "image2": str(pair.image2),
            "output": str(output),
            "error": f"{type(exc).__name__}: {exc}",
        }
    )
    return row


def _write_batch_outputs(rows, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_batch_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "summary.json").write_text(
        json.dumps(_aggregate(rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="DataSet/Synthetic_Arabic")
    parser.add_argument("--index", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--image1")
    parser.add_argument("--image2")
    parser.add_argument(
        "--pair-manifest",
        help="Optional flat CSV/JSON/JSONL with image1,image2 for generic datasets.",
    )
    parser.add_argument(
        "--arabic-manifest",
        help="Optional native ArabicDataset dataset_manifest.jsonl path.",
    )
    parser.add_argument(
        "--real-labels",
        default=None,
        help="ArabicDataset labels to keep; default is REAL_DATASET_LABELS or high_match,medium_match.",
    )
    parser.add_argument(
        "--real-min-text-score",
        type=float,
        default=float(os.environ.get("REAL_MIN_TEXT_SCORE", "0.0")),
    )
    parser.add_argument(
        "--real-text-key",
        default=os.environ.get("REAL_TEXT_KEY", "text_original_path"),
    )
    parser.add_argument(
        "--real-split",
        choices=("all", "train", "valid", "test"),
        default="test",
        help="ArabicDataset pair_id-safe split to evaluate; default: test.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=int(os.environ.get("DATASET_SPLIT_SEED", "42")),
    )
    parser.add_argument("--real-validate-paths", action="store_true")
    parser.add_argument(
        "--image1-pattern",
        default="img1_{index}.png",
        help="Fallback real-data image-1 pattern when no ArabicDataset manifest exists.",
    )
    parser.add_argument(
        "--image2-pattern",
        default="img2_{index}.png",
        help="Fallback real-data image-2 pattern when no ArabicDataset manifest exists.",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dataset-type",
        choices=("synthetic", "real"),
        default="synthetic",
    )
    parser.add_argument(
        "--feature",
        choices=("contextual", "local", "grouped"),
        default="contextual",
    )
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--gap", type=float, default=-0.30)
    parser.add_argument(
        "--output",
        default="Results/Evaluation/SW/smith_waterman.png",
    )
    parser.add_argument(
        "--output-dir",
        default="Results/Evaluation/SW/windows",
    )
    parser.add_argument(
        "--no-save-binarized-images",
        action="store_true",
        help="For real data, do not retain standalone binary line images.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch and (args.image1 or args.image2):
        raise SystemExit("--image1/--image2 are single-sample options and cannot be used with --batch")
    if bool(args.image1) != bool(args.image2):
        raise SystemExit("Provide both --image1 and --image2, or neither")
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be greater than zero")

    manifest_pairs: list[ImagePair] = []
    arabic_manifest = _arabic_manifest_path(args)
    if args.dataset_type == "real" and not args.pair_manifest and arabic_manifest.is_file():
        manifest_pairs = _load_arabic_dataset_pairs(args)
        print(
            "Loaded ArabicDataset manifest for SW evaluation: "
            f"manifest={arabic_manifest} split={args.real_split} "
            f"samples={len(manifest_pairs)} labels={args.real_labels or os.environ.get('REAL_DATASET_LABELS', 'high_match,medium_match')} "
            f"min_text_score={args.real_min_text_score}",
            flush=True,
        )
    elif args.pair_manifest:
        manifest_pairs = _load_pair_manifest(args.pair_manifest, args.data_dir)

    models = load_evaluation_models(args.weights, args.device, load_text_model=False)
    save_binarized_images = not args.no_save_binarized_images

    if not args.batch:
        if args.image1 and args.image2:
            pair = ImagePair(args.index, Path(args.image1), Path(args.image2))
        else:
            pair = _pair_for_index(args, args.index, manifest_pairs)
        _evaluate_sample(
            models,
            pair,
            args.dataset_type,
            args.feature,
            args.threshold,
            args.gap,
            Path(args.output),
            save_binarized_images,
        )
        return

    output_dir = Path(args.output_dir)
    rows = []
    selected_pairs = _batch_pairs(args, manifest_pairs)
    if not selected_pairs:
        raise SystemExit("No image pairs were selected for evaluation")
    for pair in selected_pairs:
        output = output_dir / f"pair_{pair.index}.png"
        try:
            row = _evaluate_sample(
                models,
                pair,
                args.dataset_type,
                args.feature,
                args.threshold,
                args.gap,
                output,
                save_binarized_images,
            )
        except Exception as exc:
            row = _error_row(args, pair, output, models, exc)
            print(f"[{pair.index}] failed: {row['error']}", file=sys.stderr, flush=True)
        rows.append(row)

    _write_batch_outputs(rows, output_dir)
    print(json.dumps(_aggregate(rows), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
