#!/usr/bin/env python3
"""Checkpoint-compatible Smith-Waterman local image alignment."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import sys

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
    ResizeAndBinarize,
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


def _load_pair_manifest(path: str | Path, data_dir: str | Path) -> list[ImagePair]:
    manifest = Path(path)
    suffix = manifest.suffix.lower()
    if suffix == ".csv":
        with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
            records = list(csv.DictReader(handle))
    elif suffix == ".jsonl":
        records = [
            json.loads(line)
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    elif suffix == ".json":
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        records = payload.get("pairs", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            raise ValueError("JSON manifest must be a list or contain a 'pairs' list")
    else:
        raise ValueError("Pair manifest must use .csv, .json, or .jsonl")

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
            )
        )
    return sorted(pairs, key=lambda item: item.index)


def _pair_for_index(args, index: int, manifest_pairs: list[ImagePair]) -> ImagePair:
    if manifest_pairs:
        by_index = {pair.index: pair for pair in manifest_pairs}
        if int(index) not in by_index:
            raise KeyError(f"Index {index} is not present in the pair manifest")
        return by_index[int(index)]
    if args.dataset_type == "real":
        return _real_pair_paths(args, index)
    pair = synthetic_pair_paths(args.data_dir, index)
    return ImagePair(index=int(index), image1=pair.image1, image2=pair.image2)


def _batch_pairs(args, manifest_pairs: list[ImagePair]) -> list[ImagePair]:
    if manifest_pairs:
        selected = [pair for pair in manifest_pairs if pair.index >= int(args.start_index)]
        return selected[: int(args.n_samples)]
    return [
        _pair_for_index(args, index, manifest_pairs=[])
        for index in range(args.start_index, args.start_index + args.n_samples)
    ]


def _display_image(path: str | Path, dataset_type: str) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        if str(dataset_type).lower() == "real":
            image = ResizeAndBinarize((128, 1024), enabled=True)(image)
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
):
    fig, axes = plt.subplots(2, 1, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(arr1, aspect="auto")
    axes[1].imshow(arr2, aspect="auto")
    suffix = " (binarized)" if binarized else ""
    axes[0].set_ylabel(f"line 1{suffix}", rotation=0, labelpad=50, va="center")
    axes[1].set_ylabel(f"line 2{suffix}", rotation=0, labelpad=50, va="center")
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
    fig.suptitle(
        f"Smith-Waterman local image alignment | score={score:.4f} | {input_label}"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _evaluate_sample(
    models,
    image1,
    image2,
    index,
    dataset_type,
    feature,
    threshold,
    gap,
    output,
    save_binarized_images,
):
    image1 = Path(image1)
    image2 = Path(image2)
    if not image1.is_file() or not image2.is_file():
        missing = [str(path) for path in (image1, image2) if not path.is_file()]
        raise FileNotFoundError("Missing image pair: " + ", ".join(missing))

    # get_image_features applies ResizeAndBinarize with Otsu thresholding whenever
    # dataset_type is real, so Smith-Waterman operates on binarized real inputs.
    features1 = get_image_features(models, image1, dataset_type)
    features2 = get_image_features(models, image2, dataset_type)
    similarity = compute_similarity(
        features1.select(feature),
        features2.select(feature),
    ).cpu().numpy()
    path, score, _score_matrix = smith_waterman(similarity, threshold, gap)

    arr1 = _display_image(image1, dataset_type)
    arr2 = _display_image(image2, dataset_type)
    binarized = str(dataset_type).lower() == "real"
    binary1 = binary2 = ""
    if binarized and save_binarized_images:
        saved1, saved2 = _save_binarized_inputs(arr1, arr2, output, index)
        binary1, binary2 = str(saved1), str(saved2)

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
    )

    path_similarities = [float(similarity[i, j]) for i, j in path]
    row = {
        "index": int(index),
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
        "binarization": "otsu" if binarized else "none",
        "flipped": bool(models.image_model.use_flip),
        "image1": str(image1),
        "image2": str(image2),
        "binarized_image1": binary1,
        "binarized_image2": binary2,
        "output": str(output),
        "error": "",
    }
    print(
        f"[{index}] score={score:.6f} path_steps={len(path)} "
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
            "binarization": "otsu" if args.dataset_type == "real" else "none",
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
        help="Optional CSV/JSON/JSONL with index,image1,image2 for arbitrary real filenames.",
    )
    parser.add_argument(
        "--image1-pattern",
        default="img1_{index}.png",
        help="Real-data image-1 filename pattern relative to data-dir or data-dir/images.",
    )
    parser.add_argument(
        "--image2-pattern",
        default="img2_{index}.png",
        help="Real-data image-2 filename pattern relative to data-dir or data-dir/images.",
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
        help="For real data, do not save the two standalone binarized line images.",
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

    manifest_pairs = (
        _load_pair_manifest(args.pair_manifest, args.data_dir)
        if args.pair_manifest
        else []
    )
    models = load_evaluation_models(args.weights, args.device, load_text_model=False)
    save_binarized_images = not args.no_save_binarized_images

    if not args.batch:
        if args.image1 and args.image2:
            pair = ImagePair(args.index, Path(args.image1), Path(args.image2))
        else:
            pair = _pair_for_index(args, args.index, manifest_pairs)
        _evaluate_sample(
            models,
            pair.image1,
            pair.image2,
            pair.index,
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
    for pair in _batch_pairs(args, manifest_pairs):
        output = output_dir / f"pair_{pair.index}.png"
        try:
            row = _evaluate_sample(
                models,
                pair.image1,
                pair.image2,
                pair.index,
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
