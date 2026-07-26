"""Dataset discovery and binarization helpers for SW evaluation."""
from __future__ import annotations

import csv
from dataclasses import replace
import json
import os
from pathlib import Path
import random

import numpy as np
from PIL import Image

from Evaluation.sw_core import ImagePair

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def image_root(data_dir: str | Path) -> Path:
    root = Path(data_dir)
    images = root / "images"
    return images if images.is_dir() else root


def candidate_patterns(role: int, requested: str) -> list[str]:
    defaults = [
        f"img{role}_{{index}}.png",
        f"image{role}_{{index}}.png",
        f"line{role}_{{index}}.png",
        f"{{index}}_{role}.png",
    ]
    return list(dict.fromkeys(value for value in [requested, *defaults] if value))


def with_extension_fallback(path: Path) -> list[Path]:
    values = [path]
    stem = path.with_suffix("") if path.suffix else path
    for suffix in _IMAGE_EXTENSIONS:
        candidate = stem.with_suffix(suffix)
        if candidate not in values:
            values.append(candidate)
    return values


def resolve_pattern_image(
    data_dir: str | Path, pattern: str, index: int, role: int
) -> Path:
    root = Path(data_dir)
    images = image_root(root)
    first_expected = None
    for candidate_pattern in candidate_patterns(role, pattern):
        relative = Path(candidate_pattern.format(index=int(index)))
        for base in (images, root):
            for candidate in with_extension_fallback(base / relative):
                if first_expected is None:
                    first_expected = candidate
                if candidate.is_file():
                    return candidate
    return first_expected or images / pattern.format(index=int(index))


def real_pair_paths(args, index: int) -> ImagePair:
    return ImagePair(
        index=int(index),
        image1=resolve_pattern_image(
            args.data_dir, args.image1_pattern, index, role=1
        ),
        image2=resolve_pattern_image(
            args.data_dir, args.image2_pattern, index, role=2
        ),
    )


def manifest_image_value(record: dict, role: int) -> str:
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


def resolve_manifest_image(value: str, manifest: Path, data_dir: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    candidates = [manifest.parent / path, Path(data_dir) / path, image_root(data_dir) / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def read_manifest_records(path: str | Path) -> list[dict]:
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


def load_pair_manifest(path: str | Path, data_dir: str | Path) -> list[ImagePair]:
    manifest = Path(path)
    records = read_manifest_records(manifest)
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
                image1=resolve_manifest_image(
                    manifest_image_value(record, 1), manifest, data_dir
                ),
                image2=resolve_manifest_image(
                    manifest_image_value(record, 2), manifest, data_dir
                ),
                pair_id=str(record.get("pair_id", "")),
                label_type=str(record.get("label_type", "")),
                manifest_position=position,
            )
        )
    return sorted(pairs, key=lambda item: item.index)


def arabic_manifest_path(args) -> Path:
    if args.arabic_manifest:
        return Path(args.arabic_manifest)
    root = Path(args.data_dir)
    if root.suffix.lower() == ".jsonl":
        return root
    return root / os.environ.get("REAL_MANIFEST_NAME", "dataset_manifest.jsonl")


def parse_real_labels(value: str | None):
    raw = (
        str(value).strip()
        if value is not None
        else os.environ.get("REAL_DATASET_LABELS", "high_match,medium_match").strip()
    )
    if raw.lower() in {"all", "*", "any"}:
        return None
    labels = [label.strip() for label in raw.split(",") if label.strip()]
    if not labels:
        raise ValueError(
            "--real-labels cannot be empty; use high_match,medium_match or all"
        )
    return labels


def split_lengths(length: int) -> tuple[int, int, int]:
    train_size = int(0.6 * length)
    valid_size = int(0.2 * length)
    return train_size, valid_size, length - train_size - valid_size


def random_split_pairs(pairs: list[ImagePair], seed: int):
    indices = list(range(len(pairs)))
    random.Random(int(seed)).shuffle(indices)
    train_size, valid_size, _ = split_lengths(len(indices))
    train = [pairs[index] for index in indices[:train_size]]
    valid = [pairs[index] for index in indices[train_size : train_size + valid_size]]
    test = [pairs[index] for index in indices[train_size + valid_size :]]
    return train, valid, test


def group_split_pairs(pairs: list[ImagePair], seed: int):
    """Mirror training's pair_id-safe 60/20/20 split."""
    groups: dict[str, list[ImagePair]] = {}
    for position, pair in enumerate(pairs):
        groups.setdefault(pair.pair_id or f"sample_{position}", []).append(pair)
    if len(groups) < 3:
        return random_split_pairs(pairs, seed)

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
        return random_split_pairs(pairs, seed)
    return train, valid, test


def load_arabic_dataset_pairs(args) -> list[ImagePair]:
    from RealDataSet import ArabicManifestLinePairDataset

    manifest = arabic_manifest_path(args)
    dataset = ArabicManifestLinePairDataset(
        manifest_path=manifest,
        transform=None,
        text_key=args.real_text_key,
        allowed_labels=parse_real_labels(args.real_labels),
        max_samples=None,
        paired=True,
        min_text_score=float(args.real_min_text_score),
        validate_paths=bool(args.real_validate_paths),
    )
    pairs = []
    for manifest_position, sample in enumerate(dataset.samples, start=1):
        side_a, side_b = sample["A"], sample["B"]
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

    train, valid, test = group_split_pairs(pairs, args.split_seed)
    selected = {"all": pairs, "train": train, "valid": valid, "test": test}[
        args.real_split
    ]
    return [
        replace(pair, index=position, split=args.real_split)
        for position, pair in enumerate(selected, start=1)
    ]


def pair_for_index(args, index: int, manifest_pairs: list[ImagePair]) -> ImagePair:
    if manifest_pairs:
        position = int(index) - 1
        if position < 0 or position >= len(manifest_pairs):
            raise IndexError(
                f"Requested index {index}, but selected manifest split contains "
                f"{len(manifest_pairs)} pairs"
            )
        return manifest_pairs[position]
    if args.dataset_type == "real":
        return real_pair_paths(args, index)
    from Evaluation._eval_utils import synthetic_pair_paths

    pair = synthetic_pair_paths(args.data_dir, index)
    return ImagePair(index=int(index), image1=pair.image1, image2=pair.image2)


def batch_pairs(args, manifest_pairs: list[ImagePair]) -> list[ImagePair]:
    if manifest_pairs:
        start = max(0, int(args.start_index) - 1)
        return manifest_pairs[start : start + int(args.n_samples)]
    return [
        pair_for_index(args, index, [])
        for index in range(args.start_index, args.start_index + args.n_samples)
    ]


def real_binarizer():
    from DataLoader import ResizeAndBinarize

    return ResizeAndBinarize(
        size=(128, 1024),
        enabled=True,
        method=os.environ.get("REAL_BINARIZE_METHOD", "otsu").lower(),
        fixed_threshold=int(os.environ.get("REAL_BINARIZE_THRESHOLD", "180")),
        auto_invert=env_flag("REAL_BINARIZE_AUTO_INVERT", True),
        autocontrast=env_flag("REAL_BINARIZE_AUTOCONTRAST", True),
    )


def display_image(path: str | Path, dataset_type: str) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        if str(dataset_type).lower() == "real":
            image = real_binarizer()(image)
        return np.asarray(image.convert("RGB"))


def save_binarized_inputs(
    array1: np.ndarray, array2: np.ndarray, output: str | Path, index: int
) -> tuple[Path, Path]:
    directory = Path(output).parent / "binarized"
    directory.mkdir(parents=True, exist_ok=True)
    line1 = directory / f"pair_{int(index)}_line1.png"
    line2 = directory / f"pair_{int(index)}_line2.png"
    Image.fromarray(array1).save(line1)
    Image.fromarray(array2).save(line2)
    return line1, line2


# Backward-compatible private names.
_real_pair_paths = real_pair_paths
_load_pair_manifest = load_pair_manifest
_group_split_pairs = group_split_pairs
_load_arabic_dataset_pairs = load_arabic_dataset_pairs
_display_image = display_image
