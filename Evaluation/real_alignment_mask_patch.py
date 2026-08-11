"""Use pair-specific real alignment masks with the synthetic NW mask metrics.

The component-aware Needleman-Wunsch evaluator already predicts a union of up to
three supported horizontal regions and scores that union against synthetic
binary masks.  Real augmented data now stores the equivalent ground-truth masks
in ``A.alignment_mask_path`` and ``B.alignment_mask_path`` inside a companion
``*_with_masks.jsonl`` manifest.

This patch replaces only the ground-truth mask lookup/geometry for real pairs.
The NW dynamic program, traceback, supported-component selection, visualisation,
and aggregate metric names remain unchanged, so synthetic and real evaluations
are directly comparable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from Evaluation._eval_utils import patch_range_to_pixels
from Evaluation import nw_discontinuous_regions as discontinuous
from Evaluation.real_subword_box_geometry import _geometry
from unified_line_geometry import resolved_geometry


_MASK_MAP: dict[tuple[str, str], tuple[Path, Path]] | None = None
_MASK_MANIFEST: Path | None = None
_PRINTED_MANIFEST = False


def _flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _normalise_path_value(value: object) -> Path:
    return Path(str(value or "").strip().replace("\\", "/")).expanduser()


def _resolve(value: object, manifest: Path, data_root: Path | None) -> Path:
    path = _normalise_path_value(value)
    if not str(path):
        raise FileNotFoundError("empty manifest path")
    if path.is_absolute():
        return path.resolve()

    candidates = [manifest.parent / path]
    if data_root is not None:
        candidates.extend((data_root / path, data_root.parent / path))
    candidates.append(Path.cwd() / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _companion(path: Path) -> Path:
    name = path.name
    if name.endswith("_with_masks.jsonl"):
        return path
    if name.endswith(".jsonl"):
        return path.with_name(name[:-6] + "_with_masks.jsonl")
    return path.with_name(name + "_with_masks.jsonl")


def _candidate_manifests() -> list[Path]:
    values: list[Path] = []
    explicit = os.environ.get("REAL_MASK_MANIFEST", "").strip()
    test_manifest = os.environ.get("TEST_MANIFEST", "").strip()
    data_dir = os.environ.get("REAL_DATA_DIR", "").strip()

    if explicit:
        values.append(Path(explicit).expanduser())
    if test_manifest:
        test_path = Path(test_manifest).expanduser()
        values.extend((_companion(test_path), test_path))
    if data_dir:
        root = Path(data_dir).expanduser()
        values.extend(
            (
                root / "test_manifest_with_masks.jsonl",
                root / "dataset_manifest_with_masks.jsonl",
            )
        )

    unique: list[Path] = []
    seen = set()
    for value in values:
        resolved = value.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _records(path: Path) -> Iterable[dict]:
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if raw:
            value = json.loads(raw)
            if isinstance(value, dict):
                yield value


def _mapping_from_manifest(path: Path) -> dict[tuple[str, str], tuple[Path, Path]]:
    data_root_value = os.environ.get("REAL_DATA_DIR", "").strip()
    data_root = Path(data_root_value).expanduser().resolve() if data_root_value else path.parent
    mapping: dict[tuple[str, str], tuple[Path, Path]] = {}
    for row in _records(path):
        side_a = row.get("A")
        side_b = row.get("B")
        if not isinstance(side_a, dict) or not isinstance(side_b, dict):
            continue
        if not side_a.get("alignment_mask_path") or not side_b.get("alignment_mask_path"):
            continue
        if not side_a.get("line_image_path") or not side_b.get("line_image_path"):
            continue

        image_a = _resolve(side_a["line_image_path"], path, data_root)
        image_b = _resolve(side_b["line_image_path"], path, data_root)
        mask_a = _resolve(side_a["alignment_mask_path"], path, data_root)
        mask_b = _resolve(side_b["alignment_mask_path"], path, data_root)
        key = (str(image_a), str(image_b))
        value = (mask_a, mask_b)
        previous = mapping.get(key)
        if previous is not None and previous != value:
            raise ValueError(
                "Ambiguous alignment masks for the same real image pair: "
                f"{image_a} <-> {image_b}; {previous} vs {value}"
            )
        mapping[key] = value
        mapping[(str(image_b), str(image_a))] = (mask_b, mask_a)
    return mapping


def _mask_mapping() -> dict[tuple[str, str], tuple[Path, Path]]:
    global _MASK_MAP, _MASK_MANIFEST, _PRINTED_MANIFEST
    if _MASK_MAP is not None:
        return _MASK_MAP

    explicit = os.environ.get("REAL_MASK_MANIFEST", "").strip()
    for candidate in _candidate_manifests():
        if not candidate.is_file():
            if explicit and candidate == Path(explicit).expanduser().resolve() and _flag(
                "REAL_REQUIRE_ALIGNMENT_MASKS", True
            ):
                raise FileNotFoundError(f"REAL_MASK_MANIFEST not found: {candidate}")
            continue
        mapping = _mapping_from_manifest(candidate)
        if mapping:
            _MASK_MAP = mapping
            _MASK_MANIFEST = candidate
            if not _PRINTED_MANIFEST:
                print(
                    f"Real NW mask ground truth: manifest={candidate} "
                    f"directed_pairs={len(mapping)}",
                    flush=True,
                )
                _PRINTED_MANIFEST = True
            return mapping

    _MASK_MAP = {}
    return _MASK_MAP


def _mask_intervals(mask_path: Path) -> tuple[list[tuple[float, float]], tuple[int, int]]:
    if not mask_path.is_file():
        raise FileNotFoundError(f"Alignment mask not found: {mask_path}")
    with Image.open(mask_path) as opened:
        mask = np.asarray(opened.convert("L"))
        size = opened.size
    columns = np.any(mask > 0, axis=0)
    indices = np.flatnonzero(columns)
    if not len(indices):
        return [], size

    intervals: list[tuple[float, float]] = []
    start = previous = int(indices[0])
    for value in map(int, indices[1:]):
        if value != previous + 1:
            intervals.append((float(start), float(previous + 1)))
            start = value
        previous = value
    intervals.append((float(start), float(previous + 1)))
    return intervals, size


def _merge_intervals(intervals) -> list[tuple[float, float]]:
    ordered = sorted(
        (float(min(left, right)), float(max(left, right)))
        for left, right in intervals
        if float(left) != float(right)
    )
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for left, right in ordered[1:]:
        if left <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], right)
        else:
            merged.append([left, right])
    return [(float(left), float(right)) for left, right in merged]


def _mapped_real_mask_intervals(
    mask_path: Path,
    image_path: Path,
    target_width: int,
    target_height: int,
) -> list[tuple[float, float]]:
    native, mask_size = _mask_intervals(mask_path)
    if not native:
        return []

    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        source_width = int(image.width)
        crop_box, scale_x, _scale_y, offset_x, _offset_y = _geometry(
            image, int(target_width), int(target_height)
        )
    crop_x0 = float(crop_box[0])
    mask_width = max(1, int(mask_size[0]))
    mask_to_source_x = float(source_width) / float(mask_width)

    mapped = []
    for left, right in native:
        source_left = float(left) * mask_to_source_x
        source_right = float(right) * mask_to_source_x
        target_left = (source_left - crop_x0) * float(scale_x) + float(offset_x)
        target_right = (source_right - crop_x0) * float(scale_x) + float(offset_x)
        target_left = max(0.0, min(float(target_width), target_left))
        target_right = max(0.0, min(float(target_width), target_right))
        if target_right > target_left:
            mapped.append((target_left, target_right))
    return _merge_intervals(mapped)


def _union_length(intervals) -> float:
    return float(sum(max(0.0, right - left) for left, right in intervals))


def _union_intersection(left, right) -> float:
    return float(
        sum(
            max(0.0, min(a1, b1) - max(a0, b0))
            for a0, a1 in left
            for b0, b1 in right
        )
    )


def _union_iou(predicted, target):
    predicted = _merge_intervals(predicted)
    target = _merge_intervals(target)
    if not predicted or not target:
        return None
    intersection = _union_intersection(predicted, target)
    union = _union_length(predicted) + _union_length(target) - intersection
    return float(intersection / union) if union > 0.0 else 0.0


def _empty_metrics() -> dict:
    keys = {}
    for prefix in ("line1", "line2"):
        keys.update(
            {
                f"{prefix}_pred_start_px": None,
                f"{prefix}_pred_end_px": None,
                f"{prefix}_gt_start_px": None,
                f"{prefix}_gt_end_px": None,
                f"{prefix}_region_iou": None,
                f"{prefix}_start_error_px": None,
                f"{prefix}_end_error_px": None,
            }
        )
    keys["mean_region_iou"] = None
    return keys


def _real_mask_metrics(
    path,
    similarity_shape,
    image1: Path,
    image2: Path,
    mask1: Path,
    mask2: Path,
    image_width1: int,
    image_width2: int,
    use_flip: bool,
) -> dict:
    keys = _empty_metrics()
    runs = discontinuous._path_runs(path)
    if not runs:
        return keys

    n1, n2 = map(int, similarity_shape)
    target_height = int(resolved_geometry().get("line_height", 128))
    specifications = (
        ("line1", 0, n1, int(image_width1), image1, mask1),
        ("line2", 1, n2, int(image_width2), image2, mask2),
    )
    ious = []
    for prefix, axis, n_windows, width, image_path, mask_path in specifications:
        predicted = []
        for run in runs:
            indices = [int(pair[axis]) for pair in run]
            if not indices:
                continue
            left, right = patch_range_to_pixels(
                min(indices), max(indices) + 1, n_windows, width, use_flip
            )
            predicted.append((min(left, right), max(left, right)))
        predicted = _merge_intervals(predicted)
        if not predicted:
            continue

        target = _mapped_real_mask_intervals(
            mask_path, image_path, width, target_height
        )
        pred_start = min(left for left, _ in predicted)
        pred_end = max(right for _, right in predicted)
        keys[f"{prefix}_pred_start_px"] = float(pred_start)
        keys[f"{prefix}_pred_end_px"] = float(pred_end)
        if not target:
            continue

        gt_start = min(left for left, _ in target)
        gt_end = max(right for _, right in target)
        iou = _union_iou(predicted, target)
        keys[f"{prefix}_gt_start_px"] = int(round(gt_start))
        keys[f"{prefix}_gt_end_px"] = int(round(gt_end))
        keys[f"{prefix}_region_iou"] = iou
        keys[f"{prefix}_start_error_px"] = abs(float(pred_start) - gt_start)
        keys[f"{prefix}_end_error_px"] = abs(float(pred_end) - gt_end)
        if iou is not None:
            ious.append(float(iou))

    if ious:
        keys["mean_region_iou"] = float(np.mean(ious))
    return keys


def _looks_like_real_pair(image1: Path, image2: Path) -> bool:
    root_value = os.environ.get("REAL_DATA_DIR", "").strip()
    if root_value:
        root = Path(root_value).expanduser().resolve()
        try:
            image1.resolve().relative_to(root)
            image2.resolve().relative_to(root)
            return True
        except ValueError:
            pass
    return "ArabicDatasetRealAug10K" in image1.parts or "ArabicDatasetRealAug10K" in image2.parts


def install(runner) -> None:
    """Replace synthetic-mask lookup with pair-specific real-mask lookup when available."""
    if getattr(runner, "_real_alignment_mask_patch_installed", False):
        return

    original = runner.synthetic_mask_region_metrics

    def mask_region_metrics(
        path,
        traceback,
        similarity_shape,
        image1,
        image2,
        image_width1,
        image_width2,
        use_flip,
    ):
        left = Path(image1).expanduser().resolve()
        right = Path(image2).expanduser().resolve()
        mapping = _mask_mapping()
        masks = mapping.get((str(left), str(right)))
        if masks is not None:
            return _real_mask_metrics(
                path,
                similarity_shape,
                left,
                right,
                masks[0],
                masks[1],
                int(image_width1),
                int(image_width2),
                bool(use_flip),
            )

        if _flag("REAL_REQUIRE_ALIGNMENT_MASKS", True) and _looks_like_real_pair(left, right):
            manifest = str(_MASK_MANIFEST or "<not found>")
            raise FileNotFoundError(
                "No pair-specific alignment masks found for real NW evaluation: "
                f"{left} <-> {right}. mask_manifest={manifest}. "
                "Generate *_with_masks.jsonl first or set REAL_MASK_MANIFEST."
            )
        return original(
            path,
            traceback,
            similarity_shape,
            image1,
            image2,
            image_width1,
            image_width2,
            use_flip,
        )

    runner.synthetic_mask_region_metrics = mask_region_metrics
    runner._real_alignment_mask_patch_installed = True
