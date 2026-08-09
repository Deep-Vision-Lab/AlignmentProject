"""Recover line-local subword boxes from flat page-level ``debug/bboxes.json``.

The real Arabic dataset stores one flat list of page-coordinate subword records
per A/B side.  A record has the schema::

    {"x1": ..., "y1": ..., "x2": ..., "y2": ..., "cx": ..., "cy": ..., "text": ...}

There is no explicit line id.  This module groups records into text lines by
vertical center, selects the group requested by ``line_XX.png``, keeps x
coordinates when the line crop is page-width, and recovers the page->line
vertical offset by aligning the projected boxes with actual ink in the cropped
line image.

It is deliberately conservative: horizontally cropped/rescaled line images are
rejected rather than assigned guessed coordinates.
"""
from __future__ import annotations

from difflib import SequenceMatcher
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

import numpy as np
from PIL import Image

from Evaluation.real_subword_box_metrics import BoxAnnotations, SubwordBox


_LINE_RE = re.compile(r"line[_-]?0*(\d+)", re.IGNORECASE)
_ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_REQUIRED = {"x1", "y1", "x2", "y2", "text"}


def _side_root(image_path: Path) -> Path:
    for parent in image_path.parents:
        if parent.name in {"A", "B"}:
            return parent
    return image_path.parent


def _line_index(image_path: Path) -> int | None:
    match = _LINE_RE.search(image_path.stem)
    return int(match.group(1)) if match else None


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _is_flat_record(value: Any) -> bool:
    if not isinstance(value, dict) or not _REQUIRED.issubset(value):
        return False
    if not str(value.get("text", "")).strip():
        return False
    return all(_as_float(value.get(key)) is not None for key in ("x1", "y1", "x2", "y2"))


def _cy(record: dict) -> float:
    explicit = _as_float(record.get("cy"))
    if explicit is not None:
        return explicit
    return (float(record["y1"]) + float(record["y2"])) / 2.0


def _cx(record: dict) -> float:
    explicit = _as_float(record.get("cx"))
    if explicit is not None:
        return explicit
    return (float(record["x1"]) + float(record["x2"])) / 2.0


def cluster_flat_page_boxes(records: list[dict]) -> list[list[dict]]:
    """Cluster flat page boxes into top-to-bottom text lines."""
    boxes = [record for record in records if _is_flat_record(record)]
    if not boxes:
        return []

    heights = [max(1.0, float(r["y2"]) - float(r["y1"])) for r in boxes]
    median_h = statistics.median(heights)
    center_gap_threshold = max(45.0, min(95.0, median_h * 0.95))

    ordered = sorted(boxes, key=_cy)
    groups: list[list[dict]] = []
    current: list[dict] = []
    previous_cy: float | None = None
    for record in ordered:
        center = _cy(record)
        if current and previous_cy is not None and center - previous_cy > center_gap_threshold:
            groups.append(current)
            current = []
        current.append(record)
        previous_cy = center
    if current:
        groups.append(current)

    # Merge accidental one-record clusters into their nearest neighboring line.
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for index, group in enumerate(list(groups)):
            if len(group) >= 2:
                continue
            center = statistics.mean(_cy(record) for record in group)
            choices: list[tuple[float, int]] = []
            if index > 0:
                choices.append((abs(center - statistics.mean(_cy(r) for r in groups[index - 1])), index - 1))
            if index + 1 < len(groups):
                choices.append((abs(center - statistics.mean(_cy(r) for r in groups[index + 1])), index + 1))
            if choices:
                _, target = min(choices)
                groups[target].extend(group)
                groups.pop(index)
                changed = True
                break

    groups.sort(key=lambda group: statistics.mean(_cy(record) for record in group))
    return groups


def _canonical(value: str) -> str:
    value = _ARABIC_DIACRITICS.sub("", str(value or "").replace("ـ", ""))
    return "".join(value.split())


def _candidate_transcript(image_path: Path) -> str:
    side = _side_root(image_path)
    filename = image_path.with_suffix(".txt").name
    for relative in (
        Path("text/final/original") / filename,
        Path("text/raw") / filename,
        Path("text/final/tashkeel") / filename,
    ):
        path = side / relative
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8").strip()
            except OSError:
                pass
    return ""


def _otsu_threshold(gray: np.ndarray) -> int:
    values = np.clip(gray, 0, 255).astype(np.uint8)
    hist = np.bincount(values.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 127
    probability = hist / total
    omega = np.cumsum(probability)
    mu = np.cumsum(probability * np.arange(256, dtype=np.float64))
    mu_total = mu[-1]
    denominator = omega * (1.0 - omega)
    numerator = (mu_total * omega - mu) ** 2
    score = np.zeros_like(numerator)
    valid = denominator > 1e-12
    score[valid] = numerator[valid] / denominator[valid]
    return int(np.argmax(score))


def _ink_mask(image: Image.Image) -> np.ndarray:
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    threshold = _otsu_threshold(gray)
    h, w = gray.shape
    band = max(1, min(8, h // 8, w // 8))
    border = np.concatenate((gray[:band].ravel(), gray[-band:].ravel(), gray[:, :band].ravel(), gray[:, -band:].ravel()))
    border_median = float(np.median(border)) if border.size else 255.0
    return gray <= threshold if border_median >= 127.5 else gray >= threshold


def _offset_score(ink: np.ndarray, group: list[dict], crop_top: int) -> float:
    height, width = ink.shape
    union = np.zeros((height, width), dtype=bool)
    density_sum = 0.0
    valid_boxes = 0
    for record in group:
        x0 = max(0, int(math.floor(float(record["x1"]))))
        x1 = min(width, int(math.ceil(float(record["x2"]))))
        y0 = max(0, int(math.floor(float(record["y1"]) - crop_top)))
        y1 = min(height, int(math.ceil(float(record["y2"]) - crop_top)))
        if x1 <= x0 or y1 <= y0:
            continue
        region = ink[y0:y1, x0:x1]
        if region.size == 0:
            continue
        union[y0:y1, x0:x1] = True
        density_sum += float(region.mean())
        valid_boxes += 1
    if valid_boxes != len(group):
        return -1.0
    inside = float(np.logical_and(ink, union).sum())
    # Total captured ink dominates; mean per-box density breaks broad plateaus.
    return inside + 25.0 * density_sum


def _recover_crop_top(image: Image.Image, group: list[dict]) -> tuple[int, float]:
    height = image.height
    min_page_y = min(float(record["y1"]) for record in group)
    max_page_y = max(float(record["y2"]) for record in group)

    # All selected subword boxes must fit completely inside the line crop.
    lower = int(math.ceil(max_page_y - height))
    upper = int(math.floor(min_page_y))
    if lower > upper:
        raise ValueError(
            f"line bbox band ({max_page_y - min_page_y:.1f}px) cannot fit inside "
            f"line image height {height}px"
        )

    ink = _ink_mask(image)
    best_top = lower
    best_score = -1.0
    for crop_top in range(lower, upper + 1):
        score = _offset_score(ink, group, crop_top)
        if score > best_score:
            best_score = score
            best_top = crop_top
    return best_top, best_score


def _load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, list):
        return []
    return [record for record in payload if _is_flat_record(record)]


def _flat_json_candidates(image_path: Path) -> list[Path]:
    side = _side_root(image_path)
    candidates = [
        side / "debug" / "bboxes.json",
        side / "bboxes.json",
        side / "debug" / "bbox.json",
        side / "bbox.json",
    ]
    return [path for path in candidates if path.is_file()]


def load_flat_page_line_annotations(image_path: str | Path) -> BoxAnnotations:
    """Return line-local boxes recovered from the dataset's flat page schema."""
    image_path = Path(image_path).expanduser().resolve()
    line_index = _line_index(image_path)
    if line_index is None:
        return BoxAnnotations((), "", "json-flat-page", "no_boxes", f"Cannot infer line index from {image_path.name}")
    if not image_path.is_file():
        return BoxAnnotations((), "", "json-flat-page", "missing", f"Line image does not exist: {image_path}")

    candidates = _flat_json_candidates(image_path)
    if not candidates:
        return BoxAnnotations((), "", "json-flat-page", "missing", "No side-local flat bboxes.json was found")

    errors: list[str] = []
    for path in candidates:
        try:
            records = _load_records(path)
            groups = cluster_flat_page_boxes(records)
            if not groups:
                errors.append(f"{path}: no flat text-bearing bbox records")
                continue
            if not (1 <= line_index <= len(groups)):
                errors.append(f"{path}: requested line {line_index}, inferred {len(groups)} page lines")
                continue

            group = groups[line_index - 1]
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")

            page_max_x = max(float(record["x2"]) for record in records)
            page_min_x = min(float(record["x1"]) for record in records)
            if page_min_x < -5 or page_max_x > image.width + 5:
                errors.append(
                    f"{path}: line image width={image.width} does not preserve page x coordinates "
                    f"[{page_min_x:.1f}, {page_max_x:.1f}]"
                )
                continue

            crop_top, score = _recover_crop_top(image, group)
            ordered = sorted(group, key=_cx, reverse=True)
            boxes: list[SubwordBox] = []
            for source_row, record in enumerate(ordered, start=1):
                x0 = float(record["x1"])
                x1 = float(record["x2"])
                y0 = float(record["y1"]) - crop_top
                y1 = float(record["y2"]) - crop_top
                text = str(record.get("text", "")).strip()
                if not (0 <= x0 < x1 <= image.width and 0 <= y0 < y1 <= image.height):
                    raise ValueError(
                        f"projected bbox outside {image.name}: text={text!r} "
                        f"box=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f}) crop_top={crop_top}"
                    )
                boxes.append(SubwordBox(text, x0, y0, x1, y1, source_row))

            transcript = _candidate_transcript(image_path)
            if transcript:
                bbox_text = "".join(box.text for box in boxes)
                expected = _canonical(transcript)
                observed = _canonical(bbox_text)
                ratio = SequenceMatcher(None, expected, observed).ratio() if expected or observed else 1.0
                if ratio < 0.72:
                    errors.append(
                        f"{path}: inferred line_{line_index:02d} text does not match transcript "
                        f"well enough (ratio={ratio:.3f})"
                    )
                    continue

            # Keep the normal BoxAnnotations interface.  The source field names
            # the recovery path so callers can distinguish it in diagnostics.
            return BoxAnnotations(
                tuple(boxes),
                str(path),
                f"json-flat-page:line={line_index}:crop_top={crop_top}:ink_score={score:.1f}",
                "ok",
                "",
            )
        except Exception as exc:
            errors.append(f"{path}: {type(exc).__name__}: {exc}")

    return BoxAnnotations(
        (),
        str(candidates[0]),
        "json-flat-page",
        "no_boxes",
        "; ".join(errors[:4]) or f"Could not recover boxes for {image_path.name}",
    )
