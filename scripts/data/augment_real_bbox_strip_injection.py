#!/usr/bin/env python3
"""Build real-data aligned injection samples from complete subword bboxes.

The augmentation is intentionally bbox-driven:

* every source line is normalized to exactly 128 px high while preserving width;
* an injected component is a FULL-HEIGHT vertical strip (y=0..128);
* strip x-bounds are exactly the outer x-bounds of consecutive complete subword boxes;
* a strip is accepted only when no unselected annotated subword intersects it;
* donor pixels are never squeezed to a target width: the target line width grows/shrinks
  by the true donor-strip width;
* all surviving target bboxes and all inserted donor bboxes are rebuilt in the new
  image coordinate system;
* the final transcript is rebuilt from the final bbox reading order, so every
  annotated subword in the image appears exactly once in the text.

Aligned pair augmentation uses the same canonical donor subword span for A and B,
but requires the two handwriting strips to come from different real source images.
The script can place 1--3 separated aligned injected strips in each output pair.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
import re
import shutil
import unicodedata
from typing import Iterable

from PIL import Image, ImageDraw

from Evaluation.real_subword_box_json import load_json_annotations
from Evaluation.real_subword_box_metrics import SubwordBox


_ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")


@dataclass(frozen=True)
class BoxItem:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    source_row: int
    origin_image: str
    injected: bool = False

    @property
    def center_x(self) -> float:
        return (self.x0 + self.x1) / 2.0


@dataclass(frozen=True)
class SourceLine:
    image_path: Path
    text_path: Path
    annotation_path: str
    pair_id: str
    side: str
    line_idx: int
    image: Image.Image
    boxes: tuple[BoxItem, ...]


@dataclass(frozen=True)
class DonorRun:
    line: SourceLine
    start: int
    size: int
    text: str
    canonical_text: str
    x0: int
    x1: int


@dataclass
class LineState:
    image: Image.Image
    boxes: list[BoxItem]
    injected_strips: list[tuple[int, int]]
    operations: list[dict]

    @property
    def text(self) -> str:
        return _text_from_boxes(self.boxes)


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    for candidate in (root / path, root.parent / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (root / path).resolve()


def _read_manifest(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _strip_marks(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value or ""))
    return _ARABIC_DIACRITICS.sub("", value.replace("ـ", ""))


def _canonical(value: str) -> str:
    return " ".join(_strip_marks(value).split())


def _compact(value: str) -> str:
    return "".join(_canonical(value).split())


def _text_from_boxes(boxes: Iterable[BoxItem]) -> str:
    # The augmented transcript is intentionally annotation-driven.  One bbox
    # corresponds to one complete connected subword; every surviving bbox is
    # represented exactly once in the final text and in RTL reading order.
    return " ".join(box.text.strip() for box in boxes if box.text.strip())


def _normalise_height(
    image: Image.Image,
    boxes: Iterable[SubwordBox],
    target_height: int,
    image_path: Path,
) -> tuple[Image.Image, tuple[BoxItem, ...]]:
    source = image.convert("RGB")
    if source.height <= 0:
        raise ValueError(f"invalid source image height: {image_path}")
    scale = float(target_height) / float(source.height)
    target_width = max(1, int(round(source.width * scale)))
    resized = source.resize((target_width, target_height), Image.Resampling.LANCZOS)

    scaled = []
    for row, box in enumerate(boxes, start=1):
        text = str(box.text or "").strip()
        if not text:
            raise ValueError(f"bbox without subword text in {image_path}: row={row}")
        item = BoxItem(
            text=text,
            x0=float(box.x0) * scale,
            y0=float(box.y0) * scale,
            x1=float(box.x1) * scale,
            y1=float(box.y1) * scale,
            source_row=int(getattr(box, "source_row", row) or row),
            origin_image=str(image_path),
            injected=False,
        )
        if not (0 <= item.x0 < item.x1 <= target_width and 0 <= item.y0 < item.y1 <= target_height):
            raise ValueError(f"scaled bbox outside normalized image {image_path}: {item}")
        scaled.append(item)

    # Single-line Arabic reading order is right-to-left.  Sorting by horizontal
    # center makes the final transcript independent of JSON row ordering.
    scaled.sort(key=lambda box: box.center_x, reverse=True)
    return resized, tuple(scaled)


def _load_line(
    dataset_root: Path,
    pair_id: str,
    side_name: str,
    side: dict,
    target_height: int,
) -> SourceLine:
    image_path = _resolve(dataset_root, side["line_image_path"])
    text_path = _resolve(dataset_root, side["text_original_path"])
    annotations = load_json_annotations(image_path)
    if annotations.status != "ok" or not annotations.boxes:
        raise ValueError(
            f"bbox.json unresolved for {image_path}: status={annotations.status} error={annotations.error}"
        )
    with Image.open(image_path) as opened:
        image, boxes = _normalise_height(opened, annotations.boxes, target_height, image_path)
    if not boxes:
        raise ValueError(f"no text-bearing subword boxes for {image_path}")
    return SourceLine(
        image_path=image_path,
        text_path=text_path,
        annotation_path=annotations.workbook,
        pair_id=str(pair_id),
        side=side_name,
        line_idx=int(side.get("line_idx", -1)),
        image=image,
        boxes=boxes,
    )


def _x_bounds(boxes: Iterable[BoxItem]) -> tuple[int, int]:
    values = tuple(boxes)
    if not values:
        raise ValueError("cannot compute x bounds for an empty bbox run")
    x0 = int(math.floor(min(box.x0 for box in values)))
    x1 = int(math.ceil(max(box.x1 for box in values)))
    if x1 <= x0:
        raise ValueError("invalid bbox run width")
    return x0, x1


def _overlap_width(box: BoxItem, x0: float, x1: float) -> float:
    return max(0.0, min(box.x1, x1) - max(box.x0, x0))


def _run_is_x_isolated(boxes: list[BoxItem] | tuple[BoxItem, ...], start: int, size: int) -> bool:
    selected_indices = set(range(start, start + size))
    selected = [boxes[index] for index in range(start, start + size)]
    x0, x1 = _x_bounds(selected)
    for index, box in enumerate(boxes):
        if index in selected_indices:
            continue
        # A full-height strip would physically contain/cut this subword, so the
        # run is invalid.  This is what guarantees every bbox in the final image
        # is complete rather than partially clipped by an injection boundary.
        if _overlap_width(box, x0, x1) > 0.5:
            return False
    return True


def _run_text(boxes: Iterable[BoxItem]) -> str:
    return " ".join(box.text.strip() for box in boxes if box.text.strip())


def _build_donor_index(
    lines: Iterable[SourceLine],
    max_run_boxes: int,
    min_chars: int,
    max_chars: int,
) -> dict[tuple[int, str], list[DonorRun]]:
    index: dict[tuple[int, str], list[DonorRun]] = defaultdict(list)
    for line in lines:
        boxes = line.boxes
        for size in range(1, min(max_run_boxes, len(boxes)) + 1):
            for start in range(0, len(boxes) - size + 1):
                if not _run_is_x_isolated(boxes, start, size):
                    continue
                selected = boxes[start : start + size]
                text = _run_text(selected)
                canonical = _canonical(text)
                compact_length = len(_compact(canonical))
                if not canonical or not (min_chars <= compact_length <= max_chars):
                    continue
                x0, x1 = _x_bounds(selected)
                index[(size, canonical)].append(
                    DonorRun(line, start, size, text, canonical, x0, x1)
                )

    usable: dict[tuple[int, str], list[DonorRun]] = {}
    for key, runs in index.items():
        by_image: dict[Path, DonorRun] = {}
        for run in runs:
            by_image.setdefault(run.line.image_path, run)
        if len(by_image) >= 2:
            usable[key] = list(by_image.values())
    return usable


def _copy_source(line: SourceLine) -> LineState:
    return LineState(
        image=line.image.copy(),
        boxes=[replace(box) for box in line.boxes],
        injected_strips=[],
        operations=[],
    )


def _target_candidates(
    state: LineState,
    size: int,
    donor_width: int,
    donor_text: str,
    width_ratio_min: float,
    width_ratio_max: float,
) -> list[int]:
    candidates = []
    for start in range(0, len(state.boxes) - size + 1):
        selected = state.boxes[start : start + size]
        if any(box.injected for box in selected):
            continue
        if not _run_is_x_isolated(state.boxes, start, size):
            continue
        x0, x1 = _x_bounds(selected)
        target_width = max(1, x1 - x0)
        ratio = float(donor_width) / float(target_width)
        if not (width_ratio_min <= ratio <= width_ratio_max):
            continue
        if _canonical(_run_text(selected)) == _canonical(donor_text):
            continue
        candidates.append(start)
    return candidates


def _shift_box_x(box: BoxItem, delta: float) -> BoxItem:
    return replace(box, x0=box.x0 + delta, x1=box.x1 + delta)


def _shift_existing_strips(
    strips: list[tuple[int, int]],
    target_x0: int,
    target_x1: int,
    delta: int,
) -> list[tuple[int, int]] | None:
    updated = []
    for x0, x1 in strips:
        if x1 <= target_x0:
            updated.append((x0, x1))
        elif x0 >= target_x1:
            updated.append((x0 + delta, x1 + delta))
        else:
            # Never replace or cut an already injected region.
            return None
    return updated


def _splice_full_height_run(
    state: LineState,
    donor: DonorRun,
    target_start: int,
    target_height: int,
) -> LineState | None:
    size = donor.size
    if target_start < 0 or target_start + size > len(state.boxes):
        return None
    if not _run_is_x_isolated(state.boxes, target_start, size):
        return None

    selected_target = state.boxes[target_start : target_start + size]
    target_x0, target_x1 = _x_bounds(selected_target)
    donor_x0, donor_x1 = donor.x0, donor.x1
    donor_width = donor_x1 - donor_x0
    target_width = target_x1 - target_x0
    if donor_width <= 0 or target_width <= 0:
        return None

    # FULL HEIGHT strip: y is deliberately never cropped by bbox y-bounds.
    donor_strip = donor.line.image.crop((donor_x0, 0, donor_x1, target_height))
    if donor_strip.height != target_height:
        return None

    delta = donor_width - target_width
    new_width = state.image.width + delta
    if new_width <= 0:
        return None

    shifted_strips = _shift_existing_strips(
        state.injected_strips, target_x0, target_x1, delta
    )
    if shifted_strips is None:
        return None

    canvas = Image.new("RGB", (new_width, target_height), "white")
    left = state.image.crop((0, 0, target_x0, target_height))
    right = state.image.crop((target_x1, 0, state.image.width, target_height))
    canvas.paste(left, (0, 0))
    canvas.paste(donor_strip, (target_x0, 0))
    canvas.paste(right, (target_x0 + donor_width, 0))

    selected_indices = set(range(target_start, target_start + size))
    kept: list[BoxItem] = []
    for index, box in enumerate(state.boxes):
        if index in selected_indices:
            continue
        if box.x1 <= target_x0 + 0.5:
            kept.append(box)
        elif box.x0 >= target_x1 - 0.5:
            kept.append(_shift_box_x(box, delta))
        else:
            # An unselected box would be partly destroyed by the splice.
            return None

    donor_boxes = []
    donor_selected = donor.line.boxes[donor.start : donor.start + donor.size]
    shift = float(target_x0 - donor_x0)
    for box in donor_selected:
        donor_boxes.append(
            replace(
                box,
                x0=box.x0 + shift,
                x1=box.x1 + shift,
                injected=True,
            )
        )

    # Preserve semantic RTL bbox order: replace the exact target run in-place.
    before = []
    after = []
    for index, box in enumerate(state.boxes):
        if index < target_start:
            adjusted = box
            if box.x0 >= target_x1 - 0.5:
                adjusted = _shift_box_x(box, delta)
            before.append(adjusted)
        elif index >= target_start + size:
            adjusted = box
            if box.x0 >= target_x1 - 0.5:
                adjusted = _shift_box_x(box, delta)
            after.append(adjusted)

    new_boxes = before + donor_boxes + after
    for box in new_boxes:
        if not (0 <= box.x0 < box.x1 <= new_width and 0 <= box.y0 < box.y1 <= target_height):
            return None

    inserted_strip = (target_x0, target_x0 + donor_width)
    operation = {
        "type": "bbox_full_height_strip_injection",
        "height_px": int(target_height),
        "target_x0": int(target_x0),
        "target_x1_before": int(target_x1),
        "target_width_replaced": int(target_width),
        "donor_x0": int(donor_x0),
        "donor_x1": int(donor_x1),
        "donor_width_inserted": int(donor_width),
        "new_strip_x0": int(inserted_strip[0]),
        "new_strip_x1": int(inserted_strip[1]),
        "width_delta": int(delta),
        "donor_text": donor.text,
        "donor_canonical_text": donor.canonical_text,
        "donor_image": str(donor.line.image_path),
        "donor_annotation": donor.line.annotation_path,
        "subword_count": int(donor.size),
    }
    return LineState(
        image=canvas,
        boxes=new_boxes,
        injected_strips=shifted_strips + [inserted_strip],
        operations=state.operations + [operation],
    )


def _choose_aligned_donors(
    rng: random.Random,
    donor_index: dict[tuple[int, str], list[DonorRun]],
    forbidden_images: set[Path],
    used_texts: set[str],
) -> tuple[DonorRun, DonorRun] | None:
    keys = [key for key in donor_index if key[1] not in used_texts]
    rng.shuffle(keys)
    for key in keys:
        eligible = [run for run in donor_index[key] if run.line.image_path not in forbidden_images]
        if len(eligible) < 2:
            continue
        rng.shuffle(eligible)
        first = eligible[0]
        second = next(
            (run for run in eligible[1:] if run.line.image_path != first.line.image_path),
            None,
        )
        if second is not None:
            return first, second
    return None


def _augment_aligned_pair(
    rng: random.Random,
    source_a: SourceLine,
    source_b: SourceLine,
    donor_index: dict[tuple[int, str], list[DonorRun]],
    regions: int,
    args,
) -> tuple[LineState, LineState, dict] | None:
    line_a = _copy_source(source_a)
    line_b = _copy_source(source_b)
    used_texts: set[str] = set()
    forbidden_images = {source_a.image_path, source_b.image_path}
    region_meta = []

    for region_index in range(regions):
        placed = False
        for _ in range(args.max_attempts_per_region):
            donors = _choose_aligned_donors(
                rng, donor_index, forbidden_images, used_texts
            )
            if donors is None:
                break
            donor_a, donor_b = donors
            donor_width_a = donor_a.x1 - donor_a.x0
            donor_width_b = donor_b.x1 - donor_b.x0
            candidates_a = _target_candidates(
                line_a,
                donor_a.size,
                donor_width_a,
                donor_a.text,
                args.injection_width_ratio_min,
                args.injection_width_ratio_max,
            )
            candidates_b = _target_candidates(
                line_b,
                donor_b.size,
                donor_width_b,
                donor_b.text,
                args.injection_width_ratio_min,
                args.injection_width_ratio_max,
            )
            if not candidates_a or not candidates_b:
                used_texts.add(donor_a.canonical_text)
                continue

            rng.shuffle(candidates_a)
            rng.shuffle(candidates_b)
            new_a = _splice_full_height_run(
                line_a, donor_a, candidates_a[0], args.height
            )
            new_b = _splice_full_height_run(
                line_b, donor_b, candidates_b[0], args.height
            )
            if new_a is None or new_b is None:
                continue

            line_a, line_b = new_a, new_b
            used_texts.add(donor_a.canonical_text)
            forbidden_images.update({donor_a.line.image_path, donor_b.line.image_path})
            region_meta.append(
                {
                    "region": region_index + 1,
                    "shared_text": donor_a.canonical_text,
                    "donor_A": str(donor_a.line.image_path),
                    "donor_B": str(donor_b.line.image_path),
                    "A_operation": line_a.operations[-1],
                    "B_operation": line_b.operations[-1],
                }
            )
            placed = True
            break
        if not placed:
            return None

    return line_a, line_b, {
        "mode": "aligned_bbox_full_height_strip_injection",
        "height_px": int(args.height),
        "regions": int(regions),
        "region_details": region_meta,
        "text_policy": "reconstructed from final RTL bbox sequence",
        "complete_subword_policy": "reject any strip intersecting an unselected bbox",
    }


def _bbox_payload(image_name: str, state: LineState) -> dict:
    return {
        "image": image_name,
        "height": state.image.height,
        "width": state.image.width,
        "reading_order": "rtl",
        "boxes": [
            {
                "text": box.text,
                "x0": round(float(box.x0), 3),
                "y0": round(float(box.y0), 3),
                "x1": round(float(box.x1), 3),
                "y1": round(float(box.y1), 3),
                "injected": bool(box.injected),
                "origin_image": box.origin_image,
            }
            for box in state.boxes
        ],
    }


def _save_side(root: Path, name: str, state: LineState) -> dict:
    side_root = root / name
    side_root.mkdir(parents=True, exist_ok=True)
    image_path = side_root / "line.png"
    text_path = side_root / "text_original.txt"
    bbox_path = side_root / "bbox.json"
    state.image.save(image_path)
    text_path.write_text(state.text.strip() + "\n", encoding="utf-8")
    bbox_path.write_text(
        json.dumps(_bbox_payload(image_path.name, state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {
        "line_image_path": str(image_path),
        "text_original_path": str(text_path),
        "bbox_path": str(bbox_path),
    }


def _draw_injected_strips(state: LineState) -> Image.Image:
    image = state.image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    for x0, x1 in state.injected_strips:
        draw.rectangle((x0, 0, x1 - 1, image.height - 1), outline=(220, 30, 30), width=3)
    return image


def _fit_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    copy = image.convert("RGB").copy()
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    panel = Image.new("RGB", (width, height), "white")
    panel.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    return panel


def _save_preview(
    path: Path,
    source_a: SourceLine,
    source_b: SourceLine,
    augmented_a: LineState,
    augmented_b: LineState,
    metadata: dict,
) -> None:
    panel_w, panel_h = 1100, 190
    header_h = 34
    sheet = Image.new("RGB", (panel_w * 2, header_h + panel_h * 2), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), f"{metadata['regions']} aligned full-height bbox strip injection region(s)", fill="black")
    sheet.paste(_fit_panel(source_a.image, panel_w, panel_h), (0, header_h))
    sheet.paste(_fit_panel(source_b.image, panel_w, panel_h), (panel_w, header_h))
    sheet.paste(_fit_panel(_draw_injected_strips(augmented_a), panel_w, panel_h), (0, header_h + panel_h))
    sheet.paste(_fit_panel(_draw_injected_strips(augmented_b), panel_w, panel_h), (panel_w, header_h + panel_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)

    path.with_suffix(".txt").write_text(
        "ORIGINAL A\n" + _text_from_boxes(source_a.boxes) + "\n\n"
        "ORIGINAL B\n" + _text_from_boxes(source_b.boxes) + "\n\n"
        "AUGMENTED A\n" + augmented_a.text + "\n\n"
        "AUGMENTED B\n" + augmented_b.text + "\n\n"
        "INJECTED SHARED REGIONS\n" + "\n".join(
            f"{item['region']}: {item['shared_text']}" for item in metadata["region_details"]
        ) + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("DataSet/ArabicDataset"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-pairs", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--labels", default="high_match,medium_match")
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--min-regions", type=int, default=1)
    parser.add_argument("--max-regions", type=int, default=3)
    parser.add_argument("--max-run-boxes", type=int, default=3)
    parser.add_argument("--injection-min-chars", type=int, default=4)
    parser.add_argument("--injection-max-chars", type=int, default=28)
    parser.add_argument("--injection-width-ratio-min", type=float, default=0.50)
    parser.add_argument("--injection-width-ratio-max", type=float, default=2.00)
    parser.add_argument("--max-attempts-per-output", type=int, default=120)
    parser.add_argument("--max-attempts-per-region", type=int, default=80)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.height <= 0:
        raise ValueError("--height must be positive")
    if args.num_pairs <= 0:
        raise ValueError("--num-pairs must be positive")
    if not (1 <= args.min_regions <= args.max_regions <= 3):
        raise ValueError("require 1 <= --min-regions <= --max-regions <= 3")

    dataset_root = args.data_dir.expanduser().resolve()
    manifest_path = (
        args.manifest.expanduser().resolve()
        if args.manifest is not None
        else dataset_root / "dataset_manifest.jsonl"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        if not args.overwrite:
            raise FileExistsError(f"Output is not empty: {output_root}; pass --overwrite")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    labels = {item.strip() for item in args.labels.split(",") if item.strip()}
    raw_records = [
        record
        for record in _read_manifest(manifest_path)
        if not labels or str(record.get("label_type", "")) in labels
    ]
    if not raw_records:
        raise ValueError(f"No manifest pairs remain for labels={sorted(labels)}")

    line_cache: dict[Path, SourceLine] = {}
    usable_pairs: list[tuple[dict, SourceLine, SourceLine]] = []
    rejected: Counter[str] = Counter()
    for record in raw_records:
        try:
            pair_id = str(record.get("pair_id", len(usable_pairs) + 1))
            loaded = []
            for side_name in ("A", "B"):
                side = record[side_name]
                image_path = _resolve(dataset_root, side["line_image_path"])
                if image_path not in line_cache:
                    line_cache[image_path] = _load_line(
                        dataset_root, pair_id, side_name, side, args.height
                    )
                loaded.append(line_cache[image_path])
            usable_pairs.append((record, loaded[0], loaded[1]))
        except Exception as exc:
            rejected[f"{type(exc).__name__}: {exc}"] += 1

    if not usable_pairs:
        details = "\n".join(
            f"  {count} x {reason}" for reason, count in rejected.most_common(8)
        )
        raise RuntimeError(
            "No bbox-valid real line pairs are available. This augmentation requires "
            "true line-level text-bearing subword boxes; it never estimates bbox positions.\n"
            + details
        )

    donor_index = _build_donor_index(
        line_cache.values(),
        args.max_run_boxes,
        args.injection_min_chars,
        args.injection_max_chars,
    )
    if not donor_index:
        raise RuntimeError(
            "No repeated bbox-exact subword runs were found in at least two distinct real lines."
        )

    rng = random.Random(args.seed)
    output_manifest = output_root / "dataset_manifest.jsonl"
    preview_root = output_root / "previews"
    generated = 0
    region_counts: Counter[int] = Counter()
    generation_failures: Counter[str] = Counter()

    with output_manifest.open("w", encoding="utf-8") as handle:
        while generated < args.num_pairs:
            result = None
            selected_record = selected_a = selected_b = None
            requested_regions = rng.randint(args.min_regions, args.max_regions)
            for _ in range(args.max_attempts_per_output):
                selected_record, selected_a, selected_b = rng.choice(usable_pairs)
                try:
                    result = _augment_aligned_pair(
                        rng,
                        selected_a,
                        selected_b,
                        donor_index,
                        requested_regions,
                        args,
                    )
                except Exception as exc:
                    generation_failures[f"{type(exc).__name__}: {exc}"] += 1
                    result = None
                if result is not None:
                    break
            if result is None:
                raise RuntimeError(
                    f"Could not create output {generated + 1}/{args.num_pairs} with "
                    f"{requested_regions} injected region(s) after {args.max_attempts_per_output} attempts"
                )

            augmented_a, augmented_b, metadata = result
            output_index = generated + 1
            pair_root = output_root / "pairs" / f"aug_{output_index:06d}"
            saved_a = _save_side(pair_root, "A", augmented_a)
            saved_b = _save_side(pair_root, "B", augmented_b)
            source_label = str(selected_record.get("label_type", ""))
            record = {
                "pair_id": f"{selected_record.get('pair_id', output_index)}__bbox_strip_aug_{output_index:06d}",
                "label_type": "augmented_positive",
                "source_label_type": source_label,
                "source_pair_id": selected_record.get("pair_id"),
                "augmentation": metadata,
                "A": {
                    **saved_a,
                    "line_idx": int(selected_a.line_idx),
                    "source_line_image_path": str(selected_a.image_path),
                    "source_bbox_json": selected_a.annotation_path,
                },
                "B": {
                    **saved_b,
                    "line_idx": int(selected_b.line_idx),
                    "source_line_image_path": str(selected_b.image_path),
                    "source_bbox_json": selected_b.annotation_path,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

            if args.preview:
                _save_preview(
                    preview_root / f"preview_{output_index:03d}_{requested_regions}regions.png",
                    selected_a,
                    selected_b,
                    augmented_a,
                    augmented_b,
                    metadata,
                )

            generated += 1
            region_counts[requested_regions] += 1

    summary = {
        "source_dataset": str(dataset_root),
        "source_manifest": str(manifest_path),
        "height_px": int(args.height),
        "generated_pairs": int(generated),
        "bbox_valid_source_pairs": len(usable_pairs),
        "bbox_valid_unique_lines": len(line_cache),
        "repeated_aligned_donor_runs": len(donor_index),
        "region_counts": {str(k): v for k, v in sorted(region_counts.items())},
        "strict_complete_subwords": True,
        "strip_policy": "full image height; x bounds equal selected consecutive subword bbox outer bounds",
        "image_policy": "splice donor strip at native width after height normalization; no horizontal squeezing",
        "text_policy": "rebuild from final RTL bbox sequence; every final bbox text appears once",
        "bbox_policy": "rebuild donor and shifted target coordinates; reject any partial bbox intersection",
        "rejected_source_reasons": dict(rejected),
        "generation_failures": dict(generation_failures),
        "output_manifest": str(output_manifest),
    }
    (output_root / "augmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
