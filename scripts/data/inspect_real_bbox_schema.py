#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from pprint import pprint

from PIL import Image

ARABIC_RE = re.compile(r"[\u0600-\u06ff]+")
LINE_RE = re.compile(r"line[_-]?(\d+)", re.IGNORECASE)


def walk_keys(value, prefix=""):
    out = set()
    if isinstance(value, dict):
        for k, v in value.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            out.add(path)
            out |= walk_keys(v, path)
    elif isinstance(value, list):
        for v in value[:50]:
            out |= walk_keys(v, prefix + "[]")
    return out


def _is_flat_bbox_record(value):
    if not isinstance(value, dict):
        return False
    required = {"x1", "y1", "x2", "y2", "text"}
    return required.issubset(value)


def _cluster_page_boxes(records):
    boxes = [r for r in records if _is_flat_bbox_record(r)]
    if not boxes:
        return []

    heights = [max(1.0, float(r["y2"]) - float(r["y1"])) for r in boxes]
    median_h = statistics.median(heights)
    # Same-line center variation is much smaller than the inter-line gap in this
    # dataset.  Keep the threshold data-driven but bounded for stability.
    center_gap_threshold = max(45.0, min(95.0, median_h * 0.95))

    ordered = sorted(boxes, key=lambda r: float(r.get("cy", (float(r["y1"]) + float(r["y2"])) / 2.0)))
    groups = []
    current = []
    previous_cy = None
    for rec in ordered:
        cy = float(rec.get("cy", (float(rec["y1"]) + float(rec["y2"])) / 2.0))
        if current and previous_cy is not None and cy - previous_cy > center_gap_threshold:
            groups.append(current)
            current = []
        current.append(rec)
        previous_cy = cy
    if current:
        groups.append(current)

    # Merge tiny accidental clusters into the nearest neighboring line.
    changed = True
    while changed and len(groups) > 1:
        changed = False
        for i, group in enumerate(list(groups)):
            if len(group) >= 2:
                continue
            cy = statistics.mean(float(r.get("cy", (float(r["y1"]) + float(r["y2"])) / 2.0)) for r in group)
            choices = []
            if i > 0:
                prev_cy = statistics.mean(float(r.get("cy", (float(r["y1"]) + float(r["y2"])) / 2.0)) for r in groups[i - 1])
                choices.append((abs(cy - prev_cy), i - 1))
            if i + 1 < len(groups):
                next_cy = statistics.mean(float(r.get("cy", (float(r["y1"]) + float(r["y2"])) / 2.0)) for r in groups[i + 1])
                choices.append((abs(cy - next_cy), i + 1))
            if choices:
                _, target = min(choices)
                groups[target].extend(group)
                groups.pop(i)
                changed = True
                break

    groups.sort(key=lambda g: statistics.mean(float(r.get("cy", (float(r["y1"]) + float(r["y2"])) / 2.0)) for r in g))
    return groups


def _group_summary(group):
    x0 = min(float(r["x1"]) for r in group)
    y0 = min(float(r["y1"]) for r in group)
    x1 = max(float(r["x2"]) for r in group)
    y1 = max(float(r["y2"]) for r in group)
    cy = statistics.mean(float(r.get("cy", (float(r["y1"]) + float(r["y2"])) / 2.0)) for r in group)
    ordered_rtl = sorted(group, key=lambda r: float(r.get("cx", (float(r["x1"]) + float(r["x2"])) / 2.0)), reverse=True)
    text = " ".join(str(r.get("text", "")).strip() for r in ordered_rtl if str(r.get("text", "")).strip())
    return {
        "count": len(group),
        "x0": x0,
        "y0": y0,
        "x1": x1,
        "y1": y1,
        "width": x1 - x0,
        "height": y1 - y0,
        "mean_cy": cy,
        "rtl_text": text,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--bbox-json", default=None)
    p.add_argument("--text", default=None)
    p.add_argument("--max-records", type=int, default=8)
    args = p.parse_args()

    image = Path(args.image).expanduser().resolve()
    side = next((x for x in image.parents if x.name in {"A", "B"}), image.parent)
    bbox = Path(args.bbox_json).expanduser().resolve() if args.bbox_json else side / "debug" / "bboxes.json"

    print("IMAGE:", image)
    print("SIDE:", side)
    print("BBOX JSON:", bbox)
    if not bbox.is_file():
        raise SystemExit("bbox json not found")
    if image.is_file():
        with Image.open(image) as opened:
            print("LINE IMAGE SIZE:", opened.size)

    text = ""
    if args.text:
        text_path = Path(args.text).expanduser().resolve()
        if text_path.is_file():
            text = text_path.read_text(encoding="utf-8").strip()
            print("TEXT FILE:", text_path)
            print("TEXT:", text)

    data = json.loads(bbox.read_text(encoding="utf-8-sig"))
    print("TOP TYPE:", type(data).__name__)
    if hasattr(data, "__len__"):
        print("TOP LENGTH:", len(data))

    records = data if isinstance(data, list) else [data]
    print("\n=== FIRST RECORDS ===")
    for i, rec in enumerate(records[: args.max_records]):
        print(f"\n--- RECORD {i} ---")
        if isinstance(rec, dict):
            print("KEYS:", list(rec.keys()))
        pprint(rec, width=140, sort_dicts=False)

    print("\n=== UNIQUE NESTED KEYS (sampled) ===")
    for k in sorted(walk_keys(data))[:300]:
        print(k)

    flat_boxes = [r for r in records if _is_flat_bbox_record(r)]
    if flat_boxes:
        global_x0 = min(float(r["x1"]) for r in flat_boxes)
        global_x1 = max(float(r["x2"]) for r in flat_boxes)
        global_y0 = min(float(r["y1"]) for r in flat_boxes)
        global_y1 = max(float(r["y2"]) for r in flat_boxes)
        print("\n=== PAGE BBOX EXTENT ===")
        print(f"x=[{global_x0:.1f}, {global_x1:.1f}] width={global_x1-global_x0:.1f}")
        print(f"y=[{global_y0:.1f}, {global_y1:.1f}] height={global_y1-global_y0:.1f}")

    line_match = LINE_RE.search(image.stem)
    line_idx = int(line_match.group(1)) if line_match else None
    groups = _cluster_page_boxes(records)
    print("\n=== INFERRED PAGE TEXT LINES (top to bottom) ===")
    print("line_count:", len(groups))
    for i, group in enumerate(groups, start=1):
        summary = _group_summary(group)
        marker = "  <== TARGET" if line_idx == i else ""
        print(
            f"line_{i:02d}: n={summary['count']} cy={summary['mean_cy']:.1f} "
            f"bbox=({summary['x0']:.0f},{summary['y0']:.0f})-({summary['x1']:.0f},{summary['y1']:.0f}) "
            f"size={summary['width']:.0f}x{summary['height']:.0f}{marker}"
        )
        print("  RTL text:", summary["rtl_text"])

    if line_idx is not None and 1 <= line_idx <= len(groups):
        target = _group_summary(groups[line_idx - 1])
        print("\n=== TARGET LINE GEOMETRY ===")
        print("target line index:", line_idx)
        pprint(target, sort_dicts=False)
        if image.is_file():
            with Image.open(image) as opened:
                iw, ih = opened.size
            print("line image width:", iw)
            print("page max bbox x2:", max(float(r["x2"]) for r in flat_boxes) if flat_boxes else None)
            print("target bbox width:", target["width"])
            print("line image height:", ih)
            print(
                "horizontal hint:",
                "likely page-width/uncropped" if flat_boxes and iw >= max(float(r["x2"]) for r in flat_boxes) - 5
                else "likely horizontally cropped; x-offset/scale must be recovered",
            )

    print("\n=== SAME-STEM / LINE-INDEX FILES UNDER SIDE ===")
    stem_needles = {image.stem.lower()}
    if line_idx is not None:
        stem_needles |= {f"line_{line_idx:02d}", f"line_{line_idx}"}
    shown = 0
    for item in side.rglob("*"):
        if not item.is_file():
            continue
        lowered = item.stem.lower()
        if any(n in lowered for n in stem_needles):
            print(item.relative_to(side))
            shown += 1
            if shown >= 40:
                break

    print("\n=== DEBUG DIRECTORY ===")
    debug_dir = bbox.parent
    for item in sorted(debug_dir.iterdir()):
        print(item.name)


if __name__ == "__main__":
    main()
