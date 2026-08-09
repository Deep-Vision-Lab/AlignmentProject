#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from pprint import pprint

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


def flatten_scalars(value, prefix=""):
    rows = []
    if isinstance(value, dict):
        for k, v in value.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            rows.extend(flatten_scalars(v, path))
    elif isinstance(value, list):
        if value and all(not isinstance(x, (dict, list)) for x in value):
            rows.append((prefix, value))
        else:
            for i, v in enumerate(value[:20]):
                rows.extend(flatten_scalars(v, f"{prefix}[{i}]"))
    else:
        rows.append((prefix, value))
    return rows


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

    line_match = LINE_RE.search(image.stem)
    line_idx = int(line_match.group(1)) if line_match else None
    needles = {image.name.lower(), image.stem.lower()}
    if line_idx is not None:
        needles |= {str(line_idx), f"{line_idx:02d}", f"line_{line_idx}", f"line_{line_idx:02d}"}
    text_tokens = set(ARABIC_RE.findall(text)) if text else set()

    print("\n=== RECORDS THAT LOOK RELATED TO TARGET LINE OR TEXT ===")
    hits = 0
    for i, rec in enumerate(records):
        scalars = flatten_scalars(rec)
        rendered = " ".join(str(v) for _, v in scalars if v is not None).lower()
        line_hit = any(n in rendered for n in needles if len(n) >= 2)
        token_hit = any(tok in rendered for tok in text_tokens if len(tok) >= 2)
        if line_hit or token_hit:
            print(f"\n--- HIT {i} line_hit={line_hit} token_hit={token_hit} ---")
            pprint(rec, width=140, sort_dicts=False)
            hits += 1
            if hits >= 20:
                break
    if hits == 0:
        print("NO MATCHING RECORDS FOUND")

    print("\n=== DEBUG DIRECTORY ===")
    debug_dir = bbox.parent
    for item in sorted(debug_dir.iterdir()):
        print(item.name)


if __name__ == "__main__":
    main()
