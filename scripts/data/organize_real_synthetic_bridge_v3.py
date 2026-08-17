#!/usr/bin/env python3
"""Organize RealSyntheticBridge V3 into a human-readable, self-contained layout."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

LAYOUT_VERSION = 2


def resolve(root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def copy_once(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() == dst.resolve():
        return
    if not dst.exists():
        shutil.copy2(src, dst)


def move_generated(root: Path, old_value: str, new_rel: Path) -> str:
    src = resolve(root, old_value)
    dst = root / new_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        if dst.exists():
            dst.unlink()
        shutil.move(str(src), str(dst))
    return new_rel.as_posix()


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_readme(root: Path) -> None:
    text = """# RealSyntheticBridge V3 dataset layout

The dataset is grouped by **real anchor**. The same `<anchor_id>` connects the real
line, its positive synthetic sentence/mask, and all of its negative sentences.

```text
RealSyntheticBridge_v3/
├── images/
│   ├── real/<anchor_id>/real.png
│   ├── positive/<anchor_id>/positive.png
│   └── negative/<anchor_id>/negative_00.png ... negative_07.png
├── texts/
│   ├── real/<anchor_id>/real.txt
│   ├── positive/<anchor_id>/positive.txt
│   └── negative/<anchor_id>/negative_00.txt ... negative_07.txt
├── masks/positive/<anchor_id>/positive_mask.png
├── anchors/<anchor_id>/anchor.json
├── positive/<anchor_id>/relation.json
├── negative/<anchor_id>/relations.json
├── anchor_index.jsonl
├── real_lines_index.jsonl
├── real_lines.csv
├── dataset_manifest.jsonl
└── metadata.json
```

## Scraping the real lines

The real manuscript lines are physically copied under `images/real/` and
`texts/real/`. `real_lines_index.jsonl` and `real_lines.csv` contain one row per real
anchor and point directly to those files. These index files are for inspection or
scraping; they do **not** add extra real-only rows to the training manifest.

## Training relationships

`dataset_manifest.jsonl` remains the machine-readable training source. For each
anchor it contains one positive relation plus the configured negative relations.
"""
    (root / "README_DATASET.md").write_text(text, encoding="utf-8")


def organize(root: Path, *, force: bool = False) -> None:
    manifest_path = root / "dataset_manifest.jsonl"
    metadata_path = root / "metadata.json"
    if not manifest_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Missing Bridge V3 metadata/manifest under {root}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata.get("dataset_version", 0)) != 3:
        raise RuntimeError(f"Expected dataset_version=3, got {metadata.get('dataset_version')}")

    rows = [json.loads(raw) for raw in manifest_path.read_text(encoding="utf-8").splitlines() if raw.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        anchor_id = str((row.get("bridge") or {}).get("anchor_id") or "")
        if not anchor_id:
            raise RuntimeError(f"Manifest row missing bridge.anchor_id: {row.get('pair_id')}")
        groups[anchor_id].append(row)

    rewritten: list[dict] = []
    anchor_index: list[dict] = []
    real_index: list[dict] = []

    for anchor_id in sorted(groups):
        group_rows = groups[anchor_id]
        positives = [r for r in group_rows if r.get("label_type") == "medium_match"]
        negatives = [r for r in group_rows if r.get("label_type") == "no_shared_content"]
        if len(positives) != 1:
            raise RuntimeError(f"{anchor_id}: expected exactly one positive, found {len(positives)}")
        positive = positives[0]
        pair_id = str(positive["pair_id"])
        page_id = str(positive.get("A_page_id", ""))

        real_image_src = resolve(root, positive["A"]["line_image_path"])
        real_text_src = resolve(root, positive["A"]["text_original_path"])
        if not real_image_src.is_file() or not real_text_src.is_file():
            raise FileNotFoundError(f"{anchor_id}: missing source real anchor")
        image_suffix = real_image_src.suffix.lower() or ".png"
        real_image_rel = Path("images") / "real" / anchor_id / f"real{image_suffix}"
        real_text_rel = Path("texts") / "real" / anchor_id / "real.txt"
        copy_once(real_image_src, root / real_image_rel)
        copy_once(real_text_src, root / real_text_rel)

        pos_image_rel = Path("images") / "positive" / anchor_id / "positive.png"
        pos_text_rel = Path("texts") / "positive" / anchor_id / "positive.txt"
        pos_mask_rel = Path("masks") / "positive" / anchor_id / "positive_mask.png"
        positive["B"]["line_image_path"] = move_generated(root, positive["B"]["line_image_path"], pos_image_rel)
        positive["B"]["text_original_path"] = move_generated(root, positive["B"]["text_original_path"], pos_text_rel)
        mask_old = positive["B"].get("alignment_mask_path") or (positive.get("bridge") or {}).get("alignment_mask_path")
        if not mask_old:
            raise RuntimeError(f"{anchor_id}: positive has no mask")
        positive["B"]["alignment_mask_path"] = move_generated(root, str(mask_old), pos_mask_rel)
        positive["bridge"]["alignment_mask_path"] = pos_mask_rel.as_posix()
        positive["A"]["line_image_path"] = real_image_rel.as_posix()
        positive["A"]["text_original_path"] = real_text_rel.as_posix()
        positive["bridge"]["layout_version"] = LAYOUT_VERSION
        positive["bridge"]["group_path"] = (Path("anchors") / anchor_id / "anchor.json").as_posix()

        negative_entries: list[dict] = []
        for neg_index, negative in enumerate(sorted(negatives, key=lambda r: str(r.get("B_page_id", "")))):
            neg_image_rel = Path("images") / "negative" / anchor_id / f"negative_{neg_index:02d}.png"
            neg_text_rel = Path("texts") / "negative" / anchor_id / f"negative_{neg_index:02d}.txt"
            negative["B"]["line_image_path"] = move_generated(root, negative["B"]["line_image_path"], neg_image_rel)
            negative["B"]["text_original_path"] = move_generated(root, negative["B"]["text_original_path"], neg_text_rel)
            negative["A"]["line_image_path"] = real_image_rel.as_posix()
            negative["A"]["text_original_path"] = real_text_rel.as_posix()
            negative["bridge"]["layout_version"] = LAYOUT_VERSION
            negative["bridge"]["negative_index"] = neg_index
            negative["bridge"]["group_path"] = (Path("anchors") / anchor_id / "anchor.json").as_posix()
            negative_entries.append({
                "index": neg_index,
                "image": neg_image_rel.as_posix(),
                "text": neg_text_rel.as_posix(),
                "fonts": negative["bridge"].get("fonts", []),
                "appearance_augmentation": negative["bridge"].get("appearance_augmentation", {}),
            })

        positive_entry = {
            "image": pos_image_rel.as_posix(),
            "text": pos_text_rel.as_posix(),
            "mask": pos_mask_rel.as_posix(),
            "shared_island_count": positive["bridge"].get("shared_island_count"),
            "shared_texts": positive["bridge"].get("shared_texts", []),
            "shared_boxes_px": positive["bridge"].get("shared_boxes_px", []),
            "fonts": positive["bridge"].get("fonts", []),
            "appearance_augmentation": positive["bridge"].get("appearance_augmentation", {}),
        }
        real_entry = {"page_id": page_id, "image": real_image_rel.as_posix(), "text": real_text_rel.as_posix()}
        anchor_record = {
            "anchor_id": anchor_id,
            "pair_id": pair_id,
            "real": real_entry,
            "positive": positive_entry,
            "negatives": negative_entries,
        }
        write_json(root / "anchors" / anchor_id / "anchor.json", anchor_record)
        write_json(root / "positive" / anchor_id / "relation.json", {
            "anchor_id": anchor_id, "pair_id": pair_id, "real": real_entry, "positive": positive_entry,
        })
        write_json(root / "negative" / anchor_id / "relations.json", {
            "anchor_id": anchor_id, "pair_id": pair_id, "real": real_entry, "negatives": negative_entries,
        })
        anchor_index.append(anchor_record)
        real_index.append({
            "anchor_id": anchor_id,
            "page_id": page_id,
            "image": real_image_rel.as_posix(),
            "text": real_text_rel.as_posix(),
            "excluded_from_real_only_training_rows": True,
        })
        rewritten.append(positive)
        rewritten.extend(sorted(negatives, key=lambda r: int((r.get("bridge") or {}).get("negative_index", 0))))

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rewritten:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (root / "anchor_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in anchor_index:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (root / "real_lines_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in real_index:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (root / "real_lines.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["anchor_id", "page_id", "image", "text"])
        writer.writeheader()
        for row in real_index:
            writer.writerow({key: row[key] for key in writer.fieldnames})

    for dirname in (root / "images", root / "texts", root / "masks"):
        for child in list(dirname.iterdir()):
            if child.is_file():
                child.unlink()

    metadata["layout_version"] = LAYOUT_VERSION
    metadata["layout_semantics"] = "anchor_grouped_self_contained_with_real_scrape_index"
    metadata["real_samples_copied"] = True
    metadata["relationship_key"] = "anchor_id"
    metadata["real_lines_index"] = "real_lines_index.jsonl"
    metadata["real_lines_csv"] = "real_lines.csv"
    metadata["folder_structure"] = {
        "real_images": "images/real/<anchor_id>/real.*",
        "positive_images": "images/positive/<anchor_id>/positive.png",
        "negative_images": "images/negative/<anchor_id>/negative_XX.png",
        "real_texts": "texts/real/<anchor_id>/real.txt",
        "positive_texts": "texts/positive/<anchor_id>/positive.txt",
        "negative_texts": "texts/negative/<anchor_id>/negative_XX.txt",
        "positive_masks": "masks/positive/<anchor_id>/positive_mask.png",
        "anchor_record": "anchors/<anchor_id>/anchor.json",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    build_readme(root)

    print("=== BRIDGE V3 DATASET ORGANIZED ===")
    print(f"root={root}")
    print(f"layout_version={LAYOUT_VERSION}")
    print(f"anchors={len(anchor_index)}")
    print(f"real_lines={len(real_index)}")
    print("relationship_key=anchor_id")
    print("readme=README_DATASET.md")
    print("anchor_index=anchor_index.jsonl")
    print("real_lines_index=real_lines_index.jsonl")
    print("real_lines_csv=real_lines.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    organize(Path(args.data_dir).expanduser().resolve(), force=args.force)


if __name__ == "__main__":
    main()
