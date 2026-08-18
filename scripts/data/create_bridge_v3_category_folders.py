#!/usr/bin/env python3
"""Create human-readable root-level real/positive/negative folders for Bridge V3."""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _link_or_copy(src: Path, dst: Path) -> None:
    if not src.is_file():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _resolve(root: Path, value: str) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    args = parser.parse_args()

    root = Path(args.data_dir).expanduser().resolve()
    index_path = root / "anchor_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"Missing anchor index: {index_path}")

    anchors = [
        json.loads(line)
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    for category in ("real", "positive", "negative"):
        (root / category).mkdir(parents=True, exist_ok=True)

    for row in anchors:
        anchor_id = str(row["anchor_id"])

        real = row["real"]
        real_dir = root / "real" / anchor_id
        real_image_src = _resolve(root, real["image"])
        real_text_src = _resolve(root, real["text"])
        _link_or_copy(real_image_src, real_dir / f"real{real_image_src.suffix.lower() or '.png'}")
        _link_or_copy(real_text_src, real_dir / "real.txt")

        original_value = str(real.get("original_image") or "")
        if original_value:
            original_src = _resolve(root, original_value)
            _link_or_copy(original_src, real_dir / f"real_original{original_src.suffix.lower() or '.png'}")

        variants = list(real.get("augmentation_variants") or [])
        for variant in variants:
            src = _resolve(root, variant["path"])
            _link_or_copy(src, real_dir / src.name)

        if real.get("appearance_augmentation"):
            (real_dir / "augmentation.json").write_text(
                json.dumps(real["appearance_augmentation"], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        positive = row["positive"]
        pos_dir = root / "positive" / anchor_id
        _link_or_copy(_resolve(root, positive["image"]), pos_dir / "positive.png")
        _link_or_copy(_resolve(root, positive["text"]), pos_dir / "positive.txt")
        _link_or_copy(_resolve(root, positive["mask"]), pos_dir / "positive_mask.png")

        neg_dir = root / "negative" / anchor_id
        for neg in row.get("negatives", []):
            idx = int(neg["index"])
            _link_or_copy(_resolve(root, neg["image"]), neg_dir / f"negative_{idx:02d}.png")
            _link_or_copy(_resolve(root, neg["text"]), neg_dir / f"negative_{idx:02d}.txt")

    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["human_category_folders"] = True
    metadata["human_category_layout"] = {
        "real": "real/<anchor_id>/{real.*,real_original.*,real_binarized.png,real_gaussian_noise.png,real_gaussian_blur.png,real_blur_noise.png,real_binarized_noise.png,real_contrast_gamma.png,real_salt_pepper.png,real.txt,augmentation.json}",
        "positive": "positive/<anchor_id>/{positive.png,positive.txt,positive_mask.png,relation.json}",
        "negative": "negative/<anchor_id>/{negative_00..07.png,negative_00..07.txt,relations.json}",
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("=== BRIDGE V3 HUMAN CATEGORY FOLDERS ===")
    print(f"root={root}")
    print(f"anchors={len(anchors)}")
    print("folders=real,positive,negative")
    print("real_files=training real + original + 7 appearance variants + text + augmentation metadata")
    print("CATEGORY_FOLDERS=READY")


if __name__ == "__main__":
    main()
