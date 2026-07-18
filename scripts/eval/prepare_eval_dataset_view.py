#!/usr/bin/env python3
"""Prepare a common evaluation layout for synthetic or real line-pair data.

Existing improve_neg visualizers expect this 1-based layout::

    DATA_DIR/
      images/img1_1.png
      images/img2_1.png
      texts/text1_1.txt
      texts/text2_1.txt

Synthetic_Arabic already has that layout and is returned unchanged. For the real
ArabicDataset manifest, this utility creates a lightweight symlink/copy view with
sequential evaluation indices and writes ``view_manifest.jsonl`` for traceability.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve synthetic data or materialize a real-manifest evaluation view."
    )
    parser.add_argument("--dataset-type", choices=("synthetic", "real"), required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Results/Evaluation/dataset_views/real"),
        help="Generated view directory for real data.",
    )
    parser.add_argument("--manifest-name", default="dataset_manifest.jsonl")
    parser.add_argument("--text-key", default="text_original_path")
    parser.add_argument(
        "--labels",
        default="high_match,medium_match",
        help="Comma-separated real pair labels, or 'all'.",
    )
    parser.add_argument("--min-text-score", type=float, default=0.0)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument(
        "--link-mode",
        choices=("auto", "symlink", "copy"),
        default="auto",
    )
    return parser.parse_args(argv)


def _required_synthetic_paths(root: Path, index: int = 1) -> tuple[Path, ...]:
    return (
        root / "images" / f"img1_{index}.png",
        root / "images" / f"img2_{index}.png",
        root / "texts" / f"text1_{index}.txt",
        root / "texts" / f"text2_{index}.txt",
    )


def validate_synthetic(root: Path) -> Path:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Synthetic dataset directory not found: {root}")
    missing = [path for path in _required_synthetic_paths(root) if not path.is_file()]
    if missing:
        rendered = "\n  - ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Synthetic evaluation layout is incomplete. Missing:\n  - " + rendered
        )
    print(f"Using synthetic evaluation dataset: {root}", file=sys.stderr)
    return root


def parse_labels(value: str) -> set[str] | None:
    if value.strip().lower() in {"", "all", "*"}:
        return None
    labels = {item.strip() for item in value.split(",") if item.strip()}
    if not labels:
        raise ValueError("--labels did not contain any usable labels")
    return labels


def manifest_path(data_dir: Path, manifest_name: str) -> tuple[Path, Path]:
    source = data_dir.expanduser().resolve()
    if source.is_file():
        return source, source.parent
    manifest = source / manifest_name
    if not manifest.is_file():
        raise FileNotFoundError(f"Real dataset manifest not found: {manifest}")
    return manifest, source


def candidate_paths(dataset_root: Path, raw_value: object) -> Iterable[Path]:
    value = Path(str(raw_value)).expanduser()
    if value.is_absolute():
        yield value
        return
    yield dataset_root / value
    yield Path.cwd() / value
    yield dataset_root.parent / value


def resolve_manifest_path(dataset_root: Path, raw_value: object) -> Path:
    tried: list[Path] = []
    for candidate in candidate_paths(dataset_root, raw_value):
        candidate = candidate.resolve()
        tried.append(candidate)
        if candidate.is_file():
            return candidate
    rendered = "\n  - ".join(str(path) for path in tried)
    raise FileNotFoundError(
        f"Could not resolve manifest path {raw_value!r}. Tried:\n  - {rendered}"
    )


def load_rows(
    manifest: Path,
    labels: set[str] | None,
    min_text_score: float,
    max_samples: int,
    text_key: str,
) -> list[dict]:
    selected: list[dict] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {manifest} at line {line_number}: {exc}"
                ) from exc

            label = str(row.get("label_type", ""))
            if labels is not None and label not in labels:
                continue
            text_score = float((row.get("scores") or {}).get("text_score", 0.0))
            if text_score < min_text_score:
                continue

            for side_name in ("A", "B"):
                side = row.get(side_name)
                if not isinstance(side, Mapping):
                    raise KeyError(
                        f"Manifest line {line_number} has no dictionary side {side_name!r}"
                    )
                for key in ("line_image_path", text_key):
                    if not side.get(key):
                        raise KeyError(
                            f"Manifest line {line_number}, side {side_name}, missing {key!r}"
                        )

            row["_manifest_line"] = line_number
            selected.append(row)
            if max_samples > 0 and len(selected) >= max_samples:
                break

    if not selected:
        rendered_labels = "all" if labels is None else ",".join(sorted(labels))
        raise ValueError(
            "No real evaluation pairs remain after filtering: "
            f"labels={rendered_labels}, min_text_score={min_text_score}"
        )
    return selected


def reset_view(output_dir: Path) -> tuple[Path, Path]:
    output_dir = output_dir.expanduser().resolve()
    images_dir = output_dir / "images"
    texts_dir = output_dir / "texts"
    for directory in (images_dir, texts_dir):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    manifest = output_dir / "view_manifest.jsonl"
    if manifest.exists():
        manifest.unlink()
    return images_dir, texts_dir


def link_or_copy(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()

    if mode in {"auto", "symlink"}:
        try:
            destination.symlink_to(source)
            return "symlink"
        except OSError:
            if mode == "symlink":
                raise
    shutil.copy2(source, destination)
    return "copy"


def materialize_real(args: argparse.Namespace) -> Path:
    manifest, dataset_root = manifest_path(args.data_dir, args.manifest_name)
    labels = parse_labels(args.labels)
    rows = load_rows(
        manifest=manifest,
        labels=labels,
        min_text_score=args.min_text_score,
        max_samples=max(0, int(args.max_samples)),
        text_key=args.text_key,
    )
    images_dir, texts_dir = reset_view(args.output_dir)
    output_dir = images_dir.parent
    records: list[dict] = []
    modes: set[str] = set()

    for eval_index, row in enumerate(rows, start=1):
        sides = {}
        for side_number, side_name in enumerate(("A", "B"), start=1):
            side = row[side_name]
            image_source = resolve_manifest_path(dataset_root, side["line_image_path"])
            text_source = resolve_manifest_path(dataset_root, side[args.text_key])
            image_suffix = image_source.suffix.lower() or ".png"
            expected_image_dest = images_dir / f"img{side_number}_{eval_index}.png"
            if image_suffix == ".png":
                modes.add(link_or_copy(image_source, expected_image_dest, args.link_mode))
            else:
                from PIL import Image

                with Image.open(image_source) as image:
                    image.convert("RGB").save(expected_image_dest)
                modes.add("convert")
            text_dest = texts_dir / f"text{side_number}_{eval_index}.txt"
            modes.add(link_or_copy(text_source, text_dest, args.link_mode))
            sides[side_name] = {
                "image_source": str(image_source),
                "text_source": str(text_source),
                "line_idx": side.get("line_idx"),
            }

        scores = row.get("scores") or {}
        records.append(
            {
                "eval_index": eval_index,
                "pair_id": str(row.get("pair_id", eval_index)),
                "label_type": str(row.get("label_type", "")),
                "text_score": float(scores.get("text_score", 0.0)),
                "manifest_line": int(row["_manifest_line"]),
                "A": sides["A"],
                "B": sides["B"],
            }
        )

    view_manifest = output_dir / "view_manifest.jsonl"
    with view_manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "dataset_type": "real",
        "source_manifest": str(manifest),
        "text_key": args.text_key,
        "labels": "all" if labels is None else sorted(labels),
        "min_text_score": args.min_text_score,
        "samples": len(records),
        "materialization_modes": sorted(modes),
    }
    (output_dir / "view_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Prepared real evaluation view: samples={len(records)} path={output_dir}",
        file=sys.stderr,
    )
    print(f"Index mapping: {view_manifest}", file=sys.stderr)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dataset_type == "synthetic":
        resolved = validate_synthetic(args.data_dir)
    else:
        resolved = materialize_real(args)
    print(str(resolved))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
