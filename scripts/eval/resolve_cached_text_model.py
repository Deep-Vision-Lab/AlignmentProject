#!/usr/bin/env python3
"""Resolve the local Hugging Face snapshot required by an evaluation checkpoint.

The improve_neg evaluation scripts run offline. This helper reads the checkpoint
configuration, determines the text backbone, and searches common cache locations,
including the sibling AlignmentProject_clone cache used during training.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--project-dir", type=Path, required=True)
    parser.add_argument("--model-name", default=None)
    return parser.parse_args(argv)


def checkpoint_metadata(weights: Path) -> tuple[str, str]:
    loaded = torch.load(weights, map_location="cpu")
    if not isinstance(loaded, Mapping):
        return "char", ""
    cfg = dict(loaded.get("model_config") or {})
    encoder_type = str(
        loaded.get("text_encoder_type")
        or cfg.get("text_encoder_type")
        or "arabic_span"
    ).lower()
    model_name = str(
        cfg.get("arabic_text_model_name")
        or loaded.get("arabic_text_model_name")
        or "aubmindlab/bert-base-arabertv02"
    )
    return encoder_type, model_name


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.expanduser().resolve()
        key = str(resolved)
        if key not in seen:
            result.append(resolved)
            seen.add(key)
    return result


def valid_snapshot(path: Path) -> bool:
    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    has_weights = any(
        (path / name).is_file()
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "tf_model.h5",
            "flax_model.msgpack",
        )
    )
    has_tokenizer = any(
        (path / name).is_file()
        for name in (
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
            "sentencepiece.bpe.model",
            "spiece.model",
        )
    )
    return has_weights and has_tokenizer


def candidate_homes(project_dir: Path) -> list[Path]:
    env_home = os.environ.get("HF_HOME")
    env_transformers = os.environ.get("TRANSFORMERS_CACHE")
    sibling_clone = project_dir.parent / f"{project_dir.name}_clone" / ".hf_cache"
    candidates = []
    if env_home:
        candidates.append(Path(env_home))
    candidates.extend(
        [
            project_dir / ".hf_cache",
            sibling_clone,
            Path.home() / ".cache" / "huggingface",
        ]
    )
    if env_transformers:
        candidates.append(Path(env_transformers))
    return unique_paths(candidates)


def snapshots_for_model(home: Path, model_name: str) -> list[Path]:
    slug = "models--" + model_name.replace("/", "--")
    roots = [home / "hub" / slug, home / slug]
    matches: list[Path] = []
    for root in roots:
        snapshots = root / "snapshots"
        if snapshots.is_dir():
            matches.extend(path for path in snapshots.iterdir() if path.is_dir())
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    weights = args.weights.expanduser().resolve()
    project_dir = args.project_dir.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")

    encoder_type, checkpoint_model_name = checkpoint_metadata(weights)
    if encoder_type == "char":
        print("")
        return 0

    requested = str(args.model_name or checkpoint_model_name).strip()
    direct = Path(requested).expanduser()
    if direct.is_dir() and valid_snapshot(direct.resolve()):
        resolved = direct.resolve()
        print(f"Using explicit local text model: {resolved}", file=sys.stderr)
        print(str(resolved))
        return 0

    searched: list[Path] = []
    for home in candidate_homes(project_dir):
        for snapshot in snapshots_for_model(home, requested):
            searched.append(snapshot)
            if valid_snapshot(snapshot):
                print(
                    f"Resolved cached text model {requested!r}: {snapshot}",
                    file=sys.stderr,
                )
                print(str(snapshot))
                return 0

    homes = candidate_homes(project_dir)
    rendered_homes = "\n  - ".join(str(path) for path in homes)
    rendered_snapshots = (
        "\n  - ".join(str(path) for path in searched)
        if searched
        else "(no matching snapshot directories found)"
    )
    raise RuntimeError(
        "The checkpoint requires a Hugging Face text backbone, but no complete "
        f"offline snapshot was found for {requested!r}.\n"
        f"Searched cache homes:\n  - {rendered_homes}\n"
        f"Matching snapshots inspected:\n  - {rendered_snapshots}\n\n"
        "Copy the cache from the training checkout, for example:\n"
        f"  rsync -avh --progress {project_dir.parent}/{project_dir.name}_clone/.hf_cache/ "
        f"{project_dir}/.hf_cache/"
    )


if __name__ == "__main__":
    raise SystemExit(main())
