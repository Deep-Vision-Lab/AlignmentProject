"""Deterministic offline Hugging Face model resolution.

The cluster runs with outgoing Hugging Face traffic disabled. Historically the
project only checked whether a cache contained model config/weights, then passed
an HF model id back to Transformers. That can still fail when the selected
snapshot is missing tokenizer assets or when multiple cache roots exist.

This module resolves one complete snapshot directory (config + model weights +
tokenizer assets) and exposes that exact path to every rank.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable

_WEIGHT_PATTERNS = ("model*.safetensors", "pytorch_model*.bin")
_TOKENIZER_FILES = ("tokenizer.json", "vocab.txt", "spiece.model", "sentencepiece.bpe.model")

@dataclass(frozen=True)
class HFModelResolution:
    model_id: str
    snapshot_path: Path
    cache_root: Path

def _has_any_pattern(path: Path, patterns: Iterable[str]) -> bool:
    return any(any(path.glob(pattern)) for pattern in patterns)

def is_complete_snapshot(path: Path) -> bool:
    path = Path(path).expanduser()
    return (
        path.is_dir()
        and (path / "config.json").is_file()
        and _has_any_pattern(path, _WEIGHT_PATTERNS)
        and any((path / name).is_file() for name in _TOKENIZER_FILES)
    )

def _candidate_roots(project_dir: Path | None) -> list[Path]:
    roots: list[Path] = []
    def add(value) -> None:
        if not value:
            return
        path = Path(value).expanduser()
        if path not in roots:
            roots.append(path)
    add(os.environ.get("HF_HOME"))
    add(os.environ.get("TRANSFORMERS_CACHE"))
    add(os.environ.get("TRAIN_SHARED_PROJECT_DIR") and Path(os.environ["TRAIN_SHARED_PROJECT_DIR"]) / ".hf_cache")
    add(os.environ.get("PROJECT_DIR") and Path(os.environ["PROJECT_DIR"]) / ".hf_cache")
    if project_dir is not None:
        project_dir = Path(project_dir).expanduser()
        add(project_dir / ".hf_cache")
        add(Path(str(project_dir) + "_clone") / ".hf_cache")
    add(Path.home() / ".cache" / "huggingface")
    return roots

def _snapshot_dirs(cache_root: Path, model_id: str) -> list[Path]:
    slug = "models--" + model_id.replace("/", "--")
    snapshots: list[Path] = []
    for layout in (cache_root, cache_root / "hub"):
        model_root = layout / slug
        snapshot_root = model_root / "snapshots"
        if not snapshot_root.is_dir():
            continue
        main_ref = model_root / "refs" / "main"
        if main_ref.is_file():
            try:
                commit = main_ref.read_text(encoding="utf-8").strip()
            except OSError:
                commit = ""
            if commit and (snapshot_root / commit).is_dir():
                snapshots.append(snapshot_root / commit)
        others = [path for path in snapshot_root.iterdir() if path.is_dir()]
        others.sort(key=lambda path: path.stat().st_mtime, reverse=True)
        for path in others:
            if path not in snapshots:
                snapshots.append(path)
    return snapshots

def resolve_hf_model_snapshot(model_id: str, *, project_dir: str | Path | None = None) -> HFModelResolution:
    model_id = str(model_id).strip()
    if not model_id:
        raise RuntimeError("ARABIC_TEXT_MODEL_NAME is empty")
    direct = Path(model_id).expanduser()
    if direct.is_dir():
        direct = direct.resolve()
        if not is_complete_snapshot(direct):
            raise RuntimeError(f"Local Hugging Face model directory is incomplete: {direct}")
        return HFModelResolution(model_id=model_id, snapshot_path=direct, cache_root=direct.parent)
    checked: list[str] = []
    for root in _candidate_roots(Path(project_dir) if project_dir is not None else None):
        resolved_root = root.resolve() if root.exists() else root
        checked.append(str(resolved_root))
        if not root.is_dir():
            continue
        for snapshot in _snapshot_dirs(root, model_id):
            if is_complete_snapshot(snapshot):
                return HFModelResolution(model_id=model_id, snapshot_path=snapshot.resolve(), cache_root=resolved_root)
    raise RuntimeError(
        f"Could not find a COMPLETE offline Hugging Face snapshot for {model_id}. "
        f"Checked cache roots: {', '.join(checked)}. Expected config.json, model weights, and tokenizer assets."
    )

def install_resolved_hf_environment(model_id: str, *, project_dir: str | Path | None = None) -> HFModelResolution:
    resolution = resolve_hf_model_snapshot(model_id, project_dir=project_dir)
    os.environ["HF_HOME"] = str(resolution.cache_root)
    os.environ["ARABIC_TEXT_MODEL_RESOLVED_PATH"] = str(resolution.snapshot_path)
    os.environ.pop("TRANSFORMERS_CACHE", None)
    return resolution
