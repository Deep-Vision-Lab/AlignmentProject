#!/usr/bin/env python3
"""Select a pretrained checkpoint whose actual tensor keys match a model backend.

This runs on the login node before Slurm submission.  It does not trust a weight
folder's name or optional model_config metadata: when metadata is absent it
infers the visual backend from the saved image-model state-dict keys.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


def _load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _state_dict(payload) -> dict:
    if isinstance(payload, dict):
        for key in ("image_model_state_dict", "model_state_dict", "state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(isinstance(key, str) for key in payload):
            return payload
    return {}


def _strip_prefix(key: str) -> str:
    while key.startswith("module."):
        key = key[len("module.") :]
    return key


def _infer_from_keys(state: dict) -> str:
    keys = [_strip_prefix(str(key)) for key in state]
    has_vit = any(key.startswith("vit_encoder.") for key in keys)
    has_cnn = any(key.startswith("cnn_encoder.") for key in keys)
    has_bilstm = any(key.startswith("sequence_encoder.bilstm.") for key in keys)
    if has_vit and not (has_cnn or has_bilstm):
        return "vit"
    if (has_cnn or has_bilstm) and not has_vit:
        return "cnn_bilstm"
    if has_vit and (has_cnn or has_bilstm):
        return "mixed"
    return "unknown"


def _inspect(path: Path) -> tuple[str, str, int | None]:
    payload = _load(path)
    config = payload.get("model_config", {}) if isinstance(payload, dict) else {}
    metadata_backend = str(
        config.get("model_backend", config.get("visual_encoder_type", ""))
    ).strip().lower()
    state_backend = _infer_from_keys(_state_dict(payload))
    window = config.get("window_size") if isinstance(config, dict) else None
    try:
        window = int(window) if window is not None else None
    except (TypeError, ValueError):
        window = None
    return metadata_backend, state_backend, window


def _compatible(path: Path, expected: str, window: int) -> tuple[bool, str]:
    try:
        metadata_backend, state_backend, checkpoint_window = _inspect(path)
    except Exception as exc:  # diagnostics only; never select an unreadable file
        return False, f"unreadable: {type(exc).__name__}: {exc}"

    if state_backend not in {"unknown", expected}:
        return False, f"state_dict={state_backend}"
    if state_backend == "unknown" and metadata_backend and metadata_backend != expected:
        return False, f"metadata={metadata_backend}"
    if state_backend == "unknown" and not metadata_backend:
        return False, "backend=unknown"
    if metadata_backend and metadata_backend != expected:
        return False, f"metadata={metadata_backend}, state_dict={state_backend}"
    if checkpoint_window is not None and checkpoint_window != window:
        return False, f"window={checkpoint_window}"

    detected = state_backend if state_backend != "unknown" else metadata_backend
    return True, f"backend={detected}, window={checkpoint_window or '<missing>'}"


def _candidate_paths(weights_root: Path, backend: str, preferred_dir: str) -> list[Path]:
    names = ("model_best.pth", "model_latest.pth", "checkpoint_latest.pth")
    candidates: list[Path] = []

    if preferred_dir:
        preferred = weights_root / preferred_dir
        candidates.extend(preferred / name for name in names)

    pattern = "*vit*" if backend == "vit" else "*cnn*"
    other_dirs = sorted(
        (path for path in weights_root.glob(pattern) if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for directory in other_dirs:
        if preferred_dir and directory.name == preferred_dir:
            continue
        if "real_aug10k" in directory.name.lower():
            continue
        candidates.extend(directory / name for name in names)

    seen: set[Path] = set()
    output: list[Path] = []
    for candidate in candidates:
        candidate = candidate.expanduser().resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        output.append(candidate)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--backend", choices=("vit", "cnn_bilstm"), required=True)
    parser.add_argument("--window-size", type=int, default=32)
    parser.add_argument("--preferred-dir", default="")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    expected = args.backend
    if args.checkpoint is not None:
        path = args.checkpoint.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"Checkpoint not found: {path}")
        ok, detail = _compatible(path, expected, args.window_size)
        print(f"checkpoint_check path={path} expected={expected} {detail}", file=sys.stderr)
        if not ok:
            raise SystemExit(
                f"Checkpoint is incompatible with backend {expected}: {path} ({detail})"
            )
        print(path)
        return

    candidates = _candidate_paths(
        args.weights_root.expanduser().resolve(), expected, args.preferred_dir
    )
    if not candidates:
        raise SystemExit(
            f"No pretrained checkpoint candidates found for backend {expected} under "
            f"{args.weights_root.expanduser().resolve()}"
        )

    rejected: list[tuple[Path, str]] = []
    for path in candidates:
        ok, detail = _compatible(path, expected, args.window_size)
        if ok:
            print(
                f"checkpoint_selected path={path} expected={expected} {detail}",
                file=sys.stderr,
            )
            print(path)
            return
        rejected.append((path, detail))

    print(
        f"No compatible {expected} checkpoint was found. Inspected:",
        file=sys.stderr,
    )
    for path, detail in rejected:
        print(f"  {path}: {detail}", file=sys.stderr)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
