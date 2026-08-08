#!/usr/bin/env python3
"""Evaluate deliberately unaligned line pairs and visualize false alignment masks.

Synthetic negatives are cross-pairs drawn only from the fixed-63 held-out test
split. Candidate transcripts are rejected when they share a meaningful exact
word, two-word phrase, or long compact character substring. Real negatives come
from manifest rows labelled ``no_shared_content`` and use a pair-id-safe test
split.

Prediction intentionally reuses the normal aligned-data policy:
  * synthetic: component-aware Needleman-Wunsch v3-local regions;
  * real: the current ink-aware Smith-Waterman local region.

Therefore a red rectangle in an output image is a real false-positive mask under
the same inference rule used for aligned evaluation; no special negative-only
rejection threshold is introduced here.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import re
import sys
import tempfile
import unicodedata
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_line_geometry import install_evaluation_geometry
install_evaluation_geometry()

try:
    from Evaluation.vit_evaluation import install_vit_evaluation_loader
    install_vit_evaluation_loader()
except ImportError:
    pass

from Evaluation.zero_shot_sw import (
    display_image as real_display_image,
    ink_aware_match_scores,
    install_dataset_patches,
)
install_dataset_patches()

from Evaluation._eval_utils import (
    compute_similarity,
    get_image_features,
    load_evaluation_models,
    needleman_wunsch,
    patch_range_to_pixels,
)
from Evaluation.sw_core import build_match_scores, resolve_score_mode, smith_waterman
from Evaluation.sw_dataset import load_arabic_dataset_pairs

_ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_TOKEN = re.compile(r"[\u0600-\u06ff]+")


def _normalise_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = _ARABIC_DIACRITICS.sub("", text.replace("ـ", ""))
    return " ".join(_TOKEN.findall(text))


def _tokens(text: str) -> list[str]:
    return [token for token in _normalise_text(text).split() if token]


def _compact_ngrams(text: str, size: int) -> set[str]:
    compact = "".join(_tokens(text))
    if size <= 0 or len(compact) < size:
        return set()
    return {compact[index : index + size] for index in range(len(compact) - size + 1)}


def _strictly_unaligned_text(
    left: str,
    right: str,
    *,
    min_word_chars: int,
    min_compact_chars: int,
) -> tuple[bool, dict]:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    long_left = {token for token in left_tokens if len(token) >= min_word_chars}
    long_right = {token for token in right_tokens if len(token) >= min_word_chars}
    shared_words = sorted(long_left & long_right)

    left_bigrams = {
        " ".join(left_tokens[index : index + 2])
        for index in range(max(0, len(left_tokens) - 1))
    }
    right_bigrams = {
        " ".join(right_tokens[index : index + 2])
        for index in range(max(0, len(right_tokens) - 1))
    }
    shared_bigrams = sorted(left_bigrams & right_bigrams)
    shared_compact = sorted(
        _compact_ngrams(left, min_compact_chars)
        & _compact_ngrams(right, min_compact_chars)
    )
    ok = not shared_words and not shared_bigrams and not shared_compact
    return ok, {
        "shared_words": shared_words,
        "shared_bigrams": shared_bigrams,
        "shared_compact_ngrams": shared_compact[:20],
    }


def _synthetic_paths(root: Path, role: int, index: int) -> tuple[Path, Path]:
    image_candidates = [
        root / "images" / f"img{role}_{index}.png",
        root / f"img{role}_{index}.png",
    ]
    text_candidates = [
        root / "texts" / f"text{role}_{index}.txt",
        root / "texts" / f"txt{role}_{index}.txt",
        root / f"text{role}_{index}.txt",
    ]
    image = next((path for path in image_candidates if path.is_file()), image_candidates[0])
    text = next((path for path in text_candidates if path.is_file()), text_candidates[0])
    if not image.is_file():
        raise FileNotFoundError(f"Synthetic image not found for role={role}, index={index}: {image}")
    if not text.is_file():
        raise FileNotFoundError(f"Synthetic transcript not found for role={role}, index={index}: {text}")
    return image, text


def _fixed63_test_indices(total: int, seed: int) -> list[int]:
    generator = torch.Generator().manual_seed(int(seed))
    permutation = torch.randperm(int(total), generator=generator).tolist()
    train_size = int(0.60 * total)
    valid_size = int(0.20 * total)
    # Dataset files are one-indexed.
    return [int(value) + 1 for value in permutation[train_size + valid_size :]]


def _select_synthetic_negatives(args) -> list[dict]:
    root = Path(args.data_dir).expanduser().resolve()
    indices = _fixed63_test_indices(args.num_samples, args.split_seed)
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    selected = []
    used_pairs = set()

    for source_a in indices:
        image1, text1_path = _synthetic_paths(root, 1, source_a)
        text1 = text1_path.read_text(encoding="utf-8").strip()
        candidates = list(indices)
        rng.shuffle(candidates)
        for source_b in candidates:
            if source_b == source_a or (source_a, source_b) in used_pairs:
                continue
            image2, text2_path = _synthetic_paths(root, 2, source_b)
            text2 = text2_path.read_text(encoding="utf-8").strip()
            clean, diagnostics = _strictly_unaligned_text(
                text1,
                text2,
                min_word_chars=args.min_shared_word_chars,
                min_compact_chars=args.min_common_compact_chars,
            )
            if not clean:
                continue
            selected.append(
                {
                    "image1": image1,
                    "image2": image2,
                    "text1": text1,
                    "text2": text2,
                    "source_index1": source_a,
                    "source_index2": source_b,
                    "pair_id": f"synthetic_negative_{source_a}_{source_b}",
                    "label_type": "cross_pair_no_shared_text",
                    "selection_diagnostics": diagnostics,
                }
            )
            used_pairs.add((source_a, source_b))
            break
        if len(selected) >= args.n_samples:
            break

    if len(selected) < args.n_samples:
        raise RuntimeError(
            f"Could select only {len(selected)} strict synthetic negatives; "
            f"requested {args.n_samples}. Relax --min-common-compact-chars or "
            "--min-shared-word-chars only if necessary."
        )
    return selected


def _select_real_negatives(args) -> list[dict]:
    root = Path(args.data_dir).expanduser().resolve()
    manifest = Path(args.arabic_manifest or root / "dataset_manifest.jsonl")
    namespace = SimpleNamespace(
        arabic_manifest=str(manifest),
        data_dir=str(root),
        real_text_key=args.real_text_key,
        real_labels="no_shared_content",
        real_min_text_score=0.0,
        real_validate_paths=True,
        split_seed=args.split_seed,
        real_split=args.real_split,
    )
    pairs = load_arabic_dataset_pairs(namespace)
    start = max(0, args.start_index - 1)
    pairs = pairs[start : start + args.n_samples]
    if not pairs:
        raise RuntimeError(
            "No real no_shared_content samples were found in the requested split."
        )
    return [
        {
            "image1": Path(pair.image1),
            "image2": Path(pair.image2),
            "text1": "",
            "text2": "",
            "source_index1": pair.manifest_position,
            "source_index2": pair.manifest_position,
            "pair_id": pair.pair_id,
            "label_type": "no_shared_content",
            "selection_diagnostics": {},
        }
        for pair in pairs
    ]


def _extract_components(result, match_scores: np.ndarray, gap: float) -> list[dict]:
    # CNN+BiLSTM branches use the shared component module. The pure-ViT branch
    # has the same v3-local logic in vit_fixed63_nw_eval.py.
    try:
        from Evaluation.nw_component_regions import _supported_runs
        components, _score = _supported_runs(result.steps, match_scores, gap)
        return list(components)
    except ImportError:
        from Evaluation.vit_fixed63_nw_eval import extract_components
        return list(extract_components(result, match_scores))


def _prepare_features(models, image_path: Path, dataset_type: str):
    if dataset_type == "synthetic":
        with Image.open(image_path) as opened:
            display = np.asarray(opened.convert("RGB"))
        features = get_image_features(models, image_path, "synthetic")
        return display, features

    # Match the canonical real evaluation: aspect-preserving foreground crop,
    # binarization and normalization are applied before model feature extraction.
    display = real_display_image(image_path, "real")
    temporary = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    temporary.close()
    temp_path = Path(temporary.name)
    try:
        Image.fromarray(display).save(temp_path)
        features = get_image_features(models, temp_path, "synthetic")
    finally:
        temp_path.unlink(missing_ok=True)
    return display, features


def _component_intervals(components, axis: int, n_windows: int, width: int, use_flip: bool):
    intervals = []
    for component in components:
        path = list(component.get("path", ()))
        values = [int(pair[axis]) for pair in path]
        if not values:
            continue
        left, right = patch_range_to_pixels(
            min(values), max(values) + 1, n_windows, width, use_flip
        )
        lo, hi = sorted((float(left), float(right)))
        intervals.append((max(0.0, lo), min(float(width), hi)))
    return _merge_intervals(intervals)


def _sw_intervals(path, axis: int, n_windows: int, width: int, use_flip: bool):
    values = [int(pair[axis]) for pair in path]
    if not values:
        return []
    left, right = patch_range_to_pixels(
        min(values), max(values) + 1, n_windows, width, use_flip
    )
    lo, hi = sorted((float(left), float(right)))
    return [(max(0.0, lo), min(float(width), hi))]


def _merge_intervals(intervals):
    values = sorted((float(a), float(b)) for a, b in intervals if b > a)
    merged = []
    for left, right in values:
        if not merged or left > merged[-1][1]:
            merged.append([left, right])
        else:
            merged[-1][1] = max(merged[-1][1], right)
    return [(left, right) for left, right in merged]


def _coverage(intervals, width: int) -> float:
    return float(sum(right - left for left, right in _merge_intervals(intervals)) / max(1, width))


def _draw_intervals(ax, array, intervals):
    for left, right in intervals:
        ax.add_patch(
            Rectangle(
                (left, 1),
                max(2.0, right - left),
                max(2.0, array.shape[0] - 2),
                facecolor="red",
                edgecolor="red",
                alpha=0.30,
                linewidth=1.5,
            )
        )


def _save_visualization(
    output: Path,
    arr1,
    arr2,
    similarity,
    path,
    accepted_path,
    intervals1,
    intervals2,
    title: str,
    no_mask: bool,
):
    fig = plt.figure(figsize=(20, 12))
    grid = fig.add_gridspec(3, 1, height_ratios=[2, 2, 8], hspace=0.18)
    axes = [fig.add_subplot(grid[0]), fig.add_subplot(grid[1])]
    axes[0].imshow(arr1, aspect="auto")
    axes[1].imshow(arr2, aspect="auto")
    _draw_intervals(axes[0], arr1, intervals1)
    _draw_intervals(axes[1], arr2, intervals2)
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    status = "NO MASK (correct negative)" if no_mask else "FALSE MASK DETECTED"
    axes[0].set_title(f"line A | {status}")
    axes[1].set_title("line B | red = predicted alignment mask")

    ax = fig.add_subplot(grid[2])
    matrix = np.asarray(similarity, dtype=np.float32)
    image = ax.imshow(matrix, aspect="equal", origin="upper", vmin=-1.0, vmax=1.0,
                      cmap="coolwarm", interpolation="nearest")
    if path:
        ax.plot([col for row, col in path], [row for row, col in path],
                color="black", linewidth=1.2, marker=".", markersize=2.5,
                label="alignment path")
    if accepted_path:
        ax.scatter([col for row, col in accepted_path], [row for row, col in accepted_path],
                   s=26, facecolors="cyan", edgecolors="black", linewidths=0.5,
                   label="accepted mask evidence", zorder=6)
    if path or accepted_path:
        ax.legend(loc="upper left", fontsize=8)
    ax.set_xlabel("line B windows")
    ax.set_ylabel("line A windows")
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.015, label="cosine similarity")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _evaluate_one(models, record: dict, dataset_type: str, args, output: Path) -> dict:
    arr1, features1 = _prepare_features(models, record["image1"], dataset_type)
    arr2, features2 = _prepare_features(models, record["image2"], dataset_type)
    raw_similarity = compute_similarity(
        features1.select(args.feature), features2.select(args.feature)
    ).cpu().numpy()
    resolved_mode = resolve_score_mode(args.score_mode, dataset_type)
    match_scores = build_match_scores(
        raw_similarity, resolved_mode, args.score_clip, args.threshold
    )
    match_scores = ink_aware_match_scores(
        match_scores,
        features1.ink.detach().cpu().numpy(),
        features2.ink.detach().cpu().numpy(),
    )
    use_flip = bool(models.image_model.use_flip)

    if dataset_type == "synthetic":
        result = needleman_wunsch(match_scores, gap_penalty=args.gap, similarity_offset=0.0)
        components = _extract_components(result, match_scores, args.gap)
        accepted_path = [pair for component in components for pair in component.get("path", ())]
        global_path = [
            (int(step.index1), int(step.index2))
            for step in result.steps
            if step.index1 is not None and step.index2 is not None
        ]
        intervals1 = _component_intervals(
            components, 0, raw_similarity.shape[0], arr1.shape[1], use_flip
        )
        intervals2 = _component_intervals(
            components, 1, raw_similarity.shape[1], arr2.shape[1], use_flip
        )
        predicted_components = len(components)
        score = float(result.score)
        normalized_score = float(result.normalized_score)
        algorithm = "needleman_wunsch_component_v3_local"
    else:
        path, score, _dp, _traceback = smith_waterman(
            raw_similarity,
            threshold=args.threshold,
            gap_penalty=args.gap,
            return_traceback=True,
            match_scores=match_scores,
        )
        global_path = list(path)
        accepted_path = list(path)
        intervals1 = _sw_intervals(
            path, 0, raw_similarity.shape[0], arr1.shape[1], use_flip
        )
        intervals2 = _sw_intervals(
            path, 1, raw_similarity.shape[1], arr2.shape[1], use_flip
        )
        predicted_components = 1 if path else 0
        normalized_score = float(score / max(1, max(raw_similarity.shape)))
        algorithm = "smith_waterman_current_real"

    no_mask = not intervals1 and not intervals2
    coverage1 = _coverage(intervals1, arr1.shape[1])
    coverage2 = _coverage(intervals2, arr2.shape[1])
    _save_visualization(
        output,
        arr1,
        arr2,
        raw_similarity,
        global_path,
        accepted_path,
        intervals1,
        intervals2,
        f"unaligned {dataset_type} | {algorithm} | score={score:.4f}",
        no_mask,
    )
    return {
        "pair_id": record["pair_id"],
        "label_type": record["label_type"],
        "source_index1": int(record["source_index1"]),
        "source_index2": int(record["source_index2"]),
        "algorithm": algorithm,
        "score": float(score),
        "normalized_score": normalized_score,
        "predicted_components": int(predicted_components),
        "zero_mask": bool(no_mask),
        "false_positive_mask": bool(not no_mask),
        "false_mask_fraction_line1": coverage1,
        "false_mask_fraction_line2": coverage2,
        "mean_false_mask_fraction": float(0.5 * (coverage1 + coverage2)),
        "image1": str(record["image1"]),
        "image2": str(record["image2"]),
        "output": str(output),
    }


def _aggregate(rows: list[dict], dataset_type: str) -> dict:
    total = len(rows)
    zero = sum(bool(row["zero_mask"]) for row in rows)
    false_positive = total - zero
    mean = lambda key: float(np.mean([float(row[key]) for row in rows])) if rows else 0.0
    return {
        "dataset_type": dataset_type,
        "samples": total,
        "zero_mask_pairs": zero,
        "false_positive_pairs": false_positive,
        "zero_mask_rate": float(zero / total) if total else 0.0,
        "false_positive_pair_rate": float(false_positive / total) if total else 0.0,
        "mean_predicted_components": mean("predicted_components"),
        "mean_false_mask_fraction_line1": mean("false_mask_fraction_line1"),
        "mean_false_mask_fraction_line2": mean("false_mask_fraction_line2"),
        "mean_false_mask_fraction": mean("mean_false_mask_fraction"),
        "max_predicted_components": max((int(row["predicted_components"]) for row in rows), default=0),
        "desired_behavior": "zero predicted alignment masks for every unaligned pair",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset-type", choices=("synthetic", "real"), required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--arabic-manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-samples", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=27000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-split", choices=("train", "valid", "test", "all"), default="test")
    parser.add_argument("--real-text-key", default="text_original_path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--feature", choices=("contextual", "local", "grouped"), default="contextual")
    parser.add_argument("--score-mode", choices=("auto", "raw", "centered", "mutual-z"), default="auto")
    parser.add_argument("--score-clip", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--gap", type=float, default=-0.30)
    parser.add_argument("--min-shared-word-chars", type=int, default=4)
    parser.add_argument("--min-common-compact-chars", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_samples <= 0:
        raise SystemExit("--n-samples must be positive")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Match v3-local aligned evaluation: credible local NW components are not
    # erased solely because the full-line global score is negative.
    os.environ.setdefault("NW_COMPONENT_WEAK_GLOBAL_SCORE", "-1000000.0")
    os.environ.setdefault("NW_COMPONENT_MIN_MATCHES", "7")
    os.environ.setdefault("NW_COMPONENT_MIN_SPAN_WINDOWS", "7")
    os.environ.setdefault("NW_COMPONENT_MIN_SPAN_FRACTION", "0.13")

    records = (
        _select_synthetic_negatives(args)
        if args.dataset_type == "synthetic"
        else _select_real_negatives(args)
    )
    (output_dir / "selected_unaligned_pairs.json").write_text(
        json.dumps(
            [
                {
                    **{key: value for key, value in record.items() if key not in {"image1", "image2"}},
                    "image1": str(record["image1"]),
                    "image2": str(record["image2"]),
                }
                for record in records
            ],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    models = load_evaluation_models(args.weights, args.device, load_text_model=False)
    rows = []
    for position, record in enumerate(records, start=1):
        output = output_dir / f"pair_{position:03d}.png"
        row = _evaluate_one(models, record, args.dataset_type, args, output)
        row["index"] = position
        rows.append(row)
        print(
            f"[{position}/{len(records)}] pair_id={row['pair_id']} "
            f"zero_mask={row['zero_mask']} components={row['predicted_components']} "
            f"false_coverage={row['mean_false_mask_fraction']:.4f}",
            flush=True,
        )

    fieldnames = [
        "index", "pair_id", "label_type", "source_index1", "source_index2",
        "algorithm", "score", "normalized_score", "predicted_components",
        "zero_mask", "false_positive_mask", "false_mask_fraction_line1",
        "false_mask_fraction_line2", "mean_false_mask_fraction",
        "image1", "image2", "output",
    ]
    with (output_dir / "samples.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary = _aggregate(rows, args.dataset_type)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
