#!/usr/bin/env python3
"""Transcript-supervised quantitative evaluation for real manuscript alignment.

This evaluator requires only paired real line images and their line transcripts.
It does not claim pixel-level localization accuracy. It reports:

1. Pair classification precision, recall, F1/Dice, and positive-set IoU using
   transcript overlap as the reference label and a validation-calibrated visual
   alignment threshold.
2. Transcript-defined retrieval metrics with one true transcript-overlapping
   candidate and unique transcript-negative candidates per query.
3. Word-correspondence precision, recall, F1/Dice, and IoU. Exact transcript
   token alignment supplies the reference word pairs; visual Smith-Waterman
   paths supply predicted word pairs after transcript-to-window forced alignment.

The word metrics are a transcript-supervised proxy because transcript-to-window
regions are inferred by the checkpoint rather than manually annotated in image
coordinates. They must not be reported as pixel IoU or mask Dice.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import re
import tempfile
import unicodedata
from typing import Sequence

import numpy as np

from Evaluation.quantitative_real import (
    _alignment,
    _average_precision,
    _best_f1_threshold,
    _classification_at_threshold,
    _feature_cache,
    _install_runtime,
    _json_ready,
    _mean,
    _roc_auc,
    _write_csv,
)


@dataclass(frozen=True)
class TranscriptPair:
    manifest_position: int
    pair_id: str
    label_type: str
    image1: Path
    image2: Path
    text1: str
    text2: str
    text_score: float


@dataclass(frozen=True)
class TextReference:
    tokens1: tuple[str, ...]
    tokens2: tuple[str, ...]
    matched_pairs: tuple[tuple[int, int], ...]
    shared_words: int
    coverage1: float
    coverage2: float
    min_coverage: float
    dice: float
    iou: float
    class_label: int | None


_ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_ALEF_TRANSLATION = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي"})


def normalize_transcript(text: str) -> str:
    """Normalize cleaned Arabic transcripts without introducing OCR output."""
    value = unicodedata.normalize("NFKC", str(text)).replace("ـ", "")
    value = _ARABIC_DIACRITICS.sub("", value).translate(_ALEF_TRANSLATION)
    chars = []
    for char in value:
        category = unicodedata.category(char)
        if category.startswith(("P", "S", "M")):
            chars.append(" ")
        else:
            chars.append(char)
    return " ".join("".join(chars).split())


def tokenize(text: str) -> tuple[str, ...]:
    normalized = normalize_transcript(text)
    return tuple(token for token in normalized.split() if token)


def lcs_pairs(left: Sequence[str], right: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Return deterministic exact-token LCS index pairs."""
    n, m = len(left), len(right)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if left[i] == right[j]:
                dp[i, j] = dp[i + 1, j + 1] + 1
            else:
                dp[i, j] = max(dp[i + 1, j], dp[i, j + 1])
    pairs: list[tuple[int, int]] = []
    i = j = 0
    while i < n and j < m:
        if left[i] == right[j] and dp[i, j] == dp[i + 1, j + 1] + 1:
            pairs.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1, j] >= dp[i, j + 1]:
            i += 1
        else:
            j += 1
    return tuple(pairs)


def text_reference(
    text1: str,
    text2: str,
    positive_overlap: float,
    negative_overlap: float,
    min_shared_words: int,
) -> TextReference:
    tokens1, tokens2 = tokenize(text1), tokenize(text2)
    pairs = lcs_pairs(tokens1, tokens2)
    shared = len(pairs)
    coverage1 = shared / max(1, len(tokens1))
    coverage2 = shared / max(1, len(tokens2))
    min_coverage = shared / max(1, min(len(tokens1), len(tokens2)))
    dice = 2.0 * shared / max(1, len(tokens1) + len(tokens2))
    union = len(tokens1) + len(tokens2) - shared
    iou = shared / max(1, union)
    label: int | None
    if shared >= int(min_shared_words) and min_coverage >= float(positive_overlap):
        label = 1
    elif shared == 0 or min_coverage <= float(negative_overlap):
        label = 0
    else:
        label = None
    return TextReference(
        tokens1=tokens1,
        tokens2=tokens2,
        matched_pairs=pairs,
        shared_words=shared,
        coverage1=coverage1,
        coverage2=coverage2,
        min_coverage=min_coverage,
        dice=dice,
        iou=iou,
        class_label=label,
    )


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def load_manifest_pairs(args) -> list[TranscriptPair]:
    from RealDataSet import ArabicManifestLinePairDataset

    dataset = ArabicManifestLinePairDataset(
        manifest_path=args.arabic_manifest,
        transform=None,
        text_key=args.real_text_key,
        allowed_labels=None,
        paired=True,
        min_text_score=0.0,
        validate_paths=True,
    )
    pairs: list[TranscriptPair] = []
    for position, sample in enumerate(dataset.samples, start=1):
        side1, side2 = sample["A"], sample["B"]
        image1 = dataset._resolve(side1["line_image_path"])
        image2 = dataset._resolve(side2["line_image_path"])
        text1_path = dataset._resolve(side1[args.real_text_key])
        text2_path = dataset._resolve(side2[args.real_text_key])
        pairs.append(
            TranscriptPair(
                manifest_position=position,
                pair_id=str(sample.get("pair_id", position)),
                label_type=str(sample.get("label_type", "")),
                image1=image1,
                image2=image2,
                text1=_read_text(text1_path),
                text2=_read_text(text2_path),
                text_score=float((sample.get("scores") or {}).get("text_score", 0.0)),
            )
        )
    return pairs


def group_split_pairs(pairs: Sequence[TranscriptPair], seed: int):
    groups: dict[str, list[TranscriptPair]] = defaultdict(list)
    for position, pair in enumerate(pairs):
        groups[pair.pair_id or f"sample_{position}"].append(pair)
    items = list(groups.items())
    random.Random(int(seed)).shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)
    targets = {
        "train": 0.60 * len(pairs),
        "valid": 0.20 * len(pairs),
        "test": 0.20 * len(pairs),
    }
    assigned: dict[str, list[TranscriptPair]] = {"train": [], "valid": [], "test": []}
    if len(items) >= 3:
        seeds = sorted(items[:3], key=lambda item: len(item[1]))
        for split, item in zip(("test", "valid", "train"), seeds):
            assigned[split].extend(item[1])
            items.remove(item)
    for _group_id, members in items:
        destination = max(
            ("train", "valid", "test"),
            key=lambda split: targets[split] - len(assigned[split]),
        )
        assigned[destination].extend(members)
    return assigned


def _stratified_limit(rows: Sequence[tuple[TranscriptPair, TextReference]], limit: int, seed: int):
    rows = list(rows)
    if int(limit) <= 0 or len(rows) <= int(limit):
        return rows
    positives = [row for row in rows if row[1].class_label == 1]
    negatives = [row for row in rows if row[1].class_label == 0]
    rng = random.Random(int(seed))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    positive_target = min(len(positives), max(1, int(limit) // 2))
    negative_target = min(len(negatives), int(limit) - positive_target)
    selected = positives[:positive_target] + negatives[:negative_target]
    remaining = positives[positive_target:] + negatives[negative_target:]
    rng.shuffle(remaining)
    selected.extend(remaining[: max(0, int(limit) - len(selected))])
    rng.shuffle(selected)
    return selected


def _alignment_score(aligned: dict, mode: str, min_path_steps: int) -> tuple[float, dict]:
    region = aligned["region"]
    coverage1 = region.line1_span_windows / max(1, aligned["line1_windows"])
    coverage2 = region.line2_span_windows / max(1, aligned["line2_windows"])
    joint_coverage = math.sqrt(max(0.0, coverage1 * coverage2))
    path_gate = min(1.0, region.path_steps / max(1, int(min_path_steps)))
    if mode == "coverage_sw":
        base = aligned["normalized_score"]
    elif mode == "coverage_cosine":
        base = aligned["mean_path_cosine"]
    elif mode == "coverage_hybrid":
        base = aligned["normalized_score"] + aligned["mean_path_cosine"]
    else:
        raise ValueError("ranking score must be coverage_sw, coverage_cosine, or coverage_hybrid")
    score = float(base) * joint_coverage * path_gate
    return score, {
        "joint_coverage": joint_coverage,
        "line1_coverage": coverage1,
        "line2_coverage": coverage2,
        "path_steps": int(region.path_steps),
    }


def _set_metrics(predicted: set, reference: set) -> dict:
    tp = len(predicted & reference)
    fp = len(predicted - reference)
    fn = len(reference - predicted)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    iou = tp / max(1, tp + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": f1,
        "iou": iou,
    }


def _classification_summary(labels, scores, threshold: float) -> dict:
    metrics = _classification_at_threshold(labels, scores, threshold)
    metrics["dice"] = metrics["f1"]
    metrics["positive_set_iou"] = metrics["tp"] / max(
        1, metrics["tp"] + metrics["fp"] + metrics["fn"]
    )
    metrics["auroc"] = _roc_auc(labels, scores)
    metrics["average_precision"] = _average_precision(labels, scores)
    metrics["positives"] = sum(int(label) == 1 for label in labels)
    metrics["negatives"] = sum(int(label) == 0 for label in labels)
    return metrics


def run_pair_classification(runtime, models, valid_rows, test_rows, args, output: Path):
    rows: list[dict] = []
    split_payload = [("valid", valid_rows), ("test", test_rows)]
    with tempfile.TemporaryDirectory(prefix="transcript_pair_quant_") as temp_dir:
        get_features = _feature_cache(runtime, models, "real", Path(temp_dir))
        for split_name, split_rows in split_payload:
            for order, (pair, reference) in enumerate(split_rows, start=1):
                aligned = _alignment(
                    runtime,
                    get_features(pair.image1),
                    get_features(pair.image2),
                    args,
                )
                score, details = _alignment_score(
                    aligned, args.ranking_score, args.min_path_steps
                )
                rows.append(
                    {
                        "split": split_name,
                        "order": order,
                        "manifest_position": pair.manifest_position,
                        "pair_id": pair.pair_id,
                        "manifest_label": pair.label_type,
                        "reference_label": int(reference.class_label),
                        "shared_words": reference.shared_words,
                        "transcript_min_coverage": reference.min_coverage,
                        "transcript_dice": reference.dice,
                        "transcript_iou": reference.iou,
                        "visual_score": score,
                        "normalized_sw_score": aligned["normalized_score"],
                        "mean_path_cosine": aligned["mean_path_cosine"],
                        **details,
                        "image1": str(pair.image1),
                        "image2": str(pair.image2),
                    }
                )
                print(
                    f"classification {split_name} {order}/{len(split_rows)} "
                    f"label={reference.class_label} score={score:.4f}",
                    flush=True,
                )
    valid_output = [row for row in rows if row["split"] == "valid"]
    test_output = [row for row in rows if row["split"] == "test"]
    valid_labels = [int(row["reference_label"]) for row in valid_output]
    valid_scores = [float(row["visual_score"]) for row in valid_output]
    test_labels = [int(row["reference_label"]) for row in test_output]
    test_scores = [float(row["visual_score"]) for row in test_output]
    threshold = _best_f1_threshold(valid_labels, valid_scores)
    summary = {
        "threshold_selected_on": "valid",
        "threshold": threshold,
        "validation": _classification_summary(valid_labels, valid_scores, threshold),
        "test": _classification_summary(test_labels, test_scores, threshold),
    }
    for row in rows:
        row["threshold"] = threshold
        row["predicted_label"] = int(float(row["visual_score"]) >= threshold)
    _write_csv(output / "pair_classification.csv", rows)
    return rows, summary


def _deduplicate_candidates(pairs: Sequence[TranscriptPair]) -> list[tuple[Path, str, str]]:
    """Keep one unique candidate image per page-pair identity."""
    seen_paths: set[str] = set()
    seen_pair_ids: set[str] = set()
    candidates = []
    for pair in pairs:
        resolved = str(pair.image2.resolve())
        if resolved in seen_paths or pair.pair_id in seen_pair_ids:
            continue
        seen_paths.add(resolved)
        seen_pair_ids.add(pair.pair_id)
        candidates.append((pair.image2, pair.text2, pair.pair_id))
    return candidates


def run_retrieval(runtime, models, test_pairs, args, output: Path):
    positive_queries = []
    for pair in test_pairs:
        reference = text_reference(
            pair.text1,
            pair.text2,
            args.positive_overlap,
            args.negative_overlap,
            args.min_shared_words,
        )
        if reference.class_label == 1:
            positive_queries.append((pair, reference))
    rng = random.Random(args.seed + 31)
    rng.shuffle(positive_queries)
    positive_queries = positive_queries[: args.retrieval_queries]
    candidate_universe = _deduplicate_candidates(test_pairs)
    retrieval_rows: list[dict] = []
    score_rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="transcript_retrieval_quant_") as temp_dir:
        get_features = _feature_cache(runtime, models, "real", Path(temp_dir))
        for query_order, (pair, _reference) in enumerate(positive_queries, start=1):
            true_path = pair.image2.resolve()
            negatives = []
            for candidate_path, candidate_text, candidate_pair_id in candidate_universe:
                if candidate_path.resolve() == true_path or candidate_pair_id == pair.pair_id:
                    continue
                candidate_reference = text_reference(
                    pair.text1,
                    candidate_text,
                    args.positive_overlap,
                    args.negative_overlap,
                    args.min_shared_words,
                )
                if candidate_reference.class_label == 0:
                    negatives.append((candidate_path, candidate_text, candidate_pair_id, candidate_reference))
            rng.shuffle(negatives)
            selected_negatives = negatives[: max(0, args.retrieval_pool_size - 1)]
            true_reference = text_reference(
                pair.text1,
                pair.text2,
                args.positive_overlap,
                args.negative_overlap,
                args.min_shared_words,
            )
            pool = [(pair.image2, pair.text2, pair.pair_id, true_reference, 1)]
            pool.extend((path, text, pair_id, ref, 0) for path, text, pair_id, ref in selected_negatives)
            rng.shuffle(pool)
            query_features = get_features(pair.image1)
            ranked = []
            for candidate_path, _candidate_text, candidate_pair_id, candidate_ref, is_positive in pool:
                aligned = _alignment(runtime, query_features, get_features(candidate_path), args)
                score, details = _alignment_score(
                    aligned, args.ranking_score, args.min_path_steps
                )
                ranked.append((score, is_positive, candidate_path, candidate_pair_id, candidate_ref, aligned, details))
            ranked.sort(key=lambda item: item[0], reverse=True)
            positive_rank = next(
                rank for rank, item in enumerate(ranked, start=1) if int(item[1]) == 1
            )
            retrieval_rows.append(
                {
                    "query_order": query_order,
                    "pair_id": pair.pair_id,
                    "pool_size": len(ranked),
                    "positive_rank": positive_rank,
                    "reciprocal_rank": 1.0 / positive_rank,
                    "recall_at_1": int(positive_rank <= 1),
                    "recall_at_5": int(positive_rank <= 5),
                    "recall_at_10": int(positive_rank <= 10),
                    "average_precision": 1.0 / positive_rank,
                    "query_image": str(pair.image1),
                    "true_candidate": str(pair.image2),
                }
            )
            for rank, item in enumerate(ranked, start=1):
                score, is_positive, candidate_path, candidate_pair_id, candidate_ref, aligned, details = item
                score_rows.append(
                    {
                        "query_order": query_order,
                        "rank": rank,
                        "query_pair_id": pair.pair_id,
                        "candidate_pair_id": candidate_pair_id,
                        "is_positive": int(is_positive),
                        "visual_score": score,
                        "normalized_sw_score": aligned["normalized_score"],
                        "mean_path_cosine": aligned["mean_path_cosine"],
                        **details,
                        "transcript_min_coverage": candidate_ref.min_coverage,
                        "transcript_dice": candidate_ref.dice,
                        "transcript_iou": candidate_ref.iou,
                        "query_image": str(pair.image1),
                        "candidate_image": str(candidate_path),
                    }
                )
            print(
                f"retrieval {query_order}/{len(positive_queries)} rank={positive_rank}/{len(ranked)}",
                flush=True,
            )
    summary = {
        "queries": len(retrieval_rows),
        "pool_size": args.retrieval_pool_size,
        "candidate_protocol": "one transcript-positive plus unique transcript-negative images",
        "recall_at_1": _mean(row["recall_at_1"] for row in retrieval_rows),
        "recall_at_5": _mean(row["recall_at_5"] for row in retrieval_rows),
        "recall_at_10": _mean(row["recall_at_10"] for row in retrieval_rows),
        "mrr": _mean(row["reciprocal_rank"] for row in retrieval_rows),
        "map": _mean(row["average_precision"] for row in retrieval_rows),
    }
    _write_csv(output / "transcript_retrieval.csv", retrieval_rows)
    _write_csv(output / "transcript_retrieval_scores.csv", score_rows)
    return retrieval_rows, score_rows, summary


def _window_to_words(regions, n_windows: int) -> list[list[int]]:
    mapping: list[list[int]] = [[] for _ in range(max(0, int(n_windows)))]
    for region in regions:
        start = max(0, int(region.window_start))
        end = min(len(mapping), int(region.window_end))
        for window in range(start, end):
            mapping[window].append(int(region.index))
    return mapping


def _monotonic_pairs_from_support(
    support: dict[tuple[int, int], int],
    n1: int,
    n2: int,
    min_support: int,
) -> set[tuple[int, int]]:
    weights = np.zeros((n1, n2), dtype=np.float32)
    for (i, j), count in support.items():
        if 0 <= i < n1 and 0 <= j < n2 and int(count) >= int(min_support):
            weights[i, j] = float(count)
    dp = np.zeros((n1 + 1, n2 + 1), dtype=np.float32)
    trace = np.zeros((n1 + 1, n2 + 1), dtype=np.uint8)
    for i in range(1, n1 + 1):
        for j in range(1, n2 + 1):
            up = dp[i - 1, j]
            left = dp[i, j - 1]
            match = dp[i - 1, j - 1] + weights[i - 1, j - 1]
            values = (up, left, match)
            best = int(np.argmax(values))
            dp[i, j] = values[best]
            trace[i, j] = best + 1
    pairs: set[tuple[int, int]] = set()
    i, j = n1, n2
    while i > 0 and j > 0:
        code = int(trace[i, j])
        if code == 3:
            if weights[i - 1, j - 1] > 0:
                pairs.add((i - 1, j - 1))
            i -= 1
            j -= 1
        elif code == 1:
            i -= 1
        else:
            j -= 1
    return pairs


def _micro_summary(rows: Sequence[dict], prefix: str) -> dict:
    tp = sum(int(row[f"{prefix}_tp"]) for row in rows)
    fp = sum(int(row[f"{prefix}_fp"]) for row in rows)
    fn = sum(int(row[f"{prefix}_fn"]) for row in rows)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    iou = tp / max(1, tp + fp + fn)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "dice": f1,
        "iou": iou,
    }


def run_word_alignment(runtime, models, test_pairs, args, output: Path):
    positive_pairs = []
    for pair in test_pairs:
        reference = text_reference(
            pair.text1,
            pair.text2,
            args.positive_overlap,
            args.negative_overlap,
            args.min_shared_words,
        )
        if reference.class_label == 1:
            positive_pairs.append((pair, reference))
    rng = random.Random(args.seed + 71)
    rng.shuffle(positive_pairs)
    positive_pairs = positive_pairs[: args.word_pairs]
    rows: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="transcript_word_quant_") as temp_dir:
        get_features = _feature_cache(runtime, models, "real", Path(temp_dir))
        for order, (pair, reference) in enumerate(positive_pairs, start=1):
            features1 = get_features(pair.image1)
            features2 = get_features(pair.image2)
            aligned = _alignment(runtime, features1, features2, args)
            regions1, _ = runtime.utils.extract_word_regions(
                models, pair.text1, features1, feature=args.word_feature
            )
            regions2, _ = runtime.utils.extract_word_regions(
                models, pair.text2, features2, feature=args.word_feature
            )
            map1 = _window_to_words(regions1, aligned["line1_windows"])
            map2 = _window_to_words(regions2, aligned["line2_windows"])
            support: dict[tuple[int, int], int] = defaultdict(int)
            for window1, window2 in aligned["path"]:
                if not (0 <= int(window1) < len(map1) and 0 <= int(window2) < len(map2)):
                    continue
                for word1 in map1[int(window1)]:
                    for word2 in map2[int(window2)]:
                        support[(word1, word2)] += 1
            predicted_pairs = _monotonic_pairs_from_support(
                support,
                len(reference.tokens1),
                len(reference.tokens2),
                args.word_min_support,
            )
            reference_pairs = set(reference.matched_pairs)
            pair_metrics = _set_metrics(predicted_pairs, reference_pairs)
            predicted_words1 = {i for i, _ in predicted_pairs}
            predicted_words2 = {j for _, j in predicted_pairs}
            reference_words1 = {i for i, _ in reference_pairs}
            reference_words2 = {j for _, j in reference_pairs}
            line1_metrics = _set_metrics(predicted_words1, reference_words1)
            line2_metrics = _set_metrics(predicted_words2, reference_words2)
            rows.append(
                {
                    "order": order,
                    "manifest_position": pair.manifest_position,
                    "pair_id": pair.pair_id,
                    "manifest_label": pair.label_type,
                    "tokens1": len(reference.tokens1),
                    "tokens2": len(reference.tokens2),
                    "reference_word_pairs": len(reference_pairs),
                    "predicted_word_pairs": len(predicted_pairs),
                    **{f"word_pair_{key}": value for key, value in pair_metrics.items()},
                    **{f"line1_word_{key}": value for key, value in line1_metrics.items()},
                    **{f"line2_word_{key}": value for key, value in line2_metrics.items()},
                    "mean_line_word_dice": (line1_metrics["dice"] + line2_metrics["dice"]) / 2.0,
                    "mean_line_word_iou": (line1_metrics["iou"] + line2_metrics["iou"]) / 2.0,
                    "visual_path_steps": int(aligned["region"].path_steps),
                    "forced_regions1": len(regions1),
                    "forced_regions2": len(regions2),
                    "image1": str(pair.image1),
                    "image2": str(pair.image2),
                }
            )
            print(
                f"word alignment {order}/{len(positive_pairs)} "
                f"F1={pair_metrics['f1']:.3f} IoU={pair_metrics['iou']:.3f}",
                flush=True,
            )
    summary = {
        "pairs": len(rows),
        "reference": "exact normalized transcript-token LCS",
        "prediction": "visual SW path projected through checkpoint transcript-to-window forced alignment",
        "is_spatial_ground_truth": False,
        "word_pair_micro": _micro_summary(rows, "word_pair") if rows else None,
        "word_pair_macro": {
            "precision": _mean(row["word_pair_precision"] for row in rows),
            "recall": _mean(row["word_pair_recall"] for row in rows),
            "f1": _mean(row["word_pair_f1"] for row in rows),
            "dice": _mean(row["word_pair_dice"] for row in rows),
            "iou": _mean(row["word_pair_iou"] for row in rows),
        } if rows else None,
        "mean_line_word_dice": _mean(row["mean_line_word_dice"] for row in rows),
        "mean_line_word_iou": _mean(row["mean_line_word_iou"] for row in rows),
    }
    _write_csv(output / "word_alignment.csv", rows)
    return rows, summary


def write_report(summary: dict, output: Path) -> None:
    classification = summary["pair_classification"]["test"]
    retrieval = summary["retrieval"]
    word = summary["word_alignment"]
    word_micro = word.get("word_pair_micro") if word else None
    lines = [
        "# Transcript-supervised real-data evaluation",
        "",
        f"- Checkpoint: `{summary['checkpoint']}`",
        f"- Backend: `{summary['model_backend']}`",
        f"- Transcript key: `{summary['real_text_key']}`",
        f"- Positive reference: transcript min-coverage >= {summary['positive_overlap']} with at least {summary['min_shared_words']} shared words",
        f"- Negative reference: transcript min-coverage <= {summary['negative_overlap']}",
        "",
        "## Pair classification on the held-out test split",
        "",
        f"- Precision: {classification['precision']:.4f}",
        f"- Recall: {classification['recall']:.4f}",
        f"- F1 / pair-set Dice: {classification['f1']:.4f}",
        f"- Pair-set IoU: {classification['positive_set_iou']:.4f}",
        f"- AUROC: {classification['auroc']:.4f}" if classification['auroc'] is not None else "- AUROC: n/a",
        f"- Average precision: {classification['average_precision']:.4f}" if classification['average_precision'] is not None else "- Average precision: n/a",
        "",
        "## Transcript-defined retrieval",
        "",
        f"- Queries: {retrieval['queries']}",
        f"- Recall@1: {retrieval['recall_at_1']:.4f}",
        f"- Recall@5: {retrieval['recall_at_5']:.4f}",
        f"- Recall@10: {retrieval['recall_at_10']:.4f}",
        f"- MRR: {retrieval['mrr']:.4f}",
        f"- mAP: {retrieval['map']:.4f}",
        "",
        "## Word-correspondence proxy",
        "",
    ]
    if word_micro:
        lines.extend(
            [
                f"- Evaluated pairs: {word['pairs']}",
                f"- Word-pair precision: {word_micro['precision']:.4f}",
                f"- Word-pair recall: {word_micro['recall']:.4f}",
                f"- Word-pair F1 / Dice: {word_micro['f1']:.4f}",
                f"- Word-pair IoU: {word_micro['iou']:.4f}",
                f"- Mean line-level word Dice: {word['mean_line_word_dice']:.4f}",
                f"- Mean line-level word IoU: {word['mean_line_word_iou']:.4f}",
            ]
        )
    else:
        lines.append("- Word metrics were unavailable for this checkpoint or sample selection.")
    lines.extend(
        [
            "",
            "## Interpretation limitation",
            "",
            "These metrics use transcripts as supervision. Pair and retrieval metrics are valid image-pair matching metrics. "
            "Word Dice/IoU are transcript-supervised proxy metrics because word-to-window locations are forced-aligned by "
            "the checkpoint. They are not pixel-level mask Dice or spatial interval IoU.",
            "",
        ]
    )
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--real-data-dir", default="DataSet/ArabicDataset")
    parser.add_argument("--arabic-manifest", default="DataSet/ArabicDataset/dataset_manifest.jsonl")
    parser.add_argument("--real-text-key", default="text_original_path")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive-overlap", type=float, default=0.50)
    parser.add_argument("--negative-overlap", type=float, default=0.10)
    parser.add_argument("--min-shared-words", type=int, default=2)
    parser.add_argument("--max-valid-pairs", type=int, default=0)
    parser.add_argument("--max-test-pairs", type=int, default=0)
    parser.add_argument("--retrieval-queries", type=int, default=100)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--word-pairs", type=int, default=100)
    parser.add_argument("--word-min-support", type=int, default=1)
    parser.add_argument("--feature", choices=("contextual", "local", "grouped"), default="contextual")
    parser.add_argument("--word-feature", choices=("contextual", "local", "grouped"), default="local")
    parser.add_argument(
        "--ranking-score",
        choices=("coverage_sw", "coverage_cosine", "coverage_hybrid"),
        default="coverage_sw",
    )
    parser.add_argument("--min-path-steps", type=int, default=5)
    parser.add_argument("--score-mode", default="auto")
    parser.add_argument("--score-clip", type=float, default=4.0)
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--gap", type=float, default=-0.30)
    args = parser.parse_args()
    if not 0.0 <= args.negative_overlap < args.positive_overlap <= 1.0:
        parser.error("Require 0 <= negative-overlap < positive-overlap <= 1")
    for name in ("min_shared_words", "retrieval_queries", "retrieval_pool_size", "word_pairs", "word_min_support", "min_path_steps"):
        if int(getattr(args, name)) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    runtime = _install_runtime()
    models = runtime.utils.load_evaluation_models(
        args.weights, device=args.device, load_text_model=True
    )
    pairs = load_manifest_pairs(args)
    splits = group_split_pairs(pairs, args.split_seed)

    def labeled(items):
        result = []
        for pair in items:
            reference = text_reference(
                pair.text1,
                pair.text2,
                args.positive_overlap,
                args.negative_overlap,
                args.min_shared_words,
            )
            if reference.class_label is not None:
                result.append((pair, reference))
        return result

    valid_rows = _stratified_limit(labeled(splits["valid"]), args.max_valid_pairs, args.seed + 1)
    test_rows = _stratified_limit(labeled(splits["test"]), args.max_test_pairs, args.seed + 2)
    if not valid_rows or len({row[1].class_label for row in valid_rows}) < 2:
        raise RuntimeError("Validation split must contain transcript-positive and transcript-negative pairs")
    if not test_rows or len({row[1].class_label for row in test_rows}) < 2:
        raise RuntimeError("Test split must contain transcript-positive and transcript-negative pairs")

    print(
        f"transcript_quantitative backend={models.config.get('model_backend', 'cnn_bilstm')} "
        f"manifest_pairs={len(pairs)} valid={len(valid_rows)} test={len(test_rows)}",
        flush=True,
    )
    _classification_rows, classification_summary = run_pair_classification(
        runtime, models, valid_rows, test_rows, args, output
    )
    _retrieval_rows, _retrieval_scores, retrieval_summary = run_retrieval(
        runtime, models, splits["test"], args, output
    )
    try:
        _word_rows, word_summary = run_word_alignment(
            runtime, models, splits["test"], args, output
        )
    except (RuntimeError, ValueError) as exc:
        word_summary = {
            "pairs": 0,
            "available": False,
            "error": f"{type(exc).__name__}: {exc}",
            "is_spatial_ground_truth": False,
        }
        (output / "word_alignment.csv").write_text("", encoding="utf-8")
        print(f"WARNING: word metrics unavailable: {exc}", flush=True)

    summary = {
        "checkpoint": str(Path(args.weights).resolve()),
        "model_backend": str(models.config.get("model_backend", "cnn_bilstm")),
        "manifest_pairs": len(pairs),
        "split_counts": {key: len(value) for key, value in splits.items()},
        "real_text_key": args.real_text_key,
        "positive_overlap": args.positive_overlap,
        "negative_overlap": args.negative_overlap,
        "min_shared_words": args.min_shared_words,
        "ranking_score": args.ranking_score,
        "min_path_steps": args.min_path_steps,
        "pair_classification": classification_summary,
        "retrieval": retrieval_summary,
        "word_alignment": word_summary,
        "metric_scope": {
            "pair_precision_recall_f1": "transcript-defined pair matching",
            "pair_dice_iou": "overlap of predicted-positive and transcript-positive pair sets",
            "word_dice_iou": "transcript-token correspondence proxy, not image pixels",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(summary, output)
    print(f"Transcript-supervised evaluation complete: {output}", flush=True)


if __name__ == "__main__":
    main()
