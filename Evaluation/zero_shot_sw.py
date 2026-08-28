"""Zero-shot preprocessing, sampling, and ink-aware Smith-Waterman patches."""
from __future__ import annotations

from collections import OrderedDict
import json
import os
from pathlib import Path
import random
import tempfile

import numpy as np
from PIL import Image

from Evaluation import sw_dataset
from Evaluation.trace_components import (
    component_intervals_px,
    component_metrics,
    save_alignment_visualization,
    save_numeric_evidence,
    sw_component_path,
)
from zero_shot_preprocessing import build_preprocessor, env_flag, env_float


def balanced_group_split_pairs(pairs, seed: int):
    """Allocate complete pair_id groups while avoiding a one-group test split."""
    groups = OrderedDict()
    for position, pair in enumerate(pairs):
        groups.setdefault(pair.pair_id or f"sample_{position}", []).append(pair)
    if len(groups) < 3:
        return sw_dataset.random_split_pairs(pairs, seed)

    rng = random.Random(int(seed))
    items = list(groups.items())
    rng.shuffle(items)
    items.sort(key=lambda item: len(item[1]), reverse=True)

    targets = {
        "train": 0.60 * len(pairs),
        "valid": 0.20 * len(pairs),
        "test": 0.20 * len(pairs),
    }
    assigned = {"train": [], "valid": [], "test": []}

    smallest = sorted(items[:3], key=lambda item: len(item[1]))
    for split, item in zip(("test", "valid", "train"), smallest):
        assigned[split].extend(item[1])
        items.remove(item)

    for _group_id, members in items:
        def deficit(split):
            return targets[split] - len(assigned[split])

        destination = max(("train", "valid", "test"), key=deficit)
        assigned[destination].extend(members)

    return assigned["train"], assigned["valid"], assigned["test"]


def balanced_batch_pairs(args, manifest_pairs):
    """Round-robin pair_ids so the first N outputs are not one page pair only."""
    if not manifest_pairs or not env_flag("REAL_EVAL_BALANCED", True):
        return _ORIGINAL_BATCH_PAIRS(args, manifest_pairs)

    groups = OrderedDict()
    for pair in manifest_pairs:
        groups.setdefault(pair.pair_id or f"sample_{pair.index}", []).append(pair)
    ordered = []
    positions = {key: 0 for key in groups}
    while True:
        added = False
        for key, values in groups.items():
            position = positions[key]
            if position < len(values):
                ordered.append(values[position])
                positions[key] += 1
                added = True
        if not added:
            break
    start = max(0, int(args.start_index) - 1)
    return ordered[start : start + int(args.n_samples)]


def real_binarizer():
    """Use the shared real-line preprocessing used by evaluation/training."""
    return build_preprocessor("real", training=False)


def display_image(path: str | Path, dataset_type: str) -> np.ndarray:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        if str(dataset_type).lower() == "real":
            image = real_binarizer()(image)
        return np.asarray(image.convert("RGB"))


def ink_aware_match_scores(match_scores, ink1, ink2):
    """Suppress blank-window matches that otherwise create long false paths."""
    scores = np.asarray(match_scores, dtype=np.float32).copy()
    if not env_flag("SW_INK_AWARE", True):
        return scores
    minimum = env_float("SW_MIN_INK", 0.02)
    blank_blank_score = env_float("SW_BLANK_BLANK_SCORE", -0.20)
    blank_ink_score = env_float("SW_BLANK_INK_SCORE", -0.50)
    left_blank = np.asarray(ink1, dtype=np.float32) < minimum
    right_blank = np.asarray(ink2, dtype=np.float32) < minimum
    both_blank = np.outer(left_blank, right_blank)
    one_blank = np.logical_xor(left_blank[:, None], right_blank[None, :])
    scores[both_blank] = np.minimum(scores[both_blank], blank_blank_score)
    scores[one_blank] = np.minimum(scores[one_blank], blank_ink_score)
    return scores


def _empty_mask_metrics() -> dict:
    result = {"mean_region_iou": None}
    for prefix in ("line1", "line2"):
        for suffix in (
            "pred_start_px", "pred_end_px", "gt_start_px", "gt_end_px",
            "region_iou", "start_error_px", "end_error_px",
        ):
            result[f"{prefix}_{suffix}"] = None
    return result


_COMPONENT_FIELDS = [
    "component_count",
    "full_match_steps",
    "bridged_trace_steps",
    "component_bridge_limit",
    "component_support_floor",
    "component_score",
    "mean_full_path_cosine",
    "line1_component_ranges",
    "line2_component_ranges",
    "line1_component_intervals_px",
    "line2_component_intervals_px",
    "matrix_csv_cosine",
    "matrix_csv_match",
    "evidence_json",
]


_ORIGINAL_BATCH_PAIRS = sw_dataset.batch_pairs


def install_dataset_patches() -> None:
    sw_dataset.group_split_pairs = balanced_group_split_pairs
    sw_dataset._group_split_pairs = balanced_group_split_pairs
    sw_dataset.batch_pairs = balanced_batch_pairs
    sw_dataset.real_binarizer = real_binarizer
    sw_dataset.display_image = display_image
    sw_dataset._display_image = display_image


def install_runner_patches(runner) -> None:
    """Install real preprocessing plus component-aware interpretation of true SW."""
    if getattr(runner, "_zero_shot_sw_components_installed", False):
        return

    original_fieldnames = runner.batch_fieldnames
    original_aggregate = runner.aggregate

    def evaluate_sample(
        models,
        pair,
        dataset_type,
        feature,
        threshold,
        gap,
        score_mode,
        score_clip,
        output,
        save_binary,
        show_heatmap,
        heatmap_source,
        annotate_values,
        value_decimals,
        annotation_fontsize,
    ):
        image1, image2 = Path(pair.image1), Path(pair.image2)
        if not image1.is_file() or not image2.is_file():
            missing = [str(path) for path in (image1, image2) if not path.is_file()]
            raise FileNotFoundError("Missing image pair: " + ", ".join(missing))

        binarized = str(dataset_type).lower() == "real"
        binary1 = binary2 = ""
        temporary_directory = None
        try:
            if binarized:
                arr1 = display_image(image1, "real")
                arr2 = display_image(image2, "real")
                if save_binary:
                    model_image1, model_image2 = runner.save_binarized_inputs(
                        arr1, arr2, output, pair.index
                    )
                    binary1, binary2 = str(model_image1), str(model_image2)
                else:
                    temporary_directory = tempfile.TemporaryDirectory(
                        prefix="sw_real_binary_"
                    )
                    root = Path(temporary_directory.name)
                    model_image1, model_image2 = root / "line1.png", root / "line2.png"
                    Image.fromarray(arr1).save(model_image1)
                    Image.fromarray(arr2).save(model_image2)
                feature_dataset_type = "synthetic"
            else:
                with Image.open(image1) as opened:
                    arr1 = np.asarray(opened.convert("RGB"))
                with Image.open(image2) as opened:
                    arr2 = np.asarray(opened.convert("RGB"))
                model_image1, model_image2 = image1, image2
                feature_dataset_type = "synthetic"

            features1 = runner.get_image_features(models, model_image1, feature_dataset_type)
            features2 = runner.get_image_features(models, model_image2, feature_dataset_type)
            raw_similarity = runner.compute_similarity(
                features1.select(feature), features2.select(feature)
            ).cpu().numpy()

            resolved_mode = runner.resolve_score_mode(score_mode, dataset_type)
            match_scores = runner.build_match_scores(
                raw_similarity, resolved_mode, score_clip, threshold
            )
            match_scores = ink_aware_match_scores(
                match_scores,
                features1.ink.detach().cpu().numpy(),
                features2.ink.detach().cpu().numpy(),
            )

            # This is the unchanged Smith-Waterman algorithm.  Its traceback
            # starts at the maximum accumulated DP cell and stops at zero.
            full_path, score, dp_score, traceback = runner.smith_waterman(
                raw_similarity,
                threshold=threshold,
                gap_penalty=gap,
                return_traceback=True,
                match_scores=match_scores,
            )

            use_components = binarized and env_flag("TRACE_COMPONENTS", True)
            component_path = (
                sw_component_path(full_path, traceback, match_scores)
                if use_components
                else sw_component_path(full_path, traceback, np.ones_like(match_scores))
            )
            if not use_components:
                # Synthetic compatibility: keep the complete local SW diagonal path.
                component_path.clear()
                component_path.extend(full_path)
                component_path.runs = (tuple(full_path),) if full_path else ()
                component_path.full_match_steps = len(full_path)

            displayed_matrix, displayed_label = runner.select_heatmap_matrix(
                raw_similarity,
                threshold,
                dp_score,
                heatmap_source,
                match_scores=match_scores,
                score_mode=resolved_mode + "+ink",
            )

            window_size = getattr(models.image_model, "window_size", None)
            stride = getattr(models.image_model, "stride", None)
            intervals1 = component_intervals_px(
                component_path, 0, raw_similarity.shape[0], arr1.shape[1],
                bool(models.image_model.use_flip), window_size=window_size, stride=stride,
            )
            intervals2 = component_intervals_px(
                component_path, 1, raw_similarity.shape[1], arr2.shape[1],
                bool(models.image_model.use_flip), window_size=window_size, stride=stride,
            )

            if show_heatmap or use_components:
                save_alignment_visualization(
                    arr1=arr1,
                    arr2=arr2,
                    features1=features1,
                    features2=features2,
                    full_path=full_path,
                    component_path=component_path,
                    traceback=traceback,
                    heatmap_matrix=displayed_matrix,
                    heatmap_label=displayed_label,
                    score=float(score),
                    output=output,
                    use_flip=bool(models.image_model.use_flip),
                    pair=pair,
                    score_mode=resolved_mode + "+ink",
                    algorithm="Smith-Waterman",
                    traceback_label="SW traceback: maximum DP → zero",
                    traceback_start_label="local DP maximum",
                    traceback_end_label="zero-score boundary",
                    binarized=binarized,
                    annotate_values=bool(annotate_values),
                    value_decimals=value_decimals,
                    annotation_fontsize=annotation_fontsize,
                    window_size=window_size,
                    stride=stride,
                )
            else:
                runner.save_visualization(
                    arr1, arr2, features1, features2, full_path, traceback,
                    displayed_matrix, displayed_label, score, output,
                    models.image_model.use_flip, binarized, pair,
                    resolved_mode + "+ink", show_heatmap=False,
                    annotate_values=False,
                )

            evidence_files = save_numeric_evidence(
                output,
                algorithm="Smith-Waterman",
                raw_similarity=raw_similarity,
                match_scores=match_scores,
                component_path=component_path,
                full_path=full_path,
                traceback=traceback,
                intervals1=intervals1,
                intervals2=intervals2,
            )
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()

        component_path_cosines = [
            float(raw_similarity[i, j]) for i, j in component_path
        ]
        full_path_cosines = [float(raw_similarity[i, j]) for i, j in full_path]
        component_score = float(
            sum(max(0.0, float(match_scores[i, j])) for i, j in component_path)
        )
        max_row, max_col = map(int, np.unravel_index(np.argmax(dp_score), dp_score.shape))
        region_metrics = component_metrics(component_path, raw_similarity.shape)
        mask_metrics = (
            _empty_mask_metrics()
            if binarized
            else runner.synthetic_mask_region_metrics(
                full_path, traceback, raw_similarity.shape, image1, image2,
                arr1.shape[1], arr2.shape[1], models.image_model.use_flip,
            )
        )
        row = {
            "index": int(pair.index),
            "manifest_position": int(pair.manifest_position),
            "pair_id": pair.pair_id,
            "label_type": pair.label_type,
            "text_score": float(pair.text_score),
            "split": pair.split,
            "status": "ok",
            "score": float(score),
            **region_metrics,
            "component_score": component_score,
            "mean_full_path_cosine": (
                float(np.mean(full_path_cosines)) if full_path_cosines else 0.0
            ),
            "dp_max_row": max_row,
            "dp_max_col": max_col,
            "dp_max_is_terminal": bool(
                max_row == dp_score.shape[0] - 1 and max_col == dp_score.shape[1] - 1
            ),
            "mean_path_cosine": (
                float(np.mean(component_path_cosines)) if component_path_cosines else 0.0
            ),
            "line1_windows": int(raw_similarity.shape[0]),
            "line2_windows": int(raw_similarity.shape[1]),
            "line1_component_intervals_px": json.dumps(intervals1, separators=(",", ":")),
            "line2_component_intervals_px": json.dumps(intervals2, separators=(",", ":")),
            **mask_metrics,
            **evidence_files,
            "feature": str(feature),
            "score_mode": resolved_mode,
            "score_clip": float(score_clip),
            "threshold": float(threshold),
            "gap": float(gap),
            "heatmap_source": str(heatmap_source),
            "dataset_type": str(dataset_type),
            "binarized": bool(binarized),
            "binarization": (
                os.environ.get("REAL_BINARIZE_METHOD", "otsu").lower()
                if binarized else "none"
            ),
            "flipped": bool(models.image_model.use_flip),
            "image1": str(image1),
            "image2": str(image2),
            "binarized_image1": binary1,
            "binarized_image2": binary2,
            "output": str(output),
            "error": "",
        }
        print(
            f"[{pair.index}] pair_id={pair.pair_id or '-'} label={pair.label_type or '-'} "
            f"SW={score:.6f} start=DP-max components={row['component_count']} "
            f"supported={row['path_steps']}/{row['full_match_steps']} "
            f"bridge<={row['component_bridge_limit']} "
            f"mean_cosine={row['mean_path_cosine']:.4f} saved={output}",
            flush=True,
        )
        return row

    def batch_fieldnames():
        names = list(original_fieldnames())
        for name in _COMPONENT_FIELDS:
            if name not in names:
                names.append(name)
        return names

    def aggregate(rows):
        summary = original_aggregate(rows)
        successful = [row for row in rows if row.get("status") == "ok"]

        def values(key):
            output = []
            for row in successful:
                try:
                    value = float(row.get(key))
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    output.append(value)
            return output

        summary.update(
            {
                "mean_component_count": (
                    float(np.mean(values("component_count")))
                    if values("component_count") else None
                ),
                "mean_supported_component_matches": (
                    float(np.mean(values("path_steps"))) if values("path_steps") else None
                ),
                "mean_full_sw_matches": (
                    float(np.mean(values("full_match_steps")))
                    if values("full_match_steps") else None
                ),
                "mean_component_span_fraction_line1": (
                    float(np.mean(values("line1_matched_fraction")))
                    if values("line1_matched_fraction") else None
                ),
                "mean_component_span_fraction_line2": (
                    float(np.mean(values("line2_matched_fraction")))
                    if values("line2_matched_fraction") else None
                ),
            }
        )
        return summary

    runner.evaluate_sample = evaluate_sample
    runner._evaluate_sample = evaluate_sample
    runner.batch_fieldnames = batch_fieldnames
    runner._batch_fieldnames = batch_fieldnames
    runner.aggregate = aggregate
    runner._aggregate = aggregate
    runner._zero_shot_sw_components_installed = True
