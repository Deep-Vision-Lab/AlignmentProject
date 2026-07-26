"""Zero-shot preprocessing, sampling, and ink-aware Smith-Waterman patches."""
from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
import random
import tempfile

import numpy as np
from PIL import Image

from Evaluation import sw_dataset
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

    # Seed every split with one complete group, choosing smaller groups for
    # validation/test so they retain more writer/page diversity.
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
    """Use the shared aspect-preserving preprocessing used by zero-shot training."""
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


_ORIGINAL_BATCH_PAIRS = sw_dataset.batch_pairs


def install_dataset_patches() -> None:
    sw_dataset.group_split_pairs = balanced_group_split_pairs
    sw_dataset._group_split_pairs = balanced_group_split_pairs
    sw_dataset.batch_pairs = balanced_batch_pairs
    sw_dataset.real_binarizer = real_binarizer
    sw_dataset.display_image = display_image
    sw_dataset._display_image = display_image


def install_runner_patches(runner) -> None:
    """Replace only sample scoring; retain the existing CLI and report format."""

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

            features1 = runner.get_image_features(
                models, model_image1, feature_dataset_type
            )
            features2 = runner.get_image_features(
                models, model_image2, feature_dataset_type
            )
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
            path, score, dp_score, traceback = runner.smith_waterman(
                raw_similarity,
                threshold=threshold,
                gap_penalty=gap,
                return_traceback=True,
                match_scores=match_scores,
            )
            displayed_matrix, displayed_label = runner.select_heatmap_matrix(
                raw_similarity,
                threshold,
                dp_score,
                heatmap_source,
                match_scores=match_scores,
                score_mode=resolved_mode + "+ink",
            )
            runner.save_visualization(
                arr1,
                arr2,
                features1,
                features2,
                path,
                traceback,
                displayed_matrix,
                displayed_label,
                score,
                output,
                models.image_model.use_flip,
                binarized,
                pair,
                resolved_mode + "+ink",
                show_heatmap=show_heatmap,
                annotate_values=annotate_values,
                value_decimals=value_decimals,
                annotation_fontsize=annotation_fontsize,
            )
        finally:
            if temporary_directory is not None:
                temporary_directory.cleanup()

        path_cosines = [float(raw_similarity[i, j]) for i, j in path]
        max_row, max_col = map(
            int, np.unravel_index(np.argmax(dp_score), dp_score.shape)
        )
        matched_rows = {row for row, _ in path}
        matched_cols = {col for _, col in path}
        row = {
            "index": int(pair.index),
            "manifest_position": int(pair.manifest_position),
            "pair_id": pair.pair_id,
            "label_type": pair.label_type,
            "text_score": float(pair.text_score),
            "split": pair.split,
            "status": "ok",
            "score": float(score),
            "path_steps": len(path),
            "traceback_steps": max(0, len(traceback) - 1),
            "dp_max_row": max_row,
            "dp_max_col": max_col,
            "dp_max_is_terminal": bool(
                max_row == dp_score.shape[0] - 1
                and max_col == dp_score.shape[1] - 1
            ),
            "mean_path_cosine": float(np.mean(path_cosines)) if path_cosines else 0.0,
            "line1_matched_fraction": len(matched_rows) / max(1, raw_similarity.shape[0]),
            "line2_matched_fraction": len(matched_cols) / max(1, raw_similarity.shape[1]),
            "line1_windows": int(raw_similarity.shape[0]),
            "line2_windows": int(raw_similarity.shape[1]),
            "line1_path_start": int(path[0][0]) if path else -1,
            "line1_path_end": int(path[-1][0]) if path else -1,
            "line2_path_start": int(path[0][1]) if path else -1,
            "line2_path_end": int(path[-1][1]) if path else -1,
            "feature": str(feature),
            "score_mode": resolved_mode,
            "score_clip": float(score_clip),
            "threshold": float(threshold),
            "gap": float(gap),
            "heatmap_source": str(heatmap_source),
            "dataset_type": str(dataset_type),
            "binarized": bool(binarized),
            "binarization": os.environ.get("REAL_BINARIZE_METHOD", "otsu").lower()
            if binarized
            else "none",
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
            f"score={score:.6f} score_mode={resolved_mode}+ink "
            f"dp_max=({max_row},{max_col}) terminal={row['dp_max_is_terminal']} "
            f"path_steps={len(path)} matched_fraction="
            f"({row['line1_matched_fraction']:.3f},{row['line2_matched_fraction']:.3f}) "
            f"mean_cosine={row['mean_path_cosine']:.4f} saved={output}",
            flush=True,
        )
        return row

    runner.evaluate_sample = evaluate_sample
    runner._evaluate_sample = evaluate_sample
