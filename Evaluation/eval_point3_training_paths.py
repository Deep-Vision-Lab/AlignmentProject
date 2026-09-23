#!/usr/bin/env python3
"""Point 3: inspect the actual trained image->letter DTW geometry.

For each fixed synthetic test pair this diagnostic saves:
- exact model inputs using checkpoint training preprocessing;
- the actual letter-DTW cost matrix used by training (full-alphabet NLL or cosine);
- the same position prior used by training;
- a hard minimum-cost path using the same vertical/horizontal transition penalties;
- every path point projected back to the exact 32-pixel physical window at stride 16;
- an image->image fused-cosine hard-DTW path as a separate structural diagnostic.

The image->letter hard path is an argmin interpretation of the Soft-DTW
objective; it is not itself the differentiable Soft-DTW loss.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from Evaluation import eval_img_align_nw_diagnostic as pair_loader
from Evaluation._eval_utils import (
    build_transform,
    compute_similarity,
    load_evaluation_models,
)
from Evaluation.eval_yelda import configure_geometry, synthetic_split
from Evaluation.point3_core import (
    effective_cell_costs as _position_prior,
    hard_letter_path,
    hard_monotonic_path as _hard_monotonic_path,
    sequence_to_physical_window as _physical_window,
)
from Evaluation.yelda_geometry import prepare_line, source_intervals
from vlm_restoration_positive_dtw import (
    _clean_letters,
    _soft_dtw_cost_matrix,
    letter_dtw_cost_matrix,
)


def _source_interval_or_none(x0, x1, geometry):
    """Map one canvas interval to source pixels, tolerating pure padding."""
    mapped = source_intervals([[float(x0), float(x1)]], geometry)
    if not mapped:
        return None, None, False
    return float(mapped[0][0]), float(mapped[0][1]), True


def _write_image_text_path(
    path_file,
    path,
    letters,
    effective_costs,
    *,
    geometry,
    window_size,
    stride,
    use_flip,
    sequence_indices,
    total_window_count,
    physical_indices=None,
):
    """Write DTW rows using original sequence indices after valid-token filtering."""
    width = int(geometry["canvas_width"])
    if len(sequence_indices) != int(effective_costs.shape[0]):
        raise ValueError(
            "Filtered DTW row count does not match sequence-index mapping: "
            f"rows={effective_costs.shape[0]} indices={len(sequence_indices)}"
        )
    with path_file.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "step", "dtw_window_index", "sequence_window", "physical_window",
            "logical_sequence_index", "physical_window_index",
            "canvas_x0", "canvas_x1", "source_x0", "source_x1",
            "source_mapped", "letter_index", "letter", "cell_cost"
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, (i, j) in enumerate(path):
            sequence_index = int(sequence_indices[int(i)])
            physical, x0, x1 = _physical_window(
                sequence_index if physical_indices is None else int(physical_indices[sequence_index]),
                int(total_window_count) if physical_indices is None else 1 + (width-window_size)//stride,
                width=width,
                window=window_size,
                stride=stride,
                use_flip=use_flip if physical_indices is None else False,
            )
            source_x0, source_x1, source_mapped = _source_interval_or_none(
                x0, x1, geometry
            )
            writer.writerow({
                "step": step,
                "dtw_window_index": int(i),
                "sequence_window": sequence_index,
                "physical_window": physical,
                "logical_sequence_index": sequence_index,
                "physical_window_index": physical,
                "canvas_x0": x0,
                "canvas_x1": x1,
                "source_x0": source_x0,
                "source_x1": source_x1,
                "source_mapped": int(source_mapped),
                "letter_index": j,
                "letter": letters[j],
                "cell_cost": float(effective_costs[i, j]),
            })


def _hard_image_image(similarity):
    costs = 1.0 - np.asarray(similarity, dtype=np.float64)
    return _hard_monotonic_path(
        costs,
        vertical_penalty=0.0,
        horizontal_penalty=0.0,
        disable_horizontal_when_feasible=False,
    )[0]


def _write_image_image_path(
    path_file,
    path,
    similarity,
    geometry1,
    geometry2,
    *,
    window_size,
    stride,
    use_flip,
    line1_sequence_indices,
    line2_sequence_indices,
    line1_total_window_count,
    line2_total_window_count,
):
    """Write image-image path with filtered DTW indices mapped to original windows."""
    n, m = similarity.shape
    if len(line1_sequence_indices) != n or len(line2_sequence_indices) != m:
        raise ValueError(
            "Filtered image-image matrix does not match sequence-index maps: "
            f"matrix={similarity.shape}, maps="
            f"{len(line1_sequence_indices)},{len(line2_sequence_indices)}"
        )
    with path_file.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "step",
            "line1_dtw_window_index", "line1_sequence_window",
            "line1_physical_window", "line1_canvas_x0", "line1_canvas_x1",
            "line1_logical_sequence_index", "line1_physical_window_index",
            "line1_source_x0", "line1_source_x1", "line1_source_mapped",
            "line2_dtw_window_index", "line2_sequence_window",
            "line2_physical_window", "line2_canvas_x0", "line2_canvas_x1",
            "line2_logical_sequence_index", "line2_physical_window_index",
            "line2_source_x0", "line2_source_x1", "line2_source_mapped",
            "cosine",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for step, (i, j) in enumerate(path):
            seq1 = int(line1_sequence_indices[int(i)])
            seq2 = int(line2_sequence_indices[int(j)])
            p1, a0, a1 = _physical_window(
                seq1,
                int(line1_total_window_count),
                width=int(geometry1["canvas_width"]),
                window=window_size,
                stride=stride,
                use_flip=use_flip,
            )
            p2, b0, b1 = _physical_window(
                seq2,
                int(line2_total_window_count),
                width=int(geometry2["canvas_width"]),
                window=window_size,
                stride=stride,
                use_flip=use_flip,
            )
            s1x0, s1x1, s1mapped = _source_interval_or_none(a0, a1, geometry1)
            s2x0, s2x1, s2mapped = _source_interval_or_none(b0, b1, geometry2)
            writer.writerow({
                "step": step,
                "line1_dtw_window_index": int(i),
                "line1_sequence_window": seq1,
                "line1_physical_window": p1,
                "line1_logical_sequence_index": seq1,
                "line1_physical_window_index": p1,
                "line1_canvas_x0": a0,
                "line1_canvas_x1": a1,
                "line1_source_x0": s1x0,
                "line1_source_x1": s1x1,
                "line1_source_mapped": int(s1mapped),
                "line2_dtw_window_index": int(j),
                "line2_sequence_window": seq2,
                "line2_physical_window": p2,
                "line2_logical_sequence_index": seq2,
                "line2_physical_window_index": p2,
                "line2_canvas_x0": b0,
                "line2_canvas_x1": b1,
                "line2_source_x0": s2x0,
                "line2_source_x1": s2x1,
                "line2_source_mapped": int(s2mapped),
                "cosine": float(similarity[i, j]),
            })


def _heatmap(matrix, path, output, title, xlabel, ylabel, *, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(14, 8))
    image = ax.imshow(matrix, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    ax.plot([j for i, j in path], [i for i, j in path], linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.colorbar(image, ax=ax)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _transcript_path(dataset, pair, side):
    dataset_index = int(pair.manifest_position)
    return Path(dataset) / "texts" / f"text{side}_{dataset_index}.txt"


def _extract_training_bundle(models, image_path):
    from PIL import Image
    with Image.open(image_path) as opened:
        tensor = build_transform(contract=models.contract, prepared=True)(opened).unsqueeze(0).to(models.device)
    with torch.inference_mode():
        bundle = models.image_model(tensor, return_training_bundle=True)
    return bundle


def _namespace(config):
    defaults = {
        "positive_letter_dtw_cost_mode": "full_alphabet_nll",
        "positive_letter_dtw_competition_temperature": 0.10,
        "positive_letter_dtw_vertical_penalty": 0.05,
        "positive_letter_dtw_horizontal_penalty": 0.30,
        "positive_letter_dtw_position_prior": 0.15,
        "positive_letter_dtw_disable_horizontal_when_feasible": True,
        "positive_letter_dtw_gamma_end": 0.05,
    }
    defaults.update(config)
    return SimpleNamespace(**defaults)


def monitoring_preview(output, record, transcript, image, bundle, index, model, text_encoder, P, config):
    """Fixed-ID epoch preview using the same scientific core and coordinate writer."""
    from PIL import Image
    from Evaluation.checkpoint_contract import resolve_evaluation_contract
    output.mkdir(parents=True, exist_ok=True)
    contract = resolve_evaluation_contract(config)
    source = Path(record["_root"]) / record["line_image_path"]
    prepared, geometry = contract.prepare_line(source, domain="real", preprocessing="training")
    expected = contract.tensor_transform()(prepared)
    if not torch.equal(expected, image.cpu()):
        raise ValueError(f"Monitoring preview/training pixel contract differs: {source}")
    with Image.open(source) as original:
        original.save(output / "source.png")
    prepared.save(output / "model_input.png")
    letters = _clean_letters(transcript)
    valid = bundle["token_valid"][index].bool()
    raw = letter_dtw_cost_matrix(P, text_encoder, bundle["semantic"][index][valid], letters).float().cpu().numpy()
    hard = hard_letter_path(raw,
        vertical_penalty=P.positive_letter_dtw_vertical_penalty,
        horizontal_penalty=P.positive_letter_dtw_horizontal_penalty,
        position_prior_weight=P.positive_letter_dtw_position_prior,
        disable_horizontal_when_feasible=P.positive_letter_dtw_disable_horizontal_when_feasible)
    np.save(output / "training_cell_cost.npy", raw)
    np.save(output / "effective_cell_cost.npy", hard.effective_costs)
    _write_image_text_path(output / "path.csv", hard.path, letters, hard.effective_costs,
        geometry=geometry, window_size=contract.window_size, stride=contract.stride,
        use_flip=model.use_flip, sequence_indices=torch.where(valid)[0].tolist(),
        total_window_count=len(valid), physical_indices=(bundle["physical_window_indices"][index]
                                                       if "physical_window_indices" in bundle else None))
    _heatmap(hard.effective_costs, hard.path, output / "cost_path.png",
        "Training cost + unchanged DTW prior; hard argmin (not localization accuracy)",
        "Transcript letter index", "Logical window index")
    (output / "metadata.json").write_text(json.dumps(dict(record=record, transcript=transcript,
        letters=letters, geometry=geometry, hard_objective_normalized=hard.hard_objective_normalized,
        transformer_position_mode=config.get("position_mode"), dtw_position_prior=P.positive_letter_dtw_position_prior),
        ensure_ascii=False, indent=2))
    (output / "transcript.txt").write_text(transcript, encoding="utf-8")


def select_pairs(dataset, split, training_samples, split_seed, start_index, n_samples):
    layout, pairs = pair_loader.load_pairs(Path(dataset), split)
    if layout != "synthetic":
        raise ValueError("Point-3 training-path validation currently requires the synthetic dataset")
    pairs = synthetic_split(pairs, split, training_samples, split_seed)
    start = start_index - 1
    selected = pairs[start:] if n_samples == 0 else pairs[start:start + n_samples]
    if not selected:
        raise ValueError("No pairs selected")
    return selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", choices=("train", "valid", "test"), default="test")
    ap.add_argument("--training-samples", type=int, default=6000)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--start-index", type=int, default=1)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image-preprocessing", choices=("original", "training", "cropped_1024"), default="training")
    args = ap.parse_args()

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    models = load_evaluation_models(args.weights, args.device, load_text_model=True)
    if models.text_model is None:
        raise RuntimeError("Point 3 requires the frozen training character codebook")
    if str(models.config.get("text_encoder_type", "")) != "char":
        raise RuntimeError("Point 3 expects the restoration character-codebook checkpoint")

    configure_geometry(models.config, args.image_preprocessing)
    P = _namespace(models.config)
    window_size = int(models.config.get("window_size", 32))
    stride = int(models.config.get("stride", 16))
    use_flip = bool(models.image_model.use_flip)
    selected = select_pairs(
        args.dataset, args.split, args.training_samples, args.split_seed,
        args.start_index, args.n_samples
    )
    pair_rows = []

    for pair in selected:
        pair_dir = output / f"pair_{int(pair.manifest_position):05d}"
        pair_dir.mkdir(parents=True, exist_ok=True)
        prepared = []
        geometries = []
        bundles = []
        valid_sequence_indices = []
        total_window_counts = []
        letter_objectives = []
        for side in (1, 2):
            image, geometry = prepare_line(
                getattr(pair, f"image{side}"),
                pair.preprocess_domain(side),
                args.image_preprocessing,
                contract=models.contract,
            )
            image_path = pair_dir / f"line{side}_model_input.png"
            image.save(image_path)
            prepared.append(image_path)
            geometries.append(geometry)
            bundles.append(_extract_training_bundle(models, image_path))

            text_path = _transcript_path(args.dataset, pair, side)
            if not text_path.is_file():
                raise FileNotFoundError(f"Missing transcript: {text_path}")
            text = text_path.read_text(encoding="utf-8")
            letters = _clean_letters(text)
            visual_full = bundles[-1]["semantic"][0]
            valid = bundles[-1]["token_valid"][0].bool()
            indices = torch.where(valid)[0]
            if int(indices.numel()) == 0:
                raise RuntimeError(
                    f"No valid visual tokens for pair={pair.pair_id} side={side}"
                )
            valid_sequence_indices.append([int(v) for v in indices.tolist()])
            total_window_counts.append(int(visual_full.shape[0]))
            visual = visual_full[indices]
            with torch.inference_mode():
                raw = letter_dtw_cost_matrix(P, models.text_model, visual, letters)
                soft = _soft_dtw_cost_matrix(
                    raw,
                    gamma=float(getattr(P, "positive_letter_dtw_gamma_end", 0.05)),
                    vertical_penalty=float(P.positive_letter_dtw_vertical_penalty),
                    horizontal_penalty=float(P.positive_letter_dtw_horizontal_penalty),
                    position_prior_weight=float(P.positive_letter_dtw_position_prior),
                    disable_horizontal_when_feasible=bool(
                        P.positive_letter_dtw_disable_horizontal_when_feasible
                    ),
                )
            raw_np = raw.detach().cpu().numpy()
            hard = hard_letter_path(
                raw_np,
                vertical_penalty=float(P.positive_letter_dtw_vertical_penalty),
                horizontal_penalty=float(P.positive_letter_dtw_horizontal_penalty),
                position_prior_weight=float(P.positive_letter_dtw_position_prior),
                disable_horizontal_when_feasible=bool(
                    P.positive_letter_dtw_disable_horizontal_when_feasible
                ),
            )
            effective, path, dp = hard.effective_costs, hard.path, hard.dp
            letter_objectives.append(hard)
            np.save(pair_dir / f"line{side}_letter_cost_raw.npy", raw_np.astype(np.float32))
            np.savetxt(pair_dir / f"line{side}_letter_cost_raw.csv", raw_np, delimiter=",")
            np.save(pair_dir / f"line{side}_letter_cost_effective.npy", effective.astype(np.float32))
            np.savetxt(pair_dir / f"line{side}_letter_cost_effective.csv", effective, delimiter=",")
            np.savetxt(pair_dir / f"line{side}_hard_dtw_dp.csv", dp, delimiter=",")
            _write_image_text_path(
                pair_dir / f"line{side}_image_to_text_path.csv",
                path,
                letters,
                effective,
                geometry=geometry,
                window_size=window_size,
                stride=stride,
                use_flip=use_flip,
                sequence_indices=valid_sequence_indices[-1],
                total_window_count=total_window_counts[-1],
                physical_indices=(bundles[-1]["physical_window_indices"][0]
                                  if "physical_window_indices" in bundles[-1] else None),
            )
            _heatmap(
                effective,
                path,
                pair_dir / f"line{side}_image_to_text_path.png",
                f"Line {side}: actual training cost + hard minimum path; "
                f"soft-DTW={float(soft):.5f}; hard={hard.hard_objective_normalized:.5f}",
                "Transcript letter index",
                "Image window index (model sequence order)",
            )
            (pair_dir / f"line{side}_letters.json").write_text(
                json.dumps({"letters": letters}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        first = bundles[0]["semantic"][0][
            torch.tensor(valid_sequence_indices[0], device=bundles[0]["semantic"].device)
        ]
        second = bundles[1]["semantic"][0][
            torch.tensor(valid_sequence_indices[1], device=bundles[1]["semantic"].device)
        ]
        similarity = compute_similarity(first, second).detach().cpu().numpy()
        image_path = _hard_image_image(similarity)
        np.save(pair_dir / "image_to_image_cosine.npy", similarity.astype(np.float32))
        np.savetxt(pair_dir / "image_to_image_cosine.csv", similarity, delimiter=",")
        _write_image_image_path(
            pair_dir / "image_to_image_path.csv",
            image_path,
            similarity,
            geometries[0],
            geometries[1],
            window_size=window_size,
            stride=stride,
            use_flip=use_flip,
            line1_sequence_indices=valid_sequence_indices[0],
            line2_sequence_indices=valid_sequence_indices[1],
            line1_total_window_count=total_window_counts[0],
            line2_total_window_count=total_window_counts[1],
        )
        _heatmap(
            similarity,
            image_path,
            pair_dir / "image_to_image_path.png",
            "Fused image-to-image hard DTW (structural diagnostic only)",
            "Line 2 window index",
            "Line 1 window index",
            vmin=-1.0,
            vmax=1.0,
        )
        pair_rows.append({
            "pair_id": pair.pair_id,
            "dataset_index": int(pair.manifest_position),
            "line1_windows": int(similarity.shape[0]),
            "line2_windows": int(similarity.shape[1]),
            "line1_mean_path_cell_cost": letter_objectives[0].mean_path_cell_cost,
            "line2_mean_path_cell_cost": letter_objectives[1].mean_path_cell_cost,
            "line1_hard_objective_total": letter_objectives[0].hard_objective_total,
            "line2_hard_objective_total": letter_objectives[1].hard_objective_total,
            "line1_hard_objective_normalized": letter_objectives[0].hard_objective_normalized,
            "line2_hard_objective_normalized": letter_objectives[1].hard_objective_normalized,
            "image_image_path_points": len(image_path),
        })

    with (output / "point3_pairs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0]))
        writer.writeheader()
        writer.writerows(pair_rows)

    metadata = {
        "point": 3,
        "purpose": "verify training DTW chooses plausible physical locations",
        "checkpoint": str(Path(args.weights).resolve()),
        "dataset": str(Path(args.dataset).resolve()),
        "pairs": len(selected),
        "image_preprocessing": args.image_preprocessing,
        "preprocessing_override": args.image_preprocessing != "training",
        "window_size": window_size,
        "stride": stride,
        "use_flip": use_flip,
        "training_cost_mode": str(P.positive_letter_dtw_cost_mode),
        "competition_temperature": float(P.positive_letter_dtw_competition_temperature),
        "vertical_penalty": float(P.positive_letter_dtw_vertical_penalty),
        "horizontal_penalty": float(P.positive_letter_dtw_horizontal_penalty),
        "position_prior": float(P.positive_letter_dtw_position_prior),
        "disable_horizontal_when_feasible": bool(
            P.positive_letter_dtw_disable_horizontal_when_feasible
        ),
        "hard_path_note": (
            "The saved image-to-letter hard path is the minimum-cost path under the "
            "same cell costs and transition penalties; training itself used Soft-DTW."
        ),
        "source_mapping_note": (
            "DTW rows are mapped through the original pre-filter sequence index. "
            "Windows entirely inside preprocessing padding are retained in the "
            "diagnostic with source_mapped=0 and blank source coordinates."
        ),
        "image_to_image_note": (
            "Image-to-image hard DTW is a structural diagnostic and was not a training loss."
        ),
    }
    from Evaluation.checkpoint_contract import evaluation_metadata
    metadata["evaluation_contract"] = evaluation_metadata(models, args.weights, args.image_preprocessing, args.dataset)
    (output / "summary.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
