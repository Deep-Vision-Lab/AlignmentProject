#!/usr/bin/env python3
"""Held-out real Arabic image-to-text retrieval diagnostic for Span-DTW checkpoints.

The diagnostic deliberately scores the contextual image sequence used by the
training image-text loss against the correct transcript and deterministic wrong
held-out transcripts. Lower normalized Span-DTW cost is better.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import DataLoader as DL
import Parameters as P
from RealDataSet import ArabicManifestLinePairDataset
from span_alignment_loss import SpanContrastiveSoftDTW
from Evaluation import _eval_utils

try:
    from Evaluation.vit_evaluation import install_vit_evaluation_loader
except ImportError:  # CNN-only historical branches remain supported.
    install_vit_evaluation_loader = None


LABELS = ("high_match", "medium_match")
SAMPLE_FIELDS = [
    "query_index",
    "pair_id",
    "side",
    "label_type",
    "text_score",
    "status",
    "image_path",
    "text_path",
    "correct_text",
    "correct_cost",
    "correct_norm_cost",
    "hardest_negative_cost",
    "hardest_negative_norm_cost",
    "mean_negative_norm_cost",
    "margin",
    "rank",
    "reciprocal_rank",
    "candidate_count",
    "negative_count",
    "infeasible_negative_count",
    "pos_probability",
    "image_steps",
    "correct_text_steps",
    "hardest_negative_text",
    "visual_encoder_type",
    "dtw_backend",
    "error",
]
CANDIDATE_FIELDS = [
    "query_index",
    "pair_id",
    "side",
    "label_type",
    "is_correct",
    "candidate_index",
    "candidate_text",
    "candidate_text_path",
    "candidate_pair_id",
    "candidate_side",
    "raw_cost",
    "norm_cost",
    "text_steps",
    "feasible",
]
SUMMARY_FIELDS = [
    "group",
    "queries",
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "mrr",
    "mean_correct_norm_cost",
    "mean_hardest_negative_norm_cost",
    "mean_margin",
    "mean_pos_probability",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-dir", default="DataSet/ArabicDataset")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=60,
        help="0 evaluates every held-out pair row",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=50,
        help="0 uses every distinct wrong held-out transcript",
    )
    parser.add_argument("--candidate-batch-size", type=int, default=16)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--negative-seed", type=int, default=42)
    parser.add_argument("--text-key", default="text_original_path")
    return parser.parse_args()


def _round_robin_rows(rows):
    groups = OrderedDict()
    for position, row in enumerate(rows):
        groups.setdefault(
            str(row.get("pair_id", f"sample_{position}")), []
        ).append(row)
    offsets = {key: 0 for key in groups}
    ordered = []
    while True:
        added = False
        for key, members in groups.items():
            offset = offsets[key]
            if offset < len(members):
                ordered.append(members[offset])
                offsets[key] += 1
                added = True
        if not added:
            return ordered


def _resolve_manifest(data_dir: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    return (
        data_dir
        / os.environ.get("REAL_MANIFEST_NAME", "dataset_manifest.jsonl")
    ).resolve()


def _text_record(dataset, row, side_name):
    side = row[side_name]
    text_path = dataset._resolve(side[dataset.text_key])
    text = dataset._read_text(side[dataset.text_key])
    return {
        "text": text,
        "key": text.strip(),
        "text_path": str(text_path),
        "pair_id": str(row.get("pair_id", "")),
        "side": side_name,
    }


def _query_record(dataset, row, side_name, label_type, text_score):
    side = row[side_name]
    text_record = _text_record(dataset, row, side_name)
    return {
        **text_record,
        "image_path": str(dataset._resolve(side["line_image_path"])),
        "label_type": label_type,
        "text_score": float(text_score),
    }


def _unique_text_pool(dataset, rows):
    unique = OrderedDict()
    for row in rows:
        for side_name in ("A", "B"):
            item = _text_record(dataset, row, side_name)
            if item["key"]:
                unique.setdefault(item["key"], item)
    return list(unique.values())


def _build_criterion(config):
    return SpanContrastiveSoftDTW(
        gamma=float(
            config.get(
                "contrastive_soft_dtw_gamma", P.contrastive_soft_dtw_gamma
            )
        ),
        margin=float(config.get("contrastive_margin", P.contrastive_margin)),
        temperature=float(
            config.get(
                "contrastive_temperature", P.contrastive_temperature
            )
        ),
        max_windows_per_span=int(
            config.get("max_windows_per_span", P.max_windows_per_span)
        ),
        window_count_penalty=float(
            config.get(
                "span_window_count_penalty",
                os.environ.get("SPAN_WINDOW_COUNT_PENALTY", "0.05"),
            )
        ),
        negative_grad_mode="hardest",
        backend="torch",
    )


def _image_embedding(models, dataset, image_path):
    image = dataset._read_image(image_path).unsqueeze(0).to(models.device)
    with torch.no_grad():
        contextual = models.image_model(image)
        if isinstance(contextual, (tuple, list)):
            contextual = contextual[0]
        contextual = F.normalize(contextual.float(), p=2, dim=-1)
    return contextual[0]


def _score_candidates(
    criterion,
    text_model,
    image_embedding,
    candidates,
    batch_size,
):
    rows = []
    image_steps = int(image_embedding.shape[0])
    for start in range(0, len(candidates), batch_size):
        batch = candidates[start : start + batch_size]
        texts = [item["text"] for item in batch]
        with torch.no_grad():
            encodings = criterion._encode_many(
                text_model, texts, use_cache=True
            )
            repeated = image_embedding.unsqueeze(0).expand(
                len(batch), -1, -1
            )
            costs = criterion._costs_allowing_infeasible(
                encodings, repeated
            )
        for local_index, (item, encoding, cost) in enumerate(
            zip(batch, encodings, costs)
        ):
            try:
                criterion._check_path_feasible(encoding, image_steps)
                feasible = True
            except ValueError:
                feasible = False
            raw_cost = float(cost.detach().cpu().item())
            text_steps = int(encoding.text_length)
            norm_cost = raw_cost / max(text_steps, image_steps)
            rows.append(
                {
                    **item,
                    "raw_cost": raw_cost,
                    "norm_cost": float(norm_cost),
                    "text_steps": text_steps,
                    "feasible": feasible,
                    "candidate_index": start + local_index,
                }
            )
    return rows


def _sample_negatives(pool, correct_key, count, seed):
    eligible = [item for item in pool if item["key"] != correct_key]
    if not eligible:
        raise ValueError(
            "Held-out text pool has no distinct negative transcript"
        )
    if count <= 0 or count >= len(eligible):
        return eligible
    return random.Random(int(seed)).sample(eligible, int(count))


def _auc_from_costs(positive_costs, negative_costs):
    pos = np.asarray(positive_costs, dtype=np.float64)
    neg = np.asarray(negative_costs, dtype=np.float64)
    if pos.size == 0 or neg.size == 0:
        return None
    # Lower cost is a better retrieval score. Equivalent to AUROC on -cost.
    wins = 0.0
    for value in pos:
        wins += float(np.sum(value < neg))
        wins += 0.5 * float(np.sum(value == neg))
    return wins / float(pos.size * neg.size)


def _aggregate(rows):
    rows = [row for row in rows if row.get("status") == "ok"]
    if not rows:
        return {
            "queries": 0,
            "recall_at_1": None,
            "recall_at_5": None,
            "recall_at_10": None,
            "mrr": None,
            "mean_correct_norm_cost": None,
            "mean_hardest_negative_norm_cost": None,
            "mean_margin": None,
            "mean_pos_probability": None,
        }

    def mean(key):
        return float(np.mean([float(row[key]) for row in rows]))

    ranks = np.asarray(
        [float(row["rank"]) for row in rows], dtype=np.float64
    )
    return {
        "queries": len(rows),
        "recall_at_1": float(np.mean(ranks <= 1)),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "mrr": mean("reciprocal_rank"),
        "mean_correct_norm_cost": mean("correct_norm_cost"),
        "mean_hardest_negative_norm_cost": mean(
            "hardest_negative_norm_cost"
        ),
        "mean_margin": mean("margin"),
        "mean_pos_probability": mean("pos_probability"),
    }


def _write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.n_pairs < 0:
        raise SystemExit("--n-pairs must be >= 0")
    if args.num_negatives < 0:
        raise SystemExit("--num-negatives must be >= 0")
    if args.candidate_batch_size <= 0:
        raise SystemExit("--candidate-batch-size must be > 0")

    os.environ["DATASET_SPLIT_SEED"] = str(args.split_seed)
    os.environ["REAL_TEXT_KEY"] = str(args.text_key)
    # DataLoader captures this at import time; guard against a mismatched shell.
    DL._split_seed = int(args.split_seed)

    data_dir = Path(args.data_dir).expanduser().resolve()
    manifest = _resolve_manifest(data_dir, args.manifest)
    if not manifest.is_file():
        raise SystemExit(f"Real manifest not found: {manifest}")

    if install_vit_evaluation_loader is not None:
        install_vit_evaluation_loader()
    models = _eval_utils.load_evaluation_models(
        args.weights, args.device, load_text_model=True
    )
    if models.text_model is None:
        raise SystemExit("Checkpoint did not reconstruct a text encoder")
    text_type = str(
        models.config.get("text_encoder_type", "arabic_span")
    )
    if text_type != "arabic_span":
        raise SystemExit(
            "Image-text retrieval diagnostic currently requires "
            "text_encoder_type=arabic_span; "
            f"checkpoint records {text_type!r}."
        )

    dataset = ArabicManifestLinePairDataset(
        manifest_path=manifest,
        transform=DL.real_transform,
        text_key=args.text_key,
        allowed_labels=LABELS,
        max_samples=None,
        paired=True,
        min_text_score=0.0,
        validate_paths=False,
    )
    _, _, test_subset = DL._group_split_real_dataset(dataset)
    test_indices = [int(index) for index in test_subset.indices]
    test_rows = [dataset.samples[index] for index in test_indices]
    ordered_test_rows = _round_robin_rows(test_rows)
    selected_rows = (
        ordered_test_rows
        if args.n_pairs == 0
        else ordered_test_rows[: min(args.n_pairs, len(ordered_test_rows))]
    )
    if not selected_rows:
        raise SystemExit("No held-out positive pair rows were selected")

    text_pool = _unique_text_pool(dataset, test_rows)
    if len(text_pool) < 2:
        raise SystemExit(
            "Held-out test split does not contain enough distinct transcripts"
        )

    visual_type = str(
        models.config.get("visual_encoder_type", "cnn_bilstm")
    ).strip().lower()
    criterion = _build_criterion(models.config).to(models.device)
    criterion.eval()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_manifest = output_dir / "selected_queries.jsonl"
    samples = []
    candidate_rows = []
    positive_costs = []
    negative_costs = []

    with selected_manifest.open("w", encoding="utf-8") as selected_handle:
        query_index = 0
        for row in selected_rows:
            label_type = str(row.get("label_type", ""))
            text_score = float(
                (row.get("scores") or {}).get("text_score", 0.0)
            )
            for side_name in ("A", "B"):
                query_index += 1
                query = _query_record(
                    dataset,
                    row,
                    side_name,
                    label_type,
                    text_score,
                )
                selected_handle.write(
                    json.dumps(
                        {
                            "query_index": query_index,
                            "pair_id": query["pair_id"],
                            "side": side_name,
                            "label_type": label_type,
                            "image_path": query["image_path"],
                            "text_path": query["text_path"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                base_row = {
                    "query_index": query_index,
                    "pair_id": query["pair_id"],
                    "side": side_name,
                    "label_type": label_type,
                    "text_score": text_score,
                    "status": "error",
                    "image_path": query["image_path"],
                    "text_path": query["text_path"],
                    "correct_text": query["text"].strip(),
                    "visual_encoder_type": visual_type,
                    "dtw_backend": "torch",
                    "error": "",
                }
                try:
                    image_embedding = _image_embedding(
                        models, dataset, query["image_path"]
                    )
                    negatives = _sample_negatives(
                        text_pool,
                        query["key"],
                        args.num_negatives,
                        args.negative_seed + query_index * 10007,
                    )
                    candidates = [
                        {
                            **query,
                            "is_correct": True,
                        }
                    ] + [
                        {
                            **item,
                            "is_correct": False,
                            "image_path": "",
                            "label_type": "",
                            "text_score": 0.0,
                        }
                        for item in negatives
                    ]
                    scored = _score_candidates(
                        criterion,
                        models.text_model,
                        image_embedding,
                        candidates,
                        args.candidate_batch_size,
                    )
                    correct = scored[0]
                    if not correct["feasible"]:
                        raise ValueError(
                            "Correct transcript is infeasible for the "
                            "checkpoint Span-DTW geometry"
                        )
                    neg_scored = scored[1:]
                    hardest = min(
                        neg_scored, key=lambda item: item["norm_cost"]
                    )
                    mean_neg = float(
                        np.mean([item["norm_cost"] for item in neg_scored])
                    )
                    less = sum(
                        item["norm_cost"] < correct["norm_cost"]
                        for item in neg_scored
                    )
                    rank = int(1 + less)
                    all_norm = torch.as_tensor(
                        [item["norm_cost"] for item in scored],
                        dtype=torch.float64,
                    )
                    pos_probability = float(
                        torch.softmax(-all_norm, dim=0)[0].item()
                    )

                    base_row.update(
                        {
                            "status": "ok",
                            "correct_cost": correct["raw_cost"],
                            "correct_norm_cost": correct["norm_cost"],
                            "hardest_negative_cost": hardest["raw_cost"],
                            "hardest_negative_norm_cost": hardest[
                                "norm_cost"
                            ],
                            "mean_negative_norm_cost": mean_neg,
                            "margin": hardest["norm_cost"]
                            - correct["norm_cost"],
                            "rank": rank,
                            "reciprocal_rank": 1.0 / rank,
                            "candidate_count": len(scored),
                            "negative_count": len(neg_scored),
                            "infeasible_negative_count": sum(
                                not item["feasible"]
                                for item in neg_scored
                            ),
                            "pos_probability": pos_probability,
                            "image_steps": int(image_embedding.shape[0]),
                            "correct_text_steps": correct["text_steps"],
                            "hardest_negative_text": hardest["text"].strip(),
                        }
                    )
                    positive_costs.append(correct["norm_cost"])
                    negative_costs.extend(
                        item["norm_cost"] for item in neg_scored
                    )

                    for item in scored:
                        candidate_rows.append(
                            {
                                "query_index": query_index,
                                "pair_id": query["pair_id"],
                                "side": side_name,
                                "label_type": label_type,
                                "is_correct": bool(item["is_correct"]),
                                "candidate_index": item[
                                    "candidate_index"
                                ],
                                "candidate_text": item["text"].strip(),
                                "candidate_text_path": item["text_path"],
                                "candidate_pair_id": item["pair_id"],
                                "candidate_side": item["side"],
                                "raw_cost": item["raw_cost"],
                                "norm_cost": item["norm_cost"],
                                "text_steps": item["text_steps"],
                                "feasible": bool(item["feasible"]),
                            }
                        )
                    print(
                        f"[{query_index}] pair_id={query['pair_id']} "
                        f"side={side_name} label={label_type} "
                        f"rank={rank}/{len(scored)} "
                        f"correct={correct['norm_cost']:.4f} "
                        f"hardneg={hardest['norm_cost']:.4f} "
                        f"margin={base_row['margin']:.4f}",
                        flush=True,
                    )
                except Exception as exc:
                    base_row["error"] = f"{type(exc).__name__}: {exc}"
                    print(
                        f"[{query_index}] pair_id={query['pair_id']} "
                        f"side={side_name} FAILED: {base_row['error']}",
                        flush=True,
                    )
                samples.append(base_row)

    _write_csv(output_dir / "samples.csv", SAMPLE_FIELDS, samples)
    _write_csv(
        output_dir / "candidate_scores.csv",
        CANDIDATE_FIELDS,
        candidate_rows,
    )

    successful = [
        row for row in samples if row.get("status") == "ok"
    ]
    groups = OrderedDict()
    groups["all"] = successful
    for label in LABELS:
        groups[label] = [
            row for row in successful if row["label_type"] == label
        ]
    for side_name in ("A", "B"):
        groups[f"side_{side_name}"] = [
            row for row in successful if row["side"] == side_name
        ]

    group_summaries = {
        name: _aggregate(rows) for name, rows in groups.items()
    }
    overall = dict(group_summaries["all"])
    overall.update(
        {
            "requested_pair_rows": args.n_pairs,
            "selected_pair_rows": len(selected_rows),
            "selected_queries": len(samples),
            "successful_queries": len(successful),
            "failed_queries": len(samples) - len(successful),
            "heldout_pair_rows": len(test_rows),
            "heldout_unique_transcripts": len(text_pool),
            "num_negatives_requested": args.num_negatives,
            "candidate_batch_size": args.candidate_batch_size,
            "split_seed": args.split_seed,
            "negative_seed": args.negative_seed,
            "text_key": args.text_key,
            "visual_encoder_type": visual_type,
            "dtw_backend": "torch",
            "positive_vs_negative_auc": _auc_from_costs(
                positive_costs, negative_costs
            ),
            "checkpoint": str(
                Path(args.weights).expanduser().resolve()
            ),
            "manifest": str(manifest),
        }
    )
    summary = {"overall": overall, "groups": group_summaries}
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_rows = [
        {"group": name, **values}
        for name, values in group_summaries.items()
    ]
    _write_csv(
        output_dir / "summary.csv",
        SUMMARY_FIELDS,
        summary_rows,
    )

    print(json.dumps(overall, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
