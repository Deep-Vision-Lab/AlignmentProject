#!/usr/bin/env python3
"""Point 6: same-window nearest-neighbor comparison before/after ViT.

For exactly the same query windows, compare cross-line nearest neighbors using:
  local   = ResNet18 window vector before TinyViT
  context = raw TinyViT contextual vector
  fused   = trained Local+Context fusion vector

Candidates always come from a DIFFERENT source line. Same-line/overlapping
windows are therefore excluded by construction. This is a representation
diagnostic: without independent semantic labels it must not be reported as
nearest-neighbor accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from Evaluation import eval_img_align_nw_diagnostic as pair_loader
from Evaluation._eval_utils import build_transform
from Evaluation.eval_yelda import synthetic_split
from Evaluation.point2_runtime import load_point2_visual_models
from Evaluation.yelda_geometry import prepare_line
from Evaluation.yelda_runtime import read_checkpoint


@dataclass
class LineRecord:
    line_id: str
    pair_id: str
    side: int
    source_path: Path
    image: np.ndarray
    local: torch.Tensor
    context: torch.Tensor
    fused: torch.Tensor
    valid: torch.Tensor
    ink: torch.Tensor
    seq_to_physical: list[int]
    window_size: int
    stride: int

    def representation(self, name: str) -> torch.Tensor:
        return {"local": self.local, "context": self.context, "fused": self.fused}[name]


def _finite_mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def _ink_fraction_windows(image: np.ndarray, window: int, stride: int) -> np.ndarray:
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    h, w = gray.shape
    border_h = max(1, int(round(h * 0.05)))
    border_w = max(1, int(round(w * 0.02)))
    border = np.concatenate(
        [
            gray[:border_h].reshape(-1),
            gray[-border_h:].reshape(-1),
            gray[:, :border_w].reshape(-1),
            gray[:, -border_w:].reshape(-1),
        ]
    )
    background = float(np.median(border))
    foreground = np.abs(gray - background) >= 0.08
    count = max(1, 1 + max(0, w - int(window)) // int(stride))
    fractions = []
    for physical_index in range(count):
        x0 = physical_index * int(stride)
        x1 = min(w, x0 + int(window))
        fractions.append(float(np.mean(foreground[:, x0:x1])))
    return np.asarray(fractions, dtype=np.float32)


def _encode_line(models, image: Image.Image, source_path: Path, pair_id: str, side: int):
    tensor = build_transform("synthetic")(image.convert("RGB")).unsqueeze(0).to(models.device)
    model = models.image_model
    vit = model.vit_encoder
    encoder = getattr(vit, "encode_restoration_sequence", None)
    if encoder is None:
        raise RuntimeError("Point 6 requires the original restoration ResNet->TinyViT encoder")

    with torch.inference_mode():
        fused_raw, local_raw, context_raw, _model_input, token_valid = encoder(
            tensor, use_flip=model.use_flip
        )
        local = F.normalize(model.vision_norm(local_raw).float(), p=2, dim=-1)[0].cpu()
        context = F.normalize(model.vision_norm(context_raw).float(), p=2, dim=-1)[0].cpu()
        fused = F.normalize(model.vision_norm(fused_raw).float(), p=2, dim=-1)[0].cpu()
        valid = token_valid[0].bool().cpu()

    array = np.asarray(image.convert("RGB"))
    window = int(models.config.get("window_size", 32))
    stride = int(models.config.get("stride", 16))
    physical_ink = _ink_fraction_windows(array, window, stride)
    count = int(local.shape[0])
    if len(physical_ink) != count:
        raise RuntimeError(
            f"Window-count mismatch for {source_path}: pixels={len(physical_ink)} model={count}"
        )
    if bool(model.use_flip):
        seq_to_physical = list(reversed(range(count)))
        ink = torch.from_numpy(physical_ink[::-1].copy())
    else:
        seq_to_physical = list(range(count))
        ink = torch.from_numpy(physical_ink.copy())

    return LineRecord(
        line_id=f"{pair_id}:side{side}:{source_path}",
        pair_id=str(pair_id),
        side=int(side),
        source_path=source_path,
        image=array,
        local=local,
        context=context,
        fused=fused,
        valid=valid,
        ink=ink,
        seq_to_physical=seq_to_physical,
        window_size=window,
        stride=stride,
    )


def _select_pairs(args):
    pair_loader.P.real_dataset_labels = args.labels
    pair_loader.P.dataset_split_seed = args.split_seed
    layout, pairs = pair_loader.load_pairs(Path(args.dataset), args.split)
    if layout == "synthetic":
        pairs = synthetic_split(pairs, args.split, args.training_samples, args.split_seed)
    rng = random.Random(args.seed)
    pairs = list(pairs)
    rng.shuffle(pairs)
    return layout, pairs[: min(args.line_pairs, len(pairs))]


def _build_lines(models, pairs, preprocessing):
    lines = []
    seen = set()
    with tempfile.TemporaryDirectory(prefix="point6_prepared_") as temp:
        root = Path(temp)
        for pair in pairs:
            for side in (1, 2):
                source = Path(getattr(pair, f"image{side}")).resolve()
                if str(source) in seen:
                    continue
                seen.add(str(source))
                prepared, _geometry = prepare_line(
                    source, pair.preprocess_domain(side), preprocessing
                )
                # Keep an exact PIL copy; the temporary file only gives the shared
                # transform an unambiguous already-prepared synthetic-domain input.
                path = root / f"line_{len(lines):05d}.png"
                prepared.save(path)
                with Image.open(path) as opened:
                    exact = opened.convert("RGB").copy()
                lines.append(
                    _encode_line(
                        models,
                        exact,
                        source,
                        pair.pair_id or f"pair_{pair.index}",
                        side,
                    )
                )
    return lines


def _candidate_bank(lines, representation, min_ink):
    vectors = []
    metadata = []
    for line_index, line in enumerate(lines):
        features = line.representation(representation)
        for seq_index in range(int(features.shape[0])):
            if not bool(line.valid[seq_index]):
                continue
            if float(line.ink[seq_index]) < float(min_ink):
                continue
            vectors.append(features[seq_index])
            metadata.append((line_index, seq_index))
    if not vectors:
        raise RuntimeError(f"No candidate windows survived min_ink={min_ink}")
    return torch.stack(vectors, dim=0), metadata


def _window_crop(line: LineRecord, seq_index: int) -> np.ndarray:
    physical = line.seq_to_physical[int(seq_index)]
    x0 = physical * line.stride
    x1 = min(line.image.shape[1], x0 + line.window_size)
    return line.image[:, x0:x1]


def _save_contact_sheet(query, results_by_rep, lines, output, top_k):
    reps = ("local", "context", "fused")
    cols = top_k + 1
    fig, axes = plt.subplots(len(reps), cols, figsize=(2.1 * cols, 2.4 * len(reps)))
    if len(reps) == 1:
        axes = np.asarray([axes])
    query_crop = _window_crop(query[0], query[1])
    for row, rep in enumerate(reps):
        axes[row, 0].imshow(query_crop)
        axes[row, 0].set_title(f"query\n{rep}")
        axes[row, 0].axis("off")
        for rank in range(top_k):
            ax = axes[row, rank + 1]
            items = results_by_rep.get(rep, [])
            if rank < len(items):
                item = items[rank]
                line = lines[item["candidate_line_index"]]
                ax.imshow(_window_crop(line, item["candidate_seq_window"]))
                ax.set_title(
                    f"#{rank+1} cos={item['cosine']:.2f}\n"
                    f"{line.pair_id} s{line.side} w{item['candidate_physical_window']}"
                )
            ax.axis("off")
    fig.suptitle(
        f"Point 6 query: {query[0].pair_id} side{query[0].side} "
        f"physical window {query[0].seq_to_physical[query[1]]}"
    )
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", choices=("train", "valid", "test", "all"), default="test")
    ap.add_argument("--training-samples", type=int, default=6000)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--labels", default="high_match,medium_match")
    ap.add_argument("--line-pairs", type=int, default=80)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--visualize-queries", type=int, default=20)
    ap.add_argument("--min-ink", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image-preprocessing", choices=("original", "training", "tight", "cropped_1024"), default="cropped_1024")
    args = ap.parse_args()

    os.environ["EVAL_TIGHT_NO_PADDING"] = (
        "1" if args.image_preprocessing == "tight" else "0"
    )

    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = read_checkpoint(args.weights)
    models = load_point2_visual_models(checkpoint, args.device, "restoration")
    if str(models.config.get("model_backend", "")) == "resnet18_physical_window_tinyvit_positive_dtw":
        raise RuntimeError(
            "Point 6 is intended for the selected original ResNet->TinyViT checkpoint, "
            "not the physical-window ViT ablation."
        )

    layout, pairs = _select_pairs(args)
    if len(pairs) < 2:
        raise RuntimeError("Point 6 needs at least two line pairs")
    lines = _build_lines(models, pairs, args.image_preprocessing)
    if len(lines) < 2:
        raise RuntimeError("Point 6 needs at least two unique source lines")

    rng = random.Random(args.seed + 606)
    query_candidates = []
    for line_index, line in enumerate(lines):
        for seq_index in range(int(line.local.shape[0])):
            if bool(line.valid[seq_index]) and float(line.ink[seq_index]) >= args.min_ink:
                query_candidates.append((line_index, seq_index))
    rng.shuffle(query_candidates)
    query_candidates = query_candidates[: min(args.queries, len(query_candidates))]

    rows = []
    query_summary = []
    per_rep_results_for_visuals = {}
    banks = {
        rep: _candidate_bank(lines, rep, args.min_ink)
        for rep in ("local", "context", "fused")
    }

    for query_order, (query_line_index, query_seq) in enumerate(query_candidates, start=1):
        query_line = lines[query_line_index]
        rep_neighbors = {}
        neighbor_sets = {}

        for rep in ("local", "context", "fused"):
            bank, metadata = banks[rep]
            q = query_line.representation(rep)[query_seq]
            similarities = torch.mv(bank, q)
            allowed = torch.tensor(
                [
                    int(line_index) != int(query_line_index)
                    and lines[line_index].source_path.resolve()
                    != query_line.source_path.resolve()
                    for line_index, _seq in metadata
                ],
                dtype=torch.bool,
            )
            if not bool(allowed.any()):
                raise RuntimeError("No cross-line candidates available")
            masked = similarities.clone()
            masked[~allowed] = -float("inf")
            k = min(args.top_k, int(allowed.sum().item()))
            values, indices = torch.topk(masked, k=k)
            items = []
            for rank, (value, bank_index) in enumerate(
                zip(values.tolist(), indices.tolist()), start=1
            ):
                candidate_line_index, candidate_seq = metadata[int(bank_index)]
                candidate_line = lines[candidate_line_index]
                physical = candidate_line.seq_to_physical[candidate_seq]
                item = {
                    "query_order": query_order,
                    "representation": rep,
                    "rank": rank,
                    "query_pair_id": query_line.pair_id,
                    "query_side": query_line.side,
                    "query_image": str(query_line.source_path),
                    "query_seq_window": query_seq,
                    "query_physical_window": query_line.seq_to_physical[query_seq],
                    "query_ink": float(query_line.ink[query_seq]),
                    "candidate_line_index": candidate_line_index,
                    "candidate_pair_id": candidate_line.pair_id,
                    "candidate_side": candidate_line.side,
                    "candidate_image": str(candidate_line.source_path),
                    "candidate_seq_window": candidate_seq,
                    "candidate_physical_window": physical,
                    "candidate_ink": float(candidate_line.ink[candidate_seq]),
                    "cosine": float(value),
                    "same_pair_id": int(candidate_line.pair_id == query_line.pair_id),
                }
                items.append(item)
                rows.append(item)
            rep_neighbors[rep] = items
            neighbor_sets[rep] = {
                (item["candidate_image"], item["candidate_physical_window"])
                for item in items
            }

        def overlap(a, b):
            union = neighbor_sets[a] | neighbor_sets[b]
            return (
                len(neighbor_sets[a] & neighbor_sets[b]) / len(union)
                if union
                else 1.0
            )

        query_summary.append(
            {
                "query_order": query_order,
                "query_pair_id": query_line.pair_id,
                "query_side": query_line.side,
                "query_image": str(query_line.source_path),
                "query_seq_window": query_seq,
                "query_physical_window": query_line.seq_to_physical[query_seq],
                "local_top1_cosine": rep_neighbors["local"][0]["cosine"],
                "context_top1_cosine": rep_neighbors["context"][0]["cosine"],
                "fused_top1_cosine": rep_neighbors["fused"][0]["cosine"],
                "local_context_topk_jaccard": overlap("local", "context"),
                "local_fused_topk_jaccard": overlap("local", "fused"),
                "context_fused_topk_jaccard": overlap("context", "fused"),
            }
        )
        if query_order <= args.visualize_queries:
            per_rep_results_for_visuals[query_order] = rep_neighbors
            _save_contact_sheet(
                (query_line, query_seq),
                rep_neighbors,
                lines,
                output / f"query_{query_order:04d}_neighbors.png",
                args.top_k,
            )

    with (output / "point6_neighbors.csv").open("w", newline="", encoding="utf-8") as f:
        fields = [
            "query_order", "representation", "rank", "query_pair_id", "query_side",
            "query_image", "query_seq_window", "query_physical_window", "query_ink",
            "candidate_line_index", "candidate_pair_id", "candidate_side",
            "candidate_image", "candidate_seq_window", "candidate_physical_window",
            "candidate_ink", "cosine", "same_pair_id"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with (output / "point6_queries.csv").open("w", newline="", encoding="utf-8") as f:
        fields = list(query_summary[0]) if query_summary else []
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(query_summary)

    summary = {
        "point": 6,
        "diagnostic_only": True,
        "dataset_layout": layout,
        "line_pairs_loaded": len(pairs),
        "unique_lines": len(lines),
        "query_windows": len(query_summary),
        "top_k": args.top_k,
        "min_ink": args.min_ink,
        "same_query_windows_for_all_representations": True,
        "same_source_line_candidates_excluded": True,
        "overlapping_same_source_windows_excluded": True,
        "mean_local_top1_cosine": _finite_mean(
            r["local_top1_cosine"] for r in query_summary
        ),
        "mean_context_top1_cosine": _finite_mean(
            r["context_top1_cosine"] for r in query_summary
        ),
        "mean_fused_top1_cosine": _finite_mean(
            r["fused_top1_cosine"] for r in query_summary
        ),
        "mean_local_context_topk_jaccard": _finite_mean(
            r["local_context_topk_jaccard"] for r in query_summary
        ),
        "mean_local_fused_topk_jaccard": _finite_mean(
            r["local_fused_topk_jaccard"] for r in query_summary
        ),
        "mean_context_fused_topk_jaccard": _finite_mean(
            r["context_fused_topk_jaccard"] for r in query_summary
        ),
        "interpretation": (
            "Compare the contact sheets and neighbor lists for exactly the same query "
            "windows. Top-1 cosine and neighbor-set overlap describe representation "
            "behavior only; without independent semantic window labels they are not "
            "nearest-neighbor accuracy."
        ),
    }
    (output / "point6_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
