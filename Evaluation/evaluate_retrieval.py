#!/usr/bin/env python3
"""Unified image-text retrieval evaluation for synthetic and real datasets.

The evaluator uses the current training checkpoint format and the current
``DataLoader.build_dataloaders`` interface. Synthetic and real batches are
normalised into the same list of (image, transcript) samples, so the reported
metrics are directly comparable.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate image-to-text retrieval on synthetic or real data."
    )
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument(
        "--dataset-type", choices=("synthetic", "real"), default="synthetic"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Dataset directory. Defaults to DataSet/Synthetic_Arabic for synthetic "
            "and DataSet/ArabicDataset for real."
        ),
    )
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument(
        "--sides",
        choices=("first", "second", "both"),
        default="first",
        help="Which side of paired samples enters the retrieval pool.",
    )
    parser.add_argument("--n-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--score-mode",
        choices=("d3tw", "mean"),
        default="d3tw",
        help="d3tw evaluates monotonic sequence alignment; mean is a fast smoke test.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, cuda:0, etc.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSON output path. A sibling CSV ranking report is also written.",
    )
    parser.add_argument(
        "--allow-uninitialized-text",
        action="store_true",
        help=(
            "Allow a checkpoint with no saved text-encoder state. This is normally "
            "invalid for retrieval because the image branch was trained against a "
            "specific text embedding space."
        ),
    )
    return parser.parse_args(argv)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(requested)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested ({requested}) but is unavailable")
    return selected


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["DATASET_TYPE"] = args.dataset_type
    os.environ["BATCH_SIZE"] = str(args.batch_size)
    os.environ["DATALOADER_NUM_WORKERS"] = str(args.num_workers)
    # Evaluation does not need a large generated-negative pool, but the loader's
    # collate function always creates one. One negative keeps that overhead small.
    os.environ.setdefault("NUM_NEGATIVES", "1")


def compute_stride(config: Mapping[str, object], window_size: int) -> int:
    if "stride" in config:
        return max(1, int(config["stride"]))
    mode = str(config.get("window_overlap_mode", "custom")).lower()
    ratio = float(config.get("stride_ratio", 0.5))
    if mode == "no_overlap":
        return window_size
    if mode == "light_overlap":
        return max(1, window_size // 2)
    if mode == "dense_overlap":
        return max(1, window_size // 4)
    return max(1, int(window_size * ratio))


def state_from_checkpoint(checkpoint, keys: Iterable[str]):
    if not isinstance(checkpoint, Mapping):
        return checkpoint
    for key in keys:
        state = checkpoint.get(key)
        if isinstance(state, Mapping):
            return state
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint
    return None


def load_models(weights_path: Path, device: torch.device, allow_uninitialized_text: bool):
    import Parameters as P

    # textEmbedding imports ``device`` by value, so update Parameters before
    # importing the text-encoder modules.
    P.device = str(device)

    from embeddingModel import EmbeddingModel
    from textEmbedding import TextEmbedding

    checkpoint = torch.load(weights_path, map_location=device)
    config = dict(checkpoint.get("model_config") or {}) if isinstance(checkpoint, Mapping) else {}

    window_size = int(config.get("window_size", P.window_size))
    vector_size = int(config.get("vector_size", P.vector_size))
    stride = compute_stride(config, window_size)
    language = str(config.get("lang", P.lang))
    use_bilstm = bool(config.get("use_bilstm", P.use_bilstm))
    bilstm_layers = int(config.get("bilstm_layers", P.bilstm_layers))
    bilstm_hidden_dim = config.get("bilstm_hidden_dim", P.bilstm_hidden_dim)
    if bilstm_hidden_dim is not None:
        bilstm_hidden_dim = int(bilstm_hidden_dim)

    image_model = EmbeddingModel(
        window_size=window_size,
        stride=stride,
        vector_size=vector_size,
        device=str(device),
        use_flip=(language.lower() == "arabic"),
        use_bilstm=use_bilstm,
        bilstm_layers=bilstm_layers,
        bilstm_hidden_dim=bilstm_hidden_dim,
    ).to(device)

    image_state = state_from_checkpoint(
        checkpoint,
        ("image_model_state_dict", "model_state_dict", "state_dict"),
    )
    if image_state is None:
        raise ValueError(f"No image-model state was found in checkpoint: {weights_path}")
    image_model.load_state_dict(image_state, strict=True)

    text_encoder_type = str(
        checkpoint.get("text_encoder_type", config.get("text_encoder_type", P.text_encoder_type))
        if isinstance(checkpoint, Mapping)
        else P.text_encoder_type
    ).lower()
    model_name = str(config.get("arabic_text_model_name", P.arabic_text_model_name))

    if text_encoder_type == "arabic_span":
        from arabic_span_text_encoder import ArabicSpanTextEncoder

        text_model = ArabicSpanTextEncoder(
            model_name=model_name,
            output_dim=vector_size,
            max_span_chars=int(config.get("max_text_span_chars", P.max_text_span_chars)),
            freeze_backbone=True,
            device=str(device),
            strip_text_edges=bool(config.get("strip_span_text_edges", P.strip_span_text_edges)),
            cache_size=int(config.get("span_feature_cache_size", P.span_feature_cache_size)),
            cache_dtype=str(config.get("span_feature_cache_dtype", P.span_feature_cache_dtype)),
        )
    elif text_encoder_type == "arabic_token":
        from arabic_token_text_encoder import ArabicTokenTextEncoder

        text_model = ArabicTokenTextEncoder(
            model_name=model_name,
            output_dim=vector_size,
            max_token_chars=int(config.get("max_text_token_chars", P.max_text_token_chars)),
            freeze_backbone=True,
            device=str(device),
        )
    elif text_encoder_type == "char":
        text_model = TextEmbedding(embedding_dim=vector_size)
    else:
        raise ValueError(f"Unsupported checkpoint text_encoder_type={text_encoder_type!r}")

    text_model = text_model.to(device)
    text_state = state_from_checkpoint(
        checkpoint,
        ("text_encoder_state_dict", "text_embedder_state_dict"),
    )
    if text_state is None:
        if not allow_uninitialized_text:
            raise ValueError(
                "The checkpoint has no saved text-encoder state. Retrieval would use "
                "a different embedding space. Pass --allow-uninitialized-text only "
                "for an intentional smoke test."
            )
        print("warning: evaluating with an uninitialized text encoder", file=sys.stderr)
    else:
        missing, unexpected = text_model.load_state_dict(text_state, strict=False)
        if missing or unexpected:
            print(
                "warning: non-strict text-state load: "
                f"missing={list(missing)} unexpected={list(unexpected)}",
                file=sys.stderr,
            )

    image_model.eval()
    text_model.eval()
    metadata = {
        "window_size": window_size,
        "stride": stride,
        "vector_size": vector_size,
        "language": language,
        "use_bilstm": use_bilstm,
        "bilstm_layers": bilstm_layers,
        "text_encoder_type": text_encoder_type,
        "model_name": model_name,
    }
    return image_model, text_model, metadata


def select_loader(loaders, split: str):
    return {"train": loaders[0], "valid": loaders[1], "test": loaders[2]}[split]


def samples_from_batch(batch, sides: str):
    if isinstance(batch, Mapping):
        side_names = []
        if sides in ("first", "both"):
            side_names.append(("images1", "texts1", "A"))
        if sides in ("second", "both"):
            side_names.append(("images2", "texts2", "B"))
        for image_key, text_key, side_name in side_names:
            images = batch.get(image_key)
            texts = batch.get(text_key)
            if images is None or texts is None:
                continue
            for index, text in enumerate(texts):
                yield images[index], str(text), side_name
        return

    if sides == "second":
        raise ValueError("--sides second requested, but this dataset batch has one side")
    images, texts, _negatives = batch
    for index, text in enumerate(texts):
        yield images[index], str(text), "A"


def collect_embeddings(loader, image_model, text_model, device, n_samples: int, sides: str):
    image_embeddings: list[torch.Tensor] = []
    texts: list[str] = []
    sample_sides: list[str] = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            selected = list(samples_from_batch(batch, sides))
            if not selected:
                continue
            remaining = n_samples - len(image_embeddings)
            selected = selected[:remaining]
            image_batch = torch.stack([item[0] for item in selected], dim=0).to(device)
            encoded = image_model(image_batch)
            encoded = F.normalize(encoded.float(), p=2, dim=-1).cpu()
            for index, (_, text, side_name) in enumerate(selected):
                image_embeddings.append(encoded[index])
                texts.append(text)
                sample_sides.append(side_name)
            print(
                f"embedded batch={batch_index} collected={len(image_embeddings)}/{n_samples}",
                flush=True,
            )
            if len(image_embeddings) >= n_samples:
                break

        text_embeddings: list[torch.Tensor] = []
        for index, text in enumerate(texts):
            embedding = text_model(text)
            if isinstance(embedding, (tuple, list)):
                embedding = embedding[0]
            if embedding.ndim == 3 and embedding.shape[0] == 1:
                embedding = embedding.squeeze(0)
            if embedding.ndim != 2:
                raise ValueError(
                    f"Text encoder returned shape {tuple(embedding.shape)} for sample {index}; "
                    "expected [tokens, vector_size]."
                )
            text_embeddings.append(F.normalize(embedding.float(), p=2, dim=-1).cpu())

    if len(image_embeddings) < 2:
        raise ValueError(
            f"Evaluation needs at least two samples, but collected {len(image_embeddings)}"
        )
    return image_embeddings, text_embeddings, texts, sample_sides


def d3tw_cost(image_embedding: torch.Tensor, text_embedding: torch.Tensor) -> float:
    """Hard D3TW cost: image windows advance; text may advance or stay."""
    similarity = text_embedding @ image_embedding.transpose(0, 1)
    costs = (1.0 - similarity).numpy().astype(np.float64, copy=False)
    text_len, image_len = costs.shape
    if text_len == 0 or image_len == 0:
        return float("inf")
    if text_len > image_len:
        return 1e6 + float(text_len - image_len)

    previous = np.full(text_len, np.inf, dtype=np.float64)
    previous[0] = costs[0, 0]
    for image_index in range(1, image_len):
        current = np.full(text_len, np.inf, dtype=np.float64)
        current[0] = previous[0] + costs[0, image_index]
        max_text_index = min(text_len - 1, image_index)
        for text_index in range(1, max_text_index + 1):
            current[text_index] = costs[text_index, image_index] + min(
                previous[text_index], previous[text_index - 1]
            )
        previous = current
    return float(previous[text_len - 1] / image_len)


def mean_cost(image_embedding: torch.Tensor, text_embedding: torch.Tensor) -> float:
    image_vector = F.normalize(image_embedding.mean(dim=0), p=2, dim=0)
    text_vector = F.normalize(text_embedding.mean(dim=0), p=2, dim=0)
    return float(1.0 - torch.dot(image_vector, text_vector).item())


def build_cost_matrix(image_embeddings, text_embeddings, mode: str) -> np.ndarray:
    n_samples = len(image_embeddings)
    matrix = np.empty((n_samples, n_samples), dtype=np.float64)
    scorer = d3tw_cost if mode == "d3tw" else mean_cost
    for image_index, image_embedding in enumerate(image_embeddings):
        for text_index, text_embedding in enumerate(text_embeddings):
            matrix[image_index, text_index] = scorer(image_embedding, text_embedding)
        print(f"scored query={image_index + 1}/{n_samples}", flush=True)
    return matrix


def canonical_text(text: str) -> str:
    return " ".join(text.split())


def compute_metrics(cost_matrix: np.ndarray, texts: Sequence[str]):
    n_samples = len(texts)
    canonical = [canonical_text(text) for text in texts]
    ranks: list[int] = []
    rows: list[dict[str, object]] = []
    positive_costs: list[float] = []
    hard_negative_costs: list[float] = []

    for query_index in range(n_samples):
        order = np.argsort(cost_matrix[query_index], kind="stable")
        correct = {index for index, text in enumerate(canonical) if text == canonical[query_index]}
        rank = min(int(np.where(order == index)[0][0]) + 1 for index in correct)
        top_index = int(order[0])
        positive_cost = min(float(cost_matrix[query_index, index]) for index in correct)
        negative_indices = [index for index in range(n_samples) if index not in correct]
        hard_negative = (
            min(float(cost_matrix[query_index, index]) for index in negative_indices)
            if negative_indices
            else float("nan")
        )

        ranks.append(rank)
        positive_costs.append(positive_cost)
        if math.isfinite(hard_negative):
            hard_negative_costs.append(hard_negative)
        rows.append(
            {
                "query_index": query_index,
                "rank": rank,
                "positive_cost": positive_cost,
                "hard_negative_cost": hard_negative,
                "top1_index": top_index,
                "top1_correct": canonical[top_index] == canonical[query_index],
                "query_text": texts[query_index],
                "top1_text": texts[top_index],
            }
        )

    metrics = {
        "samples": n_samples,
        "R@1": 100.0 * sum(rank <= 1 for rank in ranks) / n_samples,
        "R@5": 100.0 * sum(rank <= 5 for rank in ranks) / n_samples,
        "R@10": 100.0 * sum(rank <= 10 for rank in ranks) / n_samples,
        "MRR": sum(1.0 / rank for rank in ranks) / n_samples,
        "mean_rank": sum(ranks) / n_samples,
        "median_rank": statistics.median(ranks),
        "mean_positive_cost": statistics.fmean(positive_costs),
        "mean_hard_negative_cost": (
            statistics.fmean(hard_negative_costs) if hard_negative_costs else None
        ),
        "positive_beats_hard_negative_pct": (
            100.0
            * sum(
                row["positive_cost"] < row["hard_negative_cost"]
                for row in rows
                if math.isfinite(float(row["hard_negative_cost"]))
            )
            / max(1, len(hard_negative_costs))
        ),
    }
    return metrics, rows


def default_output(args: argparse.Namespace) -> Path:
    return (
        ROOT
        / "Results"
        / "Evaluation"
        / f"{args.dataset_type}_{args.split}_{args.score_mode}_retrieval.json"
    )


def write_outputs(path: Path, payload: Mapping[str, object], rows) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path, csv_path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_samples < 2:
        raise ValueError("--n-samples must be at least 2")
    configure_environment(args)
    device = resolve_device(args.device)

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = Path(
            "DataSet/Synthetic_Arabic"
            if args.dataset_type == "synthetic"
            else "DataSet/ArabicDataset"
        )
    data_dir = data_dir.expanduser().resolve()
    weights_path = args.weights.expanduser().resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset path not found: {data_dir}")

    image_model, text_model, model_metadata = load_models(
        weights_path,
        device,
        args.allow_uninitialized_text,
    )

    from DataLoader import build_dataloaders

    loaders = build_dataloaders(str(data_dir))
    loader = select_loader(loaders, args.split)
    image_embeddings, text_embeddings, texts, sample_sides = collect_embeddings(
        loader,
        image_model,
        text_model,
        device,
        args.n_samples,
        args.sides,
    )
    cost_matrix = build_cost_matrix(image_embeddings, text_embeddings, args.score_mode)
    metrics, rows = compute_metrics(cost_matrix, texts)

    payload = {
        "dataset": {
            "type": args.dataset_type,
            "data_dir": str(data_dir),
            "split": args.split,
            "sides": args.sides,
            "sample_sides": sample_sides,
        },
        "checkpoint": str(weights_path),
        "device": str(device),
        "score_mode": args.score_mode,
        "model": model_metadata,
        "metrics": metrics,
    }
    output_path, csv_path = write_outputs(args.output or default_output(args), payload, rows)

    print("\nRetrieval metrics")
    print("-----------------")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print(f"json: {output_path}")
    print(f"csv:  {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
