"""Shared evaluation utilities compatible with optimized Span-DTW checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import unicodedata
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

from arabic_span_text_encoder import ArabicSpanTextEncoder
from arabic_token_text_encoder import ArabicTokenTextEncoder
from embeddingModel import EmbeddingModel
from span_alignment_loss import hard_span_dtw_path
from textEmbedding import TextEmbedding

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EvaluationModels:
    image_model: EmbeddingModel
    text_model: torch.nn.Module | None
    config: dict
    checkpoint: dict | torch.Tensor
    device: torch.device


@dataclass
class ImageFeatures:
    contextual: torch.Tensor
    local: torch.Tensor
    grouped: torch.Tensor
    ink: torch.Tensor
    image_size: tuple[int, int]

    def select(self, name: str) -> torch.Tensor:
        value = str(name).lower()
        if value == "contextual":
            return self.contextual
        if value == "local":
            return self.local
        if value == "grouped":
            return self.grouped
        raise ValueError("feature must be contextual, local, or grouped")


@dataclass(frozen=True)
class NWStep:
    index1: int | None
    index2: int | None
    operation: str
    similarity: float | None


@dataclass
class NWResult:
    steps: list[NWStep]
    score: float
    normalized_score: float
    score_matrix: np.ndarray

    @property
    def pairs(self) -> list[tuple[int, int]]:
        return [
            (int(step.index1), int(step.index2))
            for step in self.steps
            if step.index1 is not None and step.index2 is not None
        ]


@dataclass
class WordRegion:
    index: int
    text: str
    char_start: int
    char_end: int
    window_start: int
    window_end: int
    embedding: torch.Tensor | None = None
    ink: float = 0.0


@dataclass
class PairPaths:
    image1: Path
    image2: Path
    text1: Path
    text2: Path
    index: int


def _device(value: str | torch.device | None) -> torch.device:
    if value is None or str(value) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return requested


def _resolve_hf_home(model_name: str) -> None:
    explicit = os.environ.get("HF_HOME", "").strip()
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend(
        [
            ROOT / ".hf_cache",
            Path(str(ROOT) + "_clone") / ".hf_cache",
            Path.home() / ".cache" / "huggingface",
        ]
    )
    slug = "models--" + model_name.replace("/", "--")
    for candidate in candidates:
        for layout in (candidate, candidate / "hub"):
            snapshots = layout / slug / "snapshots"
            if snapshots.is_dir() and any(
                (item / "config.json").is_file() for item in snapshots.iterdir()
            ):
                os.environ["HF_HOME"] = str(candidate)
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
                os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
                return
    if explicit:
        os.environ["HF_HOME"] = explicit


def _strip_module_prefix(state: dict) -> dict:
    if state and all(str(key).startswith("module.") for key in state):
        return {str(key)[7:]: value for key, value in state.items()}
    return state


def _model_state(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("image_model_state_dict", "model_state_dict", "state_dict"):
            if key in checkpoint:
                return _strip_module_prefix(checkpoint[key])
    return _strip_module_prefix(checkpoint)


def _config(checkpoint) -> dict:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_config"), dict):
        return dict(checkpoint["model_config"])
    return {}


def _bool(config: dict, key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _compute_stride(config: dict) -> int:
    if "stride" in config:
        return max(1, int(config["stride"]))
    window = int(config.get("window_size", 32))
    mode = str(config.get("window_overlap_mode", "custom"))
    ratio = float(config.get("stride_ratio", 0.5))
    if mode == "no_overlap":
        return window
    if mode == "light_overlap":
        return max(1, window // 2)
    if mode == "dense_overlap":
        return max(1, window // 4)
    return max(1, int(window * ratio))


def load_evaluation_models(
    weights_path: str | os.PathLike,
    device: str | torch.device | None = "auto",
    load_text_model: bool = True,
) -> EvaluationModels:
    """Load the exact visual/text architecture recorded in a checkpoint."""
    dev = _device(device)
    checkpoint = torch.load(weights_path, map_location="cpu")
    config = _config(checkpoint)

    use_local_grouping = _bool(config, "use_local_window_grouping", True)
    os.environ["USE_LOCAL_WINDOW_GROUPING"] = "1" if use_local_grouping else "0"
    for key, env_name, default in (
        ("span_use_blank_transitions", "SPAN_USE_BLANK_TRANSITIONS", True),
        ("span_include_space_context", "SPAN_INCLUDE_SPACE_CONTEXT", False),
        (
            "span_allow_character_space_surfaces",
            "SPAN_ALLOW_CHARACTER_SPACE_SURFACES",
            False,
        ),
    ):
        os.environ[env_name] = "1" if _bool(config, key, default) else "0"
    if "span_blank_penalty" in config:
        os.environ["SPAN_BLANK_PENALTY"] = str(config["span_blank_penalty"])
    if "max_windows_per_span" in config:
        os.environ["MAX_WINDOWS_PER_SPAN"] = str(config["max_windows_per_span"])

    image_model = EmbeddingModel(
        window_size=int(config.get("window_size", 32)),
        stride=_compute_stride(config),
        vector_size=int(config.get("vector_size", 128)),
        device=dev,
        use_flip=str(config.get("lang", "Arabic")).lower() == "arabic",
        use_bilstm=_bool(config, "use_bilstm", True),
        bilstm_layers=int(config.get("bilstm_layers", 2)),
        bilstm_hidden_dim=int(config.get("bilstm_hidden_dim", 128)),
        use_local_grouping=use_local_grouping,
        local_group_size=int(config.get("local_group_size", 3)),
    ).to(dev)
    incompatible = image_model.load_state_dict(_model_state(checkpoint), strict=False)
    serious_missing = [
        key
        for key in incompatible.missing_keys
        if not key.endswith("_use_flip_state")
        and not key.endswith("_use_local_grouping_state")
    ]
    if serious_missing:
        raise RuntimeError(
            "Checkpoint is incompatible with the reconstructed image model; "
            f"missing keys: {serious_missing[:10]}"
        )
    image_model.eval()

    text_model = None
    text_type = str(
        config.get(
            "text_encoder_type",
            checkpoint.get("text_encoder_type", "arabic_span")
            if isinstance(checkpoint, dict)
            else "arabic_span",
        )
    )
    if load_text_model:
        vector_size = int(config.get("vector_size", 128))
        if text_type == "arabic_span":
            model_name = str(
                config.get("arabic_text_model_name", "aubmindlab/bert-base-arabertv02")
            )
            _resolve_hf_home(model_name)
            text_model = ArabicSpanTextEncoder(
                model_name=model_name,
                output_dim=vector_size,
                max_span_chars=int(config.get("max_text_span_chars", 2)),
                freeze_backbone=True,
                device=dev,
                strip_text_edges=_bool(config, "strip_span_text_edges", True),
                cache_size=int(config.get("span_feature_cache_size", 8192)),
                cache_dtype=str(config.get("span_feature_cache_dtype", "float16")),
            )
        elif text_type == "arabic_token":
            model_name = str(
                config.get("arabic_text_model_name", "aubmindlab/bert-base-arabertv02")
            )
            _resolve_hf_home(model_name)
            text_model = ArabicTokenTextEncoder(
                model_name=model_name,
                output_dim=vector_size,
                max_token_chars=int(config.get("max_text_token_chars", 2)),
                freeze_backbone=True,
                device=dev,
            )
        elif text_type == "char":
            text_model = TextEmbedding(embedding_dim=vector_size)
        else:
            raise ValueError(f"Unsupported text_encoder_type={text_type!r}")

        if isinstance(checkpoint, dict):
            state = checkpoint.get("text_encoder_state_dict")
            if state is None:
                state = checkpoint.get("text_embedder_state_dict")
            if state:
                text_model.load_state_dict(_strip_module_prefix(state), strict=False)
        text_model = text_model.to(dev).eval()

    return EvaluationModels(image_model, text_model, config, checkpoint, dev)


class ResizeAndBinarize:
    def __init__(self, size=(128, 1024), enabled=True, fixed_threshold=None):
        self.height, self.width = map(int, size)
        self.enabled = bool(enabled)
        self.fixed_threshold = fixed_threshold

    @staticmethod
    def _otsu(gray: np.ndarray) -> int:
        hist = np.bincount(gray.reshape(-1), minlength=256).astype(np.float64)
        total = float(gray.size)
        levels = np.arange(256, dtype=np.float64)
        total_sum = float(np.dot(levels, hist))
        left_weight = left_sum = 0.0
        best_value, best_threshold = -1.0, 127
        for threshold in range(256):
            left_weight += hist[threshold]
            if left_weight <= 0:
                continue
            right_weight = total - left_weight
            if right_weight <= 0:
                break
            left_sum += threshold * hist[threshold]
            left_mean = left_sum / left_weight
            right_mean = (total_sum - left_sum) / right_weight
            value = left_weight * right_weight * (left_mean - right_mean) ** 2
            if value > best_value:
                best_value, best_threshold = value, threshold
        return int(best_threshold)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("L").resize((self.width, self.height), Image.BILINEAR)
        if not self.enabled:
            return image.convert("RGB")
        image = ImageOps.autocontrast(image)
        gray = np.asarray(image, dtype=np.uint8)
        threshold = (
            int(self.fixed_threshold)
            if self.fixed_threshold is not None
            else self._otsu(gray)
        )
        binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
        border = np.concatenate([binary[0], binary[-1], binary[:, 0], binary[:, -1]])
        if border.mean() < 127.5:
            binary = 255 - binary
        return Image.fromarray(binary, mode="L").convert("RGB")


def build_transform(dataset_type: str = "synthetic"):
    first = (
        ResizeAndBinarize((128, 1024), enabled=True)
        if str(dataset_type).lower() == "real"
        else transforms.Resize((128, 1024))
    )
    return transforms.Compose(
        [
            first,
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_image_features(
    models: EvaluationModels,
    image_path: str | os.PathLike,
    dataset_type: str = "synthetic",
) -> ImageFeatures:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
        original_size = image.size
        tensor = build_transform(dataset_type)(image).unsqueeze(0).to(models.device)
    with torch.no_grad():
        contextual, local, grouped, ink = models.image_model(
            tensor,
            return_local=True,
            return_grouped=True,
            return_ink=True,
        )
    return ImageFeatures(
        contextual=F.normalize(contextual[0].float(), p=2, dim=-1),
        local=F.normalize(local[0].float(), p=2, dim=-1),
        grouped=F.normalize(grouped[0].float(), p=2, dim=-1),
        ink=ink[0].float(),
        image_size=original_size,
    )


def compute_similarity(features1: torch.Tensor, features2: torch.Tensor) -> torch.Tensor:
    return F.normalize(features1.float(), p=2, dim=-1) @ F.normalize(
        features2.float(), p=2, dim=-1
    ).T


def needleman_wunsch(
    similarity: torch.Tensor | np.ndarray,
    gap_penalty: float = -0.25,
    similarity_offset: float = 0.0,
) -> NWResult:
    """Global NW alignment using continuous cosine similarity as match score."""
    matrix = (
        similarity.detach().cpu().numpy().astype(np.float32)
        if torch.is_tensor(similarity)
        else np.asarray(similarity, dtype=np.float32)
    )
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2-D similarity matrix, got {matrix.shape}")
    n, m = matrix.shape
    score = np.full((n + 1, m + 1), -np.inf, dtype=np.float32)
    trace = np.zeros((n + 1, m + 1), dtype=np.uint8)
    score[0, 0] = 0.0
    for i in range(1, n + 1):
        score[i, 0] = score[i - 1, 0] + gap_penalty
        trace[i, 0] = 2
    for j in range(1, m + 1):
        score[0, j] = score[0, j - 1] + gap_penalty
        trace[0, j] = 3

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = score[i - 1, j - 1] + float(matrix[i - 1, j - 1]) - similarity_offset
            up = score[i - 1, j] + gap_penalty
            left = score[i, j - 1] + gap_penalty
            values = (diag, up, left)
            best = int(np.argmax(values))
            score[i, j] = values[best]
            trace[i, j] = best + 1

    steps: list[NWStep] = []
    i, j = n, m
    while i > 0 or j > 0:
        code = int(trace[i, j])
        if i > 0 and j > 0 and code == 1:
            steps.append(NWStep(i - 1, j - 1, "match", float(matrix[i - 1, j - 1])))
            i -= 1
            j -= 1
        elif i > 0 and (j == 0 or code == 2):
            steps.append(NWStep(i - 1, None, "gap_in_line2", None))
            i -= 1
        else:
            steps.append(NWStep(None, j - 1, "gap_in_line1", None))
            j -= 1
    steps.reverse()
    final = float(score[n, m])
    return NWResult(
        steps=steps,
        score=final,
        normalized_score=final / max(1, max(n, m)),
        score_matrix=score,
    )


def prepare_transcript(text: str, boundary_spaces: bool = True) -> str:
    value = " ".join(str(text).replace("\n", " ").split())
    return f" {value} " if boundary_spaces else value


def read_text(path: str | os.PathLike, boundary_spaces: bool = True) -> str:
    return prepare_transcript(Path(path).read_text(encoding="utf-8"), boundary_spaces)


def _word_spans(text: str):
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def align_text_to_windows(
    models: EvaluationModels,
    text: str,
    image_features: ImageFeatures,
    include_blank_steps: bool = True,
):
    if models.text_model is None:
        raise RuntimeError("This operation requires load_text_model=True")
    if str(models.config.get("text_encoder_type", "arabic_span")) != "arabic_span":
        raise RuntimeError("Word-region extraction requires an arabic_span checkpoint")
    prepared = prepare_transcript(text, boundary_spaces=True)
    with torch.no_grad():
        encoding = models.text_model(prepared)
        path = hard_span_dtw_path(
            encoding,
            image_features.contextual,
            temperature=float(models.config.get("contrastive_temperature", 0.07)),
            max_windows=int(models.config.get("max_windows_per_span", 3)),
            window_count_penalty=float(models.config.get("span_window_count_penalty", 0.05)),
            include_blank_steps=include_blank_steps,
        )
    return prepared, encoding, path


def extract_word_regions(
    models: EvaluationModels,
    text: str,
    image_features: ImageFeatures,
    feature: str = "local",
) -> tuple[list[WordRegion], list[dict]]:
    prepared, _encoding, path = align_text_to_windows(models, text, image_features, True)
    visual = image_features.select(feature)
    regions: list[WordRegion] = []
    for index, (word, start, end) in enumerate(_word_spans(prepared)):
        overlapping = [
            step
            for step in path
            if not step.get("is_blank", False)
            and int(step["text_end"]) > start
            and int(step["text_start"]) < end
        ]
        if not overlapping:
            continue
        w0 = min(int(step["window_start"]) for step in overlapping)
        w1 = max(int(step["window_end"]) for step in overlapping)
        w0 = max(0, min(w0, int(visual.shape[0]) - 1))
        w1 = max(w0 + 1, min(w1, int(visual.shape[0])))
        ink = image_features.ink[w0:w1].clamp_min(0.0)
        if ink.numel() and float(ink.sum()) > 1e-8:
            weights = ink / ink.sum()
            pooled = (visual[w0:w1] * weights.unsqueeze(-1)).sum(dim=0)
            mean_ink = float(ink.mean().item())
        else:
            pooled = visual[w0:w1].mean(dim=0)
            mean_ink = 0.0
        regions.append(
            WordRegion(
                index=index,
                text=word,
                char_start=start,
                char_end=end,
                window_start=w0,
                window_end=w1,
                embedding=F.normalize(pooled.float(), p=2, dim=-1),
                ink=mean_ink,
            )
        )
    return regions, path


def word_similarity_matrix(
    regions1: Sequence[WordRegion], regions2: Sequence[WordRegion]
) -> torch.Tensor:
    if not regions1 or not regions2:
        return torch.empty((len(regions1), len(regions2)), dtype=torch.float32)
    left = torch.stack([region.embedding for region in regions1])
    right = torch.stack([region.embedding for region in regions2])
    return compute_similarity(left, right)


_ARABIC_DIACRITICS = re.compile("[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")


def normalize_word(text: str) -> str:
    value = unicodedata.normalize("NFKC", str(text)).replace("ـ", "")
    value = _ARABIC_DIACRITICS.sub("", value)
    return value.strip()


def transcript_reference_alignment(
    regions1: Sequence[WordRegion], regions2: Sequence[WordRegion]
) -> NWResult:
    matrix = np.empty((len(regions1), len(regions2)), dtype=np.float32)
    for i, left in enumerate(regions1):
        for j, right in enumerate(regions2):
            matrix[i, j] = (
                1.0 if normalize_word(left.text) == normalize_word(right.text) else -1.5
            )
    return needleman_wunsch(matrix, gap_penalty=-0.5, similarity_offset=0.0)


def evaluate_word_alignment(
    predicted: NWResult,
    reference: NWResult,
    regions1: Sequence[WordRegion],
    regions2: Sequence[WordRegion],
) -> dict:
    predicted_pairs = set(predicted.pairs)
    reference_pairs = set(reference.pairs)
    correct = predicted_pairs & reference_pairs
    precision = len(correct) / max(1, len(predicted_pairs))
    recall = len(correct) / max(1, len(reference_pairs))
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    exact = [
        normalize_word(regions1[i].text) == normalize_word(regions2[j].text)
        for i, j in predicted.pairs
    ]
    similarities = [step.similarity for step in predicted.steps if step.similarity is not None]
    return {
        "nw_score": predicted.score,
        "nw_normalized_score": predicted.normalized_score,
        "predicted_pairs": len(predicted_pairs),
        "reference_pairs": len(reference_pairs),
        "pair_precision": precision,
        "pair_recall": recall,
        "pair_f1": f1,
        "exact_word_accuracy": float(np.mean(exact)) if exact else 0.0,
        "mean_matched_cosine": float(np.mean(similarities)) if similarities else 0.0,
        "line1_word_coverage": len({i for i, _ in predicted_pairs}) / max(1, len(regions1)),
        "line2_word_coverage": len({j for _, j in predicted_pairs}) / max(1, len(regions2)),
    }


def patch_range_to_pixels(
    window_start: int,
    window_end: int,
    n_windows: int,
    image_width: int,
    flipped: bool,
) -> tuple[float, float]:
    if n_windows <= 0:
        return 0.0, float(image_width)
    start = max(0, min(int(window_start), n_windows - 1))
    end = max(start + 1, min(int(window_end), n_windows))
    if flipped:
        x0 = (n_windows - end) / n_windows * image_width
        x1 = (n_windows - start) / n_windows * image_width
    else:
        x0 = start / n_windows * image_width
        x1 = end / n_windows * image_width
    return float(x0), float(x1)


def synthetic_pair_paths(data_dir: str | os.PathLike, index: int) -> PairPaths:
    root = Path(data_dir)
    return PairPaths(
        image1=root / "images" / f"img1_{index}.png",
        image2=root / "images" / f"img2_{index}.png",
        text1=root / "texts" / f"text1_{index}.txt",
        text2=root / "texts" / f"text2_{index}.txt",
        index=int(index),
    )


def iter_synthetic_pairs(
    data_dir: str | os.PathLike,
    start_index: int = 1,
    n_samples: int | None = None,
) -> Iterable[PairPaths]:
    root = Path(data_dir)
    available = sorted(
        int(match.group(1))
        for path in (root / "images").glob("img1_*.png")
        if (match := re.fullmatch(r"img1_(\d+)\.png", path.name))
    )
    selected = [value for value in available if value >= int(start_index)]
    if n_samples is not None and int(n_samples) > 0:
        selected = selected[: int(n_samples)]
    for index in selected:
        pair = synthetic_pair_paths(root, index)
        if all(path.is_file() for path in (pair.image1, pair.image2, pair.text1, pair.text2)):
            yield pair


def validate_pair_paths(pair: PairPaths) -> None:
    missing = [
        str(path)
        for path in (pair.image1, pair.image2, pair.text1, pair.text2)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError("Missing pair inputs: " + ", ".join(missing))


def json_ready(value):
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value
