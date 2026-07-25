import numpy as np
import torch

from Evaluation._eval_utils import needleman_wunsch, normalize_word
from Evaluation.window_alignment import (
    BLANK_TOKEN,
    cluster_representatives,
    cosine_kmeans,
    pca_project_2d,
    summarize_cluster_tokens,
    window_alignment_metrics,
    window_token_labels,
)


def test_nw_recovers_diagonal_window_alignment():
    similarity = torch.tensor(
        [
            [0.9, -0.2, -0.3],
            [-0.1, 0.8, -0.2],
            [-0.4, -0.1, 0.95],
        ],
        dtype=torch.float32,
    )
    result = needleman_wunsch(similarity, gap_penalty=-0.3)
    assert result.pairs == [(0, 0), (1, 1), (2, 2)]
    assert result.normalized_score > 0


def test_nw_can_insert_window_gap():
    similarity = np.asarray(
        [[0.9, -0.8, -0.8], [-0.8, -0.8, 0.9]],
        dtype=np.float32,
    )
    result = needleman_wunsch(similarity, gap_penalty=-0.2)
    assert (0, 0) in result.pairs
    assert (1, 2) in result.pairs
    assert any(step.operation.startswith("gap") for step in result.steps)


def test_arabic_word_normalization_removes_tatweel_and_diacritics():
    assert normalize_word("كِتـاب") == normalize_word("كتاب")


def test_window_labels_include_blank_and_span_tokens():
    path = [
        {
            "window_start": 0,
            "window_end": 2,
            "text": "ب",
            "is_blank": False,
            "is_space": False,
        },
        {
            "window_start": 2,
            "window_end": 3,
            "text": "<BLANK>",
            "is_blank": True,
            "is_space": False,
        },
        {
            "window_start": 3,
            "window_end": 5,
            "text": "ت",
            "is_blank": False,
            "is_space": False,
        },
    ]
    assert window_token_labels(path, 5) == ["ب", "ب", BLANK_TOKEN, "ت", "ت"]


def test_cosine_kmeans_separates_two_window_groups():
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=torch.float32,
    )
    labels, centers = cosine_kmeans(features, n_clusters=2, seed=3)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]
    representatives = cluster_representatives(features, labels, centers)
    assert set(representatives) == {0, 1}


def test_cluster_token_is_majority_window_token():
    summaries = summarize_cluster_tokens(
        labels=[0, 0, 0, 1, 1],
        tokens=["ب", "ب", "ت", "ج", "ج"],
    )
    by_cluster = {item.cluster: item for item in summaries}
    assert by_cluster[0].token == "ب"
    assert np.isclose(by_cluster[0].purity, 2 / 3)
    assert by_cluster[1].token == "ج"
    assert by_cluster[1].purity == 1.0


def test_pca_projection_returns_one_dot_per_window():
    projected = pca_project_2d(torch.eye(4))
    assert projected.shape == (4, 2)
    assert np.isfinite(projected).all()


def test_window_metrics_report_token_agreement():
    similarity = torch.eye(3)
    result = needleman_wunsch(similarity, gap_penalty=-0.3)
    metrics = window_alignment_metrics(result, ["ب", "ت", "ج"], ["ب", "ت", "د"])
    assert metrics["matched_window_pairs"] == 3
    assert np.isclose(metrics["token_agreement"], 2 / 3)
