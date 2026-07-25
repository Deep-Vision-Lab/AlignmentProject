import numpy as np
import torch

from Evaluation._eval_utils import needleman_wunsch, normalize_word
from Evaluation.window_alignment import (
    BLANK_TOKEN,
    alignment_score_matrix,
    attach_raw_similarities,
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


def test_mutual_z_score_removes_row_and_column_cosine_bias():
    row_bias = torch.tensor([[0.8], [0.6], [0.4]], dtype=torch.float32)
    column_bias = torch.tensor([[0.3, 0.2, 0.1]], dtype=torch.float32)
    similarity = row_bias + column_bias + 0.25 * torch.eye(3)

    scores = alignment_score_matrix(similarity, mode="mutual-z")
    diagonal = scores.diag().mean()
    off_diagonal = (scores.sum() - scores.diag().sum()) / 6

    assert diagonal > 0
    assert diagonal > off_diagonal
    assert torch.isfinite(scores).all()


def test_raw_cosines_are_restored_after_score_based_nw():
    raw_similarity = torch.tensor(
        [[0.71, 0.20], [0.15, 0.83]],
        dtype=torch.float32,
    )
    match_scores = torch.tensor(
        [[2.0, -1.0], [-1.0, 2.0]],
        dtype=torch.float32,
    )
    scored = needleman_wunsch(match_scores, gap_penalty=-0.3)
    restored = attach_raw_similarities(scored, raw_similarity)

    matched_cosines = [
        step.similarity
        for step in restored.steps
        if step.index1 is not None and step.index2 is not None
    ]
    assert np.allclose(matched_cosines, [0.71, 0.83])
    assert restored.score == scored.score


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
