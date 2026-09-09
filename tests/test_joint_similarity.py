from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from Evaluation.joint_similarity import joint_components
from Evaluation.eval_yelda import evaluation_similarity, parse_args
from Evaluation._eval_utils import ImageFeatures, needleman_wunsch


def inputs():
    local = torch.eye(2)
    context_left = torch.eye(2)
    context_right = torch.flip(context_left, [0])
    return local, local, context_left, context_right


@pytest.mark.parametrize("weight", [0.0, 0.25, 0.5, 1.0])
def test_joint_score_is_weighted_cosines(weight):
    a, b, c, d = inputs()
    local, context, joint = joint_components(a * 3, b * 2, c * 5, d * 7, weight)
    torch.testing.assert_close(local, torch.eye(2))
    torch.testing.assert_close(context, torch.flip(torch.eye(2), [0]))
    torch.testing.assert_close(joint, weight * local + (1-weight) * context)


def test_evidence_weight_changes_one_nw_alignment():
    a, b, c, d = inputs()
    local_score = joint_components(a, b, c, d, 1.0)[2].numpy()
    context_score = joint_components(a, b, c, d, 0.0)[2].numpy()
    left = needleman_wunsch(local_score - .45, gap_penalty=-.30)
    right = needleman_wunsch(context_score - .45, gap_penalty=-.30)
    assert left.pairs == [(0, 0), (1, 1)]
    assert left.pairs != right.pairs


@pytest.mark.parametrize("weight", [-0.01, 1.01, float("nan"), float("inf")])
def test_invalid_weights_rejected(weight):
    with pytest.raises(ValueError, match="between 0 and 1"):
        joint_components(*inputs(), weight)
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "data", "--weights", "model", "--output-dir", "out", "--local-weight", str(weight)])


def test_window_correspondence_and_finite_values_required():
    a, b, c, d = inputs()
    with pytest.raises(ValueError, match="same windows"):
        joint_components(a[:1], b, c, d)
    a[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        joint_components(a, b, c, d)


def test_hook_returns_and_saves_combined_matrix(tmp_path):
    a, b, c, d = inputs()
    first = ImageFeatures(contextual=c, local=a, grouped=a, ink=torch.ones(2), image_size=(48, 128))
    second = ImageFeatures(contextual=d, local=b, grouped=b, ink=torch.ones(2), image_size=(48, 128))
    args = SimpleNamespace(representation="joint", local_weight=.25)
    scores = evaluation_similarity(first, second, args, tmp_path)
    local = np.load(tmp_path / "local_cosine_similarity.npy")
    context = np.load(tmp_path / "contextual_cosine_similarity.npy")
    np.testing.assert_allclose(scores.numpy(), .25 * local + .75 * context)
    np.testing.assert_allclose(np.load(tmp_path / "joint_similarity.npy"), scores.numpy())


def test_cli_defaults_to_joint():
    args = parse_args(["--dataset", "data", "--weights", "model", "--output-dir", "out"])
    assert args.representation == "joint" and args.local_weight == .5
