import torch

from alignment_pooling import (
    build_char_bank,
    collect_unique_chars,
    compute_char_pool_contrastive_loss,
    get_char_pool_weight,
    groups_from_assignment,
    hard_d3tw_path_from_similarity,
    normalize_assignment,
    path_to_assignment,
    pool_visual_by_assignment,
)
from textEmbedding import OrthogonalCharEmbedding


def test_restricted_path_assigns_complete_consecutive_groups():
    sim = torch.tensor([
        [5.0, 4.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 5.0, 4.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 5.0],
    ])
    path = hard_d3tw_path_from_similarity(sim)
    assignment = path_to_assignment(path, num_chars=3, num_visual=5)

    assert path[0] == (0, 0)
    assert path[-1] == (2, 4)
    assert groups_from_assignment(assignment) == [[0, 1], [2, 3], [4]]
    assert torch.equal(assignment.sum(0), torch.ones(5))


def test_assignment_pooling_is_per_character_mean_not_global_mean():
    visual = torch.tensor([
        [2.0, 0.0],
        [4.0, 0.0],
        [0.0, 3.0],
        [0.0, 5.0],
    ])
    assignment = torch.tensor([
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ])
    weights, valid, counts = normalize_assignment(assignment)
    unnormalized_pooled = weights @ visual
    pooled, pooled_valid, pooled_counts = pool_visual_by_assignment(
        visual, assignment
    )

    assert torch.allclose(unnormalized_pooled, torch.tensor([[3.0, 0.0], [0.0, 4.0]]))
    assert torch.allclose(pooled, torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert not torch.allclose(unnormalized_pooled[0], visual.mean(0))
    assert valid.tolist() == pooled_valid.tolist() == [True, True]
    assert counts.tolist() == pooled_counts.tolist() == [2.0, 2.0]


def test_detached_assignment_preserves_visual_gradient():
    visual = torch.randn(5, 8, requires_grad=True)
    assignment = torch.tensor([
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1],
    ], dtype=torch.float32, requires_grad=True)
    pooled, valid, counts = pool_visual_by_assignment(
        visual, assignment, detach_assignment=True
    )

    assert pooled.requires_grad
    assert valid.tolist() == [True, True]
    assert counts.tolist() == [2.0, 3.0]
    pooled.sum().backward()
    assert visual.grad is not None and visual.grad.abs().sum() > 0
    assert assignment.grad is None


def test_path_pool_and_character_loss_backpropagate_to_visual_windows():
    raw_visual = torch.tensor([
        [3.0, 0.1],
        [2.0, 0.1],
        [0.1, 2.0],
        [0.1, 3.0],
    ], requires_grad=True)
    visual = torch.nn.functional.normalize(raw_visual, dim=-1)
    bank = torch.eye(2)
    sim = bank @ visual.T
    path = hard_d3tw_path_from_similarity(sim)
    assignment = path_to_assignment(path, 2, 4, device=visual.device)
    pooled, valid, _ = pool_visual_by_assignment(visual, assignment)
    loss, _ = compute_char_pool_contrastive_loss(
        pooled, ["a", "b"], bank, {"a": 0, "b": 1}, valid_mask=valid
    )

    loss.backward()
    assert groups_from_assignment(assignment) == [[0, 1], [2, 3]]
    assert pooled.shape == (2, 2)
    assert raw_visual.grad is not None and raw_visual.grad.abs().sum() > 0


def test_minimum_window_count_masks_small_groups():
    visual = torch.randn(3, 4)
    assignment = torch.tensor([[1, 0, 0], [0, 1, 1]], dtype=torch.float32)
    _, valid, counts = pool_visual_by_assignment(
        visual, assignment, min_windows_per_char=2
    )
    assert valid.tolist() == [False, True]
    assert counts.tolist() == [1.0, 2.0]


def test_character_bank_loss_and_schedule():
    embedder = OrthogonalCharEmbedding(embedding_dim=16)
    chars = collect_unique_chars(["بت ب"], skip_spaces=True)
    embeddings, char_to_idx, idx_to_char = build_char_bank(embedder, chars, "cpu")
    pooled = embeddings[[char_to_idx["ب"], char_to_idx["ت"]]].clone()
    pooled.requires_grad_(True)
    loss, stats = compute_char_pool_contrastive_loss(
        pooled, ["ب", "ت"], embeddings, char_to_idx, tau=0.07
    )

    loss.backward()
    assert idx_to_char == chars
    assert not embeddings.requires_grad
    assert stats["char_pool_acc_top1"] == 1.0
    assert pooled.grad is not None
    assert get_char_pool_weight(4, 0.5, 5, 10) == 0.0
    assert get_char_pool_weight(5, 0.5, 5, 10) == 0.05
    assert get_char_pool_weight(14, 0.5, 5, 10) == 0.5
