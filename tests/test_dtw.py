import pytest
import torch

from dtw import cosine_cost_matrix, hard_dtw_path, letter_cost_matrix, soft_dtw


def test_known_diagonal_and_competition():
    vectors = torch.eye(4)
    costs = cosine_cost_matrix(vectors, vectors)
    result = hard_dtw_path(costs)
    assert result.path == [(i,i) for i in range(4)] and result.total == 0
    assert soft_dtw(costs) < soft_dtw(cosine_cost_matrix(vectors, vectors.flip(0)))
    nll = letter_cost_matrix(vectors, vectors[:2], alphabet=vectors, letter_ids=[0,1])
    torch.testing.assert_close(nll, -torch.log_softmax(vectors / .1, -1)[:,:2])


@pytest.mark.parametrize('shape', [(3,2),(2,4),(1,1),(63,7)])
def test_variable_lengths_backward_and_soft_limit(shape):
    torch.manual_seed(12)
    costs = torch.rand(shape, requires_grad=True)
    soft = soft_dtw(costs)
    soft.backward()
    assert torch.isfinite(soft) and torch.isfinite(costs.grad).all()
    hard = hard_dtw_path(costs)
    assert hard.path[0] == (0,0) and hard.path[-1] == (shape[0]-1,shape[1]-1)
    assert all((c-a,d-b) in {(0,1),(1,0),(1,1)} for (a,b),(c,d) in zip(hard.path,hard.path[1:]))
    assert float(soft_dtw(costs, gamma=1e-5)) == pytest.approx(hard.normalized, abs=1e-5)


def test_transition_penalties_in_scalar():
    assert hard_dtw_path(torch.zeros(3,1), position_prior=0).total == pytest.approx(.1)
    assert hard_dtw_path(torch.zeros(1,3), position_prior=0).total == pytest.approx(.6)
    with pytest.raises(ValueError, match='nonempty'):
        soft_dtw(torch.empty(3,0))
