"""Space-aware checkpoint evaluation: invisible gaps and multi-window gaps."""
from dataclasses import replace

import pytest
import torch

from dtw import hard_dtw_path, soft_dtw
from losses import alignment_tokens, effective_alphabet, positive_dtw_loss
from parameters import Config
from text_embedding import OrthogonalCharEmbedding, restore_transcript_spaces


def test_word_boundaries_and_legacy_tokenization():
    config = Config(space_mode='optional')
    assert alignment_tokens('  بَـا\t\n  ت  ', config) == list('با ت')
    assert alignment_tokens(' 12 با 99 ت ! ', config) == list('با ت')
    assert alignment_tokens('با ت', Config()) == list('بات')
    assert restore_transcript_spaces('با ت', 'جمح') == 'جم ح'


@pytest.mark.parametrize('gap_windows', [0, 1, 4])
def test_space_can_use_zero_one_or_many_windows(gap_windows):
    # Columns are Arabic letter, space, Arabic letter.
    costs = torch.full((gap_windows + 2, 3), 20.)
    costs[0, 0] = costs[-1, -1] = 0.
    costs[1:-1, 1] = 0.
    options = dict(space_columns=[False, True, False], space_skip_penalty=2., position_prior=0)
    hard = hard_dtw_path(costs, **options)
    assert hard.path == [(0, 0)] + [(i, 1) for i in range(1, gap_windows+1)] + [(gap_windows+1, 2)]
    assert hard.skipped_columns == ([] if gap_windows else [1])
    assert float(soft_dtw(costs, gamma=1e-5, **options)) == pytest.approx(hard.normalized, abs=1e-5)
    mean, occupancy = soft_dtw(costs.requires_grad_(), normalization='aligned_mean',
                               return_occupancy=True, **options)
    assert torch.isfinite(mean) and torch.isfinite(occupancy).all()
    if not gap_windows:
        assert float(occupancy[:, 1].sum()) < 1e-6
    mean.backward()
    assert torch.isfinite(costs.grad).all()


def test_letters_remain_mandatory_and_absent_mask_keeps_old_dtw():
    costs = torch.tensor([[0., 20., 20., 20., 20.], [20., 20., 20., 20., 0.]])
    result = hard_dtw_path(costs, space_columns=[False, True, False, True, False],
                           space_skip_penalty=1., position_prior=0)
    assert {j for _, j in result.path} >= {0, 2, 4}
    plain = hard_dtw_path(costs)
    assert hard_dtw_path(costs, space_columns=[False]*5, space_skip_penalty=1.) == plain
    torch.testing.assert_close(soft_dtw(costs, space_columns=[False]*5, space_skip_penalty=1.), soft_dtw(costs))


@pytest.mark.parametrize('shape', [(2, 5), (5, 3), (4, 7), (8, 6)])
@pytest.mark.parametrize('suppress_horizontal', [False, True])
@pytest.mark.parametrize('position_prior', [0., .15])
def test_hard_backtrace_matches_pulled_training_recurrence(shape, suppress_horizontal, position_prior):
    generator = torch.Generator().manual_seed(20261108)
    costs = torch.rand(shape, generator=generator) * 4
    spaces = [j % 2 == 1 and j < shape[1]-1 for j in range(shape[1])]
    options = dict(space_columns=spaces, space_skip_penalty=.8,
                   position_prior=position_prior,
                   disable_horizontal_when_feasible=suppress_horizontal)
    hard = hard_dtw_path(costs, **options)
    soft = soft_dtw(costs, gamma=1e-5, **options)
    assert float(soft) == pytest.approx(hard.normalized, abs=1e-5)
    assert {j for _, j in hard.path} >= {j for j, optional in enumerate(spaces) if not optional}
    assert all(spaces[j] for j in hard.skipped_columns)


def test_checkpoint_alphabet_and_space_occupancy_in_positive_loss():
    config = Config(space_mode='optional', dtw_normalization='aligned_mean',
                    negative_target_mode='uniform_ratio', position_prior=0)
    encoder = OrthogonalCharEmbedding(128, 4096)
    encoder.letter_evidence_prior = dict(vocabulary=list(' با'), prior=[.2, .4, .4])
    assert effective_alphabet(config, encoder) == list(' با')
    visual = encoder.encode('با')[None].clone().requires_grad_()
    loss, stats, _, occupancy = positive_dtw_loss(visual, ['ب ا'], encoder, config,
                                                return_alignments=True)
    assert stats['evaluated'] == 1 and occupancy[0].shape == (2, 3)
    assert float(occupancy[0][:, 1].sum()) < 1e-5
    loss.backward()
    assert torch.isfinite(visual.grad).all()
    legacy = replace(config, space_mode='off')
    _, _, _, old = positive_dtw_loss(visual, ['ب ا'], encoder, legacy, return_alignments=True)
    assert old[0].shape == (2, 2)


def test_optional_space_arguments_reject_invalid_grids():
    with pytest.raises(ValueError, match='one entry'):
        hard_dtw_path(torch.zeros(2, 3), space_columns=[False])
    with pytest.raises(ValueError, match='positive'):
        soft_dtw(torch.zeros(2, 1), space_columns=[True], space_skip_penalty=0.)
    with pytest.raises(ValueError, match='skip_penalty'):
        hard_dtw_path(torch.zeros(2, 2), space_columns=[False, True], space_skip_penalty=-1)


def test_typed_negative_letter_positions_ignore_inserted_space_columns():
    from dataset import generate_typed_negative_transcripts
    from losses import compute_loss
    config = Config(space_mode='optional', dtw_normalization='aligned_mean',
                    negative_target_mode='uniform_ratio', negative_loss_type='hybrid_typed',
                    negative_count=4, negative_curriculum_epochs=0, embedding_dim=16)
    encoder = OrthogonalCharEmbedding(16, 4096)
    from text_embedding import ARABIC_LETTERS
    encoder.letter_evidence_prior = dict(vocabulary=list(' ' + ARABIC_LETTERS))
    text = 'ابت ثجحخ'
    negatives, _, metadata = generate_typed_negative_transcripts(text)
    negatives = [restore_transcript_spaces(text, value) for value in negatives]
    for value, entry in zip(negatives, metadata):
        entry['text'] = value
    visual = encoder.encode('ابتثجحخ')[None].clone().requires_grad_()
    output = dict(fused=visual, token_valid=torch.ones(1, 7, dtype=torch.bool))
    loss, stats = compute_loss(output, [text], encoder, config,
                               negative_texts=[negatives], negative_metadata=[metadata])
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(visual.grad).all()
