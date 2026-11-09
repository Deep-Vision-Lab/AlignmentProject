"""Space is an optional transcript token, never image padding or an Arabic-letter skip."""
from collections import Counter

import pytest
import torch

from dataloader import create_dataloaders
from dataset import generate_typed_negative_transcripts
from dtw import soft_dtw
from losses import compute_loss, positive_dtw_loss
from parameters import Config, validate_objective
from test_dataset import make_book_c, make_book_lines
from text_embedding import (ARABIC_LETTERS, OrthogonalCharEmbedding,
                            clean_alignment_tokens, clean_letters,
                            restore_transcript_spaces)


def space_config(**overrides):
    return Config(embedding_dim=16, space_mode='optional',
                  dtw_normalization='aligned_mean', negative_target_mode='uniform_ratio',
                  negative_loss_type='hybrid_typed', negative_count=4,
                  negative_curriculum_epochs=0, **overrides)


def encoder():
    text = OrthogonalCharEmbedding(16, 4096)
    text.letter_evidence_prior = dict(vocabulary=list(' ' + ARABIC_LETTERS))
    return text


def test_space_tokenization_and_negative_boundaries():
    assert clean_alignment_tokens('  ا\tب   ج  ', True) == list('ا ب ج')
    assert clean_alignment_tokens('  ا\tب   ج  ') == list('ابج')
    assert clean_alignment_tokens('ابجد', True) == list('ابجد')
    assert restore_transcript_spaces('اب جد', 'اكصد') == 'اك صد'
    assert OrthogonalCharEmbedding.SPACE_TOKEN_IDX != OrthogonalCharEmbedding.PAD_TOKEN_IDX
    with pytest.raises(ValueError, match='preserve normalized letter length'):
        restore_transcript_spaces('اب جد', 'اك')


def test_space_only_skip_consumes_no_image_cell_and_has_finite_gradients():
    costs = torch.tensor([[0., 8., 8.], [8., 8., 0.]], requires_grad=True)
    old = soft_dtw(costs, position_prior=0, normalization='aligned_mean')
    new, occupancy, skipped = soft_dtw(
        costs, position_prior=0, normalization='aligned_mean',
        space_columns=[False, True, False], space_skip_penalty=1.,
        return_occupancy=True, return_space_skip_count=True)
    assert new < old and float(skipped) == pytest.approx(1., abs=1e-4)
    assert float(occupancy[:, 1].sum()) < 1e-4
    assert float(occupancy[:, 0].sum()) > .99 and float(occupancy[:, 2].sum()) > .99
    new.backward()
    assert torch.isfinite(costs.grad).all()
    torch.testing.assert_close(costs.grad[:, 1], torch.zeros_like(costs.grad[:, 1]), atol=1e-4, rtol=0)


def test_visible_space_can_align_and_nonspace_columns_cannot_skip():
    costs = torch.full((3, 3), 8.)
    costs.diagonal().fill_(0.)
    value, occupancy, skipped = soft_dtw(
        costs, position_prior=0, normalization='aligned_mean',
        space_columns=[False, True, False], space_skip_penalty=2.,
        return_occupancy=True, return_space_skip_count=True)
    assert float(skipped) < 1e-4
    assert float(occupancy[1, 1]) > .99
    old = soft_dtw(costs, position_prior=0, normalization='aligned_mean')
    disabled = soft_dtw(costs, position_prior=0, normalization='aligned_mean',
                        space_columns=[False, False, False], space_skip_penalty=2.)
    torch.testing.assert_close(old, disabled, atol=0, rtol=0)
    with pytest.raises(ValueError, match='one entry'):
        soft_dtw(costs, space_columns=[True], space_skip_penalty=1.)


def test_typed_negatives_keep_spaces_and_wrong_letter_columns_skip_spaces():
    cfg = space_config()
    validate_objective(cfg)
    text = 'سلام عليكم'
    values, stats, metadata = generate_typed_negative_transcripts(
        text, strong_count=2, local_count=1, order_count=1,
        seed=42, sample_id='line-1', epoch=1)
    assert stats['generated'] == 4
    values = [restore_transcript_spaces(text, value) for value in values]
    for value, entry in zip(values, metadata):
        entry['text'] = value
        assert value.count(' ') == 1
        assert clean_letters(value) != clean_letters(text)
    assert Counter(entry['negative_type'] for entry in metadata) == {
        'strong_global': 2, 'local_substitution': 1, 'order_negative': 1}
    visual = torch.randn(1, 16, 16, requires_grad=True)
    valid = torch.tensor([[True] * 12 + [False] * 4])
    output = dict(fused=visual, fused_pre_l2=visual, token_valid=valid)
    loss, stats = compute_loss(output, [text], encoder(), cfg,
                               negative_texts=[values], negative_metadata=[metadata])
    assert torch.isfinite(loss) and stats['space_tokens'] == 1
    assert stats['strong_count'] == 2 and stats['local_count'] == stats['order_count'] == 1
    loss.backward()
    assert torch.isfinite(visual.grad).all()
    torch.testing.assert_close(visual.grad[:, 12:], torch.zeros_like(visual.grad[:, 12:]))


def test_three_book_balancing_and_page_safe_split(tmp_path):
    root = make_book_c(make_book_lines(tmp_path / 'ArabicData', pages=60), pages=60)
    loaders = create_dataloaders(root, 'real', books=('A', 'B', 'C'), num_workers=0)
    group_sets = []
    for loader in loaders:
        counts = Counter(record['book_id'] for record in loader.dataset.records)
        assert counts['A'] == counts['B'] == counts['C'] > 0
        group_sets.append({record['group_id'] for record in loader.dataset.records})
    assert not (group_sets[0] & group_sets[1] or group_sets[0] & group_sets[2] or group_sets[1] & group_sets[2])


def test_old_checkpoint_defaults_and_invalid_space_objectives():
    assert Config().space_mode == 'off'
    for override in (dict(space_mode='optional', alignment_objective='ctc'),
                     dict(space_mode='optional', dtw_normalization='legacy'),
                     dict(space_mode='optional', dtw_cost_mode='cosine')):
        with pytest.raises(ValueError, match='Optional spaces require'):
            validate_objective(Config(**override))


def test_offline_training_path_saves_space_settings_and_balanced_books(tmp_path):
    from train import load_checkpoint, main
    root = make_book_c(make_book_lines(tmp_path / 'ArabicData', pages=60), pages=60)
    for path in root.glob('Book*/texts/*.txt'):
        path.write_text('سا لم' if path.parent.parent.name != 'BookC' else 'ا ب',
                        encoding='utf-8')
    output = main(['--dataset', str(root), '--dataset-type', 'real', '--books', 'A,B,C',
                   '--run-name', 'space_smoke', '--output-root', str(tmp_path),
                   '--device', 'cpu', '--cnn-type', 'simple', '--no-cnn-pretrained',
                   '--image-height', '32', '--image-width', '128',
                   '--embedding-dim', '16', '--transformer-layers', '1',
                   '--transformer-heads', '1', '--batch-size', '2', '--num-workers', '0',
                   '--epochs', '1', '--max-batches', '1', '--space-mode', 'optional',
                   '--space-skip-penalty-ratio', '1', '--negative-loss-type', 'hybrid_typed',
                   '--negative-count', '4', '--negative-target-mode', 'uniform_ratio',
                   '--negative-curriculum-epochs', '0', '--dtw-normalization', 'aligned_mean'])
    _, _, config, saved = load_checkpoint(output / 'checkpoint_latest.pt')
    assert config.space_mode == 'optional' and config.space_skip_penalty_ratio == 1.
    assert saved['selected_books'] == ['A', 'B', 'C']
    assert saved['letter_evidence_prior']['vocabulary'][0] == ' '
    assert saved['metrics']['train']['space_tokens'] > 0
    for ids in saved['split_ids'].values():
        counts = Counter(sample.split(':', 1)[0] for sample in ids)
        assert counts['BookA'] == counts['BookB'] == counts['BookC'] > 0
