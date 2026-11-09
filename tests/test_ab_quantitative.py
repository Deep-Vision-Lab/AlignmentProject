"""Large-cohort selection, held-out separation and mask/FPR denominators."""
import numpy as np
import pytest

from scripts.eval.ab_quantitative import select_catalog, score_masks, report_tables, model_hash
import torch


def pool(prefix, n, target, membership):
    return [dict(sample_id=f'{prefix}{i}', target=target, membership=membership) for i in range(n)]


def test_sampling_reaches_requested_counts_preserves_test_and_never_repeats():
    positives = pool('test', 27, 1, 'test') + pool('train', 468, 1, 'train') + pool('val', 134, 1, 'val')
    negatives = pool('neg', 300, 0, 'test')
    result = select_catalog(positives, negatives, 300, 300, 42)
    assert len(result) == len({p['sample_id'] for p in result}) == 600
    assert sum(p['target'] == 1 for p in result) == sum(p['target'] == 0 for p in result) == 300
    assert {p['sample_id'] for p in positives[:27]} <= {p['sample_id'] for p in result}
    assert result == select_catalog(positives[::-1], negatives[::-1], 300, 300, 42)
    assert result != select_catalog(positives, negatives, 300, 300, 43)


def test_sampling_fails_instead_of_silently_padding_or_shrinking():
    with pytest.raises(ValueError, match='Only 27 distinct positive'):
        select_catalog(pool('p', 27, 1, 'test'), pool('n', 300, 0, 'test'), 300, 300, 42)
    with pytest.raises(ValueError, match='Duplicate'):
        select_catalog(pool('p', 2, 1, 'test') * 2, pool('n', 4, 0, 'test'), 2, 2, 42)


def record(sample_id, target, membership='test', heldout=True):
    return dict(sample_id=sample_id, category='positive' if target else 'negative', bootstrap_group=sample_id,
                membership=membership, source_splits=[membership]*2, training_seen=membership=='train',
                heldout=heldout, development_related=False, presumed_negative=not target)


def prediction(a, b):
    return dict(masks=[a, b], region_count=int(a.any() or b.any()), candidate_limit_hit=False)


def test_negative_fpr_is_per_pair_even_when_only_one_side_has_a_mask():
    truth = np.zeros((3, 10), bool)
    mask = truth.copy(); mask[:, 2:4] = True
    row = score_masks(prediction(truth, mask), [truth, truth], record('neg', 0))
    assert row['negative_false_positive'] is True
    assert row['negative_masked_area_fraction'] == .1  # average of 0 and 20%.
    assert row['fp'] == 6
    assert row['pair_pixel_accuracy'] == .9


def test_mixed_and_heldout_rows_have_separate_positive_denominators():
    truth = np.zeros((3, 10), bool); truth[:, 2:6] = True
    empty = np.zeros_like(truth)
    rows = [score_masks(prediction(truth, truth), [truth, truth], record('seen', 1, 'train', False)),
            score_masks(prediction(empty, empty), [truth, truth], record('test', 1)),
            score_masks(prediction(empty, empty), [empty, empty], record('neg1', 0)),
            score_masks(prediction(empty, truth), [empty, empty], record('neg2', 0))]
    table, membership, summary = report_tables(rows)
    assert summary['catalog']['positive_pairs'] == 2
    assert summary['catalog']['positive_iou'] == .5
    assert summary['heldout']['positive_pairs'] == 1
    assert summary['heldout']['positive_iou'] == 0
    assert summary['catalog']['negative_pair_fpr'] == summary['heldout']['negative_pair_fpr'] == .5
    assert table.iloc[0]['aligned_pairs'] == 2 and table.iloc[1]['aligned_pairs'] == 1
    assert set(membership['membership']) == {'train', 'test'}


def test_gt_uses_exact_source_coordinates_and_rejects_mismatched_shapes():
    truth = np.zeros((2, 20), bool); truth[:, 4:8] = True
    mask = truth.copy(); mask[:, 12:16] = True
    row = score_masks(prediction(mask, mask), [truth, truth], record('p', 1))
    assert row['positive_iou'] == row['precision'] == .5
    assert row['recall'] == 1
    with pytest.raises(ValueError, match='source-image'):
        score_masks(prediction(mask, mask), [truth[:, :-1], truth], record('p', 1))


def test_state_verification_detects_parameter_and_buffer_changes():
    model = torch.nn.BatchNorm1d(3).eval().requires_grad_(False)
    before = model_hash(model)
    with torch.inference_mode(): model(torch.ones(2, 3))
    assert model_hash(model) == before
    model.running_var.add_(1)
    assert model_hash(model) != before
