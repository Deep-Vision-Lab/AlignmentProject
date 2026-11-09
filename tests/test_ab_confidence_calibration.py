"""Confidence-only replay, faithful compact pools, and guarded selection."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.eval.calibrate_ab_confidence import (
    BASELINE, presets, replay, compact_prediction, scores, choose, development_passed,
)
from scripts.eval.mask_quantitative import binary_metrics
from evaluate import resolved_match_settings


def test_search_changes_only_acceptance_not_generation_or_visual_passages():
    baseline = resolved_match_settings(json.loads(BASELINE.read_text()))
    allowed = {'word_max_nll', 'word_min_posterior', 'word_min_posterior_short', 'word_min_cosine',
               'word_min_distinct_types', 'word_min_total_characters', 'word_visual_weight'}
    methods = presets(baseline)
    assert len(methods) > 100
    for _, settings in methods:
        assert {k for k in settings if settings[k] != baseline[k]} <= allowed


def test_compact_replay_and_column_masks_equal_full_source_prediction():
    baseline = json.loads(BASELINE.read_text())
    config = SimpleNamespace(window_width=32, window_stride=16)
    geometry = dict(source_size=[100, 12], crop=[0, 0, 100, 12], scale_x=1.)
    candidates = []
    for word, logical, columns in [('ابتغا', [0, 4], [2, 23]), ('الله', [5, 8], [52, 78])]:
        candidates.append(dict(candidate_type='word', word=word, word_nll=[.3, .4], word_pair_posterior=.8,
            mean_cosine=.85, logical_ranges=[logical, logical],
            supported_physical=[list(range(logical[0], logical[1]+1))]*2, filled_physical=[[], []],
            source_word_ranges=[columns, columns], unused_traceback=['prose'] * 100))
    prediction = dict(pair={}, match=dict(candidates=candidates), lines=[dict(geometry=geometry, physical=np.arange(9))]*2)
    full_regions, full = replay(prediction, baseline, config)
    compact = compact_prediction(prediction)
    compact_regions, column = replay(compact, baseline, config, columns=True)
    assert {c['word'] for c in full_regions} == {c['word'] for c in compact_regions} == {'ابتغا', 'الله'}
    assert all(np.array_equal(a[:1], b) for a, b in zip(full, column))
    assert all('unused_traceback' not in c for c in compact['match']['candidates'])
    truth = np.zeros((12, 100), bool); truth[:, 2:20] = True
    a = scores(full, [truth, truth]); b = scores(column, [truth[:1], truth[:1]])
    assert np.allclose(a, b)
    direct = binary_metrics(full[0], truth)
    assert np.allclose(a, [direct[k] for k in ('iou', 'dice', 'precision', 'recall')])


def test_best_positive_iou_is_selected_only_inside_budget_and_regression_guards():
    rows = [dict(name='high_fpr', negative_fpr=.11, development_passed=True, iou=.95, recall=.99, precision=.98),
            dict(name='loses_words', negative_fpr=.01, development_passed=False, iou=.96, recall=.99, precision=.98),
            dict(name='best', negative_fpr=.09, development_passed=True, iou=.8, recall=.9, precision=.85),
            dict(name='stricter', negative_fpr=.02, development_passed=True, iou=.7, recall=.8, precision=.9)]
    assert choose(rows, .1)['name'] == 'best'
    assert choose(rows, .05)['name'] == 'stricter'
    with pytest.raises(ValueError, match='No validation preset'):
        choose(rows[:2], .1)


def test_known_three_word_example_cannot_be_silently_reduced_to_two_words():
    baseline = [dict(sample_id='pair_000016__A_line_014__B_line_003', iou=.89, words=['الله']),
                dict(sample_id='pair_000028__A_line_005__B_line_002', iou=.76, words=['ابتغا', 'الله', 'من'])]
    current = [dict(row) for row in baseline]
    current[1] = dict(current[1], iou=.8, words=['ابتغا', 'الله'])
    assert not development_passed(current, baseline)
    assert development_passed(baseline, baseline)
