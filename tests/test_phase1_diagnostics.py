"""Phase 1 tests use fixed image-only predictions; no optimization or retuning."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from test_evaluation_notebook import evaluation_fixture, arabic_folder_fixture, notebook_source
from dataset import prepare_image
from evaluation_utils import (EvaluationSession, REPRESENTATIONS, representation_diagnostics,
    indexing_sanity, recorded_negative_pairs, predict_cached_pair, sample_random_pairs,
    plot_pair_alignment, plot_representation_comparison, evaluate_population)


def diagnostic_result(matrix, paths):
    regions = [dict(pairs=path) for path in paths]
    return dict(cosine=matrix, match=dict(regions=regions),
                lines=[dict(physical=np.arange(n)[::-1]) for n in matrix.shape])


def test_raw_peaks_path_background_and_disjoint_transitions():
    matrix = np.array([[.9, .2, .1], [.1, .8, .2], [.2, .7, .1]])
    before = matrix.copy()
    result = diagnostic_result(matrix, [[(0, 0), (1, 1), (2, 1)]])
    stats = representation_diagnostics(result)
    assert stats['path_mean_cosine'] == pytest.approx(.8)
    background = np.array([.2, .1, .1, .2, .2, .1])
    assert stats['off_path_mean_cosine'] == pytest.approx(background.mean())
    assert stats['path_contrast'] == pytest.approx(.8-background.mean())
    assert stats['mean_row_peak_margin'] == pytest.approx(np.mean([.7, .6, .5]))
    assert stats['vertical_repeats'] == stats['diagonal_steps'] == 1
    assert stats['repeat_fraction'] == .5
    assert stats['physical_ranges_a'] == [[0, 2]]
    np.testing.assert_array_equal(matrix, before)
    disjoint = representation_diagnostics(diagnostic_result(matrix, [[(0, 0)], [(2, 2)]]))
    assert disjoint['transition_count'] == 0
    assert disjoint['repeat_fraction'] == 0
    empty = representation_diagnostics(diagnostic_result(matrix, []))
    assert empty['path_contrast'] is None and empty['number_of_matched_pairs'] == 0
    assert empty['off_path_mean_cosine'] == pytest.approx(matrix.mean())
    singleton = representation_diagnostics(diagnostic_result(np.array([[.9]]), [[(0, 0)]]))
    assert singleton['mean_row_peak_margin'] is None
    assert singleton['off_path_mean_cosine'] is None
    broad = representation_diagnostics(diagnostic_result(np.array([[.9, .88, .86, .84]]), []))
    unique = representation_diagnostics(diagnostic_result(np.array([[.9, .31, .17, .1]]), []))
    assert unique['mean_row_peak_margin'] > broad['mean_row_peak_margin']
    repeats = representation_diagnostics(diagnostic_result(np.full((6, 6), .9), [[(i, 0) for i in range(6)]]))
    assert repeats['repeat_fraction'] == 1
    assert any('one B window' in w for w in repeats['route_warnings'])


def test_indexing_excludes_padded_windows_and_checks_source_masks(evaluation_fixture):
    root, checkpoint = evaluation_fixture
    session = EvaluationSession(checkpoint, root, device='cpu', construct_negatives=False)
    pair = session.pairs[0]
    _, tensor, geometry = prepare_image(pair['sides'][0]['image'], (32, 128), bbox=pair['sides'][0]['bbox'])
    with torch.no_grad():
        output = session.model(tensor[None], token_valid=torch.tensor([[True, True, False, True, False, True, True]]))
    valid = output['token_valid'][0]
    line = dict(features={r: output[r][0][valid].cpu() for r in REPRESENTATIONS},
                geometry=geometry, token_valid=valid.cpu().numpy(),
                full_physical=output['physical_window_indices'][0].cpu().numpy(),
                physical=output['physical_window_indices'][0][valid].cpu().numpy(),
                logical=torch.arange(7)[valid].numpy())
    result = predict_cached_pair(session, pair, [line, line])
    rows = indexing_sanity(result, session.config)
    assert result['cosine'].shape == (5, 5)
    assert all(r['padded_windows'] == 2 and r['status'] == 'passed' for r in rows)
    corrupted = dict(line, physical=line['physical'].copy()); corrupted['physical'][0] = -1
    with pytest.raises(AssertionError):
        indexing_sanity(dict(result, lines=[corrupted, line]), session.config)
    mask = result['masks'][0].copy(); mask[0, 0] = 255
    with pytest.raises(AssertionError, match='Mask/physical'):
        indexing_sanity(dict(result, masks=[mask, result['masks'][1]]), session.config)


def test_recorded_negatives_use_existing_occurrences_and_saved_split(arabic_folder_fixture):
    root, checkpoint = arabic_folder_fixture
    for book in 'AB':
        index = root / f'Book{book}/lines.jsonl'
        rows = [json.loads(s) for s in index.read_text().splitlines()]
        for row in rows:
            for o in row['source_occurrences']:
                o['line_number'] = int(row['line_id'].split('_')[-1]) + 1
        index.write_text('\n'.join(map(json.dumps, rows)))
    (root / 'metadata').mkdir()
    rows = [dict(pair_id=f'pair-{page}', label_type='no_shared_content',
                 A=dict(line_idx=1), B=dict(line_idx=2)) for page in (0, 4)]
    (root / 'metadata/dataset_manifest_full_pairs.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    s = EvaluationSession(checkpoint, root, device='cpu', construct_negatives=False)
    negatives, report = recorded_negative_pairs(s)
    assert report['eligible'] == report['excluded'] == 1
    assert len(negatives) == 1 and negatives[0]['target'] == 0
    assert not negatives[0]['constructed_negative']
    assert all('/Book' in side['image'] and '_04_' in side['image'] for side in negatives[0]['sides'])
    s.pairs = negatives
    # No GT/text may enter image-only predictions.
    prediction = s.predict_pair(negatives[0])
    hostile = dict(negatives[0], annotations=[dict(mask='missing')] * 2, target=1)
    np.testing.assert_array_equal(prediction['cosine'], s.predict_pair(hostile)['cosine'])


def test_phase1_fixed_selection_is_exact_and_excludes_constructed():
    source = notebook_source('# PHASE 1 CONFIGURATION')
    function = source[source.index('def phase1_select'):source.index('\nif RUN_PHASE1:')]
    ns = dict(PHASE1_SEED=42, sample_random_pairs=sample_random_pairs)
    exec(function, ns)
    rows = [dict(sample_id=str(i), target=i % 2, constructed_negative=False) for i in range(10)]
    rows.append(dict(sample_id='constructed', target=0, constructed_negative=True))
    select = ns['phase1_select']
    assert select(rows, [], 3, True) == select(rows, [], 3, True)
    assert [r['sample_id'] for r in select(rows, ['9', '1'], 1, True)] == ['9', '1']
    with pytest.raises(ValueError, match='missing'):
        select(rows, ['constructed'], 1, False)


def test_phase1_reuses_frozen_matching_and_fair_heatmaps(evaluation_fixture):
    import matplotlib.pyplot as plt
    root, checkpoint = evaluation_fixture
    s = EvaluationSession(checkpoint, root, device='cpu', construct_negatives=False)
    pair = s.pairs[0]; before = dict(s.settings)
    results = {}
    for rep in REPRESENTATIONS:
        result = s.evaluate_pair(pair, rep)
        results[rep] = result
        indexing_sanity(result, s.config)
        fig = plot_pair_alignment(s, result, diagnostic=True, max_rejected_paths=0)
        assert fig.axes[0].images[0].get_clim() == (-1, 1)
        fig.canvas.draw(); plt.close(fig)
        population = evaluate_population(s, rep)
        assert population['selected_pair_ids'] == [p['sample_id'] for p in s.pairs]
    assert s.settings == before and len(s.feature_cache) == 4
    # Both views must display the exact same predictions, without another inference.
    def unexpected_prediction(*args, **kwargs):
        pytest.fail('Comparison must reuse the separately displayed predictions')
    s.predict_pair = unexpected_prediction
    fig, scores = plot_representation_comparison(s, pair, results=results,
                                                show_masks=True, max_rejected_paths=0)
    for column, rep in enumerate(REPRESENTATIONS):
        heat = fig.axes[column].images[0]
        assert heat.get_clim() == (-1, 1)
        np.testing.assert_array_equal(heat.get_array(), results[rep]['cosine'])
        for side in (0, 1):
            mask = fig.axes[(1 + 2 * side) * 3 + column].images[0].get_array()
            np.testing.assert_array_equal(mask, results[rep]['masks'][side] > 0)
    assert set(scores) == set(REPRESENTATIONS)
    fig.canvas.draw(); plt.close(fig)
    different_pair = dict(pair, sample_id='different-sample')
    with pytest.raises(ValueError, match='same pair'):
        plot_representation_comparison(s, different_pair, results=results)
    plt.close('all')


def test_phase1_conclusions_are_cautious_and_check_missing_evidence():
    source = notebook_source('def phase1_conclusions')
    ns = dict(pd=pd, REPRESENTATIONS=REPRESENTATIONS)
    exec(source.split('\nif RUN_PHASE1:')[0], ns)
    table = pd.DataFrame([dict(representation=rep.upper(), positive_path_contrast=c,
        positive_row_peak_margin=p, positive_column_peak_margin=p, positive_mask_iou=.5)
        for rep, c, p in zip(REPRESENTATIONS, [.04, .2, .26], [.02, .05, .05])]).set_index('representation')
    summaries = {r: dict(discrimination=dict(status='available', roc_auc=.5)) for r in REPRESENTATIONS}
    collapse = pd.DataFrame(dict(warning=['', '', '']), index=['LOCAL', 'CONTEXT', 'FUSED'])
    messages = ns['phase1_conclusions'](table, summaries, collapse)
    assert any('compensating' in x for x in messages)
    assert any('does not sufficiently separate' in x for x in messages)
    table['positive_path_contrast'] = None
    assert any('unavailable' in x for x in ns['phase1_conclusions'](table, summaries, collapse))


@pytest.mark.parametrize('contrasts,peaks,iou,phrase', [
    ([.3, .18, .16], [.06, .04, .04], .5, 'degrading useful local'),
    ([.02, .03, .04], [.01, .01, .01], .5, 'insufficiently discriminative'),
    ([.3, .3, .3], [.06, .06, .06], .1, 'candidate/path/region selection'),
])
def test_phase1_interpretation_progression_cases(contrasts, peaks, iou, phrase):
    source = notebook_source('def phase1_conclusions')
    ns = dict(pd=pd, REPRESENTATIONS=REPRESENTATIONS)
    exec(source.split('\nif RUN_PHASE1:')[0], ns)
    table = pd.DataFrame([dict(representation=rep.upper(), positive_path_contrast=c,
        positive_row_peak_margin=p, positive_column_peak_margin=p, positive_mask_iou=iou)
        for rep, c, p in zip(REPRESENTATIONS, contrasts, peaks)]).set_index('representation')
    summaries = {r: dict(discrimination=dict(status='available', roc_auc=.9)) for r in REPRESENTATIONS}
    collapse = pd.DataFrame(dict(warning=['', '', '']), index=['LOCAL', 'CONTEXT', 'FUSED'])
    assert any(phrase in s for s in ns['phase1_conclusions'](table, summaries, collapse))


def test_improving_localization_prevents_context_degradation_claim():
    source = notebook_source('def phase1_conclusions')
    ns = dict(pd=pd, REPRESENTATIONS=REPRESENTATIONS)
    exec(source.split('\nif RUN_PHASE1:')[0], ns)
    table = pd.DataFrame([dict(representation=rep.upper(), positive_path_contrast=c,
        positive_row_peak_margin=.05, positive_column_peak_margin=.05, positive_mask_iou=iou)
        for rep, c, iou in zip(REPRESENTATIONS, [.53, .34, .45], [.36, .58, .63])]).set_index('representation')
    summaries = {r: dict(discrimination=dict(status='available', roc_auc=.7)) for r in REPRESENTATIONS}
    collapse = pd.DataFrame(dict(warning=['', '', '']), index=['LOCAL', 'CONTEXT', 'FUSED'])
    messages = ns['phase1_conclusions'](table, summaries, collapse)
    assert any('mixed' in s for s in messages)
    assert not any('degrading useful local' in s for s in messages)
