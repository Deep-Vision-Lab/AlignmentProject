"""Inference-only window evidence and absolute eligibility regressions."""
import json
import math
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from alignment_candidates import (extend_path, merge_regions, candidate_statistics,
                                  uniqueness_matrices)
from dtw import bidirectional_window_scores
from evaluate import (MATCH_DEFAULTS, _local_repeat_path, local_repeat_regions,
                      match_features, resolved_match_settings,
                      plot_window_match_diagnostics, window_match_diagnostic_examples)
from evaluation_utils import EvaluationSession, save_pair_artifacts
from test_evaluation_notebook import evaluation_fixture, notebook_source

MODE = dict(similarity_mode='bidirectional_window', candidate_similarity_mode='bidirectional_window')


def test_hand_computed_reward():
    c = torch.tensor([[1., 0.], [0., .5]], dtype=torch.float64)
    scores = bidirectional_window_scores(c, temperature=.5, reward_offset=.17)
    # Explicit row/column denominators: exp(2)+1 and exp(1)+1.
    denominators = np.array([math.log(math.exp(2)+1), math.log(math.exp(1)+1)])
    logits = c.numpy() / .5
    row = logits - denominators[:, None]
    col = logits - denominators[None, :]
    expected = .5 * (row + col + 2*math.log(2))
    np.testing.assert_allclose(scores['log_p_B_given_A'], row, atol=2e-7)
    np.testing.assert_allclose(scores['log_p_A_given_B'], col, atol=2e-7)
    np.testing.assert_allclose(scores['bidirectional_evidence'], expected, atol=2e-7)
    np.testing.assert_allclose(scores['rewards'], expected-.17, atol=2e-7)
    assert all(v.dtype == torch.float32 for v in scores.values())


def test_swap_transposes_scores_for_unequal_lengths():
    rng = torch.Generator().manual_seed(16)
    a, b = torch.randn(4, 9, generator=rng), torch.randn(7, 9, generator=rng)
    ab = match_features(a, b, np.arange(4), np.arange(7), **MODE)
    ba = match_features(b, a, np.arange(7), np.arange(4), **MODE)
    for key in ('cosine', 'bidirectional_evidence', 'rewards', 'absolute_cosine_eligible'):
        np.testing.assert_allclose(ab[key], ba[key].T, atol=5e-7)
    np.testing.assert_allclose(ab['log_p_B_given_A'], ba['log_p_A_given_B'].T, atol=5e-7)


@pytest.mark.parametrize('n,m', [(1, 1), (3, 5), (7, 2)])
def test_uniform_is_zero_evidence(n, m):
    scores = bidirectional_window_scores(torch.full((n, m), .72), .1, .3)
    torch.testing.assert_close(scores['bidirectional_evidence'], torch.zeros(n, m), atol=2e-7, rtol=0)
    torch.testing.assert_close(scores['rewards'], torch.full((n, m), -.3), atol=2e-7, rtol=0)


def test_padding_is_removed_before_norm_and_reductions():
    a, b = torch.eye(5), torch.eye(5)[[0, 1, 3]]
    expected = match_features(a, b, np.arange(5), np.array([0, 1, 3]), **MODE)
    padded_a = torch.cat((a[:2], torch.full((2, 5), float('nan')), a[2:]))
    padded_b = torch.cat((b, torch.zeros(4, 5)))
    actual = match_features(padded_a, padded_b, np.array([0, 1, -1, -1, 2, 3, 4]),
        np.array([0, 1, 3, -1, -1, -1, -1]),
        token_valid_a=[True, True, False, False, True, True, True],
        token_valid_b=[True, True, True, False, False, False, False], **MODE)
    for key in ('cosine', 'bidirectional_evidence', 'rewards', 'absolute_cosine_eligible'):
        np.testing.assert_array_equal(actual[key], expected[key])
    assert actual['regions'] == expected['regions']


@pytest.mark.parametrize('n,m', [(0, 3), (3, 0), (0, 0)])
def test_empty_sequences_are_safe(n, m):
    match = match_features(torch.ones(n, 8), torch.ones(m, 8), np.arange(n), np.arange(m), **MODE)
    assert not match['regions'] and match['maximum_local_score'] == 0
    for key in ('cosine', 'rewards', 'bidirectional_evidence', 'absolute_cosine_eligible'):
        assert match[key].shape == (n, m)
    fig = plot_window_match_diagnostics(match)
    plt.close(fig)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), 0., 1e100])
def test_rejects_invalid_valid_embeddings(bad):
    a = torch.full((3, 4), bad, dtype=torch.float64)
    with pytest.raises(ValueError, match='finite|nonzero'):
        match_features(a, torch.ones(2, 4), np.arange(3), np.arange(2), **MODE)


@pytest.mark.parametrize('key,value', [
    ('window_match_temperature', 0), ('window_match_temperature', -.1),
    ('window_match_temperature', float('nan')), ('window_match_min_cosine', 1.1),
    ('window_match_min_cosine', -1.1), ('window_match_reward_offset', float('inf'))])
def test_parameter_validation(key, value):
    with pytest.raises(ValueError):
        resolved_match_settings(dict(MODE, **{key: value}))


def test_selection_and_compatibility():
    assert resolved_match_settings({})['similarity_mode'] == 'cosine'
    for candidate, reward in [('letter_evidence', 'cosine'), ('cosine', 'letter_evidence')]:
        result = resolved_match_settings(dict(candidate_similarity_mode=candidate, similarity_mode=reward))
        assert result['similarity_mode'] == result['candidate_similarity_mode'] == candidate
    assert resolved_match_settings(dict(similarity_mode='bidirectional_window'))['candidate_similarity_mode'] == 'bidirectional_window'
    for settings in [dict(candidate_similarity_mode='bidirectional_window'),
                     dict(similarity_mode='bidirectional_window', candidate_similarity_mode='cosine')]:
        with pytest.raises(ValueError, match='Mixed'):
            resolved_match_settings(settings)
    for decoder in ('affine', 'stretch'):
        with pytest.raises(ValueError, match='requires local_repeat_dtw'):
            resolved_match_settings(dict(MODE, decoder=decoder))
    with pytest.raises(ValueError, match='image-only'):
        resolved_match_settings(dict(MODE, verify_with_letter_evidence=True))


def test_float32_even_under_autocast():
    a = torch.eye(5)
    baseline = match_features(a, a, np.arange(5), np.arange(5), **MODE)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        actual = match_features(a, a, np.arange(5), np.arange(5), **MODE)
    np.testing.assert_array_equal(actual['cosine'], baseline['cosine'])
    np.testing.assert_array_equal(actual['rewards'], baseline['rewards'])


def test_small_nonzero_embeddings_are_l2_normalized():
    a = torch.eye(5) * 1e-15
    match = match_features(a, a, np.arange(5), np.arange(5), **MODE)
    np.testing.assert_allclose(match['cosine'], np.eye(5), atol=1e-6)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='MPS unavailable')
def test_mps_float32_scoring_without_autocast_support():
    a = torch.eye(5, device='mps')
    match = match_features(a, a, np.arange(5), np.arange(5), **MODE)
    assert match['regions']
    direct = bidirectional_window_scores(torch.eye(5, device='mps'))
    np.testing.assert_allclose(direct['rewards'].cpu(), match['rewards'], atol=1e-6)


def test_dp_gate_allows_only_existing_gaps_through_excluded_cells():
    c = np.eye(3) * .9
    rewards = np.full((3, 3), 2.)  # excluded cells still have positive evidence
    eligible = np.zeros((3, 3), bool)
    eligible[0, 0] = eligible[2, 2] = True
    raw, best, _ = _local_repeat_path(c, rewards, np.ones_like(eligible), .05, 3, .1, .05,
        match_eligible=eligible, return_candidates=True)
    assert best == pytest.approx(3.8)
    path = max(raw, key=lambda r:r['score'])['path_steps']
    assert [(s['i'], s['j']) for s in path if s['reward'] is not None] == [(0, 0), (2, 2)]
    assert {s['transition'] for s in path if s['reward'] is None} == {'gap_a', 'gap_b'}
    result = local_repeat_regions(c, np.arange(3), np.arange(3), rewards=rewards, match_eligible=eligible,
        normal_min_distinct_a=1, normal_min_distinct_b=1, normal_min_matched_pairs=1,
        use_path_anchor_check=False, enable_path_extension=True, enable_region_merging=True)
    assert result['regions']
    assert not result['mutual_anchors'][~eligible].any()
    assert all(eligible[q] for candidate in result['candidates'] for q in candidate['pairs'])


def steps(pairs, c, r):
    return [dict(i=i, j=j, transition='diagonal', cosine=float(c[i,j]), reward=float(r[i,j])) for i,j in pairs]


def test_extension_cannot_add_positive_but_excluded_cells():
    c = np.eye(5)*.9
    rewards = np.ones_like(c)
    eligible = np.eye(5, dtype=bool)
    eligible[0,0] = eligible[4,4] = False
    path = steps([(i,i) for i in range(1,4)], c, rewards)
    args = (path, c, rewards, [np.arange(5)]*2, eligible, uniqueness_matrices(c), dict(MATCH_DEFAULTS))
    _, before = extend_path(*args)
    _, after = extend_path(*args, match_eligible=eligible)
    assert (0,0) in before and (4,4) in before
    assert not after


def test_merging_cannot_promote_excluded_bridge_to_match():
    c = np.eye(7)*.9
    r = np.ones_like(c)
    eligible = np.eye(7, dtype=bool); eligible[3,3] = False
    physical = [np.arange(7)]*2
    o = dict(MATCH_DEFAULTS, normal_min_distinct_a=1, normal_min_distinct_b=1,
        normal_min_matched_pairs=1, use_path_anchor_check=False)
    u = uniqueness_matrices(c)
    regions = [candidate_statistics(steps([(i,i) for i in indices], c, r), c, r,
        physical, eligible, u, o, candidate_id=f'C{k}')
        for k, indices in enumerate((range(3), range(4,7)))]
    old = merge_regions(regions, [], c, r, physical, eligible, u, o)
    assert len(old) == 1 and (3,3) in old[0]['pairs']
    new = merge_regions(regions, [], c, r, physical, eligible, u, o, match_eligible=eligible)
    assert len(new) == 2
    assert all(eligible[q] for region in new for q in region['pairs'])


def test_repeated_and_all_below_gate_examples():
    examples = window_match_diagnostic_examples()
    repeated, negative = examples['repeated_valid'], examples['all_below_gate']
    assert repeated['regions']
    assert repeated['cosine'][0,0] == repeated['cosine'][0,1] == 1
    assert np.exp(repeated['log_p_B_given_A'][0,0]) < .51
    assert negative['bidirectional_evidence'].max() > 0  # strong relative best
    assert not negative['absolute_cosine_eligible'].any()
    assert not negative['regions'] and not negative['mutual_anchors'].any()
    assert not negative['candidates']


def test_gate_boundary_is_inclusive_and_offset_cannot_override_it():
    a = torch.eye(5)
    c = np.eye(5)*.5
    settings = dict(MODE, window_match_reward_offset=-2.)
    included = match_features(a, a, np.arange(5), np.arange(5), precomputed_cosine=c, **settings)
    assert included['regions'] and np.diag(included['absolute_cosine_eligible']).all()
    excluded = match_features(a, a, np.arange(5), np.arange(5), precomputed_cosine=c-1e-5, **settings)
    assert excluded['rewards'].max() > 0 and not excluded['regions']


def test_no_letter_assets_and_cached_parameters(evaluation_fixture, monkeypatch, tmp_path):
    import train
    import evaluate
    root, checkpoint = evaluation_fixture
    def forbidden(*args, **kwargs): raise AssertionError('loaded character vectors')
    monkeypatch.setattr(train, 'OrthogonalCharEmbedding', forbidden)
    monkeypatch.setattr('evaluation_utils.load_inference_text_encoder', forbidden)
    session = EvaluationSession(checkpoint, root, device='cpu', settings=MODE)
    pair = session.pairs[0]
    first = session.evaluate_pair(pair)
    assert session._text_encoder is None
    args = ['--mode', 'shared_regions', '--checkpoint', str(checkpoint),
            '--image-a', pair['sides'][0]['image'], '--image-b', pair['sides'][1]['image'],
            '--output', str(tmp_path/'cli'), '--similarity-mode', 'bidirectional_window',
            '--candidate-similarity-mode', 'bidirectional_window',
            '--window-match-temperature', '.1', '--window-match-min-cosine', '.5',
            '--window-match-reward-offset', '0.']
    for name, side in zip(('a', 'b'), pair['sides']):
        args += ['--bbox-'+name, *map(str, side['bbox'])]
    report = evaluate.main(args)
    assert report['settings']['similarity_mode'] == 'bidirectional_window'
    np.testing.assert_array_equal(np.load(tmp_path/'cli/alignment_scores.npy'), first['match']['rewards'])
    key = session.matching_cache_key(pair)
    def no_encoder(*args, **kwargs): raise AssertionError('reran image encoder')
    monkeypatch.setattr(session.model, 'forward', no_encoder)
    session.settings.update(window_match_temperature=.2, window_match_reward_offset=.4,
                            window_match_min_cosine=.7)
    second = session.evaluate_pair(pair)
    assert key != session.matching_cache_key(pair)
    np.testing.assert_array_equal(first['cosine'], second['cosine'])
    assert not np.array_equal(first['match']['rewards'], second['match']['rewards'])
    assert len(session.feature_cache) == 2 and session._text_encoder is None
    directory = tmp_path/'artifacts'
    save_pair_artifacts(session, second, directory)
    assert (directory/'absolute_cosine_eligible.npy').exists()
    metadata = json.loads((directory/'pair_metrics.json').read_text())
    assert metadata['settings']['window_match_temperature'] == .2
    assert metadata['settings']['similarity_mode'] == 'bidirectional_window'


def test_legacy_rewards_are_unchanged(evaluation_fixture):
    from evaluate import checkpoint_letter_evidence
    root, checkpoint = evaluation_fixture
    session = EvaluationSession(checkpoint, root, device='cpu')
    pair = session.pairs[0]
    lines = [session.get_line_features(side) for side in pair['sides']]
    a,b = [line['features']['fused'] for line in lines]
    physical = [line['physical'] for line in lines]
    cosine = match_features(a,b,*physical, score_mode='raw')
    np.testing.assert_allclose(cosine['rewards'], cosine['cosine']-.5, atol=1e-8)
    background = match_features(a,b,*physical)
    c = background['cosine']
    baseline = np.maximum(.5, np.median(c,axis=1)[:,None]+.05)
    baseline = np.maximum(baseline, np.median(c,axis=0)[None,:]+.05)
    np.testing.assert_array_equal(background['rewards'], c-baseline)
    options = resolved_match_settings(dict(similarity_mode='letter_evidence'))
    expected = checkpoint_letter_evidence(a,b,session.text_encoder,session.config,None,options)
    actual = match_features(a,b,*physical,text_encoder=session.text_encoder,config=session.config,**options)
    np.testing.assert_array_equal(actual['rewards'],expected['letter_evidence'])
    assert 'absolute_cosine_eligible' not in actual and 'absolute_cosine_eligible' not in cosine


def test_notebook_settings_roundtrip_and_examples():
    ns = {}
    exec(notebook_source('# EVALUATION-ONLY MATCHING SETTINGS.'), ns)
    ns.update(CANDIDATE_SIMILARITY_MODE='bidirectional_window', SIMILARITY_MODE='bidirectional_window',
              ALIGNMENT_MODE='local_repeat_dtw')
    import ast
    cell = notebook_source('def current_match_settings(')
    function = next(n for n in ast.parse(cell).body if isinstance(n,ast.FunctionDef) and n.name=='current_match_settings')
    exec(compile(ast.Module(body=[function],type_ignores=[]),'<notebook>','exec'), ns)
    settings = ns['current_match_settings']()
    assert settings['window_match_temperature'] == .1
    assert settings['window_match_min_cosine'] == .5
    assert settings['window_match_reward_offset'] == 0
    assert settings['similarity_mode'] == settings['candidate_similarity_mode'] == 'bidirectional_window'


def test_notebook_three_mode_comparison_reuses_checkpoint_features_and_pair_ids(evaluation_fixture):
    root, checkpoint = evaluation_fixture
    session = EvaluationSession(checkpoint, root, split='val', device='cpu')
    calls = []
    handle = session.model.register_forward_hook(lambda *args: calls.append(1))
    initial = dict(session.settings)
    namespace = dict(session=session, RUN_SIMILARITY_COMPARISON=True, SETTINGS=initial,
        REPRESENTATION='local', COMPARISON_MAX_PAIRS=0, display=lambda x:None,
        resolved_match_settings=resolved_match_settings, image_only_window_mode=lambda:False)
    try:
        exec(notebook_source('from evaluation_utils import evaluate_population, aggregate_summary, save_results, plot_candidate_crops'), namespace)
    finally:
        handle.remove()
    rows = namespace['comparison']
    assert [r['similarity_mode'] for r in rows] == ['cosine', 'letter_evidence', 'bidirectional_window']
    assert len({r['checkpoint_sha256'] for r in rows}) == 1
    assert {r['representation'] for r in rows} == {'local'}
    assert session.settings == initial
    assert len(calls) == len(session.feature_cache) == 4
    populations = namespace['comparison_populations']
    assert len({tuple(p['selected_pair_ids']) for p in populations.values()}) == 1


def test_notebook_refreshes_stale_kernel_without_rebuilding_session():
    # Isolate the simulated old module state from the rest of the test process.
    script = '''
import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
import evaluate
import evaluation_utils

book = json.load(open("notebooks/model_evaluation.ipynb"))
def cell(prefix):
    return next("".join(c["source"]) for c in book["cells"]
                if c["cell_type"] == "code" and "".join(c["source"]).startswith(prefix))

model, features, metrics = object(), {"cached": object()}, {"cached": object()}
session = SimpleNamespace(model=model, feature_cache=features, metric_cache=metrics,
                          text_encoder=object())
ns = dict(session=session, pd=pd, display=lambda value:None,
          predict_cached_pair=evaluation_utils.predict_cached_pair)
exec(cell("# EVALUATION-ONLY MATCHING SETTINGS."), ns)
ns.update(CANDIDATE_SIMILARITY_MODE="bidirectional_window",
          SIMILARITY_MODE="bidirectional_window", ALIGNMENT_MODE="local_repeat_dtw")
# Reproduce the saved traceback: an earlier imported evaluator lacks these keys.
for key in ("window_match_temperature", "window_match_min_cosine", "window_match_reward_offset"):
    evaluate.MATCH_DEFAULTS.pop(key)
evaluate.MATCH_VERSION = "old-kernel-version"
try:
    ns["resolved_match_settings"]({"window_match_temperature": .1})
except ValueError as error:
    assert "Unknown matching settings" in str(error)
else:
    raise AssertionError("Failed to reproduce stale-kernel error")

stale_predict = ns["predict_cached_pair"]
exec(cell("def current_match_settings("), ns)
assert ns["SETTINGS"]["similarity_mode"] == "bidirectional_window"
assert ns["SETTINGS"]["window_match_temperature"] == .1
assert ns["MATCH_VERSION"] != "old-kernel-version"
assert ns["predict_cached_pair"] is evaluation_utils.predict_cached_pair
assert ns["predict_cached_pair"] is not stale_predict
assert session.model is model and session.feature_cache is features and session.metric_cache is metrics
assert session._text_encoder is session.text_encoder

# Check that matching uses refreshed code and still needs no character assets.
session.split = "test"
session.config = SimpleNamespace(window_width=1, window_stride=1)
a = torch.eye(5)
lines = [dict(features={"fused":a}, physical=np.arange(5),
    geometry=dict(source_size=[5,1], crop=[0,0,5,1], scale_x=1)) for _ in range(2)]
prediction = ns["predict_cached_pair"](session, {}, lines)
assert prediction["settings"]["similarity_mode"] == "bidirectional_window"
assert "bidirectional_evidence" in prediction["match"]
assert prediction["match"]["cosine"].shape == (5,5)
'''
    subprocess.run([sys.executable, '-c', script], check=True, capture_output=True, text=True)
