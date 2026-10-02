"""GT-free evaluation regressions; checkpoints are random fixtures, never trained."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from evaluate import local_repeat_regions, match_features, region_mask, mutual_top_k_anchors
from evaluation_utils import (EvaluationSession, predict_cached_pair, evaluate_population,
                              plot_pair_alignment)
from test_evaluation_notebook import evaluation_fixture, notebook_source


def diagonal(indices, n=None, m=None, value=.95):
    n = n or max(indices)+1
    c = np.zeros((n, m or n))
    for i in indices: c[i,i] = value
    return c


def decode(c, physical=None, **kwargs):
    p = physical or (np.arange(c.shape[0])[::-1], np.arange(c.shape[1])[::-1])
    opts = dict(threshold=.5, score_mode='raw', min_windows=5,
                min_distinct_windows_a=3, min_distinct_windows_b=3, min_matched_pairs=4,
                allow_short_regions=False,use_path_anchor_check=False,enable_path_extension=False,enable_region_merging=False,
                normal_min_distinct_a=None,normal_min_distinct_b=None,normal_min_matched_pairs=None)
    opts.update(kwargs)
    return local_repeat_regions(c,*p,**opts)


def test_three_window_path_rejected_with_full_diagnostics():
    r=decode(diagonal(range(3)))
    assert not r['regions']
    c=r['rejected'][0]
    assert c['reason']=='insufficient_distinct_support'
    assert (c['support_a'],c['support_b'],c['matched_pairs'])==(3,3,3)
    assert (c['required_support_a'],c['required_support_b'],c['required_matched_pairs'])==(5,5,5)
    assert c['score']==pytest.approx(1.35) and not c['accepted']


def test_five_window_path_accepted_and_score_accounting():
    r=decode(diagonal(range(5)),use_strong_start_anchor=True)
    assert len(r['regions'])==1
    c=r['regions'][0]
    assert c['support']==[5,5] and c['matched_pairs']==5 and c['accepted']
    assert c['score']==pytest.approx(c['matching_reward']-c['gap_penalty']-c['repeat_penalty'])
    assert c['score']==pytest.approx(sum(s['delta'] for s in c['path_steps']))
    assert c['path_density']==1 and c['repeat_fraction']==0
    assert c['start_anchor_mutual_top_k']


def test_large_internal_gap_splits_and_rescores_independently():
    # Gap penalties allow the original DP traceback to bridge four windows.
    c=diagonal([0,1,2,7,8,9,10,11])
    r=decode(c,physical=(np.arange(12),np.arange(12)),gap_open=.02,gap_extend=.01,max_gap=1)
    assert [x['support'] for x in r['regions']]==[[5,5]]
    assert r['regions'][0]['pairs']==[(i,i) for i in range(7,12)]
    assert r['regions'][0]['score']==pytest.approx(5*.45)
    assert any(x['support']==[3,3] and x['reason']=='insufficient_distinct_support' for x in r['rejected'])
    assert all(x['maximum_internal_gap']<=1 for x in r['regions'])
    assert all(not set(range(3,7)) & set(x['supported_physical'][0]+x['filled_physical'][0]) for x in r['regions'])


def test_one_window_gap_allowed_filled_only_when_valid():
    r=decode(diagonal([0,1,2,4,5,6]),max_gap=1)
    assert len(r['regions'])==1
    c=r['regions'][0]
    assert c['support']==[6,6] and c['filled_physical']==[[3],[3]]
    assert c['maximum_internal_gap']==1 and c['true_gaps']==2
    assert c['path_density']==pytest.approx(6/7)
    assert c['gap_penalty']>0
    assert c['score']==pytest.approx(c['matching_reward']-c['gap_penalty']-c['repeat_penalty'])
    assert not decode(diagonal([0,1,2,4,5,6]),max_gap=0)['regions']


def test_legacy_first_start_settings_have_no_prediction_effect():
    c=diagonal(range(7));c[0,0]=.55
    plain=decode(c)
    legacy=decode(c,use_strong_start_anchor=True,start_min_reward=.2)
    assert plain['regions']==legacy['regions']
    assert plain['regions'][0]['pairs'][0]==(0,0)


def test_weak_nonanchor_extends_strong_route():
    c=diagonal(range(6),n=10)
    c[2,2]=.55
    c[2,7:10]=.99
    c[7:10,2]=.99
    r=decode(c,use_strong_start_anchor=True,start_top_k=3)
    assert not r['strong_start_anchors'][2,2]
    route=next(x for x in r['regions'] if (0,0) in x['pairs'])
    assert (2,2) in route['pairs'] and route['matched_pairs']==6


def test_split_segment_keeps_weak_prefix_when_whole_route_is_valid():
    c=diagonal(range(10),n=14);c[3,3]=.55
    p=[0,1,2,*range(7,18)]
    r=decode(c,physical=(p,p))
    region=next(x for x in r['regions'] if (4,4) in x['pairs'])
    assert region['pairs'][0]==(3,3) and region['support']==[7,7]
    assert region['maximum_internal_gap']==0


def test_padding_invalid_tokens_rectangular_and_physical_holes():
    a=torch.eye(9)
    b=torch.eye(9)[:8]
    a[3]=0; b[3]=0
    va=torch.ones(9,dtype=torch.bool);va[3]=False
    vb=torch.ones(8,dtype=torch.bool);vb[3]=False
    r=match_features(a,b,np.arange(9),np.arange(8),token_valid_a=va,token_valid_b=vb,
        alignment_mode='local_repeat_dtw',threshold=.5,score_mode='raw',min_windows=3,
        min_matched_pairs=3,gap_open=.01,gap_extend=.01)
    assert r['cosine'].shape==(8,7)
    assert all(3 not in x['supported_physical'][side]+x['filled_physical'][side]
               for x in r['regions'] for side in (0,1))
    assert all(x['maximum_internal_gap']<=1 for x in r['regions'])
    # A large invalid physical hole splits even with adjacent logical matrix cells.
    p=[0,1,2,7,8,9,10,11]
    split=decode(np.eye(8)*.95,physical=(p,p))
    assert [x['support'] for x in split['regions']]==[[5,5]]
    geo=dict(source_size=[120,20],crop=[0,0,120,20],scale_x=1)
    mask=region_mask(split['regions'],0,geo,4,10)
    assert not mask[:,30:70].any()
    with pytest.raises(ValueError,match='nonzero'):
        match_features(a,b,np.arange(9),np.arange(8))


def test_rtl_preserves_both_physical_maps_with_unequal_lengths():
    c=np.zeros((7,10))
    for i in range(5):c[i,i+3]=.95
    ltr=decode(c,physical=(np.arange(7),np.arange(10)))
    rtl=decode(c)
    assert rtl['regions'][0]['pairs']==ltr['regions'][0]['pairs']
    assert rtl['regions'][0]['score']==ltr['regions'][0]['score']
    assert rtl['regions'][0]['supported_physical']==[[2,3,4,5,6],[2,3,4,5,6]]
    assert ltr['regions'][0]['supported_physical']==[[0,1,2,3,4],[3,4,5,6,7]]


def test_optional_filters_and_secondary_score_do_not_override_support():
    c=np.zeros((20,20))
    for i in range(6):c[i,i]=.95
    for i in range(12,17):c[i,i]=.56
    r=decode(c)
    assert len(r['regions'])==2
    filtered=decode(c,min_secondary_score_ratio=.25)
    assert len(filtered['regions'])==1
    assert any(x['reason']=='below_secondary_score_ratio' for x in filtered['rejected'])
    assert not decode(diagonal(range(3)),min_secondary_score_ratio=.01)['regions']
    assert not decode(diagonal(range(5)),min_matched_pairs=8)['regions']
    assert not decode(diagonal(range(5)),min_region_score=10)['regions']
    assert not decode(diagonal(range(5)),min_mean_reward=.6)['regions']
    assert not decode(diagonal([0,1,2,4,5,6]),min_path_density=.9)['regions']
    repeated=np.zeros((7,8))
    for i in range(6):repeated[i,i]=.95
    repeated[2,3]=.95
    for i in range(3,6):repeated[i,i+1]=.95
    rr=decode(repeated,max_repeat_fraction=0)
    assert any(x['repeat_penalty']>0 and x['reason']=='excessive_repeat_fraction' for x in rr['rejected'])


def test_notebook_session_population_predictions_identical_and_gt_independent(evaluation_fixture,monkeypatch):
    root,checkpoint=evaluation_fixture
    session=EvaluationSession(checkpoint,root,device='cpu')
    # Execute the real configuration cell, including applying it to the session.
    ns=dict(session=session)
    exec(notebook_source('# EVALUATION-ONLY MATCHING SETTINGS.'),ns)
    pair=session.pairs[0]
    for candidate in session.pairs:
        for side in candidate['sides']:
            session.get_line_features(side)['features']['fused']=torch.eye(7,session.config.embedding_dim)
    lines=[session.get_line_features(s) for s in pair['sides']]
    # Controlled model-output fixture gives an actual accepted route.
    for line in lines:line['features']['fused']=torch.eye(7,session.config.embedding_dim)
    cached=predict_cached_pair(session,pair,lines,settings=ns['current_match_settings']())
    from evaluation_utils import compute_pair_metrics, load_pair_ground_truth
    ns.update(compute_pair_metrics=compute_pair_metrics,load_pair_ground_truth=load_pair_ground_truth,
              ground_truth_cache={})
    exec(notebook_source('KEY_METRICS ='),ns)
    notebook_result=ns['_rescore_cached'](dict(base=dict(pair=pair,lines=lines,
        representation='fused',cosine=cached['cosine'])),ns['current_match_settings']())
    assert notebook_result['match']['regions']==cached['match']['regions']
    for a,b in zip(notebook_result['masks'],cached['masks']):np.testing.assert_array_equal(a,b)
    evaluated=session.evaluate_pair(pair)
    assert cached['match']['regions']==evaluated['match']['regions']
    assert cached['match']['regions'][0]['support']==[7,7]
    for a,b in zip(cached['masks'],evaluated['masks']):np.testing.assert_array_equal(a,b)
    seen=[]
    original=session.evaluate_pair
    def capture(*args,**kwargs):
        result=original(*args,**kwargs); seen.append(result); return result
    monkeypatch.setattr(session,'evaluate_pair',capture)
    session.metric_cache.clear()
    population=evaluate_population(session)
    same=next(r for r in seen if r['pair']['sample_id']==pair['sample_id'])
    assert same['match']['regions']==cached['match']['regions']
    assert len(population['rows'])==3
    hostile=dict(pair,annotations=[dict(mask='/missing/gt')]*2,label='no_shared_content',target=0)
    pred=session.predict_pair(hostile)
    assert pred['match']['regions']==cached['match']['regions']
    fig=plot_pair_alignment(session,evaluated,max_rejected_paths=5)
    fig.canvas.draw();plt.close(fig)


def test_empty_valid_features_and_invalid_configuration():
    r=match_features(torch.empty(0,3),torch.eye(3),[],[0,1,2],alignment_mode='local_repeat_dtw')
    assert r['cosine'].shape==(0,3) and not r['regions']
    with pytest.raises(ValueError):decode(np.eye(5),min_windows=0)
    with pytest.raises(ValueError):decode(np.eye(5),anchor_top_k=0)



def test_deprecated_start_min_reward_does_not_reject_split_segments():
    c=diagonal(range(10));c[3,3]=.55
    p=[0,1,2,*range(7,14)]
    r=decode(c,physical=(p,p),use_strong_start_anchor=True,start_min_reward=.9)
    assert r['regions'][0]['pairs'][0]==(3,3)
    assert r['regions'][0]['support']==[7,7]


def test_standalone_pair_local_decoder_with_frozen_fixture(evaluation_fixture,tmp_path):
    from evaluate import evaluate_pair
    root,checkpoint=evaluation_fixture
    report=evaluate_pair(checkpoint,[root/'A8.png',root/'B8.png'],tmp_path/'standalone',
        alignment_mode='local_repeat_dtw',threshold=.5,min_windows=5,use_strong_start_anchor=True)
    assert report['settings']['decoder']=='local_repeat_dtw'
    assert np.load(tmp_path/'standalone/cosine.npy').shape==(7,7)
    assert (tmp_path/'standalone/cosine_heatmap.png').exists()
    assert all(min(r['support'])>=5 for r in report['regions'])
    assert report['implementation_version']=='local-path-quality-3'
