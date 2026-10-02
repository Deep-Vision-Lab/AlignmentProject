"""Whole-route evaluation regressions; no training and no production weight writes."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from alignment_candidates import (PATH_DEFAULTS,candidate_statistics,decode_candidates,
    uniqueness_matrices,validate_candidate,select_candidates,merge_regions,extend_path)
from evaluate import local_repeat_regions,match_features,region_mask,MATCH_DEFAULTS,MATCH_VERSION
from evaluation_utils import (EvaluationSession,predict_cached_pair,sample_random_pairs,
    evaluation_category,aggregate_summary,evaluate_population,compute_pair_metrics,load_pair_ground_truth)
from test_evaluation_notebook import evaluation_fixture,notebook_source


def diagonal(n,cos=.95):return np.eye(n)*cos


def decode(c,**kwargs):
    options=dict(threshold=.5,score_mode='raw',normal_min_distinct_a=5,
        normal_min_distinct_b=5,normal_min_matched_pairs=5,enable_region_merging=False)
    options.update(kwargs)
    return local_repeat_regions(c,np.arange(c.shape[0]),np.arange(c.shape[1]),**options)


def steps(c,pairs):
    return [dict(i=i,j=j,transition='diagonal' if k==0 or (i!=pairs[k-1][0] and j!=pairs[k-1][1])
                 else 'horizontal_repeat' if i==pairs[k-1][0] else 'vertical_repeat',
                 cosine=float(c[i,j]),reward=float(c[i,j]-.5)) for k,(i,j) in enumerate(pairs)]


def options(**kwargs):
    return dict(MATCH_DEFAULTS,threshold=.5,score_mode='raw',normal_min_distinct_a=5,
                normal_min_distinct_b=5,normal_min_matched_pairs=5,**kwargs)


def test_long_route_with_weak_nonanchor_first_cell():
    c=diagonal(12);c[0,0]=.6;c[0,9:12]=.98;c[9:12,0]=.98
    result=decode(c)
    r=next(r for r in result['regions'] if (1,1) in r['pairs'])
    assert r['pairs'][0]==(0,0) and not r['start_anchor_mutual_top_k']
    assert r['mutual_anchor_count']>=6 and r['mutual_anchor_fraction']>=.15
    assert all(r['reason']!='no_strong_start_anchor' for r in result['candidates'])


def test_weak_prefix_is_trimmed_when_full_route_confidence_fails():
    c=diagonal(13)
    for i in (0,1):c[i,i]=.51;c[i,10:13]=.99;c[10:13,i]=.99
    # Six coherent cells, not eleven: leave only weak prefix + six strong cells.
    for i in range(8,13):c[i,i]=0
    result=decode(c,min_mean_reward=.40)
    r=next(r for r in result['regions'] if (2,2) in r['pairs'])
    assert r['pairs'][0]==(2,2) and r['support']==[6,6]
    assert r['trimmed_prefix'] and r['variant']=='trimmed_prefix'
    assert set(r['trimmed_pairs'])=={(0,0),(1,1)}


def test_compact_four_by_four_six_match_route_accepted_as_short():
    c=np.zeros((4,4));pairs=[(0,0),(0,1),(1,1),(2,2),(2,3),(3,3)]
    for q in pairs:c[q]=.95
    r=decode(c)['regions'][0]
    assert r['candidate_type']=='short' and r['support']==[4,4]
    assert r['matched_pairs']==6 and r['maximum_internal_gap']==0
    assert r['mutual_anchor_count']>=2 and r['path_density']==1


def test_random_weak_three_by_three_rejected():
    c=np.zeros((3,3))
    for q in [(0,0),(0,1),(1,1),(2,2)]:c[q]=.60
    result=decode(c)
    assert not result['regions']
    assert any(r['reason']=='below_short_min_mean_cosine' for r in result['rejected'])


def test_central_matcher_default_uses_normal_five_and_stronger_short_class():
    # No explicit normal minima: public dispatch must still reject a weak short island.
    a=torch.eye(4);b=torch.eye(4)*.6
    c=np.zeros((4,4));c[0,0]=c[0,1]=c[1,1]=c[2,2]=.60
    r=match_features(a,b,np.arange(4),np.arange(4),
                     threshold=.5,score_mode='raw',precomputed_cosine=c)
    assert not r['regions']
    assert any(c['reason']=='below_short_min_mean_cosine' for c in r['rejected'])

def test_high_cosine_ambiguity_excludes_self_and_is_reported():
    c=np.full((6,7),.82);u=uniqueness_matrices(c)
    assert np.all(u['row_margin']==0) and np.all(u['column_margin']==0)
    p=[(0,0),(0,1),(1,1),(2,2)];o=options()
    anchors=np.ones(c.shape,bool)
    r=candidate_statistics(steps(c,p),c,c-.5,[np.arange(6),np.arange(7)],anchors,u,o)
    assert r['ambiguous'] and r['median_bidirectional_margin']==0
    assert r['fraction_positive_bidirectional_margin']==0
    assert r['pair_confidence'][0]['row_second_best']==.82
    o['min_median_bidirectional_margin_short']=.01
    assert not validate_candidate(r,o) and r['reason']=='below_min_median_bidirectional_margin_short'
    matrix=np.array([[.8,.7,.2],[.4,.9,.1]])
    unique=uniqueness_matrices(matrix)
    assert unique['row_margin'][0,0]==pytest.approx(.1)
    assert unique['row_margin'][0,1]==pytest.approx(-.1)  # competitor is .8, not self .7
    assert unique['column_margin'][0,0]==pytest.approx(.4)
    singleton=uniqueness_matrices(np.array([[.9]]))
    assert np.isnan(singleton['bidirectional_margin'][0,0])


def test_high_cosine_weak_letter_evidence_can_be_rejected():
    c=diagonal(6,.9);e=np.full(c.shape,-.3)
    baseline=decode(c,letter_evidence=e)
    assert baseline['regions'] and baseline['regions'][0]['mean_letter_evidence']==pytest.approx(-.3)
    verified=decode(c,letter_evidence=e,verify_with_letter_evidence=True,
                    min_mean_letter_evidence=0,min_letter_positive_fraction=.5)
    assert not verified['regions']
    assert any(r['reason']=='below_min_mean_letter_evidence' for r in verified['rejected'])
    with pytest.raises(ValueError,match='without image-only letter evidence'):
        decode(c,verify_with_letter_evidence=True)


def test_strong_core_extends_coherent_weaker_tail_in_both_pipeline_and_helper():
    c=diagonal(13)
    for i in (6,7):c[i,i]=.58;c[i,10:13]=.99;c[10:13,i]=.99
    for i in range(8,13):c[i,i]=0
    r=decode(c)
    assert any((7,7) in x['pairs'] and (0,0) in x['pairs'] for x in r['regions'])
    anchors=np.eye(13,dtype=bool);anchors[6,6]=anchors[7,7]=False
    path=steps(c,[(i,i) for i in range(6)])
    extended,added=extend_path(path,c,c-.5,[np.arange(13),np.arange(13)],anchors,uniqueness_matrices(c),options())
    assert added==[(6,6),(7,7)]
    assert [(s['i'],s['j']) for s in extended][-2:]==[(6,6),(7,7)]


def test_large_gap_splits_into_two_independently_accepted_regions():
    c=np.zeros((14,14))
    for i in [*range(5),*range(9,14)]:c[i,i]=.95
    result=decode(c,gap_open=.02,gap_extend=.01,enable_region_merging=True)
    assert len(result['regions'])==2
    assert all(r['support']==[5,5] and r['maximum_internal_gap']==0 for r in result['regions'])
    geo=dict(source_size=[140,10],crop=[0,0,140,10],scale_x=1)
    assert not region_mask(result['regions'],0,geo,4,10)[:,50:90].any()


def test_one_window_gap_stays_inside_normal_region():
    c=diagonal(7);c[3,3]=0
    r=decode(c)['regions'][0]
    assert r['support']==[6,6] and r['maximum_internal_gap']==1 and r['true_gaps']==2
    assert r['filled_physical']==[[3],[3]]
    assert r['score']==pytest.approx(r['matching_reward']-r['gap_penalty']-r['repeat_penalty'])
    assert r['score_per_match']==pytest.approx(r['score']/r['matched_pairs'])


def test_larger_comparable_candidate_dominates_contained_shorter_candidate():
    c=diagonal(12);c[0,0]=.80;c[0,9:12]=.98;c[9:12,0]=.98
    for i in range(7,12):c[i,i]=0
    result=decode(c)
    full=next(r for r in result['regions'] if (0,0) in r['pairs'])
    assert full['support']==[7,7]
    assert any(r['dominated_by']==full['candidate_id'] and r['reason'].startswith('dominated_by_candidate_') for r in result['rejected'])


def test_distant_true_regions_never_merge():
    c=np.zeros((20,20))
    for i in [*range(5),*range(15,20)]:c[i,i]=.95
    assert len(decode(c,enable_region_merging=True)['regions'])==2


def test_rectangular_valid_tokens_no_padding_in_cosine_margins_paths_or_masks():
    a=torch.eye(9);b=torch.eye(9)[:7];a[3]=0;a[8]=0;b[3]=0
    va=torch.ones(9,dtype=torch.bool);va[[3,8]]=False
    vb=torch.ones(7,dtype=torch.bool);vb[3]=False
    r=match_features(a,b,np.arange(9),np.arange(7),token_valid_a=va,token_valid_b=vb,
        alignment_mode='local_repeat_dtw',score_mode='raw',threshold=.5)
    assert r['cosine'].shape==r['rewards'].shape==(7,6)
    assert all(3 not in c['supported_physical'][s]+c['filled_physical'][s] for c in r['regions'] for s in (0,1))
    assert all(c['matched_pairs']==len(c['pair_confidence']) for c in r['candidates'])
    assert all(c['maximum_internal_gap']<=1 for c in r['regions'])
    assert all(c['true_gaps']==0 and c['path_density']==1 for c in r['regions'])
    with pytest.raises(ValueError,match='nonzero'):
        match_features(a,b,np.arange(9),np.arange(7))


def test_rtl_unequal_sequences_keep_physical_identity():
    c=np.zeros((7,10))
    for i in range(6):c[i,i+3]=.95
    r=local_repeat_regions(c,np.arange(7)[::-1],np.arange(10)[::-1],threshold=.5,score_mode='raw')
    region=r['regions'][0]
    assert region['pairs']==[(i,i+3) for i in range(6)]
    assert region['supported_physical']==[[1,2,3,4,5,6],[1,2,3,4,5,6]]
    assert region['physical_pairs'][0]==(6,6)


def test_same_actual_notebook_session_population_cli_and_saved_masks(evaluation_fixture,tmp_path,monkeypatch):
    import evaluate,evaluation_utils
    from train import load_checkpoint
    root,cp=evaluation_fixture
    model,text,cfg,saved=load_checkpoint(cp)
    def frozen_output(images):
        f=torch.eye(7,cfg.embedding_dim).unsqueeze(0).expand(len(images),-1,-1)
        return {**{k:f for k in ('local','context','fused')},
                'token_valid':torch.ones(len(images),7,dtype=torch.bool),
                'physical_window_indices':torch.arange(6,-1,-1).expand(len(images),-1)}
    monkeypatch.setattr(model,'forward',frozen_output)
    monkeypatch.setattr(evaluate,'load_checkpoint',lambda *args:(model,text,cfg,saved))
    monkeypatch.setattr(evaluation_utils,'load_checkpoint',lambda *args:(model,text,cfg,saved))
    session=EvaluationSession(cp,root,device='cpu')
    ns=dict(session=session)
    exec(notebook_source('# EVALUATION-ONLY MATCHING SETTINGS.'),ns)
    pair=session.pairs[0];expected=session.evaluate_pair(pair)
    ns.update(compute_pair_metrics=compute_pair_metrics,load_pair_ground_truth=load_pair_ground_truth,ground_truth_cache={})
    exec(notebook_source('KEY_METRICS ='),ns)
    sample=dict(base=dict(pair=pair,lines=expected['lines'],representation='fused',cosine=expected['cosine']))
    notebook=ns['_rescore_cached'](sample,session.settings)
    assert notebook['match']['regions']==expected['match']['regions']
    assert notebook['match']['rejected']==expected['match']['rejected']
    captured=[];original=session.evaluate_pair
    def capture(*a,**kw):
        r=original(*a,**kw);captured.append(r);return r
    monkeypatch.setattr(session,'evaluate_pair',capture)
    session.pairs=[pair]
    evaluate_population(session)
    assert captured[0]['match']['candidates']==expected['match']['candidates']
    report=evaluate.evaluate_pair(cp,[s['image'] for s in pair['sides']],tmp_path/'cli',
        line_bboxes=[s.get('bbox') for s in pair['sides']],**session.settings)
    assert report['regions']==expected['match']['regions']
    assert report['rejected']==expected['match']['rejected']
    # Exercise argument parsing and saved-record source boxes as well as the
    # public evaluate_pair function. No loader/network training takes place.
    records=[dict(sample_id=f'record_{side}',sides=[pair['sides'][side]]) for side in (0,1)]
    class RecordView:
        def __len__(self):return len(self.records)
    view=RecordView();view.records=records
    loader=SimpleNamespace(dataset=view)
    monkeypatch.setattr(evaluate,'build_loaders',lambda *args:[loader]*3)
    argv=['--mode','shared_regions','--checkpoint',str(cp),'--dataset',str(root),
          '--split','val','--record-indices','0','1','--output',str(tmp_path/'cli_main')]
    for key,value in session.settings.items():
        if value is None:continue
        if isinstance(value,bool):argv.append('--'+('' if value else 'no-')+key.replace('_','-'))
        else:argv.extend(['--'+key.replace('_','-'),str(value)])
    parsed_report=evaluate.main(argv)
    assert parsed_report['regions']==expected['match']['regions']
    assert parsed_report['rejected']==expected['match']['rejected']
    from PIL import Image
    for s in (0,1):
        np.testing.assert_array_equal(notebook['masks'][s],expected['masks'][s])
        np.testing.assert_array_equal(captured[0]['masks'][s],expected['masks'][s])
        np.testing.assert_array_equal(np.asarray(Image.open(tmp_path/'cli'/f'line_{"ab"[s]}_mask.png')),expected['masks'][s])
        np.testing.assert_array_equal(np.asarray(Image.open(tmp_path/'cli_main'/f'line_{"ab"[s]}_mask.png')),expected['masks'][s])
    assert expected['match']['regions'][0]['support']==[7,7]
    # Labels, transcripts and annotations have no prediction effect.
    hostile=dict(pair,target=0,label='no_shared_content',annotations=[dict(mask='/missing')]*2)
    assert session.predict_pair(hostile)['match']['candidates']==expected['match']['candidates']


def test_unique_sampling_caps_at_population_and_never_duplicates(capsys):
    rows=[dict(sample_id=str(i),target=1) for i in range(4)]
    result=sample_random_pairs(rows+[rows[0]],5)
    assert len(result)==len({r['sample_id'] for r in result})==4
    assert 'Requested 5 unique positive samples, but only 4' in capsys.readouterr().out
    negative=[dict(r,target=0) for r in rows]
    assert len(sample_random_pairs(negative+[negative[1]],5,aligned=False))==4


def test_partial_overlap_negative_label_is_not_local_false_positive(tmp_path):
    a=tmp_path/'a.txt';b=tmp_path/'b.txt';a.write_text('سلام عالم');b.write_text('شيء عالم')
    pair=dict(target=0,label='no_shared_content',sides=[dict(text=str(a)),dict(text=str(b))])
    assert evaluation_category(pair,[None,None])['evaluation_category']=='partial_overlap'
    assert evaluation_category(pair,[np.ones((3,8)),np.ones((3,9))])['evaluation_category']=='partial_overlap'
    assert evaluation_category(pair,[np.zeros((3,8)),np.zeros((3,9))])['evaluation_category']=='true_no_overlap'
    b.write_text('شيء مختلف')
    assert evaluation_category(pair,[None,None])['evaluation_category']=='unknown'
    rows=[]
    for category,regions,precision,recall in [('partial_overlap',1,.8,.6),('true_no_overlap',0,0,0),('unknown',1,None,None)]:
        rows.append(dict(sample_id=category,target=0,constructed_negative=False,evaluation_category=category,
            region_count=regions,pair_score=regions,**{f'{s}_{k}':v for s in ('a','b') for k,v in
                [('status','available' if precision is not None else 'unavailable'),('precision',precision),('recall',recall)]}))
    population=dict(rows=rows,split='val',population='fixture',evaluated_unique_lines=6,saved_split_unique_lines=6,distribution_stats={})
    summary=aggregate_summary(population)
    assert summary['true_negative_false_positive_rate']==0
    assert summary['partial_overlap_precision']==.8 and summary['partial_overlap_recall']==.6
    assert 'manifest_negative_false_positive_rate' not in summary


def test_merging_requires_actual_bridge_evidence_and_rescores():
    c=diagonal(11);c[5,5]=.55;o=options(enable_region_merging=True)
    physical=[np.arange(11),np.arange(11)];anchors=np.eye(11,dtype=bool);u=uniqueness_matrices(c)
    regions=[]
    for k,ids in enumerate((range(5),range(6,11))):
        r=candidate_statistics(steps(c,[(i,i) for i in ids]),c,c-.5,physical,anchors,u,o,candidate_id=f'C{k}')
        assert validate_candidate(r,o);r.update(accepted=True);regions.append(r)
    candidates=list(regions)
    merged=merge_regions(regions,candidates,c,c-.5,physical,anchors,u,o)
    assert len(merged)==1 and merged[0]['merged_from']==['C0','C1']
    assert (5,5) in merged[0]['pairs'] and merged[0]['maximum_internal_gap']==0
    assert merged[0]['score']==pytest.approx(sum(s['delta'] for s in merged[0]['path_steps']))
    c[5,5]=0
    candidates=[]
    for k,ids in enumerate((range(5),range(6,11))):
        r=candidate_statistics(steps(c,[(i,i) for i in ids]),c,c-.5,physical,anchors,uniqueness_matrices(c),o,candidate_id=f'C{k}')
        validate_candidate(r,o);r.update(accepted=True);candidates.append(r)
    assert len(merge_regions(candidates,list(candidates),c,c-.5,physical,anchors,uniqueness_matrices(c),o))==2


def test_weak_larger_competitor_does_not_win_only_by_supported_length():
    c=diagonal(7)
    c[5,5]=c[6,6]=.551
    p=[np.arange(7),np.arange(7)];a=np.eye(7,dtype=bool);u=uniqueness_matrices(c);o=options()
    candidates=[]
    for k,n in enumerate((5,7)):
        candidate=candidate_statistics(steps(c,[(i,i) for i in range(n)]),c,c-.5,p,a,u,o,candidate_id=f'C{k}')
        assert validate_candidate(candidate,o);candidates.append(candidate)
    assert select_candidates(candidates,o)==[candidates[0]]
    assert candidates[1]['reason']=='crossing_or_reused_windows'


def test_dominance_quality_tolerance_cannot_compound_through_subpaths():
    # All routes have the same physical support; extra repeat pairs increase
    # length. A 10% tolerance must not remove an .80 core via .73 then .66.
    pairs=[(i,i) for i in range(5)]
    candidates=[]
    for k,(extra,quality) in enumerate((([], .80),([(0,1)], .73),([(0,1),(1,2)], .66))):
        candidates.append(dict(candidate_id=f'C{k}',valid=True,accepted=False,dominated_by=None,
            support=[5,5],support_a=5,support_b=5,pairs=pairs+extra,matched_pairs=5+len(extra),
            score_per_match=quality,mean_reward=quality,path_density=1,mutual_anchor_fraction=1,
            median_bidirectional_margin=.1,score=quality*(5+len(extra)),logical_ranges=[(0,4),(0,4)]))
    selected=select_candidates(candidates,options())
    assert selected==[candidates[0]]
    assert candidates[0]['dominated_by'] is None
    assert candidates[1]['dominated_by']=='C2'
