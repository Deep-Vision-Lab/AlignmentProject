"""Independent passages can reorder without sharing source windows."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from alignment_candidates import compatible_regions,decode_candidates
from evaluate import MATCH_DEFAULTS,local_repeat_regions,resolved_match_settings
from evaluation_utils import predict_cached_pair,revalidate_cached_candidates
from parameters import Config


def decode(cosine,**overrides):
    settings=dict(threshold=.5,score_mode='raw',normal_min_distinct_a=3,
        normal_min_distinct_b=3,normal_min_matched_pairs=3,allow_short_regions=False,
        use_path_anchor_check=False,enable_path_extension=False,enable_region_merging=False,
        min_path_density=.95)
    settings.update(overrides)
    return local_repeat_regions(cosine,np.arange(len(cosine)),np.arange(cosine.shape[1]),**settings)


@pytest.mark.parametrize('extension,merging',[(False,False),(True,False),(False,True),(True,True)])
def test_two_passages_in_reverse_order_are_both_selected(extension,merging):
    cosine=np.full((18,18),-.4)
    first=[(i,i+12) for i in range(5)]
    second=[(i+12,i) for i in range(5)]
    for cell in first+second:cosine[cell]=.95
    assert len(decode(cosine)['regions'])==1
    result=decode(cosine,allow_reordered_regions=True,enable_path_extension=extension,enable_region_merging=merging)
    assert len(result['regions'])==2
    assert {tuple(c['pairs']) for c in result['regions']}=={tuple(first),tuple(second)}
    for side in (0,1):
        used=[q[side] for c in result['regions'] for q in c['pairs']]
        assert len(used)==len(set(used))
    assert all(all(a<=c and b<=d for (a,b),(c,d) in zip(r['pairs'],r['pairs'][1:])) for r in result['regions'])


def test_reordering_keeps_overlap_forbidden_on_both_lines():
    left=dict(logical_ranges=[(0,4),(10,14)])
    crossing=dict(logical_ranges=[(10,14),(0,4)])
    assert compatible_regions(left,crossing,True)
    assert not compatible_regions(left,crossing,False)
    for conflict in ([(3,7),(0,4)],[(10,14),(12,17)]):
        assert not compatible_regions(left,dict(logical_ranges=conflict),True)


def test_separated_short_passages_corroborate_without_accepting_an_isolated_hit():
    cosine=np.full((18,18),-.4)
    for start in (0,8):
        for offset in range(4):cosine[start+offset,start+offset]=.9
    settings=dict(allow_reordered_regions=True,split_gap_candidates=True,
        normal_min_distinct_a=7,normal_min_distinct_b=7,normal_min_matched_pairs=7,
        allow_short_regions=True,short_min_distinct_a=3,short_min_distinct_b=3,
        require_corroborated_short_regions=True,short_min_mean_cosine=.7,
        short_min_median_cosine=.7)
    assert len(decode(cosine,**settings)['regions'])==2
    cosine[8:12,8:12]=-.4
    isolated=decode(cosine,**settings)
    assert not isolated['regions']
    assert any(c['reason']=='uncorroborated_short_regions' for c in isolated['rejected'])


def test_visual_floor_rejects_low_cosine_despite_positive_letter_rewards():
    cosine=np.eye(8)*.4;rewards=np.eye(8)*2.-.5
    assert decode(cosine,rewards=rewards)['regions']
    filtered=decode(cosine,rewards=rewards,normal_min_mean_cosine=.6)
    assert not filtered['regions']
    assert any(c['reason']=='below_normal_min_mean_cosine' for c in filtered['rejected'])


def test_cli_accepts_fractional_visual_floor_and_passage_flags(monkeypatch):
    import evaluate
    captured={}
    def predict(*args,**settings):
        captured.update(settings)
        return dict(regions=[])
    monkeypatch.setattr(evaluate,'evaluate_pair',predict)
    evaluate.main(['--mode','shared_regions','--checkpoint','unused.pt','--image-a','a.png','--image-b','b.png',
        '--normal-min-mean-cosine','.6','--allow-reordered-regions',
        '--split-gap-candidates','--require-corroborated-short-regions'])
    assert captured['normal_min_mean_cosine']==.6
    assert all(captured[key] for key in ('allow_reordered_regions','split_gap_candidates',
                                       'require_corroborated_short_regions'))


def test_gap_fragments_survive_when_joined_route_fails_density():
    cosine=np.eye(10)*.95;cosine[4,4]=cosine[5,5]=-.4
    path=[dict(i=i,j=i,transition='diagonal',cosine=.95,reward=.45) for i in range(4)]
    path += [dict(i=i,j=None,transition='gap_a',cosine=None,reward=None) for i in (4,5)]
    path += [dict(i=None,j=i,transition='gap_b',cosine=None,reward=None) for i in (4,5)]
    path += [dict(i=i,j=i,transition='diagonal',cosine=.95,reward=.45) for i in range(6,10)]
    raw=[dict(path_steps=path,score=2.)]
    settings=resolved_match_settings(dict(threshold=.5,normal_min_distinct_a=3,normal_min_distinct_b=3,
        normal_min_matched_pairs=3,allow_short_regions=False,min_path_density=.95,
        max_gap=4,use_path_anchor_check=False,enable_path_extension=False,enable_region_merging=False))
    physical=[np.arange(10),np.arange(10)];anchors=np.eye(10,dtype=bool)
    old=decode_candidates(raw,cosine,cosine-.5,physical,anchors,settings)
    assert not old['regions']
    new=decode_candidates(raw,cosine,cosine-.5,physical,anchors,dict(settings,split_gap_candidates=True))
    assert len(new['regions'])==2
    assert all(c['fragmented_at_gaps'] and c['path_density']==1 for c in new['regions'])


@pytest.mark.parametrize('reordered',[False,True])
@pytest.mark.parametrize('fragments',[False,True])
@pytest.mark.parametrize('pattern',['reversed','gapped'])
def test_cached_confidence_selection_matches_complete_production_prediction(reordered,fragments,pattern):
    cosine=np.full((18,18),-.4)
    if pattern=='reversed':
        for i in range(5):cosine[i,i+12]=cosine[i+12,i]=.95
    else:
        for i in list(range(5))+list(range(7,12)):cosine[i,i]=.95
    features=torch.eye(18)
    geometry=dict(source_size=[180,4],crop=[0,0,180,4],scale_x=1.)
    lines=[dict(features=dict(fused=features),physical=np.arange(18),geometry=geometry) for _ in range(2)]
    session=SimpleNamespace(config=Config(window_width=10,window_stride=10),split='test')
    settings=resolved_match_settings(dict(threshold=.5,score_mode='raw',normal_min_distinct_a=3,
        normal_min_distinct_b=3,normal_min_matched_pairs=3,allow_short_regions=False,
        use_path_anchor_check=False,enable_path_extension=False,enable_region_merging=False,
        split_gap_candidates=True,allow_reordered_regions=True,max_gap=4,min_path_density=.95))
    pair=dict(sample_id='fixture')
    pool=predict_cached_pair(session,pair,lines,settings=settings,cosine=cosine)
    previous=[c['reason'] for c in pool['match']['candidates']]
    requested=dict(settings,allow_reordered_regions=reordered,split_gap_candidates=fragments)
    cached=revalidate_cached_candidates(session,pool,requested)
    direct=predict_cached_pair(session,pair,lines,settings=requested,cosine=cosine)
    assert [r['pairs'] for r in cached['match']['regions']]==[r['pairs'] for r in direct['match']['regions']]
    for a,b in zip(cached['masks'],direct['masks']):np.testing.assert_array_equal(a,b)
    assert previous==[c['reason'] for c in pool['match']['candidates']]
    with pytest.raises(ValueError,match='generation settings'):
        revalidate_cached_candidates(session,pool,dict(requested,repeat_penalty=.7))
