"""Whole-line competition, word identity, spacing and exact mask geometry."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from evaluate import match_features, region_mask, resolved_match_settings
from evaluation_utils import predict_cached_pair, revalidate_cached_candidates
from parameters import Config
from text_embedding import OrthogonalCharEmbedding
from word_alignment import full_span_path, word_lattice
from word_alignment import select_word_candidates


def test_two_strong_words_can_corroborate_without_a_weak_third_fragment():
    from copy import deepcopy
    options=resolved_match_settings(dict(require_corroborated_short_regions=True,
        word_min_distinct_types=3,word_min_total_characters=9,
        normal_min_distinct_a=9,normal_min_distinct_b=9,
        word_max_nll=1.6,word_min_cosine=.6))
    candidates=[]
    for word,lo,hi in [('الذين',0,4),('كفروا',5,9)]:
        candidates.append(dict(candidate_type='word',word=word,word_nll=[.4,.5],
            word_pair_posterior=.4,mean_cosine=.8,logical_ranges=[[lo,hi],[lo,hi]],
            supported_physical=[list(range(lo,hi+1))]*2,filled_physical=[[],[]]))
    assert not select_word_candidates(deepcopy(candidates),options)
    enabled=dict(options,word_strong_pair_max_nll=.6,word_strong_pair_min_cosine=.75)
    assert {c['word'] for c in select_word_candidates(deepcopy(candidates),enabled)}=={'الذين','كفروا'}
    weak=deepcopy(candidates);weak[0]['word_nll']=[.4,.7]
    assert not select_word_candidates(weak,enabled)
    weak=deepcopy(candidates);weak[0]['mean_cosine']=.7
    assert not select_word_candidates(weak,enabled)
    assert not select_word_candidates(deepcopy(candidates[:1]),enabled)
    # The exception does not bypass combined character/support requirements.
    assert not select_word_candidates(deepcopy(candidates),dict(enabled,word_min_total_characters=11))
    assert not select_word_candidates(deepcopy(candidates),dict(enabled,normal_min_distinct_a=11))
    # Enabling it cannot alter a pair that already passed ordinary corroboration.
    ordinary=dict(options,word_min_distinct_types=2)
    before=select_word_candidates(deepcopy(candidates),ordinary)
    after=select_word_candidates(deepcopy(candidates),dict(ordinary,word_strong_pair_max_nll=.6,word_strong_pair_min_cosine=.75))
    assert [c['logical_ranges'] for c in before]==[c['logical_ranges'] for c in after]


@pytest.fixture
def word_setup(tmp_path):
    vocabulary=list(dict.fromkeys(' اموالهبتيغن'))
    text=OrthogonalCharEmbedding(128,4096)
    text.letter_evidence_prior=dict(vocabulary=vocabulary,prior=[1/len(vocabulary)]*len(vocabulary))
    path=tmp_path/'words.json'
    path.write_text(json.dumps(dict(source_split='train',checkpoint_sha256='fixture',
        words={'اموالهم':10,'الله':100,'ابتغا':10,'من':100})))
    settings=resolved_match_settings(dict(alignment_mode='word_dtw',similarity_mode='letter_evidence',
        candidate_similarity_mode='letter_evidence',word_vocabulary_path=str(path),
        allow_reordered_regions=True,enable_path_extension=False,enable_region_merging=False,
        word_keep_long_regions=False,
        word_min_posterior=.04,word_min_posterior_short=.1,word_min_cosine=.3))
    return text,Config(competition_temperature=.1),settings


def predict(a,b,setup):
    text,config,settings=setup
    return match_features(text.encode(a),text.encode(b),np.arange(len(a)),np.arange(len(b)),
        text_encoder=text,config=config,word_boundaries=[np.ones(len(a)+1),np.ones(len(b)+1)],**settings)


def test_common_letters_inside_another_word_do_not_become_a_word_match(word_setup):
    result=predict('اموالهم','الله',word_setup)
    assert not result['regions']


@pytest.mark.parametrize('spaces',[0,1,4])
def test_word_order_can_change_with_zero_one_or_many_space_windows(word_setup,spaces):
    gap=' '*spaces
    result=predict('ابتغا'+gap+'الله'+gap+'من','من'+gap+'ابتغا'+gap+'الله',word_setup)
    assert {c['word'] for c in result['regions']}=={'ابتغا','الله','من'}
    for c in result['regions']:
        for side in (0,1):
            lo,hi=c['logical_ranges'][side]
            assert c['support'][side]==hi-lo+1


def test_full_word_traceback_consumes_both_ends_and_all_windows():
    cosine=np.array([[.9,.1,.1],[.8,.7,.1],[.1,.2,.8],[.1,.2,.9]])
    path=full_span_path(cosine,(0,3),(0,2))
    assert path[0]==(0,0) and path[-1]==(3,2)
    assert {a for a,_ in path}==set(range(4))
    assert {b for _,b in path}==set(range(3))


def test_word_masks_clip_to_word_edges_in_source_coordinates():
    geometry=dict(source_size=[100,3],crop=[0,0,100,3],scale_x=1.)
    region=dict(source_word_ranges=[[21.,39.],[0.,10.]],supported_physical=[[1,2],[]],filled_physical=[[],[]])
    mask=region_mask([region],0,geometry,32,16)
    assert mask[:,21:39].all() and not mask[:,:21].any() and not mask[:,39:].any()


def test_word_mode_rejects_missing_word_prior(word_setup):
    _,_,settings=word_setup
    with pytest.raises(ValueError,match='training vocabulary'):
        resolved_match_settings(dict(settings,word_vocabulary_path=None))


def test_hybrid_preserves_a_long_visual_passage_outside_the_word_vocabulary(word_setup):
    text,config,settings=word_setup
    value='يغيغيغييغيغ'
    result=predict(value,value,(text,config,dict(settings,word_keep_long_regions=True)))
    assert any(c['candidate_type']=='visual_passage' for c in result['regions'])
    assert all(c['candidate_type'] in ('word','visual_passage') for c in result['regions'])


def test_an_isolated_common_word_does_not_corroborate_a_line_pair(word_setup):
    text,config,settings=word_setup
    result=predict('الله','الله',(text,config,dict(settings,require_corroborated_short_regions=True)))
    assert not result['regions']
    assert any(c['reason']=='uncorroborated_word_regions' for c in result['rejected'])
    positive=predict('ابتغااللهمن','منابتغاالله',
        (text,config,dict(settings,require_corroborated_short_regions=True)))
    assert {c['word'] for c in positive['regions']}=={'ابتغا','الله','من'}


def test_cached_word_acceptance_matches_full_inference_and_ignores_pair_transcripts(word_setup):
    text,config,settings=word_setup
    strings=['ابتغااللهمن','منابتغاالله']
    lines=[dict(features=dict(fused=text.encode(s)),physical=np.arange(len(s)),
        geometry=dict(source_size=[len(s)*10,3],crop=[0,0,len(s)*10,3],scale_x=1.),
        word_boundaries=(np.ones(len(s)+1),np.arange(len(s)+1)*10)) for s in strings]
    session=SimpleNamespace(config=config,text_encoder=text,split='test',settings=settings,checkpoint_sha256='fixture')
    pair=dict(sample_id='fixture',sides=[dict(image='a.png'),dict(image='b.png')])
    pool=predict_cached_pair(session,pair,lines)
    requested=dict(settings,word_min_posterior=.2,word_min_posterior_short=.3)
    replay=revalidate_cached_candidates(session,pool,requested)
    hostile=dict(pair,sides=[dict(side,text='does-not-exist.txt',raw_text='wrong transcription') for side in pair['sides']],target=0)
    direct=predict_cached_pair(session,hostile,lines,settings=requested)
    assert [r['pairs'] for r in replay['match']['regions']]==[r['pairs'] for r in direct['match']['regions']]
    for a,b in zip(replay['masks'],direct['masks']):np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='generation settings'):
        revalidate_cached_candidates(session,pool,dict(requested,word_boundary_weight=3.))
