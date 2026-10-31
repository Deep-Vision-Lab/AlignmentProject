"""Source-coordinate metrics, constrained selection and exact notebook-flow parity."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import optuna
from PIL import Image
import pytest

from scripts.eval import mask_quantitative as q
from scripts.eval import tune_masks as tuning
from evaluation_utils import predict_cached_pair
from evaluate import region_mask
from test_evaluation_notebook import evaluation_fixture


def span(intervals, width=20, height=3):
    result = np.zeros((height,width),bool)
    for a,b in intervals: result[:,a:b] = True
    return result


def test_perfect_missed_overmask_and_internal_holes():
    gt = span([(3,10)])
    perfect = q.binary_metrics(gt,gt)
    assert all(perfect[k]==1 for k in ('iou','dice','precision','recall'))
    missed = q.binary_metrics(np.zeros_like(gt),gt)
    assert all(missed[k]==0 for k in ('iou','dice','precision','recall'))
    over = q.binary_metrics(span([(0,14)]),gt)
    assert over['iou']==.5 and over['recall']==1 and over['precision']==.5
    hole = q.binary_metrics(span([(3,5),(7,10)]),gt)
    assert hole['iou']==pytest.approx(5/7) and hole['precision']==1
    assert q.region_counts(span([(3,5),(7,10)]),gt)==dict(tp=0,fp=2,fn=1,matches=[])


def pair_row(name, category, masks, truth):
    scores = [q.binary_metrics(a,b) for a,b in zip(masks,truth)]
    return dict(sample_id=name,category=category,bootstrap_group=name,
        **{k:float(np.mean([s[v] for s in scores])) for k,v in [('positive_iou','iou'),('positive_dice','dice'),('precision','precision'),('recall','recall')]},
        negative_false_positive=any(s['predicted_pixels'] for s in scores),
        negative_masked_area_fraction=float(np.mean([s['foreground_fraction'] for s in scores])),
        candidate_limit_hit=False,region_eligible=True,region_tp=0,region_fp=0,region_fn=0)


def test_negative_either_side_and_all_empty_denominators():
    gt = span([(0,10)]); black = span([])
    positive = pair_row('p','positive',[gt,black],[gt,gt])
    negatives = [pair_row('n0','negative',[black,black],[black,black]),
                 pair_row('n1','negative',[gt,black],[black,black]),
                 pair_row('n2','negative',[black,gt],[black,black])]
    result = q.aggregate([positive,*negatives])
    assert result['positive_iou']==.5 and result['positive_dice']==.5
    assert result['negative_false_positive_pairs']==2 and result['negative_pair_denominator']==3
    assert result['negative_pair_fpr']==pytest.approx(2/3)
    assert result['negative_masked_area_fraction']==pytest.approx(1/6)
    ci = q.confidence_intervals([positive,negatives[0]],resamples=20)
    assert ci['estimates']['negative_fpr_binomial']['high']>0


def test_bootstrap_respects_shared_page_dependencies():
    gt=span([(0,10)]);black=span([])
    rows=[pair_row('p1','positive',[gt,gt],[gt,gt]),pair_row('p2','positive',[black,black],[gt,gt]),
          pair_row('n1','negative',[black,black],[black,black]),pair_row('n2','negative',[gt,black],[black,black])]
    rows[2]['bootstrap_group']=rows[3]['bootstrap_group']='same-page'
    report=q.confidence_intervals(rows,resamples=200,seed=42)['estimates']
    assert report['positive_iou']['low']==0 and report['positive_iou']['high']==1
    assert report['negative_pair_fpr']['low'] is None
    assert not report['negative_fpr_binomial']['available']


def test_unequal_widths_physical_indices_and_full_height_guard():
    geometry = dict(source_size=[25,4],crop=[0,0,25,4],scale_x=1.)
    mask = region_mask([dict(supported_physical=[[0,2],[0]],filled_physical=[[],[]])],0,geometry,5,5)
    assert mask.shape==(4,25)
    assert np.array_equal(mask>0,span([(0,5),(10,15)],width=25,height=4))
    # Invalid physical windows 1,3,4 are not renumbered into the valid indices.
    valid = np.array([True,False,True,False,False])
    assert np.array_equal(np.arange(5)[valid],[0,2])
    a=q.binary_metrics(mask,span([(0,5)],25,4))
    b=q.binary_metrics(span([(0,5)],10,7),span([(0,5)],10,7))
    assert (a['iou']+b['iou'])/2==.75
    irregular=span([(0,5)]); irregular[1,0]=False
    with pytest.raises(ValueError,match='full-height'): q.intervals(irregular)
    import torch
    from evaluate import match_features
    full=torch.zeros(5,2);full[0,0]=1;full[2,1]=1
    matched=match_features(full,torch.eye(2),np.arange(5),np.arange(2),token_valid_a=valid,
        score_mode='raw',threshold=.5,normal_min_distinct_a=2,normal_min_distinct_b=2,
        normal_min_matched_pairs=2,allow_short_regions=False,max_gap=1,min_mutual_anchors_long=1,
        min_mutual_anchor_fraction_long=0.,enable_path_extension=False,enable_region_merging=False)
    actual=region_mask(matched['regions'],0,geometry,5,5)
    assert np.array_equal(actual,mask)


def test_one_to_one_region_matching_and_micro_counts():
    result = q.region_counts(span([(0,5),(10,20)]),span([(0,6),(11,20)]))
    assert result['tp']==2 and result['fp']==result['fn']==0
    assert len({m[0] for m in result['matches']})==len({m[1] for m in result['matches']})==2
    assert q.region_counts(span([(0,20)]),span([(0,5),(15,20)]))['tp']==0


def trial(value, fpr, area, number):
    t=optuna.trial.create_trial(value=value,user_attrs={'metrics':dict(negative_pairs=100,
        negative_pair_fpr=fpr,negative_masked_area_fraction=area)})
    t.number=number
    return t


def test_feasibility_not_unconstrained_best_and_tie_break():
    bad=trial(.99,.06,.01,0); first=trial(.7,.05,.03,1); second=trial(.7,.04,.01,2)
    assert tuning.choose_trial([bad,first,second],.05).number==2
    assert tuning.choose_trial([bad],.05) is None


def test_no_feasible_trial_exports_tradeoffs_without_selection(tmp_path):
    study=optuna.create_study(direction='maximize');study.add_trial(trial(.99,.1,.02,0))
    (tmp_path/'selected.json').write_text('{"stale": true}')
    result=tuning.export_studies([(1,study)],tmp_path,.05,dict(version='test'))
    assert result is None
    assert json.loads((tmp_path/'selection_status.json').read_text())['status']=='NO_FEASIBLE_TRIAL'
    assert (tmp_path/'trials.csv').exists() and not (tmp_path/'selected.json').exists()
    assert list((tmp_path/'previous_selections').glob('*/selected.json'))


def test_explicit_pair_manifest_requires_review_and_saved_membership(evaluation_fixture,tmp_path):
    root,checkpoint=evaluation_fixture;configure_fixture(root)
    args=arguments(root,checkpoint,tmp_path/'explicit')
    session,prepared,_=q.build_cache(args,'val')
    source=prepared[0]['record'];path=tmp_path/'pairs.json'
    row=dict(sample_id='manual-pair',split='val',target=1,verified=False,
        sides=[dict(image=r['image']) for r in source['images']],masks=[r['path'] for r in source['masks']])
    path.write_text(json.dumps(dict(schema_version=1,pairs=[row])))
    with pytest.raises(ValueError,match='verified'):q.explicit_pairs(session,path)
    row['verified']=True;path.write_text(json.dumps(dict(schema_version=1,pairs=[row])))
    assert q.explicit_pairs(session,path)[0]['sample_id']=='manual-pair'
    row['sides'][0]['image']=str(root/'A0.png');path.write_text(json.dumps(dict(schema_version=1,pairs=[row])))
    with pytest.raises(ValueError,match='outside'):q.explicit_pairs(session,path)


def test_search_bounds_conditionals_and_anchor_aliases():
    _,baseline=q.notebook_baseline()
    study=optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=42))
    for i in range(60):
        t=study.ask(); s=tuning.suggest(t,baseline,2)
        assert s['normal_min_distinct_a']==s['normal_min_distinct_b']
        assert 4<=s['normal_min_distinct_a']<=10
        assert .01<=s['gap_extend']<=s['gap_open']<=.6
        assert s['candidate_similarity_mode']==s['similarity_mode']=='letter_evidence'
        assert s['threshold']==.5 and s['max_candidates']==512
        if s['allow_short_regions']:
            assert 3<=s['short_min_distinct_a']<=min(5,s['normal_min_distinct_a'])
            assert s['short_min_mutual_anchors']==s['min_mutual_anchors_short']
            assert s['short_min_mutual_anchor_fraction']==s['min_mutual_anchor_fraction_short']
            assert s['short_max_internal_gap']<=min(2,s['max_gap'])
        if s['enable_path_extension']: assert s['extension_max_internal_gap']<=min(4,s['max_gap'])
        if s['enable_region_merging']: assert s['merge_max_gap_a']==s['merge_max_gap_b']<=min(4,s['max_gap'])
        study.tell(t,0.)
    assert tuning.baseline_params(baseline) is not None
    assert tuning.suggest(optuna.trial.FixedTrial(tuning.baseline_params(baseline)),baseline,1)==baseline
    resolved=q.effective_settings(baseline)
    assert resolved['derived']['short_anchor_fraction']==.95
    assert resolved['derived']['extension_gap']==2


def configure_fixture(root):
    """Controlled test annotations, not automatically certified real negatives."""
    path=root/'dataset_manifest.jsonl'; records=[json.loads(line) for line in path.read_text().splitlines()]
    for row in records:
        if row.get('split') not in ('val','test'):continue
        negative=row['pair_id'] in ('pair7','pair9','negative')
        row['label_type']='no_shared_content' if negative else 'high_match'
        for side in 'AB':
            mask=root/f'{row["pair_id"]}_{side}_gt.png'
            Image.fromarray((span([] if negative else [(30,120)],160,32)*255).astype(np.uint8)).save(mask)
            row[side]['alignment_mask_path']=mask.name
    path.write_text('\n'.join(json.dumps(row) for row in records))


def arguments(root, checkpoint, output, command='prepare'):
    return tuning.parser().parse_args([command,'--dataset',str(root),'--checkpoint',str(checkpoint),
        '--output',str(output),'--device','cpu','--crop','checkpoint','--pair-source','dataset','--bootstrap','20'])


def test_exact_notebook_prediction_parity_cache_and_gt_separation(evaluation_fixture,tmp_path):
    root,checkpoint=evaluation_fixture;configure_fixture(root)
    args=arguments(root,checkpoint,tmp_path/'result')
    session,prepared,provenance=q.build_cache(args,'val')
    _,settings=q.notebook_baseline();session.settings=settings
    direct=[session.predict_pair(item['inference']) for item in prepared]
    observed=[]
    def record(*a,**kw):
        result=predict_cached_pair(*a,**kw);observed.append(result)
        assert set(a[1])=={'sample_id','sides'}
        assert all('mask' not in side for side in a[1]['sides'])
        return result
    with patch.object(q,'predict_cached_pair',side_effect=record):
        metrics,rows=q.evaluate(session,prepared,'fused',settings)
    for old,new in zip(direct,observed):
        for a,b in zip(old['masks'],new['masks']):assert np.array_equal(a,b)
    expected=q.binary_metrics(direct[0]['masks'][0],prepared[0]['truth'][0])['iou']
    assert rows[0]['positive_iou']==expected
    with patch.object(q.EvaluationSession,'get_line_features',side_effect=AssertionError('cache not reused')):
        q.build_cache(args,'val')
    assert metrics['positive_pairs']==metrics['negative_pairs']==1


def test_split_copy_leakage_is_reported(evaluation_fixture,tmp_path):
    root,checkpoint=evaluation_fixture;configure_fixture(root)
    (root/'A6.png').write_bytes((root/'A0.png').read_bytes())
    args=arguments(root,checkpoint,tmp_path/'leak')
    with pytest.raises(ValueError,match='leak'):q.build_cache(args,'val')
    assert json.loads((args.output/'split_audit.json').read_text())['image_leakage']


def test_missing_or_inconsistent_masks_are_not_silently_dropped(evaluation_fixture,tmp_path):
    root,checkpoint=evaluation_fixture;configure_fixture(root)
    (root/'pair7_A_gt.png').unlink()
    args=arguments(root,checkpoint,tmp_path/'bad')
    with pytest.raises(FileNotFoundError):q.build_cache(args,'val')
    report=json.loads((args.output/'split_audit.json').read_text())
    assert report['status']=='FAILED_SESSION_CONTRACT' and 'pair7_A_gt.png' in report['error']
    Image.fromarray((span([(0,10)],160,32)*255).astype(np.uint8)).save(root/'pair7_A_gt.png')
    with pytest.raises(ValueError,match='inconsistent'):q.build_cache(args,'val')
    report=json.loads((args.output/'val_split_manifest.json').read_text())
    assert report['inconsistent_records'][0]['reason']=='label/GT inconsistency'


def test_smoke_sqlite_resume_and_selection_lock(evaluation_fixture,tmp_path):
    root,checkpoint=evaluation_fixture;configure_fixture(root)
    output=tmp_path/'smoke';args=arguments(root,checkpoint,output,'tune')
    command=['tune','--dataset',str(root),'--checkpoint',str(checkpoint),'--output',str(output),
        '--device','cpu','--crop','checkpoint','--pair-source','dataset','--bootstrap','20',
        '--stage1-trials','2','--stage2-trials','3']
    studies=tuning.main(command)
    assert [len(s.trials) for _,s in studies]==[2,3]
    assert json.loads((output/'selected.json').read_text())['status']=='SELECTED_FEASIBLE'
    tuning.main(command+['--resume'])
    assert [len(optuna.load_study(study_name=f'quantitative-mask-iou-stage{i}',storage='sqlite:///'+str(output/'study.sqlite3')).trials) for i in (1,2)]==[2,3]
    for _,s in studies:
        for t in s.trials:
            if t.state==optuna.trial.TrialState.COMPLETE:
                assert 'metrics' in t.user_attrs and 'effective_settings' in t.user_attrs
    changed=command+['--resume','--max-negative-pair-fpr','.1']
    with pytest.raises(ValueError,match='provenance'):tuning.main(changed)
    test_command=command.copy();test_command[0]='test'
    tuning.main(test_command)
    assert (output/'paper_table.csv').exists() and (output/'paper_table.tex').exists()
    reports=json.loads((output/'test/methods.json').read_text())
    assert len(reports)==4 and reports[2]['ablation_without_retuning']
    with pytest.raises(ValueError,match='frozen'):tuning.main(command+['--resume'])


def test_failed_objective_keeps_actual_error_and_has_no_selected_trial(evaluation_fixture,tmp_path):
    root,checkpoint=evaluation_fixture;configure_fixture(root);output=tmp_path/'failed'
    original=tuning.evaluate
    def failure(*a,**kw):
        if Path(a[4]).name=='baseline':return original(*a,**kw)
        raise RuntimeError('controlled matcher failure')
    args=['tune','--dataset',str(root),'--checkpoint',str(checkpoint),'--output',str(output),
        '--device','cpu','--crop','checkpoint','--pair-source','dataset','--bootstrap','20','--stage1-trials','1','--stage2-trials','1']
    with patch.object(tuning,'evaluate',side_effect=failure):studies=tuning.main(args)
    assert all(s.trials[0].state==optuna.trial.TrialState.FAIL and s.trials[0].value is None for _,s in studies)
    assert all(s.trials[0].user_attrs['error']['message']=='controlled matcher failure' for _,s in studies)
    assert not (output/'selected.json').exists()


def test_stage2_completed_seed_preserves_exact_baseline_additions(evaluation_fixture,tmp_path):
    root,checkpoint=evaluation_fixture;configure_fixture(root);output=tmp_path/'seeded'
    studies=tuning.main(['tune','--dataset',str(root),'--checkpoint',str(checkpoint),'--output',str(output),
        '--device','cpu','--crop','checkpoint','--pair-source','dataset','--bootstrap','20',
        '--max-negative-pair-fpr','1','--stage1-trials','1','--stage2-trials','2'])
    first=studies[0][1].trials[0];seed=studies[1][1].trials[0]
    assert seed.user_attrs['stage1_seed']==first.number
    assert seed.user_attrs['effective_settings']==first.user_attrs['effective_settings']
    assert seed.user_attrs['effective_settings']['settings']['merge_max_gap_a']==1
    assert seed.user_attrs['effective_settings']['settings']['merge_max_gap_b']==2
