"""The pooled report must use both saved splits without changing selection."""
import json

import pytest

from scripts.eval.combined_mask_report import report
from scripts.eval.mask_quantitative import identity
from scripts.eval.tune_matching import write_csv, write_json


def saved_pair(sample_id, category, iou, false_positive=False):
    return dict(sample_id=sample_id, category=category, bootstrap_group=sample_id,
                data_policy='folder_masks', source_folder='alignedPairs' if category=='positive' else 'Unalignedlines',
                presumed_negative=category=='negative', negative_basis='unmatched_comparison' if category=='negative' else '',
                positive_iou=iou, positive_dice=iou, precision=iou, recall=iou,
                negative_false_positive=false_positive, negative_masked_area_fraction=.1 if false_positive else 0.,
                candidate_limit_hit=False, region_eligible=True, region_tp=0, region_fp=0, region_fn=0)


def fixture_output(root):
    output = root/'run'; output.mkdir()
    manifest = dict(checkpoint_sha256='checkpoint', bootstrap_resamples=20, seed=42)
    selected = dict(status='SELECTED_FEASIBLE', stage=1, trial=3, run_manifest=manifest,
                    metrics=dict(positive_iou=1.,negative_pair_fpr=0.))
    selected['selection_sha256'] = identity(selected)
    (output/'test').mkdir()
    write_json(output/'test/methods.json', [{'method':'Current baseline'}, {'method':'Optuna selected'}])
    for split, pairs in [('val', [('vp','positive'),('vn','negative')]),
                         ('test', [('tp','positive'),('tn','negative')])]:
        contract = dict(records=[dict(sample_id=sample,category=category,
                                      images=[dict(sha256=sample+side) for side in 'AB'])
                                 for sample,category in pairs])
        write_json(output/f'{split}_split_manifest.json', contract)
        if split == 'val': manifest['split_manifest_sha256'] = identity(contract)
        for directory in ([output/'baseline',output/'trials/stage1_00003'] if split=='val' else
                          [output/'test/method_0',output/'test/method_1']):
            directory.mkdir(parents=True)
            rows = [saved_pair(sample,category,1. if sample.endswith('p') else 0.,
                               false_positive=sample=='tn') for sample,category in pairs]
            write_csv(directory/'per_pair.csv', rows)
            write_csv(directory/'per_line.csv',
                      [dict(sample_id=sample,side=side,category=category) for sample,category in pairs for side in 'AB'])
    selected['selection_sha256'] = identity({k:v for k,v in selected.items() if k!='selection_sha256'})
    write_json(output/'run_manifest.json', manifest)
    write_json(output/'selected.json', selected)
    write_json(output/'test/selection_used.json', selected)
    return output


def test_combines_both_positive_and_negative_splits(tmp_path):
    output = fixture_output(tmp_path)
    summaries = report(output)
    for summary in summaries:
        assert summary['validation']['positive_pairs']==summary['validation']['negative_pairs']==1
        assert summary['test']['positive_pairs']==summary['test']['negative_pairs']==1
        pooled = summary['combined']
        assert pooled['positive_pairs']==pooled['negative_pairs']==2
        assert pooled['positive_iou']==1.
        assert pooled['negative_false_positive_pairs']==1
        assert pooled['negative_pair_fpr']==.5
        assert pooled['presumed_negative_pairs']==2
    lines=(output/'combined_validation_test/per_line.csv').read_text()
    assert 'val' in lines and 'test' in lines


def test_refuses_unfrozen_or_overlapping_test(tmp_path):
    output = fixture_output(tmp_path)
    (output/'test/selection_used.json').unlink()
    with pytest.raises(ValueError,match='frozen'): report(output)
    write_json(output/'test/selection_used.json',json.loads((output/'selected.json').read_text()))
    test_manifest=output/'test_split_manifest.json'
    contract=json.loads(test_manifest.read_text())
    contract['records'][0]['sample_id']='vp'
    write_json(test_manifest,contract)
    with pytest.raises(ValueError,match='overlap'): report(output)
