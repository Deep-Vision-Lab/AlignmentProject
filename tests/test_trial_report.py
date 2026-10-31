"""Parameter/metric report data retain failed trials and selection identity."""
import json

import optuna

from scripts.eval.trial_report import read_baseline, report_tables, trial_records


def test_trial_tables_keep_actual_parameters_metrics_and_selected_run():
    study=optuna.create_study(direction='maximize')
    for value,fpr in ((.99,.30),(.80,.04)):
        settings=dict(gap_open=.3,min_mean_reward=None,enable_region_merging=False)
        study.add_trial(optuna.trial.create_trial(value=value,user_attrs=dict(
            requested_settings=settings,effective_settings=dict(settings=settings),
            metrics=dict(positive_iou=value,positive_dice=.9,precision=.85,recall=.84,
                         negative_pair_fpr=fpr,negative_masked_area_fraction=.01,
                         positive_pairs=53,negative_pairs=447))))
    study.add_trial(optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL,
        user_attrs=dict(error=dict(type='RuntimeError',message='controlled test failure'))))
    records=trial_records([(1,study)],.05,dict(stage=1,trial=1))
    tables=report_tables(records)
    assert [row['feasible'] for row in tables['trials']]==[False,True,False]
    assert [row['selected'] for row in tables['trials']]==[False,True,False]
    assert tables['trials'][1]['parameter.gap_open']==.3
    assert tables['trials'][1]['parameter.enable_region_merging'] is False
    assert tables['trials'][2]['positive_iou'] is None
    assert tables['trials'][2]['error_message']=='controlled test failure'
    best=[row for row in tables['parameters'] if row['selected']]
    assert len(best)==3 and all(row['positive_iou']==.8 for row in best)
    assert next(row for row in best if row['parameter']=='min_mean_reward')['effective_value'] is None


def test_saved_baseline_is_not_invented_as_an_optuna_trial(tmp_path):
    assert read_baseline(tmp_path) is None
    folder=tmp_path/'baseline';folder.mkdir()
    (folder/'aggregate.json').write_text(json.dumps(dict(positive_iou=.714,negative_pair_fpr=.698)))
    (folder/'effective_settings.json').write_text(json.dumps(dict(settings=dict(gap_open=.3))))
    baseline=read_baseline(tmp_path)
    tables=report_tables([], [baseline])
    assert tables['trials']==[] and tables['parameters']==[]
    assert tables['baselines'][0]['positive_iou']==.714
    assert tables['baselines'][0]['parameter.gap_open']==.3
