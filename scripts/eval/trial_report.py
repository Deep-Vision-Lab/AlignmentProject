"""Prepare auditable baseline/trial tables for an offline results workbook."""
from __future__ import annotations

import json
import math
from pathlib import Path


METRIC_COLUMNS = ('positive_iou', 'positive_dice', 'precision', 'recall',
    'negative_pair_fpr', 'negative_masked_area_fraction', 'positive_pairs',
    'negative_pairs', 'negative_false_positive_pairs', 'presumed_negative_pairs',
    'candidate_limit_hits', 'evaluation_seconds')


def read_baseline(output):
    """Read saved measurements without running inference or inventing trials."""
    directory = Path(output)/'baseline'
    aggregate = directory/'aggregate.json'
    if not aggregate.is_file():
        return None
    def read(name, default=None):
        path = directory/name
        return json.loads(path.read_text()) if path.is_file() else default
    return dict(name=Path(output).name, directory=str(directory.resolve()),
        metrics=read('aggregate.json', {}), requested=read('requested_settings.json', {}),
        effective=read('effective_settings.json', {}).get('settings', {}),
        provenance=read('provenance.json', {}))


def trial_records(studies, limit, selected=None):
    """Keep failed/infeasible runs, with exactly the tuner's feasible-best rule."""
    records = []
    for stage, study in studies:
        for trial in study.trials:
            metrics = trial.user_attrs.get('metrics', {})
            value = trial.value
            feasible = (trial.state.name == 'COMPLETE' and value is not None
                and math.isfinite(value) and metrics.get('negative_pairs', 0) > 0
                and metrics.get('negative_pair_fpr', math.inf) <= limit)
            effective = trial.user_attrs.get('effective_settings', {}) or {}
            records.append(dict(run_id=f'Stage {stage} / trial {trial.number}',
                stage=stage, trial=trial.number, state=trial.state.name,
                objective_iou=value, feasible=feasible,
                selected=bool(selected and selected.get('stage') == stage and selected.get('trial') == trial.number),
                sampled_parameters=dict(trial.params),
                requested=trial.user_attrs.get('requested_settings', {}) or {},
                effective=effective.get('settings', {}),
                metrics=metrics, error=trial.user_attrs.get('error', {}),
                started_at=trial.datetime_start.isoformat() if trial.datetime_start else None,
                finished_at=trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                elapsed_seconds=trial.duration.total_seconds() if trial.duration is not None else None,
                reused_stage1_trial=trial.user_attrs.get('stage1_seed'),
                result_directory=trial.user_attrs.get('result_directory')))
    return records


def report_tables(records, baselines=()):
    """One column per effective parameter, plus a parameter/metric long table."""
    parameter_names = sorted({key for row in records for key in row['effective']}
        | {key for row in baselines for key in row['effective']})
    trials = []
    parameters = []
    for record in records:
        scores = {name:record['metrics'].get(name) for name in METRIC_COLUMNS}
        trials.append(dict(run_id=record['run_id'],stage=record['stage'],trial=record['trial'],
            state=record['state'],selected=record['selected'],feasible=record['feasible'],
            **scores,elapsed_seconds=record['elapsed_seconds'],
            **{'parameter.'+name:record['effective'].get(name) for name in parameter_names},
            reused_stage1_trial=record['reused_stage1_trial'],
            started_at=record['started_at'],finished_at=record['finished_at'],
            error_type=record['error'].get('type'),error_message=record['error'].get('message'),
            result_directory=record['result_directory']))
        for name in sorted(record['effective'].keys() | record['requested'].keys()):
            parameters.append(dict(run_id=record['run_id'],stage=record['stage'],trial=record['trial'],
                parameter=name,requested_value=record['requested'].get(name),
                effective_value=record['effective'].get(name),selected=record['selected'],
                feasible=record['feasible'],state=record['state'],**scores))
    baseline_rows = [dict(run_name=row['name'],**{name:row['metrics'].get(name) for name in METRIC_COLUMNS},
        **{'parameter.'+name:row['effective'].get(name) for name in parameter_names},
        dataset=row['provenance'].get('dataset'),split=row['provenance'].get('split'),
        checkpoint=row['provenance'].get('checkpoint'),source_directory=row['directory']) for row in baselines]
    return dict(trials=trials,parameters=parameters,baselines=baseline_rows,parameter_names=parameter_names)
