"""Resume a CUDA-cached mask study on CPU without changing its Optuna identity.

The original CUDA feature arrays and journal stay immutable. This worker creates
CPU cache aliases of those exact arrays, verifies data/code provenance and
replays completed trials before adding any new trial. Run only in SLURM.
"""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import optuna
from dataset import file_hash
from evaluation_utils import EvaluationSession
from scripts.eval import mask_quantitative as q, tune_masks as tuning
from scripts.eval.mask_batch import (BatchInterrupted, allocation_required, preflight,
                                     recover_trials, run_lease)
from scripts.eval.tune_matching import write_json

VERSION = 'cpu-resume-from-frozen-cuda-features-1'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def original_arguments(output, manifest, status, threads):
    values = ['tune', '--checkpoint', manifest['checkpoint'], '--dataset', manifest['dataset'],
              '--data-policy', manifest['data_policy'], '--negative-count', str(manifest['negative_count']),
              '--pairs-per-split', str(manifest['pairs_per_split']), '--class-balance', manifest['class_balance'],
              '--notebook', manifest['notebook_path'], '--output', str(output),
              '--cache-dir', str(output/'cpu_cache'), '--device', 'cpu', '--threads', str(threads),
              '--representation', manifest['representation'], '--crop', manifest['config']['crop'],
              '--pair-source', manifest['pair_source'], '--mask-source', manifest['mask_source'],
              '--stage1-trials', str(status['stage1_completed_budget']),
              '--stage2-trials', str(status['stage2_completed_budget']),
              '--startup-trials', str(manifest['startup_trials']), '--seed', str(manifest['seed']),
              '--bootstrap', str(manifest['bootstrap_resamples']),
              '--max-negative-pair-fpr', str(manifest['max_negative_pair_fpr']),
              '--storage', manifest['storage'], '--study-name', manifest['study_name'], '--resume', '--fail-fast']
    if manifest.get('split_manifest_input'):
        values += ['--split-manifest', manifest['split_manifest_input']]
    if manifest['resize_gt']:
        values += ['--resize-gt']
    return tuning.parser().parse_args(values)


def verify_original(output, manifest, provenance, args):
    if manifest['device'] != 'cuda' or provenance['device'] != 'cuda':
        raise ValueError('This migration requires an existing CUDA feature study')
    if q.source_identity() != manifest['source_sha256']:
        raise ValueError('Original evaluator/notebook source changed; the study cannot be resumed safely')
    if file_hash(manifest['checkpoint']) != manifest['checkpoint_sha256']:
        raise ValueError('Checkpoint changed since CUDA feature extraction')
    if tuning.stable_manifest(provenance, args) != manifest:
        raise ValueError('Original study settings/provenance do not match; no trial was added')
    if read_json(output/'baseline/provenance.json') != {k:v for k,v in provenance.items() if k!='cache_seconds'}:
        raise ValueError('Baseline provenance changed')


def backup_study(output):
    backup = output/'cpu_migration_backup'
    names = ('run_manifest.json','optuna.journal','val_provenance.json',
             'val_split_manifest.json','val_cache_index.json','batch_status.json')
    backup.mkdir(exist_ok=True)
    for name in names:
        source, target = output/name, backup/name
        if not source.is_file():
            raise FileNotFoundError(source)
        if target.exists():
            if file_hash(source) != file_hash(target) and name not in ('optuna.journal','batch_status.json'):
                raise ValueError('Immutable study backup differs: '+name)
        else:
            shutil.copy2(source,target)
    return {name:file_hash(backup/name) for name in names}


def alias_frozen_cache(output, manifest):
    """Create CPU-keyed names for byte-identical GPU NPZ arrays; never infer."""
    contract = read_json(output/'val_split_manifest.json')
    if q.identity(contract) != manifest['split_manifest_sha256']:
        raise ValueError('Validation split manifest differs from the original study')
    index = {row['sample_id']:row for row in read_json(output/'val_cache_index.json')}
    if len(index) != len(contract['records']):
        raise ValueError('Frozen feature index does not cover every validation pair')
    destination = output/'cpu_cache';destination.mkdir(exist_ok=True)
    aliases = []
    for record in contract['records']:
        sample_id = record['sample_id']; old = index[sample_id]
        old_npz = Path(old['npz']).resolve()
        if old_npz.parent != (output/'cache').resolve() or old_npz.stem != old['key']:
            raise ValueError('Frozen feature path/key does not belong to this study: '+sample_id)
        metadata = read_json(old_npz.with_suffix('.json'))
        if metadata['key'] != old['key'] or metadata['pair_id'] != sample_id or file_hash(old_npz) != metadata['npz_sha256']:
            raise ValueError('Frozen feature checksum/identity mismatch: '+sample_id)
        key = q.identity(dict(checkpoint=manifest['checkpoint_sha256'],config=manifest['config'],
            representation=manifest['representation'],
            evidence={k:manifest['baseline_settings'][k] for k in ('acceptance_offset','prior_floor')},
            sources=manifest['source_sha256'],packages=manifest['packages'],pair_id=sample_id,
            images=record['images'],seed=manifest['seed'],device='cpu'))
        target = destination/(key+'.npz')
        if target.exists():
            if file_hash(target) != metadata['npz_sha256']:
                raise ValueError('CPU alias differs from frozen CUDA array: '+sample_id)
        else:
            try: os.link(old_npz,target)
            except OSError: shutil.copy2(old_npz,target)
        cpu_metadata = dict(metadata,key=key)
        meta_path = target.with_suffix('.json')
        if meta_path.exists() and read_json(meta_path) != cpu_metadata:
            raise ValueError('CPU cache metadata differs from frozen CUDA feature: '+sample_id)
        write_json(meta_path,cpu_metadata)
        aliases.append(dict(sample_id=sample_id,cuda_key=old['key'],cpu_key=key,
                            npz_sha256=metadata['npz_sha256']))
    return aliases


def prepare_cpu(output, manifest, provenance, args):
    aliases = alias_frozen_cache(output,manifest)
    preparation = copy.copy(args)
    preparation.output = output/'cpu_preparation'
    preparation.cache_dir = output/'cpu_cache'
    def forbidden(*_args,**_kwargs):
        raise RuntimeError('A frozen feature is missing; refusing CNN/Transformer inference during CPU resume')
    with patch.object(EvaluationSession,'get_line_features',forbidden):
        session, prepared, cpu_provenance = q.build_cache(preparation,'val')
    current = {k:v for k,v in cpu_provenance.items() if k not in ('device','cache_seconds','git_commit')}
    original = {k:v for k,v in provenance.items() if k not in ('device','cache_seconds','git_commit')}
    if current != original or cpu_provenance['device'] != 'cpu':
        raise ValueError('CPU preparation changed data, code, checkpoint, package or settings provenance')
    cpu_index = read_json(preparation.output/'val_cache_index.json')
    if len(cpu_index) != len(aliases) or any(Path(row['npz']).parent != preparation.cache_dir for row in cpu_index):
        raise ValueError('CPU preparation did not use only the frozen feature aliases')
    return session, prepared, aliases


def compare_rows(observed, saved_csv, label):
    with Path(saved_csv).open(newline='',encoding='utf-8') as stream:
        saved = {r['sample_id']:r for r in csv.DictReader(stream)}
    actual = {r['sample_id']:r for r in observed}
    if set(actual) != set(saved) or len(actual) != len(observed):
        raise ValueError('Pair population changed during parity replay: '+label)
    for sample_id,row in actual.items():
        old = saved[sample_id]
        for key in ('category','negative_false_positive','candidate_limit_hit','region_eligible',
                    'region_tp','region_fp','region_fn'):
            if str(row[key]) != old[key]:
                raise ValueError(f'Parity mismatch {label} {sample_id} {key}: {row[key]} != {old[key]}')
        for key in ('positive_iou','positive_dice','precision','recall','negative_masked_area_fraction'):
            if abs(float(row[key])-float(old[key])) > 1e-9:
                raise ValueError(f'Parity mismatch {label} {sample_id} {key}: {row[key]} != {old[key]}')


def replay_parity(output, manifest, session, prepared):
    from optuna.trial import TrialState
    study = optuna.load_study(study_name=manifest['study_name']+'-stage1',
                              storage=tuning.open_storage(manifest['storage']))
    completed = [t for t in study.trials if t.state == TrialState.COMPLETE]
    if not completed:
        raise ValueError('No completed Stage-1 trials to certify')
    choices = [max(completed,key=lambda t:t.number), max(completed,key=lambda t:t.value)]
    feasible = tuning.choose_trial(completed,manifest['max_negative_pair_fpr'])
    if feasible is not None: choices.append(feasible)
    settings = [('baseline',manifest['baseline_settings'],output/'baseline/per_pair.csv')]
    for trial in {t.number:t for t in choices}.values():
        settings.append((f'stage1_{trial.number:05d}',trial.user_attrs['requested_settings'],
                         output/'trials'/f'stage1_{trial.number:05d}'/'per_pair.csv'))
    results = []
    for label,requested,saved in settings:
        metrics, rows = q.evaluate(session,prepared,manifest['representation'],requested)
        compare_rows(rows,saved,label)
        results.append(dict(label=label,positive_iou=metrics['positive_iou'],
                            negative_pair_fpr=metrics['negative_pair_fpr'],pairs=len(rows),
                            saved_csv_sha256=file_hash(saved)))
        print('PARITY PASSED '+label+' IoU='+str(metrics['positive_iou'])+' FPR='+str(metrics['negative_pair_fpr']),flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--verify-only',action='store_true',help='Prepare/check frozen features and replay trials, without adding trials')
    options = parser.parse_args(argv)
    if options.threads < 1:raise ValueError('Threads must be positive')
    allocation_required()
    output = options.output.expanduser().resolve()
    if (output/'test/selection_used.json').exists():
        raise ValueError('Final test already froze this study; further tuning is prohibited')
    manifest = read_json(output/'run_manifest.json')
    provenance = read_json(output/'val_provenance.json')
    previous_status = read_json(output/'batch_status.json')
    args = original_arguments(output,manifest,previous_status,options.threads)
    verify_original(output,manifest,provenance,args)
    with run_lease(output,True) as (owner,recovered):
        status = dict(version=VERSION,state='RUNNING',phase='preflight',owner=owner,
                      started_at=datetime.now(timezone.utc).isoformat(),device='cpu',
                      feature_extraction_device='cuda',pruner='NopPruner',
                      stage1_completed_budget=args.stage1_trials,stage2_completed_budget=args.stage2_trials)
        def save():
            write_json(output/'cpu_resume_status.json',status)
            write_json(output/'batch_status.json',status)
        def interrupted(signum,_frame):raise BatchInterrupted(signum)
        for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(signum,interrupted)
        try:
            backup = backup_study(output)
            status['backup_sha256']=backup
            status['preflight']=preflight(args,require_cuda=False)
            status['phase']='prepare_and_replay';save()
            session,prepared,aliases=prepare_cpu(output,manifest,provenance,args)
            parity=replay_parity(output,manifest,session,prepared)
            migration=dict(version=VERSION,original_manifest_sha256=file_hash(output/'cpu_migration_backup/run_manifest.json'),
                           original_journal_sha256=backup['optuna.journal'],worker_sha256=file_hash(__file__),
                           original_git_commit=manifest.get('git_commit'),
                           cpu_worker_git_commit=subprocess.check_output(
                               ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                           feature_extraction_device='cuda',trial_device='cpu',
                           frozen_features=aliases,parity=parity,checked_at=datetime.now(timezone.utc).isoformat())
            write_json(output/'cpu_migration.json',migration)
            status.update(phase='tuning',migration='cpu_migration.json',parity=parity);save()
            print(f'CPU RESUME: {len(aliases)} frozen pairs; original journal preserved; NopPruner',flush=True)
            if options.verify_only:
                status.update(state='VERIFIED',phase='finished');save();return 0
            recovered_trials = recover_trials(args,manifest,recovered)
            status['recovered_trials']=recovered_trials;save()
            studies=tuning.tune(args,prepared_run=(session,prepared,provenance))
            status['completed_trials']={str(stage):sum(t.state.name=='COMPLETE' for t in study.trials)
                                        for stage,study in studies}
            selected=read_json(output/'selection_status.json')
            status.update(state='COMPLETE' if selected['status']=='SELECTED_FEASIBLE' else 'NO_FEASIBLE_TRIAL',
                          phase='finished',selection=selected,finished_at=datetime.now(timezone.utc).isoformat())
            save()
            print('CPU tuning finished: '+status['state'],flush=True)
            return 0 if status['state']=='COMPLETE' else 3
        except BatchInterrupted as exc:
            status.update(state='INTERRUPTED',signal=exc.signum,finished_at=datetime.now(timezone.utc).isoformat());save()
            print('Interrupted; journal and frozen cache preserved. Resubmit the CPU job.',flush=True)
            return 128+exc.signum
        except Exception as exc:
            status.update(state='FAILED',error_type=type(exc).__name__,error=str(exc),
                          finished_at=datetime.now(timezone.utc).isoformat());save()
            raise


if __name__=='__main__':
    raise SystemExit(main())
