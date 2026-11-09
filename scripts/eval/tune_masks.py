"""Validation-only constrained mask tuning; final test and paper export are separate commands."""
from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import pprint
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))

import optuna
from optuna.trial import TrialState
from scripts.eval.mask_quantitative import (METRICS, VERSION, build_cache, effective_settings,
    evaluate, identity, notebook_baseline)
from scripts.eval.tune_matching import write_json, write_csv, validate_study_manifest

SEARCH_VERSION = 'two-stage-letter-evidence-3-disjoint-passages'
OBJECTIVE = 'positive pair-macro foreground IoU; negative pair FPR <= predeclared limit'
SPACE = dict(normal_min_distinct=[4,10],normal_min_matched_pairs=[4,12],allow_short_regions=[False,True],
    max_gap=[0,6],gap_open=[.05,.60],gap_extend=['.01','gap_open'],repeat_penalty=[0.,.02,.05,.10,.20],
    max_consecutive_repeats=[1,4],anchor_top_k=[1,4],min_mutual_anchors_long=[1,9],
    min_mutual_anchor_fraction_long=[.05,.10,.20,.35,.50,.70,.90],min_path_density=[.60,.70,.80,.85,.90,.95,1.],
    min_mean_reward=[None,0.,.1,.25,.5,1.],
    short_min_distinct=[3,'min(5,normal_min_distinct)'],short_min_matched_pairs=[4,7],
    short_cosine=[.70,.90,.02],short_min_mean_reward=[.10,.25,.50,1.],short_min_path_density=[.80,.90,.95,1.],
    short_max_internal_gap=[0,'min(2,max_gap)'],short_anchor_count=[1,4],short_anchor_fraction=[.40,.60,.80,.95,1.],
    enable_path_extension=[False,True],extension_min_reward=[0.,.05,.10,.25,.50],
    extension_max_weak_run=[0,3],extension_max_internal_gap=[0,'min(4,max_gap)'],
    enable_region_merging=[False,True],merge_max_gap=[0,'min(4,max_gap)'])
FIXED = dict(alignment_mode='local_repeat_dtw',candidate_similarity_mode='letter_evidence',
    similarity_mode='letter_evidence',use_path_anchor_check=True,threshold=.5,max_candidates=512)


def suggest(trial, baseline, stage):
    settings = dict(baseline,**FIXED)
    def integer(name,low=None,high=None):
        a,b = SPACE[name] if low is None else (low,high)
        return trial.suggest_int(name,a,b)
    def choice(name): return trial.suggest_categorical(name,SPACE[name])
    n = integer('normal_min_distinct')
    settings.update(normal_min_distinct_a=n,normal_min_distinct_b=n,
        normal_min_matched_pairs=integer('normal_min_matched_pairs'),allow_short_regions=choice('allow_short_regions'),
        max_gap=integer('max_gap'),gap_open=trial.suggest_float('gap_open',.05,.60))
    settings['gap_extend'] = trial.suggest_float('gap_extend',.01,settings['gap_open'])
    for name in ('repeat_penalty','min_mutual_anchor_fraction_long','min_path_density','min_mean_reward'): settings[name]=choice(name)
    for name in ('max_consecutive_repeats','anchor_top_k','min_mutual_anchors_long'): settings[name]=integer(name)
    if stage == 2:
        if settings['allow_short_regions']:
            k = integer('short_min_distinct',3,min(5,n))
            settings.update(short_min_distinct_a=k,short_min_distinct_b=k,
                short_min_matched_pairs=integer('short_min_matched_pairs'))
            cosine = trial.suggest_float('short_cosine',.70,.90,step=.02)
            settings.update(short_min_mean_cosine=cosine,short_min_median_cosine=cosine,
                short_min_mean_reward=choice('short_min_mean_reward'),short_min_path_density=choice('short_min_path_density'),
                short_max_internal_gap=integer('short_max_internal_gap',0,min(2,settings['max_gap'])))
            count = integer('short_anchor_count'); fraction = choice('short_anchor_fraction')
            settings.update(short_min_mutual_anchors=count,min_mutual_anchors_short=count,
                short_min_mutual_anchor_fraction=fraction,min_mutual_anchor_fraction_short=fraction)
        settings['enable_path_extension'] = choice('enable_path_extension')
        if settings['enable_path_extension']:
            settings.update(extension_min_reward=choice('extension_min_reward'),extension_max_weak_run=integer('extension_max_weak_run'),
                extension_max_internal_gap=integer('extension_max_internal_gap',0,min(4,settings['max_gap'])))
        settings['enable_region_merging'] = choice('enable_region_merging')
        if settings['enable_region_merging']:
            gap = integer('merge_max_gap',0,min(4,settings['max_gap']))
            settings.update(merge_max_gap_a=gap,merge_max_gap_b=gap)
    return settings


def baseline_params(settings):
    """Stage 1's baseline is enqueued only if exactly representable."""
    if settings['normal_min_distinct_a'] != settings['normal_min_distinct_b']: return None
    keys = ('normal_min_matched_pairs','allow_short_regions','max_gap','gap_open','gap_extend','repeat_penalty',
            'max_consecutive_repeats','anchor_top_k','min_mutual_anchors_long','min_mutual_anchor_fraction_long',
            'min_path_density','min_mean_reward')
    values = {k:settings[k] for k in keys}; values['normal_min_distinct']=settings['normal_min_distinct_a']
    for key in ('repeat_penalty','min_mutual_anchor_fraction_long','min_path_density','min_mean_reward'):
        if values[key] not in SPACE[key]: return None
    for key in ('normal_min_distinct','normal_min_matched_pairs','max_gap','max_consecutive_repeats','anchor_top_k','min_mutual_anchors_long'):
        if not SPACE[key][0]<=values[key]<=SPACE[key][1]: return None
    if not .05<=values['gap_open']<=.6 or not .01<=values['gap_extend']<=values['gap_open']: return None
    return values


def feasible_trials(trials, limit):
    return [t for t in trials if t.state==TrialState.COMPLETE and t.value is not None and math.isfinite(t.value)
        and t.user_attrs.get('metrics',{}).get('negative_pairs',0)>0
        and t.user_attrs['metrics']['negative_pair_fpr']<=limit]


def choose_trial(trials, limit):
    feasible = feasible_trials(trials,limit)
    return min(feasible,key=lambda t:(-t.value,t.user_attrs['metrics']['negative_masked_area_fraction'],t.number)) if feasible else None


def constraints(trial):
    return (trial.user_attrs['metrics']['negative_pair_fpr']-trial.user_attrs['max_negative_pair_fpr'],)


def storage_spec(args):
    spec=args.storage or 'sqlite:///'+str((Path(args.output)/'study.sqlite3').resolve())
    if spec.startswith('journal://'):
        return 'journal://'+str(Path(spec[len('journal://'):]).expanduser().resolve())
    return spec


def open_storage(spec):
    """Journal locking is suitable for NFS; SQLite remains a local CLI option."""
    if spec.startswith('journal://'):
        from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock
        path=Path(spec[len('journal://'):]);path.parent.mkdir(parents=True,exist_ok=True)
        return optuna.storages.JournalStorage(JournalFileBackend(str(path),lock_obj=JournalFileSymlinkLock(str(path))))
    return optuna.storages.RDBStorage(spec,heartbeat_interval=30,grace_period=90,
        engine_kwargs={'connect_args':{'timeout':60}} if spec.startswith('sqlite') else {})


def log_trial(stage,trial,metrics,elapsed,seed_from=None):
    feasible=metrics['negative_pair_fpr']<=trial.user_attrs['max_negative_pair_fpr']
    print(f'Stage {stage} trial {trial.number}: IoU={metrics["positive_iou"]:.6f} '
        f'Dice={metrics["positive_dice"]:.6f} precision={metrics["precision"]:.6f} '
        f'recall={metrics["recall"]:.6f} negative_pair_FPR={metrics["negative_pair_fpr"]:.6f} '
        f'negative_masked_area_fraction={metrics["negative_masked_area_fraction"]:.6f} '
        f'feasible={feasible} elapsed_seconds={elapsed:.3f}'+
        (f' reused_stage1_trial={seed_from}' if seed_from is not None else ''),flush=True)


def stable_manifest(provenance, args):
    result = {k:v for k,v in provenance.items() if k!='cache_seconds'}
    result.update(search_version=SEARCH_VERSION,objective=OBJECTIVE,space=SPACE,fixed=FIXED,
        max_negative_pair_fpr=args.max_negative_pair_fpr,startup_trials=args.startup_trials,
        study_name=args.study_name,storage=storage_spec(args),
        sampler='constrained TPE',pruner='NopPruner',split_manifest_input=str(Path(args.split_manifest).resolve()) if args.split_manifest else None)
    return json.loads(json.dumps(result))


def export_studies(studies, output, limit, manifest):
    records = []; candidates = []
    for stage,study in studies:
        for trial in study.trials:
            records.append(dict(stage=stage,trial=trial.number,state=trial.state.name,objective_iou=trial.value,
                feasible=trial in feasible_trials([trial],limit),seconds=trial.duration.total_seconds() if trial.duration else None,
                parameters=json.dumps(trial.params,sort_keys=True),user_attributes=json.dumps(trial.user_attrs,sort_keys=True),
                requested_settings=json.dumps(trial.user_attrs.get('requested_settings'),sort_keys=True),
                effective_settings=json.dumps(trial.user_attrs.get('effective_settings'),sort_keys=True),
                **{k:v for k,v in trial.user_attrs.get('metrics',{}).items() if not isinstance(v,dict)}))
        trial = choose_trial(study.trials,limit)
        if trial is not None: candidates.append((stage,trial))
    write_csv(output/'trials.csv',records)
    frozen=output/'test/selection_used.json'
    if frozen.exists():return json.loads(frozen.read_text())
    if not candidates:
        old=[output/name for name in ('selected.json','best_effective_settings.json','notebook_settings.py','range_diagnostics.json') if (output/name).exists()]
        if old:
            archive=output/'previous_selections'/uuid.uuid4().hex;archive.mkdir(parents=True)
            for path in old:path.replace(archive/path.name)
        write_json(output/'selection_status.json',dict(status='NO_FEASIBLE_TRIAL',limit=limit,
            message='No completed validation trial met the predeclared FPR constraint. No selected configuration exists.'))
        return None
    stage,trial = min(candidates,key=lambda x:(-x[1].value,x[1].user_attrs['metrics']['negative_masked_area_fraction'],x[0],x[1].number))
    selection = dict(status='SELECTED_FEASIBLE',stage=stage,trial=trial.number,metrics=trial.user_attrs['metrics'],
        requested_settings=trial.user_attrs['requested_settings'],effective_settings=trial.user_attrs['effective_settings'],
        max_negative_pair_fpr=limit,run_manifest=manifest)
    selection['selection_sha256'] = identity(selection)
    write_json(output/'selected.json',selection); write_json(output/'selection_status.json',dict(status=selection['status'],stage=stage,trial=trial.number))
    write_json(output/'best_effective_settings.json',selection['effective_settings'])
    settings = selection['effective_settings']['settings']
    block = ('# Paste after the notebook matching-settings cell; rerun sample cells.\n'
        'from evaluate import resolved_match_settings\nCOMPARISON_MAX_PAIRS = 0\n'
        'SETTINGS = '+pprint.pformat(settings,sort_dicts=True)+'\n'
        'session.settings = dict(SETTINGS)\n'
        'def current_match_settings(threshold=None):\n'
        '    return resolved_match_settings(dict(SETTINGS, **({} if threshold is None else {"threshold": float(threshold)})))\n')
    (output/'notebook_settings.py').write_text(block)
    bounds = {name:dict(low=SPACE[name][0],high=SPACE[name][-1],selected=trial.params.get(name))
              for name in ('gap_open','min_mean_reward','extension_min_reward','short_min_mean_reward')}
    write_json(output/'range_diagnostics.json',dict(bounds=bounds,
        note='A selected boundary is a reason to inspect validation reward distributions, not proof the optimum is outside this range. No range is changed automatically; test results never change this search.'))
    return selection


def baseline_result(args, session, prepared, provenance):
    output = Path(args.output)/'baseline'; marker = output/'provenance.json'
    stable = {k:v for k,v in provenance.items() if k!='cache_seconds'}
    stable = json.loads(json.dumps(stable))
    if marker.exists():
        if json.loads(marker.read_text()) != stable: raise ValueError('Baseline provenance changed; use a new output directory')
        return json.loads((output/'aggregate.json').read_text())
    values, settings = notebook_baseline(args.notebook)
    result,_ = evaluate(session,prepared,args.representation,settings,output,True,args.bootstrap,args.seed)
    write_json(marker,stable); write_json(output/'notebook_requested_variables.json',values)
    return result


def tune(args, prepared_run=None):
    output = Path(args.output)
    if (output/'test/selection_used.json').exists(): raise ValueError('Selection was frozen for final test. Further tuning in this output is prohibited.')
    session,prepared,provenance = prepared_run if prepared_run is not None else build_cache(args,'val')
    manifest = stable_manifest(provenance,args)
    if (output/'run_manifest.json').exists() and json.loads((output/'run_manifest.json').read_text()) != manifest:
        raise ValueError('Run provenance/objective/search space changed; use a new output directory')
    baseline_result(args,session,prepared,provenance)
    baseline = dict(provenance['baseline_settings'])
    if any(baseline[k]!=v for k,v in FIXED.items()): raise ValueError('Notebook baseline no longer matches declared fixed settings; inspect before tuning')
    if 'constraints_func' not in inspect.signature(optuna.samplers.TPESampler).parameters:
        raise ValueError('Installed Optuna lacks constrained TPE; install optuna>=4,<5')
    storage = open_storage(storage_spec(args))
    write_json(output/'run_manifest.json',manifest)
    studies = []
    for stage,total in ((1,args.stage1_trials),(2,args.stage2_trials)):
        sampler = optuna.samplers.TPESampler(seed=args.seed,n_startup_trials=args.startup_trials,constraints_func=constraints)
        study = optuna.create_study(study_name=args.study_name+f'-stage{stage}',storage=storage,direction='maximize',
            load_if_exists=args.resume,sampler=sampler,pruner=optuna.pruners.NopPruner())
        validate_study_manifest(study,dict(manifest,stage=stage),args.resume)
        if isinstance(storage,optuna.storages.RDBStorage):optuna.storages.fail_stale_trials(study)
        if any(t.state==TrialState.RUNNING for t in study.trials):
            raise ValueError('This study has a live or recently interrupted trial. Wait 90 seconds for stale-heartbeat recovery before resuming.')
        if not study.trials and total:
            if stage==1:
                params = baseline_params(baseline)
                if params is not None: study.enqueue_trial(params,user_attrs={'baseline_enqueued':True})
            else:
                seeds = sorted(feasible_trials(studies[0][1].trials,args.max_negative_pair_fpr),
                    key=lambda t:(-t.value,t.user_attrs['metrics']['negative_masked_area_fraction']))[:min(5,total)]
                for seed in seeds:
                    attrs = dict(seed.user_attrs,stage1_seed=seed.number)
                    # Completed seeds preserve EXACT baseline additions, including
                    # asymmetric merge gaps, without inventing coupled suggestions.
                    study.add_trial(optuna.trial.create_trial(params=seed.params,distributions=seed.distributions,
                        value=seed.value,user_attrs=attrs,system_attrs={'constraints':constraints(seed)}))
                    log_trial(stage,study.trials[-1],attrs['metrics'],0.,seed.number)
        studies.append((stage,study))
        def objective(trial):
            started = time.monotonic()
            trial.set_user_attr('max_negative_pair_fpr',args.max_negative_pair_fpr)
            if getattr(args,'batch_owner',None):trial.set_user_attr('batch_owner',args.batch_owner)
            try:
                requested = suggest(trial,baseline,stage)
                trial.set_user_attr('requested_settings',requested)
                trial.set_user_attr('effective_settings',effective_settings(requested))
                trial.set_user_attr('result_directory',str((output/'trials'/f'stage{stage}_{trial.number:05d}').resolve()))
                metrics,_ = evaluate(session,prepared,args.representation,requested,
                    output/'trials'/f'stage{stage}_{trial.number:05d}')
                trial.set_user_attr('metrics',metrics)
                trial.set_user_attr('evaluation_seconds',time.monotonic()-started)
                log_trial(stage,trial,metrics,time.monotonic()-started)
                return metrics['positive_iou']
            except Exception as exc:
                error = dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
                trial.set_user_attr('error',error)
                write_json(output/f'failed_stage{stage}_{trial.number:05d}.json',error)
                raise
        def callback(_study,_trial): export_studies(studies,output,args.max_negative_pair_fpr,manifest)
        done = sum(t.state==TrialState.COMPLETE for t in study.trials)
        if done>total:raise ValueError(f'Stage {stage} already has {done} completed trials, exceeding requested total {total}')
        try:
            study.optimize(objective,n_trials=max(0,total-done),timeout=args.timeout,n_jobs=1,
                catch=() if args.fail_fast else (Exception,),callbacks=[callback])
        finally: export_studies(studies,output,args.max_negative_pair_fpr,manifest)
    selection = export_studies(studies,output,args.max_negative_pair_fpr,manifest)
    print('Selected feasible validation configuration: '+str(output/'selected.json') if selection else 'NO FEASIBLE TRIAL. Trade-offs saved; the FPR requirement was not relaxed.',flush=True)
    return studies


def final_test(args):
    output = Path(args.output); selected_path = output/'selected.json'
    if not selected_path.exists(): raise ValueError('No validation-selected feasible configuration; final test is prohibited')
    selection = json.loads(selected_path.read_text()); digest = selection.pop('selection_sha256')
    if identity(selection) != digest: raise ValueError('Selected configuration hash is invalid')
    selection['selection_sha256'] = digest
    for stage in (1,2):
        study=optuna.load_study(study_name=selection['run_manifest']['study_name']+f'-stage{stage}',storage=open_storage(selection['run_manifest']['storage']))
        if any(trial.state==TrialState.RUNNING for trial in study.trials):raise ValueError('Finish or stop running tuning trials before final test')
    test = output/'test'; test.mkdir(exist_ok=True)
    marker = test/'selection_used.json'
    if marker.exists() and json.loads(marker.read_text()) != selection: raise ValueError('Test selection already frozen to another configuration')
    # Freeze before any test features/GT are accessed. No tuning can resume here.
    write_json(marker,selection)
    session,prepared,provenance = build_cache(args,'test')
    for key in ('checkpoint_sha256','dataset','config','representation','seed','source_sha256','split_audit_sha256','device','mask_source','resize_gt','bootstrap_resamples'):
        if key in selection['run_manifest'] and provenance.get(key) != selection['run_manifest'][key]:
            raise ValueError('Final-test provenance differs from validation: '+key)
    base = provenance['baseline_settings']; chosen = selection['effective_settings']['settings']
    methods = [('Current baseline',base,None),('Optuna selected',chosen,None),
        ('Selected without extension',dict(chosen,enable_path_extension=False),'Optuna selected' if not chosen['enable_path_extension'] else None),
        ('Selected without merging',dict(chosen,enable_region_merging=False),'Optuna selected' if not chosen['enable_region_merging'] else None)]
    reports = []
    for i,(name,settings,identical) in enumerate(methods):
        folder = test/f'method_{i}'
        result,_ = evaluate(session,prepared,args.representation,settings,folder,True,args.bootstrap,args.seed)
        reports.append(dict(method=name,folder=str(folder.resolve()),identical_to=identical,ablation_without_retuning=i>=2,metrics=result))
    write_json(test/'methods.json',reports)
    export_paper(output)


def export_paper(output):
    path = output/'test/methods.json'
    if not path.exists(): raise ValueError('Final-test results are absent; no paper table was produced')
    reports = json.loads(path.read_text()); rows = []; regions = []
    presumed = any(report['metrics'].get('presumed_negative_pairs',0) for report in reports)
    labels = ['Positive IoU','Positive Dice','Precision','Recall',
              'Presumed-negative pair FPR' if presumed else 'Negative-pair FPR','Negative masked-area fraction']
    latex = ['\\begin{tabular}{lrrrrrr}','Method & '+' & '.join(labels)+' \\\\','\\hline']
    for report in reports:
        m = report['metrics']; row = dict(Method=report['method'],positive_pairs=m['positive_pairs'],negative_pairs=m['negative_pairs'],
            data_policy=m.get('data_policy','reviewed'),presumed_negative_pairs=m.get('presumed_negative_pairs',0),
            negative_pair_fpr_interpretation=m.get('negative_pair_fpr_interpretation'),
            identical_to=report['identical_to'],ablation_without_retuning=report['ablation_without_retuning'])
        cells = []
        for key,label in zip(METRICS,labels):
            ci = m['confidence_intervals']['estimates'][key]
            row.update({label:m[key],label+' CI low':ci['low'],label+' CI high':ci['high']})
            cells.append(f'{m[key]:.4f} ['+(f'{ci["low"]:.4f}, {ci["high"]:.4f}' if ci['low'] is not None else 'CI unavailable')+']')
        row['negative_false_positive_pairs'] = m['negative_false_positive_pairs']
        row['negative_pair_denominator'] = m['negative_pair_denominator']
        binomial=m['confidence_intervals']['estimates']['negative_fpr_binomial']
        row.update(negative_fpr_binomial_low=binomial['low'],negative_fpr_binomial_high=binomial['high'],
                   independent_negative_pairs=binomial['available'])
        rows.append(row); regions.append(dict(Method=report['method'],**m['region_f1']))
        name = report['method']+(' (identical)' if report['identical_to'] else '')
        latex.append(name+' & '+' & '.join(cells)+' \\\\')
    latex.append('\\end{tabular}')
    write_csv(output/'paper_table.csv',rows); (output/'paper_table.tex').write_text('\n'.join(latex)+'\n')
    write_csv(output/'supplementary_region_f1.csv',regions)
    write_json(output/'aggregate_results.json',reports)
    write_csv(output/'aggregate_results.csv',rows)


def negative_pair_count(value):
    if value == 'all':
        return value
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use all or a nonnegative integer') from exc
    if count < 0:
        raise argparse.ArgumentTypeError('Use all or a nonnegative integer')
    return count


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('prepare','tune','test','export'))
    p.add_argument('--checkpoint',default='Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt')
    p.add_argument('--dataset',default='DataSet/ArabicData')
    p.add_argument('--split-manifest',help='Reviewed JSON pair list, schema in docs/quantitative_mask_evaluation.md')
    p.add_argument('--data-policy',choices=('auto','folder_masks','reviewed'),default='auto',
        help='auto uses alignedPairs/Unalignedlines folder masks for ArabicData, or reviewed labels when a manifest is supplied')
    p.add_argument('--negative-count',type=negative_pair_count,default='all',metavar='all|N',
        help='Unalignedlines comparisons per saved split: all eligible by default, or a seeded cap N (folder policy only)')
    p.add_argument('--pairs-per-split',type=negative_pair_count,default=500,metavar='all|N',
        help='Folder-pair target/upper limit per split (default 500); balanced mode uses equal classes without duplicating scarce positives')
    p.add_argument('--class-balance',choices=('balanced','natural'),default='balanced',
        help='balanced (default): equal positive/negative counts, limited by distinct available pairs; natural retains the original unequal class mix')
    p.add_argument('--notebook',type=Path,default=ROOT/'notebooks/model_evaluation.ipynb')
    p.add_argument('--output',type=Path,default=Path('Results/Evaluation/quantitative_masks'))
    p.add_argument('--cache-dir',type=Path)
    p.add_argument('--device',default='auto')
    p.add_argument('--threads',type=int,default=1)
    p.add_argument('--representation',choices=('local','context','fused'),default='fused')
    p.add_argument('--crop',choices=('content','checkpoint','auto','xml','none'),default='content')
    p.add_argument('--pair-source',choices=('auto','dataset','folders'),default='auto')
    p.add_argument('--mask-source',choices=('manifest','folders'),default='manifest')
    p.add_argument('--resize-gt',action='store_true',help='Explicit uniform GT resize to source image, nearest-neighbor only')
    p.add_argument('--quarantine-leaked-evaluation-images',action='store_true',
                   help='Exclude and report evaluation pairs using duplicated cross-split pixels; saved splits and training are unchanged')
    p.add_argument('--stage1-trials',type=int,default=300)
    p.add_argument('--stage2-trials',type=int,default=200)
    p.add_argument('--startup-trials',type=int,default=40)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--bootstrap',type=int,default=2000)
    p.add_argument('--max-negative-pair-fpr',type=float,default=.05)
    p.add_argument('--storage',help='Optuna RDB URL or journal:///absolute/path; local default is output/study.sqlite3')
    p.add_argument('--study-name',default='quantitative-mask-iou')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--fail-fast',action='store_true',help='Stop on the first failed trial after preserving completed exports')
    p.add_argument('--timeout',type=float)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0<=args.max_negative_pair_fpr<=1 or not math.isfinite(args.max_negative_pair_fpr): raise ValueError('FPR requirement must be in [0,1]')
    if min(args.stage1_trials,args.stage2_trials,args.startup_trials)<0 or args.bootstrap<1 or args.threads<1:
        raise ValueError('Invalid trial/bootstrap/thread count')
    if args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout<=0): raise ValueError('Timeout must be finite and positive')
    args.output = args.output.expanduser().resolve(); args.output.mkdir(parents=True,exist_ok=True)
    if args.command=='export': return export_paper(args.output)
    if args.command=='tune': return tune(args)
    if args.command=='test': return final_test(args)
    session,prepared,provenance = build_cache(args,'val')
    return baseline_result(args,session,prepared,provenance)


if __name__=='__main__': main()
