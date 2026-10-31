"""Allocated-node-only, offline cache/baseline/two-stage mask search worker."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import uuid

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def stamp():return datetime.now(timezone.utc).isoformat()


def allocation_required():
    if not os.environ.get('SLURM_JOB_ID') or not (os.environ.get('SLURMD_NODENAME') or os.environ.get('SLURM_JOB_NODELIST')):
        raise RuntimeError('Refusing to run outside a SLURM compute allocation. Submit with scripts/eval/tune_masks.sh; never run the full worker on a login node.')


def durable(path):
    path=Path(path).expanduser().resolve()
    roots=[Path('/tmp'),Path('/private/tmp'),Path('/var/tmp'),Path('/dev/shm')]
    roots += [Path(os.environ[key]).resolve() for key in ('SLURM_TMPDIR','TMPDIR','TMP') if os.environ.get(key)]
    if any(path==root or root in path.parents for root in roots):
        raise ValueError('Studies and caches require durable project storage, not node temporary storage: '+str(path))
    return path


def job_active(owner):
    """Failure to contact the controller is NOT evidence that a writer is dead."""
    if owner.get('uid')!=os.getuid():raise RuntimeError('Run lease belongs to another user')
    cluster=os.environ.get('SLURM_CLUSTER_NAME')
    if cluster and owner.get('cluster') and owner['cluster']!=cluster:
        raise RuntimeError('Cannot verify an old run lease from a different SLURM cluster')
    if owner['job_id']==os.environ.get('SLURM_JOB_ID') and owner['host']==socket.gethostname():
        try:os.kill(owner['pid'],0);return True
        except ProcessLookupError:return False
    response=subprocess.run(['squeue','--noheader','--user',getpass.getuser(),'--states=all','--format=%i'],
        capture_output=True,text=True,timeout=20)
    if response.returncode:raise RuntimeError('Cannot verify previous job status with squeue; preserving its run lease: '+response.stderr.strip())
    return owner['job_id'] in response.stdout.split()


@contextmanager
def admission_lock(output):
    """Brief NFS-safe mutex; a killed admission can be reclaimed after 60s."""
    from optuna.storages.journal import JournalFileSymlinkLock
    if 'grace_period' not in inspect.signature(JournalFileSymlinkLock).parameters:
        raise RuntimeError('Crash-recoverable journal locks are unavailable; install compatible Optuna 4.x or 5.x in manucripts_align before submitting')
    target=Path(output)/'.batch_admission'
    with target.open('ab'):pass
    guard=JournalFileSymlinkLock(str(target),grace_period=60)
    guard.acquire()
    try:yield
    finally:guard.release()


@contextmanager
def run_lease(output,resume):
    """Atomic symlink lease with controller-verified stale-writer recovery."""
    from scripts.eval.tune_matching import write_json
    output=Path(output).resolve();lock=output/'.mask_batch.lock';recovered=[]
    owner=dict(job_id=os.environ['SLURM_JOB_ID'],host=socket.gethostname(),pid=os.getpid(),uid=os.getuid(),
        cluster=os.environ.get('SLURM_CLUSTER_NAME'),token=uuid.uuid4().hex,started_at=stamp())
    directory=output/'batch_attempts';directory.mkdir(exist_ok=True)
    owner_file=directory/('lease_'+owner['token']+'.json');write_json(owner_file,owner)
    with admission_lock(output):
        if lock.exists() or lock.is_symlink():
            if not lock.is_symlink() or lock.resolve().parent!=directory:
                raise RuntimeError('Run lease is invalid; no lock was removed. Inspect '+str(lock))
            try:old=json.loads(lock.read_text())
            except (OSError,ValueError):raise RuntimeError('Run lease is incomplete; no lock was removed. Inspect '+str(lock)) from None
            if not resume or job_active(old):raise RuntimeError('Another run owns this output (job '+str(old.get('job_id'))+'). Stop it before resuming.')
            lock.unlink();recovered.append(old)
        os.symlink(str(owner_file),str(lock))
    try:
        yield owner,recovered
    finally:
        with admission_lock(output):
            if lock.is_symlink() and lock.resolve()==owner_file:lock.unlink()


def filesystem_probe(folder):
    """Check durable writes, exclusive symlink creation, rename and fsync locally."""
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    path=folder/('.storage_probe_'+uuid.uuid4().hex);link=path.with_suffix('.lock');renamed=path.with_suffix('.renamed')
    try:
        with path.open('xb') as stream:stream.write(b'mask-evaluation-storage-check');stream.flush();os.fsync(stream.fileno())
        os.symlink(str(path),str(link))
        try:os.symlink(str(path),str(link))
        except FileExistsError:pass
        else:raise RuntimeError('Filesystem did not enforce exclusive lock creation')
        path.replace(renamed)
        if renamed.read_bytes()!=b'mask-evaluation-storage-check':raise RuntimeError('Storage probe contents changed')
    finally:
        link.unlink(missing_ok=True);path.unlink(missing_ok=True);renamed.unlink(missing_ok=True)


def preflight(args,require_cuda=True):
    import inspect
    import optuna
    import torch
    from train import load_checkpoint
    from scripts.eval.tune_masks import storage_spec
    from scripts.eval.mask_quantitative import data_policy
    workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG')
    if args.device.startswith('cuda') and workspace not in (':4096:8',':16:8'):
        raise RuntimeError('Deterministic CUDA evaluation requires CUBLAS_WORKSPACE_CONFIG=:4096:8 '
                           'or :16:8 before Python starts; submit through scripts/eval/tune_masks.sh')
    policy=data_policy(args)
    if policy=='reviewed' and not args.split_manifest:
        raise ValueError('Reviewed evaluation requires --split-manifest with verified pairs and explicit GT masks')
    if policy=='folder_masks':
        for folder in ('alignedPairs','Unalignedlines'):
            if not (Path(args.dataset)/folder).is_dir():
                raise FileNotFoundError(f'Folder-mask evaluation requires {args.dataset}/{folder}')
    for path in (Path(args.checkpoint), *([Path(args.split_manifest)] if args.split_manifest else [])):
        if not path.is_file():raise FileNotFoundError(path)
    if not Path(args.dataset).is_dir():raise FileNotFoundError(args.dataset)
    if not storage_spec(args).startswith('journal://'):raise ValueError('The BGU worker requires journal:///durable/project/path; SQLite is not used on shared storage')
    if 'constraints_func' not in inspect.signature(optuna.samplers.TPESampler).parameters:raise RuntimeError('Optuna constrained TPE support is missing')
    from optuna.storages.journal import JournalFileBackend, JournalFileSymlinkLock  # noqa: F401
    if 'grace_period' not in inspect.signature(JournalFileSymlinkLock).parameters:
        raise RuntimeError('Installed Optuna lacks crash-recoverable journal symlink locks')
    if require_cuda and args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the allocated node; refusing a silent CPU fallback')
    # Strict reconstruction requires only the self-contained trained checkpoint.
    # initialize=False in load_checkpoint never fetches ImageNet/model assets.
    model,text,config,saved=load_checkpoint(args.checkpoint,'cpu')
    if not saved.get('split_ids'):raise RuntimeError('Checkpoint lacks the required saved split membership')
    del model,text
    result=dict(python=sys.executable,conda_environment=os.environ.get('CONDA_DEFAULT_ENV'),
        checkpoint=str(Path(args.checkpoint).resolve()),model_assets='self-contained checkpoint, strict state load passed; no downloads',
        cuda_available=torch.cuda.is_available(),visible_gpus=torch.cuda.device_count(),
        packages={name:importlib.metadata.version(name) for name in ('torch','torchvision','numpy','scipy','optuna','pillow')},
        worker_count=1,pruning=False,device=args.device,storage_backend='JournalStorage / JournalFileSymlinkLock',
        data_policy=policy,negative_label_status=('presumed, not manually reviewed' if policy=='folder_masks' else 'verified manifest'),
        cublas_workspace_config=workspace)
    return result


def recover_trials(args,manifest,recovered):
    """Only an exclusive batch owner may retire confirmed interrupted writers."""
    import optuna
    from scripts.eval.tune_masks import open_storage,storage_spec
    from scripts.eval.tune_matching import validate_study_manifest
    storage=open_storage(storage_spec(args));events=[]
    for stage in (1,2):
        try:study=optuna.load_study(study_name=args.study_name+f'-stage{stage}',storage=storage)
        except KeyError:continue
        validate_study_manifest(study,dict(manifest,stage=stage),args.resume)
        for trial in study.trials:
            if trial.state!=optuna.trial.TrialState.RUNNING:continue
            previous=trial.user_attrs.get('batch_owner')
            confirmed=bool(recovered) or bool(previous and not job_active(previous))
            if not args.resume or not confirmed:
                raise RuntimeError('Unverified RUNNING trial '+str(trial.number)+' remains; no live/unknown writer was overridden')
            study.tell(trial.number,state=optuna.trial.TrialState.FAIL)
            event=dict(stage=stage,trial=trial.number,reason='Interrupted batch owner confirmed no longer active',at=stamp())
            events.append(event);print('RECOVERED '+json.dumps(event),flush=True)
        if events:study.set_user_attr('batch_recovery_events',study.user_attrs.get('batch_recovery_events',[])+[e for e in events if e['stage']==stage])
    return events


class BatchInterrupted(KeyboardInterrupt):
    def __init__(self,signum):self.signum=signum;super().__init__('SLURM signal '+str(signum))


def main(argv=None):
    allocation_required()
    for name in ('HF_HUB_OFFLINE','HF_DATASETS_OFFLINE','TRANSFORMERS_OFFLINE','PIP_NO_INDEX'):os.environ[name]='1'
    os.environ['WANDB_MODE']='offline';os.environ['PYTHONUNBUFFERED']='1'
    from scripts.eval import tune_masks as tuning
    from scripts.eval.mask_quantitative import build_cache,source_identity
    from scripts.eval.tune_matching import write_json
    args=tuning.parser().parse_args(['tune',*(sys.argv[1:] if argv is None else argv)])
    args.output=durable(args.output);args.output.mkdir(parents=True,exist_ok=True)
    args.cache_dir=durable(args.cache_dir or args.output/'cache')
    args.storage=args.storage or 'journal://'+str(args.output/'optuna.journal')
    if not tuning.storage_spec(args).startswith('journal://'):raise ValueError('Batch storage must be an NFS-compatible journal')
    journal=durable(tuning.storage_spec(args)[len('journal://'):])
    if (args.output/'test/selection_used.json').exists():raise ValueError('This selection is frozen for final test; further tuning is prohibited')
    if not args.resume and (args.output/'run_manifest.json').exists():raise ValueError('Output already contains a run; submit with --resume')
    if min(args.stage1_trials,args.stage2_trials,args.startup_trials)<0 or args.bootstrap<1 or args.threads<1:
        raise ValueError('Invalid trial, bootstrap or thread budget')
    if not 0<=args.max_negative_pair_fpr<=1:raise ValueError('FPR requirement must lie in [0,1]')
    args.fail_fast=True
    def interrupted(number,_frame):raise BatchInterrupted(number)
    for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(signum,interrupted)
    with run_lease(args.output,args.resume) as (owner,recovered):
        args.batch_owner=owner
        attempt=args.output/'batch_attempts'/(owner['job_id']+'_'+owner['token']+'.json');attempt.parent.mkdir(exist_ok=True)
        status=dict(state='RUNNING',phase='preflight',owner=owner,started_at=stamp(),
            output=str(args.output),cache=str(args.cache_dir),journal=str(journal),
            stage1_completed_budget=args.stage1_trials,stage2_completed_budget=args.stage2_trials,seed=args.seed)
        status['allocation']={name:os.environ.get(name) for name in ('SLURM_JOB_NAME','SLURM_JOB_PARTITION','SLURM_JOB_ACCOUNT','SLURM_JOB_QOS','SLURM_CPUS_PER_TASK','SLURM_MEM_PER_NODE','SLURM_JOB_GPUS','CUDA_VISIBLE_DEVICES')}
        def save():
            write_json(attempt,status);write_json(args.output/'batch_status.json',status)
        try:
            save();print('BATCH '+json.dumps(status),flush=True)
            filesystem_probe(args.output);filesystem_probe(args.cache_dir);filesystem_probe(journal.parent)
            status['preflight']=preflight(args);status['source_sha256']=source_identity()
            from dataset import file_hash
            status['worker_sha256']=file_hash(__file__);save()
            print('PREFLIGHT '+json.dumps(status['preflight']),flush=True)
            status['phase']='cache_and_baseline';save();print('Building/reusing durable validation cache and exact baseline',flush=True)
            prepared_run=build_cache(args,'val')
            manifest=tuning.stable_manifest(prepared_run[2],args)
            prior=args.output/'run_manifest.json'
            if prior.exists() and json.loads(prior.read_text())!=manifest:raise ValueError('Run provenance/objective/search space changed; use a new output')
            tuning.baseline_result(args,*prepared_run)
            status['recovered_trials']=recover_trials(args,manifest,recovered)
            status['phase']='tuning';save();print('Running both stages: one worker, constrained TPE, no pruning',flush=True)
            studies=tuning.tune(args,prepared_run=prepared_run)
            status['completed_trials']={str(stage):sum(t.state.name=='COMPLETE' for t in study.trials) for stage,study in studies}
            selection=json.loads((args.output/'selection_status.json').read_text())
            if selection['status']!='SELECTED_FEASIBLE':
                status.update(state='NO_FEASIBLE_TRIAL',phase='finished',finished_at=stamp());save()
                print('NO FEASIBLE TRIAL: trade-off tables preserved; FPR constraint unchanged.',file=sys.stderr,flush=True);return 3
            status.update(state='COMPLETE',phase='finished',finished_at=stamp(),selection=selection);save()
            print('COMPLETE: selected.json, validation metrics, trials.csv and durable journal saved. Final test was NOT run.',flush=True)
            return 0
        except BatchInterrupted as exc:
            status.update(state='INTERRUPTED',signal=exc.signum,finished_at=stamp());save()
            print('INTERRUPTED: completed trials and caches preserved. Resubmit the same output with --resume.',file=sys.stderr,flush=True)
            return 128+exc.signum
        except Exception as exc:
            status.update(state='FAILED',error_type=type(exc).__name__,error=str(exc),finished_at=stamp());save()
            print('FAILED: '+str(exc)+'; completed results remain in '+str(args.output),file=sys.stderr,flush=True)
            raise


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr,flush=True);raise SystemExit(1)
