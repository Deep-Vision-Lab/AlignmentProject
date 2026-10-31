"""Offline BGU submission, exclusive journal recovery, and a bounded CPU worker smoke."""
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import optuna
import pytest

from scripts.eval import mask_batch as batch
from scripts.eval import tune_masks as tuning
from evaluation_utils import EvaluationSession
from dataset import file_hash
from test_evaluation_notebook import evaluation_fixture
from test_quantitative_masks import configure_fixture

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def project_output():
    parent=ROOT/'Results/Evaluation/mask_batch_tests';parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=parent) as folder:yield Path(folder)


def executable(path,source):
    path.write_text(source);path.chmod(0o755);return path


def fake_inputs(tmp_path):
    cp=tmp_path/'checkpoint.pt';cp.write_text('submission existence check only')
    data=tmp_path/'data';data.mkdir()
    manifest=tmp_path/'reviewed.json';manifest.write_text('{"schema_version":1,"pairs":[]}')
    return ['--checkpoint',str(cp),'--dataset',str(data),'--split-manifest',str(manifest)]


def test_submission_only_forwards_resources_and_persistent_paths(tmp_path,project_output):
    fake=tmp_path/'bin';fake.mkdir();captured=tmp_path/'arguments'
    executable(fake/'sbatch','#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE"\nprintf "123456;testcluster\\n"\n')
    args=fake_inputs(tmp_path)
    command=['bash','scripts/eval/tune_masks.sh',*args,'--output-dir',str(project_output),
        '--job-name','mask-review','--seed','7','--stage1-trials','11','--stage2-trials','9','--resume',
        '--cpus','4','--mem','20G','--gpus','1','--time','12:00:00','--account','provided_account','--qos','provided_qos']
    result=subprocess.run(command,cwd=ROOT,env=dict(os.environ,PATH=str(fake)+':'+os.environ['PATH'],CAPTURE=str(captured)),capture_output=True,text=True,check=True)
    assert result.stdout.strip()=='123456'
    values=captured.read_text().splitlines()
    for expected in ['--partition=rtx4090','--gpus=rtx_4090:1','--cpus-per-task=4','--mem=20G',
        '--time=12:00:00','--account=provided_account','--qos=provided_qos','--signal=B:USR1@120','--resume','--fail-fast']:
        assert expected in values
    assert values[values.index('--stage1-trials')+1]=='11'
    assert values[values.index('--stage2-trials')+1]=='9'
    assert values[values.index('--storage')+1]=='journal://'+str(project_output/'optuna.journal')
    assert str(ROOT/'scripts/eval/tune_masks.sbatch') in values
    assert '--output='+str(project_output/'logs/%x_%j.out') in values
    assert '--error='+str(project_output/'logs/%x_%j.err') in values


def test_defaults_and_dry_run_do_not_start_a_search(tmp_path,project_output):
    result=subprocess.run(['bash','scripts/eval/tune_masks.sh',*fake_inputs(tmp_path),
        '--output-dir',str(project_output),'--dry-run'],cwd=ROOT,capture_output=True,text=True,check=True)
    assert '--stage1-trials 300' in result.stdout and '--stage2-trials 200' in result.stdout
    assert '--seed 42' in result.stdout and '--startup-trials 40' in result.stdout
    assert '--negative-count all' in result.stdout
    assert '--pairs-per-split 500' in result.stdout
    assert '--gpus=rtx_4090:1' in result.stdout
    assert '--account=' not in result.stdout and '--qos=' not in result.stdout
    assert 'mask_batch.py' not in result.stdout  # sbatch invokes the guarded script.
    assert not (project_output/'optuna.journal').exists()
    cpu=subprocess.run(['bash','scripts/eval/tune_masks.sh',*[
        '--checkpoint',str(tmp_path/'checkpoint.pt'),'--dataset',str(tmp_path/'data'),
        '--split-manifest',str(tmp_path/'reviewed.json')],
        '--output-dir',str(project_output),'--gpus','0','--dry-run'],
        cwd=ROOT,capture_output=True,text=True,check=True)
    assert '--gpus=0' in cpu.stdout and '--device cpu' in cpu.stdout


def test_arabic_folder_submission_does_not_require_a_review_manifest(tmp_path,project_output):
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_text('existence check only')
    dataset=tmp_path/'ArabicData'
    for name in ('alignedPairs','Unalignedlines'):(dataset/name).mkdir(parents=True)
    result=subprocess.run(['bash','scripts/eval/tune_masks.sh','--checkpoint',str(checkpoint),
        '--dataset',str(dataset),'--output-dir',str(project_output),'--negative-count','75','--dry-run'],
        cwd=ROOT,capture_output=True,text=True,check=True)
    assert '--data-policy auto' in result.stdout and '--negative-count 75' in result.stdout
    assert '--split-manifest' not in result.stdout
    assert not (project_output/'optuna.journal').exists()
    all_pairs=subprocess.run(['bash','scripts/eval/tune_masks.sh','--checkpoint',str(checkpoint),
        '--dataset',str(dataset),'--output-dir',str(project_output),'--dry-run'],
        cwd=ROOT,capture_output=True,text=True,check=True)
    assert '--negative-count all' in all_pairs.stdout


def test_both_worker_entrypoints_refuse_login_node():
    env={k:v for k,v in os.environ.items() if not k.startswith('SLURM')}
    for command in [['bash','scripts/eval/tune_masks.sbatch'],[sys.executable,'scripts/eval/mask_batch.py']]:
        result=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
        assert result.returncode and ('Refusing' in result.stderr or 'refusing' in result.stderr)


def test_batch_activates_conda_preserves_gpu_and_forwards_exit(tmp_path):
    fake=tmp_path/'bin';fake.mkdir();conda_root=tmp_path/'conda';profile=conda_root/'etc/profile.d';profile.mkdir(parents=True)
    executable(fake/'conda','#!/usr/bin/env bash\nprintf "%s\\n" "$FAKE_CONDA_BASE"\n')
    (profile/'conda.sh').write_text('conda() { export CONDA_DEFAULT_ENV="$2"; export CUDA_VISIBLE_DEVICES=99; }\n')
    captured=tmp_path/'worker-args';environment=tmp_path/'worker-env'
    executable(fake/'python','#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE"\nprintf "%s\\n" "$CONDA_DEFAULT_ENV" "$CUDA_VISIBLE_DEVICES" "$PYTHONUNBUFFERED" "$HF_HUB_OFFLINE" > "$ENV_CAPTURE"\nexit 7\n')
    env=dict(os.environ,PATH=str(fake)+':'+os.environ['PATH'],FAKE_CONDA_BASE=str(conda_root),
        CAPTURE=str(captured),ENV_CAPTURE=str(environment),CUDA_VISIBLE_DEVICES='2',SLURM_JOB_ID='1234',
        SLURMD_NODENAME='mock-node',SLURM_SUBMIT_DIR=str(ROOT),MASK_CONDA_ENV='manucripts_align')
    result=subprocess.run(['bash','scripts/eval/tune_masks.sbatch','--stage1-trials','2'],cwd=ROOT,env=env,capture_output=True,text=True)
    assert result.returncode==7
    assert captured.read_text().splitlines()==['-u','scripts/eval/mask_batch.py','--stage1-trials','2']
    assert environment.read_text().splitlines()==['manucripts_align','2','1','1']


def test_temporary_paths_and_extra_gpus_are_rejected(tmp_path,project_output,monkeypatch):
    monkeypatch.setenv('TMPDIR',str(tmp_path))
    with pytest.raises(ValueError,match='durable'):batch.durable(tmp_path/'studies')
    redirected=project_output/'redirect';redirected.symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError,match='durable'):batch.durable(redirected/'cache')
    result=subprocess.run(['bash','scripts/eval/tune_masks.sh',*fake_inputs(tmp_path),
        '--output-dir',str(project_output),'--gpus','2','--dry-run'],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode and 'at most one GPU' in result.stderr


def test_live_lease_is_never_stolen(project_output,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','222')
    with batch.run_lease(project_output,False) as (owner,recovered):
        assert not recovered
        with pytest.raises(RuntimeError,match='owns this output'):
            with batch.run_lease(project_output,True):pass
        assert json.loads((project_output/'.mask_batch.lock').read_text())['token']==owner['token']
    assert not (project_output/'.mask_batch.lock').exists()


def test_confirmed_dead_lease_is_preserved_and_recovered(project_output,monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID','222');monkeypatch.setattr(batch,'job_active',lambda owner:False)
    directory=project_output/'batch_attempts';directory.mkdir()
    old=dict(job_id='111',host='old-node',pid=1,uid=os.getuid(),token='old')
    path=directory/'old.json';path.write_text(json.dumps(old));(project_output/'.mask_batch.lock').symlink_to(path)
    with batch.run_lease(project_output,True) as (owner,recovered):assert recovered==[old]
    assert json.loads(path.read_text())==old


def test_journal_orphans_need_verified_dead_owner(project_output):
    args=SimpleNamespace(output=project_output,storage='journal://'+str(project_output/'optuna.journal'),study_name='test',resume=True)
    storage=tuning.open_storage(tuning.storage_spec(args));study=optuna.create_study(study_name='test-stage1',storage=storage)
    manifest=dict(identity='test');study.set_user_attr('run_manifest',dict(manifest,stage=1));orphan=study.ask()
    with pytest.raises(RuntimeError,match='Unverified RUNNING'):batch.recover_trials(args,manifest,[])
    assert study.trials[0].state==optuna.trial.TrialState.RUNNING
    events=batch.recover_trials(args,manifest,[dict(job_id='confirmed-old-job')])
    assert len(events)==1 and study.trials[0].state==optuna.trial.TrialState.FAIL


def test_bounded_journal_worker_resume_and_interrupted_trial_budget(evaluation_fixture,project_output,tmp_path):
    data,checkpoint=evaluation_fixture;configure_fixture(data)
    session=EvaluationSession(checkpoint,data,split='val',device='cpu',pair_source='dataset',construct_negatives=False)
    manifest=project_output/'reviewed.json'
    manifest.write_text(json.dumps(dict(schema_version=1,pairs=[dict(sample_id=p['sample_id'],split='val',
        target=p['target'],verified=True,masks=[a['mask'] for a in p['annotations']]) for p in session.pairs])))
    output=project_output/'run';fake=tmp_path/'bin';fake.mkdir()
    executable(fake/'squeue','#!/usr/bin/env bash\nprintf "%s\\n" "$SLURM_JOB_ID"\n')
    env=dict(os.environ,PATH=str(fake)+':'+os.environ['PATH'],SLURM_JOB_ID='900001',
        SLURMD_NODENAME=socket.gethostname(),CONDA_DEFAULT_ENV='manucripts_align')
    command=[sys.executable,'scripts/eval/mask_batch.py','--dataset',str(data),'--checkpoint',str(checkpoint),
        '--split-manifest',str(manifest),'--output',str(output),'--device','cpu','--crop','checkpoint','--pair-source','dataset',
        '--stage1-trials','2','--stage2-trials','3','--max-negative-pair-fpr','1','--bootstrap','20']
    before=file_hash(checkpoint)
    first=subprocess.run(command,cwd=ROOT,env=env,capture_output=True,text=True)
    assert first.returncode==0,first.stdout+'\n'+first.stderr
    assert all(name in first.stdout for name in ('IoU=','Dice=','precision=','recall=',
        'negative_pair_FPR=','negative_masked_area_fraction=','feasible=','elapsed_seconds='))
    storage=tuning.open_storage('journal://'+str(output/'optuna.journal'))
    study=optuna.load_study(study_name='quantitative-mask-iou-stage1',storage=storage)
    original=[(t.number,t.params,t.value) for t in study.trials]
    orphan=study.ask();orphan.set_user_attr('batch_owner',dict(job_id='899999',host='old-node',pid=1,uid=os.getuid()))
    resumed=command.copy();resumed[resumed.index('--stage1-trials')+1]='3';resumed[resumed.index('--stage2-trials')+1]='4';resumed+=['--resume']
    env['SLURM_JOB_ID']='900002'
    second=subprocess.run(resumed,cwd=ROOT,env=env,capture_output=True,text=True)
    assert second.returncode==0,second.stdout+'\n'+second.stderr
    study=optuna.load_study(study_name='quantitative-mask-iou-stage1',storage=tuning.open_storage('journal://'+str(output/'optuna.journal')))
    assert [(t.number,t.params,t.value) for t in study.trials[:2]]==original
    assert sum(t.state==optuna.trial.TrialState.COMPLETE for t in study.trials)==3
    assert study.trials[orphan.number].state==optuna.trial.TrialState.FAIL
    assert len(study.trials)==4  # Interrupted work does not consume a completed-trial budget.
    third=subprocess.run(resumed,cwd=ROOT,env=env,capture_output=True,text=True)
    assert third.returncode==0,third.stdout+'\n'+third.stderr
    status=json.loads((output/'batch_status.json').read_text())
    assert status['state']=='COMPLETE' and status['completed_trials']=={'1':3,'2':4}
    assert not (output/'test').exists() and not list(output.glob('*.sqlite3'))
    assert list((output/'cache').glob('*.npz')) and (output/'selected.json').exists()
    assert file_hash(checkpoint)==before
