"""Frozen CUDA feature migration must preserve arrays and trial scores."""
import subprocess
from pathlib import Path

import pytest

from dataset import file_hash
from scripts.eval import cpu_resume_masks as cpu
from scripts.eval.mask_quantitative import identity
from scripts.eval.tune_matching import write_csv, write_json

ROOT = Path(__file__).resolve().parents[1]


def frozen_fixture(tmp_path):
    output = tmp_path/'run'; (output/'cache').mkdir(parents=True)
    sample = 'positive:1'
    record = dict(sample_id=sample, images=[dict(image='a.png',sha256='a'),dict(image='b.png',sha256='b')])
    contract = dict(records=[record])
    write_json(output/'val_split_manifest.json',contract)
    old_key = 'a'*64
    old_npz = output/'cache'/(old_key+'.npz')
    old_npz.write_bytes(b'frozen numerical features')
    write_json(old_npz.with_suffix('.json'),dict(key=old_key,pair_id=sample,
        npz_sha256=file_hash(old_npz),geometry=[dict(source_size=[4,2]),dict(source_size=[5,2])]))
    write_json(output/'val_cache_index.json',[dict(sample_id=sample,key=old_key,npz=str(old_npz))])
    manifest = dict(split_manifest_sha256=identity(contract),checkpoint_sha256='checkpoint',
        config={'window_width':2},representation='fused',
        baseline_settings={'acceptance_offset':.1,'prior_floor':.0001},
        source_sha256={'code':'unchanged'},packages={'torch':'test'},seed=42)
    return output,manifest,old_npz


def test_aliases_exact_frozen_bytes_without_reencoding(tmp_path):
    output,manifest,old_npz = frozen_fixture(tmp_path)
    aliases = cpu.alias_frozen_cache(output,manifest)
    assert len(aliases)==1 and aliases[0]['cuda_key'] != aliases[0]['cpu_key']
    target=output/'cpu_cache'/(aliases[0]['cpu_key']+'.npz')
    assert target.read_bytes()==old_npz.read_bytes()
    assert cpu.alias_frozen_cache(output,manifest)==aliases
    old_npz.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'): cpu.alias_frozen_cache(output,manifest)


def test_pair_score_parity_requires_exact_population_and_metrics(tmp_path):
    path=tmp_path/'scores.csv'
    expected=dict(sample_id='a',category='positive',negative_false_positive=False,
        candidate_limit_hit=False,region_eligible=True,region_tp=1,region_fp=0,region_fn=0,
        positive_iou=.5,positive_dice=.6,precision=.7,recall=.8,
        negative_masked_area_fraction=0.)
    write_csv(path,[expected])
    cpu.compare_rows([expected],path,'trial')
    with pytest.raises(ValueError,match='Parity mismatch'):
        cpu.compare_rows([dict(expected,positive_iou=.4)],path,'trial')
    with pytest.raises(ValueError,match='population'):
        cpu.compare_rows([dict(expected,sample_id='other')],path,'trial')


def test_offline_launcher_requests_zero_gpus_and_keeps_original_output(tmp_path):
    output,_,_ = frozen_fixture(tmp_path)
    (output/'run_manifest.json').write_text('{}')
    (output/'optuna.journal').write_text('')
    result=subprocess.run(['bash','scripts/eval/tune_masks_cpu.sh','--output-dir',str(output),
                           '--verify-only','--dry-run'],cwd=ROOT,capture_output=True,text=True,check=True)
    assert '--gpus=0' in result.stdout
    assert '--partition=cpu' in result.stdout
    assert '--verify-only' in result.stdout
    assert str(output) in result.stdout
