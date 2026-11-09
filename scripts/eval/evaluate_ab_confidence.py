"""Evaluate one frozen validation-selected preset on the prior A/B cohort.

Also audit 300 additional saved-test negative combinations, excluding all prior
negative pairs. No settings are selected or changed in this program.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil
import ast
import hashlib
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from dataset import file_hash
from evaluation_utils import EvaluationSession, arabic_data_pairs, predict_cached_pair, _side_key
from scripts.eval.ab_quantitative import evaluate_ab_quantitative, extract_features, score_masks
from scripts.eval.review_reordered import cached_session, pool_signature
from scripts.eval import mask_quantitative as q
from evaluate import resolved_match_settings, MATCH_VERSION
from dataclasses import asdict
from scripts.eval.tune_matching import write_json, write_csv

CHECKPOINT = ROOT / 'Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt'
BASELINE = ROOT / 'Results/Evaluation/ab_300_20261109'


def migrate_accepted_coarse_predictions(session, selected, output):
    """The strong-pair OR clause can only rescue previously empty predictions.

    Reuse nonempty predictions from the interrupted coarse run ONLY when all
    other confidence/generation options, features and checkpoint are identical.
    Empty predictions are always recomputed. No ground truth is consulted.
    """
    settings_path=output/'run_settings.json'
    if not settings_path.exists():return
    old=json.loads(settings_path.read_text());current=resolved_match_settings(selected['settings'])
    previous=resolved_match_settings(old['settings']);exception={'word_strong_pair_max_nll','word_strong_pair_min_cosine'}
    if any(current[k]!=previous[k] for k in current if k not in exception):return
    if previous['word_strong_pair_max_nll'] is not None:return
    if old['checkpoint_sha256']!=session.checkpoint_sha256:return
    core={name:file_hash(ROOT/name) for name in ('model.py','cnn_encoder.py','transformer_encoder.py','dataset.py')}
    feature_signature=q.identity(dict(checkpoint=session.checkpoint_sha256,config=asdict(session.config),sources=core,device=str(session.device)))
    if old['feature_signature']!=feature_signature:return
    def function(source,name):
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.get_source_segment(source,node)
    historical=subprocess.check_output(['git','show','HEAD:word_alignment.py'],cwd=ROOT,text=True)
    before=function(historical,'select_word_candidates');after=function((ROOT/'word_alignment.py').read_text(),'select_word_candidates')
    if before.split('        confirmed=(')[0]!=after.split('        strong_pair=(')[0]:return
    if before.split('            and sum(map(len,identities))',1)[1]!=after.split('            and sum(map(len,identities))',1)[1]:return
    if function(historical,'validate_word_candidate')!=function((ROOT/'word_alignment.py').read_text(),'validate_word_candidate'):return
    generation_record=ROOT/'Results/Evaluation/word_alignment_20261109/word_acceptance_rekey.json'
    if not generation_record.exists():return
    historical_generation=json.loads(generation_record.read_text())['new_generation']
    if pool_signature(CHECKPOINT,current)!=historical_generation:return
    sources={name:file_hash(ROOT/name) for name in ('evaluate.py','word_alignment.py','alignment_candidates.py',
        'evaluation_utils.py','dtw.py','scripts/eval/ab_quantitative.py')}
    assert all(old['sources'][name]==sha for name,sha in sources.items()
               if name not in ('evaluate.py','word_alignment.py','evaluation_utils.py'))
    vocab=file_hash(ROOT/current['word_vocabulary_path'])
    signature=q.identity(dict(checkpoint=session.checkpoint_sha256,settings=current,source=sources,
        vocabulary=vocab,features=feature_signature,matcher_version=MATCH_VERSION))
    if signature==old['prediction_signature']:return
    cohort=json.loads((output/'cohort.json').read_text());directory=output/'predictions';copied=[]
    for record in cohort['records']:
        images=[im['sha256'] for im in record['images']]
        old_key=q.identity(dict(signature=old['prediction_signature'],representation='fused',images=images))
        path=directory/(old_key+'.npz');meta=path.with_suffix('.json')
        if not path.exists() or not meta.exists():continue
        info=json.loads(meta.read_text())
        if info['key']!=old_key or info['sha256']!=file_hash(path):raise ValueError('Coarse prediction cache changed')
        with np.load(path,allow_pickle=False) as arrays: nonempty=bool(arrays['a'].any() or arrays['b'].any())
        if not nonempty:continue
        key=q.identity(dict(signature=signature,representation='fused',images=images));target=directory/(key+'.npz')
        shutil.copy2(path,target);write_json(target.with_suffix('.json'),dict(info,key=key,sha256=file_hash(target)))
        copied.append(record['sample_id'])
    write_json(output/'accepted_cache_migration.json',dict(
        reason='Only the corroboration OR clause changed; existing accepted words/passages are identical. Empty predictions are recomputed.',
        source_prediction_signature=old['prediction_signature'],new_prediction_signature=signature,
        unchanged_generator_signature=historical_generation,migrated_nonempty_predictions=copied,
        ground_truth_consulted=False,old_selector_sha256=hashlib.sha256(before.encode()).hexdigest(),
        new_selector_sha256=hashlib.sha256(after.encode()).hexdigest()))
    print('Verified unchanged accepted predictions reused:',len(copied),flush=True)


def new_negative_pairs(session, baseline, count=300, seed=42):
    old = {tuple(im['sha256'] for im in r['images']) for r in baseline['records'] if r['category'] == 'negative'}
    records = [r for r in session.views['test'].records if r['book_id'] in ('A', 'B')]
    catalog, aliases, statistics = arabic_data_pairs(session.dataset_path, records, negative_count='all', seed=seed)
    hashes, eligible = {}, []
    for pair in catalog:
        if pair['target']: continue
        for side in pair['sides']:
            hashes.setdefault(side['image'], file_hash(side['image']))
        key = tuple(hashes[s['image']] for s in pair['sides'])
        if key in old: continue
        pair = dict(pair, image_sha256=list(key), canonical_sides=[session.allowed[aliases[_side_key(s)]]['side'] for s in pair['sides']])
        eligible.append(pair)
    if len(eligible) < count: raise ValueError('Insufficient disjoint negative combinations')
    chosen = random.Random(seed).sample(sorted(eligible, key=lambda p: p['sample_id']), count)
    if any(tuple(p['image_sha256']) in old for p in chosen): raise RuntimeError('Negative pair overlap')
    return chosen, dict(seed=seed, statistics=statistics, available_after_excluding_old=len(eligible),
        old_negative_pairs=len(old), overlap_pairs=0,
        limitation='Different pair combinations from the same saved-test line inventory; not independent new images')


def audit_negatives(session, selected, output, count=300, seed=42):
    baseline = json.loads((BASELINE / 'cohort.json').read_text())
    pairs, protocol = new_negative_pairs(session, baseline, count, seed)
    write_json(output / 'additional_negative_protocol.json', protocol)
    prepared, signature = extract_features(session, pairs, output, 'fused')
    decoder = cached_session(CHECKPOINT, session.config, 'test')
    decoder.checkpoint_sha256 = session.checkpoint_sha256
    decoder.settings = dict(selected['settings'])
    rows, records = [], []
    for index, (item, pair) in enumerate(zip(prepared, pairs), 1):
        record = dict(sample_id=pair['sample_id'], category='negative', bootstrap_group=pair['sample_id'],
            membership='test', source_splits=['test', 'test'], training_seen=False,
            heldout=True, development_related=False, presumed_negative=True)
        with torch.inference_mode():
            prediction = predict_cached_pair(decoder, item['inference'], item['lines'], settings=selected['settings'])
        truth = [np.zeros(tuple(reversed(line['geometry']['source_size'])), bool) for line in item['lines']]
        values = dict(masks=prediction['masks'], candidate_limit_hit=bool(prediction['match']['candidate_limit_reached']),
                      region_count=len(prediction['match']['regions']))
        rows.append(score_masks(values, truth, record))
        records.append(dict(sample_id=pair['sample_id'], image_sha256=pair['image_sha256'],
            images=[s['image'] for s in pair['sides']], source_splits=['test', 'test']))
        if index % 25 == 0 or index == len(pairs): print(f'Additional test negatives: {index}/{len(pairs)}', flush=True)
    report = dict(**protocol, pairs=len(rows), false_positive_pairs=sum(r['negative_false_positive'] for r in rows),
        fpr=float(np.mean([r['negative_false_positive'] for r in rows])),
        masked_area=float(np.mean([r['negative_masked_area_fraction'] for r in rows])),
        feature_signature=signature, selected_preset=selected['name'])
    write_json(output / 'additional_negative_cohort.json', records)
    write_csv(output / 'additional_negatives.csv', rows)
    write_json(output / 'additional_negative_results.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected', type=Path, default=ROOT/'Results/Evaluation/ab_low_fpr_20261109/selected.json')
    parser.add_argument('--output', type=Path, default=ROOT/'Results/Evaluation/ab_300_low_fpr_20261109')
    args = parser.parse_args(); output = args.output; output.mkdir(parents=True, exist_ok=True)
    frozen_sha = file_hash(args.selected); selected = json.loads(args.selected.read_text())
    if selected['checkpoint_sha256'] != file_hash(CHECKPOINT): raise ValueError('Selected preset checkpoint mismatch')
    write_json(output / 'evaluation_protocol.json', dict(selection_sha256=frozen_sha, selected_preset=selected['name'],
        selection='Final validation selection frozen before the final evaluation; see refinement protocol for the stopped coarse computation',
        main_cohort='Exact same 300 aligned and 300 presumed-negative A/B pairs as baseline, seed 42',
        additional_negatives='300 saved-test A/B comparisons excluding the original 300 negative combinations',
        no_test_parameter_search=True))
    features = output / 'features'
    if features.is_symlink() and not features.exists(): features.unlink()
    if not features.exists():
        if (BASELINE/'features').is_dir():features.symlink_to((BASELINE/'features').resolve(),target_is_directory=True)
        else:features.mkdir()
    torch.set_num_threads(1)
    session = EvaluationSession(CHECKPOINT, ROOT/'DataSet/ArabicData', split='test',
        device='mps' if torch.backends.mps.is_available() else 'cpu', settings=selected['settings'],
        construct_negatives=False, pair_source='folders', crop_override='none')
    migrate_accepted_coarse_predictions(session,selected,output)
    result = evaluate_ab_quantitative(session, output, settings=selected['settings'], workers=1)
    baseline = json.loads((BASELINE / 'cohort.json').read_text())
    keys = lambda m: {tuple(im['sha256'] for im in r['images']) for r in m['records']}
    if keys(result['manifest']) != keys(baseline): raise RuntimeError('Before/after cohort changed')
    previous = json.loads((BASELINE / 'metrics.json').read_text())
    write_json(output / 'comparison.json', dict(baseline=previous, selected=result['metrics'],
        selected_name=selected['name'], validation=selected['validation_metrics']))
    print('MAIN TEST/DIAGNOSTIC:', result['metrics'], flush=True)
    additional = audit_negatives(session, selected, output)
    if file_hash(args.selected) != frozen_sha: raise RuntimeError('Selection changed after test evaluation began')
    print('ADDITIONAL NEGATIVES:', additional, flush=True)


if __name__ == '__main__': main()
