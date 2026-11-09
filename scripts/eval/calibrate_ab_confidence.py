"""Select word-confidence thresholds on saved validation data, never test scores.

Candidate generation, long visual passages, CNN weights and vocabulary stay
fixed. Previously reviewed examples are explicit development regression guards.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
from pathlib import Path
import pickle
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from dataset import file_hash
from evaluate import region_mask, resolved_match_settings
from evaluation_utils import EvaluationSession, arabic_data_pairs, load_folder_masks, predict_cached_pair, _side_key, _image_key
from scripts.eval import mask_quantitative as q
from scripts.eval.ab_quantitative import model_hash
from scripts.eval.final_fragments import DEVELOPMENT_PAIRS
from scripts.eval.review_reordered import pool_signature, candidate_pool_key
from scripts.eval.tune_matching import write_json, write_csv
from word_alignment import select_word_candidates

CHECKPOINT = ROOT / 'Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt'
BASELINE = ROOT / 'docs/evaluation/word_alignment_20261109/settings.json'
PRIOR = ROOT / 'Results/Evaluation/word_alignment_20261109'


def presets(baseline, seed=42, limit=220):
    profiles = list(itertools.product((1.2, 1.4, 1.6, 1.8), (.075, .1, .15, .2, .3),
        (.15, .25, .4), (.4, .5, .6, .7), (2, 3), (8, 10, 11), (8., 12., 20.)))
    rng = random.Random(seed)
    chosen = rng.sample(profiles, min(limit, len(profiles)))
    result = [('baseline', dict(baseline))]
    for key, values in [('word_min_posterior', (.1, .15, .2, .3, .5)),
                        ('word_min_posterior_short', (.25, .4, .6)),
                        ('word_min_cosine', (.5, .6, .7, .8)),
                        ('word_max_nll', (1.2, 1.4, 1.6)),
                        ('word_min_distinct_types', (3,)), ('word_min_total_characters', (10, 11))]:
        result.extend((f'{key}_{value}', dict(baseline, **{key: value})) for value in values)
    keys = ('word_max_nll', 'word_min_posterior', 'word_min_posterior_short', 'word_min_cosine',
            'word_min_distinct_types', 'word_min_total_characters', 'word_visual_weight')
    result += [(f'confidence_{i:03d}', dict(baseline, **dict(zip(keys, values)))) for i, values in enumerate(chosen)]
    seen, unique = set(), []
    for name, settings in result:
        settings = resolved_match_settings(settings); key = json.dumps(settings, sort_keys=True)
        if key not in seen: seen.add(key); unique.append((name, settings))
    return unique


def replay(prediction, settings, config, *, columns=False):
    """Only confidence/ranking changes; generation/long fallback stay identical."""
    candidates = [dict(c) for c in prediction['match']['candidates']]
    regions = select_word_candidates(candidates, settings)
    geometries = [dict(line['geometry'], source_size=[line['geometry']['source_size'][0], 1])
                  if columns else line['geometry'] for line in prediction['lines']]
    masks = [region_mask(regions, i, geometry, config.window_width, config.window_stride)
             for i, geometry in enumerate(geometries)]
    return regions, masks


def scores(masks, truth):
    # Full-height column scoring is exactly equal to source-pixel ratios.
    result = []
    for mask, gt in zip(masks, truth):
        pred = np.asarray(mask) > 0
        if pred.shape != gt.shape: raise ValueError('Mask shape changed')
        if np.all(pred == pred[:1]) and np.all(gt == gt[:1]): pred, gt = pred[0], gt[0]
        tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
        result.append([tp / (tp+fp+fn) if tp+fp+fn else 0.,
            2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            tp / (tp+fp) if tp+fp else 0., tp / (tp+fn) if tp+fn else 0.])
    return np.mean(result, axis=0)


def prepare_validation(session, output, negative_count, seed):
    saved = torch.load(PRIOR / 'full_line/val_prepared.pt', map_location='cpu', weights_only=False)
    if saved['provenance']['checkpoint_sha256'] != session.checkpoint_sha256 or saved['provenance']['config']['crop'] != 'none':
        raise ValueError('Prior features belong to different checkpoint/preprocessing')
    old = {item['record']['sample_id']: item for item in saved['prepared']}
    # Prime only exact, verified images in the current saved validation split.
    for item in saved['prepared']:
        for side, line, image in zip(item['inference']['sides'], item['lines'], item['record']['images']):
            alias = session.line_aliases.get(_side_key(side), _side_key(side))
            if alias in session.allowed:
                if file_hash(side['image']) != image['sha256']: raise ValueError('Validation image changed')
                session.feature_cache[_image_key(session.allowed[alias]['side'])] = line
    records = [r for r in session.views['val'].records if r['book_id'] in ('A', 'B')]
    pairs, aliases, statistics = arabic_data_pairs(session.dataset_path, records, negative_count=negative_count, seed=seed)
    session.line_aliases.update(aliases)
    prepared = []
    for pair in sorted(pairs, key=lambda p: (not p['target'], p['sample_id'])):
        masks, _ = load_folder_masks(pair)
        if pair['target'] and not all(mask.any() for mask in masks): raise ValueError('Validation positive has missing/empty GT')
        if not pair['target'] and any(mask.any() for mask in masks): raise ValueError('Validation negative has nonempty GT')
        inference = dict(sample_id=pair['sample_id'], sides=[dict(image=s['image'], bbox=None) for s in pair['sides']])
        lines = [session.get_line_features(side) for side in pair['sides']]
        if pair['sample_id'] in old:
            record = old[pair['sample_id']]['record']
        else:
            record = dict(sample_id=pair['sample_id'], category='positive' if pair['target'] else 'negative',
                images=[dict(image=s['image'], sha256=file_hash(s['image']), bbox=s.get('bbox'), group_id=s.get('group_id')) for s in pair['sides']])
        if not all(np.all(mask == mask[:1]) for mask in masks):
            raise ValueError('Column replay requires full-height validation masks')
        prepared.append(dict(inference=inference, lines=lines, truth=[m[:1].copy() for m in masks], record=record,
            target=pair['target']))
    write_json(output / 'validation_cohort.json', dict(split='val', seed=seed, statistics=statistics,
        records=[dict(sample_id=p['record']['sample_id'], target=p['target'], images=p['record']['images']) for p in prepared]))
    return prepared


def compact_prediction(prediction):
    # Acceptance and mask construction use only these immutable statistics.
    # Dropping traceback prose saves substantial RAM and compression work.
    keys = {'candidate_type', 'candidate_id', 'word', 'word_nll', 'word_pair_posterior', 'mean_cosine',
            'logical_ranges', 'supported_physical', 'filled_physical', 'source_word_ranges'}
    candidates = [{k: v for k, v in c.items() if k in keys} for c in prediction['match']['candidates']]
    return dict(pair=prediction['pair'], match=dict(candidates=candidates),
        lines=[dict(geometry=line['geometry'], physical=line['physical']) for line in prediction['lines']])


def load_pool(session, item, baseline, output, generation):
    key = candidate_pool_key(generation, item)
    local = output / 'candidate_cache' / (key + '.pkl.gz')
    if local.exists():
        try:
            with gzip.open(local, 'rb') as stream: return compact_prediction(pickle.load(stream))
        except (EOFError, OSError, pickle.UnpicklingError):
            # Only this experiment's interrupted, generated cache is removed.
            local.unlink()
    previous = PRIOR / 'word_validation/candidate_cache' / (key + '.pkl.gz')
    if previous.exists():
        with gzip.open(previous, 'rb') as stream: match = pickle.load(stream)
        prediction = dict(pair=item['inference'], match=match, lines=item['lines'])
    else:
        with torch.inference_mode():
            prediction = predict_cached_pair(session, item['inference'], item['lines'], settings=baseline)
    prediction = compact_prediction(prediction)
    temporary = local.with_suffix('.tmp')
    with gzip.open(temporary, 'wb') as stream: pickle.dump(prediction, stream)
    temporary.replace(local)
    return prediction


def development(session, methods, output):
    saved = torch.load(PRIOR / 'full_line/test_prepared.pt', map_location='cpu', weights_only=False)
    items = [item for item in saved['prepared'] if any(item['record']['sample_id'].endswith(name) for name in DEVELOPMENT_PAIRS)]
    if len(items) != 2: raise ValueError('Missing declared development regressions')
    results = {name: [] for name, _ in methods}
    for item in items:
        inference = dict(sample_id=item['record']['sample_id'], sides=[dict(image=s['image'], bbox=None) for s in item['inference']['sides']])
        for image in item['record']['images']:
            if file_hash(image['image']) != image['sha256']: raise ValueError('Development image changed')
        with torch.inference_mode(): prediction = predict_cached_pair(session, inference, item['lines'], settings=methods[0][1])
        for name, settings in methods:
            regions, masks = replay(prediction, settings, session.config)
            results[name].append(dict(sample_id=inference['sample_id'], iou=float(scores(masks, item['truth'])[0]),
                words=sorted({c['word'] for c in regions if c['candidate_type'] == 'word'})))
    write_json(output / 'development_regressions.json', results)
    return results


def choose(rows, max_fpr):
    eligible = [row for row in rows if row['negative_fpr'] <= max_fpr and row['development_passed']]
    if not eligible: raise ValueError('No validation preset meets the FPR budget and declared development guards')
    return max(eligible, key=lambda row: (row['iou'], row['recall'], row['precision'], -row['negative_fpr'], row['name']))


def development_passed(current, baseline):
    return (all(new['iou'] >= old['iou']-.05 for new, old in zip(current, baseline))
        and all(set(old['words']) <= set(new['words']) for new, old in zip(current, baseline)
                if any(old['sample_id'].endswith(p) for p in DEVELOPMENT_PAIRS[1:])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'Results/Evaluation/ab_low_fpr_20261109')
    parser.add_argument('--negative-count', type=int, default=300)
    parser.add_argument('--max-fpr', type=float, default=.10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=220)
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    (out / 'candidate_cache').mkdir(exist_ok=True)
    torch.set_num_threads(1)
    baseline = json.loads(BASELINE.read_text()); methods = presets(baseline, args.seed, args.limit)
    write_json(out / 'selection_protocol.json', dict(seed=args.seed, split='val', max_negative_fpr=args.max_fpr,
        objective='Maximum validation positive mask IoU subject to FPR budget and declared development regression guards; recall then precision break ties',
        development_guards='Each reviewed example loses at most 0.05 absolute IoU; retain all baseline whole-word identities in the three-word example',
        test_policy='No test/cohort scores used for selection; test evaluated only after selected.json is frozen',
        presets=[dict(name=name, settings=settings) for name, settings in methods]))
    session = EvaluationSession(CHECKPOINT, ROOT/'DataSet/ArabicData', split='val', device='mps' if torch.backends.mps.is_available() else 'cpu',
        settings=baseline, construct_negatives=False, pair_source='folders', crop_override='none')
    session.model.eval().requires_grad_(False); session.text_encoder.eval().requires_grad_(False)
    state = model_hash(session.model); encoder = model_hash(session.text_encoder)
    print('Preparing saved-validation A/B cohort; confidence presets:', len(methods), flush=True)
    prepared = prepare_validation(session, out, args.negative_count, args.seed)
    dev = development(session, methods, out)
    rejected = [name for name, _ in methods if not development_passed(dev[name], dev['baseline'])]
    methods = [(name, settings) for name, settings in methods if name not in rejected]
    write_json(out / 'development_rejected_presets.json', rejected)
    print('Presets passing declared development guards:', len(methods), flush=True)
    npairs = len(prepared); values = np.zeros((len(methods), npairs, 5))
    generation = pool_signature(CHECKPOINT, baseline)
    causes = dict(negative_pairs=0, detected=0, visual=0, word_only=0)
    for index, item in enumerate(prepared):
        prediction = load_pool(session, item, baseline, out, generation)
        for method, (name, settings) in enumerate(methods):
            regions, masks = replay(prediction, settings, session.config, columns=True)
            values[method, index, :4] = scores(masks, item['truth'])
            values[method, index, 4] = bool(any(np.asarray(mask).any() for mask in masks))
            if method == 0 and not item['target']:
                causes['negative_pairs'] += 1
                if regions:
                    causes['detected'] += 1
                    causes['visual' if any(c['candidate_type']=='visual_passage' for c in regions) else 'word_only'] += 1
        if (index+1) % 20 == 0 or index+1 == npairs:
            print(f'Validation candidate replay: {index+1}/{npairs} pairs, {len(methods)} presets', flush=True)
    positive = np.array([p['target'] == 1 for p in prepared]); rows = []
    for i, (name, settings) in enumerate(methods):
        means = values[i, positive, :4].mean(0)
        guard = development_passed(dev[name], dev['baseline'])
        rows.append(dict(name=name, iou=float(means[0]), dice=float(means[1]), precision=float(means[2]), recall=float(means[3]),
            positive_pairs=int(positive.sum()), negative_pairs=int((~positive).sum()),
            negative_false_positives=int(values[i, ~positive, 4].sum()), negative_fpr=float(values[i, ~positive, 4].mean()),
            positive_detection_rate=float(values[i, positive, 4].mean()), development_passed=bool(guard)))
    write_csv(out / 'validation.csv', rows); write_json(out / 'baseline_negative_causes.json', causes)
    np.savez_compressed(out / 'validation_scores.npz', scores=values, positive=positive)
    selected = choose(rows, args.max_fpr); settings = dict(methods)[selected['name']]
    frozen = dict(name=selected['name'], settings=settings, validation_metrics=selected, baseline_validation=rows[0],
        checkpoint_sha256=session.checkpoint_sha256, matcher_version=__import__('evaluate').MATCH_VERSION,
        selection='Saved validation masks plus previously declared development regressions; no test selection',
        vocabulary_sha256=file_hash(ROOT / settings['word_vocabulary_path']), development=dev[selected['name']])
    assert model_hash(session.model) == state and model_hash(session.text_encoder) == encoder
    assert file_hash(CHECKPOINT) == session.checkpoint_sha256
    write_json(out / 'selected.json', frozen)
    print('BASELINE:', rows[0], flush=True); print('SELECTED:', selected, flush=True)
    print('NEGATIVE CAUSES:', causes, flush=True)


if __name__ == '__main__': main()
