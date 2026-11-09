"""Large fixed-cohort A/B diagnostics, with saved-test scores kept separate.

No training or parameter selection. References are used by the existing catalog
builder to reject shared words in presumed negatives, never by the predictor.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pandas as pd
from PIL import Image
import torch

from dataset import file_hash, prepare_image
from evaluate import MATCH_VERSION, resolved_match_settings
from evaluation_utils import arabic_data_pairs, _side_key, load_folder_masks, folder_mask_path, predict_cached_pair
from parameters import Config
from scripts.eval import mask_quantitative as q
from scripts.eval.final_fragments import DEVELOPMENT_PAIRS
from scripts.eval.tune_matching import write_json, write_csv

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'ab-catalog-300-v1'


def model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def select_catalog(positives, negatives, positive_count, negative_count, seed):
    """Keep test positives, then seed-sample the rest; never pad/repeat pairs."""
    if min(positive_count, negative_count) < 1:
        raise ValueError('Positive and negative counts must both be positive')
    for pool, count, name in ((positives, positive_count, 'positive'), (negatives, negative_count, 'negative')):
        if len({p['sample_id'] for p in pool}) != len(pool):
            raise ValueError(f'Duplicate {name} IDs')
        if len(pool) < count:
            raise ValueError(f'Only {len(pool)} distinct {name} pairs for requested {count}; no duplication or fallback')
    rng = random.Random(seed)
    test = sorted((p for p in positives if p['membership'] == 'test'), key=lambda p: p['sample_id'])
    other = sorted((p for p in positives if p['membership'] != 'test'), key=lambda p: p['sample_id'])
    chosen = (rng.sample(test, positive_count) if positive_count < len(test)
              else test + rng.sample(other, positive_count - len(test)))
    negative = rng.sample(sorted(negatives, key=lambda p: p['sample_id']), negative_count)
    return sorted(chosen + negative, key=lambda p: (not p['target'], p['sample_id']))


def build_cohort(session, output, positive_count=300, negative_count=300, seed=42):
    """Do not change session.split, its saved membership, or its pair catalog."""
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    audit = q.audit_splits(session)
    write_json(output / 'split_audit.json', audit)
    if audit['page_group_leakage'] or any('test' in row['splits'] for row in audit['image_leakage']):
        raise ValueError('Saved test images/pages overlap other splits; see split_audit.json')
    leaked = {row['image_sha256'] for row in audit['image_leakage']}
    pixels = {im['path']: im['pixel_sha256'] for records in audit['records'].values()
              for row in records for im in row['images']}
    members = {_side_key(side): dict(split=split, record=record, side=side)
               for split, view in session.views.items() for record in view.records for side in record['sides']}
    all_ab = [r for view in session.views.values() for r in view.records if r['book_id'] in ('A', 'B')]
    test_ab = [r for r in session.views['test'].records if r['book_id'] in ('A', 'B')]
    positive_pool, aliases, positive_stats = arabic_data_pairs(session.dataset_path, all_ab, negative_count=0, seed=seed)
    negative_pool, test_aliases, negative_stats = arabic_data_pairs(session.dataset_path, test_ab,
        negative_count=negative_count, seed=seed)
    aliases.update(test_aliases)
    hashes = {}
    def digest(path):
        if path not in hashes: hashes[path] = file_hash(path)
        return hashes[path]
    development_hashes = {digest(side['image']) for pair in positive_pool
        if any(pair['sample_id'].endswith(name) for name in DEVELOPMENT_PAIRS) for side in pair['sides']}
    pools, excluded, seen = {1: [], 0: []}, [], set()
    for pair in positive_pool + [p for p in negative_pool if p['target'] == 0]:
        canonical = [members[aliases[_side_key(side)]] for side in pair['sides']]
        image_hashes = [digest(side['image']) for side in pair['sides']]
        physical_key = tuple(image_hashes)
        if physical_key in seen:
            excluded.append(dict(sample_id=pair['sample_id'], reason='duplicate physical image pair'))
            continue
        seen.add(physical_key)
        gt, _ = load_folder_masks(pair)
        provenance = [dict(path=str(path), sha256=digest(str(path)) if path.is_file() else None,
                           missing_as_black=not path.is_file())
                      for i, side in enumerate(pair['sides']) for path in [folder_mask_path(side, i)]]
        if pair['target'] and not all(mask.any() for mask in gt):
            excluded.append(dict(sample_id=pair['sample_id'], reason='aligned catalog lacks two foreground masks'))
            continue
        if not pair['target'] and any(mask.any() for mask in gt):
            raise ValueError('Unaligned comparison unexpectedly has foreground GT')
        splits = [member['split'] for member in canonical]
        membership = '+'.join(sorted(set(splits)))
        development_related = any(value in development_hashes for value in image_hashes)
        image_overlap = any(pixels[m['side']['image']] in leaked for m in canonical)
        pair = dict(pair, membership=membership, source_splits=splits,
            training_seen=any(split == 'train' for split in splits),
            development_related=development_related, leaked_image=image_overlap,
            heldout=membership == 'test' and not development_related and not image_overlap,
            canonical_sides=[dict(m['side']) for m in canonical],
            image_sha256=image_hashes, mask_provenance=provenance)
        pools[pair['target']].append(pair)
    pairs = select_catalog(pools[1], pools[0], positive_count, negative_count, seed)
    manifest = dict(version=VERSION, seed=seed, positive_count=positive_count, negative_count=negative_count,
        scope='A_B_catalog_diagnostic; mixed saved membership for positives, saved-test presumed negatives',
        selection='Retain all saved-test aligned pairs; seeded sample of remaining aligned pairs. Seeded saved-test unmatched A/B comparisons.',
        saved_test_aligned_available=sum(p['membership'] == 'test' for p in pools[1]),
        eligible_positive_pairs=len(pools[1]), positive_catalog=positive_stats, negative_catalog=negative_stats,
        exclusions=excluded, checkpoint_sha256=session.checkpoint_sha256,
        membership_counts=dict(Counter(('positive:' if p['target'] else 'negative:') + p['membership'] for p in pairs)),
        records=[dict(sample_id=p['sample_id'], category='positive' if p['target'] else 'negative',
            target=p['target'], source_splits=p['source_splits'], membership=p['membership'], training_seen=p['training_seen'],
            heldout=p['heldout'], development_related=p['development_related'], leaked_image=p['leaked_image'],
            images=[dict(image=side['image'], sha256=sha, canonical_image=canonical['image'], group_id=group)
                for side, canonical, sha, group in zip(p['sides'], p['canonical_sides'], p['image_sha256'], p['group_ids'])],
            masks=p['mask_provenance'], presumed_negative=bool(p['constructed_negative']),
            dependency_keys=[*['page:' + g for g in p['group_ids']], *['image:' + s for s in p['image_sha256']]]) for p in pairs],
        notes=['A–B only; no A–C/B–C examples.', '300 aligned pairs are not 300 held-out test pairs.',
               'Negatives are presumed unmatched comparisons, not human-verified no-overlap labels.',
               'Masks and reference text are excluded from matcher inputs; no training or parameter search.'])
    groups = q.dependency_units(manifest['records'])
    for record in manifest['records']: record['bootstrap_group'] = groups[record['sample_id']]
    write_json(output / 'cohort.json', manifest)
    print('Selected cohort:', manifest['membership_counts'], flush=True)
    return pairs, manifest


def extract_features(session, pairs, output, representation):
    """Use each verified indexed image once; feature cache contains no GT/text."""
    config = asdict(session.config)
    sources = {name: file_hash(ROOT / name) for name in ('model.py', 'cnn_encoder.py', 'transformer_encoder.py', 'dataset.py')}
    signature = q.identity(dict(checkpoint=session.checkpoint_sha256, config=config, sources=sources, device=str(session.device)))
    directory = Path(output) / 'features'; directory.mkdir(exist_ok=True)
    cache, prepared = {}, []
    for index, pair in enumerate(pairs, 1):
        lines = []
        for side, canonical, image_sha in zip(pair['sides'], pair['canonical_sides'], pair['image_sha256']):
            key = q.identity(dict(signature=signature, image=image_sha))
            if key not in cache:
                path = directory / (key + '.pt')
                if path.is_file():
                    line = torch.load(path, map_location='cpu', weights_only=False)
                    if line.get('signature') != signature or line.get('image_sha256') != image_sha:
                        raise ValueError('Frozen feature cache provenance mismatch')
                else:
                    _, tensor, geometry = prepare_image(canonical['image'], (session.config.image_height, session.config.image_width),
                        session.config.grayscale, session.config.crop, augment=False, binarize=session.config.binarize)
                    with torch.inference_mode(): value = session.model(tensor[None].to(session.device))
                    valid = value['token_valid'][0]
                    line = dict(signature=signature, image_sha256=image_sha, geometry=geometry,
                        features={r: value[r][0][valid].detach().float().cpu() for r in ('local', 'context', 'fused')},
                        physical=value['physical_window_indices'][0][valid].cpu().numpy())
                    if any(not torch.isfinite(f).all() for f in line['features'].values()):
                        raise ValueError('Nonfinite frozen image features')
                    temporary = path.with_suffix('.tmp'); torch.save(line, temporary); temporary.replace(path)
                cache[key] = line
            lines.append(cache[key])
        # Only images/geometry/features enter the prediction structure.
        inference = dict(sample_id=pair['sample_id'], sides=[dict(image=s['image'], bbox=None) for s in pair['sides']])
        prepared.append(dict(inference=inference, lines=lines))
        if index % 50 == 0 or index == len(pairs):
            print(f'Frozen features: {index}/{len(pairs)} pairs, {len(cache)} unique images', flush=True)
    return prepared, signature


def score_masks(prediction, truths, record):
    """Post-prediction scoring in full source coordinates."""
    scores = [q.binary_metrics(mask, gt) for mask, gt in zip(prediction['masks'], truths)]
    region = [q.region_counts(mask, gt) if score['full_height'] else dict(tp=0, fp=0, fn=0)
              for mask, gt, score in zip(prediction['masks'], truths, scores)]
    row = {k: record[k] for k in ('sample_id', 'category', 'bootstrap_group', 'membership', 'source_splits',
        'training_seen', 'heldout', 'development_related', 'presumed_negative')}
    row.update({name: float(np.mean([s[key] for s in scores])) for name, key in
        [('positive_iou', 'iou'), ('positive_dice', 'dice'), ('precision', 'precision'), ('recall', 'recall')]})
    row.update(negative_false_positive=any(s['predicted_pixels'] for s in scores),
        negative_masked_area_fraction=float(np.mean([s['foreground_fraction'] for s in scores])),
        candidate_limit_hit=bool(prediction['candidate_limit_hit']), region_count=prediction['region_count'],
        region_eligible=all(s['full_height'] for s in scores),
        **{'region_' + name: sum(s[name] for s in region) for name in ('tp', 'fp', 'fn')})
    totals = dict(tp=sum(s['tp'] for s in scores), fp=sum(s['fp'] for s in scores), fn=sum(s['fn'] for s in scores),
                  pixels=sum(s['pixels'] for s in scores))
    row.update(pair_pixel_accuracy=1 - (totals['fp'] + totals['fn']) / totals['pixels'], **totals)
    row['mask_coverage_a'], row['mask_coverage_b'] = [s['foreground_fraction'] for s in scores]
    return row


def _score_partition(task):
    checkpoint, config, settings, checkpoint_sha, representation, items, directory, signature, partition = task
    from evaluation_utils import load_inference_text_encoder
    torch.set_num_threads(1)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    encoder = load_inference_text_encoder(saved, Config(**config), 'cpu').eval().requires_grad_(False)
    del saved
    session = SimpleNamespace(config=Config(**config), settings=settings, text_encoder=encoder,
        checkpoint_sha256=checkpoint_sha, split='A_B_catalog_diagnostic')
    cache = Path(directory) / 'predictions'; cache.mkdir(exist_ok=True)
    rows = []
    for index, item in enumerate(items, 1):
        record = item['record']
        key = q.identity(dict(signature=signature, representation=representation,
                              images=[im['sha256'] for im in record['images']]))
        path = cache / (key + '.npz'); meta = path.with_suffix('.json')
        if path.is_file() and meta.is_file():
            info = json.loads(meta.read_text())
            if info['key'] != key or info['sha256'] != file_hash(path):
                raise ValueError('Prediction cache mismatch')
            with np.load(path, allow_pickle=False) as values:
                prediction = dict(masks=[values['a'].copy(), values['b'].copy()],
                    candidate_limit_hit=info['candidate_limit_hit'], region_count=info['region_count'])
        else:
            with torch.inference_mode():
                result = predict_cached_pair(session, item['inference'], item['lines'], representation)
            prediction = dict(masks=result['masks'], candidate_limit_hit=bool(result['match']['candidate_limit_reached']),
                              region_count=len(result['match']['regions']))
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, a=prediction['masks'][0], b=prediction['masks'][1])
            temporary.replace(path)
            write_json(meta, dict(key=key, sha256=file_hash(path), **{k:prediction[k] for k in ('candidate_limit_hit', 'region_count')}))
            del result
        # GT is loaded only once the image-only prediction is fixed.
        truths = []
        for image, mask in zip(record['images'], record['masks']):
            with Image.open(image['image']) as source: size = (source.height, source.width)
            if record['category'] == 'negative':
                truths.append(np.zeros(size, bool))
            else:
                if file_hash(mask['path']) != mask['sha256']:
                    raise ValueError('GT changed after cohort creation')
                with Image.open(mask['path']) as source: truth = np.asarray(source.convert('L')) > 0
                if truth.shape != size: raise ValueError('GT shape changed after cohort creation')
                truths.append(truth)
        rows.append(score_masks(prediction, truths, record))
        if index % 20 == 0 or index == len(items):
            print(f'A/B partition {partition}: {index}/{len(items)} scored', flush=True)
    write_csv(Path(directory) / f'partition_{partition}.csv', rows)
    return rows


def summarize(rows):
    result = q.aggregate(rows)
    positives = [r for r in rows if r['category'] == 'positive']
    result.update(positive_detected_pairs=sum(r['negative_false_positive'] for r in positives),
        positive_pair_detection_recall=float(np.mean([r['negative_false_positive'] for r in positives])),
        pair_pixel_accuracy=float(np.mean([r['pair_pixel_accuracy'] for r in rows])),
        scope='mixed-membership catalog diagnostic' if any(r['membership'] != 'test' for r in rows) else 'saved-test; development-related images excluded')
    return result


def report_tables(rows):
    """Six hundred catalog results and the genuine held-out subset are distinct."""
    rows = list(rows)
    overall = summarize(rows)
    heldout_rows = [r for r in rows if r['heldout']]
    heldout = summarize(heldout_rows) if {r['category'] for r in heldout_rows} == {'positive', 'negative'} else None
    tables = []
    for name, score in [('300 + 300 A–B catalog diagnostic', overall), ('Saved-test subset (development excluded)', heldout)]:
        if score is None: continue
        tables.append(dict(cohort=name, aligned_pairs=score['positive_pairs'], unaligned_pairs=score['negative_pairs'],
            IoU=score['positive_iou'], Dice=score['positive_dice'], Precision=score['precision'], Recall=score['recall'],
            negative_FPR=score['negative_pair_fpr'], negative_false_positives=score['negative_false_positive_pairs'],
            negative_masked_area=score['negative_masked_area_fraction'], pair_detection_recall=score['positive_pair_detection_recall']))
    membership = []
    for key in sorted({r['membership'] for r in rows}):
        chosen = [r for r in rows if r['membership'] == key]
        pos, neg = ([r for r in chosen if r['category'] == kind] for kind in ('positive', 'negative'))
        membership.append(dict(membership=key, aligned=len(pos), unaligned=len(neg),
            IoU=float(np.mean([r['positive_iou'] for r in pos])) if pos else None,
            Precision=float(np.mean([r['precision'] for r in pos])) if pos else None,
            Recall=float(np.mean([r['recall'] for r in pos])) if pos else None))
    return pd.DataFrame(tables), pd.DataFrame(membership), dict(catalog=overall, heldout=heldout)


def evaluate_ab_quantitative(session, output, *, positive_count=300, negative_count=300, seed=42,
                             representation='fused', workers=4, settings=None):
    """One shared notebook/CLI evaluation entry point; no split mutation."""
    if workers < 1: raise ValueError('workers must be positive')
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    active = resolved_match_settings(session.settings if settings is None else settings)
    checkpoint_sha = file_hash(session.checkpoint)
    if checkpoint_sha != session.checkpoint_sha256: raise ValueError('Checkpoint changed')
    session.model.eval().requires_grad_(False); session.text_encoder.eval().requires_grad_(False)
    state_before = model_hash(session.model); text_before = model_hash(session.text_encoder)
    options_before = dict(session.settings)
    source = {name: file_hash(ROOT / name) for name in ('evaluate.py', 'word_alignment.py', 'alignment_candidates.py',
        'evaluation_utils.py', 'dtw.py', 'scripts/eval/ab_quantitative.py')}
    vocabulary = ROOT / active['word_vocabulary_path'] if active['decoder'] == 'word_dtw' else None
    vocabulary_sha = file_hash(vocabulary) if vocabulary else None
    pairs, manifest = build_cohort(session, output, positive_count, negative_count, seed)
    prepared, feature_signature = extract_features(session, pairs, output, representation)
    for item, record in zip(prepared, manifest['records']): item['record'] = record
    signature = q.identity(dict(checkpoint=checkpoint_sha, settings=active, source=source,
        vocabulary=vocabulary_sha, features=feature_signature, matcher_version=MATCH_VERSION))
    write_json(output / 'run_settings.json', dict(checkpoint_sha256=checkpoint_sha, settings=active,
        feature_signature=feature_signature, prediction_signature=signature, sources=source,
        representation=representation, vocabulary_sha256=vocabulary_sha))
    count = min(workers, len(prepared)); rows = []
    tasks = [(str(session.checkpoint), asdict(session.config), active, checkpoint_sha, representation,
        prepared[i::count], str(output), signature, i) for i in range(count)]
    if count == 1:
        rows = _score_partition(tasks[0])
    else:
        # Spawn avoids inheriting the MPS/CUDA runtime. Workers restore only the
        # frozen character encoder and score CPU features, never train a model.
        with ProcessPoolExecutor(max_workers=count, mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = [executor.submit(_score_partition, task) for task in tasks]
            for future in as_completed(futures): rows.extend(future.result())
    rows.sort(key=lambda row: (row['category'] != 'positive', row['sample_id']))
    if len(rows) != positive_count + negative_count or len({r['sample_id'] for r in rows}) != len(rows):
        raise RuntimeError('Incomplete or duplicated quantitative results')
    table, membership, report = report_tables(rows)
    checks = dict(checkpoint_unchanged=file_hash(session.checkpoint) == checkpoint_sha,
        model_state_unchanged=model_hash(session.model) == state_before,
        text_encoder_unchanged=model_hash(session.text_encoder) == text_before,
        settings_unchanged=session.settings == options_before,
        vocabulary_unchanged=not vocabulary or file_hash(vocabulary) == vocabulary_sha,
        model_eval=not session.model.training, gradients_disabled=not any(p.requires_grad for p in session.model.parameters()))
    if not all(checks.values()): raise RuntimeError(f'Frozen inference contract changed: {checks}')
    verification = dict(checks=checks, training_steps=0, checkpoint_sha256=checkpoint_sha, source_sha256=source,
        settings=active, positive_count=positive_count, negative_count=negative_count,
        seed=seed, representation=representation, device=str(session.device), scope=manifest['scope'])
    write_csv(output / 'per_pair.csv', rows); write_json(output / 'metrics.json', report)
    table.to_csv(output / 'summary.csv', index=False); membership.to_csv(output / 'membership.csv', index=False)
    write_json(output / 'verification.json', verification)
    return dict(rows=rows, summary=table, membership=membership, metrics=report, manifest=manifest, verification=verification)
