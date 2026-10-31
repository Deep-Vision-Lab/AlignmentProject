"""Frozen-checkpoint mask evaluation and constrained validation-only Optuna search.

Pair JSONL rows require sample_id, image_a, image_b, mask_a, mask_b,
label (positive/negative), and verified=true. Images must belong to the
checkpoint's saved split. Masks are source-image, full-height binary spans.
No transcript, label or mask is passed to the image-only matcher.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import random
import subprocess
import time

import numpy as np
import optuna
from PIL import Image

from alignment_candidates import validate_path_options
from dataset import file_hash
from evaluate import MATCH_VERSION, resolved_match_settings
from evaluation_utils import EvaluationSession, _image_key, predict_cached_pair


SEARCH_VERSION = 'mask-optuna-v1'
FIXED_SEARCH = dict(alignment_mode='local_repeat_dtw', candidate_similarity_mode='letter_evidence',
                    similarity_mode='letter_evidence', use_path_anchor_check=True,
                    threshold=.5, max_candidates=512)
REWARD_CHOICES = [None, 0., .1, .25, .5, 1.]
REPEAT_CHOICES = [0., .02, .05, .10, .20]
LONG_FRACTIONS = [.05, .10, .20, .35, .50, .70, .90]
DENSITIES = [.60, .70, .80, .85, .90, .95, 1.]


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, data):
    Path(path).write_text(json.dumps(_jsonable(data), ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                          encoding='utf-8')


def _write_csv(path, rows):
    rows = [_jsonable(r) for r in rows]
    columns = sorted({k for row in rows for k in row})
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def notebook_settings(path='notebooks/model_evaluation.ipynb'):
    """Read literal values in the live notebook settings cell; never execute it."""
    notebook = json.loads(Path(path).read_text(encoding='utf-8'))
    source = next((''.join(c['source']) for c in notebook['cells']
                   if ''.join(c.get('source', [])).startswith('# EVALUATION-ONLY MATCHING SETTINGS.')), None)
    if source is None:
        raise ValueError('Notebook matching-settings cell not found')
    values = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if isinstance(node.value, ast.Name):
            if node.value.id in values:
                values[name] = values[node.value.id]
        else:
            try:
                values[name] = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                pass
    aliases = {'SIMILARITY_THRESHOLD': 'threshold', 'MAX_INTERNAL_GAP': 'max_gap',
               'MAX_ALIGNMENT_CANDIDATES': 'max_candidates'}
    requested = {aliases.get(k, k.lower()): v for k, v in values.items()
                 if aliases.get(k, k.lower()) in resolved_match_settings({})}
    return checked_settings(requested)


def checked_settings(requested):
    resolved = resolved_match_settings(requested)
    validate_path_options(resolved)
    if resolved['alignment_mode'] != 'local_repeat_dtw':
        raise ValueError('This tuner requires the production local_repeat_dtw matcher')
    return resolved


def search_baseline(notebook):
    """Search baseline changes only the predeclared fixed evidence settings."""
    return checked_settings(dict(notebook, **FIXED_SEARCH))


def sample_settings(trial, stage, baseline):
    if stage not in (1, 2):
        raise ValueError('stage must be 1 or 2')
    settings = dict(baseline)
    distinct = trial.suggest_int('normal_distinct', 4, 10)
    gap = trial.suggest_int('max_gap', 0, 6)
    opening = trial.suggest_float('gap_open', .05, .60)
    settings.update(normal_min_distinct_a=distinct, normal_min_distinct_b=distinct,
        normal_min_matched_pairs=trial.suggest_int('normal_min_matched_pairs', 4, 12),
        allow_short_regions=trial.suggest_categorical('allow_short_regions', [False, True]),
        max_gap=gap, gap_open=opening, gap_extend=trial.suggest_float('gap_extend', .01, opening),
        repeat_penalty=trial.suggest_categorical('repeat_penalty', REPEAT_CHOICES),
        max_consecutive_repeats=trial.suggest_int('max_consecutive_repeats', 1, 4),
        anchor_top_k=trial.suggest_int('anchor_top_k', 1, 4),
        min_mutual_anchors_long=trial.suggest_int('min_mutual_anchors_long', 1, 5),
        min_mutual_anchor_fraction_long=trial.suggest_categorical('min_mutual_anchor_fraction_long', LONG_FRACTIONS),
        min_path_density=trial.suggest_categorical('min_path_density', DENSITIES),
        min_mean_reward=trial.suggest_categorical('min_mean_reward', REWARD_CHOICES))
    if stage == 2:
        if settings['allow_short_regions']:
            short_distinct = trial.suggest_int('short_distinct', 3, min(5, distinct))
            short_cosine = trial.suggest_float('short_cosine', .70, .90, step=.02)
            short_count = trial.suggest_int('short_anchor_count', 1, 4)
            short_fraction = trial.suggest_categorical('short_anchor_fraction', [.40, .60, .80, .95, 1.])
            settings.update(short_min_distinct_a=short_distinct, short_min_distinct_b=short_distinct,
                short_min_matched_pairs=trial.suggest_int('short_min_matched_pairs', 4, 7),
                short_min_mean_cosine=short_cosine, short_min_median_cosine=short_cosine,
                short_min_mean_reward=trial.suggest_categorical('short_min_mean_reward', [.10, .25, .50, 1.]),
                short_min_path_density=trial.suggest_categorical('short_min_path_density', [.80, .90, .95, 1.]),
                short_max_internal_gap=trial.suggest_int('short_max_internal_gap', 0, min(2, gap)),
                short_min_mutual_anchors=short_count, min_mutual_anchors_short=short_count,
                short_min_mutual_anchor_fraction=short_fraction, min_mutual_anchor_fraction_short=short_fraction)
        settings['enable_path_extension'] = trial.suggest_categorical('enable_path_extension', [False, True])
        if settings['enable_path_extension']:
            settings.update(extension_min_reward=trial.suggest_categorical('extension_min_reward', [0., .05, .10, .25, .50]),
                extension_max_weak_run=trial.suggest_int('extension_max_weak_run', 0, 3),
                extension_max_internal_gap=trial.suggest_int('extension_max_internal_gap', 0, min(4, gap)))
        settings['enable_region_merging'] = trial.suggest_categorical('enable_region_merging', [False, True])
        if settings['enable_region_merging']:
            merge = trial.suggest_int('merge_max_gap', 0, min(4, gap))
            settings.update(merge_max_gap_a=merge, merge_max_gap_b=merge)
    return checked_settings(settings)


def _intervals(mask):
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.array_equal(mask, np.broadcast_to(mask[0:1], mask.shape)):
        raise ValueError('Region scoring requires a full-height column-span mask')
    active = mask[0]
    edges = np.diff(np.r_[False, active, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def region_counts(pred, truth, threshold=.5):
    """Ordered interval DP maximizes (match count, summed IoU), lexicographically."""
    left, right = _intervals(pred), _intervals(truth)
    n, m = len(left), len(right)
    dp = [[(0, 0.) for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            a, b = left[i - 1]; c, d = right[j - 1]
            intersection = max(0, min(b, d) - max(a, c))
            iou = intersection / (b - a + d - c - intersection)
            choices = [dp[i - 1][j], dp[i][j - 1]]
            if iou >= threshold:
                matches, total_iou = dp[i - 1][j - 1]
                choices.append((matches + 1, total_iou + iou))
            dp[i][j] = max(choices)
    matched = dp[n][m][0]
    return dict(tp=matched, fp=n - matched, fn=m - matched,
                matched_iou_sum=dp[n][m][1], predicted=n, ground_truth=m)


def positive_line_metrics(pred, truth):
    pred, truth = np.asarray(pred) > 0, np.asarray(truth) > 0
    if pred.shape != truth.shape or pred.ndim != 2 or not truth.any():
        raise ValueError('Positive pair requires nonempty same-size source masks')
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    region = region_counts(pred, truth)
    return dict(iou=tp / (tp + fp + fn), dice=2 * tp / (2 * tp + fp + fn),
                precision=tp / (tp + fp) if tp + fp else 0., recall=tp / (tp + fn),
                tp=tp, fp=fp, fn=fn, coverage=float(pred.mean()), **{f'region_{k}': v for k, v in region.items()})


def pair_metrics(predictions, truths, label, sample_id):
    if label not in ('positive', 'negative') or len(predictions) != 2 or len(truths) != 2:
        raise ValueError('Require a two-line verified positive or negative pair')
    if any(np.shape(p) != np.shape(t) for p, t in zip(predictions, truths)):
        raise ValueError('Prediction and GT must have identical source coordinates/resolution')
    if label == 'positive':
        if not all(np.asarray(t).any() for t in truths):
            raise ValueError(f'Positive pair has empty ground-truth mask: {sample_id}')
        lines = [positive_line_metrics(p, t) for p, t in zip(predictions, truths)]
        pair = {k: sum(line[k] for line in lines) / 2 for k in ('iou', 'dice', 'precision', 'recall')}
        pair['negative_false_positive'] = None
        pair['negative_coverage'] = None
    else:
        if any(np.asarray(t).any() for t in truths):
            raise ValueError(f'Negative pair has nonempty ground-truth mask: {sample_id}')
        lines = [dict(coverage=float((np.asarray(p) > 0).mean()),
                      predicted_pixels=int((np.asarray(p) > 0).sum())) for p in predictions]
        pair = dict(negative_false_positive=int(any(line['predicted_pixels'] for line in lines)),
                    negative_coverage=sum(line['coverage'] for line in lines) / 2)
    return pair, lines


def aggregate(rows):
    positives = [r for r in rows if r['label'] == 'positive']
    negatives = [r for r in rows if r['label'] == 'negative']
    if not positives or not negatives:
        raise ValueError('Evaluation requires both verified positive and verified negative pairs')
    result = dict(positive_pairs=len(positives), negative_pairs=len(negatives),
                  negative_false_positive_count=sum(r['negative_false_positive'] for r in negatives),
                  negative_pair_fpr=sum(r['negative_false_positive'] for r in negatives) / len(negatives),
                  negative_masked_area_fraction=float(np.mean([r['negative_coverage'] for r in negatives])))
    for key in ('iou', 'dice', 'precision', 'recall'):
        result['positive_' + key] = float(np.mean([r[key] for r in positives]))
    tp = sum(line['region_tp'] for r in positives for line in r['lines'])
    fp = sum(line['region_fp'] for r in positives for line in r['lines'])
    fn = sum(line['region_fn'] for r in positives for line in r['lines'])
    result.update(region_tp=tp, region_fp=fp, region_fn=fn,
                  region_precision=tp / (tp + fp) if tp + fp else 0.,
                  region_recall=tp / (tp + fn) if tp + fn else 0.,
                  region_f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)
    return result


def _load_mask(path, shape, *, resize_to_source=False):
    with Image.open(path) as image:
        if image.size != (shape[1], shape[0]):
            if not resize_to_source:
                raise ValueError(f'Mask {path} is {image.size}; expected {(shape[1], shape[0])}. '
                                 'Explicit resize_to_source=true is required for nearest-neighbor resize')
            image = image.resize((shape[1], shape[0]), Image.Resampling.NEAREST)
        mask = np.asarray(image.convert('L')) >= 128
    _intervals(mask)  # A 1D column equivalent is safe only for full-height spans.
    return mask


def verified_pairs(session, manifest):
    root = Path(manifest).resolve().parent
    image_to_side = defaultdict(list)
    for entry in session.allowed.values():
        side = entry['side']
        image_to_side[str(Path(side['image']).resolve())].append(side)
    pairs, truths, seen = [], {}, set()
    for number, line in enumerate(Path(manifest).read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        sample_id = str(row['sample_id'])
        if sample_id in seen:
            raise ValueError(f'Duplicate pair ID: {sample_id}')
        seen.add(sample_id)
        if row.get('verified') is not True:
            raise ValueError(f'{sample_id}: verified=true is required; presumed negatives are forbidden')
        label = row['label']
        if label not in ('positive', 'negative'):
            raise ValueError(f'{sample_id}: label must be positive or negative')
        sides = []
        for side in 'ab':
            image = Path(row[f'image_{side}'])
            image = str((image if image.is_absolute() else session.dataset_path / image).resolve())
            choices = image_to_side.get(image, [])
            if len(choices) != 1:
                raise ValueError(f'{sample_id}: {image} is absent/ambiguous in saved {session.split} split')
            sides.append(choices[0])
        if sides[0]['image'] == sides[1]['image']:
            raise ValueError(f'{sample_id}: same image on both sides')
        pair = dict(sample_id=sample_id, sides=sides, target=int(label == 'positive'),
                    label='verified_' + label, constructed_negative=False,
                    annotations=[{}, {}], annotation_provenance=str(Path(manifest).resolve()),
                    manually_verified=True, group_id=row.get('group_id'))
        line_data = [session.get_line_features(side) for side in sides]
        masks = []
        for side, item in zip('ab', line_data):
            path = Path(row[f'mask_{side}'])
            path = (path if path.is_absolute() else root / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            width, height = item['geometry']['source_size']
            masks.append(_load_mask(path, (height, width), resize_to_source=row.get('resize_to_source') is True))
        if label == 'positive' and not all(mask.any() for mask in masks):
            raise ValueError(f'{sample_id}: positive GT must be nonempty on both lines')
        if label == 'negative' and any(mask.any() for mask in masks):
            raise ValueError(f'{sample_id}: negative GT must be empty on both lines')
        pairs.append(pair)
        truths[sample_id] = masks
    if not pairs:
        raise ValueError('Verified pair manifest is empty')
    if not any(p['target'] == 0 for p in pairs) or not any(p['target'] == 1 for p in pairs):
        raise ValueError('Both verified positives and verified negatives are required')
    return pairs, truths


def cache_identity(session, pairs, representation):
    config = session.config
    return dict(version=SEARCH_VERSION, checkpoint_sha256=session.checkpoint_sha256,
                checkpoint_epoch=session.saved['epoch'], split=session.split,
                split_manifest_sha256=session.saved['split_manifest_sha256'],
                pair_ids=[p['sample_id'] for p in pairs], representation=representation,
                preprocessing={k: getattr(config, k) for k in ('crop', 'grayscale', 'binarize',
                    'image_height', 'image_width', 'window_width', 'window_stride', 'rtl',
                    'cnn_type', 'transformer_type', 'fusion_mode', 'use_gated_fusion')},
                evidence=dict(temperature=config.competition_temperature,
                    prior=session.saved.get('letter_evidence_prior')))


def prepare_cache(session, pairs, representation, path):
    identity = cache_identity(session, pairs, representation)
    path = Path(path)
    if path.is_file():
        import torch
        saved = torch.load(path, map_location='cpu', weights_only=False)
        if saved['identity'] != identity:
            raise ValueError('Feature cache identity mismatch; use a distinct output directory')
        session.feature_cache.update(saved['features'])
    for pair in pairs:
        for side in pair['sides']:
            session.get_line_features(side)
    if not path.is_file():
        import torch
        torch.save(dict(identity=identity, features=session.feature_cache), path)
    return identity


def evaluate_settings(session, pairs, truths, representation, requested):
    settings = checked_settings(requested)
    session.settings = settings
    pair_rows, line_rows = [], []
    rejection, candidate_limit = Counter(), 0
    accepted, extension, merged = Counter(), Counter(), Counter()
    distributions = defaultdict(list)
    started = time.monotonic()
    for pair in pairs:  # complete, fixed validation population; no pruning or subsampling
        lines = [session.get_line_features(side) for side in pair['sides']]
        prediction = predict_cached_pair(session, pair, lines, representation, settings=settings)
        masks = prediction['masks']
        scores, per_line = pair_metrics(masks, truths[pair['sample_id']],
                                       'positive' if pair['target'] else 'negative', pair['sample_id'])
        match = prediction['match']
        candidate_limit += int(match.get('candidate_limit_reached', False))
        for candidate in match.get('candidates', []):
            if not candidate.get('accepted'):
                rejection[str(candidate.get('reason'))] += 1
            else:
                accepted[str(candidate.get('candidate_type'))] += 1
                extension['extended'] += int(bool(candidate.get('extension_pairs')))
                merged['merged'] += int(bool(candidate.get('merged_from')))
            for key in ('path_density', 'mutual_anchor_fraction', 'mean_reward'):
                if candidate.get(key) is not None:
                    distributions[key].append(float(candidate[key]))
        pair_row = dict(sample_id=pair['sample_id'], label='positive' if pair['target'] else 'negative',
                        group_id=pair.get('group_id'), image_a=pair['sides'][0]['image'],
                        image_b=pair['sides'][1]['image'], **scores, lines=per_line,
                        accepted_regions=len(match['regions']), candidate_limit_reached=bool(match.get('candidate_limit_reached')))
        pair_rows.append(pair_row)
        for index, metrics in enumerate(per_line):
            line_rows.append(dict(sample_id=pair['sample_id'], label=pair_row['label'],
                                  side='AB'[index], image=pair['sides'][index]['image'], **metrics))
    summary = aggregate(pair_rows)
    summary.update(seconds=time.monotonic() - started, candidate_limit_hits=candidate_limit,
                   accepted_regions=dict(accepted), candidate_rejections=dict(rejection),
                   extension_outcomes=dict(extension), merging_outcomes=dict(merged),
                   candidate_distributions={key: dict(mean=float(np.mean(values)),
                       median=float(np.median(values)), min=float(np.min(values)), max=float(np.max(values)),
                       count=len(values)) for key, values in distributions.items()})
    return dict(summary=summary, pairs=pair_rows, lines=line_rows,
                requested_settings=requested, effective_settings=settings)


def feasible_trials(studies, fpr_limit):
    candidates = [trial for study in studies for trial in study.trials
                  if trial.state == optuna.trial.TrialState.COMPLETE
                  and trial.user_attrs.get('negative_pair_fpr', math.inf) <= fpr_limit]
    return sorted(candidates, key=lambda trial: (-trial.value,
        trial.user_attrs.get('negative_masked_area_fraction', math.inf), trial.number))


def _study(storage, name, stage, seed, signature, fpr_limit):
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=40,
        constraints_func=lambda trial: (trial.user_attrs.get('negative_pair_fpr', math.inf) - fpr_limit,))
    study = optuna.create_study(study_name=name, storage=storage, direction='maximize',
                                sampler=sampler, load_if_exists=True)
    expected = dict(signature=signature, stage=stage, fpr_limit=fpr_limit, search_version=SEARCH_VERSION)
    existing = study.user_attrs.get('provenance')
    if existing is not None and existing != expected:
        raise ValueError(f'Resumed study provenance mismatch: {existing} != {expected}')
    if existing is None:
        if study.trials:
            raise ValueError('Existing study has no provenance; refusing unsafe resume')
        study.set_user_attr('provenance', expected)
    return study


def _fingerprint(data):
    return hashlib.sha256(json.dumps(_jsonable(data), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _trial_rows(studies):
    return [dict(study=study.study_name, trial=trial.number, state=trial.state.name,
                 value=trial.value, parameters=trial.params, **trial.user_attrs)
            for study in studies for trial in study.trials]


def _seed_params(settings):
    """Return compatible core values; incompatible notebook baselines are not enqueued."""
    params = dict(normal_distinct=settings['normal_min_distinct_a'],
        normal_min_matched_pairs=settings['normal_min_matched_pairs'],
        allow_short_regions=settings['allow_short_regions'], max_gap=settings['max_gap'],
        gap_open=settings['gap_open'], gap_extend=settings['gap_extend'],
        repeat_penalty=settings['repeat_penalty'], max_consecutive_repeats=settings['max_consecutive_repeats'],
        anchor_top_k=settings['anchor_top_k'], min_mutual_anchors_long=settings['min_mutual_anchors_long'],
        min_mutual_anchor_fraction_long=settings['min_mutual_anchor_fraction_long'],
        min_path_density=settings['min_path_density'], min_mean_reward=settings['min_mean_reward'])
    valid = (settings['normal_min_distinct_a'] == settings['normal_min_distinct_b'] and
             4 <= params['normal_distinct'] <= 10 and 4 <= params['normal_min_matched_pairs'] <= 12 and
             0 <= params['max_gap'] <= 6 and .05 <= params['gap_open'] <= .60 and
             .01 <= params['gap_extend'] <= params['gap_open'] and
             params['repeat_penalty'] in REPEAT_CHOICES and 1 <= params['max_consecutive_repeats'] <= 4 and
             1 <= params['anchor_top_k'] <= 4 and 1 <= params['min_mutual_anchors_long'] <= 5 and
             params['min_mutual_anchor_fraction_long'] in LONG_FRACTIONS and
             params['min_path_density'] in DENSITIES and params['min_mean_reward'] in REWARD_CHOICES)
    return params if valid else None


def split_audit(session, pairs):
    views = session.views
    image_sets = {name: {str(Path(s['image']).resolve()) for r in view.records for s in r['sides']}
                  for name, view in views.items()}
    overlaps = {f'{a}-{b}': len(image_sets[a] & image_sets[b]) for a, b in
                (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    if any(overlaps.values()):
        raise ValueError(f'Image leakage across saved splits: {overlaps}')
    group_sets = {name: {r['group_id'] for r in view.records if r.get('group_id')}
                  for name, view in views.items()}
    group_overlap = {f'{a}-{b}': len(group_sets[a] & group_sets[b]) for a, b in
                     (('train', 'val'), ('train', 'test'), ('val', 'test'))}
    return dict(image_overlap=overlaps, group_overlap=group_overlap,
                grouping_limitations=('Some records lack group_id; no cross-split regrouping was performed'
                                      if any(not r.get('group_id') for v in views.values() for r in v.records)
                                      else None),
                selected_pair_ids=[p['sample_id'] for p in pairs],
                split_ids=session.saved['split_ids'],
                split_manifest_sha256=session.saved['split_manifest_sha256'])


def _bootstrap(rows, seed=42, resamples=2000):
    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        grouped[str(row.get('group_id') or row['sample_id'])].append(row)
    groups = list(grouped.values())
    values = defaultdict(list)
    for _ in range(resamples):
        chosen = [row for _ in groups for row in rng.choice(groups)]
        if not any(r['label'] == 'positive' for r in chosen) or not any(r['label'] == 'negative' for r in chosen):
            continue
        score = aggregate(chosen)
        for key in ('positive_iou', 'positive_dice', 'positive_precision', 'positive_recall',
                    'negative_pair_fpr', 'negative_masked_area_fraction', 'region_f1'):
            values[key].append(score[key])
    return {key: [float(np.quantile(v, .025)), float(np.quantile(v, .975))]
            for key, v in values.items() if v}


def _wilson(successes, total):
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0., center - half), min(1., center + half)]


def paper_exports(results, directory, seed, resamples):
    directory = Path(directory)
    columns = [('method', 'Method'), ('positive_iou', 'Positive IoU'),
               ('positive_dice', 'Positive Dice'), ('positive_precision', 'Precision'),
               ('positive_recall', 'Recall'), ('negative_pair_fpr', 'Negative-pair FPR'),
               ('negative_masked_area_fraction', 'Negative masked-area fraction')]
    rows = []
    for method, result in results.items():
        summary = result['summary']
        ci = _bootstrap(result['pairs'], seed, resamples)
        row = dict(method=method, positive_pairs=summary['positive_pairs'],
                   negative_pairs=summary['negative_pairs'])
        for key, _ in columns[1:]:
            row[key] = summary[key]
            row[key + '_ci95'] = ci.get(key)
        row['region_f1'] = summary['region_f1']
        row['region_f1_ci95'] = ci.get('region_f1')
        negative_images = [r['image_' + side] for r in result['pairs'] if r['label'] == 'negative' for side in 'ab']
        row['independent_negative_pairs'] = len(negative_images) == len(set(negative_images))
        row['negative_fpr_wilson_ci95'] = (_wilson(summary['negative_false_positive_count'],
                                                 summary['negative_pairs'])
                                            if row['independent_negative_pairs'] else None)
        rows.append(row)
    _write_csv(directory / 'paper_table.csv', rows)
    header = ' & '.join(label for _, label in columns) + ' \\\\ \\hline\n'
    body = ''.join(' & '.join([str(row['method'])] + [f'{row[key]:.3f}' for key, _ in columns[1:]]) + ' \\\\ \n'
                   for row in rows)
    (directory / 'paper_table.tex').write_text('\\begin{tabular}{lrrrrrr}\\hline\n' + header + body +
                                               '\\hline\\end{tabular}\n', encoding='utf-8')
    _write_json(directory / 'paper_table_uncertainty.json', dict(rows=rows, bootstrap_resamples=resamples,
        bootstrap_seed=seed, bootstrap_unit='group_id where supplied, otherwise pair',
        limitation='Pairs sharing images but lacking common group_id are dependent; intervals can be optimistic. '
                   'Wilson FPR intervals are reported only for image-disjoint negative pairs.'))


def _provenance(session, manifest, identity, audit, seed):
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return dict(checkpoint=str(session.checkpoint), checkpoint_sha256=session.checkpoint_sha256,
        checkpoint_epoch=session.saved['epoch'], dataset=str(session.dataset_path), split=session.split,
        pair_manifest=str(Path(manifest).resolve()), pair_manifest_sha256=file_hash(manifest),
        cache_identity=identity, split_audit=audit, seed=seed, git_commit=commit,
        packages={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'optuna', 'Pillow')},
        matcher_version=MATCH_VERSION, search_version=SEARCH_VERSION)


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('cache', 'tune', 'test', 'export'))
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--pairs-manifest', required=True, help='Verified, source-coordinate pair JSONL')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--representation', choices=('local', 'context', 'fused'), default=None)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--storage', help='Optuna SQLite URL; default output-dir/studies.sqlite3')
    parser.add_argument('--stage', type=int, choices=(1, 2), default=1)
    parser.add_argument('--stage1-trials', type=int, default=300)
    parser.add_argument('--stage2-trials', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-negative-pair-fpr', type=float, default=.05)
    parser.add_argument('--bootstrap-resamples', type=int, default=2000)
    return parser.parse_args()


def main():
    args = _arguments()
    if args.action in ('cache', 'tune') and args.split != 'val':
        raise ValueError('Cache/tuning use validation only; test is reserved for frozen final evaluation')
    if args.action == 'test' and args.split != 'test':
        raise ValueError('Final evaluation requires --split test')
    if not 0 <= args.max_negative_pair_fpr <= 1:
        raise ValueError('FPR limit must lie in [0,1]')
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    session = EvaluationSession(args.checkpoint, args.dataset, args.split, args.device,
                                seed=args.seed, construct_negatives=False, negative_count=0,
                                pair_source='external')
    representation = args.representation or ('context' if session.config.fusion_mode == 'context' else 'fused')
    if session.config.fusion_mode == 'context' and representation == 'fused':
        raise ValueError('Context-only checkpoint has no trained fusion head')
    pairs, truths = verified_pairs(session, args.pairs_manifest)
    audit = split_audit(session, pairs)
    identity = prepare_cache(session, pairs, representation, out / f'features_{args.split}.pt')
    provenance = _provenance(session, args.pairs_manifest, identity, audit, args.seed)
    _write_json(out / f'{args.split}_split_manifest.json', audit)
    _write_json(out / f'{args.split}_provenance.json', provenance)
    print(f'{args.split}: {sum(p["target"] == 1 for p in pairs)} verified positives, '
          f'{sum(p["target"] == 0 for p in pairs)} verified negatives; '
          f'{len(session.feature_cache)} cached lines; representation={representation}', flush=True)
    if args.action == 'cache':
        notebook = notebook_settings()
        baseline = evaluate_settings(session, pairs, truths, representation, notebook)
        _write_json(out / 'baseline_notebook.json', baseline)
        _write_csv(out / 'baseline_notebook_pairs.csv', baseline['pairs'])
        print('Notebook baseline:', baseline['summary'], flush=True)
        return
    storage = args.storage or f'sqlite:///{out / "studies.sqlite3"}'
    signature = _fingerprint(dict(provenance=provenance, objective='positive_iou',
        max_negative_pair_fpr=args.max_negative_pair_fpr, notebook=file_hash('notebooks/model_evaluation.ipynb')))
    notebook = notebook_settings()
    baseline = search_baseline(notebook)
    if args.action == 'tune':
        stage = args.stage
        study = _study(storage, f'{out.name}_stage{stage}', stage, args.seed, signature,
                       args.max_negative_pair_fpr)
        if stage == 1 and not study.trials:
            seed_params = _seed_params(baseline)
            if seed_params is not None:
                study.enqueue_trial(seed_params, user_attrs={'source': 'letter-evidence baseline'})
            else:
                print('Baseline outside Stage-1 distributions; evaluated separately, not enqueued.', flush=True)
        if stage == 2 and not study.trials:
            stage1 = _study(storage, f'{out.name}_stage1', 1, args.seed, signature,
                            args.max_negative_pair_fpr)
            for source in feasible_trials([stage1], args.max_negative_pair_fpr)[:5]:
                params = _seed_params(source.user_attrs['effective_settings'])
                if params is not None:
                    study.enqueue_trial(params, user_attrs={'source': f'Stage-1 trial {source.number}; '
                        'new conditional parameters retain baseline only where compatible'})
        target = args.stage1_trials if stage == 1 else args.stage2_trials
        if target < 1:
            raise ValueError('Trial count must be positive')
        def objective(trial):
            requested = None
            try:
                requested = sample_settings(trial, stage, baseline)
                result = evaluate_settings(session, pairs, truths, representation, requested)
                summary = result['summary']
                for key, value in summary.items():
                    trial.set_user_attr(key, _jsonable(value))
                trial.set_user_attr('requested_settings', requested)
                trial.set_user_attr('effective_settings', result['effective_settings'])
                trial.set_user_attr('population_ids', [p['sample_id'] for p in pairs])
                if len(result['pairs']) != len(pairs):
                    raise RuntimeError('Incomplete validation population')
                return summary['positive_iou']
            except Exception as error:
                trial.set_user_attr('actual_error', f'{type(error).__name__}: {error}')
                trial.set_user_attr('requested_settings', requested)
                raise
        remaining = max(0, target - len(study.trials))
        study.optimize(objective, n_trials=remaining, catch=(Exception,), show_progress_bar=True)
        stages = [study] if stage == 1 else [stage1, study]
        _write_csv(out / 'complete_trials.csv', _trial_rows(stages))
        selected = feasible_trials(stages, args.max_negative_pair_fpr)
        if selected:
            winner = selected[0]
            settings = winner.user_attrs['effective_settings']
            _write_json(out / 'best_effective_settings.json', dict(settings=settings,
                study=next(s.study_name for s in stages if winner in s.trials), trial=winner.number,
                validation_positive_iou=winner.value,
                validation_negative_pair_fpr=winner.user_attrs['negative_pair_fpr'],
                validation_negative_masked_area_fraction=winner.user_attrs['negative_masked_area_fraction'],
                signature=signature))
            (out / 'notebook_settings.py').write_text('SETTINGS = ' + repr(settings) + '\n', encoding='utf-8')
            print(f'Best feasible validation IoU={winner.value:.4f}, FPR={winner.user_attrs["negative_pair_fpr"]:.4f}', flush=True)
        else:
            print('NO FEASIBLE TRIAL. Trade-off CSV saved; do not run selected test evaluation.', flush=True)
        return
    if args.action == 'test':
        selected_path = out / 'best_effective_settings.json'
        if not selected_path.is_file():
            raise ValueError('No frozen feasible validation selection; test evaluation is forbidden')
        selected = json.loads(selected_path.read_text(encoding='utf-8'))
        val_provenance = out / 'val_provenance.json'
        if not val_provenance.is_file():
            raise ValueError('Validation provenance missing')
        previous = json.loads(val_provenance.read_text(encoding='utf-8'))
        if previous['checkpoint_sha256'] != session.checkpoint_sha256:
            raise ValueError('Checkpoint differs from frozen validation selection')
        settings = checked_settings(selected['settings'])
        variants = dict(current_notebook=notebook, optuna_selected=settings,
                        no_extension=checked_settings(dict(settings, enable_path_extension=False)),
                        no_merging=checked_settings(dict(settings, enable_region_merging=False)))
        results = {}
        for name, variant in variants.items():
            result = evaluate_settings(session, pairs, truths, representation, variant)
            result['identical_to_selected'] = (name in ('no_extension', 'no_merging') and variant == settings)
            results[name] = result
            _write_json(out / f'test_{name}.json', result)
            _write_csv(out / f'test_{name}_pairs.csv', result['pairs'])
            _write_csv(out / f'test_{name}_lines.csv', result['lines'])
        _write_json(out / 'test_aggregate.json', {name: value['summary'] for name, value in results.items()})
        paper_exports(results, out, args.seed, args.bootstrap_resamples)
        return
    if args.action == 'export':
        names = ('current_notebook', 'optuna_selected', 'no_extension', 'no_merging')
        results = {name: json.loads((out / f'test_{name}.json').read_text(encoding='utf-8')) for name in names}
        paper_exports(results, out, args.seed, args.bootstrap_resamples)


if __name__ == '__main__':
    main()
