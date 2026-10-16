"""Tune a frozen checkpoint's image matcher for mean mask IoU.

Run from the repository root; see docs/matching_parameter_search.md. Defaults to
validation. Explicit training-split calibration is available when validation
masks are unavailable. Never trains, modifies weights, or tunes on test.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image
import optuna
import torch
from tqdm.auto import tqdm

from dataset import file_hash
from dtw import cosine_similarity_matrix
from evaluate import MATCH_VERSION, match_features, region_mask, resolved_match_settings
from evaluation_utils import EvaluationSession


SEARCH_VERSION = 1
OBJECTIVE = 'mean((IoU_A + IoU_B) / 2) over pairs with foreground in both GT masks'
# Coupled dimensions avoid redundant/invalid combinations. In particular, the
# normal support floor stays above the short floor; both similarity modes agree.
SEARCH_SPACE = {
    'similarity_mode': ['cosine', 'letter_evidence'],
    'threshold': dict(low=0.35, high=0.80, step=0.025),
    'contrast_margin': dict(low=0.0, high=0.12, step=0.01),
    'normal_min_windows': dict(low=4, high=7),
    'normal_min_matched_pairs': dict(low=4, high=8),
    'allow_short_regions': [True, False],
    'short_quality': ['relaxed', 'balanced', 'strict'],
    'max_gap': dict(low=0, high=2),
    'gap_open': [0.05, 0.10, 0.20, 0.30],
    'gap_extend_ratio': [0.25, 0.5, 1.0],
    'repeat_penalty': [0.02, 0.05, 0.10, 0.15],
    'max_consecutive_repeats': dict(low=1, high=3),
    'anchor_top_k': dict(low=1, high=4),
    'min_mutual_anchor_fraction_long': [0.10, 0.15, 0.25, 0.40],
    'min_path_density': [None, 0.65, 0.80, 0.90],
    'enable_path_extension': [True, False],
    'extension_max_weak_run': dict(low=0, high=2),
    'enable_region_merging': [True, False],
    'merge_max_gap': dict(low=0, high=2),
    'dominance_min_quality_ratio': [0.50, 0.75, 0.90, 1.0],
    'verify_with_letter_evidence': [True, False],
    'min_mean_letter_evidence': [0.0, 0.10, 0.25],
    'min_letter_positive_fraction': [0.50, 0.60, 0.75],
}
SHORT_PROFILES = {
    'relaxed': dict(cosine=0.68, reward=0.05, density=0.70, anchors=0.25),
    'balanced': dict(cosine=0.72, reward=0.15, density=0.80, anchors=0.40),
    'strict': dict(cosine=0.78, reward=0.25, density=0.90, anchors=0.60),
}
BASELINE_PARAMS = dict(
    similarity_mode='cosine', threshold=0.5, contrast_margin=0.05,
    normal_min_windows=5, normal_min_matched_pairs=5, allow_short_regions=True,
    short_quality='balanced', max_gap=1, gap_open=0.20, gap_extend_ratio=0.25,
    repeat_penalty=0.05, max_consecutive_repeats=3, anchor_top_k=3,
    min_mutual_anchor_fraction_long=0.15, min_path_density=None,
    enable_path_extension=True, extension_max_weak_run=2,
    enable_region_merging=True, merge_max_gap=1, dominance_min_quality_ratio=0.90,
    verify_with_letter_evidence=False,
)


def suggest_settings(trial, max_candidates=512):
    def choice(name):
        return trial.suggest_categorical(name, SEARCH_SPACE[name])

    def integer(name):
        return trial.suggest_int(name, **SEARCH_SPACE[name])

    mode = choice('similarity_mode')
    settings = dict(alignment_mode='local_repeat_dtw', similarity_mode=mode,
                    candidate_similarity_mode=mode, score_mode='background',
                    max_candidates=max_candidates)
    if mode == 'cosine':
        for name in ('threshold', 'contrast_margin'):
            settings[name] = trial.suggest_float(name, **SEARCH_SPACE[name])
    support = integer('normal_min_windows')
    settings.update(normal_min_distinct_a=support, normal_min_distinct_b=support,
                    normal_min_matched_pairs=integer('normal_min_matched_pairs'),
                    allow_short_regions=choice('allow_short_regions'))
    if settings['allow_short_regions']:
        profile = SHORT_PROFILES[choice('short_quality')]
        settings.update(short_min_mean_cosine=profile['cosine'],
                        short_min_median_cosine=profile['cosine'],
                        short_min_mean_reward=profile['reward'],
                        short_min_path_density=profile['density'],
                        short_min_mutual_anchor_fraction=profile['anchors'],
                        min_mutual_anchor_fraction_short=profile['anchors'])
    settings['max_gap'] = integer('max_gap')
    settings['gap_open'] = choice('gap_open')
    settings['gap_extend'] = settings['gap_open'] * choice('gap_extend_ratio')
    for name in ('repeat_penalty', 'min_mutual_anchor_fraction_long',
                 'min_path_density', 'dominance_min_quality_ratio'):
        settings[name] = choice(name)
    for name in ('max_consecutive_repeats', 'anchor_top_k'):
        settings[name] = integer(name)
    settings['enable_path_extension'] = choice('enable_path_extension')
    if settings['enable_path_extension']:
        settings.update(extension_max_weak_run=integer('extension_max_weak_run'),
                        extension_max_internal_gap=settings['max_gap'])
    settings['enable_region_merging'] = choice('enable_region_merging')
    if settings['enable_region_merging']:
        gap = integer('merge_max_gap')
        settings.update(merge_max_gap_a=gap, merge_max_gap_b=gap)
    # Letter rewards already use the evidence. The optional second-stage check
    # is searched only for cosine rewards, and always has active thresholds.
    settings['verify_with_letter_evidence'] = (
        choice('verify_with_letter_evidence') if mode == 'cosine' else False)
    if settings['verify_with_letter_evidence']:
        for name in ('min_mean_letter_evidence', 'min_letter_positive_fraction'):
            settings[name] = choice(name)
    return resolved_match_settings(settings)


def mask_path(pair, index, mask_source):
    side = pair['sides'][index]
    if mask_source == 'folders':
        image = Path(side['image'])
        return image.parent.parent / ('masks' + 'AB'[index]) / (image.stem + '.png')
    value = pair['annotations'][index].get('mask')
    return Path(value) if value else None


def load_masks(pair, mask_source):
    """Actual source-size mask files only; missing files are black, as in the notebook."""
    masks, sources = [], []
    for index, side in enumerate(pair['sides']):
        with Image.open(side['image']) as image:
            shape = (image.height, image.width)
        path = mask_path(pair, index, mask_source)
        exists = path is not None and path.is_file()
        if exists:
            with Image.open(path) as image:
                mask = np.asarray(image.convert('L')) > 0
            if mask.shape != shape:
                raise ValueError(f'Mask/source size mismatch: {path}: {mask.shape} != {shape}')
        else:
            mask = np.zeros(shape, dtype=bool)
        masks.append(mask)
        sources.append(dict(path=str(path) if path else None, exists=exists,
                            sha256=hashlib.sha256(mask.tobytes()).hexdigest()))
    return tuple(masks), sources


@dataclass
class PreparedPair:
    pair: dict
    lines: list
    cosine: np.ndarray
    ground_truth: tuple
    category: str


def category_for_masks(masks):
    foreground = [bool(mask.any()) for mask in masks]
    return 'positive' if all(foreground) else 'negative' if not any(foreground) else 'partial'


def prepare_pairs(session, representation, mask_source, max_pairs=0):
    catalog = []
    for pair in sorted({p['sample_id']: p for p in session.pairs}.values(), key=lambda p: p['sample_id']):
        masks, sources = load_masks(pair, mask_source)
        catalog.append((pair, masks, sources, category_for_masks(masks)))
    available = {name: sum(row[3] == name for row in catalog)
                 for name in ('positive', 'negative', 'partial')}
    if not available['positive']:
        raise ValueError(f'No {session.split} pairs have foreground in BOTH ground-truth masks '
                         f'(available counts: {available}). '
                         'Check saved split membership and the mask paths. No split fallback is used. '
                         'If validation masks are unavailable, --split train explicitly enables '
                         'in-sample calibration; these scores are not held-out validation results.')
    if max_pairs and len(catalog) > max_pairs:
        # Deterministic stratification preserves positives and, where possible,
        # black-mask examples, even when the diagnostic pool is much larger.
        rng = random.Random(session.seed)
        groups = [[row for row in catalog if row[3] == name]
                  for name in ('positive', 'negative', 'partial')]
        for group in groups:
            rng.shuffle(group)
        selected = []
        while len(selected) < max_pairs and any(groups):
            for group in groups:
                if group and len(selected) < max_pairs:
                    selected.append(group.pop())
        catalog = sorted(selected, key=lambda row: row[0]['sample_id'])
    prepared, manifest = [], []
    for pair, masks, sources, category in tqdm(catalog, desc=f'Cache {session.split} features'):
        lines = [session.get_line_features(side) for side in pair['sides']]
        cosine = cosine_similarity_matrix(*(line['features'][representation] for line in lines)).cpu().numpy()
        prepared.append(PreparedPair(pair, lines, cosine, masks, category))
        manifest.append(dict(sample_id=pair['sample_id'], category=category,
                             images=[side['image'] for side in pair['sides']],
                             image_sha256=[file_hash(side['image']) for side in pair['sides']],
                             masks=sources))
    return prepared, manifest, available


def mask_iou(prediction, truth):
    pred = np.asarray(prediction) > 0
    if pred.shape != truth.shape:
        raise ValueError(f'Prediction/GT shape mismatch: {pred.shape} != {truth.shape}')
    union = np.count_nonzero(pred | truth)
    return float(np.count_nonzero(pred & truth) / union) if union else 1.0


def evaluate_settings(session, prepared, representation, settings):
    """Same matcher and mask builder as the notebook; GT enters only scoring."""
    rows = []
    for cached in prepared:
        lines = cached.lines
        match = match_features(*(line['features'][representation] for line in lines),
                               *(line['physical'] for line in lines),
                               text_encoder=session.text_encoder, config=session.config,
                               precomputed_cosine=cached.cosine, **settings)
        masks = [region_mask(match['regions'], index, line['geometry'],
                             session.config.window_width, session.config.window_stride)
                 for index, line in enumerate(lines)]
        ious = [mask_iou(mask, truth) for mask, truth in zip(masks, cached.ground_truth)]
        rows.append(dict(sample_id=cached.pair['sample_id'], category=cached.category,
                         a_iou=ious[0], b_iou=ious[1], pair_mean_iou=sum(ious) / 2,
                         predicted_nonempty=any(mask.any() for mask in masks),
                         mask_coverage_a=float((masks[0] > 0).mean()),
                         mask_coverage_b=float((masks[1] > 0).mean()),
                         region_count=len(match['regions']),
                         candidate_limit_reached=bool(match['candidate_limit_reached'])))
    positive = [row for row in rows if row['category'] == 'positive']
    negative = [row for row in rows if row['category'] == 'negative']
    if not positive:
        raise ValueError('Mean positive mask IoU requires at least one positive pair')
    summary = dict(mean_mask_iou=float(np.mean([row['pair_mean_iou'] for row in positive])),
                   a_mean_iou=float(np.mean([row['a_iou'] for row in positive])),
                   b_mean_iou=float(np.mean([row['b_iou'] for row in positive])),
                   positive_pairs=len(positive), negative_pairs=len(negative),
                   partial_pairs=len(rows) - len(positive) - len(negative),
                   negative_false_positive_rate=(float(np.mean([row['predicted_nonempty'] for row in negative]))
                                                 if negative else None),
                   candidate_limit_reached_pairs=sum(row['candidate_limit_reached'] for row in rows))
    return summary, rows


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_study_manifest(study, manifest, resume):
    previous = study.user_attrs.get('run_manifest')
    if previous is not None and previous != manifest:
        raise ValueError('This study uses different data, masks, checkpoint, code or search settings. '
                         'Use a new --output directory or --study-name.')
    if previous is None:
        if study.trials:
            raise ValueError('Existing study has trials without matching provenance; use a new study.')
        study.set_user_attr('run_manifest', manifest)
    elif not resume:
        raise ValueError('Study already exists. Use --resume to add trials or choose a new output directory.')


def export_results(study, output):
    trials = study.get_trials(deepcopy=False)
    write_csv(output / 'trials.csv', [dict(
        trial=trial.number, state=trial.state.name, mean_mask_iou=trial.value,
        duration_seconds=trial.duration.total_seconds() if trial.duration else None,
        **{key: value for key, value in trial.user_attrs.get('metrics', {}).items() if key != 'mean_mask_iou'},
        **{'param_' + key: value for key, value in trial.params.items()}) for trial in trials])
    complete = [trial for trial in trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not complete:
        return
    best = study.best_trial
    write_json(output / 'best_settings.json', best.user_attrs['settings'])
    write_json(output / 'best_result.json', dict(trial=best.number, objective=OBJECTIVE,
               value=best.value, metrics=best.user_attrs['metrics'], params=best.params,
               settings=best.user_attrs['settings'], run_manifest=study.user_attrs['run_manifest']))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--checkpoint', default='Weights/phase4_typed_local03_sum_nogate/checkpoint_best.pt')
    parser.add_argument('--dataset', default='DataSet/ArabicData')
    parser.add_argument('--output', default='Results/Evaluation/matching_optuna')
    parser.add_argument('--study-name', default='validation-mask-iou')
    parser.add_argument('--split', choices=('val', 'train'), default='val',
                        help='val is preferred; train is explicit in-sample calibration. Test is never searched')
    parser.add_argument('--n-trials', type=int, default=200, help='Additional trials, including queued baseline configurations')
    parser.add_argument('--timeout', type=float, help='Stop starting trials after this many seconds; caching is excluded')
    parser.add_argument('--device', default='auto', help='auto, cpu, mps or cuda; used for feature extraction')
    parser.add_argument('--representation', choices=('local', 'context', 'fused'), default='fused')
    parser.add_argument('--pair-source', choices=('auto', 'dataset', 'folders'), default='auto')
    parser.add_argument('--mask-source', choices=('auto', 'folders', 'manifest'), default='auto')
    parser.add_argument('--crop', choices=('checkpoint', 'auto', 'xml', 'none', 'content'), default='content')
    parser.add_argument('--negative-pairs', type=int, default=100, help='Seeded folder comparison pool; diagnostic only')
    parser.add_argument('--max-pairs', type=int, default=0, help='0 means all eligible pairs; positive/negative stratification otherwise')
    parser.add_argument('--max-candidates', type=int, default=512)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', action='store_true', help='Add trials to a study with identical provenance')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.n_trials < 1 or args.max_candidates < 1 or args.max_pairs < 0 or args.negative_pairs < 0:
        raise ValueError('Trials/candidates must be positive and pair limits nonnegative')
    if args.timeout is not None and (not np.isfinite(args.timeout) or args.timeout <= 0):
        raise ValueError('--timeout must be finite and positive')
    output = Path(args.output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Replay saved membership exactly. No test tuning or automatic split fallback.
    from train import resolve_device
    session = EvaluationSession(args.checkpoint, args.dataset, split=args.split,
        device=str(resolve_device(args.device)), seed=args.seed, pair_source=args.pair_source,
        construct_negatives=args.negative_pairs > 0, negative_count=args.negative_pairs,
        crop_override=None if args.crop == 'checkpoint' else args.crop)
    mask_source = (('folders' if session.pair_source == 'folders' else 'manifest')
                   if args.mask_source == 'auto' else args.mask_source)
    prepared, pairs, available = prepare_pairs(session, args.representation, mask_source, args.max_pairs)
    manifest = dict(search_version=SEARCH_VERSION, objective=OBJECTIVE,
        checkpoint=str(session.checkpoint), checkpoint_sha256=session.checkpoint_sha256,
        dataset=str(session.dataset_path), split=args.split, in_sample=args.split == 'train',
        split_manifest_sha256=session.saved.get('split_manifest_sha256'),
        representation=args.representation, config=asdict(session.config), seed=args.seed,
        device=str(session.device), optuna_version=optuna.__version__,
        torch_version=str(torch.__version__), numpy_version=np.__version__,
        pair_source=session.pair_source, mask_source=mask_source,
        missing_mask_policy='black', max_candidates=args.max_candidates,
        max_pairs=args.max_pairs, negative_pairs=args.negative_pairs,
        matcher_version=MATCH_VERSION, search_space=SEARCH_SPACE, short_profiles=SHORT_PROFILES,
        source_sha256={name: file_hash(ROOT / name) for name in
                       ('evaluate.py', 'alignment_candidates.py', 'dtw.py', 'evaluation_utils.py',
                        'dataset.py', 'dataloader.py', 'model.py', 'cnn_encoder.py',
                        'transformer_encoder.py', 'text_embedding.py', 'train.py',
                        'scripts/eval/tune_matching.py')},
        available_pair_counts=available, pairs=pairs)
    # JSON-normalize tuples from model config for comparison after SQLite storage.
    manifest = json.loads(json.dumps(manifest))
    study = optuna.create_study(study_name=args.study_name, direction='maximize',
        storage='sqlite:///' + str(output / 'study.sqlite3'), load_if_exists=args.resume,
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=25),
        pruner=optuna.pruners.NopPruner())
    validate_study_manifest(study, manifest, args.resume)
    if not study.trials:
        baseline = dict(BASELINE_PARAMS)
        for params in (baseline,
                       dict(baseline, verify_with_letter_evidence=True,
                            min_mean_letter_evidence=0.0, min_letter_positive_fraction=0.60),
                       dict(baseline, similarity_mode='letter_evidence')):
            study.enqueue_trial(params)
    write_json(output / 'run_manifest.json', manifest)
    if args.split == 'train':
        print('Training-split calibration: scores are in-sample, not held-out validation results.', flush=True)
    print(f'{args.split} pairs: {len(prepared)}; positive={sum(p.category == "positive" for p in prepared)}; '
          f'objective: {OBJECTIVE}', flush=True)

    def objective(trial):
        settings = suggest_settings(trial, args.max_candidates)
        trial.set_user_attr('settings', settings)
        start = time.monotonic()
        metrics, rows = evaluate_settings(session, prepared, args.representation, settings)
        trial.set_user_attr('metrics', metrics)
        write_csv(output / f'trial_{trial.number:05d}_pairs.csv', rows)
        print(f'Trial {trial.number}: mean IoU={metrics["mean_mask_iou"]:.5f} '
              f'(A={metrics["a_mean_iou"]:.5f}, B={metrics["b_mean_iou"]:.5f}); '
              f'negative FP rate={metrics["negative_false_positive_rate"]}; '
              f'{time.monotonic() - start:.1f}s', flush=True)
        return metrics['mean_mask_iou']

    def callback(study, trial):
        export_results(study, output)

    try:
        study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout,
                       n_jobs=1, callbacks=[callback])
    finally:
        # Persist usable completed results even after an interrupted trial.
        export_results(study, output)
    if any(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials):
        best = study.best_trial
        print(f'Best {args.split} mean mask IoU: {best.value:.5f} (trial {best.number})', flush=True)
        print(f'Settings: {output / "best_settings.json"}', flush=True)
    else:
        print('No trial completed before the timeout; resume to continue.', flush=True)
    return study


if __name__ == '__main__':
    try:
        main()
    except ValueError as exc:
        raise SystemExit(f'Error: {exc}') from None
