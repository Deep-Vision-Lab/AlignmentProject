"""Notebook diagnostics built on the standalone checkpoint and image matcher.

Labels/annotations select and score examples; they NEVER enter visual prediction.
All coordinates are source-image pixels unless explicitly labelled model/logical.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import uuid

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm

from dataset import AlignmentDataset, file_hash, prepare_image
from dtw import cosine_similarity_matrix, hard_dtw_path, letter_cost_matrix
from evaluate import region_mask, score_mask, shared_regions, source_interval, smith_waterman_affine
from model import extract_windows
from train import build_loaders, load_checkpoint


POSITIVE_LABELS = {'high_match', 'medium_match', 'low_match'}
NEGATIVE_LABELS = {'no_shared_content'}
REPRESENTATIONS = ('local', 'context', 'fused')
DEFAULT_MATCH_SETTINGS = dict(threshold=.6, score_mode='background', contrast_margin=.05,
                              min_windows=5, max_gap=1, gap_open=.2, gap_extend=.05)


def _side_key(side):
    return (str(Path(side['image']).resolve()), str(Path(side['text']).resolve()),
            tuple(side.get('bbox') or ()))


def _image_key(side):
    return str(Path(side['image']).resolve()), tuple(side.get('bbox') or ())


def _distribution(values):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, mean=None, std=None, median=None, q25=None, q75=None)
    return dict(count=len(values), mean=float(values.mean()), std=float(values.std()),
                median=float(np.median(values)), q25=float(np.quantile(values, .25)),
                q75=float(np.quantile(values, .75)))


def sample_random_pairs(records, n, aligned=True, seed=42):
    if n < 0:
        raise ValueError('n must be nonnegative')
    choices = sorted((r for r in records if r['target'] == int(aligned)), key=lambda r: r['sample_id'])
    return random.Random(seed).sample(choices, min(n, len(choices)))


def _constructed_negatives(pairs, line_records, n, seed):
    """Cross-group, test-only presumed negatives; never certified ground truth."""
    lines = sorted(line_records, key=lambda r: _side_key(r['side']))
    forbidden = {frozenset(_image_key(s) for s in p['sides']) for p in pairs if p['target'] == 1}
    candidates = []
    # Reservoir sampling bounds memory even for a large test population.
    rng, seen = random.Random(seed), 0
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            keys = frozenset((_image_key(a['side']), _image_key(b['side'])))
            if (a['group_id'] == b['group_id'] or len(keys) != 2 or keys in forbidden
                    or a['pair_ids'] & b['pair_ids'] or a['anchors'] & b['anchors']):
                continue
            seen += 1
            if len(candidates) < n:
                candidates.append((a, b))
            else:
                slot = rng.randrange(seen)
                if slot < n:
                    candidates[slot] = (a, b)
    result = []
    for a, b in candidates:
        sides = [a['side'], b['side']]
        identity = hashlib.sha256(json.dumps([_side_key(s) for s in sides]).encode()).hexdigest()[:20]
        result.append(dict(sample_id='constructed:' + identity, sides=sides, target=0,
                           label='constructed_negative', constructed_negative=True,
                           group_ids=[a['group_id'], b['group_id']], annotations=[{}, {}],
                           annotation_provenance='none', anchor_id=''))
    return result


class EvaluationSession:
    """Load once; cache CPU feature tensors; retain the checkpoint's exact split."""
    def __init__(self, checkpoint, dataset, split='test', device='cuda', seed=42,
                 settings=None, construct_negatives=True, negative_count=None,
                 annotation_manifest=None):
        if split not in ('train', 'val', 'test'):
            raise ValueError('split must be train, val, or test')
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.dataset_path = Path(dataset).expanduser().resolve()
        self.device = torch.device('cpu' if str(device).startswith('cuda') and not torch.cuda.is_available()
                                   else ('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else device)
        self.model, self.text_encoder, self.config, self.saved = load_checkpoint(self.checkpoint, self.device)
        self.model.eval()
        self.text_encoder.eval()
        self.split, self.seed = split, seed
        self.settings = dict(DEFAULT_MATCH_SETTINGS, **(settings or {}))
        # Saved IDs are mandatory: no random-split fallback is permitted here.
        if not self.saved.get('split_ids'):
            raise ValueError('Checkpoint has no saved split IDs; evaluation cannot reconstruct membership')
        loaders = build_loaders(self.dataset_path, replace(self.config, augmentation=False, num_workers=0),
                                self.saved['split_ids'])
        self.views = {name: loader.dataset for name, loader in zip(('train', 'val', 'test'), loaders)}
        self.view = self.views[split]
        self.split_sizes = {name: len(view) for name, view in self.views.items()}
        image_sets = [{s['image'] for r in view.records for s in r['sides']} for view in self.views.values()]
        if any(image_sets[i] & image_sets[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError('Data contract failure: source image reused across saved splits')
        self.allowed = {_side_key(s): dict(side=s, group_id=r['group_id'], pair_ids=set(),
                                           anchors={r['anchor_id']} if r['anchor_id'] else set())
                        for r in self.view.records for s in r['sides']}
        # Pair view is a catalog only. Its generated groups/splits are NEVER used.
        catalog = AlignmentDataset(self.dataset_path, self.view.dataset_type, paired=True,
                                   image_size=self.view.image_size, grayscale=self.config.grayscale,
                                   crop=self.config.crop, binarize=self.config.binarize)
        self.pairs, self.excluded_pairs, self.unknown_labels = [], 0, {}
        for r in catalog.records:
            # Preserve known page-pair/anchor relationships even when the other
            # side of an individual manifest row lies outside this split.
            pair_group = (r['sample_id'] if self.view.dataset_type == 'synthetic'
                          else r['sample_id'].rsplit(':', 1)[0])
            for s in r['sides']:
                if _side_key(s) in self.allowed:
                    self.allowed[_side_key(s)]['pair_ids'].add(pair_group)
            if not all(_side_key(s) in self.allowed for s in r['sides']):
                self.excluded_pairs += 1
                continue
            label = r['label']
            target = 1 if label in POSITIVE_LABELS else 0 if label in NEGATIVE_LABELS else None
            # Synthetic filenames alone do not certify shared content.
            if target is None:
                self.unknown_labels[str(label)] = self.unknown_labels.get(str(label), 0) + 1
            groups = [self.allowed[_side_key(s)]['group_id'] for s in r['sides']]
            self.pairs.append(dict(sample_id=r['sample_id'], sides=r['sides'], target=target,
                                   label=label, constructed_negative=False, group_ids=groups,
                                   anchor_id=r['anchor_id'], annotations=[{'mask': s['mask']} for s in r['sides']],
                                   annotation_provenance='manifest alignment_mask_path (if present)'))
        if annotation_manifest:
            self._attach_annotations(annotation_manifest)
        self.provided_negative_count = sum(p['target'] == 0 for p in self.pairs)
        if construct_negatives and not self.provided_negative_count and split == 'test':
            n = negative_count if negative_count is not None else max(len(self.allowed), sum(p['target'] == 1 for p in self.pairs))
            if n < 0:
                raise ValueError('negative_count must be nonnegative')
            self.pairs.extend(_constructed_negatives(self.pairs, list(self.allowed.values()), n, seed))
        self.feature_cache = {}
        self.metric_cache = {}
        self.checkpoint_sha256 = file_hash(self.checkpoint)

    def _attach_annotations(self, manifest):
        """Read only pair-specific supplied annotation paths/intervals, never OCR."""
        root = self.view.root
        def resolve(value):
            path = Path(value)
            return str((path if path.is_absolute() else root / path).resolve())
        by_images = {}
        for line in Path(manifest).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            sides = [row.get('A') or row['real'], row.get('B') or row['positive']]
            key = tuple(resolve(s.get('line_image_path') or s['image']) for s in sides)
            annotations = []
            meta = row.get('alignment_mask_meta') or {}
            for name, side in zip(('A', 'B'), sides):
                mask = side.get('alignment_mask_path') or side.get('mask')
                detail = meta.get(name, {})
                annotations.append(dict(mask=resolve(mask) if mask else None,
                                        intervals=detail.get('intervals_x'), source_size=detail.get('mask_size')))
            if key in by_images and by_images[key][0] != annotations:
                raise ValueError(f'Conflicting alignment annotations for {key}')
            by_images[key] = (annotations, meta.get('method', 'supplied alignment annotation'))
        for pair in self.pairs:
            key = tuple(s['image'] for s in pair['sides'])
            if key in by_images:
                pair['annotations'], pair['annotation_provenance'] = by_images[key]

    def summary(self):
        c, saved = self.config, self.saved
        checkpoint = dict(Path=str(self.checkpoint), SHA256=self.checkpoint_sha256, Epoch=saved['epoch'],
                          Best_validation_loss=saved['best_val'], Architecture=saved['architecture_family'],
                          CNN=c.cnn_type, Transformer=c.transformer_type, Embedding_dimension=c.embedding_dim,
                          Transformer_layers=len(self.model.transformer.layers), Heads=self.model.transformer.heads,
                          Window_width=c.window_width, Window_stride=c.window_stride,
                          Image_size=f'{c.image_height}x{c.image_width}', RTL=c.rtl,
                          Parameter_count=sum(p.numel() for p in self.model.parameters()), Device=str(self.device),
                          Crop=c.crop, Grayscale=c.grayscale, Binarize=c.binarize,
                          Split_manifest_SHA256=saved['split_manifest_sha256'])
        data = dict(Path=str(self.dataset_path), Dataset_type=self.view.dataset_type, Split=self.split,
                    Number_of_samples=len(self.view), Eligible_pairs=len(self.pairs),
                    Number_of_aligned_pairs=sum(p['target'] == 1 for p in self.pairs),
                    Number_of_unaligned_pairs=sum(p['target'] == 0 for p in self.pairs),
                    Constructed_negatives=sum(p['constructed_negative'] for p in self.pairs),
                    Unknown_labels=self.unknown_labels, Excluded_pairs_outside_split=self.excluded_pairs,
                    Ground_truth_masks_available=sum(bool(a.get('mask')) for p in self.pairs for a in p['annotations']),
                    Ground_truth_aligned_intervals_available=sum(a.get('intervals') is not None for p in self.pairs for a in p['annotations']),
                    Saved_split_sizes=self.split_sizes)
        for title, values in (('CHECKPOINT', checkpoint), ('EVALUATION DATASET', data)):
            print('=' * 60 + '\n' + title + '\n' + '=' * 60)
            for name, value in values.items():
                print(f'{name.replace("_", " ")}: {value}')
        print('Line/XML crop boxes are NOT alignment ground truth. Pair labels are manifest labels, not character accuracy.')
        return dict(checkpoint=checkpoint, dataset=data)

    def get_line_features(self, side):
        if _side_key(side) not in self.allowed:
            raise ValueError(f'Line outside saved {self.split} membership: {side["image"]}')
        key = _image_key(side)
        if key not in self.feature_cache:
            c = self.config
            _, tensor, geometry = prepare_image(side['image'], (c.image_height, c.image_width),
                                                c.grayscale, c.crop, bbox=side.get('bbox'),
                                                augment=False, binarize=c.binarize)
            with torch.no_grad():
                output = self.model(tensor[None].to(self.device))
            valid = output['token_valid'][0]
            features = {name: output[name][0][valid].detach().float().cpu() for name in REPRESENTATIONS}
            if any(not torch.isfinite(v).all() for v in features.values()):
                raise ValueError(f'NaN/Inf features: {side["image"]}')
            self.feature_cache[key] = dict(features=features, geometry=geometry,
                physical=output['physical_window_indices'][0][valid].cpu().numpy(),
                logical=torch.arange(len(valid), device=valid.device)[valid].cpu().numpy())
        return self.feature_cache[key]

    def predict_pair(self, pair, representation='fused'):
        if representation not in REPRESENTATIONS:
            raise ValueError(f'Unknown representation: {representation}')
        lines = [self.get_line_features(s) for s in pair['sides']]
        cosine = cosine_similarity_matrix(*(r['features'][representation] for r in lines)).numpy()
        match = shared_regions(cosine, *(r['physical'] for r in lines), **self.settings)
        masks = [region_mask(match['regions'], side, r['geometry'], self.config.window_width,
                             self.config.window_stride) for side, r in enumerate(lines)]
        return dict(pair=pair, representation=representation, lines=lines, cosine=cosine,
                    match=match, masks=masks, settings=dict(self.settings), split=self.split)

    def evaluate_pair(self, pair, representation='fused'):
        result = self.predict_pair(pair, representation)
        # Ground truth is read ONLY AFTER predictions and masks have been fixed.
        result['metrics'], result['ground_truth'] = compute_pair_metrics(result, self.config)
        return result

    def metrics(self, pair, representation='fused'):
        key = (pair['sample_id'], representation, json.dumps(self.settings, sort_keys=True))
        if key not in self.metric_cache:
            self.metric_cache[key] = self.evaluate_pair(pair, representation)['metrics']
        return self.metric_cache[key]


def _load_annotation(annotation, shape):
    if annotation.get('mask'):
        with Image.open(annotation['mask']) as image:
            gt = np.asarray(image.convert('L')) >= 128
        if gt.shape != shape:
            raise ValueError('Ground-truth/source geometry mismatch; resizing is forbidden')
        return gt
    if annotation.get('intervals') is not None:
        h, w = shape
        if annotation.get('source_size') != [w, h]:
            raise ValueError('Aligned intervals require exact source_size [width,height]')
        mask = np.zeros(shape, dtype=bool)
        for a, b in annotation['intervals']:
            if not (math.isfinite(a) and math.isfinite(b) and 0 <= a < b <= w):
                raise ValueError('Aligned interval outside source image')
            mask[:, math.floor(a):math.ceil(b)] = True
        return mask
    return None


def _binary_metrics(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    tp, union = np.logical_and(pred, gt).sum(), np.logical_or(pred, gt).sum()
    precision = float(tp / pred.sum()) if pred.any() else 0.
    recall = float(tp / gt.sum()) if gt.any() else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return dict(iou=float(tp / union) if union else 1., precision=precision, recall=recall, f1=f1)


def localization_metrics(mask, annotation, line, config, regions, side):
    gt = _load_annotation(annotation, mask.shape)
    if gt is None:
        return dict(status='unavailable', reason='no pair-specific localization annotation'), None
    # Reuse the public source-mask scorer when a file is supplied.
    pixel = (score_mask(mask, annotation['mask']) if annotation.get('mask') else _binary_metrics(mask > 0, gt))
    metrics = {k: pixel[k] for k in ('iou', 'precision', 'recall', 'f1')}
    metrics.update(status='available', dice=metrics['f1'])
    px, gx = np.flatnonzero((mask > 0).any(0)), np.flatnonzero(gt.any(0))
    if len(px) and len(gx):
        left, right = abs(int(px[0]) - int(gx[0])), abs(int(px[-1]) - int(gx[-1]))
        metrics.update(left_boundary_error_px=left, right_boundary_error_px=right,
                       normalized_boundary_error=(left + right) / (2 * mask.shape[1]))
    # Footprint occupancy >=50% defines GT windows. Prediction uses selected or
    # explicitly tiny-gap-filled windows, not overlap with adjacent footprints.
    selected = {p for r in regions for p in r['supported_physical'][side] + r['filled_physical'][side]}
    ground_windows = []
    for physical in line['physical']:
        a, b = source_interval(int(physical), line['geometry'], config.window_width, config.window_stride)
        occupancy = gt[:, max(0, math.floor(a)):min(mask.shape[1], math.ceil(b))]
        ground_windows.append(bool(occupancy.size and occupancy.mean() >= .5))
    metrics.update({'window_' + k: v for k, v in _binary_metrics(
        [int(p) in selected for p in line['physical']], ground_windows).items()})
    metrics['ground_truth_windows'] = sum(ground_windows)
    metrics['boundary_definition'] = 'outer envelope; unavailable if either mask is empty (not per-region accuracy)'
    return metrics, gt


def compute_pair_metrics(result, config):
    pair, cosine, match = result['pair'], result['cosine'], result['match']
    pairs = [tuple(p) for r in match['regions'] for p in r['pairs']]
    values = np.array([cosine[i, j] for i, j in pairs])
    off = np.ones(cosine.shape, bool)
    for i, j in pairs:
        off[i, j] = False
    background = float(cosine[off].mean()) if off.any() else None
    metrics = dict(sample_id=pair['sample_id'], label=pair['label'], target=pair['target'],
                   split=result['split'], image_a=pair['sides'][0]['image'], image_b=pair['sides'][1]['image'],
                   annotation_provenance=pair['annotation_provenance'],
                   constructed_negative=pair['constructed_negative'], representation=result['representation'],
                   # Existing accepted-region score includes rewards and affine gap costs.
                   pair_score=max((r['score'] for r in match['regions']), default=0.),
                   accepted_region_score_sum=sum(r['score'] for r in match['regions']),
                   max_local_alignment_score=smith_waterman_affine(match['rewards'],
                       result['settings']['gap_open'], result['settings']['gap_extend'])[1],
                   path_length=len(pairs), region_count=len(match['regions']),
                   path_cosine_mean=float(values.mean()) if len(values) else None,
                   path_cosine_median=float(np.median(values)) if len(values) else None,
                   path_cosine_min=float(values.min()) if len(values) else None,
                   maximum_similarity=float(cosine.max()), matrix_cosine_mean=float(cosine.mean()),
                   off_path_cosine_mean=background,
                   similarity_separation=float(values.mean()) - background if len(values) and background is not None else None)
    deltas = [(b[0]-a[0], b[1]-a[1]) for r in match['regions'] for a, b in zip(r['pairs'], r['pairs'][1:])]
    metrics.update(horizontal_skipped_windows=sum(max(0, j-1) for i, j in deltas),
                   vertical_skipped_windows=sum(max(0, i-1) for i, j in deltas),
                   largest_anchor_jump=max((max(d) for d in deltas), default=0),
                   internal_discontinuities=sum(i > 1 or j > 1 for i, j in deltas),
                   monotonic=all(i > 0 and j > 0 for i, j in deltas),
                   separation_between_regions=max(0, len(match['regions']) - 1))
    ground_truth = []
    for side, name in enumerate(('a', 'b')):
        line, mask = result['lines'][side], result['masks'][side]
        metrics[f'mask_coverage_{name}'] = float((mask > 0).mean())
        ids = sorted({int(line['physical'][p[side]]) for p in pairs})
        metrics[f'matched_windows_{name}'] = len(ids)
        metrics[f'total_windows_{name}'] = len(line['physical'])
        metrics[f'logical_ranges_{name}'] = [r['logical_ranges'][side] for r in match['regions']]
        metrics[f'pixel_ranges_{name}'] = [[min(source_interval(p, line['geometry'], config.window_width, config.window_stride)[0]
                                               for p in r['supported_physical'][side]),
                                            max(source_interval(p, line['geometry'], config.window_width, config.window_stride)[1]
                                               for p in r['supported_physical'][side])] for r in match['regions']]
        spatial, gt = localization_metrics(mask, pair['annotations'][side], line, config, match['regions'], side)
        ground_truth.append(gt)
        metrics.update({f'{name}_{k}': v for k, v in spatial.items()})
    for metric in ('iou', 'dice'):
        if all(f'{s}_{metric}' in metrics for s in ('a', 'b')):
            metrics['pair_mean_' + metric] = (metrics['a_' + metric] + metrics['b_' + metric]) / 2
    return metrics, ground_truth


def evaluate_population(session, representation='fused', max_pairs=0):
    if max_pairs < 0:
        raise ValueError('max_pairs must be >= 0')
    pairs = session.pairs if max_pairs == 0 else random.Random(session.seed).sample(
        session.pairs, min(max_pairs, len(session.pairs)))
    rows, distributions = [], {k: [] for k in ('positive_path', 'negative_matrix', 'off_path')}
    # Matrices are small; retain distributions only, not source images/graphs.
    for pair in tqdm(pairs, desc=f'{session.split} pairs ({representation})'):
        result = session.evaluate_pair(pair, representation)
        rows.append(result['metrics'])
        key = (pair['sample_id'], representation, json.dumps(session.settings, sort_keys=True))
        session.metric_cache[key] = rows[-1]
        c = result['cosine']
        path = [p for r in result['match']['regions'] for p in r['pairs']]
        off = np.ones(c.shape, bool)
        for i, j in path:
            off[i, j] = False
        if pair['target'] == 1:
            distributions['positive_path'].extend(float(c[i, j]) for i, j in path)
        if pair['target'] == 0:
            distributions['negative_matrix'].extend(c.ravel().tolist())
        distributions['off_path'].extend(c[off].tolist())
    return dict(rows=rows, distributions=distributions,
                distribution_stats={k: _distribution(v) for k, v in distributions.items()},
                population='all eligible saved-split pairs' if not max_pairs else 'explicit random pair subset',
                available_pairs=len(session.pairs), evaluated_pairs=len(rows), split=session.split,
                selected_pair_ids=[p['sample_id'] for p in pairs],
                evaluated_unique_lines=len({_image_key(s) for p in pairs for s in p['sides']}),
                saved_split_unique_lines=len(session.allowed))


def discrimination_metrics(rows, include_constructed=False, decision_threshold=None):
    selected = [r for r in rows if r['target'] in (0, 1) and (include_constructed or not r['constructed_negative'])]
    labels = [r['target'] for r in selected]
    if len(set(labels)) < 2:
        return dict(status='unavailable', reason='both positive and negative labels required', count=len(selected))
    from sklearn.metrics import average_precision_score, roc_auc_score
    scores = [r['pair_score'] for r in selected]
    result = dict(status='available', count=len(selected), roc_auc=float(roc_auc_score(labels, scores)),
                  average_precision=float(average_precision_score(labels, scores)),
                  label_source='manifest plus presumed constructed negatives' if include_constructed else 'manifest labels only')
    if decision_threshold is not None:
        result.update(threshold=float(decision_threshold), **_binary_metrics(np.array(scores) >= decision_threshold, labels))
    return result


def retrieval_metrics(session, representation='fused', negatives=20, max_queries=0):
    """Multiple relevant B lines; distractors require known negatives or other groups.

    Stable ties use pessimistic rank: every tied irrelevant candidate precedes the
    first relevant candidate. All-zero/collapsed scores cannot win by list order.
    """
    if session.split != 'test':
        return dict(status='unavailable', reason='retrieval is restricted to the saved test split')
    if negatives < 1 or max_queries < 0:
        raise ValueError('Need positive distractor count and nonnegative query cap')
    positives = [p for p in session.pairs if p['target'] == 1]
    by_a, candidates, explicit_negatives = {}, {}, set()
    for pair in session.pairs:
        a, b = map(_side_key, pair['sides'])
        candidates[b] = pair['sides'][1]
        if pair['target'] == 0 and not pair['constructed_negative']:
            explicit_negatives.add((a, b))
    for pair in positives:
        by_a.setdefault(_side_key(pair['sides'][0]), []).append(pair)
    queries = sorted(by_a)
    rng = random.Random(session.seed)
    if max_queries:
        queries = rng.sample(queries, min(max_queries, len(queries)))
    rows = []
    for a in tqdm(queries, desc='Test retrieval'):
        relevant = {_side_key(p['sides'][1]) for p in by_a[a]}
        line_a = by_a[a][0]['sides'][0]
        pool = [b for b in sorted(candidates) if b not in relevant and b != a and
                ((a, b) in explicit_negatives or
                 (session.allowed[a]['group_id'] != session.allowed[b]['group_id']
                  and not session.allowed[a]['pair_ids'] & session.allowed[b]['pair_ids']
                  and not session.allowed[a]['anchors'] & session.allowed[b]['anchors']))]
        selected = rng.sample(pool, min(negatives, len(pool)))
        if not selected:
            continue
        scores = {}
        for b in sorted(relevant) + selected:
            pair = dict(sides=[line_a, candidates[b]])
            prediction = session.predict_pair(pair, representation)
            scores[b] = max((r['score'] for r in prediction['match']['regions']), default=0.)
        best = max(scores[b] for b in relevant)
        rank = 1 + sum(scores[b] >= best for b in selected)
        rows.append(dict(query_image=line_a['image'], candidate_images=[candidates[b]['image'] for b in scores],
                         relevant_images=[candidates[b]['image'] for b in sorted(relevant)], rank=rank,
                         candidates=len(scores), relevant_count=len(relevant), top1=rank <= 1,
                         top5=rank <= 5 if len(scores) >= 5 else None,
                         reciprocal_rank=1. / rank, presumed_distractors=sum((a, b) not in explicit_negatives for b in selected)))
    if not rows:
        return dict(status='unavailable', reason='no queries with both known positives and eligible distractors')
    top5 = [r['top5'] for r in rows if r['top5'] is not None]
    return dict(status='available', queries=len(rows), top1=float(np.mean([r['top1'] for r in rows])),
                top5=float(np.mean(top5)) if top5 else None, top5_eligible=len(top5),
                mrr=float(np.mean([r['reciprocal_rank'] for r in rows])), rows=rows,
                candidate_protocol=f'all known relevant partners + at most {negatives} seeded distractors',
                population='all eligible test queries' if not max_queries else 'explicit test query subset',
                tie_policy='pessimistic; tied irrelevant candidates rank ahead',
                limitation='Cross-group distractors are presumed unrelated; unannotated true matches may exist.')


def feature_diagnostics(session, max_lines=100, max_windows=1024):
    if max_windows < 2 or max_lines < 1:
        raise ValueError('Diagnostics require >=1 line and >=2 windows')
    lines = sorted(session.allowed.values(), key=lambda r: _side_key(r['side']))
    selected = random.Random(session.seed).sample(lines, min(max_lines, len(lines)))
    report = {}
    for name in REPRESENTATIONS:
        values = [session.get_line_features(r['side'])['features'][name] for r in selected]
        if not values:
            report[name] = dict(status='unavailable', reason='empty split')
            continue
        x = torch.cat(values)
        generator = torch.Generator().manual_seed(session.seed)
        x = x[torch.randperm(len(x), generator=generator)[:max_windows]]
        norms = x.norm(dim=-1).numpy()
        c = cosine_similarity_matrix(x, x).numpy()
        pairwise = c[np.triu_indices(len(x), 1)]
        feature_std = x.std(dim=0, unbiased=False).mean().item()
        warnings = []
        if name == 'fused' and not np.allclose(norms, 1., atol=1e-4):
            warnings.append('Fused vectors are not unit norm')
        if feature_std < 1e-5:
            warnings.append('Near-zero feature variance; inspect for constant representations')
        if len(pairwise) and pairwise.mean() > .98:
            warnings.append('Very high pairwise cosine; inspect unrelated lines before diagnosing collapse')
        report[name] = dict(status='available', windows=len(x), lines=len(selected),
                            norm_mean=float(norms.mean()), norm_std=float(norms.std()),
                            norm_min=float(norms.min()), norm_max=float(norms.max()),
                            finite=bool(torch.isfinite(x).all()), feature_std_mean=feature_std,
                            pairwise_cosine=_distribution(pairwise), warnings=warnings)
    return dict(representations=report, source_images=[r['side']['image'] for r in selected],
                note='Seeded window sample; pairwise statistics include within-line and cross-line pairs, not a collapse verdict.')


def aggregate_summary(population, retrieval=None, decision_threshold=None):
    rows = population['rows']
    positive, negative = ([r for r in rows if r['target'] == value] for value in (1, 0))
    summary = dict(split=population['split'], population=population['population'],
                   evaluated_pairs=len(rows), aligned_pairs=len(positive), unaligned_pairs=len(negative),
                   evaluated_unique_lines=population['evaluated_unique_lines'],
                   saved_split_unique_lines=population['saved_split_unique_lines'],
                   constructed_negatives=sum(r['constructed_negative'] for r in rows),
                   discrimination=discrimination_metrics(rows, decision_threshold=decision_threshold), retrieval=retrieval,
                   cosine_distributions=population['distribution_stats'])
    if any(r['constructed_negative'] for r in rows):
        summary['discrimination_with_presumed_negatives'] = discrimination_metrics(
            rows, include_constructed=True, decision_threshold=decision_threshold)
    for label, values in (('aligned', positive), ('unaligned', negative)):
        for key in ('pair_score', 'path_cosine_mean', 'off_path_cosine_mean', 'similarity_separation',
                    'path_length', 'mask_coverage_a', 'mask_coverage_b'):
            summary[f'{label}_{key}'] = _distribution([r[key] for r in values if r.get(key) is not None])
    summary['localization'] = {}
    for side in ('a', 'b'):
        for key in ('iou', 'dice', 'precision', 'recall', 'window_iou', 'window_f1',
                    'left_boundary_error_px', 'right_boundary_error_px', 'normalized_boundary_error'):
            values = [r[f'{side}_{key}'] for r in rows if f'{side}_{key}' in r]
            if values:
                summary['localization'][f'{side}_{key}'] = _distribution(values)
    for key in ('pair_mean_iou', 'pair_mean_dice'):
        values = [r[key] for r in rows if key in r]
        if values:
            summary['localization'][key] = _distribution(values)
    if not summary['localization']:
        summary['localization_status'] = 'Ground-truth localization metrics unavailable for this dataset.'
    return summary


def failure_tables(rows, n=5):
    import pandas as pd
    pos, neg = ([r for r in rows if r['target'] == value] for value in (1, 0))
    columns = ['sample_id', 'label', 'constructed_negative', 'pair_score', 'mask_coverage_a',
               'mask_coverage_b', 'path_length', 'pair_mean_iou', 'pair_mean_dice']
    def table(values):
        return pd.DataFrame([{k: r.get(k) for k in columns} for r in values[:n]], columns=columns)
    return dict(strongest_aligned=table(sorted(pos, key=lambda r: r['pair_score'], reverse=True)),
                weakest_aligned=table(sorted(pos, key=lambda r: (r['pair_score'], r['path_length']))),
                strongest_negative=table(sorted(neg, key=lambda r: r['pair_score'], reverse=True)),
                worst_localization=table(sorted([r for r in pos if 'pair_mean_iou' in r],
                                                key=lambda r: (r['pair_mean_iou'], r['pair_mean_dice']))))


def _overlay(image, mask, color=(255, 70, 20), alpha=.4):
    pixels = np.array(image.convert('RGB'), dtype=float)
    active = np.asarray(mask) > 0
    pixels[active] = pixels[active] * (1-alpha) + np.array(color) * alpha
    return pixels.astype(np.uint8)


def plot_pair_alignment(session, result):
    """Full-width originals, source masks, model boundaries, windows, and matches."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch
    pair, metrics, c = result['pair'], result['metrics'], session.config
    label = 'CONSTRUCTED NEGATIVE (presumed)' if pair['constructed_negative'] else (
        'GROUND TRUTH LABEL: ALIGNED' if pair['target'] == 1 else 'GROUND TRUTH LABEL: UNALIGNED' if pair['target'] == 0 else 'LABEL UNKNOWN')
    has_gt = any(gt is not None for gt in result['ground_truth'])
    model_row = 5 if has_gt else 3
    fig = plt.figure(figsize=(19, 28 if has_gt else 24))
    fig.subplots_adjust(left=.05, right=.95, top=.95, bottom=.04, hspace=1.1, wspace=.2)
    heights = [1, 1, 1] + ([1, 1] if has_gt else []) + [1, 1.3, 3, 2, 2.1]
    grid = fig.add_gridspec(len(heights), 2, height_ratios=heights)
    connection_axes = []
    originals = []
    for side, name in enumerate(('A', 'B')):
        info = result['lines'][side]
        with Image.open(pair['sides'][side]['image']) as image:
            original = image.convert('RGB')
        originals.append(original)
        prepared, tensor, _ = prepare_image(pair['sides'][side]['image'], (c.image_height, c.image_width),
                                            c.grayscale, c.crop, pair['sides'][side].get('bbox'), False, c.binarize)
        panels = [(original, f'Line {name} original — source pixels'),
                  (result['masks'][side], f'Predicted source mask — coverage {metrics[f"mask_coverage_{name.lower()}"]:.1%}'),
                  (_overlay(original, result['masks'][side]), 'Prediction overlay (orange)')]
        if has_gt:
            panels.extend([
                  (original if result['ground_truth'][side] is None else result['ground_truth'][side],
                   'GT unavailable' if result['ground_truth'][side] is None else 'Ground truth'),
                  (original if result['ground_truth'][side] is None else
                   _overlay(Image.fromarray(_overlay(original, result['ground_truth'][side], (0, 220, 80))), result['masks'][side]),
                   'GT unavailable' if result['ground_truth'][side] is None else 'GT green + prediction orange overlay')])
        panels.append((prepared, f'Exact model input: {len(info["physical"])} windows, width={c.window_width}, stride={c.window_stride}'))
        for row, (image, title) in enumerate(panels):
            ax = fig.add_subplot(grid[row, side])
            ax.imshow(image, cmap='gray', vmin=0, vmax=1 if np.asarray(image).dtype == bool else 255, aspect='auto')
            ax.set_title(title); ax.set_yticks([])
            if row == model_row:
                for p in info['physical']:
                    ax.axvline(int(p)*c.window_stride, color='cyan', lw=.4, alpha=.5)
                    ax.axvline(int(p)*c.window_stride+c.window_width, color='orange', lw=.4, alpha=.3)
                ax.set_xlabel('Model x; boundaries in physical LTR order. Model sequence is ' + ('RTL' if c.rtl else 'LTR'))
        # Extract with the model's function; undo normalization for display only.
        windows = extract_windows(tensor[None], c.window_width, c.window_stride)[0]
        chosen = np.linspace(0, len(windows)-1, min(8, len(windows)), dtype=int)
        mean, std = ((.449,), (.226,)) if c.grayscale else ((.485,.456,.406), (.229,.224,.225))
        pixels = windows[chosen] * torch.tensor(std)[None, :, None, None] + torch.tensor(mean)[None, :, None, None]
        strip = torch.cat(list(pixels), dim=-1).permute(1, 2, 0).numpy().clip(0, 1)
        ax = fig.add_subplot(grid[model_row + 1, side]); ax.imshow(strip.squeeze(), cmap='gray', vmin=0, vmax=1)
        for index in range(1, len(chosen)):
            ax.axvline(index * c.window_width - .5, color='orange', lw=1)
        ax.set_xticks([(index + .5) * c.window_width for index in range(len(chosen))], chosen)
        ax.set_yticks([])
        ax.set_title('Actual extracted windows (display subset; physical IDs below)')
    order = 'RTL reading order' if c.rtl else 'LTR reading order'
    for side, (matrix, title) in enumerate(((result['cosine'], 'Raw cosine similarity'),
                                          (result['match']['rewards'], 'Alignment rewards (NOT cosine/probability)'))):
        ax = fig.add_subplot(grid[model_row + 2, side])
        heat = ax.imshow(matrix, origin='lower', aspect='auto', cmap='coolwarm',
                         **({'vmin': -1, 'vmax': 1} if side == 0 else {}))
        for number, region in enumerate(result['match']['regions']):
            a, b = zip(*region['pairs']); ax.plot(b, a, '.-', color=plt.get_cmap('tab10')(number % 10), ms=4, lw=1)
        ax.set(xlabel=f'Line B window indices ({order})', ylabel=f'Line A window indices ({order})',
               title=f'{title}: {matrix.shape[0]}×{matrix.shape[1]}, {result["representation"]}')
        if not result['match']['regions']:
            ax.text(.5, .95, 'NO ACCEPTED REGION', transform=ax.transAxes, ha='center', va='top', bbox=dict(facecolor='white', alpha=.8))
        fig.colorbar(heat, ax=ax, shrink=.8)
    nested = grid[model_row + 3, :].subgridspec(2, 1, hspace=.7)
    for side in (0, 1):
        ax = fig.add_subplot(nested[side]); ax.imshow(originals[side], aspect='auto')
        ax.set_yticks([])
        if side == 0:
            ax.set_title('Accepted-window connections: line A above, line B below (source pixels)')
        else:
            ax.set_xlabel('Line B source x (full width)')
        connection_axes.append(ax)
    for number, region in enumerate(result['match']['regions']):
        for i, j in region['pairs']:
            xs = []
            for side, index in enumerate((i, j)):
                line = result['lines'][side]
                interval = source_interval(int(line['physical'][index]), line['geometry'], c.window_width, c.window_stride)
                xs.append(sum(interval)/2)
            connection = ConnectionPatch((xs[0], originals[0].height-.5), (xs[1], -.5),
                                         coordsA='data', coordsB='data', axesA=connection_axes[0], axesB=connection_axes[1],
                                         color=plt.get_cmap('tab10')(number % 10), alpha=.45, lw=.7)
            fig.add_artist(connection)
    ax = fig.add_subplot(grid[model_row + 4, :]); ax.axis('off')
    fields = ['pair_score', 'path_cosine_mean', 'path_cosine_median', 'path_cosine_min', 'maximum_similarity',
              'matrix_cosine_mean', 'off_path_cosine_mean', 'similarity_separation', 'path_length', 'region_count',
              'matched_windows_a', 'matched_windows_b', 'mask_coverage_a', 'mask_coverage_b']
    text = '\n'.join(f'{k}: {metrics[k]:.5g}' if isinstance(metrics[k], float) else f'{k}: {metrics[k]}' for k in fields)
    ax.text(0, 1, text, transform=ax.transAxes, va='top', family='monospace', fontsize=10)
    localization = {k: v for k, v in metrics.items() if k.startswith(('a_', 'b_', 'pair_mean_'))}
    spatial_text = '\n'.join(f'{k}: {v:.5g}' if isinstance(v, float) else f'{k}: {v}'
                             for k, v in localization.items() if not isinstance(v, str) or k.endswith('status'))
    ax.text(.5, 1, spatial_text, transform=ax.transAxes, va='top', family='monospace', fontsize=9)
    scope = 'TRAIN (in-sample)' if session.split == 'train' else session.split.upper()
    fig.suptitle(f'{scope} | {pair["sample_id"]} | {label} ({pair["label"]}) | {result["representation"]} | score={metrics["pair_score"]:.4f}', fontsize=13)
    return fig


def plot_representation_comparison(session, pair):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), constrained_layout=True)
    scores = {}
    for name, ax in zip(REPRESENTATIONS, axes):
        result = session.predict_pair(pair, name)
        scores[name] = max((r['score'] for r in result['match']['regions']), default=0.)
        heat = ax.imshow(result['cosine'], origin='lower', aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        for r in result['match']['regions']:
            a, b = zip(*r['pairs']); ax.plot(b, a, 'k.-', ms=3)
        ax.set(title=f'{name.upper()} score={scores[name]:.4f}', xlabel='B logical windows', ylabel='A logical windows')
    fig.colorbar(heat, ax=axes, label='Raw cosine'); fig.suptitle(pair['sample_id'])
    return fig, scores


def plot_distributions(population):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    for target, label in ((1, 'Aligned'), (0, 'Unaligned / presumed negative')):
        values = [r['pair_score'] for r in population['rows'] if r['target'] == target]
        if values:
            axes[0].hist(values, bins=25, alpha=.55, label=label)
    for name, values in population['distributions'].items():
        if values:
            axes[1].hist(values, bins=np.linspace(-1, 1, 61), density=True, alpha=.45, label=name)
    axes[0].set(xlabel='Best accepted local objective (length dependent)', ylabel='Pairs')
    axes[1].set(xlabel='Raw cosine', ylabel='Window-pair density')
    for ax in axes:
        if ax.get_legend_handles_labels()[0]: ax.legend()
    return fig


def text_dtw_diagnostic(session, side):
    """Optional, separate supervised diagnostic. Never called by image matching."""
    from losses import positive_dtw_loss
    from text_embedding import ARABIC_LETTERS, clean_letters
    line = session.get_line_features(side)
    letters = clean_letters(Path(side['text']).read_text(encoding='utf-8-sig'))
    if not letters:
        return dict(status='unavailable', reason='empty cleaned transcript')
    c = session.config
    visual = line['features']['fused'].to(session.device)
    inventory = list(dict.fromkeys(ARABIC_LETTERS + ''.join(letters)))
    with torch.no_grad():
        chars = session.text_encoder.encode(''.join(letters))
        costs = letter_cost_matrix(visual, chars, alphabet=session.text_encoder.encode(''.join(inventory)),
                                  letter_ids=[inventory.index(x) for x in letters],
                                  temperature=c.competition_temperature, mode=c.dtw_cost_mode)
        soft, _ = positive_dtw_loss(visual[None], [''.join(letters)], session.text_encoder, c)
        hard = hard_dtw_path(costs, c.vertical_penalty, c.horizontal_penalty, c.position_prior,
                             c.disable_horizontal_when_feasible)
        cosine = cosine_similarity_matrix(visual, chars).cpu().numpy()
    return dict(status='available', letters=letters, cosine=cosine, costs=costs.cpu().numpy(),
                path=hard.path, positive_dtw=float(soft), hard_objective=hard.normalized,
                normalization='T+L; full alphabet NLL and saved DTW prior/penalties')


def save_results(session, population, summary, aligned, unaligned, root, diagnostics=None):
    output = Path(root) / session.checkpoint.parent.name / (session.split + '_' + uuid.uuid4().hex[:12])
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(checkpoint=str(session.checkpoint), checkpoint_sha256=session.checkpoint_sha256,
                    epoch=session.saved['epoch'], config=asdict(session.config), split=session.split,
                    split_manifest_sha256=session.saved['split_manifest_sha256'], seed=session.seed,
                    settings=session.settings, summary=summary, feature_diagnostics=diagnostics,
                    representations=sorted({r['representation'] for r in population['rows']}),
                    evaluated_pair_ids=population['selected_pair_ids'],
                    score_definition='maximum accepted region.score; zero if none; affine reward objective, length dependent',
                    limitations=['Manifest labels may be heuristic; source XML boxes are not localization GT.',
                        'Constructed negatives are presumed, not confirmed.',
                        'Greedy Smith-Waterman region extraction is not globally optimal.',
                        'Thresholds are uncalibrated; never tune on test.',
                        'Background correction may suppress broad/repeated genuine matches.',
                        'Source window widths differ after resize; one-to-one/gap matching cannot model all width variation.',
                        'Region overlap does not establish character alignment accuracy.'])
    (output / 'metrics.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    fields = sorted({k for row in population['rows'] for k in row})
    with (output / 'per_sample_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for row in population['rows']:
            writer.writerow({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    for name, pairs in (('aligned', aligned), ('unaligned', unaligned)):
        selection = [dict(sample_id=p['sample_id'], images=[s['image'] for s in p['sides']],
                          label=p['label'], constructed_negative=p['constructed_negative']) for p in pairs]
        (output / f'selected_{name}_samples.json').write_text(json.dumps(selection, indent=2))
    return output


def save_pair_artifacts(session, result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    np.save(directory / 'cosine.npy', result['cosine'])
    np.save(directory / 'alignment_rewards.npy', result['match']['rewards'])
    for side, name in enumerate(('a', 'b')):
        Image.fromarray(result['masks'][side]).save(directory / f'line_{name}_mask.png')
        record = result['pair']['sides'][side]
        with Image.open(record['image']) as image:
            original = image.convert('RGB')
        original.save(directory / f'line_{name}_original.png')
        Image.fromarray(_overlay(original, result['masks'][side])).save(directory / f'line_{name}_overlay.png')
        c = session.config
        prepared, _, _ = prepare_image(record['image'], (c.image_height, c.image_width), c.grayscale,
                                        c.crop, record.get('bbox'), False, c.binarize)
        prepared.save(directory / f'line_{name}_model_input.png')
    rows = []
    c = session.config
    for number, region in enumerate(result['match']['regions']):
        for i, j in region['pairs']:
            row = dict(region=number, cosine=float(result['cosine'][i, j]), reward=float(result['match']['rewards'][i, j]))
            for side, name, index in ((0, 'a', i), (1, 'b', j)):
                line = result['lines'][side]; physical = int(line['physical'][index])
                a, b = source_interval(physical, line['geometry'], c.window_width, c.window_stride)
                row.update({f'logical_{name}': int(line['logical'][index]), f'physical_{name}': physical,
                            f'source_{name}_x0': a, f'source_{name}_x1': b})
            rows.append(row)
    fields = ['region', 'cosine', 'reward'] + [f'{prefix}_{s}{suffix}' for s in ('a','b')
        for prefix,suffix in [('logical',''),('physical',''),('source','_x0'),('source','_x1')]]
    with (directory / 'correspondences.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (directory / 'pair_metrics.json').write_text(json.dumps(dict(metrics=result['metrics'],
        regions=result['match']['regions'], rejected=result['match']['rejected'],
        geometry=[r['geometry'] for r in result['lines']]), indent=2, allow_nan=False))
