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
import re
import unicodedata
import uuid

import numpy as np
from PIL import Image
import torch
from tqdm.auto import tqdm

from dataset import AlignmentDataset, file_hash, prepare_image
from dtw import cosine_similarity_matrix, hard_dtw_path, letter_cost_matrix
from evaluate import region_mask, score_mask, source_interval, match_features, MATCH_VERSION, MATCH_DEFAULTS, resolved_match_settings, plot_match_routes
from train import build_loaders, load_checkpoint


POSITIVE_LABELS = {'high_match', 'medium_match', 'low_match'}
NEGATIVE_LABELS = {'no_shared_content'}
REPRESENTATIONS = ('local', 'context', 'fused')
DEFAULT_MATCH_SETTINGS = dict(MATCH_DEFAULTS)


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
    unique={r['sample_id']:r for r in records if r['target']==int(aligned)}
    choices=sorted(unique.values(),key=lambda r:r['sample_id'])
    if n>len(choices):
        name='positive' if aligned else 'negative'
        print(f'Requested {n} unique {name} samples, but only {len(choices)} eligible unique samples are available.')
    return random.Random(seed).sample(choices,min(n,len(choices)))


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
            reviewed = meta.get('manually_verified_label')
            if reviewed not in (None,'aligned','unaligned'):
                raise ValueError('manually_verified_label must be aligned or unaligned')
            by_images[key] = (annotations, meta.get('method', 'supplied alignment annotation'), reviewed)
        for pair in self.pairs:
            key = tuple(s['image'] for s in pair['sides'])
            if key in by_images:
                pair['annotations'], pair['annotation_provenance'], reviewed = by_images[key]
                pair['manually_verified'] = reviewed is not None
                if reviewed is not None:
                    pair['target'] = int(reviewed=='aligned')
                    pair['label'] = 'manually_verified_'+reviewed

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
        checkpoint.update(Alignment_objective=c.alignment_objective, Cost_mode=c.dtw_cost_mode,
                          Competition_temperature=c.competition_temperature, Position_prior=c.position_prior,
                          Negative_weight=c.negative_dtw_weight,
                          Evidence_prior=(self.saved.get('letter_evidence_prior') or {}).get('source','uniform fallback'))
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
                token_valid=valid.cpu().numpy(), full_physical=output['physical_window_indices'][0].cpu().numpy(),
                physical=output['physical_window_indices'][0][valid].cpu().numpy(),
                logical=torch.arange(len(valid), device=valid.device)[valid].cpu().numpy())
        return self.feature_cache[key]

    def predict_pair(self, pair, representation='fused'):
        if representation not in REPRESENTATIONS:
            raise ValueError(f'Unknown representation: {representation}')
        lines = [self.get_line_features(s) for s in pair['sides']]
        return predict_cached_pair(self, pair, lines, representation)

    def evaluate_pair(self, pair, representation='fused'):
        result = self.predict_pair(pair, representation)
        # Ground truth is read ONLY AFTER predictions and masks have been fixed.
        gt,provenance=load_pair_ground_truth(pair,result['lines'])
        result['metrics'], result['ground_truth'] = compute_pair_metrics(result,self.config,preloaded_ground_truth=gt)
        result['metrics']['ground_truth_provenance']=provenance
        return result

    def matching_cache_key(self, pair, representation='fused'):
        return (pair['sample_id'], representation, MATCH_VERSION, self.checkpoint_sha256,
                json.dumps(resolved_match_settings(self.settings), sort_keys=True),
                json.dumps(getattr(self.text_encoder,'letter_evidence_prior',None), sort_keys=True))

    def metrics(self, pair, representation='fused'):
        key = self.matching_cache_key(pair, representation)
        if key not in self.metric_cache:
            self.metric_cache[key] = self.evaluate_pair(pair, representation)['metrics']
        return self.metric_cache[key]


def predict_cached_pair(session, pair, lines, representation='fused', *, settings=None, cosine=None):
    """One prediction flow for session and cached notebook samples; GT-free.

    ``lines`` contains only model features, valid physical identities and image
    geometry. Cosine can be reused without modifying it for plots or tuning.
    """
    match = match_features(*(r['features'][representation] for r in lines),
        *(r['physical'] for r in lines), text_encoder=session.text_encoder,
        config=session.config, precomputed_cosine=cosine,
        **(session.settings if settings is None else settings))
    masks = [region_mask(match['regions'], side, r['geometry'], session.config.window_width,
                         session.config.window_stride) for side,r in enumerate(lines)]
    return dict(pair=pair, representation=representation, lines=lines, cosine=match['cosine'],
                match=match, masks=masks, settings=match['settings'], split=session.split)


def print_candidate_diagnostics(result, max_rejected=5):
    """Print route evidence and reasons for both accepted and rejected candidates."""
    pair,match=result['pair'],result['match']
    print(f"Sample: {pair['sample_id']} | label: {pair['label']}")
    print('GT available:', [g is not None for g in result.get('ground_truth',[])])
    print('Similarity matrix:', ' × '.join(map(str,result['cosine'].shape)))
    print('Accepted regions:',len(match['regions']))
    print('Evaluation category:',result.get('metrics',{}).get('evaluation_category','unavailable'))
    fields=('candidate_id','candidate_type','status','score','score_per_match','matching_reward',
        'gap_penalty','repeat_penalty','support_a','support_b','matched_pairs','required_support_a',
        'required_support_b','required_matched_pairs','mean_cosine','median_cosine','min_cosine',
        'mean_reward','median_reward','path_density','gap_fraction','horizontal_repeats','vertical_repeats',
        'true_gaps','max_internal_gap','max_physical_discontinuity','repeat_fraction','mutual_anchor_count','mutual_anchor_fraction',
        'strong_anchor_positions','first_strong_anchor_position','last_strong_anchor_position',
        'mean_row_margin','median_row_margin','mean_column_margin','median_column_margin',
        'mean_bidirectional_margin','median_bidirectional_margin','fraction_positive_bidirectional_margin',
        'ambiguous','letter_evidence_mean','median_letter_evidence','minimum_letter_evidence',
        'letter_evidence_positive_fraction','variant','trimmed_prefix','trimmed_suffix','extension_pairs',
        'merged_from','dominated_by','accepted','reason')
    groups=[('Region',match['regions']),('Rejected candidate',sorted(match['rejected'],
            key=lambda c:c.get('score',0),reverse=True)[:max_rejected])]
    for name,candidates in groups:
        for number,candidate in enumerate(candidates):
            print(f'{name} {number}:')
            for field in fields:
                print(f'  {field}: {candidate.get(field)}')
    print('Candidate limit reached:',match.get('candidate_limit_reached',False))


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


# --- Ground truth from page-level subword boxes (dataset GT convention) -------
# Mirrors scripts/data/build_real_alignment_masks.py: page debug/bboxes.json
# boxes are clustered into page lines with page_meta.json line_threshold_used
# (largest-gap fallback to num_lines), A/B box texts are aligned by an exact
# order-preserving LCS, and consecutive matched runs become full-height white
# x-intervals on a black mask the size of the saved line image.
_ARABIC_DIACRITICS = re.compile('[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]')
_LINE_NUMBER = re.compile(r'(?:line[_\-\s]*)?0*(\d+)', re.IGNORECASE)


def annotation_ground_truth_available(annotation):
    """Usable GT from pair metadata: an existing mask image or exact-size intervals."""
    annotation = annotation or {}
    mask = annotation.get('mask')
    if mask and Path(mask).is_file():
        return True
    return annotation.get('intervals') is not None and annotation.get('source_size') is not None


def _boxes_available(side):
    return any((parent / 'debug' / 'bboxes.json').is_file() for parent in Path(side['image']).parents)


def pair_ground_truth_available(pair):
    """True when BOTH lines can obtain GT: metadata masks/intervals, or page
    subword boxes for both sides (the LCS build needs both sides' boxes)."""
    if all(annotation_ground_truth_available(a) for a in pair['annotations']):
        return True
    return all(_boxes_available(s) for s in pair['sides'])


def positive_pairs(session, require_ground_truth=True):
    """Real manifest positive pairs in the saved split; line B is always the
    recorded partner of line A, never an independently sampled line."""
    pool = [p for p in session.pairs if p['target'] == 1 and not p['constructed_negative']]
    if require_ground_truth:
        pool = [p for p in pool if pair_ground_truth_available(p)]
    return sorted(pool, key=lambda r: r['sample_id'])


def shuffled_positive_pairs(session, seed=42, require_ground_truth=True):
    """Seeded shuffle of the eligible positive pool. Iterate and keep the first
    valid samples so invalid samples are skipped automatically."""
    pool = positive_pairs(session, require_ground_truth)
    return random.Random(seed).sample(pool, len(pool))


def _normalized_unit(value):
    text = unicodedata.normalize('NFKC', str(value or ''))
    text = _ARABIC_DIACRITICS.sub('', text.replace('\u0640', ''))
    return ''.join(text.split())


def _page_line_boxes(image, line_width):
    """Subword boxes of one saved line in line-image x coordinates, RTL ordered.

    Returns (boxes, status): boxes = [(x0, x1, text)], empty with reason on failure.
    """
    image = Path(image)
    side_root = next((p for p in image.parents if (p / 'debug' / 'bboxes.json').is_file()), None)
    if side_root is None:
        return [], 'no debug/bboxes.json above the line image'
    try:
        payload = json.loads((side_root / 'debug' / 'bboxes.json').read_text(encoding='utf-8'))
    except Exception as exc:
        return [], f'bboxes.json parse error: {type(exc).__name__}: {exc}'
    boxes = []
    for row, record in enumerate(payload if isinstance(payload, list) else []):
        try:
            x0, x1 = sorted((float(record['x1']), float(record['x2'])))
            y0, y1 = sorted((float(record['y1']), float(record['y2'])))
        except (KeyError, TypeError, ValueError):
            continue
        if x1 > x0 and y1 > y0:
            boxes.append(dict(x0=x0, x1=x1, cx=(x0 + x1) / 2, cy=(y0 + y1) / 2,
                              text=str(record.get('text', '')), row=row))
    if not boxes:
        return [], 'no valid x1/y1/x2/y2 box records'
    meta = {}
    meta_path = side_root / 'page_meta.json'
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:
            meta = {}
    threshold = float(meta.get('line_threshold_used') or 42.)
    ordered = sorted(boxes, key=lambda b: (b['cy'], b['cx']))
    clusters, centers = [], []
    for box in ordered:
        if clusters and abs(box['cy'] - centers[-1]) <= threshold:
            clusters[-1].append(box)
            centers[-1] = float(np.median([b['cy'] for b in clusters[-1]]))
        else:
            clusters.append([box])
            centers.append(box['cy'])
    expected = int(meta.get('num_lines') or 0)
    if expected > 1 and len(clusters) != expected and len(ordered) > 1:
        gaps = sorted(((ordered[i + 1]['cy'] - ordered[i]['cy'], i)
                       for i in range(len(ordered) - 1)), reverse=True)
        cut = {i for _, i in gaps[:expected - 1]}
        clusters, current = [], []
        for i, box in enumerate(ordered):
            current.append(box)
            if i in cut:
                clusters.append(current)
                current = []
        if current:
            clusters.append(current)
    clusters.sort(key=lambda group: float(np.mean([b['cy'] for b in group])))
    number = _LINE_NUMBER.search(image.stem)
    index = int(number.group(1)) - 1 if number else -1
    if not 0 <= index < len(clusters):
        return [], f'line {index + 1} outside {len(clusters)} clustered page lines'
    originals = sorted(side_root.glob('original_image.*'))
    page_width = None
    if originals:
        with Image.open(originals[0]) as source:
            page_width = float(source.width)
    if page_width is None:
        page_width = max(float(line_width), max(b['x1'] for b in boxes))
    scale = float(line_width) / max(1., page_width)
    selected = [dict(b, x0=b['x0'] * scale, x1=b['x1'] * scale) for b in clusters[index]]
    selected.sort(key=lambda b: (-b['cx'], b['row']))   # RTL reading order
    return [(b['x0'], b['x1'], b['text']) for b in selected], 'ok'


def _lcs_pairs(left, right):
    n, m = len(left), len(right)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            dp[i, j] = 1 + dp[i + 1, j + 1] if left[i] and left[i] == right[j] else max(dp[i + 1, j], dp[i, j + 1])
    result, i, j = [], 0, 0
    while i < n and j < m:
        if left[i] and left[i] == right[j] and dp[i, j] == 1 + dp[i + 1, j + 1]:
            result.append((i, j)); i += 1; j += 1
        elif dp[i + 1, j] >= dp[i, j + 1]:
            i += 1
        else:
            j += 1
    return result


def _consecutive_runs(pairs):
    if not pairs:
        return []
    runs = [[pairs[0]]]
    for pair in pairs[1:]:
        previous = runs[-1][-1]
        if pair[0] == previous[0] + 1 and pair[1] == previous[1] + 1:
            runs[-1].append(pair)
        else:
            runs.append([pair])
    return runs


def _mask_intervals(mask):
    """Exclusive [start, end) column runs of a full-height boolean mask."""
    intervals = []
    for column in np.flatnonzero(np.asarray(mask, bool).any(0)):
        if intervals and column == intervals[-1][1]:
            intervals[-1][1] = int(column) + 1
        else:
            intervals.append([int(column), int(column) + 1])
    return intervals


def boxes_ground_truth(pair):
    """Build full-height GT masks for both sides from page subword boxes.

    Returns (masks, detail): masks = [bool array | None, bool array | None].
    """
    images = [Path(s['image']) for s in pair['sides']]
    sizes = []
    for image in images:
        with Image.open(image) as source:
            sizes.append(source.size)           # (width, height)
    boxes, statuses = [], []
    for (width, _), image in zip(sizes, images):
        side_boxes, status = _page_line_boxes(image, width)
        boxes.append(side_boxes)
        statuses.append(status)
    if not all(boxes):
        return [None, None], 'bbox ground truth unavailable: ' + '; '.join(statuses)
    units = [[_normalized_unit(text) for _, _, text in side] for side in boxes]
    pairs = _lcs_pairs(*units)
    runs = _consecutive_runs(pairs)
    if not runs:
        return [np.zeros((h,w),bool) for w,h in sizes], 'complete page box transcripts: no matched subword run'
    masks = []
    for side, ((width, height), side_boxes) in enumerate(zip(sizes, boxes)):
        mask = np.zeros((height, width), dtype=bool)
        for run in runs:
            selected = [side_boxes[match[side]] for match in run]
            x0 = max(0, min(width, math.floor(min(b[0] for b in selected))))
            x1 = max(0, min(width, math.ceil(max(b[1] for b in selected))))
            if x1 > x0:
                mask[:, x0:x1] = True
        masks.append(mask)
    return masks, f'page debug/bboxes.json LCS: {len(pairs)} matched units in {len(runs)} run(s)'


def load_pair_ground_truth(pair, lines):
    """Load pair-specific GT once, independently of matching and predictions."""
    masks, provenance = [], []
    fallback, detail = None, None
    for side, line in enumerate(lines):
        width, height = line['geometry']['source_size']
        shape = (height, width)
        gt = _load_annotation(pair['annotations'][side], shape)
        if gt is not None:
            source = pair['annotation_provenance']
        else:
            if fallback is None:
                fallback, detail = boxes_ground_truth(pair)
            gt = fallback[side]
            source = ('built from page debug/bboxes.json subword LCS' if gt is not None
                      else f'unavailable ({detail})')
        if gt is not None and gt.shape != shape:
            raise ValueError('Ground-truth/source geometry mismatch; resizing is forbidden')
        masks.append(gt)
        provenance.append(source)
    return masks, ' | '.join(provenance)


def attach_ground_truth(session,result):
    """Evaluation-only compatibility wrapper; prediction/masks already fixed."""
    if result['metrics'].get('ground_truth_provenance') is not None:
        return result
    masks,provenance=load_pair_ground_truth(result['pair'],result['lines'])
    result['metrics'],result['ground_truth']=compute_pair_metrics(result,session.config,preloaded_ground_truth=masks)
    result['metrics']['ground_truth_provenance']=provenance
    return result


def evaluation_category(pair,ground_truth):
    """Post-prediction local-overlap classification, never a matcher input.

    Nonempty localization GT wins over a negative manifest label. Complete empty
    annotations certify no overlap. Transcript word overlap supplies a partial
    category but cannot fabricate localization masks. A coarse negative label
    alone is unknown; transcript absence alone is not proof of no substring.
    """
    available=[g is not None for g in ground_truth]
    occupied=[bool(np.asarray(g,bool).any()) if g is not None else None for g in ground_truth]
    if all(available) and occupied[0]!=occupied[1]:
        return dict(evaluation_category='unknown',evaluation_category_reason='contradictory_side_annotations',shared_words=[])
    if any(v is True for v in occupied):
        category='positive' if pair.get('target')==1 else 'partial_overlap'
        return dict(evaluation_category=category,evaluation_category_reason='nonempty_localization_ground_truth',shared_words=[])
    if all(available):
        category='unknown' if pair.get('target')==1 else 'true_no_overlap'
        return dict(evaluation_category=category,evaluation_category_reason='complete_empty_localization_ground_truth',shared_words=[])
    words=[]
    for side in pair.get('sides',[]):
        path=side.get('text')
        if not path or not Path(path).is_file():words=[];break
        text=Path(path).read_text(encoding='utf-8')
        units={_normalized_unit(word) for word in text.split()}
        words.append(units-{''})
    shared=sorted(words[0]&words[1]) if len(words)==2 else []
    if pair.get('target')==1:
        return dict(evaluation_category='positive',evaluation_category_reason='known_manifest_positive_localization_unavailable',shared_words=shared)
    if shared:
        return dict(evaluation_category='partial_overlap',evaluation_category_reason='transcript_word_overlap_localization_unavailable',shared_words=shared)
    if pair.get('manually_verified') and not pair.get('constructed_negative') and pair.get('target')==0:
        return dict(evaluation_category='true_no_overlap',evaluation_category_reason='manually_verified_no_overlap',shared_words=[])
    return dict(evaluation_category='unknown',evaluation_category_reason='negative_manifest_is_not_localization_ground_truth',shared_words=[])


def _binary_metrics(pred, gt):
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    tp, union = np.logical_and(pred, gt).sum(), np.logical_or(pred, gt).sum()
    precision = float(tp / pred.sum()) if pred.any() else 0.
    recall = float(tp / gt.sum()) if gt.any() else 0.
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.
    return dict(iou=float(tp / union) if union else 1., precision=precision, recall=recall, f1=f1)


def localization_metrics(mask, annotation, line, config, regions, side, *, preloaded_gt=None):
    gt = preloaded_gt if preloaded_gt is not None else _load_annotation(annotation, mask.shape)
    if gt is None:
        return dict(status='unavailable', reason='no pair-specific localization annotation'), None
    if gt.shape != mask.shape:
        raise ValueError('Ground-truth/source geometry mismatch; resizing is forbidden')
    # Reuse the public source-mask scorer when a file is supplied.
    pixel = (score_mask(mask, annotation['mask']) if preloaded_gt is None and annotation.get('mask')
             else _binary_metrics(mask > 0, gt))
    metrics = {k: pixel[k] for k in ('iou', 'precision', 'recall', 'f1')}
    metrics.update(status='available', dice=metrics['f1'])
    predicted, truth = _mask_intervals(mask), _mask_intervals(gt)
    overlaps = []
    for i,(a,b) in enumerate(predicted):
        for j,(c,d) in enumerate(truth):
            overlap = max(0,min(b,d)-max(a,c))
            if overlap: overlaps.append((overlap/(max(b,d)-min(a,c)),i,j))
    used_p,used_t,errors=set(),set(),[]
    # Diagnostic region assignment: greedy descending interval IoU, not prediction.
    for iou,i,j in sorted(overlaps,reverse=True):
        if i in used_p or j in used_t: continue
        used_p.add(i); used_t.add(j)
        errors.append(dict(predicted=predicted[i],expected=truth[j],iou=iou,
                           left=abs(predicted[i][0]-truth[j][0]),right=abs(predicted[i][1]-truth[j][1])))
    metrics.update(extra_regions=len(predicted)-len(used_p),missed_regions=len(truth)-len(used_t),
                   per_region_boundary_errors=errors,
                   per_region_boundary_mean_px=np.mean([e[k] for e in errors for k in ('left','right')]).item() if errors else None)
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


def compute_pair_metrics(result, config, *, preloaded_ground_truth=None):
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
                   manually_verified=pair.get('manually_verified',False),
                   # Existing accepted-region score includes rewards and affine gap costs.
                   pair_score=max((r['score'] for r in match['regions']), default=0.),
                   accepted_region_score_sum=sum(r['score'] for r in match['regions']),
                   max_local_alignment_score=match.get(
                       'maximum_local_score', max((r['score'] for r in match['regions']), default=0.)),
                   similarity_mode=result['settings']['similarity_mode'], decoder=result['settings']['decoder'],
                   path_length=len(pairs), region_count=len(match['regions']),
                   path_cosine_mean=float(values.mean()) if len(values) else None,
                   path_cosine_median=float(np.median(values)) if len(values) else None,
                   path_cosine_min=float(values.min()) if len(values) else None,
                   maximum_similarity=float(cosine.max()) if cosine.size else None,
                   matrix_cosine_mean=float(cosine.mean()) if cosine.size else None,
                   off_path_cosine_mean=background,
                   similarity_separation=float(values.mean()) - background if len(values) and background is not None else None)
    deltas = [(b[0]-a[0], b[1]-a[1]) for r in match['regions'] for a, b in zip(r['pairs'], r['pairs'][1:])]
    metrics.update(horizontal_skipped_windows=sum(max(0, j-1) for i, j in deltas),
                   vertical_skipped_windows=sum(max(0, i-1) for i, j in deltas),
                   largest_anchor_jump=max((max(d) for d in deltas), default=0),
                   internal_discontinuities=sum(i > 1 or j > 1 for i, j in deltas),
                   monotonic=(all(i >= 0 and j >= 0 and (i > 0 or j > 0) for i, j in deltas)
                              if result['settings'].get('alignment_mode') == 'local_repeat_dtw'
                              else all(i > 0 and j > 0 for i, j in deltas)),
                   repeat_matches=sum(i == 0 or j == 0 for i, j in deltas),
                   separation_between_regions=max(0, len(match['regions']) - 1))
    if result['settings'].get('alignment_mode') == 'local_repeat_dtw':
        metrics.update(horizontal_repeats=sum(r['horizontal_repeats'] for r in match['regions']),
                       vertical_repeats=sum(r['vertical_repeats'] for r in match['regions']),
                       true_gaps=sum(r['true_gaps'] for r in match['regions']))
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
        spatial, gt = localization_metrics(
            mask, pair['annotations'][side] if preloaded_ground_truth is None else {},
            line, config, match['regions'], side,
            preloaded_gt=None if preloaded_ground_truth is None else preloaded_ground_truth[side])
        ground_truth.append(gt)
        metrics.update({f'{name}_{k}': v for k, v in spatial.items()})
    for metric in ('iou', 'dice'):
        if all(f'{s}_{metric}' in metrics for s in ('a', 'b')):
            metrics['pair_mean_' + metric] = (metrics['a_' + metric] + metrics['b_' + metric]) / 2
    metrics.update(evaluation_category(pair,ground_truth))
    return metrics, ground_truth


def evaluate_population(session, representation='fused', max_pairs=0):
    if max_pairs < 0:
        raise ValueError('max_pairs must be >= 0')
    unique = list({p['sample_id']: p for p in session.pairs}.values())
    pairs = unique if max_pairs == 0 else random.Random(session.seed).sample(unique, min(max_pairs, len(unique)))
    rows, distributions = [], {k: [] for k in ('positive_path', 'negative_matrix', 'off_path')}
    # Matrices are small; retain distributions only, not source images/graphs.
    for pair in tqdm(pairs, desc=f'{session.split} pairs ({representation})'):
        result = attach_ground_truth(session, session.evaluate_pair(pair, representation))
        rows.append(result['metrics'])
        key = session.matching_cache_key(pair, representation)
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
    categories={name:[r for r in rows if r.get('evaluation_category','unknown')==name]
                for name in ('positive','partial_overlap','true_no_overlap','unknown')}
    summary['evaluation_category_counts']={k:len(v) for k,v in categories.items()}
    true_negatives=categories['true_no_overlap']
    summary['true_negative_false_positive_rate']=(sum(r['region_count']>0 for r in true_negatives)/len(true_negatives)
                                                  if true_negatives else None)
    for category,prefix in (('positive','positive'),('partial_overlap','partial_overlap')):
        annotated=[r for r in categories[category] if all(r.get(f'{side}_status')=='available' for side in ('a','b'))]
        summary[prefix+'_localized_pairs']=len(annotated)
        for metric in ('recall','precision'):
            values=[r[f'{side}_{metric}'] for r in annotated for side in ('a','b')]
            summary[prefix+'_'+metric]=float(np.mean(values)) if values else None
    summary['localization_average_definition']='macro mean over both sides of pairs with two available localization masks'
    summary['manifest_negative_count']=sum(not r['constructed_negative'] for r in negative)
    summary['annotation_caution']='Coarse negative labels alone are unknown; box/LCS annotations are automatically derived.'
    summary['small_population_warning']=len(rows)<30
    for side in ('a', 'b'):
        for key in ('iou', 'dice', 'precision', 'recall', 'window_iou', 'window_f1',
                    'left_boundary_error_px', 'right_boundary_error_px', 'normalized_boundary_error',
                    'extra_regions','missed_regions','per_region_boundary_mean_px'):
            values = [r[f'{side}_{key}'] for r in positive if r.get(f'{side}_{key}') is not None]
            if values:
                summary['localization'][f'{side}_{key}'] = _distribution(values)
    for key in ('pair_mean_iou', 'pair_mean_dice'):
        values = [r[key] for r in positive if key in r]
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


def plot_pair_alignment(session, result, show_gt=False, max_rejected_paths=5):
    """Simplified full-line figure: originals, predicted and ground-truth
    masks/overlays, similarity heatmaps and the metrics panel.

    No model-input window grids, extracted-window strips or other
    split-window visualizations are shown.
    """
    import matplotlib.pyplot as plt
    pair, metrics = result['pair'], result['metrics']
    if pair['constructed_negative']:
        label = 'CONSTRUCTED NEGATIVE (presumed)'
    elif pair['target'] == 1:
        label = 'RANDOMLY CHOSEN POSITIVE SAMPLE (true manifest pair)'
    elif pair['target'] == 0:
        label = 'MANIFEST LABEL: ' + pair['label']
    else:
        label = 'LABEL UNKNOWN'
    label += ' | local evaluation: ' + metrics.get('evaluation_category','unknown')
    originals = []
    for side in range(2):
        with Image.open(pair['sides'][side]['image']) as image:
            originals.append(image.convert('RGB'))
    fig = plt.figure(figsize=(19, 30) if show_gt else (19, 24))
    fig.subplots_adjust(left=.05, right=.95, top=.94, bottom=.03, hspace=.65, wspace=.15)
    grid = fig.add_gridspec(7, 2, height_ratios=[1, 1, 1, 1, 1, 2.4, 4]) if show_gt else fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 2.4, 4])
    for side, name in enumerate(('A', 'B')):
        coverage = metrics[f'mask_coverage_{name.lower()}']
        rows = [
            (originals[side], False, f'Line {name} — original full line'),
            (np.asarray(result['masks'][side]) > 0, True,
             f'Predicted mask — line {name} (coverage {coverage:.1%})'),
            (_overlay(originals[side], result['masks'][side]), False,
             f'Predicted overlay — line {name} (orange on original)'),
        ]
        if show_gt:
            gt = result['ground_truth'][side]
            rows += [
                (None if gt is None else np.asarray(gt, bool), True,
                 f'Ground-truth mask — line {name}' if gt is not None else
                 f'Ground-truth mask — line {name} (UNAVAILABLE)'),
                (None if gt is None else _overlay(originals[side], gt, (0, 200, 80)), False,
                 f'Ground-truth overlay — line {name} (green on original)' if gt is not None else
                 f'Ground-truth overlay — line {name} (UNAVAILABLE)'),
            ]
        # Omitting (not just blanking) GT rows here keeps the reduced grid aligned with the heatmap/metrics panels below.
        for row, (image, binary, title) in enumerate(rows):
            ax = fig.add_subplot(grid[row, side])
            if image is None:
                continue
            elif binary:
                ax.imshow(np.asarray(image, dtype=float), cmap='gray', vmin=0, vmax=1, aspect='auto')
            else:
                ax.imshow(image, aspect='auto')
            ax.set_title(title)
            ax.set_yticks([])
    order = 'RTL reading order' if session.config.rtl else 'LTR reading order'
    for side, (matrix, title) in enumerate(((result['cosine'], 'Raw cosine similarity'),
                                            (result['match']['rewards'], 'Alignment rewards (NOT cosine/probability)'))):
        ax = fig.add_subplot(grid[5 if show_gt else 3, side])
        heat = ax.imshow(matrix, origin='lower', aspect='auto', cmap='coolwarm',
                         **({'vmin': -1, 'vmax': 1} if side == 0 else {}))
        plot_match_routes(ax, result['match'], max_rejected_paths)
        ax.set(xlabel=f'Line B window indices ({order})', ylabel=f'Line A window indices ({order})',
               title=f'{title}: {matrix.shape[0]}×{matrix.shape[1]}, {result["representation"]}')
        if not result['match']['regions']:
            ax.text(.5, .95, 'NO ACCEPTED REGION', transform=ax.transAxes, ha='center', va='top',
                    bbox=dict(facecolor='white', alpha=.8))
        fig.colorbar(heat, ax=ax, shrink=.8)
    ax = fig.add_subplot(grid[6 if show_gt else 4, :])
    ax.axis('off')
    fields = ['pair_score', 'path_cosine_mean', 'path_cosine_median', 'path_cosine_min',
              'maximum_similarity', 'matrix_cosine_mean', 'off_path_cosine_mean',
              'similarity_separation', 'path_length', 'region_count',
              'matched_windows_a', 'matched_windows_b', 'mask_coverage_a', 'mask_coverage_b']
    text = '\n'.join(f'{k}: {metrics[k]:.5g}' if isinstance(metrics[k], float) else f'{k}: {metrics[k]}'
                     for k in fields)
    ax.text(0, 1, text, transform=ax.transAxes, va='top', family='monospace', fontsize=10)
    localization = {k: v for k, v in metrics.items() if k.startswith(('a_', 'b_', 'pair_mean_'))}
    spatial_text = '\n'.join(f'{k}: {v:.5g}' if isinstance(v, float) else f'{k}: {v}'
                             for k, v in localization.items() if not isinstance(v, (list,dict))
                             and (not isinstance(v, str) or k.endswith('status')))
    ax.text(.5, 1, spatial_text, transform=ax.transAxes, va='top', family='monospace', fontsize=9)
    if metrics.get('ground_truth_provenance'):
        import textwrap
        ax.text(0, 0, textwrap.fill('GT provenance: ' + metrics['ground_truth_provenance'],150),
                transform=ax.transAxes, va='bottom', fontsize=8, style='italic')
    scope = 'TRAIN (in-sample)' if session.split == 'train' else session.split.upper()
    fig.suptitle(f'{scope} | {label} | {pair["sample_id"]} | {result["representation"]} | '
                 f'score={metrics["pair_score"]:.4f}', fontsize=13)
    return fig


def plot_candidate_crops(session, result, top_k=5):
    """Diagnostic paired SOURCE crops; annotations/expected regions never affect decoding."""
    import matplotlib.pyplot as plt
    candidates=result['match'].get('candidates',result['match']['regions'])[:top_k]
    if not candidates:
        fig,ax=plt.subplots(); ax.text(.1,.5,'No positive candidate'); ax.axis('off'); return fig
    fig,axes=plt.subplots(len(candidates),2,figsize=(16,3*len(candidates)),squeeze=False,constrained_layout=True)
    for row,candidate in enumerate(candidates):
        for side,ax in enumerate(axes[row]):
            line=result['lines'][side]
            intervals=[source_interval(p,line['geometry'],session.config.window_width,session.config.window_stride)
                       for p in candidate['supported_physical'][side]]
            left,right=math.floor(min(x[0] for x in intervals)),math.ceil(max(x[1] for x in intervals))
            with Image.open(result['pair']['sides'][side]['image']) as image:
                ax.imshow(image.crop((left,0,right,image.height)),aspect='auto',cmap='gray')
            ax.set_title(f'{"AB"[side]} source x=[{left},{right}) | {candidate.get("reason","accepted")}\n'
                f'score={candidate["score"]:.3f}, support={candidate["support"]}, '
                f'evidence={candidate.get("matching_evidence",0):.3f}, repeat={candidate.get("repeat_penalties",0):.3f}, '
                f'gap={candidate.get("gap_penalties",0):.3f}, weak={len(candidate.get("weak_spans",[]))}')
    return fig


def plot_representation_comparison(session, pair):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), constrained_layout=True)
    scores = {}
    for name, ax in zip(REPRESENTATIONS, axes):
        result = session.predict_pair(pair, name)
        scores[name] = max((r['score'] for r in result['match']['regions']), default=0.)
        heat = ax.imshow(result['cosine'], origin='lower', aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
        plot_match_routes(ax, result['match'])
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
                    settings=resolved_match_settings(session.settings), implementation_version=MATCH_VERSION,
                    prior_metadata=getattr(session.text_encoder,'letter_evidence_prior',None),
                    summary=summary, feature_diagnostics=diagnostics,
                    representations=sorted({r['representation'] for r in population['rows']}),
                    evaluated_pair_ids=population['selected_pair_ids'],
                    score_definition='maximum accepted region.score; zero if none; affine reward objective, length dependent',
                    limitations=['Manifest labels may be heuristic; source XML boxes are not localization GT.',
                        'Constructed negatives are presumed, not confirmed.',
                        'Greedy Smith-Waterman region extraction is not globally optimal.',
                        'Thresholds are uncalibrated; never tune on test.',
                        'Background correction may suppress broad/repeated genuine matches.',
                        'Bounded repeats approximate width variation; repeat evidence weighting/penalties are uncalibrated.',
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
    if 'letter_evidence' in result['match']:
        np.save(directory / 'letter_evidence.npy', result['match']['letter_evidence'])
        for side, logp in enumerate(result['match']['log_probabilities']):
            np.save(directory / f'letter_log_probs_{side}.npy', logp)
    for side, name in enumerate(('a', 'b')):
        for key in ('physical','logical','token_valid','full_physical'):
            np.save(directory / f'{name}_{key}.npy', result['lines'][side][key])
        np.save(directory / f'{name}_features.npy',result['lines'][side]['features'][result['representation']].numpy())
        Image.fromarray(result['masks'][side]).save(directory / f'line_{name}_mask.png')
        record = result['pair']['sides'][side]
        with Image.open(record['image']) as image:
            original = image.convert('RGB')
        original.save(directory / f'line_{name}_original.png')
        Image.fromarray(_overlay(original, result['masks'][side])).save(directory / f'line_{name}_overlay.png')
        gt = result['ground_truth'][side] if result.get('ground_truth') else None
        if gt is not None:
            Image.fromarray((np.asarray(gt, bool) * 255).astype(np.uint8)).save(
                directory / f'line_{name}_ground_truth_mask.png')
            Image.fromarray(_overlay(original, gt, (0, 200, 80))).save(
                directory / f'line_{name}_ground_truth_overlay.png')
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
        candidates=result['match'].get('candidates',[]),
        candidate_limit_reached=result['match'].get('candidate_limit_reached',False),
        settings=result['settings'], implementation_version=MATCH_VERSION,
        checkpoint=str(session.checkpoint), checkpoint_sha256=session.checkpoint_sha256,
        prior_metadata=result['match'].get('prior_metadata'),
        geometry=[r['geometry'] for r in result['lines']]), indent=2, allow_nan=False))
