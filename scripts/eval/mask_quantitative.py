"""Strict mask scoring and frozen-output caching around the notebook predictor.

Inference receives image features/geometry only. Supplied labels and mask files
are a separate scoring contract; missing masks never become negative labels.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import beta

from dataset import file_hash
from dtw import cosine_similarity_matrix
from evaluate import resolved_match_settings, MATCH_VERSION
from evaluation_utils import EvaluationSession, predict_cached_pair, _side_key
from text_embedding import ARABIC_LETTERS
from scripts.eval.tune_matching import write_json, write_csv

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'quantitative-mask-v1'
METRICS = ('positive_iou', 'positive_dice', 'precision', 'recall',
           'negative_pair_fpr', 'negative_masked_area_fraction')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def notebook_baseline(path=ROOT/'notebooks/model_evaluation.ipynb'):
    """Read literal settings and execute only the notebook's settings function."""
    book = json.loads(Path(path).read_text())
    cells = [''.join(c['source']) for c in book['cells'] if c['cell_type'] == 'code']
    values = {}
    for cell in cells:
        if 'ALIGNMENT_MODE =' not in cell: continue
        for node in ast.parse(cell).body:
            if isinstance(node, ast.Assign):
                try: value = ast.literal_eval(node.value)
                except (ValueError, TypeError): continue
                for target in node.targets:
                    if isinstance(target, ast.Name): values[target.id] = value
    functions = [n for cell in cells if 'def current_match_settings(' in cell
                 for n in ast.parse(cell).body if isinstance(n, ast.FunctionDef) and n.name == 'current_match_settings']
    if len(functions) != 1 or not values: raise ValueError('Cannot identify unique notebook matching settings')
    namespace = dict(values, resolved_match_settings=resolved_match_settings)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    return values, namespace['current_match_settings']()


def effective_settings(requested):
    settings = resolved_match_settings(requested)
    return dict(settings=settings, derived=dict(
        effective_reward_similarity=settings['candidate_similarity_mode'],
        normal_support_a=settings['normal_min_distinct_a'] or max(settings['min_windows'], settings['min_distinct_windows_a']),
        normal_support_b=settings['normal_min_distinct_b'] or max(settings['min_windows'], settings['min_distinct_windows_b']),
        normal_matched_pairs=settings['normal_min_matched_pairs'] or max(settings['min_windows'], settings['min_matched_pairs']),
        short_anchor_count=max(settings['short_min_mutual_anchors'], settings['min_mutual_anchors_short']),
        short_anchor_fraction=max(settings['short_min_mutual_anchor_fraction'], settings['min_mutual_anchor_fraction_short']),
        short_internal_gap=min(settings['max_gap'],settings['short_max_internal_gap']),
        extension_gap=min(settings['max_gap'], settings['extension_max_internal_gap']),
        merge_gap_a=min(settings['max_gap'], settings['merge_max_gap_a']),
        merge_gap_b=min(settings['max_gap'], settings['merge_max_gap_b']),
        mask_fill_gap=settings['max_gap'], comparison_max_pairs=0,
        inactive_in_letter_rewards=['score_mode', 'contrast_margin'] if settings['similarity_mode'] == 'letter_evidence' else [],
        retired_diagnostic_only=['use_strong_start_anchor', 'start_top_k', 'start_min_reward'],
        inactive_short_settings=not settings['allow_short_regions'],
        inactive_extension_route_settings=not settings['enable_path_extension'],
        extension_min_reward_used_by_merging=settings['enable_region_merging'],
        inactive_merging_settings=not settings['enable_region_merging']))


def binary_metrics(prediction, truth):
    pred, gt = np.asarray(prediction) > 0, np.asarray(truth) > 0
    if pred.ndim != 2 or pred.shape != gt.shape: raise ValueError('Masks must share source-image H x W coordinates')
    tp = int((pred & gt).sum()); fp = int((pred & ~gt).sum()); fn = int((~pred & gt).sum())
    def ratio(a, b): return a/b if b else 0.
    return dict(tp=tp, fp=fp, fn=fn, pixels=int(pred.size), predicted_pixels=int(pred.sum()),
                truth_pixels=int(gt.sum()), iou=ratio(tp, tp+fp+fn), dice=ratio(2*tp, 2*tp+fp+fn),
                precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                foreground_fraction=float(pred.mean()), full_height=bool(np.all(pred == pred[:1]) and np.all(gt == gt[:1])))


def intervals(mask):
    mask = np.asarray(mask) > 0
    if mask.ndim != 2 or not np.all(mask == mask[:1]): raise ValueError('Region metric requires verified full-height spans')
    edges = np.diff(np.r_[False, mask[0], False].astype(int))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def region_counts(prediction, truth, threshold=.5):
    pred, gt = intervals(prediction), intervals(truth)
    n, m = len(pred), len(gt)
    if not n or not m: return dict(tp=0, fp=n, fn=m, matches=[])
    iou = np.array([[max(0, min(b,d)-max(a,c))/(max(b,d)-min(a,c)) for c,d in gt] for a,b in pred])
    # A match's cardinality bonus dominates every possible sum-IoU difference.
    bonus = min(n,m)+1
    weights = np.zeros((n+m,n+m))
    weights[:n,:m] = np.where(iou >= threshold, bonus+iou, -bonus)
    a, b = linear_sum_assignment(-weights)
    matches = [(int(i), int(j), float(iou[i,j])) for i,j in zip(a,b) if i<n and j<m and iou[i,j]>=threshold]
    return dict(tp=len(matches), fp=n-len(matches), fn=m-len(matches), matches=matches)


def aggregate(rows):
    positives = [r for r in rows if r['category'] == 'positive']
    negatives = [r for r in rows if r['category'] == 'negative']
    if not positives or not negatives: raise ValueError('Strict evaluation requires both positive and verified negative pairs')
    result = {key: float(np.mean([r[key] for r in positives])) for key in ('positive_iou','positive_dice','precision','recall')}
    fp = sum(r['negative_false_positive'] for r in negatives)
    result.update(positive_pairs=len(positives), negative_pairs=len(negatives),
        negative_false_positive_pairs=int(fp), negative_pair_denominator=len(negatives),
        negative_pair_fpr=fp/len(negatives),
        negative_masked_area_fraction=float(np.mean([r['negative_masked_area_fraction'] for r in negatives])),
        candidate_limit_hits=sum(r['candidate_limit_hit'] for r in rows))
    eligible = all(r['region_eligible'] for r in positives)
    counts = {k: sum(r['region_'+k] for r in positives) for k in ('tp','fp','fn')}
    tp, fp, fn = (counts[k] for k in ('tp','fp','fn'))
    result['region_f1'] = dict(available=eligible, aggregation='micro interval counts over both lines of all positive pairs',
        positive_pair_denominator=len(positives), non_full_height_positive_pairs=sum(not r['region_eligible'] for r in positives),
        **counts, precision=tp/(tp+fp) if eligible and tp+fp else 0. if eligible else None,
        recall=tp/(tp+fn) if eligible and tp+fn else 0. if eligible else None,
        f1=2*tp/(2*tp+fp+fn) if eligible and 2*tp+fp+fn else 0. if eligible else None)
    return result


def dependency_units(pairs):
    parent = {}
    def find(k):
        parent.setdefault(k,k)
        if parent[k] != k: parent[k] = find(parent[k])
        return parent[k]
    for pair in pairs:
        keys = pair['dependency_keys']
        for key in keys[1:]: parent[find(key)] = find(keys[0])
    return {p['sample_id']: find(p['dependency_keys'][0]) for p in pairs}


def confidence_intervals(rows, resamples=2000, seed=42):
    rng = np.random.default_rng(seed); output = {}
    for category, names in [('positive', METRICS[:4]), ('negative', METRICS[4:])]:
        selected = [r for r in rows if r['category'] == category]
        groups = defaultdict(list)
        for row in selected: groups[row['bootstrap_group']].append(row)
        clusters = list(groups.values())
        if len(clusters) < 2:
            output.update({name: dict(low=None, high=None, units=len(clusters), reason='Fewer than two independent groups') for name in names})
            continue
        draws = {name: [] for name in names}
        for _ in range(resamples):
            sample = [r for i in rng.integers(0,len(clusters),len(clusters)) for r in clusters[i]]
            for name in names:
                values = [float(r['negative_false_positive']) if name=='negative_pair_fpr' else r[name] for r in sample]
                draws[name].append(float(np.mean(values)))
        output.update({name: dict(low=float(np.quantile(v,.025)), high=float(np.quantile(v,.975)), units=len(clusters)) for name,v in draws.items()})
    negatives = [r for r in rows if r['category']=='negative']
    independent = len({r['bootstrap_group'] for r in negatives}) == len(negatives)
    k, n = sum(r['negative_false_positive'] for r in negatives), len(negatives)
    output['negative_fpr_binomial'] = dict(available=independent, method='Clopper-Pearson exact, conditional on independent pairs',
        false_positive_pairs=int(k), denominator=n,
        low=(float(beta.ppf(.025,k,n-k+1)) if k else 0.) if independent else None,
        high=(float(beta.ppf(.975,k+1,n-k)) if k<n else 1.) if independent else None)
    return dict(method='pair bootstrap or connected manuscript/page/line groups', resamples=resamples, seed=seed,
        estimates=output, limitations='Grouping uses available source page/group IDs and repeated image identities. Missing metadata cannot establish manuscript independence; one-group intervals are unavailable.')


def source_identity():
    files = ['evaluate.py','alignment_candidates.py','evaluation_utils.py','dataset.py','dataloader.py',
             'model.py','cnn_encoder.py','transformer_encoder.py','text_embedding.py','dtw.py','parameters.py',
             'scripts/eval/mask_quantitative.py','scripts/eval/tune_masks.py', 'notebooks/model_evaluation.ipynb']
    return {name:file_hash(ROOT/name) for name in files}


def audit_splits(session):
    owners = defaultdict(set); page_owners = defaultdict(set); manuscripts = defaultdict(set); records = {}; missing = 0
    for split, view in session.views.items():
        records[split] = []
        for row in view.records:
            images = []
            for side in row['sides']:
                path = str(Path(side['image']).resolve())
                with Image.open(path) as image:
                    pixels = np.asarray(image.convert('RGB'))
                image_id = hashlib.sha256(str(pixels.shape).encode()+pixels.tobytes()).hexdigest()
                owners[image_id].add(split)
                group = side.get('group_id')
                if group: page_owners[str(group)].add(split)
                else: missing += 1
                if group and str(group).startswith('book:'):
                    manuscripts[str(group).rsplit(':',1)[0]].add(split)
                images.append(dict(path=path, pixel_sha256=image_id, group_id=group))
            records[split].append(dict(sample_id=row['sample_id'], group_id=row.get('group_id'), images=images))
    return dict(saved_split_ids=session.saved['split_ids'], records=records,
        image_leakage=[dict(image_sha256=k,splits=sorted(v)) for k,v in owners.items() if len(v)>1],
        page_group_leakage=[dict(group_id=k,splits=sorted(v)) for k,v in page_owners.items() if len(v)>1],
        manuscript_membership=[dict(manuscript_id=k,splits=sorted(v)) for k,v in sorted(manuscripts.items())],
        missing_group_records=missing,
        limitations='Page/group IDs are checked where supplied. Book/manuscript identity may span splits under the existing page-level policy; splits are never regenerated.')


def explicit_pairs(session, manifest):
    """Optional reviewed pair list, restricted to checkpoint-saved membership."""
    if manifest is None: return list(session.pairs)
    document = json.loads(Path(manifest).read_text())
    if document.get('schema_version') != 1: raise ValueError('Split manifest requires schema_version=1')
    identifiers=[p.get('sample_id') for p in document['pairs']]
    if any(not isinstance(i,str) or not i for i in identifiers) or len(set(identifiers))!=len(identifiers):
        raise ValueError('Explicit pair IDs must be nonempty and unique')
    if any(p.get('split') not in ('train','val','test') for p in document['pairs']):
        raise ValueError('Explicit pair splits must be train, val or test')
    rows = [p for p in document['pairs'] if p['split']==session.split]
    by_id = {p['sample_id']:p for p in session.pairs}
    allowed = {str(Path(v['side']['image']).resolve()):v for v in session.allowed.values()}
    pairs = []
    for row in rows:
        if row.get('verified') is not True: raise ValueError('Explicit pairs must have verified=true')
        if type(row.get('target')) is not int or row['target'] not in (0,1): raise ValueError('Explicit target must be 0 or 1')
        pair = dict(by_id.get(row['sample_id'], {}))
        if pair and 'sides' in row:
            paths=[str((Path(s['image']) if Path(s['image']).is_absolute() else session.dataset_path/s['image']).resolve()) for s in row['sides']]
            if paths!=[s['image'] for s in pair['sides']]:raise ValueError('Explicit sample ID and image paths disagree')
        if not pair:
            sides = []
            for side in row['sides']:
                path = Path(side['image']); path = path if path.is_absolute() else session.dataset_path/path
                entry = allowed.get(str(path.resolve()))
                if entry is None: raise ValueError('Explicit pair line is outside the saved split: '+str(path))
                sides.append(dict(entry['side']))
            pair = dict(sample_id=row['sample_id'],sides=sides,group_ids=[s.get('group_id') for s in sides])
        if len(row['masks']) != 2 or len(pair['sides']) != 2: raise ValueError('Every pair needs two lines and two explicit GT masks')
        annotations = []
        for value in row['masks']:
            path = Path(value); path = path if path.is_absolute() else session.dataset_path/path
            annotations.append(dict(mask=str(path.resolve())))
        pair.update(target=row['target'],annotations=annotations,constructed_negative=False,
                    label='manually_verified_'+('aligned' if row['target'] else 'unaligned'),manually_verified=True)
        pairs.append(pair)
    return pairs


def build_cache(args, split):
    """GT files are loaded separately; cached NPZ files contain no labels/masks."""
    from train import resolve_device
    if args.split_manifest and args.mask_source!='manifest':raise ValueError('Explicit pair manifests require --mask-source manifest')
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    output = Path(args.output); output.mkdir(parents=True,exist_ok=True)
    values, baseline = notebook_baseline(args.notebook)
    write_json(output/'baseline_configuration.json',dict(notebook_variables=values,**effective_settings(baseline)))
    checkpoint_before=file_hash(args.checkpoint)
    try:
        session = EvaluationSession(args.checkpoint,args.dataset,split=split,device=str(resolve_device(args.device)),
            seed=args.seed,pair_source=args.pair_source,construct_negatives=False,negative_count=0,
            crop_override=None if args.crop=='checkpoint' else args.crop)
    except Exception as exc:
        write_json(output/'split_audit.json',dict(status='FAILED_SESSION_CONTRACT',error=str(exc)))
        raise
    if checkpoint_before!=session.checkpoint_sha256:raise ValueError('Checkpoint changed during loading; use a frozen checkpoint copy')
    session.model.requires_grad_(False); session.text_encoder.requires_grad_(False)
    audit = audit_splits(session); write_json(output/'split_audit.json',audit)
    if audit['image_leakage'] or audit['page_group_leakage']:
        raise ValueError('Saved splits leak line images/page groups; see split_audit.json. No split was reorganized.')
    pixel_ids={image['path']:image['pixel_sha256'] for records in audit['records'].values() for row in records for image in row['images']}
    pairs = explicit_pairs(session,args.split_manifest)
    issues = []; selected = []; seen = set()
    for pair in sorted(pairs,key=lambda p:p['sample_id']):
        if pair['sample_id'] in seen: issues.append(dict(sample_id=pair['sample_id'],reason='duplicate pair ID')); continue
        seen.add(pair['sample_id'])
        if pair.get('constructed_negative') or pair.get('target') not in (0,1):
            issues.append(dict(sample_id=pair['sample_id'],reason='unverified or unknown pair label')); continue
        masks = []; provenance = []
        for i,side in enumerate(pair['sides']):
            path = pair['annotations'][i].get('mask')
            if args.mask_source=='folders':
                image = Path(side['image']); path = image.parent.parent/('masks'+'AB'[i])/(image.stem+'.png')
            path = Path(path) if path else None
            with Image.open(side['image']) as image: size = image.size
            if path is None or not path.is_file():
                issues.append(dict(sample_id=pair['sample_id'],side='AB'[i],reason='missing GT mask',path=str(path))); break
            with Image.open(path) as image:
                original_size = image.size
                if image.size != size:
                    if not args.resize_gt:
                        issues.append(dict(sample_id=pair['sample_id'],side='AB'[i],reason='GT/source size mismatch',path=str(path))); break
                    image = image.resize(size,Image.Resampling.NEAREST)
                masks.append(np.asarray(image.convert('L'))>0)
            provenance.append(dict(path=str(path.resolve()),sha256=file_hash(path),original_size=list(original_size),
                                   source_size=list(size),resized_nearest=original_size!=size))
        if len(masks)!=2: continue
        expected = bool(pair['target'])
        if any(bool(mask.any()) != expected for mask in masks):
            issues.append(dict(sample_id=pair['sample_id'],reason='label/GT inconsistency',target=pair['target'],foreground=[bool(m.any()) for m in masks])); continue
        groups = ['page:'+str(s['group_id']) for s in pair['sides'] if s.get('group_id')]
        groups += ['image:'+pixel_ids[session.line_aliases.get(_side_key(s),_side_key(s))[0]] for s in pair['sides']]
        record = dict(sample_id=pair['sample_id'],split=split,target=pair['target'],category='positive' if expected else 'negative',
            label=pair.get('label'),images=[dict(image=s['image'],sha256=file_hash(s['image']),bbox=s.get('bbox'),group_id=s.get('group_id')) for s in pair['sides']],
            masks=provenance,dependency_keys=groups)
        inference_sides=[{k:s[k] for k in ('image','text','raw_text','bbox','group_id') if k in s} for s in pair['sides']]
        selected.append(dict(inference=dict(sample_id=pair['sample_id'],sides=inference_sides),truth=tuple(masks),record=record))
    counts = Counter(item['record']['category'] for item in selected)
    units = dependency_units([p['record'] for p in selected])
    for item in selected:item['record']['bootstrap_group']=units[item['record']['sample_id']]
    contract = dict(split=split,counts={k:counts[k] for k in ('positive','negative')},inconsistent_records=issues,records=[p['record'] for p in selected],
                    eligible_catalog_pairs=len(pairs),saved_split_lines=session.split_sizes,
                    folder_catalog=session.folder_summary,
                    missing_mask_policy='error, never presumed black',max_pairs=0)
    write_json(output/f'{split}_split_manifest.json',contract)
    if issues: raise ValueError(f'{len(issues)} inconsistent/unverified records; see {split}_split_manifest.json. No examples silently excluded.')
    if not counts['positive'] or not counts['negative']:
        raise ValueError(f'{split} needs positive AND verified negative pairs with explicit GT masks; counts={dict(counts)}. Supply --split-manifest with reviewed negatives. Random unmatched lines are not negatives.')
    source = source_identity()
    vocabulary=list(dict.fromkeys(session.config.alphabet_inventory or ARABIC_LETTERS))
    prior=getattr(session.text_encoder,'letter_evidence_prior',None) or dict(source='explicit uniform fallback',
        vocabulary=vocabulary,prior=[1/len(vocabulary)]*len(vocabulary))
    provenance = dict(version=VERSION,matcher_version=MATCH_VERSION,checkpoint=str(session.checkpoint),checkpoint_sha256=session.checkpoint_sha256,dataset=str(session.dataset_path),
        config=asdict(session.config),representation=args.representation,split=split,seed=args.seed,device=str(session.device),
        mask_source=args.mask_source,resize_gt=args.resize_gt,bootstrap_resamples=args.bootstrap,
        notebook_path=str(Path(args.notebook).resolve()),notebook_sha256=file_hash(args.notebook),
        inference_mode='eval/no_grad, frozen model and character vectors, augmentation=False',
        letter_evidence_prior=prior,
        source_sha256=source,split_manifest_sha256=identity(contract),notebook_variables=values,baseline_settings=baseline,
        split_audit_sha256=identity(audit),packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')})
    try: provenance['git_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    except subprocess.CalledProcessError: provenance['git_commit'] = None
    cache = Path(args.cache_dir or output/'cache'); cache.mkdir(parents=True,exist_ok=True)
    start = time.monotonic(); cache_index=[]
    for item in selected:
        record = item['record']; record['bootstrap_group'] = units[record['sample_id']]
        # No GT label/path/hash is part of the frozen inference file contents.
        key = identity(dict(checkpoint=provenance['checkpoint_sha256'],config=provenance['config'],representation=args.representation,
            evidence={k:baseline[k] for k in ('acceptance_offset','prior_floor')},sources=source,packages=provenance['packages'],
            pair_id=record['sample_id'],images=record['images'],seed=args.seed,device=str(session.device)))
        path = cache/(key+'.npz'); meta = cache/(key+'.json')
        if path.exists() and meta.exists():
            info = json.loads(meta.read_text())
            if info['key'] != key or info['npz_sha256'] != file_hash(path): raise ValueError('Frozen feature cache is corrupt: '+str(path))
            with np.load(path,allow_pickle=False) as arrays:
                lines = [dict(features={r:torch.from_numpy(arrays[f'{i}_{r}'].copy()) for r in ('local','context','fused')},
                    physical=arrays[f'{i}_physical'].copy(),token_valid=arrays[f'{i}_valid'].copy(),
                    full_physical=arrays[f'{i}_full_physical'].copy(),geometry=info['geometry'][i]) for i in range(2)]
        else:
            lines = [session.get_line_features(side) for side in item['inference']['sides']]
            arrays = {}
            for i,line in enumerate(lines):
                for name,features in line['features'].items(): arrays[f'{i}_{name}'] = features.numpy()
                arrays.update({f'{i}_physical':line['physical'],f'{i}_valid':line['token_valid'],f'{i}_full_physical':line['full_physical']})
            temporary = path.with_suffix('.tmp')
            with temporary.open('wb') as stream: np.savez_compressed(stream,**arrays)
            temporary.replace(path)
            write_json(meta,dict(key=key,pair_id=record['sample_id'],npz_sha256=file_hash(path),geometry=[l['geometry'] for l in lines]))
        for line,truth in zip(lines,item['truth']):
            assert tuple(reversed(line['geometry']['source_size'])) == truth.shape
            if not np.array_equal(line['physical'],line['full_physical'][line['token_valid'].astype(bool)]):
                raise ValueError('Cache changed valid physical window identities')
        item['lines'] = lines
        item['cosine'] = cosine_similarity_matrix(*(l['features'][args.representation] for l in lines)).numpy()
        cache_index.append(dict(sample_id=record['sample_id'],key=key,npz=str(path.resolve()),
            valid_physical=[l['physical'].tolist() for l in lines],
            padded_tokens=[int((~l['token_valid'].astype(bool)).sum()) for l in lines],geometry=[l['geometry'] for l in lines]))
    provenance['cache_seconds'] = time.monotonic()-start
    if file_hash(session.checkpoint)!=checkpoint_before:raise ValueError('Checkpoint changed during feature extraction; use a frozen checkpoint copy')
    write_json(output/f'{split}_provenance.json',provenance)
    write_json(output/f'{split}_cache_index.json',cache_index)
    return session, selected, provenance


def evaluate(session, prepared, representation, requested, output=None, bootstrap=False, resamples=2000, seed=42):
    started=time.monotonic()
    resolved = effective_settings(requested); settings = resolved['settings']
    rows = []; line_rows = []; diagnostics = []
    for item in prepared:
        # EXACT notebook flow: all extension, merging, filling and source mapping included.
        prediction = predict_cached_pair(session,item['inference'],item['lines'],representation,
                                          settings=settings,cosine=item['cosine'])
        record = item['record']; category = record['category']; scores = []
        for i,(mask,truth) in enumerate(zip(prediction['masks'],item['truth'])):
            score = binary_metrics(mask,truth)
            region = region_counts(mask,truth) if score['full_height'] else dict(tp=0,fp=0,fn=0,matches=[])
            score.update(region_tp=region['tp'],region_fp=region['fp'],region_fn=region['fn'])
            scores.append(score)
            line_rows.append(dict(sample_id=record['sample_id'],side='AB'[i],category=category,**score))
        candidates = prediction['match']['candidates']; regions = prediction['match']['regions']
        row = dict(sample_id=record['sample_id'],category=category,bootstrap_group=record['bootstrap_group'],
            **{name:float(np.mean([s[key] for s in scores])) for name,key in
               [('positive_iou','iou'),('positive_dice','dice'),('precision','precision'),('recall','recall')]},
            negative_false_positive=any(s['predicted_pixels']>0 for s in scores),
            negative_masked_area_fraction=float(np.mean([s['foreground_fraction'] for s in scores])),
            candidate_limit_hit=bool(prediction['match']['candidate_limit_reached']),
            region_eligible=all(s['full_height'] for s in scores),
            **{'region_'+k:sum(s['region_'+k] for s in scores) for k in ('tp','fp','fn')})
        rows.append(row)
        distributions = {k:[float(c[k]) for c in candidates if c.get(k) is not None] for k in ('path_density','mutual_anchor_fraction','mean_reward')}
        diagnostics.append(dict(sample_id=record['sample_id'],candidate_limit_hit=row['candidate_limit_hit'],
            accepted_normal=sum(c['candidate_type']=='normal' for c in regions),accepted_short=sum(c['candidate_type']=='short' for c in regions),
            rejections=dict(Counter(c['reason'] for c in prediction['match']['rejected'])),
            extended_candidates=sum(bool(c.get('extension_pairs')) for c in candidates),
            accepted_extensions=sum(bool(c.get('extension_pairs')) for c in regions),
            merged_candidates=sum(bool(c.get('merged_from')) for c in candidates),
            merge_outcomes=dict(Counter(c['reason'] for c in candidates if c['reason'].startswith('merged_into_'))),
            distributions=distributions))
    summary = aggregate(rows)
    summary['diagnostics'] = dict(candidate_limit_hits=sum(r['candidate_limit_hit'] for r in rows),
        accepted_normal=sum(d['accepted_normal'] for d in diagnostics),accepted_short=sum(d['accepted_short'] for d in diagnostics),
        rejection_reasons=dict(sum((Counter(d['rejections']) for d in diagnostics),Counter())))
    if bootstrap: summary['confidence_intervals'] = confidence_intervals(rows,resamples,seed)
    summary['evaluation_seconds']=time.monotonic()-started
    if output:
        output = Path(output); output.mkdir(parents=True,exist_ok=True)
        write_csv(output/'per_pair.csv',rows); write_csv(output/'per_line.csv',line_rows)
        write_json(output/'aggregate.json',summary); write_csv(output/'aggregate.csv',[{k:v for k,v in summary.items() if not isinstance(v,dict)}])
        write_json(output/'diagnostics.json',diagnostics); write_json(output/'requested_settings.json',requested)
        write_json(output/'effective_settings.json',resolved)
    return summary, rows
