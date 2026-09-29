"""One evaluator: checkpoint-faithful split loss or image-only shared regions."""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import uuid

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

from dataloader import collate_samples
from dataset import prepare_image, file_hash
from dtw import cosine_similarity_matrix
from train import build_loaders, load_checkpoint, validate_one_epoch


def smith_waterman_affine(scores, gap_open=.2, gap_extend=.05):
    """Local starts/ends on BOTH lines; gap cost open+(length-1)*extend."""
    scores = np.asarray(scores,dtype=np.float64)
    if scores.ndim!=2 or np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError('Require a finite reward matrix (or forbidden -inf cells)')
    if not gap_open >= gap_extend >= 0:
        raise ValueError('Require gap_open >= gap_extend >= 0')
    n,m = scores.shape
    h = np.zeros((n+1,m+1))
    up,left = np.full_like(h,-np.inf),np.full_like(h,-np.inf)
    trace = np.zeros(h.shape,dtype=np.uint8)
    ue,le = np.zeros(h.shape,dtype=bool),np.zeros(h.shape,dtype=bool)
    for i in range(1,n+1):
        for j in range(1,m+1):
            ue[i,j] = up[i-1,j]-gap_extend > h[i-1,j]-gap_open
            le[i,j] = left[i,j-1]-gap_extend > h[i,j-1]-gap_open
            up[i,j] = up[i-1,j]-gap_extend if ue[i,j] else h[i-1,j]-gap_open
            left[i,j] = left[i,j-1]-gap_extend if le[i,j] else h[i,j-1]-gap_open
            choices = (0.,h[i-1,j-1]+scores[i-1,j-1],up[i,j],left[i,j])
            direction = int(np.argmax(choices))
            h[i,j],trace[i,j] = choices[direction],direction
    i,j = map(int,np.unravel_index(np.argmax(h),h.shape))
    best,state,steps = float(h[i,j]),0,[]
    while i>0 or j>0:
        if state==0:
            if h[i,j]<=0: break
            state=int(trace[i,j])
            if state==1:
                steps.append((i-1,j-1)); i,j,state=i-1,j-1,0
                continue
            if state==0: break
        if state==2:
            steps.append((i-1,None)); state=2 if ue[i,j] else 0; i-=1
        elif state==3:
            steps.append((None,j-1)); state=3 if le[i,j] else 0; j-=1
    return list(reversed(steps)),best


def shared_regions(cosine, physical_a, physical_b, threshold=.6, contrast_margin=.05,
                   score_mode='background', min_windows=5, max_gap=1,
                   gap_open=.2, gap_extend=.05, max_candidates=128):
    """Greedy noncrossing regions; positive anchors, NOT entire traceback spans."""
    cosine = np.asarray(cosine)
    if cosine.ndim!=2 or not np.isfinite(cosine).all():
        raise ValueError('Cosines must be a finite 2D matrix')
    if score_mode not in {'background','raw'} or min_windows<1 or max_gap<0 or max_candidates<1:
        raise ValueError('Invalid scoring/support settings')
    if not (-1<=threshold<=1 and contrast_margin>=0 and gap_open>=gap_extend>=0):
        raise ValueError('Invalid threshold, contrast or gap penalties')
    physical = [np.asarray(p,dtype=int) for p in (physical_a,physical_b)]
    for count,p in zip(cosine.shape,physical):
        if p.shape!=(count,) or len(set(p.tolist()))!=count or (p<0).any():
            raise ValueError('Physical identities must be unique nonnegative valid windows')
        if count>1 and not ((np.diff(p)>0).all() or (np.diff(p)<0).all()):
            raise ValueError('Physical identities must preserve sequence order')
    baseline = threshold
    if score_mode=='background' and cosine.size:
        baseline = np.maximum(threshold,np.median(cosine,axis=1)[:,None]+contrast_margin)
        baseline = np.maximum(baseline,np.median(cosine,axis=0)[None,:]+contrast_margin)
    rewards = cosine.astype(np.float64)-baseline
    allowed = np.ones(cosine.shape,dtype=bool)
    rows,cols = np.indices(cosine.shape)
    regions,rejected = [],[]
    for attempt in range(max_candidates):
        steps,best = smith_waterman_affine(np.where(allowed,rewards,-np.inf),gap_open,gap_extend)
        if best<=0: break
        anchors=[(k,i,j) for k,(i,j) in enumerate(steps) if i is not None and j is not None and rewards[i,j]>0]
        groups=[]
        for anchor in anchors:
            adjacent=bool(groups)
            if groups:
                previous=groups[-1][-1]
                for s,p in enumerate(physical,1):
                    lo,hi=sorted((int(p[previous[s]]),int(p[anchor[s]])))
                    adjacent &= hi-lo-1<=max_gap and set(range(lo,hi+1)).issubset(set(p.tolist()))
            if adjacent: groups[-1].append(anchor)
            else: groups.append([anchor])
        for group in groups:
            pairs=[(i,j) for _,i,j in group]
            support=[len({int(p[pair[s]]) for pair in pairs}) for s,p in enumerate(physical)]
            total,previous=0.,None
            for i,j in steps[group[0][0]:group[-1][0]+1]:
                kind='a' if i is None else 'b' if j is None else None
                total += float(rewards[i,j]) if kind is None else -(gap_extend if kind==previous else gap_open)
                previous=kind
            if min(support)<min_windows or total<=0:
                rejected.append(dict(reason='insufficient_positive_support' if min(support)<min_windows else 'nonpositive_trimmed_objective',support=support))
                continue
            spans=[(min(pair[s] for pair in pairs),max(pair[s] for pair in pairs)) for s in (0,1)]
            supported=[sorted({int(p[pair[s]]) for pair in pairs}) for s,p in enumerate(physical)]
            filled=[sorted(set(range(min(ids),max(ids)+1))-set(ids)) for ids in supported]
            regions.append(dict(pairs=pairs,logical_ranges=spans,supported_physical=supported,
                                filled_physical=filled,support=support,score=total))
            (a,b),(c,d)=spans
            allowed &= ((rows<a)&(cols<c))|((rows>b)&(cols>d))
        for _,i,j in anchors: allowed[i,j]=False
    else:
        rejected.append(dict(reason='candidate_limit_reached'))
    if not regions and not rejected: rejected.append(dict(reason='no_positive_local_alignment'))
    return dict(regions=sorted(regions,key=lambda r:r['logical_ranges'][0][0]),rejected=rejected,rewards=rewards)


def _local_repeat_path(cosine, rewards, allowed, repeat_penalty, max_consecutive_repeats,
                       gap_open, gap_extend):
    """Best local path with matched repeat states and separate affine skip states."""
    n, m = rewards.shape
    run = max_consecutive_repeats
    diagonal, gap_a, gap_b = 0, 2 * run + 1, 2 * run + 2
    states = gap_b + 1
    scores = np.full((n, m, states), -np.inf, dtype=np.float64)
    previous = np.full((n, m, states), -1, dtype=np.int16)
    best_score, endpoint = 0., None
    match_states = range(gap_a)
    for i in range(n):
        for j in range(m):
            if not allowed[i, j]:
                continue
            reward = float(rewards[i, j])
            # True gaps have no pair reward and never become matched support.
            if i:
                candidates = [(scores[i-1, j, state] - gap_open, state)
                              for state in match_states]
                candidates.append((scores[i-1, j, gap_a] - gap_extend, gap_a))
                value, state = max(candidates, key=lambda item: item[0])
                if value > 0:
                    scores[i, j, gap_a], previous[i, j, gap_a] = value, state
            if j:
                candidates = [(scores[i, j-1, state] - gap_open, state)
                              for state in match_states]
                candidates.append((scores[i, j-1, gap_b] - gap_extend, gap_b))
                value, state = max(candidates, key=lambda item: item[0])
                if value > 0:
                    scores[i, j, gap_b], previous[i, j, gap_b] = value, state
            # A matched cell must itself beat the threshold/background baseline.
            if reward <= 0:
                continue
            value, state = reward, -1  # local restart
            if i and j:
                predecessor = int(np.argmax(scores[i-1, j-1]))
                candidate = float(scores[i-1, j-1, predecessor]) + reward
                if candidate > value:
                    value, state = candidate, predecessor
            scores[i, j, diagonal], previous[i, j, diagonal] = value, state
            # Repeat runs have an increasing incremental penalty. A repeat is
            # admitted only when this pair's own evidence pays that penalty.
            if j:
                for length in range(1, run + 1):
                    penalty = repeat_penalty * length
                    if reward <= penalty:
                        continue
                    if length == 1:
                        sources = [diagonal, *range(run + 1, gap_a), gap_a]
                        predecessor = max(sources, key=lambda s: scores[i, j-1, s])
                    else:
                        predecessor = length - 1
                    candidate = float(scores[i, j-1, predecessor]) + reward - penalty
                    if candidate > 0:
                        scores[i, j, length] = candidate
                        previous[i, j, length] = predecessor
            if i:
                for length in range(1, run + 1):
                    penalty = repeat_penalty * length
                    if reward <= penalty:
                        continue
                    state_index = run + length
                    if length == 1:
                        sources = [diagonal, *range(1, run + 1), gap_b]
                        predecessor = max(sources, key=lambda s: scores[i-1, j, s])
                    else:
                        predecessor = run + length - 1
                    candidate = float(scores[i-1, j, predecessor]) + reward - penalty
                    if candidate > 0:
                        scores[i, j, state_index] = candidate
                        previous[i, j, state_index] = predecessor
            for state_index in match_states:
                if scores[i, j, state_index] > best_score:
                    best_score = float(scores[i, j, state_index])
                    endpoint = (i, j, state_index)
    if endpoint is None:
        return [], 0.
    path = []
    i, j, state = endpoint
    while state >= 0:
        predecessor = int(previous[i, j, state])
        if state == diagonal:
            transition = 'diagonal'
            step = dict(i=i, j=j, transition=transition,
                        cosine=float(cosine[i, j]), reward=float(rewards[i, j]))
            i, j = i - 1, j - 1
        elif state <= run:
            transition = 'horizontal_repeat'
            step = dict(i=i, j=j, transition=transition,
                        cosine=float(cosine[i, j]), reward=float(rewards[i, j]))
            j -= 1
        elif state <= 2 * run:
            transition = 'vertical_repeat'
            step = dict(i=i, j=j, transition=transition,
                        cosine=float(cosine[i, j]), reward=float(rewards[i, j]))
            i -= 1
        elif state == gap_a:
            step = dict(i=i, j=None, transition='gap_a', cosine=None, reward=None)
            i -= 1
        else:
            step = dict(i=None, j=j, transition='gap_b', cosine=None, reward=None)
            j -= 1
        path.append(step)
        state = predecessor
    return path[::-1], best_score


def _small_internal_fills(supported, skipped, max_gap):
    filled = []
    for left, right in zip(supported, supported[1:]):
        missing = range(left + 1, right)
        if len(missing) <= max_gap:
            filled.extend(p for p in missing if p not in skipped)
    return filled


def local_repeat_regions(cosine, physical_a, physical_b, threshold=.6,
                         contrast_margin=.05, score_mode='background', min_windows=5,
                         max_gap=1, gap_open=.2, gap_extend=.05, max_candidates=128,
                         repeat_penalty=.05, max_consecutive_repeats=3,
                         min_distinct_windows_a=3, min_distinct_windows_b=3,
                         min_matched_pairs=4):
    """Greedy noncrossing local many-to-many alignment on image cosine scores.

    Smith–Waterman remains in shared_regions. Repeats are matched cells, while
    affine gap states skip a window and are excluded from mask support.
    """
    cosine = np.asarray(cosine, dtype=np.float64)
    if cosine.ndim != 2 or not np.isfinite(cosine).all() or min(cosine.shape, default=0) == 0:
        raise ValueError('Cosines must be a nonempty finite 2D matrix')
    if score_mode not in {'background', 'raw'} or max_gap < 0 or max_candidates < 1:
        raise ValueError('Invalid scoring/support settings')
    if not (-1 <= threshold <= 1 and contrast_margin >= 0 and gap_open >= gap_extend >= 0):
        raise ValueError('Invalid threshold, contrast or gap penalties')
    if (repeat_penalty < 0 or max_consecutive_repeats < 1 or
            min_distinct_windows_a < 1 or min_distinct_windows_b < 1 or min_matched_pairs < 1):
        raise ValueError('Invalid repeat penalty, run limit or minimum support')
    physical = [np.asarray(p, dtype=int) for p in (physical_a, physical_b)]
    for count, indices in zip(cosine.shape, physical):
        if (indices.shape != (count,) or len(set(indices.tolist())) != count or
                (indices < 0).any() or
                (count > 1 and not ((np.diff(indices) > 0).all() or (np.diff(indices) < 0).all()))):
            raise ValueError('Physical identities must be unique and preserve sequence order')
    baseline = threshold
    if score_mode == 'background':
        baseline = np.maximum(threshold, np.median(cosine, axis=1)[:, None] + contrast_margin)
        baseline = np.maximum(baseline, np.median(cosine, axis=0)[None, :] + contrast_margin)
    rewards = cosine - baseline
    allowed = np.ones(cosine.shape, dtype=bool)
    rows, cols = np.indices(cosine.shape)
    regions, rejected = [], []
    for _ in range(max_candidates):
        path, score = _local_repeat_path(cosine, rewards, allowed, repeat_penalty,
                                         max_consecutive_repeats, gap_open, gap_extend)
        if not path:
            break
        matched = [step for step in path if step['i'] is not None and step['j'] is not None]
        pairs = [(step['i'], step['j']) for step in matched]
        ids = [sorted({int(physical[side][pair[side]]) for pair in pairs}) for side in (0, 1)]
        counts = [len(side) for side in ids]
        if (counts[0] < min_distinct_windows_a or counts[1] < min_distinct_windows_b or
                len(pairs) < min_matched_pairs):
            rejected.append(dict(reason='insufficient_repeat_support',
                                 support=counts, matched_pairs=len(pairs)))
            for i, j in pairs:
                allowed[i, j] = False
            continue
        spans = [(min(pair[side] for pair in pairs), max(pair[side] for pair in pairs))
                 for side in (0, 1)]
        skipped = [set(), set()]
        for step in path:
            if step['transition'] == 'gap_a':
                skipped[0].add(int(physical[0][step['i']]))
            elif step['transition'] == 'gap_b':
                skipped[1].add(int(physical[1][step['j']]))
        filled = [_small_internal_fills(ids[side], skipped[side], max_gap) for side in (0, 1)]
        repeats_h = sum(step['transition'] == 'horizontal_repeat' for step in path)
        repeats_v = sum(step['transition'] == 'vertical_repeat' for step in path)
        gaps = sum(step['transition'] in {'gap_a', 'gap_b'} for step in path)
        regions.append(dict(pairs=pairs, path_steps=path,
                            transition_types=[step['transition'] for step in matched],
                            logical_ranges=spans, supported_physical=ids,
                            filled_physical=filled, support=counts, score=score,
                            matched_pairs=len(pairs), horizontal_repeats=repeats_h,
                            vertical_repeats=repeats_v, true_gaps=gaps))
        (a, b), (c, d) = spans
        allowed &= ((rows < a) & (cols < c)) | ((rows > b) & (cols > d))
    else:
        rejected.append(dict(reason='candidate_limit_reached'))
    if not regions and not rejected:
        rejected.append(dict(reason='no_positive_local_alignment'))
    return dict(regions=sorted(regions, key=lambda r: r['logical_ranges'][0][0]),
                rejected=rejected, rewards=rewards)


def source_interval(physical,geometry,window_width,stride):
    x0,_,x1,_=geometry['crop']
    return [max(x0,x0+physical*stride/geometry['scale_x']),
            min(x1,x0+(physical*stride+window_width)/geometry['scale_x'])]


def region_mask(regions,side,geometry,window_width,stride):
    width,height=geometry['source_size']
    mask=np.zeros((height,width),dtype=np.uint8)
    for region in regions:
        for physical in region['supported_physical'][side]+region['filled_physical'][side]:
            a,b=source_interval(physical,geometry,window_width,stride)
            mask[:,max(0,math.floor(a)):min(width,math.ceil(b))]=255
    return mask


def score_mask(prediction,ground_truth=None):
    result=dict(status='unavailable',reason='no_source_size_ground_truth',coverage=float((prediction>0).mean()))
    if ground_truth is None: return result
    with Image.open(ground_truth) as image: gt=np.asarray(image.convert('L'))>=128
    pred=prediction>0
    if pred.shape!=gt.shape: raise ValueError('Ground-truth/source geometry mismatch; resizing is forbidden')
    tp=np.logical_and(pred,gt).sum(); union=np.logical_or(pred,gt).sum()
    precision=float(tp/pred.sum()) if pred.any() else 0.
    recall=float(tp/gt.sum()) if gt.any() else 0.
    result.update(status='available',reason=None,iou=float(tp/union) if union else 1.,
                  precision=precision,recall=recall,f1=2*precision*recall/(precision+recall) if precision+recall else 0.,
                  negative_false_positive_coverage=float(pred.mean()) if not gt.any() else None)
    return result


def evaluate_loss(checkpoint,dataset,split='test',device='cpu',max_batches=0):
    model,text,config,saved=load_checkpoint(checkpoint,device)
    loaders=build_loaders(dataset,replace(config,augmentation=False),saved['split_ids'])
    selected=loaders[('train','val','test').index(split)].dataset
    loader=DataLoader(selected,batch_size=config.batch_size,shuffle=False,num_workers=config.num_workers,collate_fn=collate_samples)
    stats=validate_one_epoch(model,text,loader,config,device,max_batches)
    return dict(split=split,checkpoint=str(checkpoint),epoch=saved['epoch'],metrics=stats,
                split_manifest_sha256=saved['split_manifest_sha256'])


def evaluate_pair(checkpoint,paths,output,device='cpu',representation='fused',crop_override=None,
                  ground_truth=(None,None),selection=None,**settings):
    model,_text,config,saved=load_checkpoint(checkpoint,device)
    if representation not in {'local','context','fused'}: raise ValueError('Unknown representation')
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    features,physical,geometry,originals=[],[],[],[]
    for side,path in enumerate(paths):
        # No transcript or alignment annotations enter this prediction path.
        image,tensor,geo=prepare_image(path,(config.image_height,config.image_width),config.grayscale,
                                       crop_override or config.crop,binarize=config.binarize)
        with Image.open(path) as source: original=source.convert('RGB')
        originals.append(original)
        original.save(output/f'line_{"ab"[side]}_original.png')
        image.save(output/f'line_{"ab"[side]}_model_input.png')
        with torch.no_grad():
            bundle=model(tensor[None].to(device))
        valid=bundle['token_valid'][0]
        features.append(bundle[representation][0][valid])
        physical.append(bundle['physical_window_indices'][0][valid].cpu().numpy())
        geometry.append(geo)
    cosine=cosine_similarity_matrix(*features).cpu().numpy()
    match=shared_regions(cosine,*physical,**settings)
    np.save(output/'cosine.npy',cosine)
    np.save(output/'alignment_scores.npy',match['rewards'])
    for name,matrix in [('cosine',cosine),('alignment_scores',match['rewards'])]:
        fig,ax=plt.subplots(figsize=(9,7))
        heat=ax.imshow(matrix,origin='lower',aspect='auto',cmap='coolwarm',
                       **dict(vmin=-1,vmax=1) if name=='cosine' else {})
        for region in match['regions']:
            a,b=zip(*region['pairs']);ax.plot(b,a,'k.',markersize=3)
        order='RTL' if config.rtl else 'LTR'
        ax.set(xlabel=f'Line B logical windows ({order})',ylabel=f'Line A logical windows ({order})',
               title='Raw cosine similarity' if name=='cosine' else 'Alignment rewards (not cosine/probability)')
        fig.colorbar(heat,ax=ax);fig.tight_layout();fig.savefig(output/f'{name}_heatmap.png');plt.close(fig)
    masks=[]
    for side in (0,1):
        mask=region_mask(match['regions'],side,geometry[side],config.window_width,config.window_stride)
        masks.append(mask);Image.fromarray(mask).save(output/f'line_{"ab"[side]}_mask.png')
        pixels=np.array(originals[side]);active=mask>0
        pixels[active]=(pixels[active]*.6+np.array([255,90,0])*.4).astype(np.uint8)
        Image.fromarray(pixels).save(output/f'line_{"ab"[side]}_overlay.png')
    with (output/'correspondences.csv').open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['region','logical_a','logical_b','physical_a','physical_b','cosine','reward','source_a_x0','source_a_x1','source_b_x0','source_b_x1'])
        for number,region in enumerate(match['regions']):
            for i,j in region['pairs']:
                pa,pb=int(physical[0][i]),int(physical[1][j])
                writer.writerow([number,i,j,pa,pb,float(cosine[i,j]),float(match['rewards'][i,j]),
                    *source_interval(pa,geometry[0],config.window_width,config.window_stride),
                    *source_interval(pb,geometry[1],config.window_width,config.window_stride)])
    report=dict(checkpoint=str(checkpoint),checkpoint_sha256=file_hash(checkpoint),epoch=saved['epoch'],
                config=saved['config'],images=list(map(str,paths)),representation=representation,
                geometry=geometry,settings=settings,regions=match['regions'],rejected=match['rejected'],
                selection=selection,crop_override=crop_override,
                metrics=[score_mask(mask,gt) for mask,gt in zip(masks,ground_truth)],
                limitations=['Greedy, not globally optimal, multi-region local alignment.',
                    'Uncalibrated thresholds; calibrate on validation positives and negatives, not test.',
                    'Background correction may suppress broad/repeated genuine matches.',
                    'Window widths vary in source pixels; one-to-one diagonals plus gaps are a limitation.',
                    'Overlapping window footprints may touch; region overlap is not character alignment accuracy.',
                    'Predictions are image-only, without DTW text prior or alignment ground truth.'])
    (output/'metadata.json').write_text(json.dumps(report,indent=2))
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--split',choices=['train','val','test'],default='test')
    parser.add_argument('--mode',choices=['loss','shared_regions'],default='loss')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--max-batches',type=int,default=0)
    parser.add_argument('--image-a');parser.add_argument('--image-b')
    parser.add_argument('--record-indices',nargs=2,type=int)
    parser.add_argument('--output')
    parser.add_argument('--representation',choices=['fused','local','context'],default='fused')
    parser.add_argument('--crop-override',choices=['none'],help='Explicit full-source preprocessing ablation')
    parser.add_argument('--gt-a');parser.add_argument('--gt-b')
    parser.add_argument('--threshold',type=float,default=.6)
    parser.add_argument('--score-mode',choices=['background','raw'],default='background')
    parser.add_argument('--contrast-margin',type=float,default=.05)
    parser.add_argument('--min-windows',type=int,default=5)
    parser.add_argument('--max-gap',type=int,default=1,help='0=strict consecutive positive anchors')
    parser.add_argument('--gap-open',type=float,default=.2)
    parser.add_argument('--gap-extend',type=float,default=.05)
    args=parser.parse_args(argv)
    if args.mode=='loss':
        if not args.dataset: parser.error('loss mode requires --dataset')
        report=evaluate_loss(args.checkpoint,args.dataset,args.split,args.device,args.max_batches)
        print(json.dumps(report,indent=2));return report
    selection=None
    if args.record_indices is not None:
        if not args.dataset or args.image_a or args.image_b: parser.error('Use saved record indices OR explicit images')
        _,_,config,saved=load_checkpoint(args.checkpoint,args.device)
        view=build_loaders(args.dataset,replace(config,augmentation=False),saved['split_ids'])[('train','val','test').index(args.split)].dataset
        if any(i<0 or i>=len(view) for i in args.record_indices): parser.error('Record index outside saved split')
        records=[view.records[i] for i in args.record_indices]
        paths=[r['sides'][0]['image'] for r in records]
        selection=dict(split=args.split,sample_ids=[r['sample_id'] for r in records],
                       split_manifest_sha256=saved['split_manifest_sha256'])
    else:
        if not args.image_a or not args.image_b: parser.error('Provide both --image-a and --image-b')
        paths=[args.image_a,args.image_b]
    settings={k:getattr(args,k) for k in ('threshold','score_mode','contrast_margin','min_windows','max_gap','gap_open','gap_extend')}
    output=args.output or str(Path('Results/simple_alignment')/uuid.uuid4().hex[:12])
    report=evaluate_pair(args.checkpoint,paths,output,args.device,args.representation,args.crop_override,
                         (args.gt_a,args.gt_b),selection,**settings)
    print(f'Saved {len(report["regions"])} shared regions to {output}')
    return report


if __name__=='__main__':
    main()
