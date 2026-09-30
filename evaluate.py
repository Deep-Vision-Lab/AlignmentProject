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
from dtw import cosine_similarity_matrix, alphabet_log_probabilities, letter_evidence_matrix
from text_embedding import ARABIC_LETTERS
from train import build_loaders, load_checkpoint, validate_one_epoch

MATCH_VERSION = 'local-stretch-1'
MATCH_DEFAULTS = dict(similarity_mode='cosine', decoder='stretch', threshold=.6,
    contrast_margin=.05, score_mode='background', min_windows=3, max_gap=1,
    gap_open=.2, gap_extend=.05, max_candidates=128, repeat_penalty=.05,
    max_stretch=3, acceptance_offset=.1, prior_floor=1e-4)


def resolved_match_settings(settings):
    unknown = set(settings) - set(MATCH_DEFAULTS)
    if unknown:
        raise ValueError(f'Unknown matching settings: {sorted(unknown)}')
    result = dict(MATCH_DEFAULTS, **settings)
    if any(not math.isfinite(v) for v in result.values() if isinstance(v,(float,int))):
        raise ValueError('Matching settings must be finite')
    if not -1 <= result['threshold'] <= 1 or result['contrast_margin']<0:
        raise ValueError('Cosine threshold must be in [-1,1] and contrast margin nonnegative')
    if result['acceptance_offset']<0:
        raise ValueError('acceptance_offset must be nonnegative')
    return result


def stretch_candidates(rewards, gap_open=.2, gap_extend=.05, repeat_penalty=.05,
                       max_stretch=3, max_candidates=128):
    """Local DP with bounded one-to-many matches, separately from affine skips.

    A diagonal consumes two new windows (full evidence). A repeat consumes one
    new window (half evidence minus penalty). Repeats cannot change direction
    without a diagonal: no reward-collecting zigzag. This is an experimental
    window-level objective, not a probability or character-boundary decoder.
    """
    r = np.asarray(rewards, dtype=float)
    if r.ndim != 2 or not np.isfinite(r).all():
        raise ValueError('Require finite rectangular rewards')
    if not gap_open >= gap_extend >= 0 or repeat_penalty < 0 or max_stretch < 1:
        raise ValueError('Invalid gap/repeat penalties or max_stretch')
    n, m = r.shape
    # state 0 diagonal; 1..R repeat A; R+1..2R repeat B; last two true skips.
    R = max_stretch - 1
    ga, gb, states = 2*R+1, 2*R+2, 2*R+3
    dp = np.full((n+1,m+1,states), -np.inf)
    back = np.full((n+1,m+1,states,3), -1, dtype=int)
    endpoints = []
    def put(i,j,s,choices,gain):
        value, predecessor = max(choices, key=lambda x:x[0])
        if value + gain > 0:
            dp[i,j,s] = value + gain
            back[i,j,s] = predecessor
    for i in range(1,n+1):
        for j in range(1,m+1):
            put(i,j,0, [(0.,(-1,-1,-1))] + [(dp[i-1,j-1,s],(i-1,j-1,s)) for s in range(states)], r[i-1,j-1])
            if r[i-1,j-1] > 0:
                for k in range(1,R+1):
                    put(i,j,k,[(dp[i-1,j,k-1],(i-1,j,k-1))], .5*r[i-1,j-1]-repeat_penalty)
                    prev = 0 if k == 1 else R+k-1
                    put(i,j,R+k,[(dp[i,j-1,prev],(i,j-1,prev))], .5*r[i-1,j-1]-repeat_penalty)
            put(i,j,ga,[(dp[i-1,j,s]-(gap_extend if s==ga else gap_open),(i-1,j,s))
                         for s in list(range(2*R+1))+[ga]],0.)
            put(i,j,gb,[(dp[i,j-1,s]-(gap_extend if s==gb else gap_open),(i,j-1,s))
                         for s in list(range(2*R+1))+[gb]],0.)
            for s in range(2*R+1):
                if dp[i,j,s] > 0:
                    endpoints.append((float(dp[i,j,s]),i,j,s))
    # Retain competing endpoints BEFORE suppression. Report truncation explicitly.
    candidates = []
    for score,i,j,s in sorted(endpoints, reverse=True)[:max_candidates]:
        path = []
        while i >= 0 and s >= 0:
            pi,pj,ps = back[i,j,s]
            previous_score = dp[pi,pj,ps] if ps >= 0 else 0.
            kind = 'gap_a' if s==ga else 'gap_b' if s==gb else 'repeat_a' if 0<s<=R else 'repeat_b' if R<s<=2*R else 'match'
            path.append(dict(a=i-1 if s!=gb else None, b=j-1 if s!=ga else None,
                             kind=kind, delta=float(dp[i,j,s]-previous_score)))
            i,j,s = int(pi),int(pj),int(ps)
        path.reverse()
        assert np.isclose(sum(step['delta'] for step in path),score)
        candidates.append(dict(path=path, score=score))
    return candidates, len(endpoints) > max_candidates


def decode_stretched_regions(rewards, physical_a, physical_b, *, min_windows=3, max_gap=1,
                             gap_open=.2, gap_extend=.05, repeat_penalty=.05,
                             max_stretch=3, max_candidates=128):
    physical = [np.asarray(p,dtype=int) for p in (physical_a,physical_b)]
    for size,p in zip(np.shape(rewards),physical):
        if p.shape!=(size,) or len(set(p))!=size or (p<0).any():
            raise ValueError('Physical indices must be unique, nonnegative valid windows')
        if size>1 and not ((np.diff(p)>0).all() or (np.diff(p)<0).all()):
            raise ValueError('Physical indices must preserve reading order')
    if min_windows<1 or max_gap<0 or max_candidates<1:
        raise ValueError('Invalid support/candidate settings')
    raw, limited = stretch_candidates(rewards,gap_open,gap_extend,repeat_penalty,max_stretch,max_candidates)
    proposals, seen = [], set()
    for candidate in raw:
        steps = candidate['path']
        anchors = [k for k,s in enumerate(steps) if s['a'] is not None and s['b'] is not None
                   and rewards[s['a'],s['b']] > 0]
        groups = []
        for k in anchors:
            prev = steps[groups[-1][-1]] if groups else None
            adjacent = prev is not None and all(abs(int(p[steps[k][key]])-int(p[prev[key]]))<=max_gap+1
                                                for p,key in zip(physical,('a','b')))
            if adjacent: groups[-1].append(k)
            else: groups.append([k])
        for group in groups:
            path = [dict(s) for s in steps[group[0]:group[-1]+1]]
            path[0]['kind'] = 'match'
            path[0]['delta'] = float(rewards[path[0]['a'],path[0]['b']])
            pairs = [(steps[k]['a'],steps[k]['b']) for k in group]
            signature = tuple(pairs)
            if signature in seen: continue
            seen.add(signature)
            supported = [sorted({int(p[pair[side]]) for pair in pairs}) for side,p in enumerate(physical)]
            # Fill only weak diagonal evidence between close anchors, never true skips.
            filled = [sorted({int(p[s[key]]) for s in path if s['kind']=='match'
                               and s['a'] is not None and s['b'] is not None and rewards[s['a'],s['b']]<=0}
                              - set(supported[side])) for side,(p,key) in enumerate(zip(physical,('a','b')))]
            score = sum(s['delta'] for s in path)
            proposals.append(dict(pairs=pairs, path=path, score=score,
                logical_ranges=[(min(q[s] for q in pairs),max(q[s] for q in pairs)) for s in (0,1)],
                supported_physical=supported, filled_physical=filled, support=list(map(len,supported)),
                matching_evidence=sum(float(rewards[s['a'],s['b']])*(.5 if s['kind'].startswith('repeat') else 1)
                    for s in path if not s['kind'].startswith('gap')),
                repeat_penalties=repeat_penalty*sum(s['kind'].startswith('repeat') for s in path),
                gap_penalties=-sum(s['delta'] for s in path if s['kind'].startswith('gap')),
                weak_spans=[s for s in path if s['kind']=='match' and s['delta']<=0],
                reason='insufficient_distinct_support' if min(map(len,supported))<min_windows else
                       'nonpositive_objective' if score<=0 else 'eligible'))
    accepted, diagnostics = [], []
    for candidate in sorted(proposals,key=lambda c:c['score'],reverse=True):
        if candidate['reason']=='eligible':
            (a,b),(c,d)=candidate['logical_ranges']
            compatible = all((b<r['logical_ranges'][0][0] and d<r['logical_ranges'][1][0]) or
                             (a>r['logical_ranges'][0][1] and c>r['logical_ranges'][1][1]) for r in accepted)
            candidate['reason']='accepted' if compatible else 'crossing_or_reused_windows'
            if compatible: accepted.append(candidate)
        diagnostics.append(candidate)
    rejected = [c for c in diagnostics if c['reason']!='accepted']
    if not diagnostics: rejected=[dict(reason='no_positive_local_alignment')]
    return dict(regions=sorted(accepted,key=lambda c:c['logical_ranges'][0][0]), rejected=rejected,
                candidates=diagnostics, rewards=np.asarray(rewards), candidate_limit_reached=limited,
                maximum_local_score=max((c['score'] for c in raw),default=0.))


def match_features(features_a, features_b, physical_a, physical_b, *, text_encoder=None,
                   config=None, prior_metadata=None, **settings):
    """SINGLE image-only matcher dispatch, used by CLI, session, population and exports."""
    options = resolved_match_settings(settings)
    cosine = cosine_similarity_matrix(features_a,features_b).detach().cpu().numpy()
    extra = {}
    if options['similarity_mode']=='cosine':
        if options['score_mode'] not in {'raw','background'}: raise ValueError('Unknown cosine score_mode')
        baseline=options['threshold']
        if options['score_mode']=='background' and cosine.size:
            baseline=np.maximum(baseline,np.median(cosine,axis=1)[:,None]+options['contrast_margin'])
            baseline=np.maximum(baseline,np.median(cosine,axis=0)[None,:]+options['contrast_margin'])
        rewards=cosine-baseline
    elif options['similarity_mode']=='letter_evidence':
        if text_encoder is None or config is None: raise ValueError('Letter evidence requires checkpoint character vectors/config')
        prior_metadata = prior_metadata or getattr(text_encoder,'letter_evidence_prior',None)
        vocabulary = (prior_metadata['vocabulary'] if prior_metadata else
                      list(dict.fromkeys(config.alphabet_inventory or ARABIC_LETTERS)))
        if not vocabulary or len(set(vocabulary))!=len(vocabulary): raise ValueError('Invalid fixed vocabulary')
        with torch.no_grad():
            alphabet=text_encoder.encode(''.join(vocabulary)).to(features_a.device)
            logp=[alphabet_log_probabilities(v,alphabet,config.competition_temperature,
                    config.ctc_blank_logit if config.alignment_objective=='ctc' else None)
                  for v in (features_a,features_b)]
            # Do NOT renormalize after removing blank: retain uncertainty/no-emission mass.
            logp=[v[:,:len(vocabulary)] for v in logp]
            evidence=letter_evidence_matrix(*logp,prior=prior_metadata['prior'] if prior_metadata else None,
                acceptance_offset=options['acceptance_offset'],prior_floor=options['prior_floor']).cpu().numpy()
        rewards=evidence
        extra=dict(letter_evidence=evidence,log_probabilities=[v.cpu().numpy() for v in logp],
                   vocabulary=vocabulary, prior_metadata=prior_metadata or dict(source='explicit uniform fallback',
                   vocabulary=vocabulary, prior=[1/len(vocabulary)]*len(vocabulary)))
    else: raise ValueError('similarity_mode must be cosine or letter_evidence')
    if options['decoder']=='affine':
        # Legacy implementation accepts a reward via raw mode / zero threshold.
        if options['similarity_mode']=='letter_evidence':
            raise ValueError('Letter evidence requires stretch decoder; affine baseline is cosine-only')
        match=shared_regions(cosine,physical_a,physical_b,**{k:options[k] for k in
            ('threshold','contrast_margin','score_mode','min_windows','max_gap','gap_open','gap_extend','max_candidates')})
        match['maximum_local_score']=smith_waterman_affine(match['rewards'],options['gap_open'],options['gap_extend'])[1]
    elif options['decoder']=='stretch':
        match=decode_stretched_regions(rewards,physical_a,physical_b,**{k:options[k] for k in
            ('min_windows','max_gap','gap_open','gap_extend','repeat_penalty','max_stretch','max_candidates')})
    else: raise ValueError('decoder must be affine or stretch')
    return dict(match, cosine=cosine, settings=options, implementation_version=MATCH_VERSION, **extra)


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
    features,physical,geometry,originals,validity,full_physical=[],[],[],[],[],[]
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
        validity.append(valid.cpu().numpy())
        full_physical.append(bundle['physical_window_indices'][0].cpu().numpy())
        features.append(bundle[representation][0][valid])
        physical.append(bundle['physical_window_indices'][0][valid].cpu().numpy())
        geometry.append(geo)
    match=match_features(*features,*physical,text_encoder=_text,config=config,**settings)
    cosine=match['cosine']
    np.save(output/'cosine.npy',cosine)
    np.save(output/'alignment_scores.npy',match['rewards'])
    if 'letter_evidence' in match:
        np.save(output/'letter_evidence.npy',match['letter_evidence'])
        for side,logp in enumerate(match['log_probabilities']):
            np.save(output/f'letter_log_probs_{side}.npy',logp)
    for side,p in enumerate(physical):
        np.save(output/f'physical_{side}.npy',p)
        np.save(output/f'valid_{side}.npy',validity[side])
        np.save(output/f'full_physical_{side}.npy',full_physical[side])
        np.save(output/f'features_{side}.npy',features[side].detach().cpu().numpy())
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
                geometry=geometry,settings=match['settings'],regions=match['regions'],rejected=match['rejected'],
                candidates=match.get('candidates',[]), candidate_limit_reached=match.get('candidate_limit_reached',False),
                implementation_version=MATCH_VERSION,prior_metadata=match.get('prior_metadata'),
                selection=selection,crop_override=crop_override,
                metrics=[score_mask(mask,gt) for mask,gt in zip(masks,ground_truth)],
                limitations=['Greedy, not globally optimal, multi-region local alignment.',
                    'Uncalibrated thresholds; calibrate on validation positives and negatives, not test.',
                    'Background correction may suppress broad/repeated genuine matches.',
                    'Source widths vary: bounded repeat matches approximate stretching; settings are uncalibrated.',
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
    parser.add_argument('--min-windows',type=int,default=3)
    parser.add_argument('--max-gap',type=int,default=1,help='0=strict consecutive positive anchors')
    parser.add_argument('--gap-open',type=float,default=.2)
    parser.add_argument('--gap-extend',type=float,default=.05)
    for key in ('similarity_mode','decoder','repeat_penalty','max_stretch','acceptance_offset','prior_floor','max_candidates'):
        value=MATCH_DEFAULTS[key]
        parser.add_argument('--'+key.replace('_','-'),type=type(value),default=value)
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
    settings={k:getattr(args,k) for k in MATCH_DEFAULTS}
    output=args.output or str(Path('Results/simple_alignment')/uuid.uuid4().hex[:12])
    report=evaluate_pair(args.checkpoint,paths,output,args.device,args.representation,args.crop_override,
                         (args.gt_a,args.gt_b),selection,**settings)
    print(f'Saved {len(report["regions"])} shared regions to {output}')
    return report


if __name__=='__main__':
    main()
