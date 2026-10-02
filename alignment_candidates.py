"""Image-only local candidate validation and selection (no dataset/GT access).

The DP proposes competing routes without suppressing image evidence. Physical
segmentation, trimming, bounded continuation, adaptive validation, dominance and
optional merging operate on those routes before masks are constructed.
"""
from __future__ import annotations

import math
import numpy as np

PATH_DEFAULTS = dict(
    use_path_anchor_check=True, anchor_top_k=3,
    min_mutual_anchors_long=2, min_mutual_anchor_fraction_long=.15,
    min_mutual_anchors_short=2, min_mutual_anchor_fraction_short=.40,
    normal_min_distinct_a=5, normal_min_distinct_b=5, normal_min_matched_pairs=5,
    allow_short_regions=True, short_min_distinct_a=3, short_min_distinct_b=3,
    short_min_matched_pairs=4, short_min_mean_cosine=.72, short_min_median_cosine=.72,
    short_min_mean_reward=.15, short_min_path_density=.80, short_max_internal_gap=0,
    short_min_mutual_anchors=2, short_min_mutual_anchor_fraction=.40,
    min_median_bidirectional_margin_long=None, min_median_bidirectional_margin_short=None,
    enable_path_extension=True, extension_min_reward=0., extension_max_weak_run=2,
    extension_max_internal_gap=1, enable_region_merging=True, merge_max_gap_a=1,
    merge_max_gap_b=1, dominance_min_quality_ratio=.90,
    candidate_similarity_mode=None, verify_with_letter_evidence=False,
    min_mean_letter_evidence=None, min_letter_positive_fraction=None)


def validate_path_options(options):
    for key, value in options.items():
        if isinstance(value,(int,float,np.number)) and not math.isfinite(value):
            raise ValueError(f'{key} must be finite')
    positive=('min_windows','max_candidates','max_consecutive_repeats','min_distinct_windows_a',
              'min_distinct_windows_b','min_matched_pairs','anchor_top_k',
              'normal_min_distinct_a','normal_min_distinct_b','normal_min_matched_pairs',
              'short_min_distinct_a','short_min_distinct_b','short_min_matched_pairs')
    nonnegative=('max_gap','short_max_internal_gap','extension_max_internal_gap',
                 'extension_max_weak_run','merge_max_gap_a','merge_max_gap_b',
                 'min_mutual_anchors_long','min_mutual_anchors_short','short_min_mutual_anchors')
    for key in positive+nonnegative:
        v=options[key]
        if v is not None and (not isinstance(v,(int,np.integer)) or v < (1 if key in positive else 0)):
            raise ValueError(f'{key} must be a valid integer')
    for key in ('short_min_path_density','min_path_density','max_repeat_fraction',
                'min_mutual_anchor_fraction_long','min_mutual_anchor_fraction_short',
                'short_min_mutual_anchor_fraction','min_letter_positive_fraction','dominance_min_quality_ratio'):
        v=options[key]
        if v is not None and not 0<=v<=1: raise ValueError(f'{key} must be in [0,1]')
    if not -1<=options['threshold']<=1 or options['contrast_margin']<0:
        raise ValueError('Invalid cosine threshold/contrast margin')
    if not options['gap_open']>=options['gap_extend']>=0 or options['repeat_penalty']<0:
        raise ValueError('Invalid gap/repeat penalties')
    if options['score_mode'] not in ('raw','background'): raise ValueError('Invalid score mode')
    if options['min_secondary_score_ratio'] is not None and options['min_secondary_score_ratio']<0:
        raise ValueError('Secondary score ratio must be nonnegative')


def uniqueness_matrices(cosine):
    """Competing maxima exclude the matched cell, even when it is not top-1.

    No competitor (a singleton row/column) is unavailable, not infinite evidence.
    Stable top-K ties can still be anchors; margins expose their ambiguity.
    """
    c=np.asarray(cosine,float)
    n,m=c.shape
    data={}
    for name,axis,size in (('row',1,m),('column',0,n)):
        if not size:
            best=np.full(n if axis==1 else m,np.nan);second=best.copy()
            competitor=np.full(c.shape,np.nan)
        else:
            ordered=np.sort(c,axis=axis)
            best=np.take(ordered,-1,axis=axis)
            second=np.take(ordered,-2,axis=axis) if size>1 else np.full(best.shape,np.nan)
            expanded=best[:,None] if axis==1 else best[None,:]
            expanded_second=second[:,None] if axis==1 else second[None,:]
            # For tied maxima the second value is equal, so ties have margin zero.
            competitor=np.where(c==expanded,expanded_second,expanded)
        data[name+'_best']=best
        data[name+'_second_best']=second
        data[name+'_margin']=c-competitor
    data['bidirectional_margin']=np.minimum(data['row_margin'],data['column_margin'])
    return data


def _stat(values,kind='mean'):
    v=np.asarray(values,float);v=v[np.isfinite(v)]
    if not len(v): return None
    return float(np.median(v) if kind=='median' else v.min() if kind=='min' else v.mean())


def _number(value):
    return float(value) if np.isfinite(value) else None


def physical_gaps(left,right,physical):
    """Spatial discontinuity guard; missing identities cannot join giant masks."""
    return [max(0,abs(int(p[right[s]])-int(p[left[s]]))-1) for s,p in enumerate(physical)]


def valid_window_gaps(left,right,physical):
    """Skipped valid windows; removed invalid/padded identities are not gaps."""
    gaps=[]
    for side,p in enumerate(physical):
        lo,hi=sorted((int(p[left[side]]),int(p[right[side]])))
        gaps.append(int(((p>lo)&(p<hi)).sum()))
    return gaps


def segment_path(path,physical,max_gap):
    groups=[]
    for k,s in enumerate(path):
        if s['i'] is None or s['j'] is None: continue
        q=(s['i'],s['j'])
        prev=path[groups[-1][-1]] if groups else None
        if prev and max(physical_gaps((prev['i'],prev['j']),q,physical))<=max_gap:
            groups[-1].append(k)
        else: groups.append([k])
    return [path[g[0]:g[-1]+1] for g in groups]


def candidate_statistics(path,cosine,rewards,physical,anchors,uniqueness,options,
                         letter_evidence=None,**provenance):
    """Transparent objective; density never counts fills or repeated support.

    Density=min(distinct support/available valid physical span) on A/B. Gap fraction
    = skipped physical windows/(supported physical windows+skipped windows),
    summed across both sides. Score per match includes gap and repeat costs.
    """
    steps=[dict(s) for s in path]
    steps[0]['transition']='diagonal'
    reward_sum=gap_cost=repeat_cost=0.
    previous=None;run=0
    for s in steps:
        kind=s['transition']
        if kind.startswith('gap'):
            cost=options['gap_extend'] if previous==kind else options['gap_open']
            gap_cost+=cost;s['delta']=-cost;run=0
        else:
            reward_sum+=float(s['reward'])
            run=run+1 if previous==kind and 'repeat' in kind else int('repeat' in kind)
            cost=options['repeat_penalty']*run
            repeat_cost+=cost;s['delta']=float(s['reward'])-cost
        previous=kind
    matched=[s for s in steps if s['i'] is not None and s['j'] is not None]
    pairs=[(s['i'],s['j']) for s in matched]
    a,b=np.array(pairs).T
    ids=[sorted({int(p[q[side]]) for q in pairs}) for side,p in enumerate(physical)]
    gaps=[g for x,y in zip(pairs,pairs[1:]) for g in valid_window_gaps(x,y,physical)]
    filled=[]
    for side,p in enumerate(physical):
        valid=set(p.tolist());fills=[]
        for x,y in zip(ids[side],ids[side][1:]):
            if y-x-1<=options['max_gap']:fills.extend(k for k in range(x+1,y) if k in valid)
        filled.append(fills)
    cos=cosine[a,b];r=rewards[a,b]
    strong=[k for k,q in enumerate(pairs) if anchors[q]]
    rh=sum(s['transition']=='horizontal_repeat' for s in matched)
    rv=sum(s['transition']=='vertical_repeat' for s in matched)
    counts=list(map(len,ids));score=reward_sum-gap_cost-repeat_cost
    c=dict(candidate_id=None,accepted=False,reason='unvalidated',status='visual_candidate',
        candidate_type=None,pairs=pairs,physical_pairs=[(int(physical[0][i]),int(physical[1][j])) for i,j in pairs],
        path_steps=steps,transition_types=[s['transition'] for s in matched],
        logical_ranges=[(min(q[s] for q in pairs),max(q[s] for q in pairs)) for s in (0,1)],
        supported_physical=ids,filled_physical=filled,support=counts,support_a=counts[0],support_b=counts[1],
        distinct_windows_a=counts[0],distinct_windows_b=counts[1],matched_pairs=len(pairs),
        score=score,total_score=score,score_per_match=score/len(pairs),matching_reward=reward_sum,
        gap_penalty=gap_cost,repeat_penalty=repeat_cost,mean_cosine=float(cos.mean()),
        median_cosine=float(np.median(cos)),min_cosine=float(cos.min()),minimum_cosine=float(cos.min()),
        mean_reward=float(r.mean()),median_reward=float(np.median(r)),
        path_density=min(len(x)/int(((p>=x[0])&(p<=x[-1])).sum()) for x,p in zip(ids,physical)),true_gaps=sum(gaps),
        max_physical_discontinuity=max((g for x,y in zip(pairs,pairs[1:])
            for g in physical_gaps(x,y,physical)),default=0),
        number_of_true_gaps=sum(gaps),max_internal_gap=max(gaps,default=0),
        maximum_internal_gap=max(gaps,default=0),gap_fraction=sum(gaps)/(sum(counts)+sum(gaps)),
        horizontal_repeats=rh,vertical_repeats=rv,repeat_fraction=(rh+rv)/len(pairs),
        mutual_anchor_count=len(strong),mutual_anchor_fraction=len(strong)/len(pairs),
        strong_anchor_positions=strong,first_strong_anchor_position=strong[0] if strong else None,
        last_strong_anchor_position=strong[-1] if strong else None,
        start_anchor=list(pairs[0]),start_anchor_mutual_top_k=bool(anchors[pairs[0]]),
        start_anchor_eligible=None,trimmed_prefix=False,trimmed_suffix=False,variant='original',
        trimmed_pairs=[],extension_pairs=[],merged_from=[],dominated_by=None,parent_candidate_id=None,
        split_from_path=False,parent_path_score=None,trimmed_to_strong_start=False,
        mean_letter_evidence=None,median_letter_evidence=None,minimum_letter_evidence=None,
        letter_evidence_mean=None,letter_evidence_positive_fraction=None,
        required_support_a=None,required_support_b=None,required_matched_pairs=None)
    for name in ('row','column','bidirectional'):
        values=uniqueness[name+'_margin'][a,b]
        c['mean_'+name+'_margin']=_stat(values)
        c['median_'+name+'_margin']=_stat(values,'median')
    margins=uniqueness['bidirectional_margin'][a,b]
    finite=margins[np.isfinite(margins)]
    c['fraction_positive_bidirectional_margin']=float((finite>0).mean()) if len(finite) else None
    c['ambiguous']=c['median_bidirectional_margin'] is not None and c['median_bidirectional_margin']<=0
    c['pair_confidence']=[dict(row_best=_number(uniqueness['row_best'][i]),
        row_second_best=_number(uniqueness['row_second_best'][i]),
        column_best=_number(uniqueness['column_best'][j]),column_second_best=_number(uniqueness['column_second_best'][j]),
        row_margin=_number(uniqueness['row_margin'][i,j]),column_margin=_number(uniqueness['column_margin'][i,j]),
        bidirectional_margin=_number(uniqueness['bidirectional_margin'][i,j])) for i,j in pairs]
    if letter_evidence is not None:
        evidence=letter_evidence[a,b]
        c.update(mean_letter_evidence=float(evidence.mean()),letter_evidence_mean=float(evidence.mean()),
                 median_letter_evidence=float(np.median(evidence)),minimum_letter_evidence=float(evidence.min()),
                 letter_evidence_positive_fraction=float((evidence>0).mean()))
    c.update(provenance)
    return c


def validate_candidate(c,o):
    """Support classes are disjoint; short never bypasses its stronger filters."""
    normal=[o['normal_min_distinct_a'] or max(o['min_windows'],o['min_distinct_windows_a']),
            o['normal_min_distinct_b'] or max(o['min_windows'],o['min_distinct_windows_b']),
            o['normal_min_matched_pairs'] or max(o['min_windows'],o['min_matched_pairs'])]
    is_normal=c['support_a']>=normal[0] and c['support_b']>=normal[1] and c['matched_pairs']>=normal[2]
    c['candidate_type']='normal' if is_normal else 'short'
    required=normal if is_normal or not o['allow_short_regions'] else [o['short_min_distinct_a'],o['short_min_distinct_b'],o['short_min_matched_pairs']]
    c.update(required_support_a=required[0],required_support_b=required[1],required_matched_pairs=required[2])
    reason=None
    if c['support_a']<required[0] or c['support_b']<required[1] or c['matched_pairs']<required[2]:
        reason='insufficient_distinct_support'
    elif _maximum_repeat_run(c['path_steps'])>o['max_consecutive_repeats']:reason='excessive_consecutive_repeats'
    elif c['score']<=0:reason='nonpositive_objective'
    elif c['maximum_internal_gap']>o['max_gap']:reason='excessive_internal_gap'
    elif not is_normal and o['allow_short_regions']:
        short_checks=(('mean_cosine','short_min_mean_cosine'),('median_cosine','short_min_median_cosine'),
                      ('mean_reward','short_min_mean_reward'),('path_density','short_min_path_density'))
        for metric,setting in short_checks:
            if o[setting] is not None and c[metric]<o[setting]:reason='below_'+setting;break
        if not reason and c['maximum_internal_gap']>o['short_max_internal_gap']:reason='excessive_short_internal_gap'
    if not reason and o['use_path_anchor_check']:
        count=o['min_mutual_anchors_long'] if is_normal else max(o['min_mutual_anchors_short'],o['short_min_mutual_anchors'])
        fraction=o['min_mutual_anchor_fraction_long'] if is_normal else max(o['min_mutual_anchor_fraction_short'],o['short_min_mutual_anchor_fraction'])
        if c['mutual_anchor_count']<count or c['mutual_anchor_fraction']<fraction:reason='insufficient_path_anchors'
    for metric,setting,upper in (('score','min_region_score',False),('mean_reward','min_mean_reward',False),
         ('path_density','min_path_density',False),('repeat_fraction','max_repeat_fraction',True),
         ('median_bidirectional_margin','min_median_bidirectional_margin_long' if is_normal else 'min_median_bidirectional_margin_short',False)):
        if reason:break
        limit=o[setting]
        if limit is not None and (c[metric] is None or (c[metric]>limit if upper else c[metric]<limit)):
            reason='excessive_repeat_fraction' if setting=='max_repeat_fraction' else 'above_'+setting if upper else 'below_'+setting
    if not reason and o['verify_with_letter_evidence']:
        for metric,setting in (('mean_letter_evidence','min_mean_letter_evidence'),
                               ('letter_evidence_positive_fraction','min_letter_positive_fraction')):
            if o[setting] is not None and (c[metric] is None or c[metric]<o[setting]):reason='below_'+setting;break
    c.update(valid=reason is None,reason=reason or 'eligible',status='visual_candidate' if reason is None else 'rejected_visual_candidate')
    return c['valid']


def _bridge(left,right,cosine,rewards):
    """Skip only consumed valid logical windows between matched anchors."""
    i,j=left;u,v=right
    path=[]
    for a in range(i+1,u):path.append(dict(i=a,j=None,transition='gap_a',cosine=None,reward=None))
    for b in range(j+1,v):path.append(dict(i=None,j=b,transition='gap_b',cosine=None,reward=None))
    return path


def _maximum_repeat_run(path):
    previous=None;run=maximum=0
    for step in path:
        kind=step['transition']
        run=run+1 if kind==previous and 'repeat' in kind else int('repeat' in kind)
        maximum=max(maximum,run);previous=kind
    return maximum


def extend_path(path,cosine,rewards,physical,anchors,uniqueness,o):
    """Bounded monotonic continuation from both boundaries, never a new seed.

    Negative evidence cannot become a supported match. Nonanchor continuation is
    permitted for at most extension_max_weak_run cells at a boundary. Physical
    jumps, repeat run lengths and local neighboring slope bound each move. Gaps
    are charged explicitly, and only positive incremental objective is extended.
    """
    result=[dict(s) for s in path];added=[]
    bound=min(o['max_gap'],o['extension_max_internal_gap'])
    n,m=cosine.shape
    for direction in (-1,1):
        weak_run=0
        while True:
            matched=[s for s in result if s['i'] is not None and s['j'] is not None]
            edge=matched[0 if direction<0 else -1];q=(edge['i'],edge['j'])
            neighbor=matched[1 if direction<0 else -2] if len(matched)>1 else edge
            slope=(abs(edge['i']-neighbor['i']),abs(edge['j']-neighbor['j']))
            choices=[]
            for di in range(bound+2):
                for dj in range(bound+2):
                    if not (di or dj):continue
                    x,y=q[0]+direction*di,q[1]+direction*dj
                    if not (0<=x<n and 0<=y<m) or max(physical_gaps(q,(x,y),physical))>bound:continue
                    reward=float(rewards[x,y])
                    if reward<o['extension_min_reward'] or reward<0 or cosine[x,y]<o['threshold']:continue
                    weak=not anchors[x,y] or reward<=0
                    if weak and weak_run>=o['extension_max_weak_run']:continue
                    transition='horizontal_repeat' if di==0 else 'vertical_repeat' if dj==0 else 'diagonal'
                    repeat=0
                    boundary=matched if direction<0 else matched[::-1]
                    for s in boundary:
                        if s['transition']!=transition:break
                        repeat+=1
                    if direction<0 and 'repeat' in transition:
                        repeat=0
                        for l,r in zip([(s['i'],s['j']) for s in matched],[(s['i'],s['j']) for s in matched][1:]):
                            same=(l[0]==r[0]) if transition=='horizontal_repeat' else (l[1]==r[1])
                            if not same:break
                            repeat+=1
                    if 'repeat' in transition and repeat>=o['max_consecutive_repeats']:continue
                    gaps=physical_gaps(q,(x,y),physical)
                    gap_cost=sum(o['gap_open']+(k-1)*o['gap_extend'] for k in gaps if k)
                    cost=o['repeat_penalty']*(repeat+1) if 'repeat' in transition else 0.
                    gain=reward-gap_cost-cost
                    if gain<=0:continue
                    margin=uniqueness['bidirectional_margin'][x,y]
                    # Tie-break toward neighboring slope, then uniqueness. No GT.
                    choices.append((gain,-abs(di-slope[0])-abs(dj-slope[1]),margin if np.isfinite(margin) else -math.inf,x,y,transition,weak))
            if not choices:break
            _,_,_,x,y,kind,weak=max(choices)
            step=dict(i=x,j=y,transition=kind,cosine=float(cosine[x,y]),reward=float(rewards[x,y]))
            if direction>0:result+=_bridge(q,(x,y),cosine,rewards)+[step]
            else:
                first=dict(result[0]);first['transition']=kind
                result=[dict(step,transition='diagonal')]+_bridge((x,y),q,cosine,rewards)+[first]+result[1:]
            added.append((x,y));weak_run=weak_run+1 if weak else 0
    return result,added


def _rank(c):
    # Comparable larger continuations already win through dominance. Remaining
    # competitors must win on normalized evidence before length or raw score.
    # Support contributes sublinearly so a small high-cosine island cannot
    # automatically suppress a coherent word, and an arbitrarily long weak
    # route cannot win solely through accumulated reward. Density discounts
    # unsupported physical coverage. This is a ranking utility, not a calibrated
    # probability or a confidence acceptance threshold.
    utility=c['score_per_match']*math.sqrt(min(c['support']))*c['path_density']
    return (utility,c['score_per_match'],c['mean_reward'],c['path_density'],
            c['mutual_anchor_fraction'],
            c['median_bidirectional_margin'] if c['median_bidirectional_margin'] is not None else -math.inf,
            min(c['support']),sum(c['support']),c['score'])


def select_candidates(candidates,o):
    """Validation/dominance on the complete pool precedes overlap selection."""
    valid=[c for c in candidates if c['valid']]
    # Strict supersets precede their subsets, including equal-support routes
    # with extra repeat matches. A dominated route cannot dominate another:
    # otherwise the quality tolerance compounds across a chain of subpaths.
    dominance_order=lambda c:(min(c['support']),sum(c['support']),c['matched_pairs'],_rank(c))
    for c in sorted(valid,key=dominance_order,reverse=True):
        if c['dominated_by'] is not None:continue
        pairs=set(c['pairs'])
        for other in valid:
            if other is c or other['dominated_by'] is not None:continue
            if (set(other['pairs'])<pairs and c['support_a']>=other['support_a'] and c['support_b']>=other['support_b']
                    and c['score_per_match']>=other['score_per_match']*o['dominance_min_quality_ratio']
                    and c['path_density']>=other['path_density']*o['dominance_min_quality_ratio']):
                other.update(dominated_by=c['candidate_id'],reason='dominated_by_candidate_'+str(c['candidate_id']),status='rejected_visual_candidate')
    selected=[]
    for c in sorted(valid,key=_rank,reverse=True):
        if c['dominated_by'] is not None:continue
        (a,b),(d,e)=c['logical_ranges']
        compatible=all((b<r['logical_ranges'][0][0] and e<r['logical_ranges'][1][0]) or
                       (a>r['logical_ranges'][0][1] and d>r['logical_ranges'][1][1]) for r in selected)
        c.update(accepted=compatible,reason='accepted' if compatible else 'crossing_or_reused_windows',
                 status='accepted_alignment_candidate' if compatible else 'rejected_visual_candidate')
        if compatible:selected.append(c)
    return sorted(selected,key=lambda c:c['logical_ranges'][0][0])


def merge_regions(regions,candidates,cosine,rewards,physical,anchors,uniqueness,o,letter_evidence=None):
    """Merge only ordered disjoint physical routes with supported small bridges."""
    merged=[]
    for right in regions:
        if not merged:merged.append(right);continue
        left=merged[-1];a,b=left['pairs'][-1],right['pairs'][0]
        gaps=physical_gaps(a,b,physical)
        if not (b[0]>a[0] and b[1]>a[1] and gaps[0]<=min(o['merge_max_gap_a'],o['max_gap'])
                and gaps[1]<=min(o['merge_max_gap_b'],o['max_gap'])):
            merged.append(right);continue
        bridge=_bridge(a,b,cosine,rewards)
        # Evidence for a missing bridge must actually be present; touching masks
        # or positive endpoint scores do not certify a blank bridge.
        interior=[(i,j) for i in range(a[0]+1,b[0]) for j in range(a[1]+1,b[1])
                  if rewards[i,j]>o['extension_min_reward']]
        if any(gaps) and not interior:
            merged.append(right);continue
        if len(interior)==1:
            i,j=interior[0]
            bridge=[dict(i=i,j=j,transition='diagonal',cosine=float(cosine[i,j]),reward=float(rewards[i,j]))]
        path=left['path_steps']+bridge+right['path_steps']
        candidate=candidate_statistics(path,cosine,rewards,physical,anchors,uniqueness,o,letter_evidence,
                                      merged_from=(left['merged_from'] or [left['candidate_id']])+(right['merged_from'] or [right['candidate_id']]))
        candidate['candidate_id']=f'C{len(candidates)}'
        candidates.append(candidate)
        if validate_candidate(candidate,o):
            for original in (left,right):original.update(accepted=False,reason='merged_into_'+candidate['candidate_id'],status='rejected_visual_candidate')
            candidate.update(accepted=True,reason='accepted',status='accepted_alignment_candidate')
            merged[-1]=candidate
        else:merged.append(right)
    return merged


def extend_selected_regions(regions,candidates,cosine,rewards,physical,anchors,uniqueness,o,letter_evidence=None):
    """Continue selected reliable cores using the bounded continuation rules.

    A moderately weaker tail need not beat its core's mean reward. It must pay
    its own transition costs, preserve whole-path validity, and leave every
    other selected region disjoint and ordered. No evidence is suppressed.
    """
    selected=list(regions)
    for k,core in enumerate(selected):
        extended,added=extend_path(core['path_steps'],cosine,rewards,physical,anchors,uniqueness,o)
        if not added:continue
        candidate=candidate_statistics(extended,cosine,rewards,physical,anchors,uniqueness,o,letter_evidence,
            candidate_id=f'C{len(candidates)}',extension_pairs=added,parent_candidate_id=core['candidate_id'],
            variant=core['variant'],trimmed_prefix=core['trimmed_prefix'],trimmed_suffix=core['trimmed_suffix'],
            trimmed_pairs=[q for q in core['trimmed_pairs'] if q not in added])
        if not validate_candidate(candidate,o):continue
        (a,b),(d,e)=candidate['logical_ranges']
        if not all((b<r['logical_ranges'][0][0] and e<r['logical_ranges'][1][0]) or
                   (a>r['logical_ranges'][0][1] and d>r['logical_ranges'][1][1])
                   for j,r in enumerate(selected) if j!=k):continue
        # Preserve a single diagnostic record per pair sequence.
        existing=next((c for c in candidates if c['pairs']==candidate['pairs']),None)
        if existing is not None:
            candidate['candidate_id']=existing['candidate_id'];existing.update(candidate);candidate=existing
        else:candidates.append(candidate)
        core.update(accepted=False,reason='extended_into_'+candidate['candidate_id'],status='rejected_visual_candidate')
        candidate.update(accepted=True,reason='accepted',status='accepted_alignment_candidate',dominated_by=None)
        selected[k]=candidate
    return selected


def decode_candidates(raw,cosine,rewards,physical,anchors,o,letter_evidence=None):
    uniqueness=uniqueness_matrices(cosine)
    candidates=[];seen=set()
    for route in raw:
        segments=segment_path(route['path_steps'],physical,o['max_gap'])
        for path in segments:
            matched=[k for k,s in enumerate(path) if s['i'] is not None and s['j'] is not None]
            strong=[k for k in matched if anchors[path[k]['i'],path[k]['j']]]
            variants=[(0,len(path),'original')]
            if strong:
                variants += [(strong[0],len(path),'trimmed_prefix'),(0,strong[-1]+1,'trimmed_suffix'),
                             (strong[0],strong[-1]+1,'trimmed_both')]
            original_pairs=[(path[k]['i'],path[k]['j']) for k in matched]
            parent_id=None
            for begin,end,variant in variants:
                if end<=begin:continue
                trimmed=[dict(s) for s in path[begin:end]]
                pairset={(s['i'],s['j']) for s in trimmed if s['i'] is not None and s['j'] is not None}
                trimmed_pairs=[q for q in original_pairs if q not in pairset]
                versions=[(trimmed,[],variant)]
                if o['enable_path_extension'] and strong:
                    extended,added=extend_path(trimmed,cosine,rewards,physical,anchors,uniqueness,o)
                    if added:versions.append((extended,added,variant))
                for version,added,kind in versions:
                    signature=tuple((s['i'],s['j']) for s in version if s['i'] is not None and s['j'] is not None)
                    if signature in seen:continue
                    seen.add(signature)
                    c=candidate_statistics(version,cosine,rewards,physical,anchors,uniqueness,o,letter_evidence,
                        candidate_id=f'C{len(candidates)}',variant=kind,
                        trimmed_prefix=begin>0,trimmed_suffix=end<len(path),trimmed_pairs=trimmed_pairs,
                        extension_pairs=added,parent_candidate_id=parent_id,
                        split_from_path=len(segments)>1,parent_path_score=route['score'])
                    if parent_id is None:parent_id=c['candidate_id']
                    validate_candidate(c,o);candidates.append(c)
    regions=select_candidates(candidates,o)
    if o['enable_path_extension']:
        regions=extend_selected_regions(regions,candidates,cosine,rewards,physical,anchors,uniqueness,o,letter_evidence)
    if o['enable_region_merging']:regions=merge_regions(regions,candidates,cosine,rewards,physical,anchors,uniqueness,o,letter_evidence)
    if regions and o['min_secondary_score_ratio'] is not None:
        best=max(regions,key=lambda c:c['score'])
        for c in list(regions):
            if c is not best and c['score']<best['score']*o['min_secondary_score_ratio']:
                c.update(accepted=False,reason='below_secondary_score_ratio',status='rejected_visual_candidate');regions.remove(c)
    rejected=[c for c in candidates if not c['accepted']]
    if not candidates:rejected=[dict(candidate_id=None,accepted=False,reason='no_positive_local_alignment',status='rejected_visual_candidate')]
    return dict(regions=regions,candidates=candidates,rejected=rejected,rewards=rewards,
                strong_start_anchors=anchors,mutual_anchors=anchors)
