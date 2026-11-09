"""Image-only word correspondence with a fixed, training-only vocabulary.

The complete line competes for word spans before two lines are compared. This
prevents a cheap local letter match from automatically becoming a word match.
Lattice posteriors are relative decoder evidence, not calibrated correctness.
No pair transcripts, alignment masks or annotated boxes enter these functions.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, grey_closing
from scipy.special import logsumexp

NORMALIZATION = str.maketrans({'آ':'ا','أ':'ا','إ':'ا','ؤ':'و','ئ':'ي','ى':'ي','ة':'ه','ء':''})


def normalize_word(value):
    return value.translate(NORMALIZATION)


@lru_cache(maxsize=8)
def _vocabulary(path, digest):
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=digest:
        raise ValueError('Word vocabulary changed while being loaded')
    saved=json.loads(raw)
    if saved.get('source_split')!='train' or not saved.get('checkpoint_sha256'):
        raise ValueError('Word vocabulary requires explicit training provenance')
    words=saved['words']
    if not words or any(not isinstance(n,int) or n<1 or len(w)<2 or ' ' in w for w,n in words.items()):
        raise ValueError('Invalid word vocabulary')
    return saved


def load_vocabulary(path):
    path=Path(path)
    if not path.is_absolute():path=Path(__file__).resolve().parent/path
    raw=path.read_bytes()
    return _vocabulary(str(path),hashlib.sha256(raw).hexdigest())


def image_word_boundaries(path, geometry, physical, window_width, stride):
    """Soft boundary evidence from local ink valleys, in logical/RTL order.

    Local background removal avoids interpreting scan shading as a word. A
    valley is evidence, not a guaranteed Arabic word break: disconnected letter
    forms also create valleys. The full-line word lattice resolves competition.
    """
    gray=np.asarray(Image.open(path).convert('L'))
    width,height=geometry['source_size']
    if gray.shape!=(height,width):raise ValueError('Word boundary image geometry changed')
    size=max(3,height//8)
    background=grey_closing(gray,size=(size,size)).astype(float)
    contrast=background-gray
    ink=contrast>max(12.,.2*np.percentile(contrast,95))
    edge=max(1,round(width*.08))
    interior=ink[:,edge:-edge] if 2*edge<width else ink
    mass=interior.sum(1).astype(float)
    peak=int(gaussian_filter1d(mass,max(1,height/12)).argmax())
    lo=max(0,peak-round(height*.2));hi=min(height,peak+round(height*.2)+1)
    projection=gaussian_filter1d(ink[lo:hi].sum(0).astype(float),1)
    x0,_,x1,_=geometry['crop']
    centers=x0+(np.asarray(physical)*stride+window_width/2)/geometry['scale_x']
    rtl=len(centers)<2 or centers[0]>centers[-1]
    pixels=np.r_[x1 if rtl else x0,(centers[1:]+centers[:-1])/2,x0 if rtl else x1]
    radius=stride/geometry['scale_x']
    scores=[]
    for x in pixels:
        left=max(0,int(x-radius*.4));right=min(width,int(x+radius*.4)+1)
        local=projection[max(0,int(x-3*radius)):min(width,int(x+3*radius)+1)]
        reference=float(np.percentile(local,80)) if len(local) else 0.
        valley=float(projection[left:right].min()) if right>left else 0.
        scores.append(float(np.clip(1-valley/max(1.,reference),0,1)))
    scores[0]=scores[-1]=1.
    return np.asarray(scores),pixels


@lru_cache(maxsize=128)
def _word_spots(logp_bytes, shape, vocabulary, word_counts):
    logp=np.frombuffer(logp_bytes,dtype=np.float32).reshape(shape)
    alphabet=sorted({normalize_word(c) for c in vocabulary}-{''})
    posterior=np.stack([logsumexp(logp[:,[i for i,c in enumerate(vocabulary) if normalize_word(c)==a]],axis=1)
                        for a in alphabet],axis=1)
    words=[w for w,_ in word_counts];length=np.array(list(map(len,words)))
    max_length=int(length.max());ids=np.zeros((len(words),max_length),int)
    for k,w in enumerate(words):ids[k,:len(w)]=[alphabet.index(c) for c in w]
    valid=np.arange(max_length)[None,:]<length[:,None]
    costs=np.where(valid[None,:,:],-posterior[:,ids],np.inf)
    spots=[];indices=np.arange(len(words));n=len(logp)
    for start in range(n-1):
        previous=np.full(ids.shape,np.inf);counts=np.zeros_like(previous)
        for duration in range(1,min(30,n-start)+1):
            current=np.full_like(previous,np.inf);visits=np.zeros_like(previous)
            for j in range(max_length):
                diagonal=previous[:,j-1] if j else np.full(len(words),0. if duration==1 else np.inf)
                vertical=previous[:,j]+.08
                horizontal=current[:,j-1]+.30 if j else np.full(len(words),np.inf)
                choices=np.stack([diagonal,vertical,horizontal]);which=choices.argmin(0)
                current[:,j]=choices.min(0)+costs[start+duration-1,:,j]
                before=np.stack([counts[:,j-1] if j else np.zeros(len(words)),counts[:,j],
                                 visits[:,j-1] if j else np.zeros(len(words))])
                visits[:,j]=before[which,indices]+1
            mean=current[indices,length-1]/np.maximum(1,visits[indices,length-1])
            mean=np.where((duration>=np.maximum(2,np.ceil(length*.6)))&(duration<=length*4),mean,np.inf)
            # An explicit bounded lexical beam, independent of acceptance settings.
            for k in np.argsort(mean,kind='stable')[:8]:
                if mean[k]<3.6:spots.append((words[k],start,start+duration-1,float(mean[k])))
            previous,counts=current,visits
    return tuple(spots)


def word_lattice(logp, vocabulary, word_counts, boundaries, boundary_weight=2., temperature=1.):
    """Forward/backward over whole-word spans and unrecognized-window steps."""
    logp=np.asarray(logp,dtype=np.float32)
    spots=_word_spots(logp.tobytes(),logp.shape,tuple(vocabulary),tuple(sorted(word_counts.items())))
    n=len(logp);endpoints={};starts={};costs=[];total=sum(word_counts.values())
    fallback=-logp.max(1)+2.
    for k,(word,start,end,nll) in enumerate(spots):
        cost=nll*(end-start+1)+2.+.2*np.log(total/word_counts[word])
        cost+=boundary_weight*(2-boundaries[start]-boundaries[end+1])
        costs.append(cost/temperature)
        endpoints.setdefault(end+1,[]).append(k);starts.setdefault(start,[]).append(k)
    forward=np.full(n+1,-np.inf);forward[0]=0.
    backward=np.full(n+1,-np.inf);backward[n]=0.
    for t in range(1,n+1):
        forward[t]=logsumexp([forward[t-1]-fallback[t-1]/temperature]+
            [forward[spots[k][1]]-costs[k] for k in endpoints.get(t,[])])
    for t in range(n-1,-1,-1):
        backward[t]=logsumexp([backward[t+1]-fallback[t]/temperature]+
            [backward[spots[k][2]+1]-costs[k] for k in starts.get(t,[])])
    return [dict(word=w,start=s,end=e,nll=c,posterior=float(np.exp(
        np.clip(forward[s]-costs[k]+backward[e+1]-forward[n],-745,0))))
        for k,(w,s,e,c) in enumerate(spots)]


def full_span_path(cosine, a, b):
    """A global visual check: every window of BOTH proposed words is consumed."""
    aa,ab=a;ba,bb=b;n=ab-aa+1;m=bb-ba+1
    costs=np.full((n+1,m+1),np.inf);costs[0,0]=0.;back=np.zeros((n+1,m+1),np.int8)
    for i in range(1,n+1):
        for j in range(1,m+1):
            alternatives=(costs[i-1,j-1],costs[i-1,j]+.06,costs[i,j-1]+.06)
            choice=int(np.argmin(alternatives));back[i,j]=choice
            costs[i,j]=alternatives[choice]+1-cosine[aa+i-1,ba+j-1]
    pairs=[];i,j=n,m
    while i and j:
        pairs.append((aa+i-1,ba+j-1));choice=back[i,j]
        if choice==0:i-=1;j-=1
        elif choice==1:i-=1
        else:j-=1
    if i or j:raise RuntimeError('Incomplete whole-word traceback')
    return pairs[::-1]


def validate_word_candidate(candidate, options):
    if candidate['candidate_type']=='visual_passage':
        candidate.update(accepted=False,reason='eligible',status='visual_candidate',valid=True,dominated_by=None)
        return True
    threshold=(options['word_min_posterior_short'] if len(candidate['word'])<=2
               else options['word_min_posterior'])
    reason=('word_lexical_cost' if max(candidate['word_nll'])>options['word_max_nll'] else
            'word_context_evidence' if candidate['word_pair_posterior']<threshold else
            'word_visual_sequence' if candidate['mean_cosine']<options['word_min_cosine'] else None)
    candidate.update(valid=reason is None,accepted=False,dominated_by=None,
        reason=reason or 'eligible',status='rejected_visual_candidate' if reason else 'visual_candidate')
    return reason is None


def select_word_candidates(candidates, options):
    from alignment_candidates import compatible_regions
    selected=[]
    for c in candidates:
        if c['candidate_type']=='visual_passage':
            validate_word_candidate(c,options)
            c.update(accepted=True,reason='accepted',status='accepted_alignment_candidate');selected.append(c)
    words=[c for c in candidates if c['candidate_type']=='word']
    for c in words:
        c['word_selection_score']=float(np.log(max(1e-300,c['word_pair_posterior']))+options['word_visual_weight']*c['mean_cosine'])
    for c in sorted(words,key=lambda c:(c['word_selection_score'],c['word_pair_posterior']),reverse=True):
        if not validate_word_candidate(c,options):continue
        compatible=all(compatible_regions(c,r,options['allow_reordered_regions']) for r in selected)
        c.update(accepted=compatible,reason='accepted' if compatible else 'word_reuses_span',
                 status='accepted_alignment_candidate' if compatible else 'rejected_visual_candidate')
        if compatible:selected.append(c)
    if (options['require_corroborated_short_regions'] and selected
            and not any(c['candidate_type']=='visual_passage' for c in selected)):
        identities={c['word'] for c in selected}
        support=[len({p for c in selected for p in c['supported_physical'][side]}) for side in (0,1)]
        confirmed=(len(identities)>=options['word_min_distinct_types']
            and sum(map(len,identities))>=options['word_min_total_characters']
            and support[0]>=options['normal_min_distinct_a']
            and support[1]>=options['normal_min_distinct_b'])
        if not confirmed:
            for c in selected:c.update(accepted=False,reason='uncorroborated_word_regions',status='rejected_visual_candidate')
            selected=[]
    return sorted(selected,key=lambda c:c['logical_ranges'][0][0])


def long_visual_passages(cosine,rewards,physical,options,return_metadata=False):
    """Preserve coherent multi-word visual paths, including unknown vocabulary."""
    if not options['word_keep_long_regions']:
        return ([],dict(candidate_limit_reached=False,raw_candidate_count=0)) if return_metadata else []
    from alignment_candidates import PATH_DEFAULTS
    from evaluate import local_repeat_regions
    keys=('threshold','contrast_margin','score_mode','min_windows','max_gap','gap_open','gap_extend',
          'max_candidates','repeat_penalty','max_consecutive_repeats','min_distinct_windows_a',
          'min_distinct_windows_b','min_matched_pairs','min_region_score','min_mean_reward',
          'min_path_density','max_repeat_fraction','min_secondary_score_ratio')
    settings={key:options[key] for key in (*keys,*PATH_DEFAULTS)}
    settings['allow_short_regions']=False
    match=local_repeat_regions(cosine,*physical,rewards=rewards,**settings)
    regions=[dict(c,candidate_type='visual_passage',candidate_id='P'+str(i),accepted=False,
        dominated_by=None,visual_passage_source='Validated normal path; short visual-only paths disabled')
        for i,c in enumerate(match['regions'])]
    metadata={key:match[key] for key in ('candidate_limit_reached','raw_candidate_count')}
    return (regions,metadata) if return_metadata else regions


def word_regions(log_probabilities, vocabulary, cosine, rewards, physical, boundaries, options):
    from alignment_candidates import candidate_statistics, uniqueness_matrices
    from evaluate import mutual_top_k_anchors
    saved=load_vocabulary(options['word_vocabulary_path']);counts=saved['words']
    unsupported=set(''.join(counts))-{normalize_word(c) for c in vocabulary}
    if unsupported:raise ValueError('Word vocabulary is incompatible with checkpoint alphabet')
    lattices=[word_lattice(lp,vocabulary,counts,b,options['word_boundary_weight'],options['word_lattice_temperature'])
              for lp,b in zip(log_probabilities,boundaries)]
    # Prune only negligible lattice mass, with an explicit floor below every
    # permitted acceptance threshold. Replaying thresholds cannot add new spots.
    targets={}
    for c in lattices[1]:
        if c['posterior']>=.001:targets.setdefault(c['word'],[]).append(c)
    uniqueness=uniqueness_matrices(cosine);anchors=mutual_top_k_anchors(cosine,options['anchor_top_k'])&(rewards>0)
    candidates=[]
    for a in lattices[0]:
        if a['posterior']<.001:continue
        for b in targets.get(a['word'],[]):
            pairs=full_span_path(cosine,(a['start'],a['end']),(b['start'],b['end']))
            steps=[];previous=None
            for i,j in pairs:
                transition=('diagonal' if previous is None or (i>previous[0] and j>previous[1]) else
                            'horizontal_repeat' if i==previous[0] else 'vertical_repeat')
                steps.append(dict(i=i,j=j,transition=transition,cosine=float(cosine[i,j]),reward=float(rewards[i,j])))
                previous=(i,j)
            c=candidate_statistics(steps,cosine,rewards,physical,anchors,uniqueness,options,
                candidate_id=f'W{len(candidates)}',candidate_type='word',word=a['word'],
                word_nll=[a['nll'],b['nll']],word_posteriors=[a['posterior'],b['posterior']],
                word_pair_posterior=float(np.sqrt(a['posterior']*b['posterior'])))
            candidates.append(c)
    passages,visual_metadata=long_visual_passages(cosine,rewards,physical,options,return_metadata=True)
    candidates+=passages
    regions=select_word_candidates(candidates,options)
    return dict(regions=regions,candidates=candidates,rejected=[c for c in candidates if not c['accepted']],
        rewards=rewards,mutual_anchors=anchors,strong_start_anchors=anchors,maximum_local_score=max(
        (c.get('word_pair_posterior',0.) for c in candidates),default=0.),
        candidate_limit_reached=visual_metadata['candidate_limit_reached'],
        visual_candidate_count=visual_metadata['raw_candidate_count'],
        word_vocabulary_source=saved['source_split'],word_vocabulary_size=len(counts),
        word_vocabulary_checkpoint=saved['checkpoint_sha256'],word_lattice_beam=8)
