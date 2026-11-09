"""Build a fixed word prior using only the checkpoint's saved training lines."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import torch
from dataset import file_hash
from parameters import Config
from train import build_loaders
from text_embedding import clean_letters
from word_alignment import normalize_word


def build(checkpoint,dataset,output):
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    counts=Counter();books=Counter();spaced=Counter();hashes={}
    for record in build_loaders(dataset,Config(**saved['config']),saved['split_ids'])[0].dataset.records:
        side=record['sides'][0];book=record['sample_id'].split(':',1)[0]
        text=Path(side['text']).read_text(encoding='utf-8') if side.get('text') else side['raw_text']
        books[book]+=1;hashes[record['sample_id']]=hashlib.sha256(text.encode()).hexdigest()
        if ' ' not in text.strip():continue
        spaced[book]+=1
        for value in text.split():
            word=normalize_word(''.join(clean_letters(value)))
            if 2<=len(word)<=12:counts[word]+=1
    words={w:n for w,n in counts.items() if n>=2 and (len(w)>2 or n>=10)}
    for word,n in list(words.items()):
        if any(word[:i] in words and word[i:] in words and
               min(words[word[:i]],words[word[i:]])>=2*n for i in range(2,len(word)-1)):
            del words[word]
    result=dict(schema_version=1,source_split='train',checkpoint_sha256=file_hash(checkpoint),
        training_split_sha256=hashlib.sha256(json.dumps(saved['split_ids']['train'],sort_keys=True).encode()).hexdigest(),
        current_training_transcripts_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        book_training_lines=dict(books),training_lines_with_spaces=dict(spaced),words=dict(sorted(words.items())),
        limitation='Fixed training vocabulary; unknown words can be missed. Local transcript hashes describe this extraction, not a proven historical training snapshot. No evaluated line transcripts or masks are consulted at inference.')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'Weights/real_abc_balanced_optional_space_20261108/checkpoint_best.pt')
    p.add_argument('--dataset',type=Path,default=ROOT/'DataSet/ArabicData')
    p.add_argument('--output',type=Path,default=ROOT/'docs/evaluation/word_alignment_20261109/vocabulary.json')
    a=p.parse_args();result=build(a.checkpoint,a.dataset,a.output)
    print('Saved',len(result['words']),'training words to',a.output)
