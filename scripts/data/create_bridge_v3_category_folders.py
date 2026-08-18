#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,shutil
from pathlib import Path

def link_or_copy(src,dst):
    src=Path(src); dst=Path(dst); dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() or dst.is_symlink(): dst.unlink()
    try: os.link(src,dst)
    except OSError: shutil.copy2(src,dst)
def resolve(root,v):
    p=Path(str(v)).expanduser(); return p if p.is_absolute() else root/p

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); a=ap.parse_args(); root=Path(a.data_dir).expanduser().resolve()
    anchors=[json.loads(x) for x in (root/'anchor_index.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    for c in ('real','positive','negative'): (root/c).mkdir(parents=True,exist_ok=True)
    for row in anchors:
        aid=str(row['anchor_id']); rd=root/'real'/aid; ris=resolve(root,row['real']['image']); link_or_copy(ris,rd/f"real{ris.suffix.lower() or '.png'}"); link_or_copy(resolve(root,row['real']['text']),rd/'real.txt')
        pd=root/'positive'/aid; p=row['positive']; link_or_copy(resolve(root,p['image']),pd/'positive.png'); link_or_copy(resolve(root,p['text']),pd/'positive.txt'); link_or_copy(resolve(root,p['mask']),pd/'positive_mask.png')
        nd=root/'negative'/aid
        for n in row.get('negatives',[]):
            i=int(n['index']); link_or_copy(resolve(root,n['image']),nd/f'negative_{i:02d}.png'); link_or_copy(resolve(root,n['text']),nd/f'negative_{i:02d}.txt')
    mp=root/'metadata.json'; m=json.loads(mp.read_text(encoding='utf-8')); m['human_category_folders']=True; mp.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print('CATEGORY_FOLDERS=READY')
if __name__=='__main__': main()
