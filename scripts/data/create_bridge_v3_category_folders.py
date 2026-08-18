#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,os,shutil
from pathlib import Path
def _link_or_copy(src:Path,dst:Path)->None:
    if not src.is_file(): raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True,exist_ok=True)
    if dst.exists() or dst.is_symlink(): dst.unlink()
    try: os.link(src,dst)
    except OSError: shutil.copy2(src,dst)
def _resolve(root:Path,value:str)->Path:
    p=Path(str(value)).expanduser(); return p if p.is_absolute() else root/p
def main():
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); a=p.parse_args(); root=Path(a.data_dir).expanduser().resolve(); rows=[json.loads(x) for x in (root/'anchor_index.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    for c in ('real','positive','negative'): (root/c).mkdir(parents=True,exist_ok=True)
    for row in rows:
        aid=str(row['anchor_id']); real=row['real']; rd=root/'real'/aid; ris=_resolve(root,real['image']); _link_or_copy(ris,rd/f"real{ris.suffix.lower() or '.png'}"); _link_or_copy(_resolve(root,real['text']),rd/'real.txt')
        if real.get('original_image'):
            oi=_resolve(root,real['original_image']); _link_or_copy(oi,rd/f"real_original{oi.suffix.lower() or '.png'}")
        if real.get('appearance_augmentation'): (rd/'augmentation.json').write_text(json.dumps(real['appearance_augmentation'],ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        pos=row['positive']; pd=root/'positive'/aid; _link_or_copy(_resolve(root,pos['image']),pd/'positive.png'); _link_or_copy(_resolve(root,pos['text']),pd/'positive.txt'); _link_or_copy(_resolve(root,pos['mask']),pd/'positive_mask.png')
        nd=root/'negative'/aid
        for neg in row.get('negatives',[]):
            i=int(neg['index']); _link_or_copy(_resolve(root,neg['image']),nd/f'negative_{i:02d}.png'); _link_or_copy(_resolve(root,neg['text']),nd/f'negative_{i:02d}.txt')
    mp=root/'metadata.json'; m=json.loads(mp.read_text(encoding='utf-8')); m['human_category_folders']=True; m['human_category_layout']={'real':'real/<anchor_id>/{real.png,real_original.png,real.txt,augmentation.json}','positive':'positive/<anchor_id>/{positive.png,positive.txt,positive_mask.png,relation.json}','negative':'negative/<anchor_id>/{negative_00..07.png,negative_00..07.txt,relations.json}'}; mp.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print('CATEGORY_FOLDERS=READY')
if __name__=='__main__': main()
