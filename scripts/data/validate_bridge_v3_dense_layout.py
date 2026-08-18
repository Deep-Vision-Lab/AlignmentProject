#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image

def _resolve(root: Path, value: str) -> Path:
    p=Path(str(value)).expanduser(); return p if p.is_absolute() else root/p

def _span(path: Path) -> float:
    with Image.open(path) as im: arr=np.asarray(im.convert('L'),dtype=np.uint8)
    cols=np.where((arr>128).sum(axis=0)>=2)[0]
    return 0.0 if cols.size==0 else float(cols[-1]-cols[0]+1)/float(arr.shape[1])

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',required=True); ap.add_argument('--min-recorded-fill',type=float,default=.90); ap.add_argument('--min-pixel-span',type=float,default=.84); ap.add_argument('--expected-negatives',type=int,default=8); a=ap.parse_args()
    root=Path(a.data_dir).expanduser().resolve(); m=json.loads((root/'metadata.json').read_text(encoding='utf-8'))
    if float(m.get('min_line_fill_ratio',0))<a.min_recorded_fill: raise RuntimeError('Dense fill policy missing')
    if m.get('human_category_folders') is not True: raise RuntimeError('Root real/positive/negative folders missing')
    rows=[json.loads(x) for x in (root/'dataset_manifest.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]; groups={}; mn=1.0
    for r in rows:
        aid=str((r.get('bridge') or {}).get('anchor_id')); groups.setdefault(aid,[]).append(r); s=_span(_resolve(root,r['B']['line_image_path'])); mn=min(mn,s)
        if s<a.min_pixel_span: raise RuntimeError(f"{r.get('pair_id')}: foreground_span_ratio={s:.3f} < {a.min_pixel_span:.3f}")
    for aid,g in groups.items():
        if len([r for r in g if r.get('label_type')=='medium_match'])!=1 or len([r for r in g if r.get('label_type')=='no_shared_content'])!=a.expected_negatives: raise RuntimeError(f'{aid}: bad relation counts')
        rd=root/'real'/aid; pd=root/'positive'/aid; nd=root/'negative'/aid
        if not rd.is_dir() or not pd.is_dir() or not nd.is_dir(): raise RuntimeError(f'{aid}: missing category folder')
        if not (rd/'real.txt').is_file() or not any(rd.glob('real.*')): raise RuntimeError(f'{aid}: missing real bundle')
        for f in ('positive.png','positive.txt','positive_mask.png','relation.json'):
            if not (pd/f).is_file(): raise RuntimeError(f'{aid}: missing {f}')
        if not (nd/'relations.json').is_file(): raise RuntimeError(f'{aid}: missing relations.json')
        for i in range(a.expected_negatives):
            if not (nd/f'negative_{i:02d}.png').is_file() or not (nd/f'negative_{i:02d}.txt').is_file(): raise RuntimeError(f'{aid}: missing negative {i}')
    print(f'minimum_observed_pixel_span={mn:.4f}'); print('DENSE_LAYOUT_TEST=PASS')
if __name__=='__main__': main()
