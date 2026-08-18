#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
from PIL import Image,ImageChops
def _r(root:Path,v:str)->Path:
 p=Path(str(v)).expanduser(); return p if p.is_absolute() else root/p
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); a=p.parse_args(); root=Path(a.data_dir).expanduser().resolve(); m=json.loads((root/'metadata.json').read_text(encoding='utf-8')); pol=m.get('real_line_augmentation') or {}
 if pol.get('enabled') is not True or pol.get('geometric') is not False or pol.get('operations')!=['gaussian_blur','gaussian_noise']: raise RuntimeError('invalid real augmentation policy')
 rows=[json.loads(x) for x in (root/'anchor_index.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]; changed=0
 for row in rows:
  aid=str(row['anchor_id']); real=row['real']; aug=real.get('appearance_augmentation') or {}; ap=_r(root,real.get('image','')); op=_r(root,real.get('original_image',''))
  if not ap.is_file() or not op.is_file(): raise RuntimeError(f'{aid}: missing augmented/original real image')
  if aug.get('geometric_transform') is not False or float(aug.get('gaussian_blur_radius',0))<=0 or float(aug.get('gaussian_noise_sigma',0))<=0: raise RuntimeError(f'{aid}: invalid augmentation metadata')
  with Image.open(ap) as x, Image.open(op) as y:
   if x.size!=y.size: raise RuntimeError(f'{aid}: geometry changed')
   if ImageChops.difference(x.convert('RGB'),y.convert('RGB')).getbbox() is not None: changed+=1
  hd=root/'real'/aid
  if not (hd/'real.txt').is_file() or not (hd/'augmentation.json').is_file() or not any(hd.glob('real_original.*')): raise RuntimeError(f'{aid}: incomplete real folder')
 if changed!=len(rows): raise RuntimeError(f'Only {changed}/{len(rows)} real images changed')
 if int(m.get('real_augmented_count',-1))!=len(rows): raise RuntimeError('real_augmented_count mismatch')
 print(f'anchors_checked={len(rows)}'); print('operations=gaussian_blur+gaussian_noise'); print('REAL_AUGMENTATION_TEST=PASS')
if __name__=='__main__': main()
