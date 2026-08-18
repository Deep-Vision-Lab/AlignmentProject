#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from PIL import Image,ImageFilter
def seed_for(base,aid): return (int(base)+int.from_bytes(hashlib.sha256(aid.encode()).digest()[:8],'big'))%(2**32-1)
def r(root,v): p=Path(str(v)).expanduser(); return p if p.is_absolute() else root/p
def main():
 p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--seed',type=int,default=4242); p.add_argument('--blur-min-radius',type=float,default=.15); p.add_argument('--blur-max-radius',type=float,default=1.0); p.add_argument('--noise-min-sigma',type=float,default=2.0); p.add_argument('--noise-max-sigma',type=float,default=8.0); a=p.parse_args(); root=Path(a.data_dir).expanduser().resolve(); aip=root/'anchor_index.jsonl'; mp=root/'dataset_manifest.jsonl'; md=root/'metadata.json'; anchors=[json.loads(x) for x in aip.read_text(encoding='utf-8').splitlines() if x.strip()]; by={}
 for row in anchors:
  aid=str(row['anchor_id']); real=row['real']; rp=r(root,real['image']); op=rp.with_name(f"real_original{rp.suffix.lower() or '.png'}");
  if not op.exists(): shutil.copy2(rp,op)
  rng=np.random.default_rng(seed_for(a.seed,aid)); blur=float(rng.uniform(a.blur_min_radius,a.blur_max_radius)); sigma=float(rng.uniform(a.noise_min_sigma,a.noise_max_sigma)); ns=int(rng.integers(0,2**32-1))
  with Image.open(op) as src: im=src.convert('RGB' if src.mode not in {'L','RGB'} else src.mode)
  im=im.filter(ImageFilter.GaussianBlur(blur)); nr=np.random.default_rng(ns); arr=np.asarray(im,dtype=np.float32); arr+=nr.normal(0.0,sigma,arr.shape).astype(np.float32); Image.fromarray(np.clip(arr,0,255).astype(np.uint8),mode=im.mode).save(rp)
  aug={'geometric_transform':False,'gaussian_blur_radius':round(blur,5),'gaussian_noise_sigma':round(sigma,5),'gaussian_noise_seed':ns,'base_seed':a.seed,'original_image':op.relative_to(root).as_posix(),'augmented_image':rp.relative_to(root).as_posix()}; by[aid]=aug; real['original_image']=aug['original_image']; real['appearance_augmentation']=aug
  aj=root/'anchors'/aid/'anchor.json'
  if aj.is_file(): q=json.loads(aj.read_text(encoding='utf-8')); q['real']['original_image']=aug['original_image']; q['real']['appearance_augmentation']=aug; aj.write_text(json.dumps(q,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 with aip.open('w',encoding='utf-8') as h:
  for x in anchors: h.write(json.dumps(x,ensure_ascii=False)+'\n')
 rows=[json.loads(x) for x in mp.read_text(encoding='utf-8').splitlines() if x.strip()]
 for x in rows: x.setdefault('bridge',{})['real_appearance_augmentation']=by[str((x.get('bridge') or {}).get('anchor_id') or '')]
 with mp.open('w',encoding='utf-8') as h:
  for x in rows: h.write(json.dumps(x,ensure_ascii=False)+'\n')
 m=json.loads(md.read_text(encoding='utf-8')); m['real_line_augmentation']={'enabled':True,'geometric':False,'operations':['gaussian_blur','gaussian_noise'],'blur_radius_range':[a.blur_min_radius,a.blur_max_radius],'noise_sigma_range':[a.noise_min_sigma,a.noise_max_sigma],'seed':a.seed,'originals_preserved':True}; m['real_augmented_count']=len(anchors); md.write_text(json.dumps(m,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(f'real_lines_augmented={len(anchors)}'); print('REAL_AUGMENTATION=READY')
if __name__=='__main__': main()
