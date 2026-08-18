#!/usr/bin/env python3
"""Apply deterministic non-geometric augmentation to copied Bridge V3 real anchors."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter

def _anchor_seed(base_seed:int, anchor_id:str)->int:
    return (int(base_seed)+int.from_bytes(hashlib.sha256(anchor_id.encode('utf-8')).digest()[:8],'big'))%(2**32-1)
def _resolve(root:Path,value:str)->Path:
    p=Path(str(value)).expanduser(); return p if p.is_absolute() else root/p
def _write_json(path:Path,value:dict)->None:
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--seed',type=int,default=4242); p.add_argument('--blur-min-radius',type=float,default=0.15); p.add_argument('--blur-max-radius',type=float,default=1.0); p.add_argument('--noise-min-sigma',type=float,default=2.0); p.add_argument('--noise-max-sigma',type=float,default=8.0); a=p.parse_args()
    if not 0<=a.blur_min_radius<=a.blur_max_radius: p.error('invalid blur radius range')
    if not 0<=a.noise_min_sigma<=a.noise_max_sigma: p.error('invalid Gaussian-noise sigma range')
    root=Path(a.data_dir).expanduser().resolve(); aip=root/'anchor_index.jsonl'; mp=root/'dataset_manifest.jsonl'; md=root/'metadata.json'
    anchors=[json.loads(x) for x in aip.read_text(encoding='utf-8').splitlines() if x.strip()]; by={}
    for r in anchors:
        aid=str(r['anchor_id']); real=r['real']; rp=_resolve(root,real['image']); suffix=rp.suffix.lower() or '.png'; op=rp.with_name(f'real_original{suffix}')
        if not op.exists(): shutil.copy2(rp,op)
        seed=_anchor_seed(a.seed,aid); rng=np.random.default_rng(seed); blur=float(rng.uniform(a.blur_min_radius,a.blur_max_radius)); sigma=float(rng.uniform(a.noise_min_sigma,a.noise_max_sigma)); ns=int(rng.integers(0,2**32-1))
        with Image.open(op) as src: image=src.convert('RGB' if src.mode not in {'L','RGB'} else src.mode)
        image=image.filter(ImageFilter.GaussianBlur(radius=blur)); nrng=np.random.default_rng(ns); arr=np.asarray(image,dtype=np.float32); arr+=nrng.normal(0.0,sigma,arr.shape).astype(np.float32); Image.fromarray(np.clip(arr,0,255).astype(np.uint8),mode=image.mode).save(rp)
        aug={'geometric_transform':False,'gaussian_blur_radius':round(blur,5),'gaussian_noise_sigma':round(sigma,5),'gaussian_noise_seed':ns,'base_seed':int(a.seed),'original_image':op.relative_to(root).as_posix(),'augmented_image':rp.relative_to(root).as_posix()}; by[aid]=aug; real['original_image']=aug['original_image']; real['appearance_augmentation']=aug
        aj=root/'anchors'/aid/'anchor.json'
        if aj.is_file(): q=json.loads(aj.read_text(encoding='utf-8')); q['real']['original_image']=aug['original_image']; q['real']['appearance_augmentation']=aug; _write_json(aj,q)
    with aip.open('w',encoding='utf-8') as h:
        for r in anchors: h.write(json.dumps(r,ensure_ascii=False)+'\n')
    rows=[json.loads(x) for x in mp.read_text(encoding='utf-8').splitlines() if x.strip()]
    for r in rows: aid=str((r.get('bridge') or {}).get('anchor_id') or ''); r.setdefault('bridge',{})['real_appearance_augmentation']=by[aid]
    with mp.open('w',encoding='utf-8') as h:
        for r in rows: h.write(json.dumps(r,ensure_ascii=False)+'\n')
    ri=root/'real_lines_index.jsonl'
    if ri.is_file():
        rows=[json.loads(x) for x in ri.read_text(encoding='utf-8').splitlines() if x.strip()]
        with ri.open('w',encoding='utf-8') as h:
            for r in rows: aug=by[str(r['anchor_id'])]; r['original_image']=aug['original_image']; r['appearance_augmentation']=aug; h.write(json.dumps(r,ensure_ascii=False)+'\n')
    meta=json.loads(md.read_text(encoding='utf-8')); meta['real_line_augmentation']={'enabled':True,'geometric':False,'operations':['gaussian_blur','gaussian_noise'],'blur_radius_range':[a.blur_min_radius,a.blur_max_radius],'noise_sigma_range':[a.noise_min_sigma,a.noise_max_sigma],'seed':a.seed,'originals_preserved':True}; meta['real_augmented_count']=len(anchors); _write_json(md,meta)
    print(f'real_lines_augmented={len(anchors)}'); print('REAL_AUGMENTATION=READY')
if __name__=='__main__': main()
