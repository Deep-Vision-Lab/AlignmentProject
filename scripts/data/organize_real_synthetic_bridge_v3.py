#!/usr/bin/env python3
"""Reorganize RealSyntheticBridge V3 into a human-readable anchor-grouped layout.

The same ``anchor_id`` is the key connecting the copied real line, its one positive
sample, the positive mask, and all negative samples.  The global manifest remains
the training source while per-anchor JSON files make manual inspection simple.
"""
from __future__ import annotations
import argparse, json, shutil
from collections import defaultdict
from pathlib import Path
LAYOUT_VERSION=1

def resolve(root,value):
    p=Path(str(value)).expanduser(); return p if p.is_absolute() else root/p

def cp(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    if not dst.exists(): shutil.copy2(src,dst)

def mv(root,old,new):
    src=resolve(root,old); dst=root/new; dst.parent.mkdir(parents=True,exist_ok=True)
    if src.resolve()!=dst.resolve():
        if dst.exists(): dst.unlink()
        shutil.move(str(src),str(dst))
    return new.as_posix()

def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")

def readme(root):
    root.joinpath("README_DATASET.md").write_text("""# RealSyntheticBridge V3 layout

`<anchor_id>` connects the same real line across every directory.

```text
images/real/<anchor_id>/real.png
images/positive/<anchor_id>/positive.png
images/negative/<anchor_id>/negative_00.png ...
texts/real/<anchor_id>/real.txt
texts/positive/<anchor_id>/positive.txt
texts/negative/<anchor_id>/negative_00.txt ...
masks/positive/<anchor_id>/positive_mask.png
anchors/<anchor_id>/anchor.json
positive/<anchor_id>/relation.json
negative/<anchor_id>/relations.json
anchor_index.jsonl
dataset_manifest.jsonl
metadata.json
```

Open `anchors/<anchor_id>/anchor.json` to see the real sample, its positive sample,
mask/shared islands, and all negatives together.
""",encoding="utf-8")

def organize(root,force=False):
    man=root/"dataset_manifest.jsonl"; meta_path=root/"metadata.json"; meta=json.loads(meta_path.read_text(encoding="utf-8"))
    if int(meta.get("dataset_version",0))!=3: raise RuntimeError("Expected Bridge V3")
    if int(meta.get("layout_version",0))==1 and not force: readme(root); return
    rows=[json.loads(x) for x in man.read_text(encoding="utf-8").splitlines() if x.strip()]; groups=defaultdict(list)
    for r in rows: groups[str(r["bridge"]["anchor_id"])].append(r)
    rewritten=[]; index=[]
    for aid in sorted(groups):
        rs=groups[aid]; pos=[r for r in rs if r.get("label_type")=="medium_match"]; neg=[r for r in rs if r.get("label_type")=="no_shared_content"]
        if len(pos)!=1: raise RuntimeError(f"{aid}: expected one positive")
        p=pos[0]; pair=str(p["pair_id"]); page=str(p.get("A_page_id","")); ris=resolve(root,p["A"]["line_image_path"]); rts=resolve(root,p["A"]["text_original_path"])
        rir=Path("images")/"real"/aid/f"real{ris.suffix.lower() or '.png'}"; rtr=Path("texts")/"real"/aid/"real.txt"; cp(ris,root/rir); cp(rts,root/rtr)
        pir=Path("images")/"positive"/aid/"positive.png"; ptr=Path("texts")/"positive"/aid/"positive.txt"; pmr=Path("masks")/"positive"/aid/"positive_mask.png"
        p["B"]["line_image_path"]=mv(root,p["B"]["line_image_path"],pir); p["B"]["text_original_path"]=mv(root,p["B"]["text_original_path"],ptr); mo=p["B"].get("alignment_mask_path") or p["bridge"].get("alignment_mask_path"); p["B"]["alignment_mask_path"]=mv(root,mo,pmr); p["bridge"]["alignment_mask_path"]=pmr.as_posix(); p["A"]["line_image_path"]=rir.as_posix(); p["A"]["text_original_path"]=rtr.as_posix(); p["bridge"]["layout_version"]=1; p["bridge"]["group_path"]=(Path("anchors")/aid/"anchor.json").as_posix()
        ne=[]
        for i,n in enumerate(sorted(neg,key=lambda r:str(r.get("B_page_id","")))):
            nir=Path("images")/"negative"/aid/f"negative_{i:02d}.png"; ntr=Path("texts")/"negative"/aid/f"negative_{i:02d}.txt"; n["B"]["line_image_path"]=mv(root,n["B"]["line_image_path"],nir); n["B"]["text_original_path"]=mv(root,n["B"]["text_original_path"],ntr); n["A"]["line_image_path"]=rir.as_posix(); n["A"]["text_original_path"]=rtr.as_posix(); n["bridge"]["layout_version"]=1; n["bridge"]["negative_index"]=i; n["bridge"]["group_path"]=(Path("anchors")/aid/"anchor.json").as_posix(); ne.append({"index":i,"image":nir.as_posix(),"text":ntr.as_posix(),"fonts":n["bridge"].get("fonts",[]),"appearance_augmentation":n["bridge"].get("appearance_augmentation",{})})
        pe={"image":pir.as_posix(),"text":ptr.as_posix(),"mask":pmr.as_posix(),"shared_island_count":p["bridge"].get("shared_island_count"),"shared_texts":p["bridge"].get("shared_texts",[]),"shared_boxes_px":p["bridge"].get("shared_boxes_px",[]),"fonts":p["bridge"].get("fonts",[]),"appearance_augmentation":p["bridge"].get("appearance_augmentation",{})}; re={"page_id":page,"image":rir.as_posix(),"text":rtr.as_posix()}; ar={"anchor_id":aid,"pair_id":pair,"real":re,"positive":pe,"negatives":ne}; dump(root/"anchors"/aid/"anchor.json",ar); dump(root/"positive"/aid/"relation.json",{"anchor_id":aid,"pair_id":pair,"real":re,"positive":pe}); dump(root/"negative"/aid/"relations.json",{"anchor_id":aid,"pair_id":pair,"real":re,"negatives":ne}); index.append(ar); rewritten.append(p); rewritten.extend(sorted(neg,key=lambda r:int(r["bridge"].get("negative_index",0))))
    with man.open("w",encoding="utf-8") as h:
        for r in rewritten: h.write(json.dumps(r,ensure_ascii=False)+"\n")
    with (root/"anchor_index.jsonl").open("w",encoding="utf-8") as h:
        for r in index: h.write(json.dumps(r,ensure_ascii=False)+"\n")
    for d in (root/"images",root/"texts",root/"masks"):
        for c in list(d.iterdir()):
            if c.is_file(): c.unlink()
    meta.update({"layout_version":1,"layout_semantics":"anchor_grouped_self_contained","real_samples_copied":True,"relationship_key":"anchor_id"}); meta["folder_structure"]={"real_images":"images/real/<anchor_id>/real.*","positive_images":"images/positive/<anchor_id>/positive.png","negative_images":"images/negative/<anchor_id>/negative_XX.png","real_texts":"texts/real/<anchor_id>/real.txt","positive_texts":"texts/positive/<anchor_id>/positive.txt","negative_texts":"texts/negative/<anchor_id>/negative_XX.txt","positive_masks":"masks/positive/<anchor_id>/positive_mask.png","anchor_record":"anchors/<anchor_id>/anchor.json"}; meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+"\n",encoding="utf-8"); readme(root); print(f"organized anchors={len(index)} relationship_key=anchor_id")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",required=True); p.add_argument("--force",action="store_true"); a=p.parse_args(); organize(Path(a.data_dir).expanduser().resolve(),a.force)
if __name__=="__main__": main()
