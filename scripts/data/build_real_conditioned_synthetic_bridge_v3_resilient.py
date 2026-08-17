#!/usr/bin/env python3
"""Resilient Bridge V3 builder.

Uses the canonical V3 scientific/rendering primitives, but rejects and resamples
individual sentence/layout candidates instead of aborting the complete dataset when
one candidate cannot satisfy the readable-font/full-width constraints.
"""
from __future__ import annotations
import json, random, shutil
from pathlib import Path
from scripts.data import build_real_conditioned_synthetic_bridge_v3 as core

_RENDER_RETRY_MARKERS=("Full sentence does not fit","Generated sentence is too sparse")

def _render_with_retries(segments,fonts,output,mask_output,rng,args,attempts=8):
    for _ in range(max(1,int(attempts))):
        try: return core.render_segments(segments,fonts,output,mask_output,rng,args)
        except RuntimeError as exc:
            if not any(marker in str(exc) for marker in _RENDER_RETRY_MARKERS): raise
    return None

def _ordered(values,args):
    midpoint=(int(args.min_sentence_chars)+int(args.max_sentence_chars))/2.0
    unique={}
    for value in values:
        key=core.normalize_match_text(value)
        if key: unique.setdefault(key,value)
    return sorted(unique.values(),key=lambda v:(abs(len(core.compact(v))-midpoint),len(core.compact(v))))

def _choose_positive(candidates,spans,fonts,outdir,img_rel,mask_rel,rng,args):
    attempts=0
    counts=list(range(1,min(3,int(args.max_shared_islands))+1))
    for base in _ordered(candidates,args):
        rng.shuffle(counts)
        for requested in counts:
            for _ in range(3):
                attempts+=1
                shared=core.choose_nonoverlapping_shared(spans,rng,requested)
                segments=core.positive_segments(base,shared,rng,args.max_font_chunk_words)
                result=_render_with_retries(segments,fonts,outdir/img_rel,outdir/mask_rel,rng,args)
                if result is None: continue
                rendered,size,aug=result
                return {"base":base,"shared":shared,"segments":segments,"text":" ".join(s["text"] for s in segments).strip(),"rendered":rendered,"size":size,"aug":aug,"attempts":attempts}
    return None

def _render_negatives(anchor_index,anchor_text,candidates,selected_base,pool,fonts,anchor_id,outdir,rng,args):
    needed=int(args.negatives_per_anchor); results=[]; seen={core.normalize_match_text(selected_base)}; attempts=0
    def try_one(sentence):
        nonlocal attempts
        key=core.normalize_match_text(sentence)
        if not key or key in seen: return False
        seen.add(key); segs=core.sentence_segments(sentence,rng,args.max_font_chunk_words); idx=len(results)
        img=Path("images")/f"{anchor_id}_neg_{idx:02d}.png"; txt=Path("texts")/f"{anchor_id}_neg_{idx:02d}.txt"; attempts+=1
        result=_render_with_retries(segs,fonts,outdir/img,None,rng,args)
        if result is None: return False
        rendered,size,aug=result; results.append({"text":sentence,"rendered":rendered,"size":size,"aug":aug,"img":img,"txt":txt}); return True
    for candidate in _ordered(candidates,args):
        if len(results)>=needed: break
        try_one(candidate)
    compose_attempts=0
    while len(results)<needed and compose_attempts<160:
        compose_attempts+=1
        sentence=core.compose_safe_sentence(anchor_index,anchor_text,pool,fonts,seen,rng,args)
        if sentence is None: break
        try_one(sentence)
    return (results if len(results)==needed else None),attempts,compose_attempts

def build(args):
    data_dir=Path(args.data_dir).expanduser().resolve(); out=Path(args.output_dir).expanduser().resolve()
    if out.exists():
        if not args.overwrite: raise FileExistsError(f"Output exists: {out}; pass --overwrite")
        shutil.rmtree(out)
    for name in ("images","texts","masks"): (out/name).mkdir(parents=True,exist_ok=True)
    fonts=core._font_candidates(args.fonts); valid,test=core._positive_eval_pages(str(data_dir)); held=set(valid)|set(test)
    records=[r for r in core._all_unique_records(str(data_dir)) if str(r["page_id"]) not in held]; records.sort(key=lambda r:str(r["image_path"]))
    if args.max_anchors>0: records=records[:args.max_anchors]
    usable=[]; texts=[]; spans=[]
    for record in records:
        text=core._read(Path(record["text_path"])); cand=core.candidate_span_records(text,min_chars=args.min_positive_chars,max_chars=args.max_phrase_chars,max_words=args.max_phrase_words); cand=[c for c in cand if core.supported_fonts(c["text"],fonts)]
        if cand: usable.append(record); texts.append(text); spans.append(cand)
    if not usable: raise RuntimeError("No leakage-safe usable anchors with glyph-safe shared spans")
    rng=random.Random(args.seed); pool=core.phrase_pool(texts,fonts); rng.shuffle(pool)
    stats={"dataset_version":core.DATASET_VERSION,"dataset_revision":core.DATASET_REVISION,"dataset_semantics":"full_sentence_multi_island_mixed_font_white_on_black_glyphsafe_fullwidth_resampled","anchors_considered":len(records),"anchors_written":0,"positive_rows":0,"negative_rows":0,"positive_shared_islands_1":0,"positive_shared_islands_2":0,"positive_shared_islands_3":0,"positive_full_sentence_rows":0,"mixed_font_positive_rows":0,"mixed_font_negative_rows":0,"positive_render_attempts":0,"negative_render_attempts":0,"negative_compose_attempts":0,"layout_resampling":True,"negatives_per_anchor":args.negatives_per_anchor,"negative_ngram":args.negative_ngram,"min_overlap_word_chars":args.min_overlap_word_chars,"sentence_min_words":args.sentence_min_words,"sentence_max_words":args.sentence_max_words,"min_sentence_chars":args.min_sentence_chars,"max_sentence_chars":args.max_sentence_chars,"min_line_fill_ratio":args.min_line_fill_ratio,"font_size":args.font_size,"min_font_size":args.min_font_size,"max_font_size":args.max_font_size,"image_polarity":"white_text_on_black_background","font_mixing":"per_segment_glyph_safe","font_validation":"fonttools_unicode_cmap_on_shaped_text","appearance_augmentation":{"geometric":False,"blur_prob":args.blur_prob,"blur_max_radius":args.blur_max_radius,"noise_prob":args.noise_prob,"noise_sigma_max":args.noise_sigma_max,"contrast_range":[args.contrast_min,args.contrast_max],"brightness_range":[args.brightness_min,args.brightness_max]},"mask_semantics":"white=shared synthetic x-region; black=distractor/gap/background","heldout_page_count":len(held),"fonts":[p.name for p in fonts],"seed":args.seed}
    manifest_path=out/"dataset_manifest.jsonl"
    with manifest_path.open("w",encoding="utf-8") as manifest:
        for ai,(record,anchor_text,span_records) in enumerate(zip(usable,texts,spans)):
            candidates=core.choose_safe_sentences(ai,anchor_text,texts,pool,fonts,1+int(args.negatives_per_anchor)+8,rng,args)
            if len(candidates)<1+int(args.negatives_per_anchor): raise RuntimeError(f"Could not construct enough safe sentence candidates for {record['image_path']}")
            anchor_id=core._anchor_id(record); pair_id=f"bridge_{anchor_id}"; pos_img=Path("images")/f"{anchor_id}_pos_00.png"; pos_txt=Path("texts")/f"{anchor_id}_pos_00.txt"; pos_mask=Path("masks")/f"{anchor_id}_pos_00_mask.png"
            pos=_choose_positive(candidates,span_records,fonts,out,pos_img,pos_mask,rng,args)
            if pos is None: raise RuntimeError(f"Could not render a readable positive for {record['image_path']} after resampling")
            negs,na,nc=_render_negatives(ai,anchor_text,candidates,pos["base"],pool,fonts,anchor_id,out,rng,args)
            if negs is None: raise RuntimeError(f"Could not render {args.negatives_per_anchor} readable negatives for {record['image_path']} after resampling")
            stats["positive_render_attempts"]+=pos["attempts"]; stats["negative_render_attempts"]+=na; stats["negative_compose_attempts"]+=nc
            (out/pos_txt).write_text(core.clean_render_text(pos["text"]),encoding="utf-8"); shared_boxes=[s["bbox_px"] for s in pos["rendered"] if s["kind"]=="shared"]; shared_texts=[s["text"] for s in pos["rendered"] if s["kind"]=="shared"]; pos_fonts=sorted({s["font"] for s in pos["rendered"]}); shared_chars=sum(len(core.compact(t)) for t in shared_texts); anchor_chars=max(1,len(core.compact(anchor_text))); positive_chars=max(1,len(core.compact(pos["text"])))
            row={"pair_id":pair_id,"label_type":"medium_match","A_page_id":str(record["page_id"]),"B_page_id":f"synthetic:{pair_id}","A":core._side(str(Path(record["image_path"]).resolve()),str(Path(record["text_path"]).resolve())),"B":core._side(pos_img.as_posix(),pos_txt.as_posix(),mask_path=pos_mask.as_posix()),"scores":{"text_score":1.0,"avg_sim":1.0,"coverage_A":min(1.0,shared_chars/anchor_chars),"coverage_B":min(1.0,shared_chars/positive_chars)},"bridge":{"dataset_version":core.DATASET_VERSION,"dataset_revision":core.DATASET_REVISION,"relation":"positive_full_sentence_multi_island","anchor_id":anchor_id,"base_sentence":pos["base"],"positive_full_sentence":pos["text"],"shared_island_count":len(pos["shared"]),"shared_texts":shared_texts,"shared_boxes_px":shared_boxes,"segments":pos["rendered"],"alignment_mask_path":pos_mask.as_posix(),"fonts":pos_fonts,"font_size":pos["size"],"image_polarity":"white_text_on_black_background","appearance_augmentation":pos["aug"],"glyph_safe":True,"layout_resampled":pos["attempts"]>1,"render_attempts":pos["attempts"],"negative_ngram_guarantee":args.negative_ngram,"min_overlap_word_chars":args.min_overlap_word_chars}}
            manifest.write(json.dumps(row,ensure_ascii=False)+"\n"); stats["positive_rows"]+=1; stats["positive_full_sentence_rows"]+=1; stats[f"positive_shared_islands_{len(pos['shared'])}"]+=1; stats["mixed_font_positive_rows"]+=int(len(pos_fonts)>1)
            for ni,neg in enumerate(negs):
                (out/neg["txt"]).write_text(core.clean_render_text(neg["text"]),encoding="utf-8"); nf=sorted({s["font"] for s in neg["rendered"]}); nrow={"pair_id":pair_id,"label_type":"no_shared_content","A_page_id":str(record["page_id"]),"B_page_id":f"synthetic:{pair_id}:neg{ni}","A":core._side(str(Path(record["image_path"]).resolve()),str(Path(record["text_path"]).resolve())),"B":core._side(neg["img"].as_posix(),neg["txt"].as_posix()),"scores":{"text_score":0.0,"avg_sim":0.0,"coverage_A":0.0,"coverage_B":0.0},"bridge":{"dataset_version":core.DATASET_VERSION,"dataset_revision":core.DATASET_REVISION,"relation":"negative_full_sentence_no_shared_content","anchor_id":anchor_id,"negative_text":neg["text"],"segments":neg["rendered"],"fonts":nf,"font_size":neg["size"],"image_polarity":"white_text_on_black_background","appearance_augmentation":neg["aug"],"glyph_safe":True,"negative_ngram_guarantee":args.negative_ngram,"min_overlap_word_chars":args.min_overlap_word_chars}}
                manifest.write(json.dumps(nrow,ensure_ascii=False)+"\n"); stats["negative_rows"]+=1; stats["mixed_font_negative_rows"]+=int(len(nf)>1)
            stats["anchors_written"]+=1
    (out/"metadata.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print("=== REAL-SYNTHETIC BRIDGE V3 RESILIENT ===")
    for k,v in stats.items(): print(f"{k}={v}")
    print(f"output={out}"); print(f"manifest={manifest_path}")

if __name__=="__main__": build(core.parse_args())
