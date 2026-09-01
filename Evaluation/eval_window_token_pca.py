#!/usr/bin/env python3
"""Evaluate which Arabic letter/token each image window is closest to.

Primary evidence is direct cosine nearest-neighbour retrieval in the shared
image/text embedding space plus a joint PCA visualization. Synthetic63 does not
store per-character pixel boxes, so optional hard Span-DTW labels are reported
only as *path consistency*, never as ground-truth token accuracy.
"""
from __future__ import annotations

import argparse, csv, json, math, os, re, sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Parameters as P
P.export_environment()
from unified_line_geometry import install_evaluation_geometry
install_evaluation_geometry()
from Evaluation.vit_evaluation import install_vit_evaluation_loader
install_vit_evaluation_loader()
from Evaluation._eval_utils import get_image_features, load_evaluation_models
from fast_hard_alignment import hard_span_dtw_path_fast

ARABIC_LETTERS = tuple("ءآأؤإئابتثجحخدذرزسشصضطظعغفقكلمنهويىةٱ")
SUFFIXES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

@dataclass(frozen=True)
class Line:
    sample: int
    side: int
    image: Path
    text_path: Path
    text: str


def read_text(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return " " + f.read().strip() + " "


def prepared(model, text):
    return text.strip() if bool(getattr(model, "strip_text_edges", False)) else text


def is_token(ch):
    return len(ch) == 1 and not ch.isspace() and not ch.startswith("<")


def discover(root, start_index, n_samples, side):
    images, texts = root / "images", root / "texts"
    if not images.is_dir() or not texts.is_dir():
        raise SystemExit("Expected Synthetic63-style dataset with images/ and texts/.")
    ids = []
    rx = re.compile(r"^img1_(\d+)\.(png|jpg|jpeg|tif|tiff|bmp)$", re.I)
    for p in images.iterdir():
        m = rx.match(p.name)
        if m:
            ids.append(int(m.group(1)))
    ids = sorted(set(ids))[max(0, start_index - 1):][:max(1, n_samples)]
    sides = (1, 2) if side == "both" else (int(side),)
    out = []
    for idx in ids:
        for s in sides:
            image = next((images / f"img{s}_{idx}{ext}" for ext in SUFFIXES if (images / f"img{s}_{idx}{ext}").is_file()), None)
            text_path = texts / f"text{s}_{idx}.txt"
            if image and text_path.is_file():
                out.append(Line(idx, s, image, text_path, read_text(text_path)))
    if not out:
        raise SystemExit("No complete image/text samples found.")
    return out


def encode_many(model, texts):
    with torch.no_grad():
        return model.encode_many(texts) if hasattr(model, "encode_many") else [model(t) for t in texts]


def occurrences(model, text, enc):
    src = prepared(model, text)
    context = getattr(enc, "context_embeddings", None)
    context = enc.embeddings if context is None else context
    out = []
    for i, (start, length) in enumerate(zip(enc.starts, enc.lengths)):
        start, length = int(start), int(length)
        if start >= 0 and length == 1 and start < len(src):
            token = src[start:start + 1]
            if is_token(token):
                out.append((token, start, enc.embeddings[i], context[i]))
    return out


def token_vector(model, token, enc):
    rows = occurrences(model, token, enc)
    if not rows:
        raise RuntimeError(f"No one-character span for token {token!r}")
    return rows[0][2]


def build_prototypes(model, lines, vocab_mode):
    dataset_tokens = sorted({c for line in lines for c in prepared(model, line.text) if is_token(c)})
    if vocab_mode == "dataset":
        tokens = dataset_tokens
    elif vocab_mode == "arabic":
        tokens = list(ARABIC_LETTERS)
    else:
        tokens = sorted(set(dataset_tokens).union(ARABIC_LETTERS))
    token_enc = encode_many(model, tokens)
    core = F.normalize(torch.stack([token_vector(model, t, e) for t, e in zip(tokens, token_enc)]).float(), p=2, dim=-1)
    line_enc = encode_many(model, [line.text for line in lines])
    buckets, counts = defaultdict(list), defaultdict(int)
    for line, enc in zip(lines, line_enc):
        for token, _start, _core, ctx in occurrences(model, line.text, enc):
            buckets[token].append(ctx)
            counts[token] += 1
    context = []
    for token, fallback in zip(tokens, core):
        vec = torch.stack(buckets[token]).float().mean(0) if buckets.get(token) else fallback
        context.append(F.normalize(vec, p=2, dim=-1))
    return tokens, core, F.normalize(torch.stack(context).float(), p=2, dim=-1), counts, line_enc


def path_labels(model, text, enc, image_contextual):
    labels = {}
    src = prepared(model, text)
    try:
        path = hard_span_dtw_path_fast(enc, image_contextual, include_blank_steps=False)
    except Exception:
        return labels
    for step in path:
        a, b = int(step.get("text_start", -1)), int(step.get("text_end", -1))
        if step.get("is_blank") or step.get("is_space") or a < 0 or b - a != 1:
            continue
        token = src[a:b]
        if not is_token(token):
            continue
        for w in range(int(step["window_start"]), int(step["window_end"])):
            labels[w] = token
    return labels


def font():
    path = ROOT / "Fonts" / "Amiri-Regular.ttf"
    return FontProperties(fname=str(path)) if path.is_file() else None


def save_matrix_csv(path, tokens, matrix):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["window_index", *tokens])
        for i, row in enumerate(matrix):
            w.writerow([i, *[f"{float(v):.7f}" for v in row]])


def save_heatmap(path, tokens, matrix, title):
    fig, ax = plt.subplots(figsize=(max(10, matrix.shape[0] * .12), max(6, len(tokens) * .23)))
    im = ax.imshow(matrix.T, aspect="auto", interpolation="nearest", vmin=-1, vmax=1)
    ax.set_title(title); ax.set_xlabel("Image window (logical Arabic order)"); ax.set_ylabel("Letter/token")
    ax.set_yticks(np.arange(len(tokens))); ax.set_yticklabels(tokens, fontproperties=font())
    step = max(1, matrix.shape[0] // 20); ax.set_xticks(np.arange(0, matrix.shape[0], step))
    fig.colorbar(im, ax=ax, label="cosine similarity"); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def pca(matrix):
    x = np.asarray(matrix, np.float64); centered = x - x.mean(0, keepdims=True)
    _u, s, vt = np.linalg.svd(centered, full_matrices=False)
    var = s ** 2 / max(1, len(x) - 1); ratio = var / max(float(var.sum()), 1e-12)
    return (centered @ vt[:2].T).astype(np.float32), ratio


def save_pca(path, windows, prototypes, tokens, predicted, title):
    coords, ratio = pca(np.concatenate([windows, prototypes], 0)); wc, pc = coords[:len(windows)], coords[len(windows):]
    cmap, denom = plt.get_cmap("nipy_spectral"), max(1, len(tokens) - 1)
    colors = [cmap(int(x) / denom) for x in predicted]
    fig, ax = plt.subplots(figsize=(13, 10)); ax.scatter(wc[:,0], wc[:,1], c=colors, s=18, alpha=.55, linewidths=0)
    ax.scatter(pc[:,0], pc[:,1], marker="*", s=150, c="black", edgecolors="white", linewidths=.7)
    for i, token in enumerate(tokens):
        ax.annotate(token, pc[i], xytext=(4,4), textcoords="offset points", fontsize=11, fontproperties=font())
    ax.set_xlabel(f"PC1 ({ratio[0]*100:.1f}% variance)"); ax.set_ylabel(f"PC2 ({ratio[1]*100:.1f}% variance)")
    ax.set_title(title + "\nWindow color = nearest prototype in original embedding space"); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=190); plt.close(fig); return ratio


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "run"


def consistency(rows, prefix, top_k):
    eligible = [r for r in rows if r["path_reference_token"]]
    out = {f"{prefix}_path_reference_windows": len(eligible)}
    for k in (1, 3, 5):
        kk = min(k, top_k)
        out[f"{prefix}_path_top{k}_consistency"] = None if not eligible else sum(
            r["path_reference_token"] in [r[f"{prefix}_top{i}_token"] for i in range(1, kk + 1)] for r in eligible
        ) / len(eligible)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True); p.add_argument("--weights", required=True); p.add_argument("--output-dir")
    p.add_argument("--device", default="auto"); p.add_argument("--start-index", type=int, default=1)
    p.add_argument("--n-samples", type=int, default=min(10, int(getattr(P, "evaluation_n_samples", 10))))
    p.add_argument("--side", choices=("1","2","both"), default="both")
    p.add_argument("--feature", choices=("contextual","local","grouped"), default=str(getattr(P, "evaluation_feature", "contextual")))
    p.add_argument("--vocab-mode", choices=("dataset","arabic","union"), default="union")
    p.add_argument("--top-k", type=int, default=5); p.add_argument("--max-pca-windows", type=int, default=1500)
    p.add_argument("--seed", type=int, default=int(getattr(P, "train_seed", 42))); p.add_argument("--no-path-reference", action="store_true")
    return p.parse_args()


def main():
    args = parse_args(); dataset = Path(args.dataset).expanduser().resolve(); weights = Path(args.weights).expanduser().resolve()
    if not dataset.is_dir(): raise SystemExit(f"Dataset does not exist: {dataset}")
    if not weights.is_file(): raise SystemExit(f"Weights do not exist: {weights}")
    lines = discover(dataset, args.start_index, args.n_samples, args.side)
    models = load_evaluation_models(weights, args.device, load_text_model=True)
    if models.text_model is None or not hasattr(models.text_model, "encode_many"):
        raise RuntimeError("Current token diagnostic requires the Arabic span text encoder.")
    output = Path(args.output_dir).expanduser().resolve() if args.output_dir else ROOT/"Results"/"Evaluation"/"TokenPCA"/safe_name(dataset.name)/safe_name(weights.parent.name)
    (output/"lines").mkdir(parents=True, exist_ok=True)

    tokens, core_proto, ctx_proto, counts, text_encodings = build_prototypes(models.text_model, lines, args.vocab_mode)
    top_k = max(1, min(args.top_k, len(tokens))); rows, windows, core_pred, ctx_pred = [], [], [], []
    for line, text_enc in zip(lines, text_encodings):
        feat = get_image_features(models, line.image, "synthetic"); image = F.normalize(feat.select(args.feature).float(), p=2, dim=-1)
        core_sim, ctx_sim = image @ core_proto.T, image @ ctx_proto.T
        cv, ci = torch.topk(core_sim, top_k, dim=1); xv, xi = torch.topk(ctx_sim, top_k, dim=1)
        ref = {} if args.no_path_reference else path_labels(models.text_model, line.text, text_enc, F.normalize(feat.contextual.float(), p=2, dim=-1))
        core_np, ctx_np = core_sim.cpu().numpy(), ctx_sim.cpu().numpy(); stem = f"sample_{line.sample:05d}_line{line.side}"
        save_matrix_csv(output/"lines"/f"{stem}_core_cosine.csv", tokens, core_np); save_matrix_csv(output/"lines"/f"{stem}_context_cosine.csv", tokens, ctx_np)
        save_heatmap(output/"lines"/f"{stem}_core_cosine.png", tokens, core_np, f"{stem}: window ↔ pure letter cosine")
        save_heatmap(output/"lines"/f"{stem}_context_cosine.png", tokens, ctx_np, f"{stem}: window ↔ contextual letter cosine")
        ci, cv, xi, xv = ci.cpu().numpy(), cv.cpu().numpy(), xi.cpu().numpy(), xv.cpu().numpy(); ink = feat.ink.cpu().numpy(); n = len(image)
        for widx in range(n):
            row = {"sample_index":line.sample,"side":line.side,"image":str(line.image),"text":prepared(models.text_model,line.text),
                   "window_index":widx,"physical_window_index":n-1-widx if bool(getattr(models.image_model,"use_flip",False)) else widx,
                   "ink":float(ink[widx]),"path_reference_token":ref.get(widx,"")}
            for rank in range(top_k):
                row[f"core_top{rank+1}_token"], row[f"core_top{rank+1}_cosine"] = tokens[int(ci[widx,rank])], float(cv[widx,rank])
                row[f"context_top{rank+1}_token"], row[f"context_top{rank+1}_cosine"] = tokens[int(xi[widx,rank])], float(xv[widx,rank])
            row["core_margin_top1_top2"] = float(cv[widx,0]-cv[widx,1]) if top_k > 1 else None
            row["context_margin_top1_top2"] = float(xv[widx,0]-xv[widx,1]) if top_k > 1 else None; rows.append(row)
        windows.append(image.cpu()); core_pred.extend(ci[:,0]); ctx_pred.extend(xi[:,0]); print(f"[{line.sample}:line{line.side}] windows={n} tokens={len(tokens)} feature={args.feature}", flush=True)

    window_np = torch.cat(windows).numpy(); rng = np.random.default_rng(args.seed)
    sel = np.sort(rng.choice(len(window_np), args.max_pca_windows, replace=False)) if len(window_np) > args.max_pca_windows else np.arange(len(window_np))
    core_ratio = save_pca(output/"pca_core_letters.png", window_np[sel], core_proto.cpu().numpy(), tokens, np.asarray(core_pred)[sel], "PCA: image windows + pure letter prototypes")
    ctx_ratio = save_pca(output/"pca_context_letters.png", window_np[sel], ctx_proto.cpu().numpy(), tokens, np.asarray(ctx_pred)[sel], "PCA: image windows + contextual letter prototypes")
    fig, ax = plt.subplots(figsize=(10,5)); k=min(20,len(core_ratio)); ax.bar(np.arange(1,k+1),core_ratio[:k]*100); ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance (%)"); ax.set_title("Core-prototype joint PCA"); fig.tight_layout(); fig.savefig(output/"pca_explained_variance.png",dpi=180); plt.close(fig)

    fields=[]
    for r in rows:
        for key in r:
            if key not in fields: fields.append(key)
    with (output/"windows_nearest_tokens.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    with (output/"token_prototypes.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.writer(f); cp, xp=core_proto.cpu().numpy(),ctx_proto.cpu().numpy(); w.writerow(["token","context_occurrences",*[f"core_{i}" for i in range(cp.shape[1])],*[f"context_{i}" for i in range(xp.shape[1])]])
        for i,t in enumerate(tokens): w.writerow([t,int(counts.get(t,0)),*cp[i],*xp[i]])
    np.save(output/"window_embeddings.npy",window_np); np.save(output/"core_token_prototypes.npy",core_proto.cpu().numpy()); np.save(output/"context_token_prototypes.npy",ctx_proto.cpu().numpy())

    def mean(key):
        vals=[float(r[key]) for r in rows if r.get(key) is not None and math.isfinite(float(r[key]))]; return float(np.mean(vals)) if vals else None
    summary={"dataset":str(dataset),"weights":str(weights),"feature":args.feature,"vocab_mode":args.vocab_mode,"lines_evaluated":len(lines),"windows_evaluated":len(rows),"token_count":len(tokens),"tokens":tokens,
             "core_mean_top1_cosine":mean("core_top1_cosine"),"context_mean_top1_cosine":mean("context_top1_cosine"),"core_mean_top1_top2_margin":mean("core_margin_top1_top2"),"context_mean_top1_top2_margin":mean("context_margin_top1_top2"),
             "core_pca_pc1_ratio":float(core_ratio[0]),"core_pca_pc1_pc2_ratio":float(core_ratio[:2].sum()),"context_pca_pc1_ratio":float(ctx_ratio[0]),"context_pca_pc1_pc2_ratio":float(ctx_ratio[:2].sum()),"path_reference_is_ground_truth":False,
             **consistency(rows,"core",top_k),**consistency(rows,"context",top_k)}
    with (output/"summary.json").open("w",encoding="utf-8") as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    print(f"\nToken diagnostic complete\noutput={output}\nwindows={len(rows)} tokens={len(tokens)}",flush=True)
    print("Path-reference consistency is model-derived, not GT token accuracy.",flush=True)

if __name__ == "__main__":
    main()
