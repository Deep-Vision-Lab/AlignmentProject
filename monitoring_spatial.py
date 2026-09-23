"""Reuse image-pair localization metrics on eligible, actual monitor members.

No pair labels, line crop boxes, or DTW scores are treated as character GT.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=None)
def xml_lines(path):
    from real_line_bbox_crop import parse_xml_boxes, group_boxes_into_lines
    return group_boxes_into_lines(parse_xml_boxes(path))[0]


def score_xml_transcript(record, transcript, bundle, index, text_encoder, P, config):
    """Only exact whole-line identity matches authorize XML part-of-word scoring.

    These annotations are NOT automatically words or individual characters.
    No fuzzy matching or inferred correction is used to create ground truth.
    """
    from PIL import Image
    from vlm_restoration_positive_dtw import _clean_letters, letter_dtw_cost_matrix
    from Evaluation.checkpoint_contract import resolve_evaluation_contract
    from Evaluation.point3_core import hard_letter_path, sequence_to_physical_window
    from Evaluation.yelda_geometry import source_intervals
    from Evaluation.point3_spatial_metrics import (_interval_iou, _binary_metrics,
        _column_labels, _region_matches, source_window_intervals)
    if not record.get("line_image_path"):
        return {"status": "unavailable", "reason": "no_native_line_identity"}
    source = Path(record["_root"]) / record["line_image_path"]
    xml = source.parent.parent / "original.xml"
    if source.parent.name != "linesImages" or not xml.is_file():
        return {"status": "unavailable", "reason": "no_native_XML"}
    lines = xml_lines(str(xml))
    line_idx = int(record["line_idx"]) - 1
    if not 0 <= line_idx < len(lines):
        return {"status": "unavailable", "reason": "XML_index_missing"}
    boxes = sorted(lines[line_idx], key=lambda b: -b["cx"])
    target = _clean_letters(transcript)
    labels = [_clean_letters(b["text"]) for b in boxes]
    if sum(labels, []) != target:
        return {"status": "unavailable", "reason": "XML_part_text_does_not_exactly_match_own_transcript"}
    page_paths = sorted(source.parent.parent.glob("original_image.*"))
    if not page_paths:
        return {"status": "unavailable", "reason": "missing_source_page_geometry"}
    with Image.open(page_paths[0]) as page, Image.open(source) as line:
        xscale = line.width / page.width
    contract = resolve_evaluation_contract(config)
    _, geometry = contract.prepare_line(source)
    valid = bundle["token_valid"][index].bool()
    logical = torch.where(valid)[0].tolist()
    costs = letter_dtw_cost_matrix(P, text_encoder, bundle["semantic"][index][valid], target)
    hard = hard_letter_path(costs.float().cpu().numpy(),
        vertical_penalty=P.positive_letter_dtw_vertical_penalty,
        horizontal_penalty=P.positive_letter_dtw_horizontal_penalty,
        position_prior_weight=P.positive_letter_dtw_position_prior,
        disable_horizontal_when_feasible=P.positive_letter_dtw_disable_horizontal_when_feasible)
    physical = source_window_intervals(geometry, contract.window_size, contract.stride)
    regions, offset = [], 0
    for box, label in zip(boxes, labels):
        if not label:
            continue
        rows = {i for i,j in hard.path if offset <= j < offset+len(label)}
        intervals = []
        for i in rows:
            physical_indices = bundle.get("physical_window_indices")
            index_value = logical[i] if physical_indices is None else int(physical_indices[index,logical[i]])
            _, x0, x1 = sequence_to_physical_window(index_value,
                len(valid) if physical_indices is None else contract.window_count, width=contract.line_width,
                window=contract.window_size, stride=contract.stride,
                use_flip=str(config.get("lang", "Arabic")).lower()=="arabic" if physical_indices is None else False)
            intervals.extend(source_intervals([[x0,x1]], geometry))
        pred = [(min(a for a,b in intervals), max(b for a,b in intervals))] if intervals else []
        gt = (box["x1"]*xscale, box["x2"]*xscale)
        width = int(geometry["source_width"])
        metrics = _binary_metrics(_column_labels(pred,width), _column_labels([gt],width))
        match = _region_matches(pred,[gt],width,5,.5,physical_windows=physical)[0]
        metrics.update(interval_iou=_interval_iou(pred[0],gt) if pred else 0.,
            center_error_px=abs(sum(pred[0])-sum(gt))/2 if pred else None,
            boundary_error_px=(abs(pred[0][0]-gt[0])+abs(pred[0][1]-gt[1]))/2 if pred else None,
            consecutive_window_support=match["consecutive_window_support"],
            success_iou_and_support=match["success"],
            annotation_type="XML PartOfWord (single letter)" if len(label)==1 else "XML PartOfWord (not character-level GT)",
            letters="".join(label), letter_start=offset, letter_end=offset+len(label), prediction=pred, target=gt)
        regions.append(metrics)
        offset += len(label)
    return dict(status="available", record_id=record["record_id"], regions=regions,
        identity_rule="exact cleaned RTL XML box concatenation equals own full transcript",
        coordinate_system="source line pixels; page XML x scaled to saved line width",
        min_consecutive_windows=5, min_region_iou=.5)


@lru_cache(maxsize=None)
def source_line_key(path):
    path = Path(path).resolve()
    if path.parent.name == "linesImages":
        pages = sorted(path.parent.parent.glob("original_image.*"))
        if pages:
            return hashlib.sha1(pages[0].read_bytes()).hexdigest(), path.stem
    return str(path), ""


def evaluate_spatial_splits(model, manifests, config, output):
    from Evaluation import eval_img_align_nw_diagnostic as base
    from Evaluation.eval_yelda import parse_args, evaluate_pair
    from Evaluation._eval_utils import EvaluationModels
    from Evaluation.checkpoint_contract import resolve_evaluation_contract, evaluation_environment
    from Evaluation.point3_spatial_metrics import score_source_regions
    from epoch_monitoring import deterministic_evaluation

    dataset = Path(config["dataset_path"])
    contract = resolve_evaluation_contract(config)
    manifest = dataset / config.get("real_manifest_name", "dataset_manifest_full_pairs.jsonl")
    # Reading all pair records does not assign or regenerate any split.
    if manifest.is_file():
        _, pairs = base.load_pairs(manifest, "all")
    else:
        _, pairs = base.load_pairs(dataset, "all")
    result = {}
    models = EvaluationModels(model, None, config, {}, next(model.parameters()).device, contract)
    models.pair_cross_attention = None
    args = parse_args(["--dataset", str(dataset), "--weights", "in-memory-epoch",
                       "--output-dir", str(output), "--representation", "primary",
                       "--image-preprocessing", "training"])
    args.feature = "contextual"
    for split, rows in manifests.items():
        members = {source_line_key(str(Path(r["_root"]) / r["line_image_path"]))
                   for r in rows if r.get("line_image_path")}
        selected = [p for p in pairs if source_line_key(str(p.image1)) in members
                    and source_line_key(str(p.image2)) in members]
        annotated = [p for p in selected if p.gt_mask1 is not None or p.gt_mask2 is not None]
        values = []
        with deterministic_evaluation(model, torch.nn.Identity(), 31415), evaluation_environment({
                "TRACE_COMPONENT_MIN_MATCHES": args.min_aligned_windows}):
            for pair in annotated:
                destination = output / split / f"pair_{pair.index:05d}"
                row = evaluate_pair(base, models, pair, args, destination)
                for side in (1, 2):
                    gt = getattr(pair, f"gt_mask{side}")
                    if gt is not None:
                        values.append(score_source_regions(row[f"line{side}_source_intervals_px"], gt,
                            row["geometry"][side-1], contract.window_size, contract.stride,
                            args.min_aligned_windows, 0.5))
        metric_keys = ("iou", "precision", "recall", "f1", "center_error_px", "region_success_rate")
        result[split] = dict(in_sample=split == "train_eval", eligible_pairs=len(selected),
            annotated_pairs=len(annotated), eligible_sides=len(values),
            pair_eligibility="both source-line identities must be actual members of this monitoring split",
            region_metrics={k: sum(v[k] for v in values if v[k] is not None)/sum(v[k] is not None for v in values)
                            if any(v[k] is not None for v in values) else None for k in metric_keys},
            localization_status="available region masks" if values else "unavailable: no matching source-space region masks in eligible pair records",
            character_localization="unavailable: no matching character/word identity-to-box annotations in current manifest",
            region_masks_are_character_accuracy=False)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(result, indent=2))
    return result
