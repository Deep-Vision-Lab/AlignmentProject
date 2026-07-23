import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from Parameters import (
    contrastive_margin,
    contrastive_soft_dtw_gamma,
    contrastive_temperature,
    max_windows_per_span,
    span_dtw_bucket_text_lengths,
    span_dtw_max_text_bucket,
    span_dtw_text_bucket_size,
)


_SPAN_DTW_MEM_DEBUG = os.environ.get("SPAN_DTW_MEM_DEBUG", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_SPAN_DTW_MEM_DEBUG_INTERVAL = max(
    1, int(os.environ.get("SPAN_DTW_MEM_DEBUG_INTERVAL", "50"))
)
_JAX_DENSE_CALL_COUNT = 0
_JAX_DENSE_SHAPE_COUNTS = {}


def _env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def softmin(values, gamma):
    return -gamma * torch.logsumexp(-values / gamma, dim=0)


def _alignment_span_embeddings(span_encoding):
    """Use overlap context for global DTW, never for local supervision.

    ``SpanEncoding.embeddings`` contains the visible core. New encoders expose
    ``context_embeddings`` separately. Legacy encoders keep working because the
    fallback is the original embeddings tensor.
    """
    contextual = getattr(span_encoding, "context_embeddings", None)
    return contextual if contextual is not None else span_encoding.embeddings


def _current_memory_summary():
    parts = []
    try:
        import psutil

        process = psutil.Process(os.getpid())
        parts.append(f"rss_mb={process.memory_info().rss / (1024 ** 2):.1f}")
    except Exception:
        pass

    if torch.cuda.is_available():
        parts.extend(
            [
                f"cuda_alloc_mb={torch.cuda.memory_allocated() / (1024 ** 2):.1f}",
                f"cuda_reserved_mb={torch.cuda.memory_reserved() / (1024 ** 2):.1f}",
                f"cuda_max_alloc_mb={torch.cuda.max_memory_allocated() / (1024 ** 2):.1f}",
            ]
        )
    return " ".join(parts)


def _bucket_length(length, bucket_size, max_bucket, enabled=True):
    length = int(length)
    if not enabled:
        return length
    bucket_size = max(1, int(bucket_size))
    max_bucket = max(1, int(max_bucket))
    bucketed = ((length + bucket_size - 1) // bucket_size) * bucket_size
    if length <= max_bucket:
        bucketed = min(bucketed, max_bucket)
    return max(length, bucketed)


def _debug_jax_dense_tensor(
    span_encoding, image_embeddings, dense, text_steps_padded=None
):
    global _JAX_DENSE_CALL_COUNT
    _JAX_DENSE_CALL_COUNT += 1
    shape = tuple(dense.shape)
    previous_shape_count = _JAX_DENSE_SHAPE_COUNTS.get(shape, 0)
    _JAX_DENSE_SHAPE_COUNTS[shape] = previous_shape_count + 1
    should_print = (
        previous_shape_count == 0
        or (_JAX_DENSE_CALL_COUNT % _SPAN_DTW_MEM_DEBUG_INTERVAL == 0)
    )
    if not should_print:
        return
    dense_mb = dense.numel() * dense.element_size() / (1024**2)
    print(
        "[SPAN_MEM_DEBUG] "
        f"jax_dense_call={_JAX_DENSE_CALL_COUNT} "
        f"new_shape={previous_shape_count == 0} "
        f"unique_dense_shapes={len(_JAX_DENSE_SHAPE_COUNTS)} "
        f"dense_shape={shape} dense_mb={dense_mb:.2f} "
        f"actual_text_length={span_encoding.text_length} "
        f"bucketed_text_length={text_steps_padded or span_encoding.text_length} "
        f"num_valid_spans={len(span_encoding.starts)} "
        f"image_windows={image_embeddings.shape[0]} "
        f"{_current_memory_summary()}",
        flush=True,
    )


def _debug_jax_cost(span_encoding, image_embeddings, cost):
    if not _SPAN_DTW_MEM_DEBUG:
        return
    detached = cost.detach()
    is_bad = (not torch.isfinite(detached).all().item()) or detached.item() < -1e-3
    if not is_bad:
        return
    print(
        "[SPAN_MEM_DEBUG_BAD_COST] "
        f"cost={detached.item()} "
        f"text_length={span_encoding.text_length} "
        f"image_windows={image_embeddings.shape[0]} "
        f"num_valid_spans={len(span_encoding.starts)} "
        f"{_current_memory_summary()}",
        flush=True,
    )


def span_index_by_start_and_length(span_encoding):
    return {
        (start, length): idx
        for idx, (start, length) in enumerate(
            zip(span_encoding.starts, span_encoding.lengths)
        )
    }


def _spans_by_start(span_encoding):
    spans = {}
    for start, length in zip(span_encoding.starts, span_encoding.lengths):
        spans.setdefault(start, []).append(length)
    return spans


def _min_required_spans(span_encoding):
    text_length = span_encoding.text_length
    inf = float("inf")
    min_spans = [inf] * (text_length + 1)
    min_spans[0] = 0
    spans_by_start = _spans_by_start(span_encoding)
    for index in range(text_length):
        if min_spans[index] == inf:
            continue
        for span_length in spans_by_start.get(index, []):
            next_index = index + span_length
            if next_index <= text_length:
                min_spans[next_index] = min(
                    min_spans[next_index], min_spans[index] + 1
                )
    return min_spans[text_length]


def _span_no_path_message(
    span_encoding,
    image_steps,
    max_windows_per_span,
    min_required_spans,
    reason=None,
):
    max_span_chars = getattr(span_encoding, "max_span_chars", "unknown")
    prefix = "No valid span-DTW path. "
    if reason == "too_few_windows":
        detail = "There are too few image windows for the text length. "
        fix = (
            "Fix by: 1. decreasing STRIDE_RATIO; 2. increasing "
            "MAX_TEXT_SPAN_CHARS; 3. using a smaller WINDOW_SIZE only if it "
            "increases usable windows. Do not fix this by increasing "
            "MAX_WINDOWS_PER_SPAN."
        )
    elif reason == "too_many_windows":
        detail = (
            "There are too many image windows for this text under "
            "MAX_WINDOWS_PER_SPAN. "
        )
        fix = (
            "Increase MAX_WINDOWS_PER_SPAN, reduce image windows, or treat the "
            "transcript as an infeasible/easy negative."
        )
    else:
        detail = "The text/window constraints are infeasible. "
        fix = (
            "Check STRIDE_RATIO, MAX_TEXT_SPAN_CHARS, "
            "MAX_WINDOWS_PER_SPAN, and transcript length."
        )
    return (
        prefix
        + detail
        + f"text_length={span_encoding.text_length}, image_windows={image_steps}, "
        + f"minimum_required_spans={min_required_spans}, "
        + f"max_text_span_chars={max_span_chars}, "
        + f"max_windows_per_span={max_windows_per_span}. "
        + fix
    )


def _infeasible_negative_cost(image_embeddings, text_length, temperature):
    return image_embeddings.new_tensor(
        max(1, int(image_embeddings.shape[0]), int(text_length))
        * (2.0 / temperature)
    )


def _precompute_transition_costs(
    span_encoding,
    image_embeddings,
    temperature,
    max_windows_per_span,
    window_count_penalty,
):
    # Global Span-DTW uses context-aware surfaces. Local hard-negative training
    # reads span_encoding.embeddings directly and therefore uses visible cores.
    span_embeddings = F.normalize(
        _alignment_span_embeddings(span_encoding).float(), p=2, dim=-1
    )
    image_embeddings = F.normalize(image_embeddings.float(), p=2, dim=-1)
    image_steps = image_embeddings.shape[0]
    max_windows = min(max_windows_per_span, image_steps)

    cosine_similarities = torch.matmul(span_embeddings, image_embeddings.T)
    window_costs = (1.0 - cosine_similarities) / temperature
    prefix = torch.cat(
        [
            torch.zeros(
                window_costs.shape[0],
                1,
                device=window_costs.device,
                dtype=window_costs.dtype,
            ),
            window_costs.cumsum(dim=1),
        ],
        dim=1,
    )

    costs_by_window_count = {}
    for window_count in range(1, max_windows + 1):
        range_sums = prefixRv–æF÷uö6÷VçC¥ÒÒ&Vf—…³¢¢×v–æF÷uö6÷VçEÐ¢&ævUöÖVç2Ò&ævU÷7V×2òv–æF÷uö6÷Vç@¢6÷7G5ö'•÷v–æF÷uö6÷VçE·v–æF÷uö6÷VçEÒÒ€¢&ævUöÖVç2²v–æF÷uö6÷VçE÷VæÇG’¢‡v–æF÷uö6÷VçBÒ¢¢&WGW&â6÷7G5ö'•÷v–æF÷uö6÷Vç@  ¦FVböFVç6U÷G&ç6—F–öåö6÷7G2€¢7åöVæ6öF–ærÀ¢–ÖvUöVÖ&VFF–æw2À¢FV×W&GW&RÀ¢Ö…÷v–æF÷w5÷W%÷7âÀ¢v–æF÷uö6÷VçE÷VæÇG’À¢FW‡E÷7FW5÷FFVCÔæöæRÀ¢“ ¢7GVÅ÷FW‡E÷7FW2Ò7åöVæ6öF–ærçFW‡EöÆVæwF€¢FVç6U÷FW‡E÷7FW2Ò–çB‡FW‡E÷7FW5÷FFVB÷"7GVÅ÷FW‡E÷7FW2¢–bFVç6U÷FW‡E÷7FW2Â7GVÅ÷FW‡E÷7FW3 ¢&—6RfÇVTW'&÷"€¢b'FW‡E÷7FW5÷FFVC×¶FVç6U÷FW‡E÷7FW7Ò—26ÖÆÆW"F†â ¢b&7GVÂFW‡BÆVæwF‚¶7GVÅ÷FW‡E÷7FW7Òâ ¢¢–ÖvU÷7FW2Ò–ÖvUöVÖ&VFF–æw2ç6†U³Ð¢Ö…÷7åö6†'2ÒÖ‚€¢–çB†vWFGG"‡7åöVæ6öF–ærÂ&Ö…÷7åö6†'2"Â’’À¢Ö‚‡7åöVæ6öF–æræÆVæwF‡2ÂFVfVÇCÓ’À¢¢G&ç6—F–öåö6÷7G2Ò÷&V6ö×WFU÷G&ç6—F–öåö6÷7G2€¢7åöVæ6öF–ærÀ¢–ÖvUöVÖ&VFF–æw2À¢FV×W&GW&RÀ¢Ö…÷v–æF÷w5÷W%÷7âÀ¢v–æF÷uö6÷VçE÷VæÇG’À¢¢FVç6RÒ–ÖvUöVÖ&VFF–æw2ææWuögVÆÂ€¢€¢Ö…÷7åö6†'2²À¢Ö…÷v–æF÷w5÷W%÷7â²À¢FVç6U÷FW‡E÷7FW2²À¢–ÖvU÷7FW2²À¢’À¢fÆöB‚&–æb"’À¢¢f÷"7åö–G‚Â‡7F'BÂ7åöÆVâ’–âVçVÖW&FR€¢¦—‡7åöVæ6öF–ærç7F'G2Â7åöVæ6öF–æræÆVæwF‡2¢“ ¢–b7åöÆVââÖ…÷7åö6†'2÷"7F'B²7åöÆVââ7GVÅ÷FW‡E÷7FW3 ¢6öçF–çVP¢f÷"v–æF÷uö6÷VçBÂ6÷7G2–âG&ç6—F–öåö6÷7G2æ—FV×2‚“ ¢–b6÷7G2ç6†U³ÒÓÒ ¢6öçF–çVP¢FVç6U·7åöÆVâÂv–æF÷uö6÷VçBÂ7F'BÂ¢6÷7G2ç6†U³ÕÒÒ6÷7G5·7åö–G…Ð¢&WGW&âFVç6P  ¦FVb†&E÷7åöGGu÷F‚€¢7åöVæ6öF–ærÀ¢–ÖvUöVÖ&VFF–æw2À¢FV×W&GW&SÖ6öçG&7F—fU÷FV×W&GW&RÀ¢Ö…÷v–æF÷w3ÖÖ…÷v–æF÷w5÷W%÷7âÀ¢v–æF÷uö6÷VçE÷VæÇG“ÔæöæRÀ¢“ ¢–bv–æF÷uö6÷VçE÷VæÇG’—2æöæS ¢v–æF÷uö6÷VçE÷VæÇG’ÒöVçeöfÆöB‚%5åõt”äDõuô4õTåEõTäÅE’"ÂãR¢7åöÆöö·WÒ7åö–æFW…ö'•÷7F'EöæEöÆVæwF‚‡7åöVæ6öF–ær¢FW‡E÷7FW2Ò7åöVæ6öF–ærçFW‡EöÆVæwF€¢–ÖvU÷7FW2Ò–ÖvUöVÖ&VFF–æw2ç6†U³Ð¢Ö–å÷&WV—&VBÒöÖ–å÷&WV—&VE÷7ç2‡7åöVæ6öF–ær¢–bÖ–å÷&WV—&VBÓÒfÆöB‚&–æb"’÷"–ÖvU÷7FW2ÂÖ–å÷&WV—&VC ¢&—6RfÇVTW'&÷"€¢÷7åöæõ÷F…öÖW76vR€¢7åöVæ6öF–ærÀ¢–ÖvU÷7FW2À¢Ö…÷v–æF÷w2À¢Ö–å÷&WV—&VBÀ¢&V6öãÒ'FöõöfWu÷v–æF÷w2"À¢¢¢Ö…÷÷76–&ÆU÷v–æF÷w2Ò7åöVæ6öF–ærçFW‡EöÆVæwF‚¢Ö…÷v–æF÷w0¢–b–ÖvU÷7FW2âÖ…÷÷76–&ÆU÷v–æF÷w3 ¢&—6RfÇVTW'&÷"€¢÷7åöæõ÷F…öÖW76vR€¢7åöVæ6öF–ærÀ¢–ÖvU÷7FW2À¢Ö…÷v–æF÷w2À¢Ö–å÷&WV—&VBÀ¢&V6öãÒ'FöõöÖç•÷v–æF÷w2"À¢¢¢G&ç6—F–öåö6÷7G2Ò÷&V6ö×WFU÷G&ç6—F–öåö6÷7G2€¢7åöVæ6öF–ærÀ¢–ÖvUöVÖ&VFF–æw2À¢FV×W&GW&RÀ¢Ö…÷v–æF÷w2À¢v–æF÷uö6÷VçE÷VæÇG’À¢¢GÒF÷&6‚ægVÆÂ‚‡FW‡E÷7FW2²Â–ÖvU÷7FW2²’ÂfÆöB‚&–æb"’¢&6²Òµ´æöæRf÷"ò–â&ævR†–ÖvU÷7FW2²•Òf÷"ò–â&ævR‡FW‡E÷7FW2²•Ð¢G³ÂÒÒã  ¢f÷"–æFW…÷FW‡B–â&ævR‡FW‡E÷7FW2²“ ¢f÷"–æFW…ö–ÖvR–â&ævR†–ÖvU÷7FW2²“ ¢–bæ÷BF÷&6‚æ—6f–æ—FR†G¶–æFW…÷FW‡BÂ–æFW…ö–ÖvUÒ“ ¢6öçF–çVP¢f÷"7åöÆVæwF‚–â&ævRƒÂFW‡E÷7FW2Ò–æFW…÷FW‡B²“ ¢7åö–G‚Ò7åöÆöö·WævWB‚†–æFW…÷FW‡BÂ7åöÆVæwF‚’¢–b7åö–G‚—2æöæS ¢6öçF–çVP¢f÷"v–æF÷uö6÷VçB–â&ævR€¢ÂÖ–â†Ö…÷v–æF÷w2Â–ÖvU÷7FW2Ò–æFW…ö–ÖvR’²¢“ ¢æW‡E÷FW‡BÒ–æFW…÷FW‡B²7åöÆVæwF€¢æW‡Eö–ÖvRÒ–æFW…ö–ÖvR²v–æF÷uö6÷Vç@¢6÷7BÒG&ç6—F–öåö6÷7G5·v–æF÷uö6÷VçEÕ·7åö–G‚Â–æFW…ö–ÖvUÐ¢6æF–FFRÒG¶–æFW…÷FW‡BÂ–æFW…ö–ÖvUÒ²6÷7BæFWF6‚‚’æ7R‚¢–b6æF–FFRÂG¶æW‡E÷FW‡BÂæW‡Eö–ÖvUÓ ¢G¶æW‡E÷FW‡BÂæW‡Eö–ÖvUÒÒ6æF–FFP¢&6µ¶æW‡E÷FW‡EÕ¶æW‡Eö–ÖvUÒÒ€¢–æFW…÷FW‡BÀ¢–æFW…ö–ÖvRÀ¢7åö–G‚À¢ ¢–b&6µ·FW‡E÷7FW5Õ¶–ÖvU÷7FW5Ò—2æöæS ¢&—6RfÇVTW'&÷"€¢÷7åöæõ÷F…öÖW76vR€¢7åöVæ6öF–ærÀ¢–ÖvU÷7FW2À¢Ö…÷v–æF÷w2À¢Ö–å÷&WV—&VBÀ¢&V6öãÒ&vVæW&–2"À¢¢ ¢F‚ÒµÐ¢–æFW…÷FW‡BÂ–æFW…ö–ÖvRÒFW‡E÷7FW2Â–ÖvU÷7FW0¢v†–ÆR&6µ¶–æFW…÷FW‡EÕ¶–æFW…ö–ÖvUÒ—2æ÷BæöæS ¢&We÷FW‡BÂ&Weö–ÖvRÂ7åö–G‚Ò&6µ¶–æFW…÷FW‡EÕ¶–æFW…ö–ÖvUÐ¢6÷&U÷FW‡BÒ7åöVæ6öF–ærçFW‡G5·7åö–G…Ð¢7W&f6U÷FW‡G2ÒvWFGG"‡7åöVæ6öF–ærÂ'7W&f6U÷FW‡G2"ÂæöæR¢&rÒvWFGG"‡7åöVæ6öF–ærÂ'&u÷FW‡G2"ÂæöæR¢—5÷76RÒvWFGG"‡7åöVæ6öF–ærÂ&—5÷76R"ÂæöæR¢F‚æVæB€¢°¢'FW‡E÷7F'B#¢&We÷FW‡BÀ¢'FW‡EöVæB#¢–æFW…÷FW‡BÀ¢'v–æF÷u÷7F'B#¢&Weö–ÖvRÀ¢'v–æF÷uöVæB#¢–æFW…ö–ÖvRÀ¢'7åö–G‚#¢7åö–G‚À¢'FW‡B#¢6÷&U÷FW‡BÀ¢'7W&f6U÷FW‡B#¢€¢7W&f6U÷FW‡G5·7åö–G…Ð¢–b7W&f6U÷FW‡G2—2æ÷BæöæP¢VÇ6R6÷&U÷FW‡@¢’À¢'&u÷FW‡B#¢&u·7åö–G…Ò–b&r—2æ÷BæöæRVÇ6R6÷&U÷FW‡BÀ¢&—5÷76R#¢&ööÂ†—5÷76U·7åö–G…Ò’–b—5÷76R—2æ÷BæöæRVÇ6RfÇ6RÀ¢Ð¢¢–æFW…÷FW‡BÂ–æFW…ö–ÖvRÒ&We÷FW‡BÂ&Weö–ÖvP¢F‚ç&WfW'6R‚¢&WGW&âF€  ¦6Æ727ä6öçG&7F—fU6ögDEEr†æâäÖöGVÆR“ ¢FVbõö–æ—Eõò€¢6VÆbÀ¢vÖÖÖ6öçG&7F—fU÷6ögEöGGuövÖÖÀ¢Ö&v–ãÖ6öçG&7F—fUöÖ&v–âÀ¢FV×W&GW&SÖ6öçG&7F—fU÷FV×W&GW&RÀ¢Ö…÷v–æF÷w5÷W%÷7ãÖÖ…÷v–æF÷w5÷W%÷7âÀ¢v–æF÷uö6÷VçE÷VæÇG“ÔæöæRÀ¢æVvF—fUöw&EöÖöFSÒ&†&FW7B"À¢&6¶VæCÒ'F÷&6‚"À¢“ ¢7WW"‚’åõö–æ—Eõò‚¢6VÆbævÖÖÒvÖÖ¢6VÆbæÖ&v–âÒÖ&v–à¢6VÆbçFV×W&GW&RÒFV×W&GW&P¢6VÆbæÖ…÷v–æF÷w5÷W%÷7âÒÖ…÷v–æF÷w5÷W%÷7à¢6VÆbçv–æF÷uö6÷VçE÷VæÇG’Ò€¢öVçeöfÆöB‚%5åõt”äDõuô4õTåEõTäÅE’"ÂãR¢–bv–æF÷uö6÷VçE÷VæÇG’—2æöæP¢VÇ6RfÆöB‡v–æF÷uö6÷VçE÷VæÇG’¢¢6VÆbææVvF—fUöw&EöÖöFRÒ7G"†æVvF—fUöw&EöÖöFR’æÆ÷vW"‚¢6VÆbæ&6¶VæBÒ7G"†&6¶VæB’æÆ÷vW"‚¢6VÆbå÷v&æVEö¦…ö&6¶VæBÒfÇ6P¢–b6VÆbææVvF—fUöw&EöÖöFRæ÷B–â²&ÆÂ"Â&†&FW7B"Â&æöæR'Ó ¢&—6RfÇVTW'&÷"€¢&æVvF—fUöw&EöÖöFR×W7B&RÆÂÂ†&FW7BÂ÷"æöæS² ¢b&v÷B¶æVvF—fUöw&EöÖöFR'Ò ¢¢–b6VÆbæ&6¶VæBæ÷B–â²'F÷&6‚"Â&¦‚'Ó ¢&—6RfÇVTW'&÷"‚&&6¶VæB×W7B&RwF÷&6‚r÷"v¦‚r" ¢FVb÷7åöGGuö6÷7B‡6VÆbÂ7åöVæ6öF–ærÂ–ÖvUöVÖ&VFF–æw2“ ¢–b6VÆbæ&6¶VæBÓÒ'F÷&6‚# ¢&WGW&â6VÆbå÷7åöGGuö6÷7E÷F÷&6‚‡7åöVæ6öF–ærÂ–ÖvUöVÖ&VFF–æw2¢&WGW&â6VÆbå÷7åöGGuö6÷7Eö¦‚‡7åöVæ6öF–ærÂ–ÖvUöVÖ&VFF–æw2 ¢FVbö6†V6µ÷F…öfV6–&ÆR‡6VÆbÂ7åöVæ6öF–ærÂ–ÖvU÷7FW2“ ¢Ö–å÷&WV—&VBÒöÖ–å÷&WV—&VE÷7ç2‡7åöVæ6öF–ær¢–bÖ–å÷&WV—&VBÓÒfÆöB‚&–æb"’÷"–ÖvU÷7FW2ÂÖ–å÷&WV—&VC ¢&—6RfÇVTW'&÷"€¢÷7åöæõ÷F…öÖW76vR€¢7åöVæ6öF–ærÀ¢–ÖvU÷7FW2À¢6VÆbæÖ…÷v–æF÷w5÷W%÷7âÀ¢Ö–å÷&WV—&VBÀ¢&V6öãÒ'FöõöfWu÷v–æF÷w2"À¢¢¢Ö…÷÷76–&ÆU÷v–æF÷w2Ò7åöVæ6öF–ærçFW‡EöÆVæwF‚¢6VÆbæÖ…÷v–æF÷w5÷W%÷7à¢–b–ÖvU÷7FW2âÖ…÷÷76–&ÆU÷v–æF÷w3 ¢&—6RfÇVTW'&÷"€¢÷7åöæõ÷F…öÖW76vR€¢7åöVæ6öF–ærÀ¢–ÖvU÷7FW2À¢6VÆbæÖ…÷v–æF÷w5÷W%÷7âÀ¢Ö–å÷&WV—&VBÀ¢&V6öãÒ'FöõöÖç•÷v–æF÷w2"À¢¢¢&WGW&âÖ–å÷&WV—&V@ ¢FVb÷7åöGGuö6÷7E÷F÷&6‚‡6VÆbÂ7åöVæ6öF–ærÂ–ÖvUöVÖ&VFF–æw2“ ¢7åöÆöö·WÒ7åö–æFW…ö'•÷7F'EöæEöÆVæwF‚‡7åöVæ6öF–ær¢FW‡E÷7FW2Ò7åöVæ6öF–ærçFW‡EöÆVæwF€¢–ÖvU÷7FW2Ò–ÖvUöVÖ&VFF–æw2ç6†U³Ð¢FWf–6RÒ–ÖvUöVÖ&VFF–æw2æFWf–6P¢Ö–å÷&WV—&VBÒ6VÆbåö6†V6µ÷F…öfV6–&ÆR‡7åöVæ6öF–ærÂ–ÖvU÷7FW2¢G&ç6—F–öåö6÷7G2Ò÷&V6ö×WFU÷G&ç6—F–öåö6÷7G2€¢7åöVæ6öF–ærÀ¢–ÖvUöVÖ&VFF–æw2À¢6VÆbçFV×W&GW&RÀ¢6VÆbæÖ…÷v–æF÷w5÷W%÷7âÀ¢6VÆbçv–æF÷uö6÷VçE÷VæÇG’À¢¢¦W&òÒF÷&6‚ç¦W&÷2‚‚’ÂFWf–6SÖFWf–6RÂGG—SÖ–ÖvUöVÖ&VFF–æw2æGG—R¢GÒµ´æöæRf÷"ò–â&ævR†–ÖvU÷7FW2²•Òf÷"ò–â&ævR‡FW‡E÷7FW2²•Ð¢G³Õ³ÒÒ¦W&ð ¢f÷"–æFW…÷FW‡B–â&ævR‡FW‡E÷7FW2²“ ¢f÷"–æFW…ö–ÖvR–â&ævR†–ÖvU÷7FW2²“ ¢7W'&VçBÒG¶–æFW…÷FW‡EÕ¶–æFW…ö–ÖvUÐ¢–b7W'&VçB—2æöæS ¢6öçF–çVP¢f÷"7åöÆVæwF‚–â&ævRƒÂFW‡E÷7FW2Ò–æFW…÷FW‡B²“ ¢7åö–G‚Ò7åöÆöö·WævWB‚†–æFW…÷FW‡BÂ7åöÆVæwF‚’¢–b7åö–G‚—2æöæS ¢6öçF–çVP¢f÷"v–æF÷uö6÷VçB–â&ævR€¢À¢Ö–â‡6VÆbæÖ…÷v–æF÷w5÷W%÷7âÂ–ÖvU÷7FW2Ò–æFW…ö–ÖvR’²À¢“ ¢æW‡E÷FW‡BÒ–æFW…÷FW‡B²7åöÆVæwF€¢æW‡Eö–ÖvRÒ–æFW…ö–ÖvR²v–æF÷uö6÷Vç@¢G&ç6—F–öâÒG&ç6—F–öåö6÷7G5·v–æF÷uö6÷VçEÕ·7åö–G‚Â–æFW…ö–ÖvUÐ¢6æF–FFRÒ7W'&VçB²G&ç6—F–öà¢&Wf–÷W2ÒG¶æW‡E÷FW‡EÕ¶æW‡Eö–ÖvUÐ¢–b&Wf–÷W2—2æöæS ¢G¶æW‡E÷FW‡EÕ¶æW‡Eö–ÖvUÒÒ6æF–FFP¢VÇ6S ¢G¶æW‡E÷FW‡EÕ¶æW‡Eö–ÖvUÒÒ6ögFÖ–â€¢F÷&6‚ç7F6²‚‡&Wf–÷W2Â6æF–FFR’’À¢6VÆbævÖÖÀ¢ ¢–bG·FW‡E÷7FW5Õ¶–ÖvU÷7FW5Ò—2æöæS ¢&—6RfÇVTW'&÷"€¢÷7åöæõ÷F…öÖW76vR€¢7åöVæ6öF–ærÀ¢–ÖvU÷7FW2À¢6VÆbæÖ…÷v–æF÷w5÷W%÷7âÀ¢Ö–å÷&WV—&VBÀ¢&V6öãÒ&vVæW&–2"À¢¢¢&WGW&âG·FW‡E÷7FW5Õ¶–ÖvU÷7FW5Ð ¢FVb÷7åöGGuö6÷7Eö¦‚‡6VÆbÂ7åöVæ6öF–ærÂ–ÖvUöVÖ&VFF–æw2“ ¢–bæ÷B6VÆbå÷v&æVEö¦…ö&6¶VæC ¢&–çB€¢%5åôEEuô$4´TäCÖ¦‚6ö×–ÆW2'’FVç6RFVç6÷"6†RâFW‡B ¢&ÆVæwF‡2&R'V6¶WFVBv†Vâ5åôEEuô%T4´UEõDU…EôÄTäuD…3Ó² ¢&¶VWv–æF÷ræB7âvVöÖWG'’f—†VBf÷"6ö×–ÆF–öâ&WW6Râ"À¢fÇW6ƒÕG'VRÀ¢¢6VÆbå÷v&æVEö¦…ö&6¶VæBÒG'VP¢–ÖvU÷7FW2Ò–ÖvUöVÖ&VFF–æw2ç6†U³Ð¢6VÆbåö6†V6µ÷F…öfV6–&ÆR‡7åöVæ6öF–ærÂ–ÖvU÷7FW2¢7GVÅ÷FW‡E÷7FW2Ò–çB‡7åöVæ6öF–ærçFW‡EöÆVæwF‚¢FW‡E÷7FW5÷FFVBÒö'V6¶WEöÆVæwF‚€¢7GVÅ÷FW‡E÷7FW2À¢7åöGGu÷FW‡Eö'V6¶WE÷6—¦RÀ¢7åöGGuöÖ…÷FW‡Eö'V6¶WBÀ¢Væ&ÆVC×7åöGGuö'V6¶WE÷FW‡EöÆVæwF‡2À¢¢FVç6RÒöFVç6U÷G&ç6—F–öåö6÷7G2€¢7åöVæ6öF–ærÀ¢–ÖvUöVÖ&VFF–æw2À¢6VÆbçFV×W&GW&RÀ¢6VÆbæÖ…÷v–æF÷w5÷W%÷7âÀ¢6VÆbçv–æF÷uö6÷VçE÷VæÇG’À¢FW‡E÷7FW5÷FFVC×FW‡E÷7FW5÷FFVBÀ¢¢–bõ5åôEEuôÔTÕôDT%Ts ¢öFV'Vuö¦…öFVç6U÷FVç6÷"€¢7åöVæ6öF–ærÀ¢–ÖvUöVÖ&VFF–æw2À¢FVç6RÀ¢FW‡E÷7FW5÷FFVC×FW‡E÷7FW5÷FFVBÀ¢¢G'“ ¢g&öÒ¦…÷7åöGGr–×÷'B¦…7äEEtgVæ7F–öà¢W†6WB'VçF–ÖTW'&÷# ¢&—6P¢W†6WB–×÷'DW'&÷"2W†3 ¢&—6R'VçF–ÖTW'&÷"€¢%5åôEEuô$4´TäCÖ¦‚&WV—&W2¤‚âW6RF÷&6‚÷"–ç7FÆÂ¤‚â ¢’g&öÒW†0¢æVVG5öw&F–VçBÒF÷&6‚æ—5öw&EöVæ&ÆVB‚’æBFVç6Rç&WV—&W5öw&@¢6÷7BÒ¦…7äEEtgVæ7F–öâæÇ’€¢FVç6RÀ¢7GVÅ÷FW‡E÷7FW2À¢–çB†–ÖvU÷7FW2’À¢6VÆbævÖÖÀ¢æVVG5öw&F–VçBÀ¢¢öFV'Vuö¦…ö6÷7B‡7åöVæ6öF–ærÂ–ÖvUöVÖ&VFF–æw2Â6÷7B¢&WGW&â6÷7@ ¢FVbf÷'v&E÷f&ÆVâ‡6VÆbÂFW‡EöVæ6öFW"Âæ÷&Õö–ÖrÂ÷5÷FW‡G2ÂæVu÷FW‡G2“ ¢Æ÷76W2ÒµÐ¢÷5ö6÷7E÷fÇVW2ÒµÐ¢æVuö6÷7E÷fÇVW2ÒµÐ¢æ÷&Õ÷÷5÷fÇVW2ÒµÐ¢æ÷&ÕöæVu÷fÇVW2ÒµÐ¢÷5÷&ö%÷fÇVW2ÒµÐ¢v÷fÇVW2ÒµÐ ¢f÷"6×ÆUö–G‚Â÷5÷FW‡B–âVçVÖW&FR‡÷5÷FW‡G2“ ¢÷5öVæ6öF–ærÒFW‡EöVæ6öFW"€¢÷5÷FW‡BÂW6Uö66†SÔfÇ6R–bFW‡EöVæ6öFW"çG&–æ–ærVÇ6RæöæP¢¢6×ÆUö–ÖvRÒæ÷&Õö–Öu·6×ÆUö–G…Ð¢÷5ö6÷7BÒ6VÆbå÷7åöGGuö6÷7B‡÷5öVæ6öF–ærÂ6×ÆUö–ÖvR¢æ÷&Õ÷÷2Ò÷5ö6÷7BòÖ‚€¢÷5öVæ6öF–ærçFW‡EöÆVæwF‚Â6×ÆUö–ÖvRç6†U³Ð¢ ¢æVu÷FW‡EöÆ—7BÒÆ—7B†æVu÷FW‡G5·6×ÆUö–G…Ò¢–bæ÷BæVu÷FW‡EöÆ—7C ¢&—6RfÇVTW'&÷"€¢%7ä6öçG&7F—fU6ögDEEr&WV—&W2BÆV7BöæRæVvF—fRFW‡BW"6×ÆRâ ¢ ¢–b6VÆbææVvF—fUöw&EöÖöFRÓÒ&ÆÂ# ¢æVuö6÷7G2ÒµÐ¢æ÷&ÕöæVw2ÒµÐ¢f÷"æVu÷FW‡B–âæVu÷FW‡EöÆ—7C ¢æVuöVæ6öF–ærÒFW‡EöVæ6öFW"†æVu÷FW‡BÂW6Uö66†SÔfÇ6R¢G'“ ¢æVuö6÷7BÒ6VÆbå÷7åöGGuö6÷7B€¢æVuöVæ6öF–ærÂ6×ÆUö–ÖvP¢¢W†6WBfÇVTW'&÷# ¢æVuö6÷7BÒö–æfV6–&ÆUöæVvF—fUö6÷7B€¢6×ÆUö–ÖvRÀ¢æVuöVæ6öF–ærçFW‡EöÆVæwF‚À¢6VÆbçFV×W&GW&RÀ¢¢æVuö6÷7G2æVæB†æVuö6÷7B¢æ÷&ÕöæVw2æVæB€¢æVuö6÷7@¢òÖ‚†æVuöVæ6öF–ærçFW‡EöÆVæwF‚Â6×ÆUö–ÖvRç6†U³Ò¢¢æVuö6÷7G2ÒF÷&6‚ç7F6²†æVuö6÷7G2¢æ÷&ÕöæVw2ÒF÷&6‚ç7F6²†æ÷&ÕöæVw2¢7FG5öæVuö6÷7G2ÒæVuö6÷7G2æFWF6‚‚¢7FG5öæ÷&ÕöæVw2Òæ÷&ÕöæVw2æFWF6‚‚¢VÇ6S ¢66÷&VEöæVuö6÷7G2ÒµÐ¢66÷&VEöæ÷&ÕöæVw2ÒµÐ¢æVuöfV6–&ÆRÒµÐ¢v—F‚F÷&6‚ææõöw&B‚“ ¢f÷"æVu÷FW‡B–âæVu÷FW‡EöÆ—7C ¢æVuöVæ6öF–ærÒFW‡EöVæ6öFW"†æVu÷FW‡BÂW6Uö66†SÔfÇ6R¢G'“ ¢æVuö6÷7BÒ6VÆbå÷7åöGGuö6÷7B€¢æVuöVæ6öF–ærÂ6×ÆUö–ÖvP¢¢fV6–&ÆRÒG'VP¢W†6WBfÇVTW'&÷# ¢æVuö6÷7BÒö–æfV6–&ÆUöæVvF—fUö6÷7B€¢6×ÆUö–ÖvRÀ¢æVuöVæ6öF–ærçFW‡EöÆVæwF‚À¢6VÆbçFV×W&GW&RÀ¢¢fV6–&ÆRÒfÇ6P¢66÷&VEöæVuö6÷7G2æVæB†æVuö6÷7BæFWF6‚‚’¢66÷&VEöæ÷&ÕöæVw2æVæB€¢€¢æVuö6÷7@¢òÖ‚€¢æVuöVæ6öF–ærçFW‡EöÆVæwF‚À¢6×ÆUö–ÖvRç6†U³ÒÀ¢¢’æFWF6‚‚¢¢æVuöfV6–&ÆRæVæB†fV6–&ÆR¢66÷&VEöæVuö6÷7G2ÒF÷&6‚ç7F6²‡66÷&VEöæVuö6÷7G2¢66÷&VEöæ÷&ÕöæVw2ÒF÷&6‚ç7F6²‡66÷&VEöæ÷&ÕöæVw2¢†&Eö–G‚Ò–çB‡F÷&6‚æ&vÖ–â‡66÷&VEöæ÷&ÕöæVw2’æ—FVÒ‚’¢–b6VÆbææVvF—fUöw&EöÖöFRÓÒ&†&FW7B# ¢†&EöVæ6öF–ærÒFW‡EöVæ6öFW"€¢æVu÷FW‡EöÆ—7E¶†&Eö–G…ÒÂW6Uö66†SÔfÇ6P¢¢†&Eö6÷7BÒ€¢6VÆbå÷7åöGGuö6÷7B††&EöVæ6öF–ærÂ6×ÆUö–ÖvR¢–bæVuöfV6–&ÆU¶†&Eö–G…Ð¢VÇ6R66÷&VEöæVuö6÷7G5¶†&Eö–G…Ð¢¢†&Eöæ÷&ÒÒ†&Eö6÷7BòÖ‚€¢†&EöVæ6öF–ærçFW‡EöÆVæwF‚Â6×ÆUö–ÖvRç6†U³Ð¢¢æVuö6÷7G2Ò†&Eö6÷7Bçf–Wrƒ¢æ÷&ÕöæVw2Ò†&Eöæ÷&Òçf–Wrƒ¢VÇ6S ¢æVuö6÷7G2Ò66÷&VEöæVuö6÷7G5¶†&Eö–G…Òçf–Wrƒ¢æ÷&ÕöæVw2Ò66÷&VEöæ÷&ÕöæVw5¶†&Eö–G…Òçf–Wrƒ¢7FG5öæVuö6÷7G2Ò66÷&VEöæVuö6÷7G0¢7FG5öæ÷&ÕöæVw2Ò66÷&VEöæ÷&ÕöæVw0 ¢6×ÆUöÆ÷72ÒF÷&6‚æ6Æ×€¢æ÷&Õ÷÷2Òæ÷&ÕöæVw2²6VÆbæÖ&v–âÂÖ–ãÓã ¢’æÖVâ‚¢Æ÷76W2æVæB‡6×ÆUöÆ÷72¢ÆÅö6÷7G2ÒF÷&6‚æ6B€¢¶æ÷&Õ÷÷2æFWF6‚‚’çf–Wrƒ’Â7FG5öæ÷&ÕöæVw5ÒÂF–ÓÓ ¢¢÷5ö6÷7E÷fÇVW2æVæB‡÷5ö6÷7BæFWF6‚‚’¢æVuö6÷7E÷fÇVW2æVæB‡7FG5öæVuö6÷7G2æÖVâ‚’æFWF6‚‚’¢æ÷&Õ÷÷5÷fÇVW2æVæB†æ÷&Õ÷÷2æFWF6‚‚’¢æ÷&ÕöæVu÷fÇVW2æVæB‡7FG5öæ÷&ÕöæVw2æÖVâ‚’æFWF6‚‚’¢÷5÷&ö%÷fÇVW2æVæB€¢F÷&6‚ç6ögFÖ‚‚ÖÆÅö6÷7G2ÂF–ÓÓ•³ÒæFWF6‚‚¢¢v÷fÇVW2æVæB€¢†æ÷&Õ÷÷2æFWF6‚‚’Ò7FG5öæ÷&ÕöæVw2æÖVâ‚’’æFWF6‚‚¢ ¢Æ÷72ÒF÷&6‚ç7F6²†Æ÷76W2’æÖVâ‚¢&WGW&âÆ÷72Â°¢&6÷7E÷÷2#¢F÷&6‚ç7F6²‡÷5ö6÷7E÷fÇVW2’æÖVâ‚’æ—FVÒ‚’À¢&6÷7EöæVr#¢F÷&6‚ç7F6²†æVuö6÷7E÷fÇVW2’æÖVâ‚’æ—FVÒ‚’À¢'÷5÷&ö"#¢F÷&6‚ç7F6²‡÷5÷&ö%÷fÇVW2’æÖVâ‚’æ—FVÒ‚’À¢&v#¢F÷&6‚ç7F6²†v÷fÇVW2’æÖVâ‚’æ—FVÒ‚’À¢&æ÷&Õ÷÷2#¢F÷&6‚ç7F6²†æ÷&Õ÷÷5÷fÇVW2’æÖVâ‚’æ—FVÒ‚’À¢&æ÷&ÕöæVr#¢F÷&6‚ç7F6²†æ÷&ÕöæVu÷fÇVW2’æÖVâ‚’æ—FVÒ‚’À¢&6öçG&7F—fR#¢Æ÷72æ—FVÒ‚’À¢Ð 