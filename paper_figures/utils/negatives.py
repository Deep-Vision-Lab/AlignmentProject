"""
Hard-negative generation mirroring newDataLoader.py logic.

Negative types
--------------
crop     – keep first ~half the words
drop     – remove one random word
shuffle  – swap two random words
random   – sample from a separate text list (if provided)
"""
import random


# ── Individual transformations ────────────────────────────────────────────────

def _crop(text):
    words = text.split()
    if len(words) < 2:
        return text
    return " ".join(words[: max(1, len(words) // 2)])


def _drop(text):
    words = text.split()
    if len(words) < 2:
        return text
    idx = random.randint(0, len(words) - 1)
    return " ".join(words[:idx] + words[idx + 1:])


def _shuffle(text):
    words = text.split()
    if len(words) < 2:
        return text
    i, j = random.sample(range(len(words)), 2)
    words[i], words[j] = words[j], words[i]
    return " ".join(words)


_HARD_NEG_FNS  = [_crop, _drop, _shuffle]
_HARD_NEG_NAMES = ["cropped", "dropped", "shuffled"]


# ── Guarantee uniqueness ──────────────────────────────────────────────────────

def _ensure_different(neg, pos):
    if neg.strip() != pos.strip():
        return neg
    chars = list(pos.strip())
    if len(chars) > 1:
        random.shuffle(chars)
        candidate = "".join(chars)
        if candidate.strip() != pos.strip():
            return candidate
    if len(pos.strip()) > 1:
        return pos.strip()[:-1]
    return pos + "‌"   # zero-width non-joiner as last resort


# ── Public API ────────────────────────────────────────────────────────────────

def generate_hard_negatives(text, all_texts=None, k=10):
    """
    Generate k hard-negative (neg_text, neg_type) pairs for a positive *text*.

    Args:
        text:      Positive transcript string.
        all_texts: Optional list of other transcripts for random negatives.
        k:         Total number of negatives to return.

    Returns:
        List of (neg_text: str, neg_type: str) tuples, length == k.
    """
    results = []

    # 1. Structured hard negatives (up to one per type)
    num_hard = min(len(_HARD_NEG_FNS), k)
    for i in range(num_hard):
        fn   = _HARD_NEG_FNS[i]
        name = _HARD_NEG_NAMES[i]
        neg  = _ensure_different(fn(text), text)
        results.append((neg, name))

    # 2. Fill remaining with random in-pool negatives
    num_random = k - len(results)
    pool = [t for t in (all_texts or []) if t.strip() != text.strip()]

    for _ in range(num_random):
        if pool:
            neg = random.choice(pool)
        else:
            neg = _ensure_different(_shuffle(text), text)
        results.append((neg, "random"))

    return results[:k]
