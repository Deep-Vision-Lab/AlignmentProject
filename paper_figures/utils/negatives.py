"""
Hard-negative generation for Arabic transcript-image alignment evaluation.

Negative modes
--------------
mixed            – crop + drop + shuffle + random in-batch (legacy default).
length_controlled – word-shuffle, preserving character count (prevents length
                   bias where shorter negatives get unfairly lower DTW cost).
dot_confusion    – substitutes visually-confusable Arabic letters (ب↔ت↔ث,
                   ج↔ح↔خ, etc.). Hardest negatives for character identity.
same_length_random – random Arabic characters preserving exact char count.
shuffle_only     – word shuffle only (no crop/drop).
"""
import random


# ── DOT CONFUSION dictionary ─────────────────────────────────────────────────
# Maps each Arabic letter to its visually-confusable relatives (same base
# glyph, different dot count / position).
DOT_CONFUSIONS = {
    "ب": ["ت", "ث", "ن", "ي"],
    "ت": ["ب", "ث", "ن", "ي"],
    "ث": ["ب", "ت", "ن", "ي"],
    "ن": ["ب", "ت", "ث", "ي"],
    "ي": ["ب", "ت", "ث", "ن"],
    "ج": ["ح", "خ"],
    "ح": ["ج", "خ"],
    "خ": ["ج", "ح"],
    "د": ["ذ"],
    "ذ": ["د"],
    "ر": ["ز"],
    "ز": ["ر"],
    "س": ["ش"],
    "ش": ["س"],
    "ص": ["ض"],
    "ض": ["ص"],
    "ط": ["ظ"],
    "ظ": ["ط"],
    "ع": ["غ"],
    "غ": ["ع"],
    "ف": ["ق"],
    "ق": ["ف"],
}

# Random Arabic letters used for same_length_random substitution
_ARABIC_LETTERS = list("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")


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


_HARD_NEG_FNS   = [_crop, _drop, _shuffle]
_HARD_NEG_NAMES  = ["cropped", "dropped", "shuffled"]


# ── Length-controlled negatives ───────────────────────────────────────────────

def _word_shuffle(text):
    """Shuffle all words — preserves word lengths and total char count."""
    words = text.split()
    if len(words) < 2:
        return text
    random.shuffle(words)
    return " ".join(words)


def _char_shuffle_within_words(text):
    """Shuffle characters inside each word — preserves space structure."""
    parts = []
    for word in text.split(" "):
        if len(word) > 1:
            chars = list(word)
            random.shuffle(chars)
            parts.append("".join(chars))
        else:
            parts.append(word)
    return " ".join(parts)


def _same_length_random(text):
    """Replace every non-space character with a random Arabic letter."""
    out = []
    for ch in text:
        if ch == " ":
            out.append(" ")
        else:
            out.append(random.choice(_ARABIC_LETTERS))
    return "".join(out)


_LENGTH_CTRL_FNS   = [_word_shuffle, _char_shuffle_within_words, _same_length_random]
_LENGTH_CTRL_NAMES  = ["word_shuffled", "char_shuffled", "same_length_random"]


# ── Dot-confusion negatives ───────────────────────────────────────────────────

def make_dot_confusion_negative(text, p=0.25):
    """
    Replace each Arabic letter that has a confusable relative with probability p.
    Always replaces at least one letter so the result is guaranteed different.
    """
    chars = list(text)
    confused_any = False
    for i, ch in enumerate(chars):
        if ch in DOT_CONFUSIONS and random.random() < p:
            chars[i] = random.choice(DOT_CONFUSIONS[ch])
            confused_any = True

    if not confused_any:
        # Force at least one substitution
        confusable = [(i, c) for i, c in enumerate(chars) if c in DOT_CONFUSIONS]
        if confusable:
            idx, ch = random.choice(confusable)
            chars[idx] = random.choice(DOT_CONFUSIONS[ch])
    return "".join(chars)


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

def generate_hard_negatives(text, all_texts=None, k=10,
                             negative_mode="mixed"):
    """
    Generate k (neg_text, neg_type) pairs for positive *text*.

    Args:
        text:          Positive transcript string.
        all_texts:     Optional pool of other transcripts for random negatives.
        k:             Total number of negatives to return.
        negative_mode: 'mixed' | 'length_controlled' | 'dot_confusion' |
                       'same_length_random' | 'shuffle_only'.

    Returns:
        List of (neg_text: str, neg_type: str) tuples, length == k.
    """
    results = []
    mode = (negative_mode or "mixed").lower()

    if mode == "mixed":
        # Legacy: crop + drop + shuffle (up to one each), then random pool.
        num_hard = min(len(_HARD_NEG_FNS), k)
        for i in range(num_hard):
            neg  = _ensure_different(_HARD_NEG_FNS[i](text), text)
            results.append((neg, _HARD_NEG_NAMES[i]))

    elif mode == "length_controlled":
        # Word-shuffle, char-shuffle, same-length random (no crop/drop).
        num_lc = min(len(_LENGTH_CTRL_FNS), k)
        for i in range(num_lc):
            neg = _ensure_different(_LENGTH_CTRL_FNS[i](text), text)
            results.append((neg, _LENGTH_CTRL_NAMES[i]))

    elif mode == "dot_confusion":
        # Fill ALL k slots with dot-confusion negatives.
        for _ in range(k):
            neg = _ensure_different(make_dot_confusion_negative(text), text)
            results.append((neg, "dot_confusion"))
        return results[:k]

    elif mode == "same_length_random":
        for _ in range(k):
            neg = _ensure_different(_same_length_random(text), text)
            results.append((neg, "same_length_random"))
        return results[:k]

    elif mode == "shuffle_only":
        for _ in range(k):
            neg = _ensure_different(_word_shuffle(text), text)
            results.append((neg, "word_shuffled"))
        return results[:k]

    else:
        raise ValueError(f"Unknown negative_mode {mode!r}.")

    # Fill remaining slots with random pool samples
    num_random = k - len(results)
    pool = [t for t in (all_texts or []) if t.strip() != text.strip()]
    for _ in range(num_random):
        if pool:
            neg = _ensure_different(random.choice(pool), text)
        else:
            neg = _ensure_different(_word_shuffle(text), text)
        results.append((neg, "random"))

    return results[:k]
