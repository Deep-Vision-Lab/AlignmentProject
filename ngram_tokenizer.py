"""N-gram text-unit vocabulary and greedy tokenizer for D3TW alignment."""

from collections import Counter
import json


DEFAULT_ARABIC_LIGATURES = ["لا", "لل", "ال"]


def collect_ngram_tokens(
    texts,
    min_n=1,
    max_n=3,
    min_freq=2,
    max_vocab_size=5000,
    skip_spaces=True,
    include_ligatures=True,
    ligatures=None,
):
    """Build an n-gram vocabulary from training transcripts.

    Single-character tokens are always kept so greedy tokenization can always
    fall back without producing unknown units.
    """
    min_n = max(1, int(min_n))
    max_n = max(min_n, int(max_n))
    min_freq = max(1, int(min_freq))
    max_vocab_size = int(max_vocab_size)
    ligatures = list(DEFAULT_ARABIC_LIGATURES if ligatures is None else ligatures)

    counts = Counter()
    single_chars = set()
    for text in texts:
        chars = list(text)
        single_chars.update(chars)
        for n in range(2, max_n + 1):
            if len(chars) < n:
                continue
            for start in range(0, len(chars) - n + 1):
                token = "".join(chars[start:start + n])
                if skip_spaces and " " in token:
                    continue
                counts[token] += 1

    for ch in single_chars:
        counts[ch] += 10**12  # force single-character retention and fallback.

    tokens = []
    for token, count in counts.items():
        if len(token) == 1 or count >= min_freq:
            tokens.append(token)

    if include_ligatures:
        for token in ligatures:
            if token and (not skip_spaces or " " not in token):
                counts[token] = max(counts.get(token, 0), min_freq)
                if token not in tokens:
                    tokens.append(token)

    tokens.sort(key=lambda tok: (-len(tok), -counts[tok], tok))

    if max_vocab_size > 0:
        singles = [tok for tok in tokens if len(tok) == 1]
        nonsingles = [tok for tok in tokens if len(tok) > 1]
        remaining = max(0, max_vocab_size - len(singles))
        kept = nonsingles[:remaining] + singles
        kept.sort(key=lambda tok: (-len(tok), -counts[tok], tok))
        tokens = kept

    return tokens


class NGramTokenizer:
    """Greedy longest-match tokenizer over a fixed n-gram vocabulary."""

    def __init__(self, tokens, mode="greedy_longest", unk_token=None):
        if mode != "greedy_longest":
            raise ValueError("Only ngram_tokenizer_mode='greedy_longest' is supported.")
        self.tokens = list(dict.fromkeys(tokens))
        self.token_set = set(self.tokens)
        self.mode = mode
        self.unk_token = unk_token
        self.max_len = max((len(token) for token in self.tokens), default=1)

    def tokenize(self, text):
        """Return token list and character spans in logical text order."""
        units = []
        spans = []
        chars = list(text)
        pos = 0
        while pos < len(chars):
            matched = None
            max_len_here = min(self.max_len, len(chars) - pos)
            for width in range(max_len_here, 0, -1):
                candidate = "".join(chars[pos:pos + width])
                if candidate in self.token_set:
                    matched = candidate
                    break
            if matched is None:
                matched = self.unk_token if self.unk_token is not None else chars[pos]
                width = 1
            units.append(matched)
            spans.append((pos, pos + width))
            pos += width
        return units, spans


def save_ngram_vocab_json(path, tokens):
    payload = {
        "tokens": list(tokens),
        "token_to_idx": {token: idx for idx, token in enumerate(tokens)},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False)
    return path


def load_ngram_vocab_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["tokens"]
