"""Frozen deterministic character directions; approximately, not exactly, orthogonal.

Unicode index mapping and seeded Gaussian codebook match the former character
encoder. Fixed LayerNorm followed by L2 reproduces the directions used by DTW.
Space is index 0, padding index 1; neither is an unknown-character token.
"""
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F


ARABIC_LETTERS = 'ءآأؤإئابتثجحخدذرزسشصضطظعغفقكلمنهويىة'


def clean_letters(text):
    """NFKC, Arabic Unicode letters only; omit tatweel, spaces, marks and digits."""
    return [c for c in unicodedata.normalize('NFKC', str(text))
            if c != 'ـ' and unicodedata.category(c).startswith('L')
            and any(lo <= ord(c) <= hi for lo, hi in
                    ((0x600,0x6ff),(0x750,0x77f),(0x8a0,0x8ff),(0xfb50,0xfdff),(0xfe70,0xfeff)))]


class OrthogonalCharEmbedding(nn.Module):
    SPACE_TOKEN_IDX = 0
    PAD_TOKEN_IDX = 1

    def __init__(self, embedding_dim=128, vocab_size=65536, seed=1234):
        super().__init__()
        if vocab_size < 2304 or embedding_dim < 1:
            raise ValueError('Require positive dimension and vocabulary covering normalized Arabic')
        self.embedding_dim, self.vocab_size, self.seed = embedding_dim, vocab_size, seed
        generator = torch.Generator().manual_seed(seed)
        weights = F.normalize(torch.randn(vocab_size, embedding_dim, generator=generator), dim=-1)
        weights[self.PAD_TOKEN_IDX].zero_()
        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True, padding_idx=self.PAD_TOKEN_IDX)

    def char_to_index(self, char):
        return self.SPACE_TOKEN_IDX if char == ' ' else ord(char) % (self.vocab_size - 2) + 2

    def encode(self, text):
        indices = torch.tensor([self.char_to_index(c) for c in text], dtype=torch.long,
                               device=self.embedding.weight.device)
        return self(indices)

    def encode_batch(self, texts):
        indices = torch.full((len(texts), max(map(len, texts), default=0)), self.PAD_TOKEN_IDX,
                             dtype=torch.long, device=self.embedding.weight.device)
        for row, text in enumerate(texts):
            indices[row, :len(text)] = torch.tensor([self.char_to_index(c) for c in text],
                                                   device=indices.device, dtype=torch.long)
        return self(indices), indices != self.PAD_TOKEN_IDX

    def forward(self, value):
        if isinstance(value, str):
            return self.encode(value)
        vectors = self.embedding(value)
        return F.normalize(F.layer_norm(vectors, (self.embedding_dim,)), dim=-1)
