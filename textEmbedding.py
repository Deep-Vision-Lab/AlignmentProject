import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from Parameters import device


class TextEmbedding(nn.Module):
    """
    Text embedding model that converts each character to a vector 
    based solely on the character value, ignoring position in the word.
    
    Includes special handling for SPACE character which maps to black patches
    in the image alignment task.
    """
    # Special token indices (reserved at the start of vocab)
    SPACE_TOKEN_IDX = 0  # Space character - maps to black patches in images
    PAD_TOKEN_IDX = 1    # Padding token for batching
    
    def __init__(self, embedding_dim, vocab_size=65536):
        """
        Args:
            embedding_dim: Dimension of the character embedding vectors
            vocab_size: Maximum number of unique characters (default supports Unicode)
            include_spaces: If True, spaces in text are preserved and embedded
                           If False, spaces are stripped from text (legacy behavior)
        """
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Initialize the SPACE token embedding to a distinct learned vector
        # This will be matched against black patches in images
        with torch.no_grad():
            # Initialize space embedding to small random values so it has a direction
            # that can be learned and normalized (zeros would cause NaN after L2 norm)
            self.embedding.weight[self.SPACE_TOKEN_IDX].normal_(0, 0.02)
            # Initialize padding to zeros
            self.embedding.weight[self.PAD_TOKEN_IDX] = torch.zeros(embedding_dim)

        # Text embedding is frozen during training (not in the optimizer).
        # Why: training treats letters as context-free anchors; we want stable
        # text targets while the image branch (CNN+BiLSTM) learns to match them.
        # Freezing also drops the per-step gradient buffer for the full
        # vocab_size x embedding_dim table — a real VRAM win.
        self.embedding.weight.requires_grad_(False)
        
        # Layer normalization for similarity computation
        self.text_norm = nn.LayerNorm(embedding_dim).to(device)
    
    def get_space_embedding(self):
        """
        Get the current space token embedding vector.
        This is the vector that black image patches should match.
        
        Returns:
            torch.Tensor: Space embedding vector [embedding_dim]
        """
        return self.embedding.weight[self.SPACE_TOKEN_IDX]

    def char_to_index(self, char):
        """
        Convert a character to its embedding index.
        Spaces map to SPACE_TOKEN_IDX, other chars use Unicode value.
        """
        if char == ' ':
            return self.SPACE_TOKEN_IDX
        # Offset by 2 to reserve indices 0,1 for special tokens
        return (ord(char) % (self.vocab_size - 2)) + 2

    def text_to_indices(self, text):
        """
        Convert a string to a tensor of character indices.
        Spaces are preserved and mapped to SPACE_TOKEN_IDX.
        """
        indices = [self.char_to_index(char) for char in text]
        
        return torch.tensor(indices, dtype=torch.long, device=device)

    def forward(self, text):
        """
        Convert text to character embeddings.
        
        Args:
            text: A string or list of strings
            
        Returns:
            Tensor of shape (seq_len, embedding_dim) for single string
            or (batch_size, max_seq_len, embedding_dim) for batch
        """
        if isinstance(text, str):
            # Single string: convert each character to embedding
            indices = self.text_to_indices(text).to(device)
            emb = self.embedding(indices)
        
        elif isinstance(text, (list, tuple)):
            # Batch of strings: pad to max length
            batch_indices = [self.text_to_indices(t) for t in text]
            
            if not batch_indices:
                return torch.empty(0, 0, self.embedding_dim, device=device)
            
            # Pad sequences to same length using PAD token
            padded = nn.utils.rnn.pad_sequence(
                batch_indices, 
                batch_first=True, 
                padding_value=self.PAD_TOKEN_IDX
            )
            emb = self.embedding(padded.to(device))
        
        else:
            # Already a tensor of indices
            emb = self.embedding(text)
        
        # Apply LayerNorm before computing similarity
        return self.text_norm(emb)


# ============================================================================
# FastText pretrained character embedding (frozen)
# ============================================================================
class FastTextCharEmbedding(nn.Module):
    """
    Pretrained character embedding backed by facebook/fasttext-ar-vectors.

    fastText publishes word vectors with internal character n-gram subwords,
    which means it returns a vector for any string — including single
    characters — derived from those subwords. Here we use those single-char
    vectors as a frozen lookup table.

    The fastText vectors are 300-d; if `embedding_dim` differs, a (default
    trainable) Linear projection is added. The projection is NOT fastText
    fine-tuning; it's a downstream adapter so the rest of the network stays
    at `vector_size`. Set `projection_trainable=False` for a truly frozen
    text branch.

    Mirrors the public API of `TextEmbedding` so it's a drop-in replacement:
    same `SPACE_TOKEN_IDX`, `PAD_TOKEN_IDX`, `char_to_index`, `text_to_indices`,
    and forward() accepting str, list[str], or LongTensor.
    """

    SPACE_TOKEN_IDX = 0
    PAD_TOKEN_IDX = 1

    # HuggingFace repo for the pretrained model
    HF_REPO_ID = "facebook/fasttext-ar-vectors"
    HF_FILENAME = "model.bin"

    # Eagerly-precomputed unicode ranges (everything we'll plausibly see in
    # Arabic + ASCII text). Any char outside these ranges still works — it's
    # filled lazily on first encounter inside `text_to_indices`.
    PRECOMPUTE_RANGES = (
        (0x0020, 0x007F),   # printable ASCII
        (0x00A0, 0x00FF),   # Latin-1 supplement
        (0x0600, 0x06FF),   # Arabic
        (0x0750, 0x077F),   # Arabic supplement
        (0xFB50, 0xFDFF),   # Arabic presentation forms-A
        (0xFE70, 0xFEFF),   # Arabic presentation forms-B
    )

    def __init__(self, embedding_dim, vocab_size=65536,
                 model_path=None, projection_trainable=True,
                 cache_dir=None):
        super(FastTextCharEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        ft_model = self._load_fasttext(model_path=model_path, cache_dir=cache_dir)
        self._ft_dim = int(ft_model.get_dimension())
        self._ft_model = ft_model
        # Don't register `_ft_model` as a module attribute Pytorch tracks — it
        # isn't a Tensor/Module. Plain attribute is fine.

        # Frozen lookup table indexed by char_to_index().
        self.embedding = nn.Embedding(vocab_size, self._ft_dim).to(device)
        with torch.no_grad():
            self.embedding.weight.zero_()
            # Space → fastText vector for " " (could be near-zero in some
            # ft models; fall back to a small random direction to keep L2
            # normalisation safe).
            space_vec = self._char_vec_from_ft(" ")
            if torch.linalg.vector_norm(space_vec) < 1e-6:
                space_vec = torch.empty(self._ft_dim, device=device).normal_(0, 0.02)
            self.embedding.weight[self.SPACE_TOKEN_IDX] = space_vec
            self.embedding.weight[self.PAD_TOKEN_IDX].zero_()

            # Precompute over common ranges so the embedding works without
            # waiting for the first batch's chars to be seen.
            for lo, hi in self.PRECOMPUTE_RANGES:
                for cp in range(lo, hi + 1):
                    ch = chr(cp)
                    if ch == " ":
                        continue  # already filled above
                    idx = self._unicode_index(ch)
                    if idx == self.PAD_TOKEN_IDX:
                        continue
                    self.embedding.weight[idx] = self._char_vec_from_ft(ch)

        # The lookup table itself is frozen — this is the "no fine-tune"
        # guarantee the caller asked for.
        self.embedding.weight.requires_grad_(False)

        # Track which indices have been populated so unknown chars can be
        # filled lazily inside text_to_indices().
        self._populated = torch.zeros(vocab_size, dtype=torch.bool, device='cpu')
        self._populated[self.SPACE_TOKEN_IDX] = True
        self._populated[self.PAD_TOKEN_IDX] = True
        for lo, hi in self.PRECOMPUTE_RANGES:
            for cp in range(lo, hi + 1):
                self._populated[self._unicode_index(chr(cp))] = True

        # Output adapter: only present when fastText dim != requested dim.
        # Identity if dims match. Marked frozen if `projection_trainable=False`.
        if self._ft_dim == embedding_dim:
            self.projection = None
        else:
            self.projection = nn.Linear(self._ft_dim, embedding_dim, bias=False).to(device)
            if not projection_trainable:
                self.projection.weight.requires_grad_(False)

        # Layer normalization for similarity computation
        self.text_norm = nn.LayerNorm(embedding_dim).to(device)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _load_fasttext(model_path=None, cache_dir=None):
        try:
            import fasttext
        except ImportError as e:
            raise ImportError(
                "FastTextCharEmbedding requires the `fasttext` package. "
                "Install with `pip install fasttext huggingface_hub`."
            ) from e

        if model_path is None or not os.path.isfile(model_path):
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:
                raise ImportError(
                    "FastTextCharEmbedding needs `huggingface_hub` to download "
                    "facebook/fasttext-ar-vectors. Install with "
                    "`pip install huggingface_hub`, or set "
                    "`text_embedder_model_path` in Parameters.py to a local "
                    "model.bin."
                ) from e
            model_path = hf_hub_download(
                repo_id=FastTextCharEmbedding.HF_REPO_ID,
                filename=FastTextCharEmbedding.HF_FILENAME,
                cache_dir=cache_dir,
            )

        # fasttext.load_model prints a noisy stderr warning on newer versions;
        # we can't suppress it cleanly without redirecting fd 2, so we leave it.
        return fasttext.load_model(model_path)

    def _char_vec_from_ft(self, ch):
        vec = self._ft_model.get_word_vector(ch)
        return torch.from_numpy(vec).to(device=device, dtype=torch.float32)

    def _unicode_index(self, ch):
        if ch == " ":
            return self.SPACE_TOKEN_IDX
        return (ord(ch) % (self.vocab_size - 2)) + 2

    # --------------------------------------------------------- public API
    def get_space_embedding(self):
        vec = self.embedding.weight[self.SPACE_TOKEN_IDX]
        if self.projection is not None:
            vec = self.projection(vec)
        return vec

    def char_to_index(self, char):
        return self._unicode_index(char)

    def _ensure_populated(self, ch):
        """Lazily fill a fastText vector for chars outside the precomputed ranges."""
        idx = self._unicode_index(ch)
        if self._populated[idx].item():
            return
        with torch.no_grad():
            self.embedding.weight[idx] = self._char_vec_from_ft(ch)
        self._populated[idx] = True

    def text_to_indices(self, text):
        indices = []
        for ch in text:
            self._ensure_populated(ch)
            indices.append(self._unicode_index(ch))
        return torch.tensor(indices, dtype=torch.long, device=device)

    def forward(self, text):
        if isinstance(text, str):
            idx = self.text_to_indices(text).to(device)
            emb = self.embedding(idx)
        elif isinstance(text, (list, tuple)):
            batch_indices = [self.text_to_indices(t) for t in text]
            if not batch_indices:
                return torch.empty(0, 0, self.embedding_dim, device=device)
            padded = nn.utils.rnn.pad_sequence(
                batch_indices, batch_first=True, padding_value=self.PAD_TOKEN_IDX
            )
            emb = self.embedding(padded.to(device))
        else:
            emb = self.embedding(text)

        if self.projection is not None:
            emb = self.projection(emb)
        
        # Apply LayerNorm before computing similarity
        return self.text_norm(emb)


# ============================================================================
# Orthogonal frozen character embedding
# ============================================================================
class OrthogonalCharEmbedding(nn.Module):
    """
    Fully frozen character embedding with deterministic approximately-orthogonal
    vectors. Each Unicode codepoint gets a unit-length vector drawn from a fixed
    random seed (no training, no fine-tuning). The space and pad tokens are
    treated identically to TextEmbedding.

    Rationale: FastText vectors are linguistic. For a visual-alignment baseline
    we want a text target space that carries zero linguistic information —
    every character just gets a distinct, stable direction in R^D. If the image
    branch learns to match these directions it must be encoding visual identity,
    not linguistic co-occurrence.

    API matches TextEmbedding / FastTextCharEmbedding.
    """

    SPACE_TOKEN_IDX = 0
    PAD_TOKEN_IDX   = 1

    def __init__(self, embedding_dim, vocab_size=65536, seed=1234):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size    = vocab_size

        # Deterministic random unit vectors — same every run.
        g      = torch.Generator().manual_seed(seed)
        weight = torch.randn(vocab_size, embedding_dim, generator=g)
        weight = torch.nn.functional.normalize(weight, p=2, dim=-1)
        weight[self.PAD_TOKEN_IDX].zero_()          # pad → zero vector
        # Space: keep its unit-length random direction (not zero).

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(weight, requires_grad=False)

        self.text_norm = nn.LayerNorm(embedding_dim)
        for p in self.text_norm.parameters():
            p.requires_grad_(False)

    def char_to_index(self, char):
        if char == " ":
            return self.SPACE_TOKEN_IDX
        return (ord(char) % (self.vocab_size - 2)) + 2

    def text_to_indices(self, text):
        indices = [self.char_to_index(c) for c in text]
        return torch.tensor(indices, dtype=torch.long, device=device)

    def get_space_embedding(self):
        return self.embedding.weight[self.SPACE_TOKEN_IDX]

    def forward(self, text):
        if isinstance(text, str):
            idx = self.text_to_indices(text).to(device)
            emb = self.embedding(idx)
        elif isinstance(text, (list, tuple)):
            batch_indices = [self.text_to_indices(t) for t in text]
            if not batch_indices:
                return torch.empty(0, 0, self.embedding_dim, device=device)
            padded = nn.utils.rnn.pad_sequence(
                batch_indices, batch_first=True,
                padding_value=self.PAD_TOKEN_IDX,
            )
            emb = self.embedding(padded.to(device))
        else:
            emb = self.embedding(text)
        return self.text_norm(emb)


# ============================================================================
# Random frozen character embedding (non-normalised baseline)
# ============================================================================
class RandomFrozenCharEmbedding(nn.Module):
    """
    Like OrthogonalCharEmbedding but vectors are NOT L2-normalised at init.
    Used as an ablation: tests whether the unit-sphere geometry of
    OrthogonalCharEmbedding matters, vs. raw Gaussian random directions.
    """

    SPACE_TOKEN_IDX = 0
    PAD_TOKEN_IDX   = 1

    def __init__(self, embedding_dim, vocab_size=65536, seed=5678):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size    = vocab_size

        g      = torch.Generator().manual_seed(seed)
        weight = torch.randn(vocab_size, embedding_dim, generator=g) * 0.1
        weight[self.PAD_TOKEN_IDX].zero_()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(weight, requires_grad=False)

        self.text_norm = nn.LayerNorm(embedding_dim)
        for p in self.text_norm.parameters():
            p.requires_grad_(False)

    def char_to_index(self, char):
        if char == " ":
            return self.SPACE_TOKEN_IDX
        return (ord(char) % (self.vocab_size - 2)) + 2

    def text_to_indices(self, text):
        indices = [self.char_to_index(c) for c in text]
        return torch.tensor(indices, dtype=torch.long, device=device)

    def get_space_embedding(self):
        return self.embedding.weight[self.SPACE_TOKEN_IDX]

    def forward(self, text):
        if isinstance(text, str):
            idx = self.text_to_indices(text).to(device)
            emb = self.embedding(idx)
        elif isinstance(text, (list, tuple)):
            batch_indices = [self.text_to_indices(t) for t in text]
            if not batch_indices:
                return torch.empty(0, 0, self.embedding_dim, device=device)
            padded = nn.utils.rnn.pad_sequence(
                batch_indices, batch_first=True,
                padding_value=self.PAD_TOKEN_IDX,
            )
            emb = self.embedding(padded.to(device))
        else:
            emb = self.embedding(text)
        return self.text_norm(emb)


# ============================================================================
# Rendered glyph prototype embedding (experimental visual baseline)
# ============================================================================
class GlyphPrototypeEmbedding(nn.Module):
    """
    Frozen deterministic embedding from rendered glyph images.

    This experimental Arabic visual-alignment embedder renders each character or
    n-gram token as a grayscale glyph image, flattens it, and applies a fixed
    random projection. It carries visual shape information but no learned
    linguistic semantics.
    """

    SPACE_TOKEN_IDX = 0
    PAD_TOKEN_IDX = 1
    _warned_font = False
    _warned_reshaper = False

    def __init__(
        self,
        embedding_dim,
        vocab_size=65536,
        font_path=None,
        image_height=32,
        image_width=96,
        seed=2468,
    ):
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.vocab_size = int(vocab_size)
        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self._cache = {}

        generator = torch.Generator().manual_seed(int(seed))
        projection = torch.randn(
            self.image_height * self.image_width,
            self.embedding_dim,
            generator=generator,
        )
        projection = F.normalize(projection, p=2, dim=0)
        self.register_buffer("projection", projection)
        self.text_norm = nn.LayerNorm(self.embedding_dim, elementwise_affine=False)
        self.fallback = OrthogonalCharEmbedding(
            embedding_dim=self.embedding_dim,
            vocab_size=self.vocab_size,
        )
        self.font = self._load_font(font_path)

    def _load_font(self, font_path):
        candidates = []
        if font_path:
            candidates.append(font_path)
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        ])
        font_size = max(10, int(self.image_height * 0.75))
        for candidate in candidates:
            if candidate and os.path.isfile(candidate):
                try:
                    return ImageFont.truetype(candidate, font_size)
                except Exception:
                    continue
        if not GlyphPrototypeEmbedding._warned_font:
            print(
                "[GlyphPrototypeEmbedding] WARNING: no usable glyph font found; "
                "falling back to PIL default font."
            )
            GlyphPrototypeEmbedding._warned_font = True
        return ImageFont.load_default()

    def _shape_text(self, token):
        try:
            import arabic_reshaper
            from bidi.algorithm import get_display

            return get_display(arabic_reshaper.reshape(token))
        except Exception:
            if not GlyphPrototypeEmbedding._warned_reshaper:
                print(
                    "[GlyphPrototypeEmbedding] WARNING: arabic_reshaper/python-bidi "
                    "not available; rendering raw text."
                )
                GlyphPrototypeEmbedding._warned_reshaper = True
            return token

    def _render_token_image(self, token):
        canvas = Image.new("L", (self.image_width, self.image_height), color=255)
        if token:
            draw = ImageDraw.Draw(canvas)
            shaped = self._shape_text(token)
            try:
                bbox = draw.textbbox((0, 0), shaped, font=self.font)
            except Exception:
                bbox = self.font.getbbox(shaped)
            text_w = max(1, bbox[2] - bbox[0])
            text_h = max(1, bbox[3] - bbox[1])
            x = max(0, (self.image_width - text_w) // 2 - bbox[0])
            y = max(0, (self.image_height - text_h) // 2 - bbox[1])
            draw.text((x, y), shaped, fill=0, font=self.font)
        values = torch.tensor(list(canvas.getdata()), dtype=torch.float32)
        return (255.0 - values) / 255.0

    def char_to_index(self, char):
        if char == " ":
            return self.SPACE_TOKEN_IDX
        return (ord(char) % (self.vocab_size - 2)) + 2

    def text_to_indices(self, text):
        indices = [self.char_to_index(c) for c in text]
        return torch.tensor(indices, dtype=torch.long, device=device)

    def embed_token(self, token):
        token = str(token)
        if token not in self._cache:
            glyph = self._render_token_image(token).to(self.projection.device)
            vec = glyph @ self.projection
            vec = self.text_norm(vec)
            vec = F.normalize(vec, p=2, dim=-1)
            self._cache[token] = vec.detach()
        return self._cache[token].to(self.projection.device)

    def get_space_embedding(self):
        return self.embed_token(" ")

    def forward(self, text):
        if isinstance(text, str):
            return torch.stack([self.embed_token(ch) for ch in text], dim=0)
        if isinstance(text, (list, tuple)):
            sequences = [self.forward(item) for item in text]
            if not sequences:
                return torch.empty(0, 0, self.embedding_dim, device=self.projection.device)
            return nn.utils.rnn.pad_sequence(
                sequences,
                batch_first=True,
                padding_value=0.0,
            )
        if torch.is_tensor(text):
            return self.fallback(text.to(device)).to(self.projection.device)
        raise TypeError(
            "GlyphPrototypeEmbedding.forward supports str, list/tuple[str], or tensor indices."
        )


# ============================================================================
# Shape-group Arabic character embedding
# ============================================================================
# Letters that share a base glyph shape and differ only by dot count/position.
_ARABIC_SHAPE_GROUPS = [
    "بتثني",   # ba/ta/tha/nun/ya family
    "جحخ",     # jim/ha/kha
    "دذ",      # dal/thal
    "رز",      # ra/zay
    "سش",      # sin/shin
    "صض",      # sad/dad
    "طظ",      # ta/dha
    "عغ",      # ain/ghain
    "فق",      # fa/qaf
]

def _build_shape_group_table(vocab_size, embedding_dim, seed=9012):
    """Return weight tensor with shape-aware vectors for Arabic letters."""
    g      = torch.Generator().manual_seed(seed)
    weight = torch.randn(vocab_size, embedding_dim, generator=g)
    weight = torch.nn.functional.normalize(weight, p=2, dim=-1)
    weight[1].zero_()  # PAD

    # Within each group, blend all members toward their centroid so
    # visually similar letters have similar (not identical) embeddings.
    char_to_idx = lambda c: (ord(c) % (vocab_size - 2)) + 2
    for group in _ARABIC_SHAPE_GROUPS:
        indices = [char_to_idx(c) for c in group]
        group_vecs = weight[indices]                        # [G, D]
        centroid   = torch.nn.functional.normalize(
            group_vecs.mean(0, keepdim=True), p=2, dim=-1  # [1, D]
        )
        # 50% blend: individual direction + group direction
        blended = torch.nn.functional.normalize(
            0.5 * group_vecs + 0.5 * centroid, p=2, dim=-1
        )
        weight[indices] = blended
    return weight


class ShapeGroupCharEmbedding(nn.Module):
    """
    Shape-aware Arabic character embedding. Letters that share a base glyph
    (e.g. ب ت ث) receive embeddings that are blended toward a shared group
    centroid so that visually-confusable pairs are closer in embedding space.

    This provides a softer target than pure orthogonal embeddings and may help
    the model learn coarser glyph structure before fine-grained dot distinction.
    """

    SPACE_TOKEN_IDX = 0
    PAD_TOKEN_IDX   = 1

    def __init__(self, embedding_dim, vocab_size=65536, seed=9012):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.vocab_size    = vocab_size

        weight = _build_shape_group_table(vocab_size, embedding_dim, seed)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(weight, requires_grad=False)

        self.text_norm = nn.LayerNorm(embedding_dim)
        for p in self.text_norm.parameters():
            p.requires_grad_(False)

    def char_to_index(self, char):
        if char == " ":
            return self.SPACE_TOKEN_IDX
        return (ord(char) % (self.vocab_size - 2)) + 2

    def text_to_indices(self, text):
        indices = [self.char_to_index(c) for c in text]
        return torch.tensor(indices, dtype=torch.long, device=device)

    def get_space_embedding(self):
        return self.embedding.weight[self.SPACE_TOKEN_IDX]

    def forward(self, text):
        if isinstance(text, str):
            idx = self.text_to_indices(text).to(device)
            emb = self.embedding(idx)
        elif isinstance(text, (list, tuple)):
            batch_indices = [self.text_to_indices(t) for t in text]
            if not batch_indices:
                return torch.empty(0, 0, self.embedding_dim, device=device)
            padded = nn.utils.rnn.pad_sequence(
                batch_indices, batch_first=True,
                padding_value=self.PAD_TOKEN_IDX,
            )
            emb = self.embedding(padded.to(device))
        else:
            emb = self.embedding(text)
        return self.text_norm(emb)


# ============================================================================
# Factory
# ============================================================================
def build_text_embedder(kind=None, embedding_dim=None, **kwargs):
    """
    Construct the text embedder selected in Parameters.py.

    Supported kinds:
        'char'               — alias for orthogonal_char, the recommended
                               deterministic visual-alignment baseline.
        'fasttext'           — facebook/fasttext-ar-vectors (frozen, linguistic).
        'orthogonal_char'    — deterministic unit-sphere random vectors (frozen).
                               Recommended default for visual alignment: no
                               linguistic bias and stable across runs.
        'glyph_prototype'    — deterministic rendered glyph image projection
                               for experimental Arabic visual alignment.
        'random_frozen_char' — raw Gaussian random frozen vectors (ablation).
        'shape_group_char'   — shape-aware Arabic embeddings (visually-confusable
                               letters blended toward shared centroid).

    Falls back to Parameters.text_embedder_type / vector_size if args are None.
    """
    if kind is None or embedding_dim is None:
        # Local import avoids a circular-import risk at module load time.
        from Parameters import text_embedder_type as _default_kind
        from Parameters import vector_size as _default_dim
        if kind is None:
            kind = _default_kind
        if embedding_dim is None:
            embedding_dim = _default_dim

    kind = kind.lower()
    if kind in ("char", "default", "learned"):
        model = OrthogonalCharEmbedding(embedding_dim=embedding_dim, **kwargs)
    elif kind in ("fasttext", "ft", "fasttext-ar"):
        from Parameters import (
            text_embedder_model_path,
            fasttext_projection_trainable,
            text_embedder_cache_dir,
        )
        model = FastTextCharEmbedding(
            embedding_dim=embedding_dim,
            model_path=text_embedder_model_path,
            projection_trainable=fasttext_projection_trainable,
            cache_dir=text_embedder_cache_dir,
            **kwargs,
        )
    elif kind in ("orthogonal_char", "orthogonal"):
        model = OrthogonalCharEmbedding(embedding_dim=embedding_dim, **kwargs)
    elif kind in ("glyph_prototype", "glyph", "rendered_glyph"):
        from Parameters import (
            glyph_font_path,
            glyph_image_height,
            glyph_image_width,
            glyph_projection_seed,
        )
        model = GlyphPrototypeEmbedding(
            embedding_dim=embedding_dim,
            font_path=glyph_font_path,
            image_height=glyph_image_height,
            image_width=glyph_image_width,
            seed=glyph_projection_seed,
            **kwargs,
        )
    elif kind in ("random_frozen_char", "random_frozen"):
        model = RandomFrozenCharEmbedding(embedding_dim=embedding_dim, **kwargs)
    elif kind in ("shape_group_char", "shape_group"):
        model = ShapeGroupCharEmbedding(embedding_dim=embedding_dim, **kwargs)
    else:
        raise ValueError(
            f"Unknown text_embedder_type {kind!r}. "
            f"Expected 'char', 'fasttext', 'orthogonal_char', 'glyph_prototype', "
            f"'random_frozen_char', or 'shape_group_char'."
        )
    return model


if __name__ == "__main__":
    # Example usage
    print("Testing TextEmbedding with space handling...")
    
    # Test with spaces included (new behavior)
    model = TextEmbedding(embedding_dim=128)
    sample_text = "Hello World"
    embedded = model(sample_text)
    print(f"Embedded shape for '{sample_text}' (with space): {embedded.shape}")
    print(f"  -> Should be (11, 128) - 11 chars including space")
    
    # Verify space token
    space_idx = sample_text.index(' ')
    space_embedding = embedded[space_idx]
    print(f"Space embedding at index {space_idx}: norm = {space_embedding.norm().item():.4f}")
    
    # Test without spaces (legacy behavior)
    model_no_spaces = TextEmbedding(embedding_dim=128)
    embedded_no_spaces = model_no_spaces(sample_text)
    print(f"Embedded shape for '{sample_text}' (no space): {embedded_no_spaces.shape}")
    print(f"  -> Should be (10, 128) - 10 chars without space")

    # Batch test
    batch_text = ["Hello World", "Test", "PyTorch ML"]
    embedded_batch = model(batch_text)
    print(f"Embedded shape for batch of strings: {embedded_batch.shape}")
    print(f"  -> Should be (3, 11, 128) - padded to longest string")
