import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from Parameters import device


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer.
    """
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class StrokeContextEncoder(nn.Module):
    """
    Encodes image strokes with context from previous strokes using self-attention.
    This helps the model understand the sequence of strokes and their relationships.
    """
    def __init__(self, embed_dim, num_heads=4, num_layers=2, dropout=0.1):
        super(StrokeContextEncoder, self).__init__()
        
        self.embed_dim = embed_dim
        
        # Positional encoding for stroke sequence
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        
        # Transformer encoder for stroke context
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, strokes, causal_mask=True):
        """
        Args:
            strokes: [B, num_strokes, embed_dim] - sequence of stroke embeddings
            causal_mask: If True, each stroke can only attend to previous strokes
        
        Returns:
            contextualized_strokes: [B, num_strokes, embed_dim]
        """
        B, seq_len, _ = strokes.shape
        
        # Add positional encoding
        strokes = self.pos_encoding(strokes)
        
        # Create causal mask if needed (each stroke can see itself and previous strokes)
        if causal_mask:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=strokes.device), diagonal=1).bool()
        else:
            mask = None
        
        # Apply transformer encoder
        contextualized = self.transformer_encoder(strokes, mask=mask)
        contextualized = self.norm(contextualized)
        
        return contextualized


class SimilarityTransformer(nn.Module):
    """
    Transformer-based model that computes similarity matrix between image strokes and text embeddings.
    
    The model takes image strokes (tokens) and uses transformer attention to:
    1. Encode strokes with context from previous strokes
    2. Cross-attend between text embeddings and contextualized strokes
    3. Predict which letter corresponds to each stroke
    
    Takes:
        - text_embeddings: [B, seq_len_text, embed_dim] - letter/character embeddings
        - image_tokens: [B, seq_len_img, embed_dim] - stroke embeddings from image
    
    Returns:
        - similarity_matrix: [B, seq_len_text, seq_len_img] - similarity scores for alignment
    
    Label-Aware Design:
        When context_free_text=True, text embeddings are NOT passed through self-attention.
        This ensures that the SAME letter (e.g., 'Alif') always produces the SAME embedding
        regardless of its position in the text. This is crucial for handling repeated letters:
        - If text is "A B A", both 'A' positions will have identical embeddings
        - The Soft-DTW loss can then correctly align image patches to ALL matching positions
        - Prevents the model from being confused by "negative" examples that are actually the same letter
    """
    
    def __init__(self, embed_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1,
                 context_free_text=True):
        """
        Args:
            embed_dim: Input embedding dimension
            hidden_dim: Hidden dimension for transformer
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            context_free_text: If True (default), text embeddings are NOT contextualized.
                              This ensures same letter = same embedding (Label-Aware design).
                              If False, text goes through self-attention (legacy behavior).
        """
        super(SimilarityTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_free_text = context_free_text
        
        # Input projections to hidden dimension
        self.text_input_proj = nn.Linear(embed_dim, hidden_dim)
        self.image_input_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Positional encodings
        # For context-free text, we still add positional encoding for position awareness
        # but without mixing embeddings between positions
        self.text_pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        self.image_pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Stroke context encoder - encodes each stroke with context from previous strokes
        # (Image NEEDS context to understand stroke sequences)
        self.stroke_context_encoder = StrokeContextEncoder(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Text encoder - self-attention over text sequence
        # Only used when context_free_text=False
        if not context_free_text:
            text_encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            self.text_encoder = nn.TransformerEncoder(text_encoder_layer, num_layers=num_layers)
        else:
            # For context-free text, use a simple MLP instead of self-attention
            # This transforms the embedding but keeps each position independent
            self.text_encoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
        
        # Cross-attention: text queries, image keys/values
        # This helps predict which letter each stroke belongs to
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projections for similarity computation
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.image_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Learnable temperature for scaling similarity
        self.temperature = nn.Parameter(torch.ones(1) * 0.07)
        
        # Layer norms
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, text_embeddings, image_tokens, use_cross_attention=True):
        """
        Args:
            text_embeddings: [B, seq_len_text, embed_dim] - character/letter embeddings
            image_tokens: [B, seq_len_img, embed_dim] - stroke embeddings from image patches
            use_cross_attention: If True, use cross-attention for richer representations
                                 NOTE: When context_free_text=True, cross-attention only flows
                                 from image->text, not text->image (to preserve label-aware property)
        
        Returns:
            similarity_matrix: [B, seq_len_text, seq_len_img]
        """
        B = text_embeddings.size(0)
        seq_len_text = text_embeddings.size(1)
        seq_len_img = image_tokens.size(1)
        
        # Project inputs to hidden dimension
        text_hidden = self.text_input_proj(text_embeddings)  # [B, seq_len_text, hidden_dim]
        image_hidden = self.image_input_proj(image_tokens)   # [B, seq_len_img, hidden_dim]
        
        # Add positional encodings
        # For context-free text: positional encoding adds position info but doesn't mix embeddings
        text_hidden = self.text_pos_encoding(text_hidden)
        image_hidden = self.image_pos_encoding(image_hidden)
        
        # Encode text
        if self.context_free_text:
            # Context-free: simple MLP transformation, each position is independent
            # Same letter at different positions will have same embedding + different position encoding
            text_encoded = self.text_encoder(text_hidden)  # [B, seq_len_text, hidden_dim]
        else:
            # Legacy: self-attention over text (mixes context between positions)
            text_encoded = self.text_encoder(text_hidden)  # [B, seq_len_text, hidden_dim]
        
        # Encode image strokes with context from previous strokes (causal attention)
        # This allows each stroke to "see" the strokes that came before it
        # (Image NEEDS context to distinguish strokes and understand the writing sequence)
        image_encoded = self.stroke_context_encoder(image_hidden, causal_mask=True)  # [B, seq_len_img, hidden_dim]
        
        if use_cross_attention:
            # Cross-attention: let image strokes attend to text to predict which letter
            # Query: image strokes, Key/Value: text characters
            image_cross, _ = self.cross_attention(
                query=image_encoded,
                key=text_encoded,
                value=text_encoded
            )
            # Residual connection
            image_encoded = image_encoded + image_cross
            
            if not self.context_free_text:
                # Only apply text->image cross-attention when NOT using context-free text
                # (otherwise we would re-introduce context mixing through the image branch)
                text_cross, _ = self.cross_attention(
                    query=text_encoded,
                    key=image_encoded,
                    value=image_encoded
                )
                text_encoded = text_encoded + text_cross
            # When context_free_text=True, we skip text->image cross-attention
            # This preserves the "label-aware" property: same letter = same text embedding
        
        # Apply layer norms
        text_encoded = self.text_norm(text_encoded)
        image_encoded = self.image_norm(image_encoded)
        
        # Project to final representations
        text_final = self.text_proj(text_encoded)   # [B, seq_len_text, hidden_dim]
        image_final = self.image_proj(image_encoded) # [B, seq_len_img, hidden_dim]
        
        # Normalize for cosine similarity
        text_norm = F.normalize(text_final, p=2, dim=-1)
        image_norm = F.normalize(image_final, p=2, dim=-1)
        
        # Compute similarity matrix: [B, seq_len_text, seq_len_img]
        similarity_matrix = torch.bmm(text_norm, image_norm.transpose(1, 2))
        
        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature.clamp(min=0.01)
        
        return similarity_matrix


class StrokeLetterPredictor(nn.Module):
    """
    A transformer model that predicts which letter each stroke belongs to.
    Uses the previous strokes as context to make predictions.
    
    This can be used as an auxiliary task or for explicit letter prediction.
    """
    
    def __init__(self, embed_dim, num_classes, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super(StrokeLetterPredictor, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Input projection
        self.input_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer decoder - each stroke can attend to previous strokes
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, strokes, memory=None):
        """
        Args:
            strokes: [B, num_strokes, embed_dim] - sequence of stroke embeddings
            memory: [B, seq_len, hidden_dim] - optional memory from encoder (e.g., text)
        
        Returns:
            logits: [B, num_strokes, num_classes] - letter predictions for each stroke
        """
        B, seq_len, _ = strokes.shape
        
        # Project and add positional encoding
        x = self.input_proj(strokes)
        x = self.pos_encoding(x)
        
        # Create causal mask (each stroke can only see previous strokes)
        tgt_mask = torch.triu(torch.ones(seq_len, seq_len, device=strokes.device), diagonal=1).bool()
        
        if memory is not None:
            # Use transformer decoder with cross-attention to memory
            x = self.transformer_decoder(x, memory, tgt_mask=tgt_mask)
        else:
            # Self-attention only (treat as encoder with causal mask)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=4,
                dim_feedforward=self.hidden_dim * 4,
                batch_first=True
            ).to(strokes.device)
            x = encoder_layer(x, src_mask=tgt_mask)
        
        # Predict letter for each stroke
        logits = self.classifier(x)
        
        return logits


class BilinearSimilarityTransformer(nn.Module):
    """
    Alternative transformer model using bilinear similarity computation.
    Combines transformer encoding with learned bilinear similarity.
    """
    
    def __init__(self, embed_dim, hidden_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super(BilinearSimilarityTransformer, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # Input projections
        self.text_proj = nn.Linear(embed_dim, hidden_dim)
        self.image_proj = nn.Linear(embed_dim, hidden_dim)
        
        # Positional encodings
        self.text_pos = PositionalEncoding(hidden_dim, dropout=dropout)
        self.image_pos = PositionalEncoding(hidden_dim, dropout=dropout)
        
        # Transformer encoders
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.image_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=num_layers
        )
        
        # Learnable bilinear weight matrix: sim(t, i) = t^T W i
        self.W = nn.Parameter(torch.randn(hidden_dim, hidden_dim) * 0.02)
        
        # Layer norms
        self.text_norm = nn.LayerNorm(hidden_dim)
        self.image_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, text_embeddings, image_tokens):
        """
        Args:
            text_embeddings: [B, seq_len_text, embed_dim]
            image_tokens: [B, seq_len_img, embed_dim]
        
        Returns:
            similarity_matrix: [B, seq_len_text, seq_len_img]
        """
        # Project and add positional encoding
        text_hidden = self.text_proj(text_embeddings)
        text_hidden = self.text_pos(text_hidden)
        
        image_hidden = self.image_proj(image_tokens)
        image_hidden = self.image_pos(image_hidden)
        
        # Encode with transformers
        text_encoded = self.text_encoder(text_hidden)
        
        # Causal mask for image strokes
        seq_len_img = image_tokens.size(1)
        causal_mask = torch.triu(torch.ones(seq_len_img, seq_len_img, device=image_tokens.device), diagonal=1).bool()
        image_encoded = self.image_encoder(image_hidden, mask=causal_mask)
        
        # Layer norm
        text_encoded = self.text_norm(text_encoded)
        image_encoded = self.image_norm(image_encoded)
        
        # Bilinear similarity: text @ W @ image.T
        text_transformed = torch.matmul(text_encoded, self.W)  # [B, seq_len_text, hidden_dim]
        # Normalize for cosine similarity
        text_norm = F.normalize(text_transformed, p=2, dim=-1)
        image_norm = F.normalize(image_encoded, p=2, dim=-1)
        
        similarity_matrix = torch.bmm(text_norm, image_norm.transpose(1, 2))  # [B, seq_len_text, seq_len_img]
        
        return similarity_matrix
