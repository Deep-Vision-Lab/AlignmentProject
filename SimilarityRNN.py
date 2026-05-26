import torch
import torch.nn as nn
import torch.nn.functional as F

from Parameters import device


class SimilarityRNN(nn.Module):
    """
    RNN-based model that computes similarity matrix between image tokens and text embeddings.
    
    Takes:
        - text_embeddings: [B, seq_len_text, embed_dim]
        - image_tokens: [B, seq_len_img, embed_dim]
    
    Returns:
        - similarity_matrix: [B, seq_len_text, seq_len_img]
    """
    
    def __init__(self, embed_dim, hidden_dim=128, num_layers=2, bidirectional=True, dropout=0.1):
        super(SimilarityRNN, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # RNN encoders for text and image sequences
        self.text_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.image_rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output dimension after RNN (hidden_dim * num_directions)
        rnn_output_dim = hidden_dim * self.num_directions
        
        # Projection layers to create query/key representations
        self.text_proj = nn.Linear(rnn_output_dim, hidden_dim)
        self.image_proj = nn.Linear(rnn_output_dim, hidden_dim)
        
        # Optional: Learnable similarity computation
        # Instead of just dot product, use a small MLP
        self.similarity_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Temperature parameter for scaling similarity
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Use dot-product similarity (simpler) or MLP-based (more expressive)
        self.use_mlp_similarity = False  # Set to True for MLP-based similarity
        
    def forward(self, text_embeddings, image_tokens, use_dot_product=True):
        """
        Args:
            text_embeddings: [B, seq_len_text, embed_dim]
            image_tokens: [B, seq_len_img, embed_dim]
            use_dot_product: If True, use scaled dot-product similarity.
                             If False, use MLP-based pairwise similarity.
        
        Returns:
            similarity_matrix: [B, seq_len_text, seq_len_img]
        """
        B = text_embeddings.size(0)
        seq_len_text = text_embeddings.size(1)
        seq_len_img = image_tokens.size(1)
        
        # Encode sequences with RNNs
        text_encoded, _ = self.text_rnn(text_embeddings)  # [B, seq_len_text, hidden*dirs]
        image_encoded, _ = self.image_rnn(image_tokens)   # [B, seq_len_img, hidden*dirs]
        
        # Project to common space
        text_proj = self.text_proj(text_encoded)   # [B, seq_len_text, hidden_dim]
        image_proj = self.image_proj(image_encoded) # [B, seq_len_img, hidden_dim]
        
        if use_dot_product or not self.use_mlp_similarity:
            # Scaled dot-product similarity (like attention)
            # Normalize for cosine similarity
            text_norm = F.normalize(text_proj, p=2, dim=-1)
            image_norm = F.normalize(image_proj, p=2, dim=-1)
            
            # Compute similarity: [B, seq_len_text, seq_len_img]
            similarity_matrix = torch.bmm(text_norm, image_norm.transpose(1, 2))
            similarity_matrix = similarity_matrix * self.temperature
        else:
            # MLP-based pairwise similarity (more expressive but slower)
            # Expand and concatenate for pairwise comparison
            # text_proj: [B, seq_len_text, 1, hidden_dim]
            # image_proj: [B, 1, seq_len_img, hidden_dim]
            text_expanded = text_proj.unsqueeze(2).expand(-1, -1, seq_len_img, -1)
            image_expanded = image_proj.unsqueeze(1).expand(-1, seq_len_text, -1, -1)
            
            # Concatenate: [B, seq_len_text, seq_len_img, hidden_dim * 2]
            combined = torch.cat([text_expanded, image_expanded], dim=-1)
            
            # Compute similarity scores: [B, seq_len_text, seq_len_img, 1] -> [B, seq_len_text, seq_len_img]
            similarity_matrix = self.similarity_mlp(combined).squeeze(-1)
        
        return similarity_matrix


class BilinearSimilarityRNN(nn.Module):
    """
    Alternative RNN model using bilinear similarity computation.
    """
    
    def __init__(self, embed_dim, hidden_dim=128, num_layers=2, bidirectional=True, dropout=0.1):
        super(BilinearSimilarityRNN, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
        
        # RNN encoders
        self.text_rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.image_rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )
        
        rnn_output_dim = hidden_dim * self.num_directions
        
        # Bilinear layer for similarity computation
        self.bilinear = nn.Bilinear(rnn_output_dim, rnn_output_dim, 1, bias=True)
        
        # Or use a learnable weight matrix W: sim(t, i) = t^T W i
        self.W = nn.Parameter(torch.randn(rnn_output_dim, rnn_output_dim) * 0.01)
        
    def forward(self, text_embeddings, image_tokens, use_bilinear=False):
        """
        Args:
            text_embeddings: [B, seq_len_text, embed_dim]
            image_tokens: [B, seq_len_img, embed_dim]
        
        Returns:
            similarity_matrix: [B, seq_len_text, seq_len_img]
        """
        B = text_embeddings.size(0)
        seq_len_text = text_embeddings.size(1)
        seq_len_img = image_tokens.size(1)
        
        # Encode sequences
        text_encoded, _ = self.text_rnn(text_embeddings)  # [B, seq_len_text, hidden*dirs]
        image_encoded, _ = self.image_rnn(image_tokens)   # [B, seq_len_img, hidden*dirs]
        
        if use_bilinear:
            # Bilinear similarity (pairwise, slower)
            similarity_matrix = torch.zeros(B, seq_len_text, seq_len_img, device=text_embeddings.device)
            for i in range(seq_len_text):
                for j in range(seq_len_img):
                    similarity_matrix[:, i, j] = self.bilinear(
                        text_encoded[:, i, :], 
                        image_encoded[:, j, :]
                    ).squeeze(-1)
        else:
            # Use weight matrix W: sim = text @ W @ image.T
            # [B, seq_len_text, hidden] @ [hidden, hidden] = [B, seq_len_text, hidden]
            text_transformed = torch.matmul(text_encoded, self.W)
            # Normalize for cosine similarity
            text_norm = F.normalize(text_transformed, p=2, dim=-1)
            image_norm = F.normalize(image_encoded, p=2, dim=-1)
            # [B, seq_len_text, hidden] @ [B, hidden, seq_len_img] = [B, seq_len_text, seq_len_img]
            similarity_matrix = torch.bmm(text_norm, image_norm.transpose(1, 2))
        
        return similarity_matrix
