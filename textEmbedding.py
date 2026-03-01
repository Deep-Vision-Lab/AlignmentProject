import torch
import torch.nn as nn
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
            return self.embedding(indices)
        
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
            return self.embedding(padded.to(device))
        
        else:
            # Already a tensor of indices
            return self.embedding(text)
        


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