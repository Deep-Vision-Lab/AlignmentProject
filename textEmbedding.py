import torch
import torch.nn as nn
from Parameters import device


class TextEmbedding(nn.Module):
    """
    Text embedding model that converts each character to a vector 
    based solely on the character value, ignoring position in the word.
    """
    def __init__(self, embedding_dim, vocab_size=65536):
        """
        Args:
            embedding_dim: Dimension of the character embedding vectors
            vocab_size: Maximum number of unique characters (default supports Unicode)
        """
        super(TextEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

    def char_to_index(self, char):
        """Convert a character to its embedding index based on Unicode value."""
        return ord(char) % self.vocab_size

    def text_to_indices(self, text):
        """Convert a string to a tensor of character indices."""
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
            
            # Pad sequences to same length
            padded = nn.utils.rnn.pad_sequence(batch_indices, batch_first=True, padding_value=0)
            return self.embedding(padded.to(device))
        
        else:
            # Already a tensor of indices
            return self.embedding(text)
        


if __name__ == "__main__":
    # Example usage
    model = TextEmbedding(embedding_dim=128)
    sample_text = "Hello, World!"
    embedded = model(sample_text)
    print(f"Embedded shape for single string: {embedded.shape}")

    batch_text = ["Hello", "World", "PyTorch"]
    embedded_batch = model(batch_text)
    print(f"Embedded shape for batch of strings: {embedded_batch.shape}")