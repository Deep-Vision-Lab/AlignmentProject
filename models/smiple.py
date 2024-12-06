import torch
import torch.nn as nn


class TextlinePatchSimilarityTransformer(nn.Module):
    def __init__(self, patch_width=10, d_model=512, nhead=8, num_layers=6):
        super(TextlinePatchSimilarityTransformer, self).__init__()

        self.patch_width = patch_width
        self.flatten_patches = nn.Flatten(start_dim=2)

        # Linear embedding layer for patches
        self.patch_embedding = nn.Linear(50 * patch_width, d_model)

        # Transformer Encoders for each image sequence
        self.encoderA = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )

        self.encoderB = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )

        # Cross-Attention Layer
        self.cross_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead)

    def extract_patches(self, images):
        # Extract patches from the images (Assuming images of shape [batch_size, channels, height=50, width=512])
        batch_size, channels, height, width = images.shape
        assert height == 50, "Height of the input images must be 50."

        # Divide the width into patches of 50x10
        patches = images.unfold(3, self.patch_width,
                                self.patch_width)  # [batch_size, channels, height=50, num_patches=51, patch_width=10]
        patches = patches.permute(0, 3, 1, 2,
                                  4).contiguous()  # [batch_size, num_patches=51, channels, height=50, patch_width=10]
        patches = patches.view(batch_size, patches.shape[1],
                               -1)  # Flatten each patch into a vector [batch_size, num_patches, 50*patch_width]
        return patches

    def forward(self, imgA, imgB):
        # Extract and embed patches for both images
        patchesA = self.extract_patches(imgA)  # [batch_size, num_patches=51, 50*patch_width]
        patchesB = self.extract_patches(imgB)  # [batch_size, num_patches=51, 50*patch_width]

        embedA = self.patch_embedding(patchesA)  # [batch_size, num_patches=51, d_model]
        embedB = self.patch_embedding(patchesB)  # [batch_size, num_patches=51, d_model]

        # Permute to match transformer input shape
        embedA = embedA.permute(1, 0, 2)  # [num_patches=51, batch_size, d_model]
        embedB = embedB.permute(1, 0, 2)  # [num_patches=51, batch_size, d_model]

        # Encode patches
        encA = self.encoderA(embedA)  # [num_patches=51, batch_size, d_model]
        encB = self.encoderB(embedB)  # [num_patches=51, batch_size, d_model]

        # Cross-attention to find similarity
        attn_output, attn_weights = self.cross_attention(encA, encB, encB)

        # attn_weights is the similarity matrix [batch_size, num_patches_A=51, num_patches_B=51]
        return attn_weights


# Example usage:
batch_size = 32
channels = 1  # Assuming grayscale images
height, width = 50, 512
imgA = torch.randn(batch_size, channels, height, width)  # Randomly generated textline images
imgB = torch.randn(batch_size, channels, height, width)

model = TextlinePatchSimilarityTransformer()
similarity_matrix = model(imgA, imgB)  # Output shape: [batch_size, num_patches_A=51, num_patches_B=51]