from Parameters import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet34, ResNet34_Weights
import time

import gc


# ============================================================================
# GPU TIMER FOR IMAGE EMBEDDING PROFILING
# ============================================================================
class ImageEmbedTimer:
    """
    GPU-aware timer for profiling image embedding phases.
    Uses CUDA events for accurate GPU timing.
    """
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.use_cuda = torch.cuda.is_available()
        self.timings = {}
        self.starts = {}

    def start(self, name):
        if not self.enabled:
            return
        if self.use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.starts[name] = start_event
        else:
            self.starts[name] = time.time()

    def stop(self, name):
        if not self.enabled or name not in self.starts:
            return
        if self.use_cuda:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()
            duration_ms = self.starts[name].elapsed_time(end_event)
        else:
            duration_ms = (time.time() - self.starts[name]) * 1000

        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration_ms)
        del self.starts[name]

    def get_summary(self):
        """Return timing summary as dict"""
        summary = {}
        for name, durations in self.timings.items():
            summary[name] = {
                'avg_ms': sum(durations) / len(durations) if durations else 0,
                'total_ms': sum(durations),
                'count': len(durations)
            }
        return summary

    def reset(self):
        self.timings = {}
        self.starts = {}


# Global timer for image embedding (set enabled=True to profile)
img_embed_timer = ImageEmbedTimer(enabled=False)


# ============================================================================
# BI-LSTM SEQUENCE CONTEXT ENCODER
# ============================================================================

class BiLSTMEncoder(nn.Module):
    """
    Bi-directional LSTM sequence encoder for CRNN architecture.

    Applies BiLSTM over CNN patch features to capture local sequential context,
    then projects back to embed_dim with residual connections and layer normalization.
    """
    def __init__(self, embed_dim, hidden_dim=None, lstm_layers=1, dropout=0.1):
        super(BiLSTMEncoder, self).__init__()

        self.embed_dim = embed_dim

        if hidden_dim is None:
            hidden_dim = embed_dim // 2  # BiLSTM concatenates forward+backward

        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )

        lstm_output_dim = hidden_dim * 2  # bidirectional
        self.lstm_proj = nn.Linear(lstm_output_dim, embed_dim)
        self.lstm_norm = nn.LayerNorm(embed_dim)
        self.lstm_dropout = nn.Dropout(dropout)

        self.final_norm = nn.LayerNorm(embed_dim)
        self.final_dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B, seq_len, embed_dim] - CNN patch embeddings
        Returns:
            output: [B, seq_len, embed_dim]
        """
        input_residual = x

        self.bilstm.flatten_parameters()
        lstm_out, _ = self.bilstm(x)  # [B, seq_len, hidden_dim * 2]

        lstm_out = self.lstm_proj(lstm_out)    # [B, seq_len, embed_dim]
        lstm_out = self.lstm_dropout(lstm_out)
        lstm_out = self.lstm_norm(lstm_out + x)  # skip connection after BiLSTM

        output = self.final_norm(lstm_out)
        output = self.final_dropout(output)
        output = output + input_residual         # final skip connection

        return output


# Sliding window function to divide image into patches (subwindows)
def sliding_window(image, window_size, stride, debug_mode=False, save_dir=False):
    patches = []
    # Unfolding the image into patches of size window_size
    for i  in range(image.shape[0]):
        image_windows = []
        for j in range(0, image.shape[3] - window_size + 1, stride):
            image_i = image[i, :, :, j:j + window_size]
            image_windows.append(image_i)
            if debug_mode:
                torchvision.utils.save_image(image_i, f"{save_dir}/patch_b{i}_w{j}.png")
        patches.append(torch.stack(image_windows, dim=0))
        del image_windows
    result = torch.stack(patches, dim=0)
    del patches
    return result

# CNN model for extracting features from sliding window patches
def calculate_conv_output_size(input_size, kernel_size, stride, padding):
    return (input_size - kernel_size + 2 * padding) // stride + 1



class EmbeddingModel(nn.Module):
    def __init__(self, window_size=128, stride=64, vector_size=512,
                  device='cuda', use_flip=False, use_bilstm=True,
                  bilstm_layers=1, dropout=0.1):
        """
        Image embedding model: ResNet34 CNN + Bi-LSTM sequence encoder (CRNN).

        Args:
            window_size: Size of sliding window for patch extraction
            stride: Stride for sliding window
            vector_size: Dimension of output feature vectors
            device: Device to run on ('cuda' or 'cpu')
            use_flip: Flip patches left-to-right before CNN
            use_bilstm: Enable BiLSTM for sequence context
            bilstm_layers: Number of BiLSTM layers
            dropout: Dropout rate
        """
        super(EmbeddingModel, self).__init__()

        self.device = device
        self.use_bilstm = use_bilstm
        self.use_flip = use_flip

        # CNN encoder (ResNet34) for patch feature extraction
        self.cnn_encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        cnn_feature_dim = self.cnn_encoder.fc.in_features
        self.cnn_encoder.fc = nn.Linear(cnn_feature_dim, vector_size)
        self.cnn_encoder = self.cnn_encoder.to(device)

        # Sequence context encoder: Bi-LSTM or None
        if use_bilstm:
            self.sequence_encoder = BiLSTMEncoder(
                embed_dim=vector_size,
                hidden_dim=vector_size // 2,
                lstm_layers=bilstm_layers,
                dropout=dropout,
            ).to(device)
            print(f"[EmbeddingModel] Using BiLSTM encoder (layers={bilstm_layers})")
        else:
            self.sequence_encoder = None
            print("[EmbeddingModel] No sequence encoder (CNN features only)")

        self.window_size = window_size
        self.stride = stride
        self.vector_size = vector_size
        self._flip_verified = False



    # Maximum number of patches to push through the CNN in one shot.
    # Keeps peak GPU allocation bounded when using multi-scale windows.
    CNN_CHUNK_SIZE = 512

    def _process_cnn_branch(self, tokens_a, show_dims=False):
        """Process CNN branch (chunked to limit peak GPU memory)."""
        batches_num, windows_num, Channels, H, W = tokens_a.shape

        total_patches = batches_num * windows_num

        # Reshape patches
        reshaped_tokens_a = tokens_a.reshape(total_patches, Channels, H, W)
        if show_dims:
            print(f"Patches after reshaping: {reshaped_tokens_a.shape}")

        del tokens_a

        # --- Chunked CNN forward to cap peak memory ---
        if total_patches <= self.CNN_CHUNK_SIZE:
            encoded_tokens_a = self.cnn_encoder(reshaped_tokens_a)
        else:
            chunks = reshaped_tokens_a.split(self.CNN_CHUNK_SIZE, dim=0)
            del reshaped_tokens_a          # free the big tensor early
            encoded_chunks = []
            for chunk in chunks:
                encoded_chunks.append(self.cnn_encoder(chunk))
            encoded_tokens_a = torch.cat(encoded_chunks, dim=0)
            del encoded_chunks, chunks
            reshaped_tokens_a = None       # already deleted above

        if show_dims:
            print(f"Tokens after CNN: {encoded_tokens_a.shape}")

        if reshaped_tokens_a is not None:
            del reshaped_tokens_a

        return encoded_tokens_a, batches_num, windows_num

    def forward(self, image_a, show_dims=False, debug=False, timer=None):
        t = timer if timer is not None else img_embed_timer

        # ==================== PHASE: Sliding Window (Patch Extraction) ====================
        t.start('img_1a_sliding_window')
        tokens_a = sliding_window(image_a, self.window_size, self.stride, debug_mode=debug).to(device)
        t.stop('img_1a_sliding_window')

        if show_dims:
            print(f"Patches: {tokens_a.shape}")

        batches_num, windows_num = tokens_a.shape[:2]

        if self.use_flip:
            tokens_a = torch.flip(tokens_a, dims=[1])

        # One-shot sanity check: save patches[0] (rightmost patch after flip) and
        # patches[-1] (leftmost patch after flip) so we can verify the flip maps
        # patch index 0 to the first Arabic character (rightmost in the image).
        if self.use_flip and not self._flip_verified:
            import os, torchvision
            os.makedirs("debug_flip", exist_ok=True)
            # patches[0] after flip = was the RIGHTMOST patch (first Arabic char)
            torchvision.utils.save_image(tokens_a[0, 0].float().cpu(), "debug_flip/patch0_after_flip.png")
            # patches[-1] after flip = was the LEFTMOST patch (last Arabic char)
            torchvision.utils.save_image(tokens_a[0, -1].float().cpu(), "debug_flip/patchLast_after_flip.png")
            # Also save the full first image for reference
            torchvision.utils.save_image(image_a[0].float().cpu(), "debug_flip/full_image_ref.png")
            print(f"[FlipCheck] Saved debug_flip/patch0_after_flip.png  ← should show the FIRST Arabic character (rightmost in image)")
            print(f"[FlipCheck] Saved debug_flip/patchLast_after_flip.png ← should show the LAST Arabic character (leftmost in image)")
            print(f"[FlipCheck] Saved debug_flip/full_image_ref.png        ← original image for comparison")
            print(f"[FlipCheck] tokens_a shape after flip: {tokens_a.shape}  (B, num_patches, C, H, W)")
            self._flip_verified = True

        # ==================== PHASE: CNN Encoding ====================
        t.start('img_1c_cnn_encoding')
        encoded_tokens_a, batches_num, windows_num = self._process_cnn_branch(
            tokens_a, show_dims
        )
        t.stop('img_1c_cnn_encoding')

        features_vector_a = encoded_tokens_a.view(batches_num, windows_num, self.vector_size)
        del encoded_tokens_a

        # ==================== PHASE: Sequence Context (BiLSTM) ====================
        t.start('img_1f_sequence_encoder')
        if self.sequence_encoder is not None:
            features_vector_a = self.sequence_encoder(features_vector_a)
        t.stop('img_1f_sequence_encoder')

        return features_vector_a

    def forward_at_scale(self, image_a, scale_window_size, scale_stride,
                         show_dims=False, debug=False, timer=None):
        """
        Forward pass at an arbitrary window/stride scale.

        Re-uses the SAME CNN and sequence encoder weights —
        only the sliding-window granularity changes, producing a different
        sequence length for each scale.

        Args:
            image_a:           [B, C, H, W] input images
            scale_window_size: patch width for this scale
            scale_stride:      stride for this scale
        Returns:
            features_vector_a: [B, seq_len_scale, vec_size]
        """
        t = timer if timer is not None else img_embed_timer

        # ---- Sliding window at the requested scale ----
        t.start(f'img_scale_{scale_window_size}_sw')
        tokens_a = sliding_window(
            image_a, scale_window_size, scale_stride, debug_mode=debug
        ).to(self.device)
        t.stop(f'img_scale_{scale_window_size}_sw')

        batches_num, windows_num = tokens_a.shape[:2]

        if self.use_flip:
            tokens_a = torch.flip(tokens_a, dims=[1])

        # ---- CNN encoding (shared weights) ----
        t.start(f'img_scale_{scale_window_size}_cnn')
        encoded_tokens_a, batches_num, windows_num = self._process_cnn_branch(
            tokens_a, show_dims
        )
        t.stop(f'img_scale_{scale_window_size}_cnn')

        features_vector_a = encoded_tokens_a.view(
            batches_num, windows_num, self.vector_size
        )
        del encoded_tokens_a

        # ---- Sequence encoder (BiLSTM) ----
        t.start(f'img_scale_{scale_window_size}_seq_enc')
        if self.sequence_encoder is not None:
            features_vector_a = self.sequence_encoder(features_vector_a)
        t.stop(f'img_scale_{scale_window_size}_seq_enc')

        return features_vector_a
