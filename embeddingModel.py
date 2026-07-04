from Parameters import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import resnet34, ResNet34_Weights
import time
import math

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


# ============================================================================
# TRANSFORMER SEQUENCE CONTEXT ENCODER
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding for [B, S, D] sequences."""
    def __init__(self, embed_dim, max_len=4096, dropout=0.1):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        if embed_dim > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds transformer_max_len={self.max_len}."
            )
        return self.dropout(x + self.pe[:, :seq_len].to(dtype=x.dtype, device=x.device))


class LearnablePositionalEncoding(nn.Module):
    """Learned positional embedding for [B, S, D] sequences."""
    def __init__(self, embed_dim, max_len=4096, dropout=0.1):
        super(LearnablePositionalEncoding, self).__init__()
        self.max_len = max_len
        self.position_embedding = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds transformer_max_len={self.max_len}."
            )
        positions = torch.arange(seq_len, device=x.device)
        pos = self.position_embedding(positions).unsqueeze(0).to(dtype=x.dtype)
        return self.dropout(x + pos)


class TransformerSequenceEncoder(nn.Module):
    """
    Transformer encoder for CNN patch sequences.

    Attention weights are not exposed by PyTorch's stock TransformerEncoderLayer.
    The module keeps the layer stack directly accessible at
    ``transformer_encoder.layers`` so attention hooks or a custom layer can be
    added later without changing EmbeddingModel's public interface.
    """
    def __init__(self, embed_dim, num_layers=2, num_heads=4, ff_dim=512,
                 dropout=0.1, activation="gelu", norm_first=True,
                 positional_encoding="sinusoidal", max_len=4096,
                 return_attention_weights=False):
        super(TransformerSequenceEncoder, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"vector_size ({embed_dim}) must be divisible by "
                f"transformer_num_heads ({num_heads})."
            )

        pe_type = str(positional_encoding).lower()
        if pe_type == "sinusoidal":
            self.positional_encoding = SinusoidalPositionalEncoding(
                embed_dim, max_len=max_len, dropout=dropout
            )
        elif pe_type == "learnable":
            self.positional_encoding = LearnablePositionalEncoding(
                embed_dim, max_len=max_len, dropout=dropout
            )
        else:
            raise ValueError(
                "transformer_positional_encoding must be 'sinusoidal' or 'learnable', "
                f"got {positional_encoding!r}."
            )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.return_attention_weights = return_attention_weights
        self.last_attention_weights = None
        if return_attention_weights:
            # TODO: Replace stock TransformerEncoderLayer with a small custom
            # layer that stores per-layer self-attention maps.
            print(
                "[TransformerSequenceEncoder] return_attention_weights=True was "
                "requested, but stock nn.TransformerEncoderLayer does not expose "
                "weights. self.transformer_encoder.layers is hook-friendly.",
                flush=True,
            )

    def forward(self, x):
        x = self.positional_encoding(x)
        self.last_attention_weights = None
        return self.transformer_encoder(x)


# Sliding window function to divide image into patches (subwindows)
def sliding_window(image, window_size, stride, debug_mode=False, save_dir=False):
    """Vectorized sliding window using torch.Tensor.unfold.

    Args:
        image: [B, C, H, W]
    Returns:
        patches: [B, num_windows, C, H, window_size]
    """
    # Unfold along the width dimension: [B, C, H, num_windows, window_size]
    patches = image.unfold(dimension=3, size=window_size, step=stride)
    # Reorder to [B, num_windows, C, H, window_size]
    patches = patches.permute(0, 3, 1, 2, 4).contiguous()

    if debug_mode and save_dir:
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                torchvision.utils.save_image(patches[i, j], f"{save_dir}/patch_b{i}_w{j * stride}.png")

    return patches

class ModifiedOCRResNet34(nn.Module):
    """
    An ImageNet ResNet34 adapted specifically for sequence alignment and OCR.
    Alters deep horizontal layer strides to maintain high-resolution text tracking.
    """
    def __init__(self, vector_size=512):
        super(ModifiedOCRResNet34, self).__init__()
        
        # 1. Load standard ResNet34 with pre-trained weights when available.
        # Figure/eval jobs immediately load a checkpoint afterwards, so falling
        # back to an uninitialised backbone is fine if the ImageNet cache is
        # unavailable on a read-only home filesystem.
        try:
            base_resnet = torchvision.models.resnet34(
                weights=torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            )
        except Exception as exc:
            print(
                f"[ModifiedOCRResNet34] ImageNet weights unavailable ({exc}); "
                "falling back to weights=None.",
                flush=True,
            )
            base_resnet = torchvision.models.resnet34(weights=None)
        
        # 2. THE FIX: Change horizontal strides from 2 to 1 in later stages.
        # Format: (vertical_stride, horizontal_stride)
        
        # Modify Layer 3 (First block downsamples height, leaves width open)
        base_resnet.layer3[0].conv1.stride = (2, 1)
        base_resnet.layer3[0].downsample[0].stride = (2, 1)
        
        # Modify Layer 4 (First block downsamples height, leaves width open)
        base_resnet.layer4[0].conv1.stride = (2, 1)
        base_resnet.layer4[0].downsample[0].stride = (2, 1)
        
        # 3. Strip away the global average pooling and fully connected classification layers
        self.backbone = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool,
            base_resnet.layer1,
            base_resnet.layer2,
            base_resnet.layer3,
            base_resnet.layer4
        )

        # 4. Project features to the required vector_size
        self.feature_proj = nn.Linear(512, vector_size)
        
        # 5. Squash height to 1 while leaving width dynamic
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        
    def forward(self, x):
        # Input shape: [Batch, 3, Height, Width]
        
        # Pass through modified layers
        features = self.backbone(x)          # Shape: [Batch, 512, Height/8, Width/8]
        
        # Collapse the vertical height dimension entirely
        features = self.adaptive_pool(features)  # Shape: [Batch, 512, 1, Width/8]
        features = features.squeeze(2)            # Shape: [Batch, 512, Width/8]
        
        # Project to vector_size
        features = features.permute(0, 2, 1)      # Shape: [Batch, Width/8, 512]
        features = self.feature_proj(features)    # Shape: [Batch, Width/8, vector_size]
        
        return features



class EmbeddingModel(nn.Module):
    def __init__(self, window_size=128, stride=64, vector_size=512,
                  device='cuda', use_flip=False, use_bilstm=True,
                  bilstm_layers=1, bilstm_hidden_dim=None, dropout=0.1,
                  sequence_encoder_type=None,
                  transformer_num_layers=2,
                  transformer_num_heads=4,
                  transformer_ff_dim=512,
                  transformer_dropout=0.1,
                  transformer_activation="gelu",
                  transformer_norm_first=True,
                  transformer_positional_encoding="sinusoidal",
                  transformer_max_len=4096,
                  return_attention_weights=False):
        """
        Image embedding model: ResNet34 CNN + optional sequence encoder.

        Args:
            window_size: Size of sliding window for patch extraction
            stride: Stride for sliding window
            vector_size: Dimension of output feature vectors
            device: Device to run on ('cuda' or 'cpu')
            use_flip: Flip patches left-to-right before CNN
            use_bilstm: Backward-compatible BiLSTM flag
            bilstm_layers: Number of BiLSTM layers
            dropout: Dropout rate
            sequence_encoder_type: "none", "bilstm", or "transformer".
                If None, infer from use_bilstm for old callers.
        """
        super(EmbeddingModel, self).__init__()

        self.device = device
        self.use_flip = use_flip
        if sequence_encoder_type is None:
            sequence_encoder_type = "bilstm" if use_bilstm else "none"
        sequence_encoder_type = str(sequence_encoder_type).lower()
        if sequence_encoder_type not in {"none", "bilstm", "transformer"}:
            raise ValueError(
                "sequence_encoder_type must be one of: 'none', 'bilstm', "
                f"'transformer'. Got {sequence_encoder_type!r}."
            )
        self.sequence_encoder_type = sequence_encoder_type
        self.use_bilstm = sequence_encoder_type == "bilstm"
        self.transformer_positional_encoding = transformer_positional_encoding
        self._sequence_debug_printed = False

        # CNN encoder (ResNet34) for patch feature extraction
        self.cnn_encoder = ModifiedOCRResNet34(vector_size=vector_size)
        self.cnn_encoder = self.cnn_encoder.to(device)

        # Sequence context encoder: None, Bi-LSTM, or Transformer
        if sequence_encoder_type == "bilstm":
            _hidden = bilstm_hidden_dim if bilstm_hidden_dim is not None else vector_size // 2
            self.sequence_encoder = BiLSTMEncoder(
                embed_dim=vector_size,
                hidden_dim=_hidden,
                lstm_layers=bilstm_layers,
                dropout=dropout,
            ).to(device)
            print(f"[EmbeddingModel] Using BiLSTM encoder (layers={bilstm_layers}, hidden_dim={_hidden})")
        elif sequence_encoder_type == "transformer":
            self.sequence_encoder = TransformerSequenceEncoder(
                embed_dim=vector_size,
                num_layers=transformer_num_layers,
                num_heads=transformer_num_heads,
                ff_dim=transformer_ff_dim,
                dropout=transformer_dropout,
                activation=transformer_activation,
                norm_first=transformer_norm_first,
                positional_encoding=transformer_positional_encoding,
                max_len=transformer_max_len,
                return_attention_weights=return_attention_weights,
            ).to(device)
            print(
                "[EmbeddingModel] Using Transformer encoder "
                f"(layers={transformer_num_layers}, heads={transformer_num_heads}, "
                f"ff_dim={transformer_ff_dim}, pos={transformer_positional_encoding})",
                flush=True,
            )
        else:
            self.sequence_encoder = None
            print("[EmbeddingModel] No sequence encoder (CNN features only)")

        # Layer normalization for similarity computation
        self.vision_norm = nn.LayerNorm(vector_size).to(device)

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

        # Reshape patches: [B, W, C, H, ws] -> [B*W, C, H, ws]
        reshaped_tokens_a = tokens_a.reshape(total_patches, Channels, H, W)
        if show_dims:
            print(f"Patches after reshaping: {reshaped_tokens_a.shape}")

        # --- Chunked CNN forward to cap peak memory ---
        if total_patches <= self.CNN_CHUNK_SIZE:
            encoded_tokens_a = self.cnn_encoder(reshaped_tokens_a) # [Total_patches, seq_len, vector_size]
        else:
            # Pre-allocate output buffer to avoid the extra concat copy.
            first_chunk = reshaped_tokens_a[:self.CNN_CHUNK_SIZE]
            first_out = self.cnn_encoder(first_chunk) # [Chunk_size, seq_len, vector_size]
            seq_len = first_out.shape[1]
            out_dim = first_out.shape[2]
            encoded_tokens_a = torch.empty(
                total_patches, seq_len, out_dim,
                device=first_out.device, dtype=first_out.dtype,
            )
            encoded_tokens_a[:self.CNN_CHUNK_SIZE] = first_out
            for start in range(self.CNN_CHUNK_SIZE, total_patches, self.CNN_CHUNK_SIZE):
                end = min(start + self.CNN_CHUNK_SIZE, total_patches)
                encoded_tokens_a[start:end] = self.cnn_encoder(reshaped_tokens_a[start:end])

        if show_dims:
            print(f"Tokens after CNN: {encoded_tokens_a.shape}")

        return encoded_tokens_a, batches_num, windows_num

    def _apply_sequence_encoder(self, x):
        """Apply the configured sequence encoder to [B, S, D] CNN features."""
        if self.sequence_encoder_type == "transformer" and not self._sequence_debug_printed:
            print(
                f"sequence_encoder_type=transformer "
                f"positional_encoding={self.transformer_positional_encoding} "
                f"input_seq_len={x.shape[1]}",
                flush=True,
            )
            self._sequence_debug_printed = True
        if self.sequence_encoder is None:
            return x
        return self.sequence_encoder(x)

    def forward(self, image_a, show_dims=False, debug=False, timer=None):
        t = timer if timer is not None else img_embed_timer

        # ==================== PHASE: Sliding Window (Patch Extraction) ====================
        t.start('img_1a_sliding_window')
        tokens_a = sliding_window(image_a, self.window_size, self.stride, debug_mode=debug)
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

        # encoded_tokens_a shape: [batches_num * windows_num, seq_len_per_patch, self.vector_size]
        # We need to flatten to [batches_num, windows_num * seq_len_per_patch, self.vector_size]
        features_vector_a = encoded_tokens_a.view(batches_num, -1, self.vector_size)
        del encoded_tokens_a

        # ==================== PHASE: Sequence Context ====================
        t.start('img_1f_sequence_encoder')
        features_vector_a = self._apply_sequence_encoder(features_vector_a)
        t.stop('img_1f_sequence_encoder')

        # Apply LayerNorm before computing similarity
        features_vector_a = self.vision_norm(features_vector_a)

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
        )
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
            batches_num, -1, self.vector_size
        )
        del encoded_tokens_a

        # ---- Sequence encoder ----
        t.start(f'img_scale_{scale_window_size}_seq_enc')
        features_vector_a = self._apply_sequence_encoder(features_vector_a)
        t.stop(f'img_scale_{scale_window_size}_seq_enc')

        # Apply LayerNorm before computing similarity
        features_vector_a = self.vision_norm(features_vector_a)

        return features_vector_a
