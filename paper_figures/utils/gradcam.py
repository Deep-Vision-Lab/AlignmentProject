"""
gradcam.py
==========
Explainability utilities used by fig12_window_feature_concentration.py.

Provides:
  GradCAMExtractor     — Grad-CAM at the last ResNet34 conv block (layer4).
  compute_saliency     — Full-image input-gradient saliency, window-cropped.
  compute_occlusion    — Occlusion sensitivity inside a window region.
  normalize_map        — Normalize a 2-D numpy array to [0, 1].
  resize_map           — Bilinear resize of a 2-D float map.
  overlay_heatmap_on_image — Blend a heatmap over an RGB image array.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Generic utilities
# ─────────────────────────────────────────────────────────────────────────────

def normalize_map(x):
    """Normalize a numpy array to [0, 1].  Returns zeros for constant maps."""
    x = np.array(x, dtype=np.float32)
    lo, hi = x.min(), x.max()
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def resize_map(heatmap, target_h, target_w):
    """Bilinear resize of a [H, W] float32 map in [0, 1] to (target_h, target_w)."""
    pil = Image.fromarray(
        (np.clip(heatmap, 0, 1) * 255).astype(np.uint8), mode="L"
    )
    pil = pil.resize((target_w, target_h), Image.BILINEAR)
    return np.array(pil, dtype=np.float32) / 255.0


def overlay_heatmap_on_image(image_arr, heatmap, alpha=0.45, cmap="jet"):
    """
    Blend a [H, W] heatmap (values in [0, 1]) onto a [H, W, 3] uint8 image.

    Returns a [H, W, 3] uint8 image.
    """
    hm_norm  = normalize_map(heatmap)
    colormap = cm.get_cmap(cmap)
    colored  = (colormap(hm_norm)[:, :, :3] * 255).astype(np.float32)
    base     = image_arr.astype(np.float32)
    blended  = (alpha * colored + (1.0 - alpha) * base).clip(0, 255).astype(np.uint8)
    return blended


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM
# ─────────────────────────────────────────────────────────────────────────────

class GradCAMExtractor:
    """
    Grad-CAM extractor targeting a configurable layer of ModifiedOCRResNet34.

    The CNN inside EmbeddingModel processes all windows as a flat batch
    [B*num_windows, C, H, W_patch], so the hook captures activations and
    gradients for every window simultaneously.  The target window's slice
    is extracted after the backward pass.

    Because the backward pass flows through:
        LayerNorm → BiLSTM → flatten → CNN outputs → backbone
    the resulting CAM weights reflect full-sequence context, not just the
    isolated window.

    Args:
        model:       EmbeddingModel in eval mode.
        layer_name:  Which ResNet34 layer to hook. One of
                     'layer1', 'layer2', 'layer3', 'layer4' (default: 'layer3').
                     layer4 is too coarse for 16px windows; layer3 gives 2× finer maps.

    Limitation
    ----------
    If the CNN runs in multiple chunks (total_patches > CNN_CHUNK_SIZE=512,
    which happens only for very large batches), the hook fires per chunk and
    the stored tensors correspond to the *last* chunk.  For typical figure
    generation (B=1, ~127 windows) this limit is never reached.
    """

    _LAYER_MAP = {"layer1": 4, "layer2": 5, "layer3": 6, "layer4": 7}

    def __init__(self, model, layer_name="layer3"):
        self.model        = model
        self.layer_name   = layer_name
        self._activations = None
        self._gradients   = None
        self._hooks: list = []

    def _attach(self):
        layer_idx = self._LAYER_MAP.get(self.layer_name, 7)
        target    = self.model.cnn_encoder.backbone[layer_idx]

        def _fwd(module, inp, out):
            # out: [B*num_windows, C, H', W']
            self._activations = out.detach().clone()

        def _bwd(module, grad_in, grad_out):
            # grad_out[0]: gradient of loss w.r.t. layer output
            self._gradients = grad_out[0].detach().clone()

        self._hooks.append(target.register_forward_hook(_fwd))
        self._hooks.append(target.register_full_backward_hook(_bwd))

    def _detach(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def compute(self, img_tensor, window_idx_model, score_fn):
        """
        Compute Grad-CAM for one window.

        Args:
            img_tensor:        [1, C, H, W] on device.
            window_idx_model:  Window index in model order (0 = first char for RTL).
            score_fn:          Callable(output [1, S, D]) → differentiable scalar.
                               Must NOT be wrapped in torch.no_grad().

        Returns:
            cam: [H_feat, W_feat] numpy float32 array in [0, 1].
                 Upsample to window pixel size with resize_map().

        Raises:
            RuntimeError: if hooks did not fire.
        """
        self._activations = None
        self._gradients   = None
        self._attach()

        self.model.zero_grad()
        # cuDNN LSTM backward requires training mode; disable cuDNN for the
        # entire forward+backward so the non-cuDNN path is used throughout.
        with torch.backends.cudnn.flags(enabled=False):
            output = self.model(img_tensor)    # hooks fire here
            score  = score_fn(output)
            score.backward()                   # backward hooks fire here
        self._detach()

        if self._activations is None:
            raise RuntimeError(
                "GradCAM forward hook did not fire. "
                "Verify cnn_encoder.backbone[-1] is a valid module."
            )
        if self._gradients is None:
            raise RuntimeError(
                "GradCAM backward hook did not fire. "
                "Ensure the score is connected to the CNN backbone in the graph."
            )

        # [num_windows, C, H', W'] — extract target window
        acts  = self._activations[window_idx_model]    # [C, H', W']
        grads = self._gradients[window_idx_model]      # [C, H', W']

        # Global average pool of gradients → channel importance weights
        weights = grads.mean(dim=(1, 2), keepdim=True)   # [C, 1, 1]
        cam     = (weights * acts).sum(dim=0)             # [H', W']
        cam     = F.relu(cam)
        return normalize_map(cam.cpu().numpy())

    def __del__(self):
        self._detach()


# ─────────────────────────────────────────────────────────────────────────────
# Saliency Map
# ─────────────────────────────────────────────────────────────────────────────

def compute_saliency(model, img_tensor, col_start, col_end, score_fn):
    """
    Full-image input-gradient saliency, cropped to the selected window region.

    Running the saliency on the full image (rather than an isolated window)
    keeps BiLSTM sequence context in the gradient signal.

    Args:
        model:       EmbeddingModel in eval mode.
        img_tensor:  [1, C, H, W] on device.
        col_start:   Window's leftmost column in the full image.
        col_end:     Window's rightmost column (exclusive).
        score_fn:    Callable(output [1, S, D]) → differentiable scalar.

    Returns:
        [H, W_window] numpy float32 array in [0, 1].
    """
    img_in = img_tensor.clone().requires_grad_(True)
    model.zero_grad()
    with torch.backends.cudnn.flags(enabled=False):
        output = model(img_in)
        score  = score_fn(output)
        score.backward()

    # Max-abs over channel dimension → [H, W]
    sal_full = img_in.grad.abs().max(dim=1)[0].squeeze(0).cpu().numpy()
    sal_crop = sal_full[:, col_start:col_end]
    return normalize_map(sal_crop)


# ─────────────────────────────────────────────────────────────────────────────
# Occlusion Sensitivity
# ─────────────────────────────────────────────────────────────────────────────

def compute_occlusion(model, img_tensor, col_start, col_end,
                       score_fn, window_h, window_w,
                       patch_size=8, occ_stride=4, fill="mean",
                       batch_size=4):
    """
    Occlusion sensitivity map for the selected window region.

    Slides a small rectangular patch over the window area, measures how much
    the target score drops when that region is masked, and builds a 2-D
    importance map.

    Args:
        model:        EmbeddingModel in eval mode.
        img_tensor:   [1, C, H, W] on device.
        col_start:    Window's leftmost column in the full image.
        col_end:      Window's rightmost column (exclusive).
        score_fn:     Callable(output [1, S, D]) → scalar.  No grad needed.
        window_h:     Window height in pixels.
        window_w:     Window width in pixels.
        patch_size:   Occluding patch side length (pixels).
        occ_stride:   Stride of the occlusion scan.
        fill:         'zero' | 'mean' — value to fill the masked region with.
        batch_size:   Number of masked images evaluated per model forward.

    Returns:
        occ_map:        [window_h, window_w] numpy float32 array in [0, 1].
        original_score: float — score with no masking.
        max_drop:       float — maximum observed score drop.
    """
    with torch.no_grad():
        orig_emb       = model(img_tensor)
        original_score = score_fn(orig_emb).item()

    fill_val = 0.0
    if fill == "mean":
        fill_val = img_tensor[:, :, :, col_start:col_end].mean().item()

    drop_acc  = np.zeros((window_h, window_w), dtype=np.float32)
    count_acc = np.zeros((window_h, window_w), dtype=np.float32)

    positions = [
        (y0, x0)
        for y0 in range(0, window_h - patch_size + 1, occ_stride)
        for x0 in range(0, window_w - patch_size + 1, occ_stride)
    ]
    total = len(positions)

    for start in range(0, total, batch_size):
        batch_positions = positions[start:start + batch_size]
        masked = img_tensor.repeat(len(batch_positions), 1, 1, 1)
        for batch_idx, (y0, x0) in enumerate(batch_positions):
            masked[batch_idx, :,
                   y0:y0 + patch_size,
                   col_start + x0:col_start + x0 + patch_size] = fill_val

        with torch.no_grad():
            embeddings = model(masked)
            scores = [
                score_fn(embeddings[i:i + 1]).item()
                for i in range(len(batch_positions))
            ]

        for (y0, x0), score in zip(batch_positions, scores):
            drop = original_score - score
            drop_acc[y0:y0 + patch_size, x0:x0 + patch_size]  += drop
            count_acc[y0:y0 + patch_size, x0:x0 + patch_size] += 1.0

        done = min(start + batch_size, total)
        print(f"      occlusion {done}/{total}", end="\r", flush=True)

    print()
    mask = count_acc > 0
    drop_acc[mask] /= count_acc[mask]

    occ_map  = normalize_map(np.clip(drop_acc, 0, None))
    max_drop = float(np.clip(drop_acc, 0, None).max())
    return occ_map, original_score, max_drop
