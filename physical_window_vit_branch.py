"""Architecture patch for direct physical-window ViT contextualization.

Local branch:
    exact 128x32 RGB windows, stride 16 -> shared pretrained ResNet-18 -> L_t

Context branch:
    the SAME exact 128x32 RGB windows, stride 16 -> direct linear window
    embedding -> pretrained ViT-Tiny transformer sequence -> C_t

Fusion:
    concat/projection of L_t and C_t -> positive letter-DTW

The contextual branch never consumes ResNet vectors and never subdivides a
128x32 physical window into smaller custom patches.
"""
from __future__ import annotations

from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from physical_window_vit_encoder import PhysicalWindowEmbedding
from pretrained_tiny_vit import initialize_tiny_vit_from_pretrained
from resnet18_window_encoder import ResNet18WindowEncoder
from restoration_recommended_components import LocalContextFusion, line_padding_masks


def attach_physical_window_vit_stages(model, P):
    """Install the physical-window contextual ViT while preserving local ResNet."""
    if getattr(model, "_physical_window_vit_installed", False):
        return model
    if not hasattr(model, "vit_encoder"):
        raise TypeError("This branch expects the shared ViT window container")

    vit = model.vit_encoder
    dim = int(vit.embed_dim)
    if dim != 192:
        raise ValueError(
            f"ViT-Tiny requires a 192-D token space, got embed_dim={dim}."
        )
    if int(vit.input_height) != 128 or int(vit.window_size) != 32:
        raise ValueError(
            "Physical-window ViT branch requires 128x32 physical windows."
        )

    reference_parameter = next(vit.patch_embedding.parameters())
    device = reference_parameter.device
    dtype = reference_parameter.dtype

    # Keep patch_embedding as the LOCAL ResNet encoder so the existing shared
    # gradient diagnostic still reports ResNet18 parameter_norm correctly.
    vit.patch_embedding = ResNet18WindowEncoder(
        input_height=int(vit.input_height),
        window_size=int(vit.window_size),
        stride=int(vit.stride),
        embed_dim=dim,
        pretrained=bool(getattr(P, "resnet18_pretrained", True)),
        local_files_only=bool(getattr(P, "pretrained_local_only", True)),
    ).to(device=device, dtype=dtype)

    # Register the raw physical-window projection under the ViT encoder. The
    # shared diagnostic defines ViT-Tiny parameter_norm from encoder.parameters(),
    # so this keeps the new input projection inside the correct gradient group.
    vit.encoder.context_window_embedding = PhysicalWindowEmbedding(
        input_height=int(vit.input_height),
        window_size=int(vit.window_size),
        stride=int(vit.stride),
        embed_dim=dim,
    ).to(device=device, dtype=dtype)

    vit.restoration_local_encoder = "resnet18"
    vit.local_encoder_type = "resnet18"
    vit.context_input_type = "direct_physical_rgb_window"
    vit.context_window_height = int(vit.input_height)
    vit.context_window_width = int(vit.window_size)
    vit.context_window_stride = int(vit.stride)
    vit.context_window_subdivision = "none"

    if len(vit.encoder.layers) != 12:
        raise ValueError(
            f"Expected 12 ViT-Tiny transformer layers, got {len(vit.encoder.layers)}"
        )
    first_layer = vit.encoder.layers[0]
    if int(first_layer.self_attn.num_heads) != 3:
        raise ValueError(
            f"Expected 3 ViT-Tiny heads, got {first_layer.self_attn.num_heads}"
        )
    if int(first_layer.linear1.out_features) != 768:
        raise ValueError(
            f"Expected ViT-Tiny MLP width 768, got {first_layer.linear1.out_features}"
        )
    vit.vit_variant = "vit_tiny_192d_12l_3h_physical_window_input"

    if bool(getattr(P, "tiny_vit_pretrained", False)):
        initialize_tiny_vit_from_pretrained(
            vit,
            model_name=str(
                getattr(
                    P,
                    "tiny_vit_pretrained_model",
                    "facebook/deit-tiny-patch16-224",
                )
            ),
            local_files_only=bool(getattr(P, "pretrained_local_only", True)),
        )
    else:
        vit.context_pretrained = False
        vit.context_pretrained_model = ""

    vit.semantic_adapter = nn.Identity().to(device=device)
    vit.restoration_semantic_adapter = "identity"
    vit.fusion_head = LocalContextFusion(dim).to(device=device, dtype=dtype)

    def encode_physical_window_sequence(self, image, *, use_flip):
        if image.ndim != 4 or int(image.shape[1]) != 3:
            raise ValueError("Expected image [B,3,H,W]")
        if int(image.shape[2]) != self.input_height:
            raise ValueError("Unexpected input height")
        if int(image.shape[3]) < self.window_size:
            raise ValueError("Input width is smaller than the window width")

        model_input = image

        # LOCAL: ResNet sees explicit 128x32 / stride-16 physical windows.
        local_tokens = self.patch_embedding(model_input)
        if local_tokens.shape[2] != 1:
            raise RuntimeError("ResNet local encoder must produce one token row")
        local = local_tokens.squeeze(2).transpose(1, 2).contiguous()

        # CONTEXT INPUT: ViT receives the SAME explicit physical windows directly.
        context_tokens = self.encoder.context_window_embedding(model_input)
        if context_tokens.shape[2] != 1:
            raise RuntimeError("Physical-window ViT embedding must produce one token row")
        context_seed = context_tokens.squeeze(2).transpose(1, 2).contiguous()

        if local.shape != context_seed.shape:
            raise RuntimeError(
                "Local and contextual physical-window sequences must match exactly: "
                f"local={tuple(local.shape)} context={tuple(context_seed.shape)}"
            )

        if use_flip:
            local = torch.flip(local, dims=[1])
            context_seed = torch.flip(context_seed, dims=[1])

        # Keep the local representation normalized exactly as in the baseline.
        local = self.local_norm(local)

        token_valid, _pixel_valid = line_padding_masks(
            model_input,
            window_size=self.window_size,
            stride=self.stride,
            use_flip=use_flip,
        )
        if token_valid.shape != local.shape[:2]:
            raise RuntimeError(
                "Physical-window validity mask and token geometry differ: "
                f"mask={tuple(token_valid.shape)} tokens={tuple(local.shape[:2])}"
            )

        positional = context_seed + self._position_tokens(context_seed.shape[1]).to(
            dtype=context_seed.dtype,
            device=context_seed.device,
        )
        contextual = self.encoder(
            self.input_dropout(positional),
            src_key_padding_mask=~token_valid,
        )

        fused = self.fusion_head(local, contextual)
        return fused, local, contextual, model_input, token_valid

    def minimal_window_forward(self, image, *, use_flip, return_model_input=False):
        fused, local, _contextual, model_input, _token_valid = (
            self.encode_physical_window_sequence(image, use_flip=use_flip)
        )
        if return_model_input:
            return fused, local, model_input
        return fused, local

    vit.encode_physical_window_sequence = MethodType(
        encode_physical_window_sequence, vit
    )
    # Compatibility with shared diagnostics/utilities that call this name.
    vit.encode_restoration_sequence = vit.encode_physical_window_sequence
    vit.forward = MethodType(minimal_window_forward, vit)

    def model_forward(
        self,
        image,
        show_dims=False,
        return_local=False,
        return_ink=False,
        return_grouped=False,
        return_training_bundle=False,
    ):
        fused, local, contextual, model_input, token_valid = (
            self.vit_encoder.encode_physical_window_sequence(
                image, use_flip=self.use_flip
            )
        )

        fused_out = F.normalize(
            self.vision_norm(fused).float(), p=2, dim=-1
        ).to(dtype=fused.dtype)
        local_out = self.vision_norm(local)

        if not return_training_bundle:
            if show_dims:
                print(
                    "image embeddings: ResNet18-local + physical-window-ViT-context "
                    f"fused={tuple(fused_out.shape)} local={tuple(local_out.shape)} "
                    f"context={tuple(contextual.shape)} window=128x32 "
                    f"stride={self.vit_encoder.stride}",
                    flush=True,
                )
            outputs = [fused_out]
            if return_local:
                outputs.append(local_out)
            if return_grouped:
                outputs.append(local_out)
            if return_ink:
                outputs.append(token_valid.float())
            return outputs[0] if len(outputs) == 1 else tuple(outputs)

        contextual_out = self.vision_norm(contextual)

        if self.training and torch.is_grad_enabled():
            local.retain_grad()
            contextual.retain_grad()
            fused.retain_grad()
            fused_out.retain_grad()
            if not hasattr(self, "_gradient_probe_records"):
                self._gradient_probe_records = []
            self._gradient_probe_records.append(
                {
                    "after_resnet18": local,
                    "after_vit_tiny": contextual,
                    "after_fusion": fused,
                    "final_fused": fused_out,
                    "token_valid": token_valid,
                }
            )

        return {
            "semantic": fused_out,
            "fused": fused_out,
            "primitive": local_out,
            "primitive_raw": local,
            "contextual": contextual_out,
            "ink": token_valid.float(),
            "token_valid": token_valid,
            "model_input": model_input,
        }

    model.forward = MethodType(model_forward, model)
    model._restoration_positive_dtw_installed = True
    model._physical_window_vit_installed = True
    return model
