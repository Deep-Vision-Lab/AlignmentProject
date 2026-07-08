import os

import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    AutoModel = None
    AutoTokenizer = None


class ArabicTokenTextEncoder(nn.Module):
    def __init__(
        self,
        model_name="aubmindlab/bert-base-arabertv02",
        output_dim=128,
        max_token_chars=3,
        freeze_backbone=True,
        device="cpu",
    ):
        super().__init__()
        self.model_name = model_name
        self.max_token_chars = max_token_chars
        self.freeze_backbone = freeze_backbone
        self.device = torch.device(device)

        if AutoTokenizer is None or AutoModel is None:
            raise ImportError(
                "transformers is required for ArabicTokenTextEncoder. "
                "Install it or use TEXT_ENCODER_TYPE=char."
            )

        cache_dir = os.environ.get("HF_HOME") or os.environ.get("TRANSFORMERS_CACHE") or None
        local_files_only = (
            os.environ.get("HF_HUB_OFFLINE", "0").lower() in {"1", "true", "yes", "on"}
            or os.environ.get("TRANSFORMERS_OFFLINE", "0").lower() in {"1", "true", "yes", "on"}
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        self.backbone = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        hidden_size = self.backbone.config.hidden_size

        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad_(False)
            self.backbone.eval()

        self.projection = nn.Linear(hidden_size, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.to(self.device)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def tokenize_visual_units(self, text):
        tokens = []
        current_word = []

        def flush_word():
            if not current_word:
                return
            word = "".join(current_word)
            for i in range(0, len(word), self.max_token_chars):
                tokens.append(word[i:i + self.max_token_chars])
            current_word.clear()

        for char in text:
            if char.isspace():
                flush_word()
                tokens.append(" ")
            else:
                current_word.append(char)
        flush_word()
        return tokens

    def _encoded_inputs(self, tokens):
        encoded = self.tokenizer(
            tokens,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def _pool_non_special_tokens(self, hidden, encoded):
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        if "special_tokens_mask" in encoded:
            non_special = (1 - encoded["special_tokens_mask"]).unsqueeze(-1).float()
            pool_mask = attention_mask * non_special
            empty_rows = pool_mask.sum(dim=1, keepdim=True) == 0
            pool_mask = torch.where(empty_rows, attention_mask, pool_mask)
        else:
            pool_mask = attention_mask

        summed = (hidden * pool_mask).sum(dim=1)
        counts = pool_mask.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def forward(self, text):
        tokens = self.tokenize_visual_units(text)
        if not tokens:
            return torch.empty(0, self.projection.out_features, device=self.device)

        encoded = self._encoded_inputs(tokens)
        backbone_inputs = {
            key: value
            for key, value in encoded.items()
            if key != "special_tokens_mask"
        }

        if self.freeze_backbone:
            with torch.no_grad():
                outputs = self.backbone(**backbone_inputs)
        else:
            outputs = self.backbone(**backbone_inputs)

        pooled = self._pool_non_special_tokens(outputs.last_hidden_state, encoded)
        projected = self.projection(pooled)
        return self.norm(projected)
