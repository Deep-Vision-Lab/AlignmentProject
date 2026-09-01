"""Checkpoint-aware ViT reconstruction using the canonical EmbeddingModel."""
from __future__ import annotations
import os
import torch
from Evaluation import _eval_utils
from embeddingModel import EmbeddingModel
def _bool(value,default=False):
    if value is None:return bool(default)
    if isinstance(value,str):return value.strip().lower() in {"1","true","yes","on"}
    return bool(value)
def _checkpoint_config(weights_path):
    checkpoint=torch.load(weights_path,map_location="cpu")
    return dict(checkpoint["model_config"]) if isinstance(checkpoint,dict) and isinstance(checkpoint.get("model_config"),dict) else {}
def install_vit_evaluation_loader():
    if getattr(_eval_utils,"_vit_evaluation_loader_installed",False):return
    original_loader=_eval_utils.load_evaluation_models
    def load_evaluation_models(weights_path,device="auto",load_text_model=True):
        config=_checkpoint_config(weights_path); encoder_type=str(config.get("visual_encoder_type",os.environ.get("VISUAL_ENCODER_TYPE","vit"))).strip().lower()
        if encoder_type!="vit":return original_loader(weights_path,device,load_text_model)
        previous_constructor=_eval_utils.EmbeddingModel
        def vit_constructor(window_size=32,stride=16,vector_size=128,device="cpu",use_flip=False,**_ignored):
            return EmbeddingModel(window_size=int(window_size),stride=int(stride),vector_size=int(vector_size),device=device,use_flip=_bool(use_flip),input_height=int(config.get("vit_input_height",128)),vit_layers=int(config.get("vit_layers",4)),vit_heads=int(config.get("vit_heads",4)),vit_mlp_dim=int(config.get("vit_mlp_dim",512)),vit_dropout=float(config.get("vit_dropout",.10)),vit_max_tokens=int(config.get("vit_max_tokens",256)),vit_position_base_tokens=int(config.get("vit_position_base_tokens",63)))
        try:
            _eval_utils.EmbeddingModel=vit_constructor
            return original_loader(weights_path,device,load_text_model)
        finally:_eval_utils.EmbeddingModel=previous_constructor
    _eval_utils.load_evaluation_models=load_evaluation_models; _eval_utils._vit_evaluation_loader_installed=True
