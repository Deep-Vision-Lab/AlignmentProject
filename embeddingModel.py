"""Canonical pure-ViT image embedding model for AlignmentProject ViT branches."""
from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
_IMAGENET_MEAN=(0.485,0.456,0.406); _IMAGENET_STD=(0.229,0.224,0.225)
def _env_flag(name,default=False):
    value=os.environ.get(name); return bool(default) if value is None else value.strip().lower() in {"1","true","yes","on"}
def _env_int(name,default):
    try:return int(os.environ.get(name,default))
    except (TypeError,ValueError):return int(default)
def _env_float(name,default):
    try:return float(os.environ.get(name,default))
    except (TypeError,ValueError):return float(default)
def _parameter(name,default):
    try:
        import Parameters as P
        return getattr(P,name,default)
    except Exception:return default
def sliding_window(image,window_size,stride): return image.unfold(3,int(window_size),int(stride)).permute(0,3,1,2,4).contiguous()
def _denormalize_imagenet_patches(patches):
    if patches.ndim!=5: raise ValueError(f"Expected [B,S,C,H,W], got {tuple(patches.shape)}")
    if patches.shape[2]!=3: raise ValueError("Ink estimation expects RGB patches")
    mean=patches.new_tensor(_IMAGENET_MEAN).view(1,1,3,1,1); std=patches.new_tensor(_IMAGENET_STD).view(1,1,3,1,1); return (patches.float()*std+mean).clamp(0,1)
def _patch_background_level(gray):
    h,w=int(gray.shape[-2]),int(gray.shape[-1]); bh=max(1,int(round(h*.05))); bw=max(1,int(round(w*.05))); border=torch.cat([gray[...,:bh,:].flatten(2),gray[...,-bh:,:].flatten(2),gray[...,:,:bw].flatten(2),gray[...,:,-bw:].flatten(2)],-1); return border.median(-1).values.unsqueeze(-1).unsqueeze(-1)
def window_ink_ratio_from_patches(patches,contrast_threshold=None):
    threshold=float(os.environ.get("INK_CONTRAST_THRESHOLD",_parameter("ink_contrast_threshold",.15))) if contrast_threshold is None else float(contrast_threshold); threshold=max(0,min(1,threshold))
    with torch.no_grad():
        rgb=_denormalize_imagenet_patches(patches.detach()); gray=.2989*rgb[:,:,0]+.5870*rgb[:,:,1]+.1140*rgb[:,:,2]; return (gray-_patch_background_level(gray)).abs().ge(threshold).float().mean((2,3))
class LineWindowViT(nn.Module):
    def __init__(self,*,input_height,window_size,stride,embed_dim,num_layers,num_heads,mlp_dim,dropout,max_tokens,position_base_tokens):
        super().__init__(); input_height=int(input_height); window_size=int(window_size); stride=int(stride); embed_dim=int(embed_dim); num_layers=int(num_layers); num_heads=int(num_heads); mlp_dim=int(mlp_dim); max_tokens=int(max_tokens); position_base_tokens=int(position_base_tokens)
        if min(input_height,window_size,stride,num_layers,num_heads,mlp_dim,max_tokens,position_base_tokens)<=0: raise ValueError("ViT dimensions must be positive")
        if embed_dim%num_heads: raise ValueError(f"VIT_HEADS={num_heads} must divide VECTOR_SIZE={embed_dim}")
        if position_base_tokens>max_tokens: raise ValueError("VIT_POSITION_BASE_TOKENS must not exceed VIT_MAX_TOKENS")
        self.input_height=input_height; self.window_size=window_size; self.stride=stride; self.position_base_tokens=position_base_tokens
        self.patch_embedding=nn.Conv2d(3,embed_dim,kernel_size=(input_height,window_size),stride=(input_height,stride),bias=True); self.local_norm=nn.LayerNorm(embed_dim); self.position_embedding=nn.Parameter(torch.zeros(1,max_tokens,embed_dim)); self.input_dropout=nn.Dropout(float(dropout)); layer=nn.TransformerEncoderLayer(d_model=embed_dim,nhead=num_heads,dim_feedforward=mlp_dim,dropout=float(dropout),activation="gelu",batch_first=True,norm_first=True); self.encoder=nn.TransformerEncoder(layer,num_layers=num_layers,norm=nn.LayerNorm(embed_dim)); nn.init.trunc_normal_(self.position_embedding,std=.02); nn.init.xavier_uniform_(self.patch_embedding.weight); nn.init.zeros_(self.patch_embedding.bias)
    def _position_tokens(self,count):
        count=int(count); base=self.position_embedding[:,:self.position_base_tokens]
        if count<=0: raise ValueError("position token count must be positive")
        return base if count==self.position_base_tokens else F.interpolate(base.transpose(1,2),size=count,mode="linear",align_corners=False).transpose(1,2)
    def forward(self,image,*,use_flip):
        if image.ndim!=4 or image.shape[1]!=3: raise ValueError(f"ViT input must be [B,3,H,W], got {tuple(image.shape)}")
        if int(image.shape[2])!=self.input_height: raise ValueError(f"ViT expects input height {self.input_height}, got {image.shape[2]}")
        tokens=self.patch_embedding(image)
        if tokens.shape[2]!=1: raise RuntimeError(f"Full-height patch embedding produced {tuple(tokens.shape)}")
        tokens=tokens.squeeze(2).transpose(1,2).contiguous(); tokens=torch.flip(tokens,[1]) if use_flip else tokens; local=self.local_norm(tokens); contextual=local+self._position_tokens(local.shape[1]).to(dtype=local.dtype,device=local.device); return self.encoder(self.input_dropout(contextual)),local
class EmbeddingModel(nn.Module):
    visual_encoder_type="vit"
    def __init__(self,window_size=32,stride=16,vector_size=128,device="cuda",use_flip=False,input_height=None,vit_layers=None,vit_heads=None,vit_mlp_dim=None,vit_dropout=None,vit_max_tokens=None,vit_position_base_tokens=None,*,use_bilstm=False,bilstm_layers=None,bilstm_hidden_dim=None,use_local_grouping=False,local_group_size=None,**_ignored):
        super().__init__()
        if use_bilstm: raise ValueError("Pure ViT EmbeddingModel does not support BiLSTM")
        if use_local_grouping: raise ValueError("Pure ViT EmbeddingModel does not support local grouping")
        self.device=device; self.window_size=int(window_size); self.stride=int(stride); self.vector_size=int(vector_size); self.use_bilstm=False; self.input_height=int(_parameter("vit_input_height",128) if input_height is None else input_height); self.vit_layers=int(_parameter("vit_layers",_env_int("VIT_LAYERS",4)) if vit_layers is None else vit_layers); self.vit_heads=int(_parameter("vit_heads",_env_int("VIT_HEADS",4)) if vit_heads is None else vit_heads); self.vit_mlp_dim=int(_parameter("vit_mlp_dim",_env_int("VIT_MLP_DIM",512)) if vit_mlp_dim is None else vit_mlp_dim); self.vit_dropout=float(_parameter("vit_dropout",_env_float("VIT_DROPOUT",.1)) if vit_dropout is None else vit_dropout); self.vit_max_tokens=int(_parameter("vit_max_tokens",_env_int("VIT_MAX_TOKENS",256)) if vit_max_tokens is None else vit_max_tokens); self.vit_position_base_tokens=int(_parameter("vit_position_base_tokens",_env_int("VIT_POSITION_BASE_TOKENS",63)) if vit_position_base_tokens is None else vit_position_base_tokens); self.register_buffer("_use_flip_state",torch.tensor(1 if use_flip else 0,dtype=torch.uint8)); self.register_buffer("_use_local_grouping_state",torch.tensor(0,dtype=torch.uint8)); self.vit_encoder=LineWindowViT(input_height=self.input_height,window_size=self.window_size,stride=self.stride,embed_dim=self.vector_size,num_layers=self.vit_layers,num_heads=self.vit_heads,mlp_dim=self.vit_mlp_dim,dropout=self.vit_dropout,max_tokens=self.vit_max_tokens,position_base_tokens=self.vit_position_base_tokens).to(device); self.vision_norm=nn.LayerNorm(self.vector_size).to(device)
    @property
    def use_flip(self): return bool(int(self._use_flip_state.item()))
    @property
    def use_local_grouping(self): return False
    def forward(self,image,show_dims=False,return_local=False,return_ink=False,return_grouped=False):
        contextual,local=self.vit_encoder(image,use_flip=self.use_flip); ink=None
        if return_ink:
            patches=sliding_window(image,self.window_size,self.stride); patches=torch.flip(patches,[1]) if self.use_flip else patches; ink=window_ink_ratio_from_patches(patches)
            if ink.shape[1]!=local.shape[1]: raise RuntimeError(f"ViT token/ink count mismatch: {local.shape[1]} != {ink.shape[1]}")
        contextual=self.vision_norm(contextual); local=self.vision_norm(local); outputs=[contextual]
        if show_dims: print(f"image embeddings: encoder=vit contextual={tuple(contextual.shape)} local={tuple(local.shape)} flip={self.use_flip}",flush=True)
        if return_local: outputs.append(local)
        if return_grouped: outputs.append(local)
        if return_ink: outputs.append(ink)
        return outputs[0] if len(outputs)==1 else tuple(outputs)
    def model_config(self): return {"visual_encoder_type":"vit","use_bilstm":False,"use_local_window_grouping":False,"vit_input_height":self.input_height,"vit_layers":self.vit_layers,"vit_heads":self.vit_heads,"vit_mlp_dim":self.vit_mlp_dim,"vit_dropout":self.vit_dropout,"vit_max_tokens":self.vit_max_tokens,"vit_position_base_tokens":self.vit_position_base_tokens}
ViTEmbeddingModel=EmbeddingModel
def build_vit_from_environment(*,window_size,stride,vector_size,device,use_flip): return EmbeddingModel(window_size=window_size,stride=stride,vector_size=vector_size,device=device,use_flip=use_flip)
def prepare_vit_model(model):
    global window_ink_ratio_from_patches
    from training_optimizations import fast_window_ink_ratio_from_patches
    window_ink_ratio_from_patches=fast_window_ink_ratio_from_patches
    if _env_flag("TORCH_COMPILE_VISUAL",False) and hasattr(torch,"compile"):
        try:model.vit_encoder=torch.compile(model.vit_encoder,mode=os.environ.get("TORCH_COMPILE_MODE","reduce-overhead"),dynamic=False)
        except Exception as exc:print(f"torch.compile visual ViT failed: {exc}",flush=True)
    return model
