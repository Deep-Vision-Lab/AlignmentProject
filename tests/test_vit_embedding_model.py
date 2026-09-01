import torch
from Evaluation.vit_evaluation import install_vit_evaluation_loader
from embeddingModel import EmbeddingModel

def _model(use_flip=False):
    return EmbeddingModel(window_size=32,stride=16,vector_size=64,device="cpu",use_flip=use_flip,input_height=128,vit_layers=2,vit_heads=4,vit_mlp_dim=128,vit_dropout=0.0,vit_max_tokens=128)

def test_vit_emits_one_token_per_existing_sliding_window():
    model=_model().eval(); image=torch.randn(1,3,128,1024)
    with torch.no_grad(): contextual,local,grouped,ink=model(image,return_local=True,return_grouped=True,return_ink=True)
    assert contextual.shape==(1,63,64); assert local.shape==(1,63,64); assert grouped.shape==(1,63,64); assert ink.shape==(1,63); torch.testing.assert_close(grouped,local)

def test_embedding_model_is_the_pure_vit_model():
    model=_model(); assert model.visual_encoder_type=="vit"; assert model.use_bilstm is False; assert model.use_local_grouping is False; assert hasattr(model,"vit_encoder"); assert not hasattr(model,"cnn_encoder"); assert not any(isinstance(module,torch.nn.LSTM) for module in model.modules()); assert model.model_config()["visual_encoder_type"]=="vit"

def test_arabic_flip_reverses_local_window_token_order():
    model=_model(False).eval(); image=torch.randn(1,3,128,256)
    with torch.no_grad():
        _,local_forward=model(image,return_local=True); model._use_flip_state.fill_(1); _,local_flipped=model(image,return_local=True)
    torch.testing.assert_close(local_flipped,torch.flip(local_forward,dims=[1]))

def test_vit_heads_must_divide_embedding_dimension():
    try: EmbeddingModel(vector_size=62,vit_heads=4,device="cpu")
    except ValueError as exc: assert "must divide" in str(exc)
    else: raise AssertionError("Expected invalid ViT head configuration to fail")

def test_evaluation_reconstructs_vit_from_checkpoint_config(tmp_path):
    model=_model(True); checkpoint_path=tmp_path/"vit.pth"; torch.save({"image_model_state_dict":model.state_dict(),"model_config":{"visual_encoder_type":"vit","window_size":32,"stride":16,"vector_size":64,"lang":"Arabic","use_bilstm":False,"use_local_window_grouping":False,"vit_input_height":128,"vit_layers":2,"vit_heads":4,"vit_mlp_dim":128,"vit_dropout":0.0,"vit_max_tokens":128}},checkpoint_path); install_vit_evaluation_loader(); from Evaluation import _eval_utils; loaded=_eval_utils.load_evaluation_models(checkpoint_path,device="cpu",load_text_model=False); assert isinstance(loaded.image_model,EmbeddingModel); assert loaded.image_model.use_flip is True
