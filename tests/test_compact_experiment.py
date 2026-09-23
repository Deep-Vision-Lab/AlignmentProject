"""Executed architecture and deterministic monitoring acceptance tests (offline)."""
import copy
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from architecture_experiment import VARIANT, metadata
from embeddingModel import EmbeddingModel
from epoch_monitoring import evaluate_split, assert_disjoint, clean_view
from textEmbedding import OrthogonalCharEmbedding
from vlm_restoration_positive_dtw import attach_restoration_dtw_stages, positive_letter_dtw_loss, strong_sigreg_loss

torch.set_num_threads(2)


def configuration():
    return SimpleNamespace(architecture_variant=VARIANT, local_dropout=.1,
        resnet18_pretrained=False, tiny_vit_pretrained=False, visual_input_channels=1,
        positive_letter_dtw_gamma=.05, positive_letter_dtw_vertical_penalty=.05,
        positive_letter_dtw_horizontal_penalty=.3, positive_letter_dtw_position_prior=.15,
        positive_letter_dtw_disable_horizontal_when_feasible=True,
        positive_letter_dtw_cost_mode="full_alphabet_nll", positive_letter_dtw_competition_temperature=.1,
        positive_letter_dtw_weight=1., restoration_contrastive_weight=0., num_negatives=0,
        sigreg_weight=.2, sigreg_sketch_dim=16, sigreg_num_knots=5, sigreg_t_min=0.,
        sigreg_t_max=3., sigreg_min_samples=2, sigreg_slice_chunk=8)


def model_and_text():
    p = configuration()
    model = EmbeddingModel(vector_size=128, device="cpu", use_flip=True,
        vit_layers=5, vit_heads=1, vit_mlp_dim=512, vit_dropout=0., vit_binarize_input=False)
    model = attach_restoration_dtw_stages(model, p)
    text = OrthogonalCharEmbedding(128, 4096, 1234)
    return model, text, p


@pytest.fixture(autouse=True)
def flags(monkeypatch):
    monkeypatch.setenv("PACK_VALID_WINDOWS", "0")
    monkeypatch.setenv("FULL_IMAGE_NO_PADDING", "1")


def test_executed_dimensions_gradients_and_same_local_draw():
    model, text, p = model_and_text()
    seen = {}
    model.vit_encoder.encoder.register_forward_pre_hook(lambda _, args: seen.update(context_input=args[0]))
    model.vit_encoder.fusion_head.register_forward_pre_hook(lambda _, args: seen.update(fusion_local=args[0], fusion_context=args[1]))
    model.vit_encoder._position_tokens = lambda *a: pytest.fail("Position tokens executed")
    bundle = model(torch.randn(1, 1, 128, 1024), return_training_bundle=True)
    assert bundle["local_raw"] is seen["context_input"] is seen["fusion_local"]
    assert bundle["contextual_raw"] is seen["fusion_context"]
    assert bundle["semantic"].shape == (1, 63, 128)
    assert bundle["token_valid"].sum() == 63
    assert bundle["physical_window_indices"][0,[0,1,2,60,61,62]].tolist() == [62,61,60,2,1,0]
    assert isinstance(model.vit_encoder.local_norm, nn.Identity)
    assert isinstance(model.vision_norm, nn.Identity)
    assert model.vit_encoder.fusion_head.projection.weight.shape == (128, 256)
    assert model.vit_encoder.patch_embedding.projection[0].weight.shape == (128, 512)
    assert isinstance(model.vit_encoder.patch_embedding.projection[1], nn.Dropout)
    assert model.vit_encoder.position_embedding is None
    assert not torch.equal(model.vit_encoder.encoder.layers[0].linear1.weight, model.vit_encoder.encoder.layers[1].linear1.weight)
    assert all(not p.requires_grad for p in text.parameters())
    assert torch.allclose(bundle["fused_pre_l2"].norm(dim=-1).mean(), torch.tensor(128.).sqrt(), atol=.02)
    assert torch.allclose(bundle["semantic"].norm(dim=-1), torch.ones(1, 63), atol=1e-6)
    dtw, _ = positive_letter_dtw_loss(p, text, bundle["semantic"], bundle["token_valid"], ["سلام"])
    sig, _ = strong_sigreg_loss(bundle["fused_pre_l2"], bundle["token_valid"], sketch_dim=16, distributed_statistics=False)
    assert torch.isfinite(dtw + sig)
    (dtw + .2*sig).backward()
    for parameter in (model.vit_encoder.patch_embedding.backbone.conv1.weight,
                      model.vit_encoder.patch_embedding.projection[0].weight,
                      model.vit_encoder.encoder.layers[0].linear1.weight,
                      model.vit_encoder.fusion_head.projection.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_dropout_permutation_equivariance_eval_repeat_and_rtl():
    model, _, _ = model_and_text()
    projection = model.vit_encoder.patch_embedding.projection
    r = torch.randn(2, 7, 512)
    assert projection(r).shape == (2, 7, 128)
    assert not torch.equal(projection(r), projection(r))
    model.eval()
    assert torch.equal(projection(r), projection[0](r))
    x = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        first = model(x, return_training_bundle=True)
        second = model(x, return_training_bundle=True)
        assert torch.equal(first["semantic"], second["semantic"])
        physical = model.vit_encoder.patch_embedding(x).squeeze(2).transpose(1, 2)
        assert torch.equal(first["local_raw"], physical.flip(1))
        local = first["local_raw"]
        perm = torch.tensor([2, 4, 0, 6, 3, 5, 1])
        encoder = model.vit_encoder.encoder
        assert torch.allclose(encoder(local[:, perm]), encoder(local)[:, perm], atol=2e-6)


def test_packed_encoder_keeps_shape_mask_and_rtl():
    model, _, _ = model_and_text()
    encoder = model.vit_encoder.patch_embedding.eval()
    images = torch.randn(2, 1, 128, 128)
    valid = torch.tensor([[0,1,1,1,0,0,0], [0,0,1,1,1,1,0]], dtype=torch.bool)
    with torch.no_grad():
        full = encoder(images).squeeze(2).transpose(1, 2)
        packed, mask = encoder.forward_packed(images, valid, use_flip=True)
    assert packed.shape == (2,128,1,4)
    assert mask.sum(1).tolist() == [3,4]
    assert torch.allclose(packed[0,:,0,:3].T, full[0,valid[0]].flip(0), atol=1e-6)


def test_evaluation_checkpoint_roundtrip_and_incompatible_metadata(tmp_path):
    from Evaluation._eval_utils import load_evaluation_models
    model, text, p = model_and_text()
    config = dict(metadata(p), architecture_family="restoration-positive-dtw-window-encoder",
        model_backend_variant="resnet_token", model_backend="resnet18_tinyvit_positive_dtw",
        visual_input_channels=1, visual_grayscale=True, window_size=32, stride=16,
        line_geometry_mode="xml-bbox-gray-full-resize", zero_shot_preserve_aspect=False,
        line_height=128,line_width=1024, full_image_no_padding=True, pack_valid_windows=False,
        vit_max_tokens=model.vit_max_tokens, text_encoder_type="char", letter_codebook_seed=1234,
        letter_codebook="frozen-orthogonal-character-identities",
        letter_codebook_vocab_size=4096, lang="Arabic", vit_binarize_input=False)
    path = tmp_path / "model.pth"
    payload = dict(model_config=config, model_state_dict=model.state_dict(),text_encoder_state_dict=text.state_dict())
    torch.save(payload, path)
    loaded = load_evaluation_models(path, "cpu", load_text_model=True)
    model.eval()
    with patch.dict(os.environ, {"PACK_VALID_WINDOWS":"1", "FULL_IMAGE_NO_PADDING":"0"}), torch.no_grad():
        x = torch.randn(1,1,128,1024)
        b = loaded.image_model(x, return_training_bundle=True)
        assert b["semantic"].shape == (1,63,128) and b["token_valid"].sum() == 63
    with torch.no_grad():
        assert torch.equal(model(x), b["semantic"])
    assert all(not p.requires_grad for p in loaded.text_model.parameters())
    payload["model_config"]["vit_layers"] = 12
    torch.save(payload, path)
    with pytest.raises(ValueError, match="requires"):
        load_evaluation_models(path, "cpu")


class Lines(Dataset):
    def __init__(self):
        self.images = torch.randn(5,1,128,64)
        self.texts = ["سلام", "", "ب", "كتاب", "ت"]
    def __len__(self): return 5
    def __getitem__(self, i): return self.images[i], self.texts[i], []


def collate(items):
    x, t, n = zip(*items)
    return torch.stack(x), list(t), list(n)


def test_monitor_full_membership_unequal_batches_no_mutation_rng_or_collectives():
    model, text, p = model_and_text()
    dataset = Lines()
    before = {k:v.clone() for k,v in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    loader = DataLoader(dataset,batch_size=2,collate_fn=collate)
    with patch("torch.distributed.is_initialized", return_value=True), patch("torch.distributed.get_world_size", return_value=2), patch("torch.distributed.get_rank", return_value=0), patch("torch.distributed.all_reduce", side_effect=AssertionError("collective")):
        result = evaluate_split(model,text,loader,p,seed=17,device="cpu")
    assert result["evaluated"] == 4 and result["skipped"] == 1 and result["invalid"] == 0
    assert torch.equal(rng,torch.get_rng_state()) and model.training
    assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
    # Per-line DTW aggregation must be invariant to an unequal final batch.
    other = evaluate_split(model,text,DataLoader(dataset,batch_size=1,collate_fn=collate),p,seed=17,device="cpu")
    assert result["positive_dtw"] == pytest.approx(other["positive_dtw"], abs=1e-6)


def test_leakage_fails_not_reshuffles():
    row = dict(record_id="a",pair_id="page",_root="/tmp",line_image_path="a.png")
    with pytest.raises(ValueError,match="leakage"):
        assert_disjoint({"train":[row],"valid":[dict(row,record_id="b")]})
    with pytest.raises(ValueError,match="leakage"):
        assert_disjoint({"train":[row],"valid":[dict(row,record_id="b",pair_id="other",line_image_path="aug.png",augmentation_parent_id="a")]})


def test_repeat_and_scan_wrapper_are_not_evaluation_population():
    from AugmentedRealDataLoader import RepeatToLengthDataset, AllPageLinesScanAugmentedSubset
    from torch.utils.data import Subset
    ds = Lines()
    view = clean_view(RepeatToLengthDataset(AllPageLinesScanAugmentedSubset(ds,[0,3,4],object()),9))
    assert isinstance(view,Subset) and list(view.indices)==[0,3,4] and len(view)==3


def test_monitor_defaults_to_full_saved_membership_not_sampler(tmp_path, monkeypatch):
    from epoch_monitoring import EpochMonitor
    from torch.utils.data import Subset
    ds = Lines()
    ds.samples = [dict(line_image_path=f"line{i}.png",text_path=f"line{i}.txt",pair_id=f"page{i}") for i in range(5)]
    ds.root = tmp_path
    p = configuration()
    p.positive_letter_dtw_gamma_end = .05
    module = SimpleNamespace(P=p,CTX=SimpleNamespace(is_main=True,world_size=1))
    monkeypatch.setenv("MONITOR_OUTPUT",str(tmp_path/"monitor"))
    monkeypatch.setenv("MONITOR_MAX_RECORDS","0")
    train = DataLoader(Subset(ds,[0,1,2]),batch_size=2,sampler=[0],collate_fn=collate)
    valid = DataLoader(Subset(ds,[3,4]),batch_size=2,collate_fn=collate)
    config = {}
    monitor = EpochMonitor(module,train,valid,config,"fixture")
    assert len(monitor.loaders["train_eval"].dataset) == 3
    assert len(monitor.loaders["val_eval"].dataset) == 2
    assert config["monitoring"]["population"] == "full-split"
    assert len(list(monitor.loaders["train_eval"].sampler)) == 3
    with pytest.raises(RuntimeError,match="monitoring failed"):
        EpochMonitor(module,train,valid,{},"fixture")


def test_packed_executed_wrapper_preserves_original_physical_indices(monkeypatch):
    model, _, _ = model_and_text()
    model.eval()
    monkeypatch.setenv("PACK_VALID_WINDOWS","1")
    monkeypatch.setenv("FULL_IMAGE_NO_PADDING","0")
    valid = torch.tensor([[0,1,1,1,0,0,0],[0,0,1,1,1,1,0]],dtype=torch.bool)
    with patch("vlm_restoration_positive_dtw.line_padding_masks",return_value=(valid,None)), torch.no_grad():
        result = model(torch.randn(2,1,128,128), return_training_bundle=True)
    assert result["semantic"].shape == (2,4,128)
    assert result["physical_window_indices"].tolist() == [[3,2,1,-1],[5,4,3,2]]
    assert result["token_valid"].sum(1).tolist() == [3,4]


def test_zero_local_dropout_is_supported_and_fused_norm_is_applied_once():
    model, _, _ = model_and_text()
    projection = model.vit_encoder.patch_embedding.projection
    projection[1].p = 0.
    x = torch.randn(5,512)
    assert torch.equal(projection(x),projection[0](x))
    bundle = model(torch.randn(1,1,128,64),return_training_bundle=True)
    fusion = model.vit_encoder.fusion_head
    expected = fusion.norm(fusion.projection(torch.cat((bundle["local_raw"],bundle["contextual_raw"]),dim=-1)))
    assert torch.equal(bundle["fused_pre_l2"], expected)


def test_nonzero_transformer_dropout_eval_is_still_deterministic():
    model, _, _ = model_and_text()
    for layer in model.vit_encoder.encoder.layers:
        layer.self_attn.dropout = .2
    model.eval()
    with torch.no_grad():
        x = torch.randn(1,1,128,64)
        assert torch.equal(model(x), model(x))


def test_source_space_support_uses_resized_model_windows():
    from Evaluation.point3_spatial_metrics import _region_matches
    # Five model windows map to widths 320 / stride 160 in source pixels.
    windows = [(i*160,(i*160)+320) for i in range(5)]
    m = _region_matches([(0,320)],[(0,320)],10240,5,.5,physical_windows=windows)
    assert m[0]["consecutive_window_support"] == 2
    assert not m[0]["success"]  # A spurious source-pixel 32/16 grid would pass.


def test_xml_fragment_gt_requires_exact_full_transcript_identity(tmp_path):
    from PIL import Image
    from monitoring_spatial import score_xml_transcript
    side = tmp_path / "A"
    (side/"linesImages").mkdir(parents=True)
    page = Image.new("L", (400,200), 200)
    page.save(side/"original_image.png")
    page.crop((0,43,400,87)).save(side/"linesImages/line_01.png")
    (side/"original.xml").write_text("<ArrayOfDocumentElement>"
        "<DocumentElement><X>200</X><Y>50</Y><Width>60</Width><Height>30</Height><Transcript>اب</Transcript></DocumentElement>"
        "<DocumentElement><X>100</X><Y>50</Y><Width>60</Width><Height>30</Height><Transcript>ت</Transcript></DocumentElement>"
        "</ArrayOfDocumentElement>")
    record = dict(_root=str(tmp_path),line_image_path="A/linesImages/line_01.png",line_idx=1,record_id="fixture")
    config = dict(line_geometry_mode="xml-bbox-gray-full-resize",visual_grayscale=True,
        visual_input_channels=1,line_height=128,line_width=1024,window_size=32,stride=16,
        zero_shot_foreground_crop=False,zero_shot_preserve_aspect=False,full_image_no_padding=True,
        pack_valid_windows=False,real_bbox_crop=True)
    bundle = dict(semantic=torch.nn.functional.normalize(torch.randn(1,63,128),dim=-1),token_valid=torch.ones(1,63,dtype=torch.bool))
    text = OrthogonalCharEmbedding(128,4096,1234)
    result = score_xml_transcript(record,"ابت",bundle,0,text,configuration(),config)
    assert result["status"] == "available" and len(result["regions"]) == 2
    mismatch = score_xml_transcript(record,"ابتب",bundle,0,text,configuration(),config)
    assert mismatch["status"] == "unavailable" and "match" in mismatch["reason"]


def _gloo_monitor_worker(rank, init_file, result_path):
    from datetime import timedelta
    from torch.nn.parallel import DistributedDataParallel
    from epoch_monitoring import broadcast_monitor_result
    torch.distributed.init_process_group("gloo", init_method="file://"+init_file,
        rank=rank,world_size=2,timeout=timedelta(seconds=60))
    try:
        model, text, p = model_and_text()
        ddp = DistributedDataParallel(model, broadcast_buffers=False)
        payload = [None]
        if rank == 0:
            with patch("torch.distributed.all_reduce", side_effect=AssertionError("evaluation collective")):
                payload[0] = evaluate_split(ddp.module,text,
                    DataLoader(Lines(),batch_size=2,collate_fn=collate),p,seed=17,device="cpu")
        result = broadcast_monitor_result(payload)
        assert result["evaluated"] == 4 and result["skipped"] == 1
        failure = [{"error":"intentional diagnostic failure"} if rank == 0 else None]
        with pytest.raises(RuntimeError,match="intentional"):
            broadcast_monitor_result(failure)
        if rank == 0:
            torch.save(result,result_path)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(os.environ.get("RUN_GLOO_MONITOR_TEST") != "1", reason="Explicit local two-process Gloo test")
def test_two_rank_monitor_handoff_uneven_batches_and_error_broadcast(tmp_path):
    import torch.multiprocessing as mp
    mp.spawn(_gloo_monitor_worker,args=(str(tmp_path/"gloo"),str(tmp_path/"result.pth")),nprocs=2,join=True)
    assert torch.load(tmp_path/"result.pth")["evaluated"] == 4
