"""Offline fusion acceptance checks through the real compact ResNet/DTW path."""
import argparse
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from architecture_experiment import (
    LinearContextFusion, SumContextFusion, add_fusion_arguments, build_fusion,
    gate_statistics, metadata, resolve_fusion,
)
from embeddingModel import EmbeddingModel
from restoration_recommended_components import LocalContextFusion
from textEmbedding import OrthogonalCharEmbedding
from vlm_restoration_positive_dtw import (
    attach_restoration_dtw_stages, positive_letter_dtw_loss, strong_sigreg_loss,
)
from test_compact_experiment import configuration

torch.set_num_threads(2)
MODES = [("concat", 0), ("sum", 0), ("sum", 1)]


@pytest.fixture(autouse=True)
def runtime_flags(monkeypatch):
    monkeypatch.setenv("PACK_VALID_WINDOWS", "0")
    monkeypatch.setenv("FULL_IMAGE_NO_PADDING", "1")
    monkeypatch.setenv("FUSION_DROPOUT", "0")


def make_model(mode, gated):
    p = configuration()
    p.fusion_mode, p.use_gated_fusion, p.gate_diagnostics = mode, gated, True
    model = EmbeddingModel(vector_size=128, device="cpu", use_flip=True,
        vit_layers=5, vit_heads=1, vit_mlp_dim=512, vit_dropout=0., vit_binarize_input=False)
    return attach_restoration_dtw_stages(model, p), OrthogonalCharEmbedding(128, 4096, 1234), p


@pytest.mark.parametrize("compact,dim", [(True, 128), (False, 192)])
def test_concat_is_exact_original_graph_rng_state_and_output(compact, dim):
    torch.manual_seed(103)
    original = LinearContextFusion() if compact else LocalContextFusion(dim)
    old_rng = torch.get_rng_state().clone()
    torch.manual_seed(103)
    new = build_fusion(SimpleNamespace(), dim, compact=compact)
    assert torch.equal(old_rng, torch.get_rng_state())
    assert type(original) is type(new)
    assert list(original.state_dict()) == list(new.state_dict())
    assert all(torch.equal(value, new.state_dict()[key]) for key, value in original.state_dict().items())
    local, context = torch.randn(2, 7, dim), torch.randn(2, 7, dim)
    assert torch.equal(original(local, context), new(local, context))


@pytest.mark.parametrize("gated", [False, True])
def test_exact_sum_equation_local_residual_and_no_window_mixing(gated):
    head = SumContextFusion(128, gated, capture_gate=True)
    local, context = torch.randn(2, 9, 128), torch.randn(2, 9, 128)
    actual = head(local, context)
    alpha = head.gate(torch.cat((local, context), -1)) if gated else 1
    assert torch.equal(actual, head.norm(local + alpha * context))
    assert not any("projection" in key for key in head.state_dict())
    changed = context.clone()
    changed[:, 3] += torch.randn(2, 128)
    other = head(local, changed)
    untouched = [0, 1, 2, 4, 5, 6, 7, 8]
    assert torch.equal(actual[:, untouched], other[:, untouched])
    assert not torch.equal(actual[:, 3], other[:, 3])
    if gated:
        assert sum(p.numel() for p in head.gate.parameters()) == 49408
        assert head.last_gate.shape == (2, 9, 128)
        assert not head.last_gate.requires_grad
        assert ((head.last_gate >= 0) & (head.last_gate <= 1)).all()
        with torch.no_grad():
            for p in head.gate.parameters():
                p.zero_()
            head.gate[2].bias.fill_(-100)
        assert torch.equal(head(local, context), head.norm(local))
        with torch.no_grad():
            head.gate[2].bias.fill_(100)
        assert torch.equal(head(local, context), head.norm(local + context))


def test_diagnostics_bins_padding_and_opt_in():
    alpha = torch.tensor([[[0., .25, .5, .75], [1., 1., 1., 1.]]])
    stats = gate_statistics(alpha, torch.tensor([[True, False]]))
    assert stats["mean"] == .375 and stats["min"] == 0 and stats["max"] == .75
    assert all(stats[key] == 25 for key in stats if key.startswith("pct_"))
    assert gate_statistics(alpha, torch.zeros(1, 2, dtype=torch.bool)) == {}
    off = SumContextFusion(4, gated=True)
    on = SumContextFusion(4, gated=True, capture_gate=True)
    on.load_state_dict(off.state_dict(), strict=True)
    x = torch.randn(1, 2, 4)
    assert torch.equal(off(x, x), on(x, x))
    assert off.last_gate is None and on.last_gate is not None


def test_cli_defaults_validation_and_historical_metadata():
    parser = argparse.ArgumentParser()
    add_fusion_arguments(parser)
    assert resolve_fusion(parser.parse_args([])) == ("concat", False)
    for mode, gate in MODES:
        args = parser.parse_args(["--fusion-mode", mode, "--use-gated-fusion", str(gate)])
        assert resolve_fusion(args) == (mode, bool(gate))
    for old in ("single-linear-final-layernorm", "concat_projection_norm"):
        assert resolve_fusion({"fusion_mode": old}) == ("concat", False)
    with pytest.raises(ValueError, match="requires fusion_mode=sum"):
        resolve_fusion({"fusion_mode": "concat", "use_gated_fusion": 1})
    with pytest.raises(ValueError, match="must be 0 or 1"):
        resolve_fusion({"fusion_mode": "sum", "use_gated_fusion": "0"})
    with pytest.raises(SystemExit):
        parser.parse_args(["--use-gated-fusion", "2"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--fusion-mode", "attention"])
    with pytest.raises(ValueError, match="equal shape"):
        SumContextFusion(128)(torch.zeros(1, 5, 128), torch.zeros(1, 4, 128))


@pytest.mark.parametrize("sigreg_weight", [0., .2])
@pytest.mark.parametrize("mode,gated", MODES)
def test_full_model_forward_backward(mode, gated, sigreg_weight):
    torch.manual_seed(2718)
    model, text, config = make_model(mode, gated)
    config.sigreg_weight = sigreg_weight
    model.vit_encoder._position_tokens = lambda *a: pytest.fail("Added positional encoding")
    bundle = model(torch.randn(2, 1, 128, 1024), return_training_bundle=True)
    for key in ("local_raw", "contextual_raw", "fused_pre_l2", "semantic"):
        assert bundle[key].shape == (2, 63, 128)
        assert torch.isfinite(bundle[key]).all()
    assert bundle["token_valid"].all()
    assert bundle["physical_window_indices"][0].tolist() == list(reversed(range(63)))
    assert torch.allclose(bundle["semantic"].norm(dim=-1), torch.ones(2, 63), atol=1e-6)
    head = model.vit_encoder.fusion_head
    assert torch.equal(bundle["fused_pre_l2"], head(bundle["local_raw"], bundle["contextual_raw"]))
    assert torch.equal(bundle["semantic"], F.normalize(bundle["fused_pre_l2"], dim=-1))
    dtw, _ = positive_letter_dtw_loss(config, text, bundle["semantic"], bundle["token_valid"], ["سلام", "كتاب"])
    sig, _ = strong_sigreg_loss(bundle["fused_pre_l2"], bundle["token_valid"],
                              sketch_dim=16, distributed_statistics=False)
    loss = dtw + config.sigreg_weight * sig
    assert torch.isfinite(loss)
    loss.backward()
    groups = {"resnet": model.vit_encoder.patch_embedding.backbone,
              "local_projection": model.vit_encoder.patch_embedding.projection,
              "context": model.vit_encoder.encoder, "fusion": head}
    if gated:
        groups["gate_mlp"] = head.gate
        assert (head.last_gate >= 0).all() and (head.last_gate <= 1).all()
        assert torch.equal(model._gradient_probe_records[0]["gate_alpha"], head.last_gate)
    gradients = {}
    for name, module in groups.items():
        params = [p for p in module.parameters() if p.requires_grad]
        assert params and all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        gradients[name] = torch.stack([p.grad.float().square().sum() for p in params]).sum().sqrt().item()
        assert gradients[name] > 0
    assert all(p.grad is None for p in text.parameters())
    print("FUSION_SMOKE " + json.dumps(dict(mode=mode, gated=gated,
        shape=list(bundle["semantic"].shape), dtw=float(dtw.detach()), sigreg=float(sig.detach()),
        sigreg_weight=sigreg_weight,
        loss=float(loss.detach()), gradients=gradients, parameters=sum(p.numel() for p in model.parameters()),
        gate_stats=gate_statistics(getattr(head, "last_gate", None), bundle["token_valid"]))))


@pytest.mark.parametrize("mode,gated", MODES)
def test_strict_evaluation_checkpoint_roundtrip_and_mismatch(tmp_path, monkeypatch, mode, gated):
    from Evaluation._eval_utils import load_evaluation_models
    model, text, config = make_model(mode, gated)
    saved = dict(metadata(config), architecture_family="restoration-positive-dtw-window-encoder",
        model_backend_variant="resnet_token", visual_input_channels=1, visual_grayscale=True,
        window_size=32, stride=16, line_geometry_mode="xml-bbox-gray-full-resize",
        zero_shot_preserve_aspect=False, line_height=128, line_width=1024,
        full_image_no_padding=True, pack_valid_windows=False, vit_max_tokens=model.vit_max_tokens,
        text_encoder_type="char", letter_codebook_seed=1234,
        letter_codebook="frozen-orthogonal-character-identities", letter_codebook_vocab_size=4096,
        lang="Arabic", vit_binarize_input=False)
    payload = dict(model_config=saved, model_state_dict=model.state_dict(), text_encoder_state_dict=text.state_dict())
    path = tmp_path / "model.pth"
    torch.save(payload, path)
    monkeypatch.setenv("FUSION_MODE", "wrong")  # evaluation must use metadata, not shell
    monkeypatch.setenv("USE_GATED_FUSION", str(1-gated))
    loaded = load_evaluation_models(path, "cpu", load_text_model=True)
    assert loaded.config["fusion_mode"] == mode and loaded.config["use_gated_fusion"] == bool(gated)
    assert loaded.config["gate_parameter_count"] == (49408 if gated else 0)
    image = torch.randn(1, 1, 128, 128)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(image), loaded.image_model(image))
        assert torch.equal(loaded.image_model(image), loaded.image_model(image))
    if mode == "concat":
        for historical_mode in (None, "single-linear-final-layernorm"):
            payload["model_config"].pop("use_gated_fusion", None)
            payload["model_config"].pop("fusion_mode", None)
            if historical_mode:
                payload["model_config"]["fusion_mode"] = historical_mode
            torch.save(payload, path)
            historical = load_evaluation_models(path, "cpu")
            with torch.no_grad():
                assert torch.equal(model(image), historical.image_model(image))
    payload["model_config"]["fusion_mode"] = "sum"
    payload["model_config"]["use_gated_fusion"] = 1-gated
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="state_dict"):
        load_evaluation_models(path, "cpu")


@pytest.mark.parametrize("mode,gated", MODES)
def test_packed_rtl_correspondence_and_padding(mode, gated, monkeypatch):
    model, _, _ = make_model(mode, gated)
    model.eval()
    monkeypatch.setenv("PACK_VALID_WINDOWS", "1")
    monkeypatch.setenv("FULL_IMAGE_NO_PADDING", "0")
    valid = torch.tensor([[0, 1, 1, 1, 0, 0, 0], [0, 0, 1, 1, 1, 1, 0]], dtype=torch.bool)
    with patch("vlm_restoration_positive_dtw.line_padding_masks", return_value=(valid, None)), torch.no_grad():
        result = model(torch.randn(2, 1, 128, 128), return_training_bundle=True)
    assert result["semantic"].shape == (2, 4, 128)
    assert result["physical_window_indices"].tolist() == [[3, 2, 1, -1], [5, 4, 3, 2]]
    assert result["token_valid"].sum(1).tolist() == [3, 4]
    assert torch.isfinite(result["semantic"]).all()


@pytest.mark.parametrize("mode,gated", MODES)
def test_sbatch_forwards_settings_without_training(mode, gated):
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ, FUSION_MODE=mode, USE_GATED_FUSION=str(gated),
               SLURM_JOB_ID="test_no_submit", SLURM_SUBMIT_DIR=str(root),
               PYTHON_BIN="/bin/echo", DATASET="fixture_not_opened")
    result = subprocess.run(["bash", "scripts/train.sbatch", "--gate-diagnostics", "1"],
                            cwd=root, env=env, capture_output=True, text=True, check=True)
    assert "--nproc_per_node=2 train.py" in result.stdout
    assert f"--fusion-mode {mode} --use-gated-fusion {gated}" in result.stdout
    assert "--gate-diagnostics 1" in result.stdout
