"""Real CPU inference, checkpoint fidelity, held-out splits, and mask geometry."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch
import torch.nn.functional as F

from embeddingModel import EmbeddingModel
from Evaluation import eval_img_align_nw_diagnostic as base
from Evaluation import yelda_runtime as runtime
from Evaluation.eval_yelda import configure_geometry, main, synthetic_split
from Evaluation.yelda_geometry import prepare_line, source_intervals

torch.set_num_threads(1)


def make_checkpoint(cross=False):
    config = dict(letter_depiction_enabled=True, visual_encoder_type="vit", window_size=32,
                  stride=16, vector_size=128, lang="Arabic", vit_layers=4, vit_heads=4,
                  vit_mlp_dim=512, vit_dropout=0.1, vit_input_height=128,
                  vit_max_tokens=256, vit_position_base_tokens=63, vit_binarize_input=False,
                  line_height=128, line_width=1024, target_ink_height_ratio=0.72,
                  cross_attention_enabled=cross)
    model = EmbeddingModel(device="cpu", use_flip=True,
                           vit_layers=4, vit_heads=4, vit_mlp_dim=512, vit_dropout=0.1,
                           vit_max_tokens=256, vit_position_base_tokens=63,
                           vit_binarize_input=False).eval()
    from vlm_letter_grounding import attach_depiction_head
    model = attach_depiction_head(model).eval()
    checkpoint = {"model_config": config, "model_state_dict": model.state_dict(), "epoch": 3}
    pair = None
    if cross:
        module = pytest.importorskip("vlm_pair_cross_attention")
        pair = module.SymmetricPairCrossAttention(128).eval()
        checkpoint["text_encoder_state_dict"] = {
            "pair_cross_attention." + key: value for key, value in pair.state_dict().items()
        }
        # There are deliberately no AraBERT weights/tokenizer files.
    return checkpoint, model, pair


@pytest.mark.parametrize("cross", [False, True])
def test_exact_checkpoint_forward_and_pair_fusion(tmp_path, cross):
    checkpoint, original, pair = make_checkpoint(cross)
    path = tmp_path / "model.pth"
    torch.save(checkpoint, path)
    restored = runtime.read_checkpoint(path)
    models = runtime.load_visual_models(restored, "cpu")
    assert models.text_model is None
    assert len(restored["_evaluation_sha256"]) == 64
    x = torch.randn(1, 3, 128, 1024)
    with torch.no_grad():
        expected = original(x)
        actual = models.image_model(x)
    torch.testing.assert_close(actual, expected)
    a, b = tmp_path / "a.png", tmp_path / "b.png"
    Image.fromarray(np.random.default_rng(42).integers(0, 256, (128, 1024, 3), dtype=np.uint8)).save(a)
    Image.fromarray(np.random.default_rng(43).integers(0, 256, (128, 1024, 3), dtype=np.uint8)).save(b)
    independent = runtime.pair_features(models, a, b, "independent")
    fused = runtime.pair_features(models, a, b)
    joint = runtime.pair_features(models, a, b, "joint")
    torch.testing.assert_close(joint[0].contextual, fused[0].contextual)
    torch.testing.assert_close(joint[0].local, independent[0].local)
    assert fused[0].contextual.shape == (63, 128)
    torch.testing.assert_close(fused[0].local, independent[0].local)
    if cross:
        with torch.no_grad():
            f1, f2, _, _ = pair(independent[0].contextual[None], independent[1].contextual[None],
                                ink1=independent[0].ink[None], ink2=independent[1].ink[None], return_weights=False)
        torch.testing.assert_close(fused[0].contextual, F.normalize(f1[0], dim=-1))
        torch.testing.assert_close(fused[1].contextual, F.normalize(f2[0], dim=-1))
    else:
        torch.testing.assert_close(fused[0].contextual, independent[0].contextual)
    with pytest.raises(ValueError):
        runtime.load_visual_models(restored, "cpu", "hierarchy" if cross else "cross")


def test_missing_pair_weights_rejected():
    checkpoint, _, _ = make_checkpoint(True)
    checkpoint["text_encoder_state_dict"].pop("pair_cross_attention.gate_logit")
    with pytest.raises(RuntimeError, match="gate_logit"):
        runtime.load_visual_models(checkpoint, "cpu")


def test_missing_patch_projection_weights_rejected():
    checkpoint, _, _ = make_checkpoint()
    key = next(key for key in checkpoint["model_state_dict"] if "patch_embedding.weight" in key)
    del checkpoint["model_state_dict"][key]
    with pytest.raises(RuntimeError, match="Missing key"):
        runtime.load_visual_models(checkpoint, "cpu")


def test_synthetic_split_matches_training_and_excludes_unused_source():
    pairs = [base.Pair(i, Path(f"a{i}"), Path(f"b{i}"), "synthetic", "synthetic", "synthetic", manifest_position=i)
             for i in range(1, 10001)]
    expected = torch.utils.data.random_split(range(6000), [3600, 1200, 1200],
                                            generator=torch.Generator().manual_seed(42))
    actual = [synthetic_split(pairs, name, 6000, 42) for name in ("train", "valid", "test")]
    for subset, reference in zip(actual, expected):
        assert [p.manifest_position for p in subset] == [i + 1 for i in reference.indices]
    assert set(p.manifest_position for p in actual[0]).isdisjoint(p.manifest_position for p in actual[2])
    with pytest.raises(ValueError, match="missing synthetic IDs"):
        synthetic_split(pairs[1:], "test", 6000, 42)


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width", [180, 1800])
def test_geometry_maps_foreground_back_to_source(tmp_path, dark, width):
    configure_geometry({})
    image = Image.new("RGB", (width, 200), "black" if dark else "white")
    rectangle = (width // 4, 70, 3 * width // 4 - 1, 129)
    ImageDraw.Draw(image).rectangle(rectangle, fill="white" if dark else "black")
    path = tmp_path / "line.png"
    image.save(path)
    prepared, mapping = prepare_line(path, "synthetic")
    ink = np.asarray(prepared.convert("L")) < 128
    cols = np.nonzero(ink.any(axis=0))[0]
    inverse = source_intervals([[int(cols.min()), int(cols.max()) + 1]], mapping)[0]
    assert abs(inverse[0] - width // 4) < 3
    assert abs(inverse[1] - 3 * width // 4) < 3
    assert prepared.size == (1024, 128)


@pytest.mark.parametrize("cross", [False, True])
def test_end_to_end_image_only_report(tmp_path, monkeypatch, cross):
    checkpoint, _, _ = make_checkpoint(cross)
    weights = tmp_path / "model.pth"
    torch.save(checkpoint, weights)
    dataset = tmp_path / "data"
    (dataset / "images").mkdir(parents=True)
    (dataset / "masks").mkdir()
    for role in (1, 2):
        image = Image.new("RGB", (1024, 128), "white")
        ImageDraw.Draw(image).rectangle((200, 35, 800, 95), fill="black")
        image.save(dataset / "images" / f"img{role}_1.png")
        mask = Image.new("L", (1024, 128), 0)
        ImageDraw.Draw(mask).rectangle((200, 0, 800, 127), fill=255)
        mask.save(dataset / "masks" / f"mask{role}_1.png")
    # Rendering is separately smoke-tested; keep this regression fast.
    monkeypatch.setattr(base, "_visualize", lambda *a, **kw: None)
    output = tmp_path / "evaluation"
    assert main(["--dataset", str(dataset), "--weights", str(weights), "--output-dir", str(output),
                 "--split", "all", "--device", "cpu"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["successful"] == 1 and summary["failed"] == 0
    assert summary["text_encoder_loaded"] is False
    assert summary["feature_stage"] == ("joint_local_fused_contextual" if cross else "joint_local_contextual")
    assert summary["line1_gt_count"] == 1
    evidence = np.load(output / "pair_00001" / "cosine_similarity.npy")
    np.testing.assert_allclose(np.diag(evidence), 1, atol=1e-5)
    pair_dir = output / "pair_00001"
    local = np.load(pair_dir / "local_cosine_similarity.npy")
    contextual = np.load(pair_dir / "contextual_cosine_similarity.npy")
    np.testing.assert_allclose(evidence, .5 * local + .5 * contextual, atol=1e-6)
    np.testing.assert_allclose(evidence, np.load(pair_dir / "joint_similarity.npy"))
    assert (output / "pair_00001" / "line1_source_pred_mask.png").exists()
    assert (output / "selected_pairs.json").exists()


def test_failed_second_image_cannot_contaminate_next_pair(tmp_path):
    checkpoint, _, _ = make_checkpoint(True)
    models = runtime.load_visual_models(checkpoint, "cpu")
    image = tmp_path / "image.png"
    Image.new("RGB", (1024, 128), "white").save(image)
    with pytest.raises(FileNotFoundError):
        runtime.pair_features(models, image, tmp_path / "missing.png")
    first, second = runtime.pair_features(models, image, image)
    torch.testing.assert_close(first.contextual, second.contextual)


def test_real_manifest_uses_images_and_reports_missing_gt(tmp_path, monkeypatch):
    checkpoint, _, _ = make_checkpoint()
    weights = tmp_path / "model.pth"
    torch.save(checkpoint, weights)
    dataset = tmp_path / "real"
    dataset.mkdir()
    image = Image.new("RGB", (600, 160), "white")
    ImageDraw.Draw(image).rectangle((100, 40, 499, 100), fill="black")
    image.save(dataset / "line.png")
    (dataset / "text.txt").write_text("unused transcript", encoding="utf-8")
    records = [dict(pair_id=f"page_{i}", label_type="high_match", scores={"text_score": 1},
                    A=dict(line_image_path="line.png", text_original_path="text.txt"),
                    B=dict(line_image_path="line.png", text_original_path="text.txt")) for i in range(6)]
    (dataset / "dataset_manifest.jsonl").write_text("\n".join(json.dumps(row) for row in records))
    monkeypatch.setattr(base, "_visualize", lambda *a, **kw: None)
    output = tmp_path / "real_eval"
    assert main(["--dataset", str(dataset), "--weights", str(weights), "--output-dir", str(output),
                 "--split", "test", "--n-samples", "1", "--device", "cpu"]) == 0
    summary = json.loads((output / "summary.json").read_text())
    assert summary["successful"] == 1 and summary["line1_gt_count"] == 0
    assert summary["mean_mask_iou"] is None
    assert summary["split_policy"] == "balanced_page_pair_groups"


def test_missing_depiction_weights_rejected():
    checkpoint, _, _ = make_checkpoint()
    del checkpoint["model_state_dict"]["vit_encoder.depiction_projection.0.weight"]
    with pytest.raises(RuntimeError, match="depiction_projection"):
        runtime.load_visual_models(checkpoint, "cpu")


def test_window_cnn_checkpoint_rejected(tmp_path):
    checkpoint, _, _ = make_checkpoint()
    checkpoint["model_config"]["window_cnn_enabled"] = True
    path = tmp_path / "incompatible.pth"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="window-CNN branch"):
        runtime.read_checkpoint(path)
