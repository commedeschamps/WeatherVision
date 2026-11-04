"""Real pretrained inference and controlled-protocol regression checks."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from src.siglip2 import (
    ASSET_DIR,
    SiglipDataset,
    load_encoder,
    load_siglip_checkpoint,
    predict_siglip,
    patch_occlusion,
    siglip_input,
    siglip_overlay,
)
from src.utils import ROOT, sha256_file


@pytest.fixture(scope="module")
def pretrained():
    if not (ASSET_DIR / "provenance.json").exists():
        pytest.skip("Official pretrained assets not installed")
    return load_encoder()


def test_processor_and_actual_features(pretrained):
    encoder, processor, provenance = pretrained
    image = Image.new("RGB", (380, 170), (255, 0, 128))
    inputs = processor(images=image, return_tensors="pt")["pixel_values"]
    assert inputs.shape == (1, 3, 224, 224)
    torch.testing.assert_close(
        inputs[0, :, 100, 100],
        torch.tensor([1.0, -1.0, 128 / 127.5 - 1]),
        atol=1e-6,
        rtol=0,
    )
    with torch.inference_mode():
        outputs = encoder(pixel_values=inputs)
    assert outputs.last_hidden_state.shape == (1, 196, 768)
    assert outputs.pooler_output.shape == (1, encoder.config.hidden_size)
    assert all(not p.requires_grad for p in encoder.parameters())
    assert provenance["revision"] == "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"


def test_dynamic_augmentation_is_epoch_seeded(pretrained):
    _, processor, _ = pretrained
    frame = pd.read_csv(ROOT / "data/manifest.csv")
    a = SiglipDataset(frame, processor, "train", augment=True, seed=42, epoch=1)[0][0]
    b = SiglipDataset(frame, processor, "train", augment=True, seed=42, epoch=1)[0][0]
    c = SiglipDataset(frame, processor, "train", augment=True, seed=42, epoch=2)[0][0]
    assert torch.equal(a, b) and not torch.equal(a, c)
    assert torch.equal(
        SiglipDataset(frame, processor, "val", seed=42)[0][0],
        SiglipDataset(frame, processor, "val", seed=44)[0][0],
    )


def test_real_head_roundtrip_and_probability_semantics(pretrained, tmp_path):
    checkpoint_path = ROOT / "outputs/siglip2_aug/best.pt"
    if not checkpoint_path.exists():
        pytest.skip("Trained head required")
    model, checkpoint = load_siglip_checkpoint(
        checkpoint_path, "cpu", encoder_bundle=pretrained
    )
    torch.save(checkpoint, tmp_path / "head.pt")
    restored, _ = load_siglip_checkpoint(
        tmp_path / "head.pt", "cpu", encoder_bundle=pretrained
    )
    image = Image.new("RGB", (320, 180), (90, 140, 190))
    result = predict_siglip(model, image, "cpu")
    again = predict_siglip(restored, image, "cpu")
    assert result["probabilities"] == again["probabilities"] and sum(
        result["probabilities"]
    ) == pytest.approx(1.0, abs=1e-6)
    assert result["score_type"] == "softmax" and result["latency_ms"] > 0
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 4 * (
        checkpoint["feature_dim"] + 1
    )
    zero, _ = load_siglip_checkpoint(
        ROOT / "outputs/siglip2_zero_shot/best.pt", "cpu", encoder_bundle=pretrained
    )
    zero_result = predict_siglip(zero, image, "cpu")
    assert zero_result["score_type"] == "sigmoid_matching"
    assert np.allclose(
        zero_result["probabilities"],
        torch.tensor(zero_result["logits"]).sigmoid().numpy(),
    )


class Corner(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        score = x[:, 0, :32, :32].mean((1, 2)) * self.scale
        return torch.stack([score, -score, score * 0, score * 0.5], 1)


def test_occlusion_exact_fixed_class_determinism_and_geometry():
    model = Corner().eval()
    x = torch.ones(1, 3, 224, 224)
    heatmap, a = patch_occlusion(model, x, 0)
    repeated, b = patch_occlusion(model, x, 0, batch_size=1)
    np.testing.assert_array_equal(heatmap, repeated)
    assert a == b
    assert a["signed_logit_drops"][0][0] == 1
    assert np.count_nonzero(np.array(a["signed_logit_drops"])) == 1
    assert a["target_class"] == 0 and model.scale.grad is None
    image = Image.new("RGB", (380, 170), (90, 140, 190))
    overlay = siglip_overlay(image, heatmap)
    assert overlay.size == image.size
    assert np.array_equal(
        np.asarray(siglip_overlay(image, heatmap, 0)), np.asarray(image)
    )
    for grid in [0, 6, True]:
        with pytest.raises(ValueError):
            patch_occlusion(model, x, 0, grid=grid)


def test_completed_ablation_and_metric_identity():
    directory = ROOT / "outputs/siglip2"
    if not (directory / "per_seed.csv").exists():
        pytest.skip("Completed three-seed protocol required")
    from src.engine import calculate_metrics

    per_seed = pd.read_csv(directory / "per_seed.csv")
    assert len(per_seed) == 6 and set(per_seed.seed) == {42, 43, 44}
    for seed in [42, 43, 44]:
        paths = [
            ROOT
            / "outputs"
            / ("siglip2_" + treatment + ("" if seed == 42 else f"_s{seed}"))
            for treatment in ["aug", "noaug"]
        ]
        statuses = [json.loads((p / "status.json").read_text()) for p in paths]
        assert statuses[0]["protocol"] == statuses[1]["protocol"]
        assert statuses[0]["initial_head_sha256"] == statuses[1]["initial_head_sha256"]
        for p in paths:
            metrics = json.loads((p / "test_metrics.json").read_text())
            pred = pd.read_csv(p / "test_predictions.csv")
            reconstructed = calculate_metrics(
                pred.class_id.tolist(), pred.predicted_class_id.tolist()
            )
            assert metrics["f1_macro"] == reconstructed["f1_macro"]
            assert metrics["checkpoint_sha256"] == sha256_file(p / "best.pt")


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Apple MPS unavailable"
)
def test_mps_cpu_trained_siglip_predictions(pretrained):
    path = ROOT / "outputs/siglip2_aug/best.pt"
    if not path.exists():
        pytest.skip("Trained head required")
    model, _ = load_siglip_checkpoint(path, "cpu", encoder_bundle=pretrained)
    image = Image.new("RGB", (300, 220), (110, 140, 180))
    cpu = predict_siglip(model, image, "cpu")
    model.to("mps")
    mps = predict_siglip(model, image, "mps")
    model.cpu()
    np.testing.assert_allclose(
        cpu["probabilities"], mps["probabilities"], atol=2e-4, rtol=1e-3
    )


def test_completed_run_reuse_and_partial_test_guard(tmp_path):
    from src.siglip_experiments import run_siglip_experiments
    from src.utils import write_json

    protocol = json.loads((ROOT / "configs/siglip2.json").read_text())
    config_path = tmp_path / "config.json"
    write_json(config_path, protocol)
    research = tmp_path / "outputs/siglip2"
    research.mkdir(parents=True)
    (research / "ablation_summary.csv").write_text("augment,accuracy_mean\nTrue,0.98\n")
    write_json(
        research / "selection_lock.json",
        {
            "config": protocol,
            "manifest_sha256": sha256_file(ROOT / "data/manifest.csv"),
        },
    )
    assert (
        len(run_siglip_experiments(config_path, output_root=tmp_path / "outputs")) == 1
    )
    changed = dict(protocol, epochs=24)
    write_json(config_path, changed)
    with pytest.raises(ValueError, match="different protocol"):
        run_siglip_experiments(config_path, output_root=tmp_path / "outputs")
    (research / "ablation_summary.csv").unlink()
    (research / "test_metrics.json").write_text("{}")
    with pytest.raises(ValueError, match="Partial test evaluation"):
        run_siglip_experiments(config_path, output_root=tmp_path / "outputs")
