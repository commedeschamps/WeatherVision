from dataclasses import replace
import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.config import Experiment, load_config, validate_ablation
from src.engine import atomic_checkpoint, calculate_metrics, load_checkpoint, predict_image, train
from src.evaluation import benchmark, compare_runs, evaluate_run
from src.explainability import gradcam_convnext, overlay_image
from src.models import build_model, set_training_mode
from src.utils import CLASSES, ROOT


def test_config():
    configs = load_config(ROOT / "configs/experiments.yaml")
    assert len(configs) == 4
    validate_ablation(configs["convnext_aug"], configs["convnext_noaug"])
    with pytest.raises(ValueError, match="only in augmentation"):
        validate_ablation(configs["convnext_aug"], replace(configs["convnext_noaug"], lr=0.01))


@pytest.mark.parametrize("values", [{"epochs": 0}, {"batch_size": -1}, {"augment": "false"}, {"image_size": 112}, {"device": "magic"}])
def test_config_rejects_invalid(values):
    with pytest.raises(ValueError):
        Experiment(model="convnext", **values)


@pytest.mark.parametrize("architecture", ["cnn", "mobilenet", "convnext"])
def test_model_outputs_and_freeze(architecture):
    model = build_model(architecture, pretrained=False, freeze_backbone=architecture != "cnn").eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 224, 224)).shape == (2, 4)
    if architecture != "cnn":
        set_training_mode(model, architecture, True)
        assert not model.features.training
        assert not any(p.requires_grad for p in model.features.parameters())
        assert any(p.requires_grad for p in model.classifier.parameters())


@pytest.mark.parametrize("architecture", ["cnn", "mobilenet", "convnext"])
def test_checkpoint_inference(tmp_path, architecture):
    cnn = architecture == "cnn"
    config = Experiment(model=architecture, pretrained=False, freeze_backbone=not cnn)
    model = build_model(architecture, False, not cnn).eval()
    checkpoint = {"format_version": 1, "classes": list(CLASSES), "config": config.to_dict(), "model_state": model.state_dict()}
    path = tmp_path / "model.pt"
    atomic_checkpoint(path, checkpoint)
    restored, _ = load_checkpoint(path)
    tensor = torch.rand(1, 3, 224, 224)
    with torch.no_grad():
        torch.testing.assert_close(model(tensor), restored(tensor))
    result = predict_image(restored, Image.new("RGB", (300, 220), (120, 160, 200)), architecture, torch.device("cpu"))
    assert sum(result["probabilities"]) == pytest.approx(1, abs=1e-6)
    assert result["label"] in CLASSES
    assert result["latency_ms"] > 0


def test_bad_checkpoint_mapping(tmp_path):
    path = tmp_path / "bad.pt"
    atomic_checkpoint(path, {"format_version": 1, "classes": ["rain", "cloudy"]})
    with pytest.raises(ValueError, match="class mapping"):
        load_checkpoint(path)


def test_metrics():
    metrics = calculate_metrics([0, 1, 2, 3], [0, 1, 2, 2])
    assert metrics["accuracy"] == .75
    assert metrics["precision_macro"] == .625
    assert metrics["recall_macro"] == .75
    assert metrics["f1_macro"] == pytest.approx(2 / 3)
    assert np.asarray(metrics["confusion_matrix"]).sum() == 4
    assert "sunrise" in metrics["per_class"]
    with pytest.raises(ValueError):
        calculate_metrics([], [])


def test_gradcam_frozen_convnext():
    model = build_model("convnext", pretrained=False, freeze_backbone=True).eval()
    x = torch.rand(1, 3, 224, 224)
    hooks = len(model.features[-1]._forward_hooks)
    cam = gradcam_convnext(model, x, 0)
    assert cam.shape == (224, 224)
    assert np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1
    assert len(model.features[-1]._forward_hooks) == hooks
    assert all(p.grad is None for p in model.parameters())
    image = Image.new("RGB", (400, 240), (80, 100, 120))
    overlay = overlay_image(image, cam)
    assert overlay.size == image.size
    assert np.asarray(image)[0, 0].tolist() == [80, 100, 120]
    # Far side border was not seen by center crop and must be unchanged.
    assert np.array_equal(np.asarray(image)[:, 0], np.asarray(overlay)[:, 0])
    assert np.array_equal(np.asarray(image), np.asarray(overlay_image(image, np.zeros((224, 224)))))


def test_training_evaluation_contract(synthetic_data):
    root, frame = synthetic_data
    manifest = root / "data/manifest.csv"
    config = Experiment(model="cnn", pretrained=False, freeze_backbone=False, epochs=2, batch_size=8, device="cpu")
    outputs = root / "outputs"
    status = train("test_cnn", config, manifest, outputs / "test_cnn", root)
    assert status["state"] == "complete" and status["epochs_ran"] == 2
    assert len(json.loads((outputs / "test_cnn/status.json").read_text())["manifest_sha256"]) == 64
    metrics = evaluate_run("test_cnn", manifest, outputs, root, "cpu", 1, 2)
    assert metrics["split"] == "test"
    assert metrics["test_samples"] == len(frame[frame.split == "test"])
    comparison = compare_runs(["test_cnn"], outputs)
    assert len(comparison) == 1
    assert train("test_cnn", config, manifest, outputs / "test_cnn", root, skip_complete=True)["state"] == "complete"
    with pytest.raises(ValueError, match="different config"):
        train("test_cnn", replace(config, lr=.02), manifest, outputs / "test_cnn", root, skip_complete=True)


def test_resume_matches_uninterrupted(synthetic_data, monkeypatch):
    import src.engine as engine
    root, _ = synthetic_data
    config = Experiment(model="cnn", pretrained=False, freeze_backbone=False, epochs=2, batch_size=8, device="cpu")
    manifest = root / "data/manifest.csv"
    original_epoch = engine._epoch
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("Simulated interruption after saved epoch 1")
        return original_epoch(*args, **kwargs)

    monkeypatch.setattr(engine, "_epoch", interrupted)
    with pytest.raises(RuntimeError, match="Simulated"):
        train("resumed", config, manifest, root / "outputs/resumed", root)
    monkeypatch.setattr(engine, "_epoch", original_epoch)
    train("resumed", config, manifest, root / "outputs/resumed", root, resume=True)
    train("uninterrupted", config, manifest, root / "outputs/uninterrupted", root)
    a = torch.load(root / "outputs/resumed/last.pt", weights_only=True)
    b = torch.load(root / "outputs/uninterrupted/last.pt", weights_only=True)
    for key in a["model_state"]:
        torch.testing.assert_close(a["model_state"][key], b["model_state"][key], rtol=0, atol=0)
