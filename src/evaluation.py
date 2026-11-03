from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .config import Experiment, validate_ablation
from .data import WeatherDataset, verify_content
from .engine import calculate_metrics, load_checkpoint
from .plots import comparison_figure, confusion_figure, training_curves
from .utils import CLASSES, ROOT, environment, seed_everything, select_device, sha256_file, synchronize, write_json


def benchmark(model, sample, device, warmup=5, repeats=30):
    if warmup < 1 or repeats < 2:
        raise ValueError("Latency requires at least one warm-up and two repetitions")
    sample = sample.to(device)
    measurements = []
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        synchronize(device)
        for _ in range(repeats):
            synchronize(device)
            start = time.perf_counter()
            model(sample)
            synchronize(device)
            measurements.append((time.perf_counter() - start) * 1000)
    return {"median_ms": float(np.median(measurements)), "mean_ms": float(np.mean(measurements)),
            "std_ms": float(np.std(measurements)), "p95_ms": float(np.percentile(measurements, 95)),
            "measurements_ms": measurements, "warmup": warmup, "repeats": repeats,
            "batch_size": 1, "includes_preprocessing": False, "includes_device_transfer": False,
            "synchronization": device.type if device.type in {"cuda", "mps"} else "not_required",
            "protocol": "Same first test tensor, resident on device; forward pass only, inference_mode; no softmax, disk I/O, or model loading",
            "environment": environment(device)}


def evaluate_run(run, manifest_path, output_root, root=ROOT, device_name="auto", warmup=5, repeats=30):
    seed_everything(42)
    device = select_device(device_name)
    run_dir = Path(output_root) / run
    status = json.loads((run_dir / "status.json").read_text())
    if status["state"] != "complete":
        raise ValueError(f"Training incomplete: {run}")
    frame = pd.read_csv(manifest_path)
    verify_content(frame, root)
    model, checkpoint = load_checkpoint(run_dir / "best.pt", device)
    if checkpoint["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("Evaluation manifest differs from training manifest")
    data = WeatherDataset(frame, root, "test", False, checkpoint["config"]["model"])
    loader = DataLoader(data, batch_size=checkpoint["config"]["batch_size"], shuffle=False, num_workers=0)
    probabilities, targets = [], []
    with torch.inference_mode():
        for x, y in loader:
            probabilities.extend(model(x.to(device)).softmax(1).cpu().tolist())
            targets.extend(y.tolist())
    predictions = np.argmax(probabilities, axis=1).tolist()
    metrics = calculate_metrics(targets, predictions)
    latency = benchmark(model, data[0][0].unsqueeze(0), device, warmup, repeats)
    prediction_frame = data.frame[["path", "label", "class_id", "group"]].copy()
    prediction_frame["predicted_class_id"] = predictions
    prediction_frame["predicted_label"] = [CLASSES[p] for p in predictions]
    prediction_frame["correct"] = prediction_frame.class_id == prediction_frame.predicted_class_id
    for j, label in enumerate(CLASSES):
        prediction_frame[f"score_{label}"] = [p[j] for p in probabilities]
    test_hash = __import__("hashlib").sha256("\n".join(data.frame.path).encode()).hexdigest()
    metrics.update(run=run, split="test", test_samples=len(data), test_membership_sha256=test_hash,
                   manifest_sha256=checkpoint["manifest_sha256"], checkpoint_sha256=sha256_file(run_dir / "best.pt"),
                   best_epoch=checkpoint["best_epoch"], latency=latency)
    prediction_frame.to_csv(run_dir / "test_predictions.csv", index=False)
    write_json(run_dir / "test_metrics.json", metrics)
    confusion_figure(metrics["confusion_matrix"], run_dir / "confusion_matrix.png")
    return metrics


def compare_runs(runs, output_root, ablation=("convnext_aug", "convnext_noaug")):
    output_root = Path(output_root)
    rows, membership, manifests = [], set(), set()
    for run in runs:
        directory = output_root / run
        metrics = json.loads((directory / "test_metrics.json").read_text())
        status = json.loads((directory / "status.json").read_text())
        if metrics["checkpoint_sha256"] != sha256_file(directory / "best.pt"):
            raise ValueError(f"Stale evaluation for {run}")
        membership.add(metrics["test_membership_sha256"])
        manifests.add(metrics["manifest_sha256"])
        rows.append({"run": run, "model": status["config"]["model"], "augment": status["config"]["augment"],
                     **{key: metrics[key] for key in ["accuracy", "precision_macro", "recall_macro", "f1_macro", "test_samples"]},
                     "latency_mean_ms": metrics["latency"]["mean_ms"],
                     "latency_median_ms": metrics["latency"]["median_ms"],
                     "parameters": status["parameters"], "trainable_parameters": status["trainable_parameters"],
                     "training_seconds": status["training_seconds"], "best_epoch": status["best_epoch"],
                     "validation_f1_macro": status["best_val_f1_macro"],
                     "device": metrics["latency"]["environment"]["device"]})
    if len(membership) != 1 or len(manifests) != 1:
        raise ValueError("Cannot compare models with different dataset/test membership")
    frame = pd.DataFrame(rows)
    frame.to_csv(output_root / "model_comparison.csv", index=False)
    write_json(output_root / "model_comparison.json", rows)
    comparison_figure(frame, output_root / "model_comparison.png")
    if set(ablation).issubset(set(runs)):
        configs = [Experiment(**json.loads((output_root / run / "status.json").read_text())["config"]) for run in ablation]
        validate_ablation(*configs)
        subset = frame.set_index("run").loc[list(ablation)].reset_index()
        subset.to_csv(output_root / "ablation.csv", index=False)
        delta = float(subset.iloc[0].f1_macro - subset.iloc[1].f1_macro)
        write_json(output_root / "ablation_analysis.json", {
            "comparison": "augmentation minus no augmentation", "test_f1_macro_delta": delta,
            "interpretation": f"Augmentation {'increased' if delta > 0 else 'decreased' if delta < 0 else 'did not change'} test macro-F1 by {abs(delta):.4f} under the shared evaluation protocol.",
            "control": "All serialized configuration fields except augment are identical; identical manifest and test membership verified",
            "selection": "Each configuration selected its own best validation macro-F1 checkpoint; no test-based tuning"})
    return frame


def regenerate_figures(output_root):
    output_root = Path(output_root)
    for directory in output_root.iterdir():
        if not directory.is_dir():
            continue
        if (directory / "history.csv").exists():
            training_curves(pd.read_csv(directory / "history.csv").to_dict("records"), directory / "training_curves.png")
        if (directory / "test_metrics.json").exists():
            metrics = json.loads((directory / "test_metrics.json").read_text())
            confusion_figure(metrics["confusion_matrix"], directory / "confusion_matrix.png")
    if (output_root / "model_comparison.csv").exists():
        comparison_figure(pd.read_csv(output_root / "model_comparison.csv"), output_root / "model_comparison.png")
