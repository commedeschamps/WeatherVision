from __future__ import annotations

import json
import logging
import os
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
from torch import nn
from torch.utils.data import DataLoader

from .config import Experiment
from .data import WeatherDataset, transform, verify_content
from .models import build_model, parameter_counts, set_training_mode
from .plots import training_curves
from .utils import CLASSES, ROOT, environment, seed_everything, select_device, sha256_file, synchronize, write_json

LOG = logging.getLogger("weathervision")


def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        raise ValueError("Metric inputs must be nonempty and have equal length")
    if any(int(v) not in range(4) for v in list(y_true) + list(y_pred)):
        raise ValueError("Class indices must be in [0, 3]")
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(4)), average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f1),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(range(4))).tolist(),
            "per_class": classification_report(y_true, y_pred, labels=list(range(4)), target_names=CLASSES,
                                               zero_division=0, output_dict=True)}


def atomic_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path, device="cpu"):
    device = torch.device(device)
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (pickle.UnpicklingError, EOFError) as error:
        raise ValueError("Cannot decode checkpoint; download the project weights again") from error
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1 or checkpoint.get("classes") != list(CLASSES):
        raise ValueError("Unsupported checkpoint format or class mapping")
    if not isinstance(checkpoint.get("config"), dict) or not isinstance(checkpoint.get("model_state"), dict):
        raise ValueError("Checkpoint metadata or model weights are missing")
    try:
        config = Experiment(**checkpoint["config"])
    except TypeError as error:
        raise ValueError("Malformed checkpoint configuration") from error
    # All weights are saved, so inference never downloads anything.
    model = build_model(config.model, pretrained=False, freeze_backbone=config.freeze_backbone)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def _epoch(model, loader, device, optimizer=None):
    total_loss, labels, predictions = 0.0, [], []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss; inspect data and learning rate")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(targets)
            labels.extend(targets.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(1).cpu().tolist())
    if not labels:
        raise ValueError("Empty data loader")
    result = calculate_metrics(labels, predictions)
    return {"loss": total_loss / len(labels), **{k: result[k] for k in ("accuracy", "f1_macro")}}


def train(run: str, config: Experiment, manifest_path: Path, output_dir: Path,
          root: Path = ROOT, resume=False, skip_complete=False):
    seed_everything(config.seed)
    device = select_device(config.device)
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    frame = pd.read_csv(manifest_path)
    verify_content(frame, root)
    manifest_hash = sha256_file(manifest_path)
    status_path = output_dir / "status.json"
    if status_path.exists():
        status = json.loads(status_path.read_text())
        same = status["config"] == config.to_dict() and status["manifest_sha256"] == manifest_hash
        if not same:
            raise ValueError("Existing run has different config or data. Use a new run/output directory")
        if status["state"] == "complete":
            if skip_complete and (output_dir / "best.pt").is_file():
                LOG.info("Skipping completed run %s", run)
                return status
            raise ValueError("Run is already complete. Use --skip-complete or a new run name")
        if not resume:
            raise ValueError("Incomplete run exists; use --resume or a new run name")
    elif resume:
        raise ValueError("Nothing to resume: status.json is missing")
    elif output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory is nonempty; choose a new run name")
    output_dir.mkdir(parents=True, exist_ok=True)
    continuing = resume and (output_dir / "last.pt").is_file()
    model = build_model(config.model, pretrained=config.pretrained and not continuing,
                        freeze_backbone=config.freeze_backbone).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=config.lr, weight_decay=config.weight_decay)
    history, best, bad, best_epoch, start, elapsed = [], -1.0, 0, 0, 1, 0.0
    if continuing:
        saved = torch.load(output_dir / "last.pt", map_location="cpu", weights_only=True)
        if saved["config"] != config.to_dict() or saved["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume checkpoint config/data mismatch")
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        # Optimizer state tensors must follow parameters to the selected device.
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor) and key != "step":
                    state[key] = value.to(device)
        history, best, bad = saved["history"], saved["best_score"], saved["bad_epochs"]
        best_epoch, start, elapsed = saved["best_epoch"], saved["epoch"] + 1, saved["training_seconds"]
    metadata = {"run": run, "config": config.to_dict(), "manifest_sha256": manifest_hash,
                "environment": environment(device), "command": sys.argv,
                "pretrained_weights": ({"mobilenet": "MobileNet_V2_Weights.IMAGENET1K_V1",
                                         "convnext": "ConvNeXt_Tiny_Weights.IMAGENET1K_V1"}.get(config.model)
                                       if config.pretrained else None),
                "selection": "maximum validation macro-F1; earliest epoch wins ties",
                "source_sha256": {p.name: sha256_file(p) for p in (root / "src").glob("*.py")},
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), **parameter_counts(model)}
    write_json(status_path, metadata | {"state": "running"})
    train_data = WeatherDataset(frame, root, "train", config.augment, config.model)
    val_data = WeatherDataset(frame, root, "val", False, config.model)
    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    for epoch in range(start, config.epochs + 1):
        if bad >= config.patience:
            break
        # Absolute epoch seeds make restart reproducible; shuffle RNG is independent
        # of augmentation RNG. ConvNeXt's head is linear with no dropout.
        seed_everything(config.seed + epoch)
        generator = torch.Generator().manual_seed(config.seed + epoch)
        train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True,
                                  generator=generator, num_workers=config.num_workers)
        synchronize(device)
        started = time.perf_counter()
        set_training_mode(model, config.model, config.freeze_backbone)
        tr = _epoch(model, train_loader, device, optimizer)
        model.eval()
        va = _epoch(model, val_loader, device)
        synchronize(device)
        duration = time.perf_counter() - started
        elapsed += duration
        row = {"epoch": epoch, "seconds": duration,
               **{f"train_{k}": v for k, v in tr.items()}, **{f"val_{k}": v for k, v in va.items()}}
        history.append(row)
        improved = va["f1_macro"] > best
        if improved:
            best, bad, best_epoch = va["f1_macro"], 0, epoch
        else:
            bad += 1
        checkpoint = {"format_version": 1, "classes": list(CLASSES), "run": run,
                      "config": config.to_dict(), "manifest_sha256": manifest_hash,
                      "model_state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                      "optimizer_state": optimizer.state_dict(), "epoch": epoch, "history": history,
                      "best_score": best, "bad_epochs": bad, "best_epoch": best_epoch,
                      "training_seconds": elapsed, "metadata": metadata}
        if improved:
            atomic_checkpoint(output_dir / "best.pt", checkpoint)
        atomic_checkpoint(output_dir / "last.pt", checkpoint)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        LOG.info("%s epoch %d/%d train loss %.4f val F1 %.4f (%0.1fs)",
                 run, epoch, config.epochs, tr["loss"], va["f1_macro"], duration)
    training_curves(history, output_dir / "training_curves.png")
    result = metadata | {"state": "complete", "epochs_ran": len(history),
                         "best_epoch": best_epoch, "best_val_f1_macro": best,
                         "training_seconds": elapsed, "stopped_early": len(history) < config.epochs,
                         "stop_reason": "early_stopping" if bad >= config.patience else "epoch_limit"}
    write_json(status_path, result)
    return result


def predict_image(model, image, architecture, device):
    if any(module.training for module in model.modules()):
        raise ValueError("Call model.eval() before inference")
    if image.width * image.height > 25_000_000:
        raise ValueError("Use an image smaller than 25 megapixels")
    if max(image.size) / min(image.size) > 64:
        raise ValueError("Use an image with an aspect ratio no greater than 64:1")
    x = transform("test", False, architecture)(image.convert("RGB")).unsqueeze(0).to(device)
    synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        probabilities = model(x).softmax(1)[0]
    synchronize(device)
    latency = (time.perf_counter() - started) * 1000
    values = probabilities.cpu().tolist()
    if len(values) != 4 or not np.isfinite(values).all():
        raise RuntimeError("Model returned invalid probabilities")
    return {"class_id": int(np.argmax(values)), "label": CLASSES[int(np.argmax(values))],
            "confidence": max(values), "probabilities": values, "latency_ms": latency}
