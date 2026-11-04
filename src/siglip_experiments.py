"""Controlled frozen-feature experiments; validation selection precedes test access."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import verify_content
from .engine import calculate_metrics, atomic_checkpoint
from .evaluation import benchmark
from .plots import confusion_figure, training_curves
from .siglip2 import (
    MODEL_ID,
    REVISION,
    ASSET_DIR,
    SiglipDataset,
    SiglipWeather,
    load_encoder,
    load_siglip_checkpoint,
)
from .utils import (
    CLASSES,
    ROOT,
    environment,
    seed_everything,
    sha256_file,
    select_device,
    synchronize,
    write_json,
)


def experiment_name(augment, seed):
    return ("siglip2_aug" if augment else "siglip2_noaug") + (
        "" if seed == 42 else f"_s{seed}"
    )


def feature_cache(
    encoder,
    processor,
    frame,
    split,
    device,
    *,
    augment=False,
    seed=42,
    epoch=0,
    batch_size=16,
):
    data = SiglipDataset(
        frame, processor, split, augment=augment, seed=seed, epoch=epoch
    )
    identity = {
        "revision": REVISION,
        "manifest": sha256_file(ROOT / "data/manifest.csv"),
        "processor": json.loads(json.dumps(processor.to_dict())),
        "split": split,
        "augment": augment,
        "seed": seed if augment else None,
        "epoch": epoch if augment else None,
        "representation": "pretrained_attention_pooler_output",
    }
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = ROOT / ".cache/siglip2_features" / f"{key}.pt"
    if path.is_file():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity or saved["paths"] != data.frame.path.tolist():
            raise ValueError("Feature-cache identity mismatch")
        return saved["features"], saved["targets"], 0.0, True
    encoder.eval()
    features = []
    targets = []
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        for images, labels in DataLoader(
            data, batch_size=batch_size, shuffle=False, num_workers=0
        ):
            features.append(encoder(pixel_values=images.to(device)).pooler_output.cpu())
            targets.append(labels)
    synchronize(device)
    elapsed = time.perf_counter() - start
    values, labels = torch.cat(features), torch.cat(targets)
    if not torch.isfinite(values).all():
        raise RuntimeError("Non-finite pretrained features")
    atomic_checkpoint(
        path,
        {
            "identity": identity,
            "paths": data.frame.path.tolist(),
            "features": values,
            "targets": labels,
        },
    )
    return values, labels, elapsed, False


def checkpoint_metadata(processor, provenance, manifest_hash, feature_dim):
    return {
        "format_version": 2,
        "model_id": MODEL_ID,
        "revision": REVISION,
        "license": "Apache-2.0",
        "classes": list(CLASSES),
        "feature_dim": feature_dim,
        "manifest_sha256": manifest_hash,
        "encoder_sha256": provenance["files"]["model.safetensors"],
        "processor": json.loads(json.dumps(processor.to_dict())),
        "dependencies": {"transformers": "4.57.6", "torch": str(torch.__version__)},
        "feature_representation": "pretrained attention-pooled pooler_output; no added feature pooling",
    }


def train_probe(
    run,
    augment,
    seed,
    config,
    frame,
    encoder,
    processor,
    provenance,
    device,
    output_root=None,
):
    directory = (ROOT / "outputs" if output_root is None else Path(output_root)) / run
    if (directory / "status.json").is_file():
        status = json.loads((directory / "status.json").read_text())
        if status["state"] == "complete":
            if (
                status["seed"] != seed
                or status["augment"] != augment
                or status["protocol"] != config
            ):
                raise ValueError("Existing experiment protocol differs")
            return status
        raise ValueError(
            "Incomplete run: use a fresh experiment directory after inspecting its history"
        )
    directory.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    start = time.perf_counter()
    extraction_seconds = 0.0
    optimization_seconds = 0.0
    val_x, val_y, seconds, _ = feature_cache(
        encoder,
        processor,
        frame,
        "val",
        device,
        batch_size=config["feature_batch_size"],
    )
    extraction_seconds += seconds
    deterministic = None
    if not augment:
        x, y, seconds, _ = feature_cache(
            encoder,
            processor,
            frame,
            "train",
            device,
            batch_size=config["feature_batch_size"],
        )
        deterministic = (x, y)
        extraction_seconds += seconds
    # Backbone construction never changes the shared seeded head initialization.
    seed_everything(seed)
    head = nn.Linear(val_x.shape[1], len(CLASSES))
    initial_hash = hashlib.sha256(
        b"".join(p.detach().numpy().tobytes() for p in head.parameters())
    ).hexdigest()
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    base = checkpoint_metadata(
        processor, provenance, sha256_file(ROOT / "data/manifest.csv"), val_x.shape[1]
    )
    history = []
    best = -1.0
    bad = 0
    best_epoch = 0
    write_json(
        directory / "status.json",
        {"state": "running", "seed": seed, "augment": augment, "protocol": config},
    )
    for epoch in range(1, config["epochs"] + 1):
        if deterministic is None:
            x, y, seconds, hit = feature_cache(
                encoder,
                processor,
                frame,
                "train",
                device,
                augment=True,
                seed=seed,
                epoch=epoch,
                batch_size=config["feature_batch_size"],
            )
            extraction_seconds += seconds
        else:
            x, y = deterministic
        epoch_start = time.perf_counter()
        head.train()
        loss_sum = 0.0
        labels = []
        predicted = []
        permutation = torch.randperm(
            len(y), generator=torch.Generator().manual_seed(seed + epoch)
        )
        for indices in permutation.split(config["batch_size"]):
            optimizer.zero_grad(set_to_none=True)
            logits = head(x[indices])
            loss = nn.functional.cross_entropy(logits, y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite head loss")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
            labels += y[indices].tolist()
            predicted += logits.detach().argmax(1).tolist()
        train_metrics = calculate_metrics(labels, predicted)
        head.eval()
        with torch.inference_mode():
            val_logits = head(val_x)
            val_loss = float(nn.functional.cross_entropy(val_logits, val_y))
        val_metrics = calculate_metrics(val_y.tolist(), val_logits.argmax(1).tolist())
        seconds = time.perf_counter() - epoch_start
        optimization_seconds += seconds
        row = {
            "epoch": epoch,
            "seconds": seconds,
            "train_loss": loss_sum / len(y),
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
        }
        history.append(row)
        if val_metrics["f1_macro"] > best:
            best = val_metrics["f1_macro"]
            bad = 0
            best_epoch = epoch
            payload = base | {
                "run": run,
                "mode": "linear_probe",
                "config": {"model": "siglip2"},
                "seed": seed,
                "augment": augment,
                "protocol": config,
                "head_state": {
                    k: v.detach().clone() for k, v in head.state_dict().items()
                },
                "best_epoch": epoch,
                "validation_f1_macro": best,
                "initial_head_sha256": initial_hash,
            }
            atomic_checkpoint(directory / "best.pt", payload)
        else:
            bad += 1
        pd.DataFrame(history).to_csv(directory / "history.csv", index=False)
        print(
            f'{run} epoch {epoch}: validation macro-F1={val_metrics["f1_macro"]:.4f}; best={best:.4f}',
            flush=True,
        )
        if bad >= config["patience"]:
            break
    elapsed = time.perf_counter() - start
    status = {
        "state": "complete",
        "run": run,
        "seed": seed,
        "augment": augment,
        "protocol": config,
        "config": {"model": "siglip2", "augment": augment},
        "best_epoch": best_epoch,
        "best_val_f1_macro": best,
        "epochs_ran": len(history),
        "training_seconds": elapsed,
        "feature_extraction_seconds": extraction_seconds,
        "head_optimization_seconds": optimization_seconds,
        "parameters": provenance["parameters"]
        + sum(p.numel() for p in head.parameters()),
        "trainable_parameters": sum(p.numel() for p in head.parameters()),
        "initial_head_sha256": initial_hash,
        "environment": environment(device),
        "checkpoint_sha256": sha256_file(directory / "best.pt"),
    }
    write_json(directory / "status.json", status)
    training_curves(history, directory / "training_curves.png")
    return status


def select_zero_shot(
    config, frame, encoder, processor, provenance, device, output_root=None
):
    directory = (
        ROOT / "outputs" if output_root is None else Path(output_root)
    ) / "siglip2_zero_shot"
    directory.mkdir(parents=True, exist_ok=True)
    banks = torch.load(
        ASSET_DIR / "prompt_banks.pt", weights_only=True, map_location="cpu"
    )
    if banks["prompt_sets"] != config["prompt_sets"] or banks["revision"] != REVISION:
        raise ValueError("Prompt-bank identity mismatch")
    val_x, val_y, _, _ = feature_cache(
        encoder,
        processor,
        frame,
        "val",
        device,
        batch_size=config["feature_batch_size"],
    )
    rows = []
    for index, embeddings in enumerate(banks["text_embeddings"]):
        logits = (
            nn.functional.normalize(val_x, dim=-1)
            @ embeddings.T
            * banks["logit_scale"].exp()
            + banks["logit_bias"]
        )
        result = calculate_metrics(val_y.tolist(), logits.argmax(1).tolist())
        rows.append(
            {
                "prompt_set": index,
                "validation_accuracy": result["accuracy"],
                "validation_f1_macro": result["f1_macro"],
            }
        )
    selected = max(
        range(len(rows)), key=lambda index: rows[index]["validation_f1_macro"]
    )
    pd.DataFrame(rows).to_csv(directory / "prompt_validation.csv", index=False)
    payload = checkpoint_metadata(
        processor, provenance, sha256_file(ROOT / "data/manifest.csv"), val_x.shape[1]
    ) | {
        "run": "siglip2_zero_shot",
        "mode": "zero_shot",
        "config": {"model": "siglip2_zero_shot"},
        "selected_prompt_set": selected,
        "prompts": banks["prompt_sets"][selected],
        "text_embeddings": banks["text_embeddings"][selected],
        "logit_scale": banks["logit_scale"],
        "logit_bias": banks["logit_bias"],
        "validation_f1_macro": rows[selected]["validation_f1_macro"],
        "selection": "maximum validation macro-F1; first prompt set wins ties",
    }
    atomic_checkpoint(directory / "best.pt", payload)
    write_json(
        directory / "status.json",
        {
            "state": "complete",
            "run": "siglip2_zero_shot",
            "config": {"model": "siglip2_zero_shot"},
            "parameters": provenance["parameters"],
            "trainable_parameters": 0,
            "training_seconds": 0.0,
            "best_epoch": 0,
            "best_val_f1_macro": rows[selected]["validation_f1_macro"],
            "selected_prompt_set": selected,
        },
    )
    return payload


def evaluate_siglip_run(
    run,
    frame,
    encoder,
    processor,
    provenance,
    device,
    test_features=None,
    output_root=None,
):
    directory = (ROOT / "outputs" if output_root is None else Path(output_root)) / run
    if (directory / "test_metrics.json").exists():
        raise FileExistsError(f"Refusing to overwrite completed test evaluation: {run}")
    checkpoint_path = directory / "best.pt"
    model, checkpoint = load_siglip_checkpoint(
        checkpoint_path, device, encoder_bundle=(encoder, processor, provenance)
    )
    if checkpoint["manifest_sha256"] != sha256_file(ROOT / "data/manifest.csv"):
        raise ValueError("Manifest differs from training")
    test = frame[frame.split == "test"].reset_index(drop=True)
    features, targets = test_features
    with torch.inference_mode():
        if model.mode == "linear_probe":
            logits = model.head(features.to(device)).cpu()
        else:
            logits = (
                nn.functional.normalize(features, dim=-1)
                @ checkpoint["text_embeddings"].T
                * checkpoint["logit_scale"].exp()
                + checkpoint["logit_bias"]
            )
        scores = logits.sigmoid() if model.mode == "zero_shot" else logits.softmax(1)
    predictions = logits.argmax(1)
    metrics = calculate_metrics(targets.tolist(), predictions.tolist())
    output = test[["path", "label", "class_id", "group"]].copy()
    output["predicted_class_id"] = predictions.tolist()
    output["predicted_label"] = [CLASSES[i] for i in predictions]
    output["correct"] = output.class_id == output.predicted_class_id
    for index, label in enumerate(CLASSES):
        output["score_" + label] = scores[:, index].tolist()
    output.to_csv(directory / "test_predictions.csv", index=False)
    sample = SiglipDataset(frame, processor, "test")[0][0].unsqueeze(0)
    latency = benchmark(model, sample, device)
    metrics.update(
        run=run,
        split="test",
        test_samples=len(test),
        manifest_sha256=checkpoint["manifest_sha256"],
        test_membership_sha256=hashlib.sha256(
            "\n".join(test.path).encode()
        ).hexdigest(),
        checkpoint_sha256=sha256_file(checkpoint_path),
        best_epoch=checkpoint.get("best_epoch", 0),
        score_type="sigmoid_matching" if model.mode == "zero_shot" else "softmax",
        latency=latency,
    )
    write_json(directory / "test_metrics.json", metrics)
    confusion_figure(metrics["confusion_matrix"], directory / "confusion_matrix.png")
    return metrics


def run_siglip_experiments(
    config_path=ROOT / "configs/siglip2.json", device_name="auto", output_root=None
):
    output_root = ROOT / "outputs" if output_root is None else Path(output_root)
    research_dir = output_root / "siglip2"
    research_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text())
    if config["model_id"] != MODEL_ID or config["revision"] != REVISION:
        raise ValueError(
            "The experiment must use the pinned official SigLIP 2 checkpoint"
        )
    if (research_dir / "ablation_summary.csv").is_file():
        lock = json.loads((research_dir / "selection_lock.json").read_text())
        if lock["config"] != config or lock["manifest_sha256"] != sha256_file(
            ROOT / "data/manifest.csv"
        ):
            raise ValueError(
                "Completed output directory uses a different protocol or dataset"
            )
        return pd.read_csv(research_dir / "ablation_summary.csv")
    if any(output_root.glob("siglip2*/test_metrics.json")):
        raise ValueError(
            "Partial test evaluation exists; preserve the selection lock and inspect it before continuing"
        )
    seed_everything(42)
    device = select_device(device_name)
    frame = pd.read_csv(ROOT / "data/manifest.csv")
    verify_content(frame)
    encoder, processor, provenance = load_encoder()
    encoder.to(device).eval()
    select_zero_shot(config, frame, encoder, processor, provenance, device, output_root)
    runs = []
    for seed in config["seeds"]:
        for augment in [True, False]:
            run = experiment_name(augment, seed)
            runs.append(run)
            train_probe(
                run,
                augment,
                seed,
                config,
                frame,
                encoder,
                processor,
                provenance,
                device,
                output_root,
            )
    selections = {
        run: sha256_file(output_root / run / "best.pt")
        for run in runs + ["siglip2_zero_shot"]
    }
    write_json(
        research_dir / "selection_lock.json",
        {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "config": config,
            "manifest_sha256": sha256_file(ROOT / "data/manifest.csv"),
            "checkpoints": selections,
            "test_evaluated_before_lock": False,
            "main_run_predeclared": config["main_run"],
        },
    )
    features, targets, _, _ = feature_cache(
        encoder,
        processor,
        frame,
        "test",
        device,
        batch_size=config["feature_batch_size"],
    )
    for run in runs + ["siglip2_zero_shot"]:
        evaluate_siglip_run(
            run,
            frame,
            encoder,
            processor,
            provenance,
            device,
            test_features=(features, targets),
            output_root=output_root,
        )
    rows = []
    for run in runs:
        m = json.loads((output_root / run / "test_metrics.json").read_text())
        s = json.loads((output_root / run / "status.json").read_text())
        rows.append(
            {
                "run": run,
                "seed": s["seed"],
                "augment": s["augment"],
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "best_epoch": s["best_epoch"],
                "training_seconds": s["training_seconds"],
            }
        )
    pd.DataFrame(rows).to_csv(research_dir / "per_seed.csv", index=False)
    summary = (
        pd.DataFrame(rows)
        .groupby("augment")
        .agg(
            seeds=("seed", "count"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            f1_mean=("f1_macro", "mean"),
            f1_std=("f1_macro", "std"),
        )
        .reset_index()
    )
    summary.to_csv(research_dir / "ablation_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
