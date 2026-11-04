"""Small UI helpers shared by the application and its tests."""
from __future__ import annotations

import hashlib
import io
import json
from collections.abc import MutableMapping
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from .utils import LABELS

MODEL_SPECS = {
    "SigLIP 2 Linear Probe": {"run": "siglip2_aug", "architecture": "siglip2", "description": "Primary A2 method · frozen 2025 Transformer and trained weather head"},
    "SigLIP 2 Zero-Shot": {"run": "siglip2_zero_shot", "architecture": "siglip2_zero_shot", "description": "Vision-language matching · prompts selected on validation data"},
    "Simple CNN": {"run": "cnn_baseline", "architecture": "cnn", "description": "Compact CNN trained from scratch"},
    "MobileNetV2": {"run": "mobilenet_improved", "architecture": "mobilenet", "description": "Efficient ImageNet feature extractor"},
    "ConvNeXt-Tiny": {"run": "convnext_aug", "architecture": "convnext", "description": "Historical A2 experiment · frozen ImageNet CNN"},
}
RUN_LABELS = {spec["run"]: name for name, spec in MODEL_SPECS.items()} | {
    "convnext_aug": "ConvNeXt-Tiny + augmentation (historical)",
    "convnext_noaug": "ConvNeXt-Tiny · no augmentation (historical)"}

for _seed in (42, 43, 44):
    for _treatment in ('aug', 'noaug'):
        _run = f'siglip2_{_treatment}' + ('' if _seed == 42 else f'_s{_seed}')
        RUN_LABELS[_run] = f'SigLIP 2 {"+ aug." if _treatment == "aug" else "no aug."} · seed {_seed}'


def decode_upload(content: bytes) -> Image.Image:
    if not content:
        raise ValueError("The image file is empty.")
    if len(content) > 12 * 1024 * 1024:
        raise ValueError("Choose a file smaller than 12 MB.")
    with Image.open(io.BytesIO(content)) as image:
        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise ValueError("Choose a JPG, PNG, or WebP image.")
        if image.width * image.height > 25_000_000:
            raise ValueError("Choose an image smaller than 25 megapixels.")
        if max(image.size) / min(image.size) > 64:
            raise ValueError("Choose an image with an aspect ratio no greater than 64:1.")
        image.load()
        return ImageOps.exif_transpose(image).convert("RGB")


def example_catalog(manifest: Path, root: Path, limit=6) -> dict[str, list[str]]:
    """Offer only existing test images located inside the project."""
    result = {label: [] for label in LABELS}
    if not manifest.is_file():
        return result
    frame = pd.read_csv(manifest)
    if not {"path", "label", "split"}.issubset(frame.columns):
        return result
    root = root.resolve()
    for row in frame[frame.split == "test"].itertuples():
        label = str(row.label).title()
        if label not in result or len(result[label]) >= limit:
            continue
        path = (root / str(row.path)).resolve()
        if path.is_relative_to(root) and path.is_file():
            result[label].append(str(path))
    return result


def sync_image_state(state: MutableMapping, content: bytes | None) -> str | None:
    """Keep results across display changes, discard them when the input changes."""
    digest = hashlib.sha256(content).hexdigest() if content is not None else None
    if state.get("active_image") != digest or "image_predictions" not in state:
        state["active_image"] = digest
        state["image_predictions"] = {}
        state["image_heatmaps"] = {}
        state["comparison_shown"] = False
    return digest


def result_key(digest: str, run: str, device: str, modified: int) -> tuple:
    return digest, run, device, modified


def export_prediction(prediction, model_name: str, device: str, filename: str, size) -> bytes:
    document = {
        "image": {"name": Path(filename).name, "width": size[0], "height": size[1]},
        "model": model_name,
        "predicted_class": prediction["label"].title(),
        "softmax_confidence": prediction["confidence"],
        "class_scores": dict(zip(LABELS, prediction["probabilities"], strict=True)),
        "device": device.upper(),
        "forward_ms": prediction["latency_ms"],
        "timing": "One synchronized forward including score conversion; excludes decoding, preprocessing, transfer, loading, and explanations",
    }
    document["score_type"] = prediction.get("score_type", "softmax")
    if document["score_type"] == "sigmoid_matching":
        document["sigmoid_matching_score"] = document.pop("softmax_confidence")
        document["score_interpretation"] = "Independent image-text matching scores; uncalibrated and do not sum to one"
    if "logits" in prediction:
        document["class_logits"] = dict(zip(LABELS, prediction["logits"], strict=True))
    if 'environment' in prediction:
        document['environment'] = prediction['environment']
    if 'server_prediction_ms' in prediction:
        document['server_prediction_ms'] = prediction['server_prediction_ms']
        document['server_timing'] = 'Model retrieval, preprocessing and forward; excludes upload, decoding, browser rendering and optional benchmark'
    document['browser_user_agent'] = prediction.get('browser_user_agent', 'Unavailable')
    if 'benchmark' in prediction:
        document['benchmark'] = prediction['benchmark']
    return (json.dumps(document, indent=2, allow_nan=False) + "\n").encode()
