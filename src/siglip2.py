"""Pinned SigLIP 2 vision encoder, weather adapters and image-region occlusion."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from .data import open_rgb
from .utils import CLASSES, ROOT, sha256_file, synchronize

MODEL_ID = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
ASSET_DIR = ROOT / "models/siglip2"


def load_encoder(asset_dir=ASSET_DIR):
    from transformers import AutoImageProcessor, SiglipVisionModel, SiglipVisionConfig

    asset_dir = Path(asset_dir)
    provenance_path = asset_dir / "provenance.json"
    if not provenance_path.is_file():
        raise FileNotFoundError(
            "SigLIP 2 pretrained assets are missing. Run python scripts/install_siglip2.py."
        )
    provenance = json.loads(provenance_path.read_text())
    if provenance["model_id"] != MODEL_ID or provenance["revision"] != REVISION:
        raise ValueError(
            "SigLIP 2 asset identity/revision does not match the pinned checkpoint"
        )
    for name, digest in provenance["files"].items():
        if sha256_file(asset_dir / "vision" / name) != digest:
            raise ValueError(f"SigLIP 2 asset integrity check failed: {name}")
    encoder = SiglipVisionModel.from_pretrained(
        asset_dir / "vision",
        local_files_only=True,
        config=SiglipVisionConfig.from_pretrained(
            asset_dir / "vision", local_files_only=True
        ),
        attn_implementation="sdpa",
    ).eval()
    processor = AutoImageProcessor.from_pretrained(
        asset_dir / "vision", local_files_only=True, use_fast=False
    )
    encoder.requires_grad_(False)
    return encoder, processor, provenance


class SiglipWeather(nn.Module):
    """Frozen official attention-pooled features and a linear or zero-shot head."""

    def __init__(
        self,
        encoder,
        processor,
        *,
        head=None,
        text_embeddings=None,
        logit_scale=None,
        logit_bias=None,
    ):
        super().__init__()
        self.encoder = encoder.requires_grad_(False).eval()
        self.processor = processor
        self.mode = "zero_shot" if text_embeddings is not None else "linear_probe"
        if self.mode == "linear_probe":
            if head is None:
                raise ValueError("A trained head must be supplied")
            self.head = head
        else:
            self.register_buffer("text_embeddings", text_embeddings)
            self.register_buffer("logit_scale", logit_scale)
            self.register_buffer("logit_bias", logit_bias)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def features(self, inputs):
        with torch.no_grad():
            return self.encoder(pixel_values=inputs).pooler_output

    def forward(self, inputs):
        features = self.features(inputs)
        if self.mode == "linear_probe":
            return self.head(features)
        return (
            F.normalize(features, dim=-1)
            @ self.text_embeddings.T
            * self.logit_scale.exp()
            + self.logit_bias
        )


def load_siglip_checkpoint(
    path, device="cpu", asset_dir=ASSET_DIR, encoder_bundle=None
):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("format_version") != 2
        or checkpoint.get("model_id") != MODEL_ID
        or checkpoint.get("revision") != REVISION
        or checkpoint.get("classes") != list(CLASSES)
    ):
        raise ValueError("Unsupported SigLIP 2 checkpoint or class mapping")
    encoder, processor, provenance = encoder_bundle or load_encoder(asset_dir)
    if checkpoint["encoder_sha256"] != provenance["files"]["model.safetensors"]:
        raise ValueError("The trained head belongs to different pretrained assets")
    if checkpoint["processor"] != processor.to_dict():
        raise ValueError(
            "Checkpoint preprocessing differs from the installed processor"
        )
    if checkpoint["mode"] == "linear_probe":
        head = nn.Linear(checkpoint["feature_dim"], len(CLASSES))
        head.load_state_dict(checkpoint["head_state"], strict=True)
        model = SiglipWeather(encoder, processor, head=head)
    elif checkpoint["mode"] == "zero_shot":
        model = SiglipWeather(
            encoder,
            processor,
            text_embeddings=checkpoint["text_embeddings"],
            logit_scale=checkpoint["logit_scale"],
            logit_bias=checkpoint["logit_bias"],
        )
    else:
        raise ValueError("Unknown SigLIP 2 classification mode")
    return model.to(device).eval(), checkpoint


class SiglipDataset(Dataset):
    def __init__(
        self, frame, processor, split, root=ROOT, augment=False, seed=42, epoch=0
    ):
        self.frame = frame[frame.split == split].reset_index(drop=True)
        self.processor, self.root = processor, Path(root)
        self.augment, self.seed, self.epoch = augment, seed, epoch
        self.augmentation = T.Compose(
            [
                T.RandomResizedCrop(
                    224,
                    scale=(0.8, 1.0),
                    ratio=(0.9, 1.1),
                    interpolation=InterpolationMode.BILINEAR,
                ),
                T.RandomHorizontalFlip(0.5),
            ]
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = open_rgb(self.root / row.path)
        if self.augment:
            # Every image gets a reproducible new view in every seed/epoch.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.seed * 1_000_000 + self.epoch * 10_000 + index)
                image = self.augmentation(image)
        tensor = self.processor(images=image, return_tensors="pt")["pixel_values"][0]
        return tensor, int(row.class_id)


def siglip_input(model, image):
    image = image.convert("RGB")
    if (
        image.width * image.height > 25_000_000
        or max(image.size) / min(image.size) > 64
    ):
        raise ValueError("Choose an image below 25 megapixels and 64:1 aspect ratio")
    return model.processor(images=image, return_tensors="pt")["pixel_values"]


def predict_siglip(model, image, device):
    if model.training:
        raise ValueError("Call model.eval() before inference")
    inputs = siglip_input(model, image).to(device)
    synchronize(torch.device(device))
    started = time.perf_counter()
    with torch.inference_mode():
        logits = model(inputs)[0]
        scores = logits.sigmoid() if model.mode == "zero_shot" else logits.softmax(0)
    synchronize(torch.device(device))
    elapsed = (time.perf_counter() - started) * 1000
    values = scores.cpu().tolist()
    if len(values) != 4 or not np.isfinite(values).all():
        raise RuntimeError("Invalid SigLIP 2 scores")
    target = int(logits.argmax())
    return {
        "class_id": target,
        "label": CLASSES[target],
        "confidence": values[target],
        "probabilities": values,
        "score_type": "sigmoid_matching" if model.mode == "zero_shot" else "softmax",
        "logits": logits.cpu().tolist(),
        "latency_ms": elapsed,
    }


def patch_occlusion(model, inputs, target=None, grid=7, batch_size=7):
    """Fixed-class logit sensitivity to equal image regions; not attention."""
    if model.training or tuple(inputs.shape) != (1, 3, 224, 224):
        raise ValueError("Occlusion requires eval mode and one 224-square RGB input")
    if (
        type(grid) is not int
        or grid < 1
        or 224 % grid
        or 224 // grid % 16
        or type(batch_size) is not int
        or batch_size < 1
    ):
        raise ValueError("The grid must divide 224 into patch-aligned regions")
    if not torch.isfinite(inputs).all():
        raise ValueError("Non-finite input")
    device = inputs.device
    with torch.inference_mode():
        original = model(inputs)[0]
        target = int(original.argmax()) if target is None else target
        if type(target) is not int or target not in range(4):
            raise ValueError("Target must be a class index")
        side = 224 // grid
        variants = inputs.repeat(grid * grid, 1, 1, 1)
        for region in range(grid * grid):
            y, x = divmod(region, grid)
            variants[
                region, :, y * side : (y + 1) * side, x * side : (x + 1) * side
            ] = 0
        scores = torch.cat(
            [
                model(variants[i : i + batch_size])[:, target]
                for i in range(0, len(variants), batch_size)
            ]
        )
        signed = (original[target] - scores).reshape(grid, grid).cpu().numpy()
    positive = np.maximum(signed, 0)
    normalized = positive / positive.max() if positive.max() else positive
    heatmap = np.asarray(
        Image.fromarray(normalized).resize((224, 224), Image.Resampling.BILINEAR)
    ).copy()
    return heatmap, {
        "method": "Class-conditioned image-region occlusion",
        "target_class": target,
        "grid": grid,
        "region_pixels": side,
        "replacement": "normalized zero = RGB mean 0.5",
        "original_logit": float(original[target].cpu()),
        "signed_logit_drops": signed.tolist(),
        "device": str(device),
    }


def siglip_overlay(image, heatmap, alpha=0.5):
    """Invert the processor's direct resize, preserving the full photograph."""
    import matplotlib.pyplot as plt

    if (
        not 0 <= alpha <= 1
        or np.asarray(heatmap).shape != (224, 224)
        or not np.isfinite(heatmap).all()
        or np.any(np.asarray(heatmap) < 0)
        or np.any(np.asarray(heatmap) > 1)
    ):
        raise ValueError("Invalid opacity or heatmap")
    mapped = np.asarray(
        Image.fromarray(heatmap).resize(image.size, Image.Resampling.BILINEAR)
    ).clip(0, 1)
    opacity = (alpha * mapped)[..., None]
    color = plt.get_cmap("inferno")(mapped)[..., :3] * 255
    return Image.fromarray(
        np.rint(np.asarray(image.convert("RGB")) * (1 - opacity) + color * opacity)
        .clip(0, 255)
        .astype(np.uint8)
    )
