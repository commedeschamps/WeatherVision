from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF

from .data import open_rgb, resize_size, transform
from .engine import load_checkpoint
from .plots import save
from .utils import CLASSES, ROOT, select_device, sha256_file, write_json


def gradcam_convnext(model, x, target=None):
    """Gradient-weighted final spatial-feature maps (ConvNeXt features[-1])."""
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("Grad-CAM expects one NCHW input image")
    captured = []
    # The whole final feature stage returns B,C,H,W before pooling/LayerNorm/head.
    handle = model.features[-1].register_forward_hook(lambda module, inputs, output: captured.append(output))
    training = model.training
    model.eval()
    try:
        with torch.enable_grad():
            # Frozen extractors still need an input graph. Detach avoids reusing an
            # inference-mode tensor, and autograd.grad avoids mutating .grad fields.
            inputs = x.detach().clone().requires_grad_(True)
            logits = model(inputs)
            index = int(logits.argmax(1).item()) if target is None else int(target)
            if index not in range(4):
                raise ValueError("Target class must be in [0, 3]")
            activation = captured[-1]
            gradient = torch.autograd.grad(logits[0, index], activation)[0]
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            heatmap = F.relu((weights * activation).sum(1, keepdim=True))
            heatmap = F.interpolate(heatmap, size=inputs.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            heatmap = heatmap - heatmap.min()
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            return heatmap.detach().cpu().numpy()
    finally:
        handle.remove()
        model.train(training)


def overlay_image(original, heatmap, architecture="convnext", alpha=0.40):
    """Map the center-cropped model attribution back onto the original canvas.

    Image regions outside the model's field of view retain their original pixels.
    A zero CAM is shown as no overlay, rather than a misleading solid blue map.
    """
    original = original.convert("RGB")
    resized = TF.resize(original, resize_size(architecture))
    width, height = resized.size
    left, top = int(round((width - 224) / 2)), int(round((height - 224) / 2))
    cam = Image.fromarray(np.asarray(heatmap, dtype=np.float32)).resize((224, 224), Image.Resampling.BILINEAR)
    canvas = Image.new("F", (width, height), 0)
    canvas.paste(cam, (left, top))
    mapped = np.asarray(canvas.resize(original.size, Image.Resampling.BILINEAR)).clip(0, 1)
    color = plt.get_cmap("inferno")(mapped)[..., :3]
    # Attribution strength controls opacity; unseen borders remain untouched.
    opacity = (alpha * mapped)[..., None]
    overlay = np.asarray(original) / 255 * (1 - opacity) + color * opacity
    return Image.fromarray(np.uint8(np.clip(overlay * 255, 0, 255)))


def generate_examples(run_dir, manifest_path, root=ROOT, count=6, device_name="auto"):
    import json
    run_dir = Path(run_dir)
    predictions = pd.read_csv(run_dir / "test_predictions.csv")
    metrics = json.loads((run_dir / "test_metrics.json").read_text())
    if metrics["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("Grad-CAM manifest differs from evaluation")
    if metrics["checkpoint_sha256"] != sha256_file(run_dir / "best.pt"):
        raise ValueError("Checkpoint changed after evaluation")
    device = select_device(device_name)
    model, checkpoint = load_checkpoint(run_dir / "best.pt", device)
    if checkpoint["config"]["model"] != "convnext":
        raise ValueError("Grad-CAM examples require ConvNeXt")
    # Deterministic post-evaluation selection includes errors when available.
    wrong = predictions[predictions.class_id != predictions.predicted_class_id]
    right = predictions[predictions.class_id == predictions.predicted_class_id]
    selected = pd.concat([wrong.head(max(1, count // 2)), right.head(count)]).head(count)
    selected.to_csv(run_dir / "gradcam_examples.csv", index=False)
    records = []
    for i, row in enumerate(selected.itertuples()):
        original = open_rgb(root / row.path)
        tensor = transform(architecture="convnext")(original).unsqueeze(0).to(device)
        cam = gradcam_convnext(model, tensor, row.predicted_class_id)
        np.save(run_dir / f"gradcam_{i + 1:02d}.npy", cam)
        overlay = overlay_image(original, cam)
        fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), layout="constrained")
        axes[0].imshow(original)
        axes[0].set_title(f"Original · true: {row.label.title()}", fontsize=10)
        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM · predicted: {row.predicted_label.title()}", fontsize=10)
        for ax in axes:
            ax.axis("off")
        name = f"gradcam_{i + 1:02d}.png"
        save(fig, run_dir / name)
        records.append({"figure": name, "path": row.path, "true": row.label,
                        "predicted": row.predicted_label, "correct": row.label == row.predicted_label,
                        "heatmap_min": float(cam.min()), "heatmap_max": float(cam.max())})
    write_json(run_dir / "gradcam_metadata.json", {"target_layer": "features[-1] (final spatial stage, before pooling)",
               "checkpoint_sha256": metrics["checkpoint_sha256"], "examples": records,
               "errors_available": len(wrong), "note": "Post-evaluation examples; approximate feature attribution, not segmentation or causal proof. Original image preserved; center-crop mapping inverted for overlay."})
    return records
