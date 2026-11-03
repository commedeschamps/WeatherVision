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
from .utils import ROOT, select_device, sha256_file, write_json


TARGET_STAGES = {"final": 7, "earlier": 5}


def gradcam_convnext(model, x, target=None, layer="final", return_details=False):
    """Grad-CAM of a class logit, with the original final-stage default.

    alpha_k = mean_ij(d logit_c / d A_kij); L = ReLU(sum_k alpha_k A_k).
    Bilinear interpolation is followed by the existing within-map min/max scale.
    Earlier-stage maps are complementary attributions, not equivalent resolutions.
    """
    import operator
    from torchvision.models.convnext import ConvNeXt
    if not isinstance(model, ConvNeXt) or layer not in TARGET_STAGES:
        raise ValueError("Choose final or earlier on a TorchVision ConvNeXt model")
    if x.ndim != 4 or tuple(x.shape[:2]) != (1, 3) or min(x.shape[-2:]) < 32:
        raise ValueError("Grad-CAM expects one RGB NCHW input image")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError("Grad-CAM input must contain finite floating-point values")
    if target is not None:
        try:
            if isinstance(target, (bool, np.bool_)):
                raise TypeError
            target = operator.index(target)
        except TypeError as error:
            raise ValueError("Target class must be an integer in [0, 3]") from error
        if target not in range(4):
            raise ValueError("Target class must be in [0, 3]")
    captured = []
    stage = model.features[TARGET_STAGES[layer]]
    handle = stage.register_forward_hook(lambda module, inputs, output: captured.append(output))
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        # Also works when the caller entered inference_mode. Only the cloned input
        # gains a graph; autograd.grad does not populate parameter .grad buffers.
        with torch.inference_mode(False), torch.enable_grad():
            inputs = x.detach().clone().requires_grad_(True)
            logits = model(inputs)
            if logits.shape != (1, 4) or not torch.isfinite(logits).all():
                raise ValueError("Grad-CAM requires four finite class logits")
            index = int(logits.argmax(1).item()) if target is None else target
            if len(captured) != 1 or captured[0].ndim != 4:
                raise ValueError("Target stage must return one spatial NCHW activation")
            activation = captured[0]
            gradient = torch.autograd.grad(logits[0, index], activation)[0]
            if not torch.isfinite(activation).all() or not torch.isfinite(gradient).all():
                raise ValueError("Non-finite Grad-CAM activations or gradients")
            weights = gradient.mean(dim=(2, 3), keepdim=True)
            native = F.relu((weights * activation).sum(1, keepdim=True))
            heatmap = F.interpolate(native, size=inputs.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            heatmap = heatmap - heatmap.min()
            peak = heatmap.max()
            if peak > 0:
                heatmap = heatmap / peak
            result = heatmap.detach().cpu().numpy()
            if not return_details:
                return result
            return result, {"target_class": index, "target_logit": float(logits[0, index].detach().cpu()),
                            "target_layer": f"features[{TARGET_STAGES[layer]}]", "stage": layer,
                            "activation_shape": list(activation.shape), "gradient_shape": list(gradient.shape),
                            "native_heatmap": native[0, 0].detach().cpu().numpy(),
                            "gradient_abs_max": float(gradient.abs().max().detach().cpu()),
                            "spatially_constant": bool(peak == 0),
                            "normalization": "ReLU; bilinear resize, align_corners=False; subtract min then divide by max if positive"}
    finally:
        handle.remove()
        # Restore every flag, including a training head with an eval-mode backbone.
        for module, training in modes:
            module.training = training


def map_heatmap_to_original(original, heatmap, architecture="convnext"):
    """Invert resize/center-crop geometry; zero attribution outside crop bounds.

    The geometric field of view is defined at original-pixel centers. Resampling
    may otherwise bleed a fractional tint beyond its edge.
    """
    values = np.asarray(heatmap, dtype=np.float32)
    if values.shape != (224, 224) or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("Heatmap must be finite, normalized to [0, 1], and 224 by 224")
    if original.width * original.height > 25_000_000 or max(original.size) / min(original.size) > 64:
        raise ValueError("Use an image under 25 megapixels and an aspect ratio at most 64:1")
    resized = TF.resize(original, resize_size(architecture))
    width, height = resized.size
    left, top = int(round((width - 224) / 2)), int(round((height - 224) / 2))
    canvas = Image.new("F", (width, height), 0)
    canvas.paste(Image.fromarray(values), (left, top))
    mapped = np.asarray(canvas.resize(original.size, Image.Resampling.BILINEAR)).copy().clip(0, 1)
    y, x = np.ogrid[:original.height, :original.width]
    visible = (((x + .5) * width / original.width >= left) & ((x + .5) * width / original.width < left + 224)
               & ((y + .5) * height / original.height >= top) & ((y + .5) * height / original.height < top + 224))
    mapped[~visible] = 0
    return mapped


def heatmap_image(heatmap):
    """Inferno rendering of unchanged normalized attribution values."""
    values = np.asarray(heatmap)
    if values.ndim != 2 or not np.isfinite(values).all() or values.min() < 0 or values.max() > 1:
        raise ValueError("Heatmap must be a finite normalized 2D array")
    return Image.fromarray(np.rint(plt.get_cmap("inferno")(values)[..., :3] * 255).astype(np.uint8))


def overlay_image(original, heatmap, architecture="convnext", alpha=0.40):
    """Strength-weighted overlay; alpha changes display only, never attribution."""
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("Overlay opacity must be between zero and one")
    original = original.convert("RGB")
    mapped = map_heatmap_to_original(original, heatmap, architecture)
    pixels = np.asarray(original)
    result = pixels.copy()
    # Bound colormap/blending temporaries for large accepted photographs.
    cmap = plt.get_cmap("inferno")
    for top in range(0, original.height, 128):
        values = mapped[top:top + 128]
        active = alpha * values > 0
        if active.any():
            opacity = (alpha * values[active])[:, None]
            color = cmap(values[active])[:, :3]
            source = pixels[top:top + 128][active]
            blended = source * (1 - opacity) + color * 255 * opacity
            result[top:top + 128][active] = np.rint(blended).clip(0, 255).astype(np.uint8)
    return Image.fromarray(result)


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
