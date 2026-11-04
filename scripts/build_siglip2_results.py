"""Verify final metrics, benchmark models consistently and build SigLIP 2 figures."""

from pathlib import Path
import sys, json, hashlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np, pandas as pd, torch
from src.data import open_rgb, transform, verify_content
from src.engine import load_checkpoint, calculate_metrics
from src.evaluation import benchmark
from src.siglip2 import (
    load_encoder,
    load_siglip_checkpoint,
    siglip_input,
    patch_occlusion,
    siglip_overlay,
    REVISION,
)
from src.siglip_experiments import experiment_name
from src.utils import (
    CLASSES,
    ROOT,
    seed_everything,
    select_device,
    sha256_file,
    write_json,
)


def build_architecture_figure(directory):
    """Render the verified architecture without benchmarking or model inference."""
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.set(xlim=(0, 10), ylim=(0, 3.4))
    ax.axis("off")
    boxes = [
        (0.1, "RGB photograph", "224 × 224 × 3"),
        (2.1, "16 × 16 patches", "196 tokens × 768"),
        (4.1, "Vision Transformer", "12 layers / 12 heads"),
        (6.1, "Pretrained\nattention pool", "768 features"),
        (8.1, "New linear head", "4 logits → Softmax"),
    ]
    for index, (x, title, detail) in enumerate(boxes):
        ax.add_patch(
            FancyBboxPatch(
                (x, 2.1),
                1.7,
                0.85,
                boxstyle="round,pad=.04",
                fc="#E8F1ED",
                ec="#55796D",
            )
        )
        ax.text(x + 0.85, 2.67, title, ha="center", va="center", fontsize=8.7, weight="bold")
        ax.text(x + 0.85, 2.32, detail, ha="center", fontsize=8.5)
        if index:
            ax.annotate(
                "",
                xy=(x - 0.08, 2.52),
                xytext=(x - 0.32, 2.52),
                arrowprops={"arrowstyle": "->", "color": "#355F51"},
            )
    ax.text(
        0.1,
        1.5,
        "Frozen SigLIP 2 vision encoder (2025); only the four-class linear head trains.",
        fontsize=11,
    )
    ax.text(
        0.1,
        0.98,
        "Zero-shot branch: normalize pooled image features and pretrained text features;",
        fontsize=10.5,
    )
    ax.text(
        0.1,
        0.58,
        "dot product → pretrained scale + bias → four independent Sigmoid matching scores.",
        fontsize=10.5,
    )
    ax.text(
        0.1,
        0.13,
        "Processor: full-image bilinear resize to 224 × 224; RGB mean/std 0.5; no center crop.",
        fontsize=10,
    )
    fig.savefig(directory / "architecture.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / "architecture.svg", bbox_inches="tight")
    plt.close(fig)


def build_results(device_name="auto"):
    seed_everything(42)
    device = select_device(device_name)
    directory = ROOT / "outputs/siglip2"
    frame = pd.read_csv(ROOT / "data/manifest.csv")
    verify_content(frame)
    test = frame[frame.split == "test"].reset_index(drop=True)
    expected_membership = hashlib.sha256("\n".join(test.path).encode()).hexdigest()
    config = json.loads((ROOT / "configs/siglip2.json").read_text())
    runs = [
        "cnn_baseline",
        "mobilenet_improved",
        "convnext_aug",
        "convnext_noaug",
        "siglip2_zero_shot",
    ] + [experiment_name(a, s) for s in config["seeds"] for a in [True, False]]
    encoder, processor, provenance = load_encoder()
    encoder.to(device)
    bundle = (encoder, processor, provenance)
    rows = []
    heads = []
    image = open_rgb(ROOT / test.iloc[0].path)
    for run in runs:
        run_dir = ROOT / "outputs" / run
        m = json.loads((run_dir / "test_metrics.json").read_text())
        status = json.loads((run_dir / "status.json").read_text())
        pred = pd.read_csv(run_dir / "test_predictions.csv")
        assert (
            pred.path.tolist() == test.path.tolist()
            and m["test_membership_sha256"] == expected_membership
        )
        assert m["checkpoint_sha256"] == sha256_file(run_dir / "best.pt")
        recomputed = calculate_metrics(
            pred.class_id.tolist(), pred.predicted_class_id.tolist()
        )
        for key in [
            "accuracy",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "confusion_matrix",
        ]:
            np.testing.assert_allclose(recomputed[key], m[key], atol=1e-12)
        if run.startswith("siglip2"):
            model, checkpoint = load_siglip_checkpoint(
                run_dir / "best.pt", device, encoder_bundle=bundle
            )
            sample = siglip_input(model, image)
            with torch.inference_mode():
                actual = model(sample.to(device))[0].argmax().item()
            assert actual == int(pred.iloc[0].predicted_class_id)
            heads.append(
                {
                    "run": run,
                    "path": f"outputs/{run}/best.pt",
                    "bytes": (run_dir / "best.pt").stat().st_size,
                    "sha256": sha256_file(run_dir / "best.pt"),
                }
            )
        else:
            model, checkpoint = load_checkpoint(run_dir / "best.pt", device)
            sample = transform(architecture=checkpoint["config"]["model"])(
                image
            ).unsqueeze(0)
        timing = benchmark(model, sample, device)
        write_json(directory / "benchmarks" / f"{run}.json", timing)
        rows.append(
            {
                "run": run,
                "model": status["config"]["model"],
                "augment": status.get(
                    "augment", status["config"].get("augment", False)
                ),
                **{
                    key: m[key]
                    for key in [
                        "accuracy",
                        "precision_macro",
                        "recall_macro",
                        "f1_macro",
                        "test_samples",
                    ]
                },
                "latency_mean_ms": timing["mean_ms"],
                "latency_median_ms": timing["median_ms"],
                "parameters": status["parameters"],
                "trainable_parameters": status["trainable_parameters"],
                "training_seconds": status["training_seconds"],
                "best_epoch": status["best_epoch"],
                "validation_f1_macro": status["best_val_f1_macro"],
                "device": str(device),
                "seed": status.get("seed", 42),
            }
        )
        print(run, m["accuracy"], m["f1_macro"], flush=True)
    comparison = pd.DataFrame(rows)
    comparison.to_csv(directory / "model_comparison.csv", index=False)
    write_json(
        ROOT / "configs/siglip2_artifacts.json",
        {
            "model_id": config["model_id"],
            "revision": REVISION,
            "license": "Apache-2.0",
            "heads": heads,
            "pretrained_install": "python scripts/install_siglip2.py",
            "vision_weights_sha256": provenance["files"]["model.safetensors"],
        },
    )
    seed_rows = pd.read_csv(directory / "per_seed.csv")
    summary = pd.read_csv(directory / "ablation_summary.csv")
    aug = summary[summary.augment].iloc[0]
    plain = summary[~summary.augment].iloc[0]
    interpretation = f"Across three seeds, augmentation mean accuracy was {aug.accuracy_mean:.2%} (sample SD {aug.accuracy_std:.2%}), versus {plain.accuracy_mean:.2%} (SD {plain.accuracy_std:.2%}) without augmentation. Mean macro-F1 was {aug.f1_mean:.4f} versus {plain.f1_mean:.4f}. These three-seed descriptive results do not establish statistical significance."
    write_json(
        directory / "ablation_analysis.json",
        {"interpretation": interpretation, "seeds": config["seeds"], "std_ddof": 1},
    )
    class_rows = []
    for seed in config["seeds"]:
        a = json.loads(
            (ROOT / "outputs" / experiment_name(True, seed) / "status.json").read_text()
        )
        b = json.loads(
            (
                ROOT / "outputs" / experiment_name(False, seed) / "status.json"
            ).read_text()
        )
        assert a["initial_head_sha256"] == b["initial_head_sha256"]
        assert a["protocol"] == b["protocol"]
        for augment in [True, False]:
            m = json.loads(
                (
                    ROOT
                    / "outputs"
                    / experiment_name(augment, seed)
                    / "test_metrics.json"
                ).read_text()
            )
            for label in CLASSES:
                class_rows.append(
                    {
                        "seed": seed,
                        "augment": augment,
                        "class": label,
                        **m["per_class"][label],
                    }
                )
    pd.DataFrame(class_rows).to_csv(directory / "per_class_seeds.csv", index=False)
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    names = [
        "Simple CNN",
        "MobileNetV2",
        "ConvNeXt (historical)",
        "SigLIP 2 zero-shot",
        "SigLIP 2 probe / s42",
    ]
    chosen = comparison.set_index("run").loc[
        [
            "cnn_baseline",
            "mobilenet_improved",
            "convnext_aug",
            "siglip2_zero_shot",
            "siglip2_aug",
        ]
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, metric, title in zip(
        axes, ["accuracy", "f1_macro"], ["Test accuracy", "Test macro-F1"]
    ):
        bars = ax.barh(
            names,
            chosen[metric],
            color=["#767676", "#9EADBA", "#9AB7A8", "#AADCA9", "#0F4D92"],
        )
        ax.set(xlim=(0, 1.08), xlabel=title)
        ax.invert_yaxis()
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    for suffix in ["png", "svg"]:
        fig.savefig(directory / f"comparison.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 3.7), layout="constrained")
    for i, (field, label) in enumerate(
        [("accuracy", "Accuracy"), ("f1_macro", "Macro-F1")]
    ):
        for j, augment in enumerate([True, False]):
            vals = seed_rows[seed_rows.augment == augment][field].to_numpy()
            position = i * 3 + j
            ax.bar(
                position,
                vals.mean(),
                yerr=vals.std(ddof=1),
                capsize=4,
                color="#0F4D92" if augment else "#AADCA9",
                edgecolor="#272727",
                linewidth=0.7,
            )
            ax.scatter(
                np.linspace(position - 0.12, position + 0.12, len(vals)),
                vals,
                color="#272727",
                s=20,
                zorder=3,
            )
    ax.set(
        xticks=[0, 1, 3, 4],
        xticklabels=["Aug.", "No aug."] * 2,
        ylim=(0, 1.08),
        ylabel="Test score",
        title="Three seeds: mean, sample SD and individual runs",
    )
    ax.text(0.21, -0.19, "Accuracy", transform=ax.transAxes, ha="center")
    ax.text(0.79, -0.19, "Macro-F1", transform=ax.transAxes, ha="center")
    fig.savefig(directory / "ablation.png", dpi=300, bbox_inches="tight")
    fig.savefig(directory / "ablation.svg", bbox_inches="tight")
    plt.close(fig)
    build_architecture_figure(directory)
    model, _ = load_siglip_checkpoint(
        ROOT / "outputs/siglip2_aug/best.pt", device, encoder_bundle=bundle
    )
    predictions = pd.read_csv(ROOT / "outputs/siglip2_aug/test_predictions.csv")
    chosen = []
    for label in CLASSES:
        subset = predictions[(predictions.label == label) & predictions.correct]
        if len(subset):
            chosen.append(subset.iloc[0])
    chosen += [row for _, row in predictions[~predictions.correct].head(2).iterrows()]
    metadata = []
    for index, row in enumerate(chosen):
        original = open_rgb(ROOT / row.path)
        inputs = siglip_input(model, original).to(device)
        heatmap, details = patch_occlusion(model, inputs, int(row.predicted_class_id))
        overlay = siglip_overlay(original, heatmap)
        if index == 0:
            changed = inputs.clone()
            changed[:, :, :32, :32] = 0
            with torch.inference_mode():
                drop = float(
                    (
                        model(inputs)[0, int(row.predicted_class_id)]
                        - model(changed)[0, int(row.predicted_class_id)]
                    ).cpu()
                )
            assert abs(drop - details["signed_logit_drops"][0][0]) < 1e-4
        np.save(directory / f"occlusion_{index+1:02d}.npy", heatmap)
        write_json(directory / f"occlusion_{index+1:02d}.json", details)
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), layout="constrained")
        for a, picture, title in zip(
            axes,
            [original, overlay, heatmap],
            [
                f"True: {row.label.title()}",
                f"Predicted: {row.predicted_label.title()}",
                "Positive logit sensitivity",
            ],
        ):
            a.imshow(
                picture, cmap="inferno" if isinstance(picture, np.ndarray) else None
            )
            a.set_title(title, fontsize=11)
            a.axis("off")
        figure = f"occlusion_{index+1:02d}.png"
        fig.savefig(directory / figure, dpi=300, bbox_inches="tight")
        plt.close(fig)
        metadata.append(
            {
                "path": row.path,
                "true": row.label,
                "predicted": row.predicted_label,
                "correct": bool(row.correct),
                "figure": figure,
            }
        )
    write_json(
        directory / "occlusion_examples.json",
        {
            "method": "Class-conditioned image-region occlusion",
            "grid": 7,
            "region_pixels": 32,
            "replacement": "RGB mean / normalized zeros",
            "device": str(device),
            "independent_region_recheck": "passed",
            "examples": metadata,
        },
    )
    print(interpretation, flush=True)


if __name__ == "__main__":
    build_results()
