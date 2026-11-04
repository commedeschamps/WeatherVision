"""Build the eight-page A2 report from completed SigLIP 2 experiments only."""

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from pypdf import PdfReader
from build_reports import render_pdf, table


def build_report():
    out = ROOT / "outputs/siglip2"
    required = [
        "model_comparison.csv",
        "per_seed.csv",
        "ablation_summary.csv",
        "occlusion_examples.json",
        "selection_lock.json",
    ]
    if any(not (out / name).is_file() for name in required):
        raise RuntimeError(
            "Complete training, locked test evaluation and figure generation before replacing the official A2 report"
        )
    comparison = pd.read_csv(out / "model_comparison.csv").set_index("run")
    seeds = pd.read_csv(out / "per_seed.csv")
    summary = pd.read_csv(out / "ablation_summary.csv")
    assert len(seeds) == 6 and set(seeds.seed) == {42, 43, 44}
    aug = comparison.loc["siglip2_aug"]
    plain = comparison.loc["siglip2_noaug"]
    zero = comparison.loc["siglip2_zero_shot"]
    conv = comparison.loc["convnext_aug"]
    prompts = pd.read_csv(ROOT / "outputs/siglip2_zero_shot/prompt_validation.csv")
    selected = int(prompts.loc[prompts.validation_f1_macro.idxmax(), "prompt_set"])
    examples = json.loads((out / "occlusion_examples.json").read_text())["examples"]
    good = next(e for e in examples if e["correct"])
    bad = next((e for e in examples if not e["correct"]), None)
    provenance = json.loads((ROOT / "models/siglip2/provenance.json").read_text())
    augstatus = json.loads((ROOT / "outputs/siglip2_aug/status.json").read_text())
    relation = (
        "higher than"
        if aug.accuracy > conv.accuracy
        else "equal to" if aug.accuracy == conv.accuracy else "lower than"
    )
    classification = table(
        ["Model", "Accuracy", "Macro P", "Macro R", "Macro-F1"],
        [
            [
                name,
                f'{comparison.loc[run,"accuracy"]:.2%}',
                f'{comparison.loc[run,"precision_macro"]:.4f}',
                f'{comparison.loc[run,"recall_macro"]:.4f}',
                f'{comparison.loc[run,"f1_macro"]:.4f}',
            ]
            for run, name in [
                ("cnn_baseline", "Simple CNN / A1"),
                ("mobilenet_improved", "MobileNetV2 / A1"),
                ("convnext_aug", "ConvNeXt + aug. / historical"),
                ("convnext_noaug", "ConvNeXt no aug. / historical"),
                ("siglip2_zero_shot", "SigLIP 2 zero-shot"),
                ("siglip2_aug", "SigLIP 2 + aug. / seed 42"),
                ("siglip2_noaug", "SigLIP 2 no aug. / seed 42"),
            ]
        ],
    )
    seedtable = table(
        ["Seed", "Aug.", "Accuracy", "Macro-F1", "Best epoch"],
        [
            [
                int(r.seed),
                "Yes" if r.augment else "No",
                f"{r.accuracy:.2%}",
                f"{r.f1_macro:.4f}",
                int(r.best_epoch),
            ]
            for r in seeds.itertuples()
        ],
    )
    summarytable = table(
        ["Treatment", "Accuracy mean / SD", "Macro-F1 mean / SD", "Seeds"],
        [
            [
                "Augmentation" if r.augment else "No augmentation",
                f"{r.accuracy_mean:.2%} / {r.accuracy_std:.2%}",
                f"{r.f1_mean:.4f} / {r.f1_std:.4f}",
                int(r.seeds),
            ]
            for r in summary.itertuples()
        ],
    )
    costs = table(
        ["Model", "Parameters / trainable", "Train s", "Mean forward ms"],
        [
            [
                name,
                f'{int(comparison.loc[run,"parameters"]):,} / {int(comparison.loc[run,"trainable_parameters"]):,}',
                f'{comparison.loc[run,"training_seconds"]:.1f}',
                f'{comparison.loc[run,"latency_mean_ms"]:.2f}',
            ]
            for run, name in [
                ("cnn_baseline", "Simple CNN"),
                ("mobilenet_improved", "MobileNetV2"),
                ("convnext_aug", "ConvNeXt / historical"),
                ("siglip2_zero_shot", "SigLIP 2 zero-shot"),
                ("siglip2_aug", "SigLIP 2 probe / s42"),
            ]
        ],
    )
    pages = []
    pages.append(
        f"""# WeatherVision: SigLIP 2 for Weather Image Classification

**Assignment 2 - Rakhman Seidigali, SE-2411**

Computer Vision, Astana IT University. Instructor: Nurym Kaiyrkhan.

## 1. Introduction and motivation

WeatherVision assigns Cloudy, Rain, Shine or Sunrise to an outdoor photograph. Assignment 1 established a Simple CNN and frozen-feature MobileNetV2 comparison. This revision introduces Google's SigLIP 2 (2025) as the primary advanced method, while preserving ConvNeXt-Tiny as a historical A2 experiment. The goal is to evaluate modern visual-semantic representations on a small labeled dataset, rather than assume a newer model must win.

SigLIP 2 is a vision-language model pretrained on WebLI with an image-text objective and additional representation-learning techniques [2]. Its visual encoder can be used with text descriptions without fitting a weather classifier, or with a newly trained linear head. Comparing these routes separates pretrained semantic matching from supervised adaptation to the four labels.

**Measured findings.** On the fixed 165-image test set, zero-shot SigLIP 2 obtains {zero.accuracy:.2%} accuracy and {zero.f1_macro:.4f} macro-F1. The predeclared seed-42 augmented linear probe obtains {aug.accuracy:.2%} and {aug.f1_macro:.4f}. Its accuracy is {relation} historical augmented ConvNeXt's 98.18%. Three matched seeds were completed for each augmentation treatment. All figures and tables derive from saved predictions and histories; no original experiment was retrained or relabeled.

The contribution is a reproducible comparison, a functioning Transformer-based application and an explicit image-region sensitivity explanation. It is not a claim of universal superiority, calibrated certainty or causal understanding. The following sections describe the unchanged data, official preprocessing, frozen-backbone protocol and measured outcomes."""
    )
    pages.append(
        """## 2. Dataset and preprocessing

Ajayi's Multi-class Weather Dataset [1] is used through the resized PerceptiLabs mirror, pinned to commit ad78ac8fcf1b1aae229ca6e5310aa75b06783787. The collection has 1,123 valid photographs. Removing 26 exact copies retains 1,097 images in 1,011 detected groups. File and decoded-pixel SHA-256 identify exact duplicates. Conservative pHash/dHash, aspect-ratio and thumbnail-color criteria group near duplicates before splitting. Detected groups remain in a single partition.

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Cloudy | 204 | 43 | 44 |
| Rain | 143 | 31 | 31 |
| Shine | 175 | 37 | 37 |
| Sunrise | 246 | 53 | 53 |
| Total | 768 | 164 | 165 |

![Figure 1. Original training photographs; no generated photographs enter training.](../data/inspection.png)

The original manifest and seed 42 remain unchanged. SigLIP 2 uses its official image processor: bilinear full-image resize to 224 by 224, rescaling by 1/255 and RGB means/standard deviations of 0.5. Unlike historical ConvNeXt, it has no center crop or ImageNet channel statistics. This resize can distort aspect ratio, a limitation of the selected fixed-resolution checkpoint.

Training augmentation applies random resized crop (scale 0.8-1.0, ratio 0.9-1.1) and horizontal flip with probability 0.5 before the official processor. Weather color is preserved by omitting color jitter; no rotation introduces black corners. Validation and test images are deterministic."""
    )
    pages.append(
        f"""## 3. SigLIP 2 architecture and adaptation

The exact checkpoint is google/siglip2-base-patch16-224, revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2, Apache-2.0 [3]. Transformers 4.57.6 provides its fixed-resolution SiglipVisionModel/SiglipModel API. The API's siglip model type does not mean older SigLIP weights were substituted. Genuine CPU and MPS forward passes verified the cached official weights and preprocessing.

A Vision Transformer divides the image into 196 non-overlapping 16-square patches [4]. Patch embeddings and positional information feed twelve Transformer layers with twelve attention heads and 768-dimensional features. Self-attention relates spatial tokens across the image; feed-forward blocks transform their representations. The pretrained attention-pooling head produces one 768-element image vector. This model does not require an invented CLS/register-token average.

![Figure 2. Verified image path and the supervised and zero-shot branches.](../outputs/siglip2/architecture.png)

**Zero-shot branch.** The pretrained text encoder produces vectors for four English descriptions. Image/text vectors are normalized; their dot products receive the pretrained logit scale and bias. Sigmoid produces four independent image-text matching scores. They do not sum to one and are not calibrated class probabilities. Three prespecified prompt sets were compared on validation data only; set {selected} was selected by macro-F1. The selected prompts are cloudy weather, rainy weather, sunny weather and sunrise. Its frozen text vectors are cached for inference.

**Linear probe.** The official attention-pooled visual representation feeds a new Linear(768,4) head. Cross-entropy trains only {int(aug.trainable_parameters):,} head parameters; the {provenance['parameters']:,}-parameter encoder remains frozen and in evaluation mode. Supervised Softmax scores sum to one but remain uncalibrated. This is linear probing, not full-backbone fine-tuning."""
    )
    pages.append(
        f"""## 4. Experimental setup and selection

Both supervised treatments use seeds 42, 43 and 44, cross-entropy, AdamW, learning rate 0.001, weight decay 0.0001, head batch size 32, up to 25 epochs and early-stopping patience five. The larger epoch budget allows adaptation beyond historical ConvNeXt's six epochs. All settings are fixed across augmentation treatments; the main demonstration is the predeclared augmented seed-42 model, independent of test performance.

The frozen encoder runs on Apple MPS, with correct CPU/CUDA support. Deterministic train-without-augmentation and validation features may be cached. Augmented training representations are generated anew for every seed and epoch. No single cached deterministic view is misrepresented as dynamic augmentation. The lightweight head is optimized on CPU over the extracted features, with independent epoch-seeded shuffling. Backbone gradients are never stored.

Highest validation macro-F1 selects the checkpoint; ties keep the earliest epoch. Initial-head hashes match within each seed's augmentation pair. Prompt selection and all six supervised runs completed before a selection lock was written and new methods accessed the test set. Test labels select neither prompts nor epochs.

![Figure 3. Seed-42 augmented probe: original training and validation curves.](../outputs/siglip2_aug/training_curves.png)

The augmented seed-42 run selected epoch {int(aug.best_epoch)} and reached validation macro-F1 {aug.validation_f1_macro:.4f}. Its recorded wall time was {aug.training_seconds:.1f} s, including feature extraction and head optimization. Cache reuse makes elapsed times dependent on cache state; historical training times cover a narrower original scope. All six histories, selected heads and test predictions are retained. Single-device timing is descriptive, not a portable hardware guarantee."""
    )
    pages.append(
        f"""## 5. Results and comparison with Assignment 1

**Table 1. Fixed-test classification results.**

{classification}

The supervised seed-42 SigLIP 2 accuracy is {relation} ConvNeXt's historical result. The frozen visual-semantic representation and small weather head operate under a different pretraining and preprocessing recipe; model year alone does not determine performance. Zero-shot uses no weather-label optimization, while its prompt choice uses validation labels. The supervised head has access to the 768 labeled training images.

**Table 2. Recorded model size, training time and new common-protocol latency.**

{costs}

The latency column is a new common-device benchmark ({str(comparison.loc['siglip2_aug','device']).upper()}): batch one, five warm-ups and thirty synchronized network forwards on the same first test photograph with each model's correct processor. It excludes decoding, preprocessing, transfer, loading and Softmax/Sigmoid. Zero-shot text vectors are cached; text encoding is excluded. Historical benchmark files remain unchanged. Training-time scopes and feature-cache reuse differ, so these values should not be interpreted as a controlled training-efficiency experiment.

![Figure 4. Original seed-42 augmented SigLIP 2 confusion matrix; true labels are rows.](../outputs/siglip2_aug/confusion_matrix.png)"""
    )
    pages.append(
        f"""## 6. Three-seed augmentation ablation

Each pair changes only the augmentation treatment. Architecture, revision, split, head initialization seed, optimizer, batch size, epoch cap and validation selection are fixed. Per-seed checkpoint selection remains independent of test performance.

**Table 3. Individual runs.**

{seedtable}

**Table 4. Descriptive aggregation (sample standard deviation, ddof=1).**

{summarytable}

![Figure 5. Mean, sample standard deviation and individual seed results.](../outputs/siglip2/ablation.png)

Three seeds expose variation hidden by a single run, but remain too few for a broad significance claim. Differences should be read together with the 165-image denominator and class-wise precision/recall/F1. The repository retains per-class scores for every seed. Augmentation changes the views presented to a frozen encoder; it can improve robustness or remove useful weather cues. Neither direction is assumed before measurement."""
    )
    explanation_images = f"![Figure 6. Correct prediction: {good['true']} classified as {good['predicted']}.](../outputs/siglip2/{good['figure']})"
    if bad:
        explanation_images += f"\n\n![Figure 7. Error: {bad['true']} classified as {bad['predicted']}.](../outputs/siglip2/{bad['figure']})"
    pages.append(
        f"""## 7. Transformer-compatible explainability

Class-conditioned image-region occlusion measures sensitivity of the original predicted class logit. The model-visible 224-square tensor is divided into 49 equal 32-square regions, each covering two by two native ViT patches. One region at a time is replaced by normalized zeros, the checkpoint RGB mean of 0.5. Batched inference measures original logit minus occluded logit while holding the target class fixed.

Signed drops are retained. Positive drops are normalized for display and mapped through the processor's full-image resize onto the photograph. Opacity changes blending, not measured importance. A single-region independent recomputation matches the batched result; deterministic settings and raw signed maps are saved. The interface explicitly calls this occlusion, not Grad-CAM or native Transformer attention.

{explanation_images}

These interventions create artificial images. Regions interact through attention, and positive-only overlays omit suppressive responses. A coarse grid, one replacement baseline and selected examples do not establish segmentation or causal understanding. The map probes the trained model's response, not ground-truth weather localization. Historical ConvNeXt Grad-CAM remains available as a different explanation method."""
    )
    screenshot = (
        "![Figure 8. Updated working SigLIP 2 application.](../outputs/siglip2/app.png)"
        if (out / "app.png").exists()
        else ""
    )
    pages.append(
        f"""## 8. Application, ethics, limitations and conclusion

The Streamlit application defaults to the genuinely trained SigLIP 2 Linear Probe. It also offers zero-shot matching, Simple CNN, MobileNetV2 and historical ConvNeXt. Upload, camera and dataset-example workflows show four clearly labeled scores, device/environment information, latency and downloadable results. Model/processor caching avoids reloads on ordinary reruns. Missing assets produce a setup error; another model is never presented as SigLIP 2. Inference works offline after installation.

{screenshot}

The dataset is attributed under CC BY 4.0; the pretrained checkpoint is Apache-2.0. No generated photographs enter training. Uploaded images stay in server memory. Users should avoid sensitive photographs on shared servers. The previous video demonstrates ConvNeXt and is labeled historical; the revised recording materials describe actual SigLIP 2 behavior.

Limitations include small and ambiguous class labels, heuristic duplicate grouping, only three head-training seeds, frozen features, fixed-resolution distortion and imperfect explanations. Pretrained visual-semantic representations are relevant because they support both label-free semantic matching and adaptation with limited labeled data. The measured outcomes support that comparison without asserting universal superiority. Source code, checkpoints, histories, prompt selection, predictions and an executed complete notebook document reproducibility.

**References**

[1] Ajayi (2018). Multi-class Weather Dataset. DOI: 10.17632/4drtyfjtfy.1. PerceptiLabs mirror, pinned commit ad78ac8.

[2] Tschannen et al. (2025). SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features. arXiv:2502.14786.

[3] Google. Official checkpoint/model card: https://huggingface.co/google/siglip2-base-patch16-224. Transformers 4.57 SigLIP API: https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/siglip.

[4] Dosovitskiy et al. (2020/2021). An Image is Worth 16x16 Words. arXiv:2010.11929. Official SigLIP 2 implementation: google-research/big_vision.

[5] Sandler et al. (2018), MobileNetV2, arXiv:1801.04381; Liu et al. (2022), ConvNeXt, arXiv:2201.03545; Selvaraju et al. (2017), Grad-CAM, arXiv:1610.02391."""
    )
    text = "\n\n<!-- pagebreak -->\n\n".join(pages) + "\n"
    target = ROOT / "reports/assignment2_report.pdf"
    (ROOT / "reports/assignment2_report.md").write_text(text)
    render_pdf(text, target, "Rakhman Seidigali")
    count = len(PdfReader(target).pages)
    if not 6 <= count <= 8:
        raise RuntimeError(f"Report must be 6-8 pages, got {count}")
    print("Updated A2 report:", count, "pages")


if __name__ == "__main__":
    build_report()
