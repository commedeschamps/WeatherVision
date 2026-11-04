# Grad-CAM method and validation

## Class-specific attribution

WeatherVision uses [Grad-CAM](https://arxiv.org/abs/1610.02391v3) to explain a ConvNeXt class logit. Channel weights are the spatial mean of the selected logit's derivatives with respect to each activation map. The positive weighted sum is resized bilinearly to 224×224 and normalized within that map. Constant maps remain zero.

The default final stage, `features[7]`, has 768 channels on a 7×7 grid. The experimental earlier stage, `features[5]`, has 384 channels on a 14×14 grid. Each map is normalized independently; color intensities are not comparable across layers or images. Higher resolution alone does not imply greater faithfulness.

The implementation enables gradients on a cloned image input while keeping the backbone frozen. It uses `autograd.grad` without populating parameter gradients, removes its hook after calculation and restores the original module modes. The overlay inverts the model's resize/center-crop geometry, leaving unseen pixels unchanged. Opacity affects blending only. The app offers original, overlay and raw views with the predicted target class, spatial resolution and downloads.

A bright region is approximate attribution to the chosen class. It is not a segmentation mask or proof of causal understanding. If an image labeled Cloudy is predicted as Sunrise, its explanation targets the Sunrise score; horizon or background responses can therefore be relevant to that prediction.

## Controlled occlusion experiment

The saved diagnostic uses 22 held-out images: four seeded correct predictions per class, all three errors and the six original Grad-CAM examples, with duplicate IDs removed. It deliberately includes errors and is not a representative population sample. Membership is recorded in `outputs/explainability_validation/samples.csv`.

Each model-visible 224×224 image is divided into 49 equal 32×32 cells. Five cells (10.20% of visible pixels) are replaced per intervention. Cell-mean attribution ranks the high and low masks with deterministic tie-breaking; ten random masks sample cells without replacement. The seed is 42 plus the fixed sample index, and random masks are shared between layers and replacement baselines.

Mean replacement uses normalized zeros, equivalent to ImageNet RGB means [0.485, 0.456, 0.406]. Blur replacement uses a Gaussian-blurred copy of the same input (31-square kernel, sigma 10), replacing only selected cells. The original predicted class stays fixed. The score is `original target logit - perturbed target logit`; a positive value means the intervention reduced that score. Softmax changes are secondary measurements.

The experiment used the saved ConvNeXt checkpoint in evaluation mode on Apple MPS. It produced 1,056 perturbation records without updating model tensors. Random repeats are first averaged within each image; summary means then give every image equal weight.

| Stage | Replacement | High logit drop | Low logit drop | Random logit drop | High > random |
|---|---|---:|---:|---:|---:|
| Earlier | Blur | 0.323 | 0.034 | 0.156 | 16/22 |
| Earlier | Mean | 0.320 | -0.074 | -0.017 | 22/22 |
| Final | Blur | 0.346 | 0.085 | 0.156 | 18/22 |
| Final | Mean | 0.381 | -0.037 | -0.017 | 21/22 |

![Equal-area occlusion results; bars are image means and points are individual images.](../outputs/explainability_validation/occlusion_comparison.png)

The earlier stage does not consistently improve the perturbation comparison, so the final stage remains the default. Baseline sensitivity also matters: for misclassified `cloud_267.jpeg`, the final-stage high/low/random drops are +0.248/-0.159/-0.005 with mean replacement, and -0.118/-0.120/-0.101 with blur. The latter interventions all increase the Sunrise logit.

## Limits and reproduction

Occlusion creates an artificial distribution shift. Equal area fixes the deletion budget but does not control region shape, location or context interactions. The study uses one checkpoint, one selected sample and one spatial scale. Random masks are controls, not population confidence intervals. Positive-only Grad-CAM omits suppressive contributions. No parameter- or label-randomization study was performed; visual plausibility alone does not validate an explanation ([Adebayo et al.](https://arxiv.org/abs/1810.03292)). Occlusion probes follow the motivation of [Zeiler and Fergus](https://arxiv.org/abs/1311.2901).

The validation directory retains checkpoint/manifest hashes and settings, sample IDs, raw attribution arrays, per-mask `scores.csv`, image-level paired scores, summary statistics and figures. The CSV is the maintained per-mask export.

```bash
python scripts/download_checkpoints.py --all
python weather_cv.py prepare
python scripts/validate_explanations.py --device auto
```

Use `--device cpu` for CPU execution. The script replaces only the diagnostic outputs in `outputs/explainability_validation/`; it does not train or rewrite official classification results.
