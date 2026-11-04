# WeatherVision

WeatherVision classifies outdoor photographs as **Cloudy, Rain, Shine or Sunrise**. Assignment 2 uses genuine **Google SigLIP 2 (2025)** for zero-shot matching and supervised linear probing. The original Assignment 1 CNN/MobileNetV2 and historical ConvNeXt experiments remain intact.

Rakhman Seidigali · SE-2411 · Computer Vision, Astana IT University<br>
Instructor: Nurym Kaiyrkhan

[Assignment 2 report](reports/assignment2_report.pdf) · [Assignment 1 report](reports/assignment1_foundation.pdf) · [Complete notebook](notebooks/WeatherVision_Complete_A1_A2.ipynb)

![SigLIP 2 application](outputs/siglip2/app.png)

## Dataset

Gbeminiyi Ajayi's [Multi-class Weather Dataset](https://doi.org/10.17632/4drtyfjtfy.1) (2018, CC BY 4.0) is accessed through the [PerceptiLabs mirror](https://github.com/PerceptiLabs/Weather-Analysis), pinned to `ad78ac8fcf1b1aae229ca6e5310aa75b06783787`. Of 1,123 valid photographs, 26 exact copies were removed. The remaining 1,097 images form 1,011 detected groups. File/pixel hashes and conservative perceptual similarity rules identify duplicates before splitting; detected groups stay together. Near-duplicate detection remains heuristic.

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| Cloudy | 204 | 43 | 44 |
| Rain | 143 | 31 | 31 |
| Shine | 175 | 37 | 37 |
| Sunrise | 246 | 53 | 53 |
| Total | 768 | 164 | 165 |

The original seed-42 manifest, content audit and test membership are unchanged. `python weather_cv.py prepare` downloads and verifies the pinned data; `--local-zip /path/to/archive.zip` accepts a local source archive.

## Completed results

Every row below uses the same 165 held-out images. Validation macro-F1 selects checkpoints, retaining the earliest epoch on ties. Zero-shot prompts and all six new supervised checkpoints were selected before a test-selection lock was written. The primary run was predeclared as augmented seed 42.

| Method | Accuracy | Macro precision | Macro recall | Macro-F1 | New MPS mean forward, ms |
|---|---:|---:|---:|---:|---:|
| Simple CNN / A1 | 95.15% | 0.9489 | 0.9473 | 0.9477 | 1.08 |
| MobileNetV2 / A1 | 96.36% | 0.9624 | 0.9625 | 0.9620 | 5.59 |
| ConvNeXt + augmentation / historical | 98.18% | 0.9866 | 0.9795 | 0.9827 | 6.23 |
| ConvNeXt without augmentation / historical | 97.58% | 0.9798 | 0.9727 | 0.9759 | 6.10 |
| SigLIP 2 zero-shot | 93.33% | 0.9330 | 0.9421 | 0.9366 | 21.97 |
| SigLIP 2 linear probe + augmentation / seed 42 | **98.18%** | 0.9866 | 0.9819 | **0.9839** | 19.24 |
| SigLIP 2 linear probe without augmentation / seed 42 | 98.18% | 0.9866 | 0.9819 | 0.9839 | 19.18 |

The main probe correctly predicts 162 images, equal to historical augmented ConvNeXt. The slightly higher macro-F1 reflects different class-wise decisions, not more correct images. Zero-shot correctly predicts 154 images. No universal superiority or statistical significance is claimed.

New latency measurements use batch one, five warm-ups and thirty synchronized MPS network forwards on the same photograph with each model's correct processor. They exclude decoding, preprocessing, transfer, loading and score conversion; zero-shot uses cached text vectors. Raw measurements are in [outputs/siglip2/benchmarks](outputs/siglip2/benchmarks). Historical timing files remain unchanged.

### Three-seed augmentation ablation

| Seed | With augmentation: accuracy / F1 | Without augmentation: accuracy / F1 | Selected epoch: with / without |
|---|---:|---:|---:|
| 42 | 98.18% / 0.9839 | 98.18% / 0.9839 | 7 / 5 |
| 43 | 98.18% / 0.9839 | 98.18% / 0.9839 | 6 / 11 |
| 44 | 98.18% / 0.9839 | 97.58% / 0.9769 | 7 / 4 |
| Mean ± sample SD | 98.18% ± 0.00 pp / 0.9839 ± 0.0000 | 97.98% ± 0.35 pp / 0.9816 ± 0.0040 | |

Architecture, revision, split, optimizer, initialization seed, epoch budget and checkpoint selection are identical within each augmentation pair. Seed 42 is neutral. Three seeds and a 165-image test set support a descriptive comparison; they do not establish a statistically significant augmentation benefit. Per-class results, histories, predictions and timing are retained for every run.

![Three-seed ablation](outputs/siglip2/ablation.png)

## Model and protocol

The model and processor ID is [`google/siglip2-base-patch16-224`](https://huggingface.co/google/siglip2-base-patch16-224), revision **`75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`**, Apache-2.0. Transformers **4.57.6**, huggingface-hub **0.36.2** and safetensors **0.8.0** are pinned. This official fixed-resolution SigLIP 2 checkpoint uses `SiglipVisionModel` and `model_type=siglip`; the API name does not substitute older SigLIP weights.

The official processor performs RGB bilinear full-image resize to 224×224, 1/255 rescaling and mean/std `[0.5,0.5,0.5]`, without center cropping. A real forward pass verifies 196 tokens from 16×16 patches and a 768-dimensional pretrained attention-pooled vector. Historical CNN processors retain their original center crops and ImageNet statistics.

- **Zero-shot:** normalize pretrained image/text vectors, apply the pretrained logit scale and bias, then independent Sigmoid matching scores. Three prespecified English prompt sets were compared on validation data only; the selected labels are “cloudy weather”, “rainy weather”, “sunny weather” and “sunrise”. Scores are uncalibrated and **do not sum to one**. Text vectors are cached; the large text encoder is not loaded during normal app inference.
- **Supervised:** frozen 92,884,224-parameter vision encoder plus `Linear(768,4)` with 3,076 trainable parameters. Cross-entropy and AdamW use learning rate 0.001, weight decay 0.0001, batch size 32, at most 25 epochs and patience five, seeds 42/43/44. This is **linear probing**, not full fine-tuning.
- **Augmentation:** random resized crop (scale 0.8–1.0, aspect ratio 0.9–1.1, bilinear) and horizontal flip (p=0.5) before the official processor. Fresh views are extracted for each seed and epoch. Deterministic train/validation representations may be cached. The head is optimized on CPU over MPS-extracted features; CPU and CUDA paths are also supported.

The main run took 155.8 s including feature extraction and head optimization. Deterministic-cache reuse makes non-augmented times much shorter, especially for later seeds; these are actual recorded wall times, not a controlled training-efficiency comparison. [The method note](docs/SIGLIP2_METHOD.md) and [configuration](configs/siglip2.json) give the protocol.

## Installation and offline operation

```bash
git clone https://github.com/commedeschamps/WeatherVision.git
cd WeatherVision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TORCH_HOME="$PWD/.cache/torch"
python scripts/download_checkpoints.py --all
python scripts/download_checkpoints.py --siglip2
python weather_cv.py prepare
```

The first downloader restores unchanged historical release checkpoints using `configs/checkpoints.json`. `--siglip2` downloads the pinned official Hugging Face assets, verifies the original full-weight SHA-256, builds a vision-only inference copy and caches genuine prompt embeddings. The initial download is approximately 1.5 GB; the app's vision-only weights are approximately 354 MiB. These large files are excluded from Git. Seven small, trained head/prompt checkpoints are included and verified against [configs/siglip2_artifacts.json](configs/siglip2_artifacts.json).

Use `python scripts/download_checkpoints.py --siglip2 --offline` to rebuild/verify from an existing complete cache. Inference works offline after installation. A missing head, processor or encoder produces a clear setup error; another architecture is never relabeled as SigLIP 2. Runtime dependencies are in `requirements.txt`; development/notebook/PDF tools are in `requirements-dev.txt`. `requirements-lock-local.txt` preserves the original pre-migration environment. The migration was tested with Python 3.14.5, PyTorch 2.10.0 and TorchVision 0.25.0 on Apple Silicon.

## Application

```bash
python -m streamlit run app.py
```

The default is the trained **SigLIP 2 Linear Probe**. Choose a dataset example, upload JPG/PNG/WebP, or capture a camera image. Select zero-shot or any historical model from the same control. CPU is the default; Automatic selects MPS/CUDA when available. Model and processor resources are cached, with shared vision weights for both SigLIP 2 approaches.

Four-class supervised Softmax and independent zero-shot matching scores have distinct labels. **Compare on this image** runs five models with the correct individual preprocessing. **Timing and environment** provides device details and an optional warmed benchmark. Single-prediction forward time includes score conversion; the benchmark excludes it. CSV, prediction JSON, overlay PNG and signed-importance JSON downloads are available. The Project tab provides both report downloads. Images stay in server memory.

**Show patch occlusion** replaces each of 49 model-input regions (32×32 pixels, aligned to 2×2 native patches) with normalized zero, the RGB mean. The original predicted class remains fixed; original minus occluded logit is saved, including signed values. Positive drops form the displayed overlay, mapped through the full-image resize. Opacity changes display only. Determinism, geometry and an independent single-region recomputation are verified. This measures sensitivity to artificial image replacement, not attention, segmentation or causal proof.

Historical **ConvNeXt Grad-CAM** remains available. Its original 22-image study is documented in [EXPLAINABILITY_VALIDATION.md](docs/EXPLAINABILITY_VALIDATION.md). It is not presented as a Transformer explanation. `deploy_app.py` is an optional startup-download entrypoint; no public deployment is claimed.

## Reproduction and verification

```bash
# New experiments: isolated outputs preserve the completed submission.
python scripts/run_siglip2.py --device auto --outputs outputs/reproduction_siglip2

# Verify/build completed submission artifacts.
python scripts/build_siglip2_results.py --device auto
python scripts/build_reports.py
python scripts/build_siglip2_notebook.py

# Tests and browser checks (start Streamlit separately).
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/verify_project.py --full-inference
python -m pip install -r requirements-ui.txt
python scripts/check_app.py --url http://127.0.0.1:8501
```

`run_siglip2.py` reuses an already complete directory only when its protocol and manifest match, and refuses to overwrite partial test evaluation. Source definitions and small checkpoints carry model revision, processor, class mapping, seed, validation selection, manifest and integrity hashes. `outputs/siglip2/selection_lock.json` binds all seven selected checkpoints before test access.

The original CLI remains available for isolated historical reproduction:

```bash
python weather_cv.py train --run cnn_baseline --outputs outputs/reproduction_a1
python weather_cv.py train --run mobilenet_improved --outputs outputs/reproduction_a1
python weather_cv.py evaluate --outputs outputs/reproduction_a1 --runs cnn_baseline mobilenet_improved
```

`src/siglip2.py` implements loading, preprocessing, inference and occlusion. `src/siglip_experiments.py` implements feature extraction, training, validation selection and evaluation. The scripts install assets, run experiments and build evidence-based figures, notebook and report. Original experiment configuration/checkpoint hashes are preserved. Generated caches and test logs are ignored.

## Submission artifacts

The [complete notebook](notebooks/WeatherVision_Complete_A1_A2.ipynb) contains **87 cells, including 61 executed code cells**, with full visible data/model/training/evaluation/explanation implementations and the complete app/CLI inspection appendices. No executable cell imports project algorithms from `src`. Demonstration mode loads completed checkpoints and runs real CPU predictions without retraining; optional full reproduction writes separately. All 61 demonstration cells passed in a fresh local kernel. Hosted Google Colab execution is not claimed. Rebuilding the notebook resets execution outputs; execute it before exporting with nbconvert.

The [eight-page Assignment 2 PDF](reports/assignment2_report.pdf) includes the actual results, curves, confusion matrix, ablation, occlusion examples and references. [The ConvNeXt report](reports/history/assignment2_convnext.pdf) and [previous complete notebook](notebooks/history/WeatherVision_ConvNeXt_A1_A2.ipynb) are archived. The original [Colab notebook](notebooks/WeatherVision_Colab.ipynb), A1 PDF, checkpoints and all original metrics remain unchanged.

`python scripts/package_project.py` creates a source archive; `--results` also includes historical full checkpoints; the seven small SigLIP 2 heads are included in the default source archive. Raw photographs, caches and secrets are excluded. Pretrained SigLIP 2 assets are installed separately.

## Limitations and references

Small, overlapping classes, heuristic duplicate grouping, a single fixed split, three head-training seeds and pretrained-data overlap limit generalization claims. No out-of-distribution detector or probability calibration was trained. The fixed-resolution processor distorts aspect ratio. Coarse occlusion introduces artificial inputs and region interactions; positive-only display omits suppressive effects. No generated photographs enter training.

[SigLIP 2 paper](https://arxiv.org/abs/2502.14786) · [Official implementation](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md) · [Tested fixed-resolution API](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/siglip) · [MobileNetV2](https://arxiv.org/abs/1801.04381) · [ConvNeXt](https://arxiv.org/abs/2201.03545) · [Grad-CAM](https://arxiv.org/abs/1610.02391). Bibliographic entries are in [docs/references.bib](docs/references.bib).
