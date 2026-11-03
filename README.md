# WeatherVision

WeatherVision classifies outdoor photographs as **Cloudy, Rain, Shine, or Sunrise**. It compares a Simple CNN with frozen ImageNet features from MobileNetV2 and ConvNeXt-Tiny, and provides a Streamlit app with model comparison and Grad-CAM explanations.

Rakhman Seidigali · SE-2411 · Computer Vision, Astana IT University

Instructor: Nurym Kaiyrkhan

[Assignment 2 report](reports/assignment2_report.pdf) · [Assignment 1 report](reports/assignment1_foundation.pdf) · [Demo video (4:08)](https://github.com/commedeschamps/WeatherVision/releases/download/v1.0.0/WeatherVision_demo.mp4) · [Model downloads](https://github.com/commedeschamps/WeatherVision/releases)

![WeatherVision application](outputs/app/desktop.png)

## Dataset

The project uses Gbeminiyi Ajayi's [Multi-class Weather Dataset for Image Classification](https://doi.org/10.17632/4drtyfjtfy.1) (2018, CC BY 4.0), through the resized [PerceptiLabs mirror](https://github.com/PerceptiLabs/Weather-Analysis), pinned to commit `ad78ac8fcf1b1aae229ca6e5310aa75b06783787`.

Of 1,123 valid photographs, 26 exact duplicates were removed. The remaining 1,097 images belong to 1,011 detected groups. Exact copies are identified by file and decoded-pixel hashes; near duplicates use pHash, dHash, aspect ratio and thumbnail color distance. Detected groups stay in one partition. The manifest fixes membership and class order with seed 42.

| Category | Train | Validation | Test |
|---|---:|---:|---:|
| Cloudy | 204 | 43 | 44 |
| Rain | 143 | 31 | 31 |
| Shine | 175 | 37 | 37 |
| Sunrise | 246 | 53 | 53 |
| Total | 768 | 164 | 165 |

`data/` contains the manifest, source provenance, content hashes, duplicate audit and statistics. `python weather_cv.py prepare` downloads the pinned images and verifies an existing manifest without rebuilding the split. A manually downloaded archive can be supplied with `--local-zip /path/to/archive.zip`.

Inputs use a 224-square center crop and ImageNet normalization. Shorter-side resize is 256 for Simple CNN and MobileNetV2, and 236 for ConvNeXt. Training augmentation applies horizontal flips, rotations up to 8 degrees and mild color jitter; validation and test preprocessing are deterministic.

## Models and results

Assignment 1 compares Simple CNN and MobileNetV2. Assignment 2 adds ConvNeXt-Tiny (2022) and its augmentation ablation. **Only the new linear heads are trained in MobileNetV2 and ConvNeXt; their pretrained backbones remain frozen.** These are the four completed experiments; no Transformer experiment is included.

| Experiment | Accuracy | Macro precision | Macro recall | Macro-F1 | Mean / median latency, ms |
|---|---:|---:|---:|---:|---:|
| Simple CNN | 95.15% | 0.9489 | 0.9473 | 0.9477 | 0.729 / 0.742 |
| MobileNetV2 | 96.36% | 0.9624 | 0.9625 | 0.9620 | 12.171 / 10.269 |
| ConvNeXt-Tiny + augmentation | 98.18% | 0.9866 | 0.9795 | 0.9827 | 6.716 / 6.494 |
| ConvNeXt-Tiny without augmentation | 97.58% | 0.9798 | 0.9727 | 0.9759 | 6.310 / 6.273 |

All runs use the same 165 test images. Checkpoints are selected by validation macro-F1, retaining the earliest epoch on a tie. The paired ConvNeXt configurations differ only in augmentation. The accuracy difference is one additional correct prediction; the macro-F1 difference is 0.0068 in this single-seed experiment.

Latency was measured on Apple MPS: batch size one, five warm-ups, then 30 synchronized forwards of the same device-resident test tensor, with four CPU threads. It excludes decoding, preprocessing, transfer, loading and softmax. Per-run `test_metrics.json` includes raw timings, hardware information, per-class scores and a confusion matrix. `outputs/model_comparison.csv` also records parameter counts, trainable parameters and training time; histories and individual predictions remain in each run directory.

## Installation

```bash
git clone https://github.com/commedeschamps/WeatherVision.git
cd WeatherVision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
export TORCH_HOME="$PWD/.cache/torch"
python scripts/download_checkpoints.py --all
```

Weights are stored in GitHub Releases, outside Git history. The downloader verifies file sizes and SHA-256 hashes against `configs/checkpoints.json`. All four best checkpoints total about 221 MiB; omit `--all` to download only the three app models (about 115 MiB). Existing files with unexpected hashes are rejected. Inference works offline once checkpoints and dependencies are installed.

Runtime requirements are in `requirements.txt`; tests, notebooks and PDF tools use `requirements-dev.txt`. `requirements-lock-local.txt` records the original local environment: Python 3.14.5, PyTorch 2.10.0 and TorchVision 0.25.0.

## Running the application

```bash
python -m streamlit run app.py
```

Open the URL printed by Streamlit. Choose an example, upload a JPG/PNG/WebP, or use the camera; select a model and click **Classify image**. Dataset examples require `python weather_cv.py prepare`; uploads work without the dataset. The app shows four softmax scores, prediction latency and device/browser details. CPU is the default; automatic selection supports CUDA or Apple MPS when available.

**Compare on this image** runs the three model families on the same photograph. **Show Grad-CAM** explains ConvNeXt predictions. Results, comparisons and heatmaps can be downloaded; the Experiments and Project tabs provide test results, figures and both reports. Models are cached, and missing checkpoints produce a setup message. Uploaded images stay in server memory.

The optional **Measure average latency** action runs five warm-ups and 30 forwards on the selected device. The ordinary prediction timer includes softmax and excludes image preparation and loading. For hosting with a startup download, `deploy_app.py` is the optional entrypoint; no public app deployment is claimed.

## Training and evaluation

`configs/experiments.yaml` fixes seed 42, cross-entropy, AdamW, learning rate 0.001 and weight decay 0.0001. Epoch limits are 12/8/6/6, batch sizes 32/32/16/16, and early-stopping patience is four. Frozen feature extractors stay in evaluation mode during training.

Use a separate output directory to reproduce the experiments while preserving the released results:

```bash
python weather_cv.py prepare
for run in cnn_baseline mobilenet_improved convnext_aug convnext_noaug; do
    python weather_cv.py train --run "$run" --outputs outputs/reproduction
done
python weather_cv.py evaluate --outputs outputs/reproduction --runs cnn_baseline mobilenet_improved convnext_aug convnext_noaug
python weather_cv.py explain --outputs outputs/reproduction --run convnext_aug --count 6
```

Add `--resume` to continue an interrupted run. Training and evaluation verify manifest and image hashes. `scripts/run_all.py --resume` runs the original end-to-end workflow in the default directories; evaluation and figure/report generation replace their generated outputs there.

To verify existing results without retraining:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python scripts/verify_project.py --full-inference
```

This checks dataset integrity, reconstructs metrics from saved predictions and, with `--full-inference`, compares every test prediction against actual CPU checkpoint inference. Optional browser checks use `requirements-ui.txt` and `python scripts/check_app.py` after launching Streamlit. Generated verification files go to ignored `tmp/`.

## Explainability

ConvNeXt Grad-CAM differentiates the selected class logit with respect to spatial activations. The default final stage is 7×7; an earlier 14×14 stage is available as an experimental view. Input gradients enable explanations through the frozen backbone without updating weights. Heatmaps map through the actual center-crop geometry; opacity changes display only.

[Explainability methods and validation](docs/EXPLAINABILITY_VALIDATION.md) describes the preserved 22-image occlusion experiment and its limitations. Raw maps, perturbation scores and settings remain in `outputs/explainability_validation/`. Reproduce it with `python scripts/validate_explanations.py --device auto`.

## Repository structure

```text
app.py, deploy_app.py    Streamlit application and optional hosting entrypoint
weather_cv.py           dataset, training, evaluation and explanation CLI
src/                    model, data, training, evaluation and plotting code
configs/                experiments, checkpoint hashes and submission links
data/                   fixed manifest, provenance and dataset audit
outputs/                trained checkpoints, results, figures and explanation data
reports/                final Assignment 1 and 2 PDFs and Markdown sources
notebooks/              executed WeatherVision notebook
scripts/                reproducibility, browser checks and artifact generation
tests/                  automated tests
docs/                   defense guide, demo script, explanation method and references
```

The [notebook](notebooks/WeatherVision_Colab.ipynb) retains its executed outputs from a local kernel; hosted Colab execution is not claimed. `python scripts/package_project.py` builds a source archive for its upload workflow; add `--results` for checkpoints, or `--output /path/to/archive.zip` to choose a destination. Raw images, caches and secrets are excluded. `python scripts/build_reports.py` rebuilds reports from stored results and `configs/submission.yaml`.

## Limitations and references

The comparison uses one seed and a small test set. Labels overlap visually, duplicate detection is heuristic, and unseen locations or lighting may reduce accuracy. Softmax scores are uncalibrated; no out-of-distribution rejection was trained. Grad-CAM is coarse attribution rather than segmentation or causal proof. No generated photographs were used for training. The demo combines an actual app recording with disclosed synthetic English narration; its local copy is `dist/WeatherVision_demo.mp4`, with [captions](docs/WeatherVision_demo.srt) and a [script](docs/DEMO_SCRIPT.md).

Model references: [MobileNetV2](https://arxiv.org/abs/1801.04381), [ConvNeXt](https://arxiv.org/abs/2201.03545v2), [Grad-CAM](https://arxiv.org/abs/1610.02391v3), and [TorchVision ConvNeXt implementation](https://docs.pytorch.org/vision/0.25/_modules/torchvision/models/convnext.html). Bibliographic entries are in [docs/references.bib](docs/references.bib); presentation notes are in the [defense guide](docs/DEFENSE_GUIDE.md).
