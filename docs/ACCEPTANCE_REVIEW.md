# WeatherVision validation review

**Author:** Rakhman Seidigali  
**Group:** SE-2411

| Component | Validation | Evidence |
|---|---|---|
| Dataset preparation | Download, decoding, corruption detection, exact and near-duplicate grouping | `data/source.json`, `statistics.json`, `image_audit.csv`, `duplicate_pairs.csv` |
| Shared split | 768 training / 164 validation / 165 test images; seed 42; detected groups remain in one partition | `data/manifest.csv`, `outputs/verification.json` |
| Simple CNN | Full 12-epoch training; validation-selected epoch 9 | `outputs/cnn_baseline/` |
| MobileNetV2 | Full 8-epoch training; validation-selected epoch 8 | `outputs/mobilenet_improved/` |
| ConvNeXt + augmentation | Full 6-epoch training; validation-selected epoch 3 | `outputs/convnext_aug/` |
| ConvNeXt without augmentation | Full 6-epoch training; validation-selected epoch 5 | `outputs/convnext_noaug/` |
| Ablation control | The two configurations differ only in augmentation | `configs/experiments.yaml`, `outputs/ablation.csv` |
| Metrics | Recalculated from saved predictions and checked against JSON outputs | `outputs/verification.json` |
| CPU inference | All four trained checkpoints load and predict on CPU | `outputs/verification.json` |
| Grad-CAM | Six examples with finite 224x224 heatmaps and crop-aware overlays | `outputs/convnext_aug/gradcam_metadata.json` |
| Application | Prediction, Grad-CAM, comparison, mobile layout, and invalid-upload checks | `outputs/app/browser_checks.json` |
| Resume | Interrupted CPU training reproduces the uninterrupted result | `tests/test_pipeline.py` |
| Automated tests | Test suite covers data, models, metrics, configuration, app behavior, and checkpoint restoration | `outputs/pytest.xml` |
| Notebook | Valid schema; every code cell compiles; shared pipeline validated locally | `notebooks/WeatherVision_Colab.ipynb` |
| Reports | PDF generation, team metadata, page counts, figures, and layout checked | `reports/` |
| Packaging | Source/results archives exclude raw data, caches, virtual environments, and smoke outputs | `dist/`, `outputs/package_verification.json` |

## Measured test results

| Model | Correct / total | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Simple CNN | 157 / 165 | 0.9515 | 0.9477 |
| MobileNetV2 | 159 / 165 | 0.9636 | 0.9620 |
| ConvNeXt + augmentation | 162 / 165 | 0.9818 | 0.9827 |
| ConvNeXt without augmentation | 161 / 165 | 0.9758 | 0.9759 |

## Verification commands

```bash
python -m pytest -q --junitxml=outputs/pytest.xml
python scripts/verify_project.py
python scripts/build_reports.py
```
