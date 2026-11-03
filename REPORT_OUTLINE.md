# WeatherVision reports

**Authors:** Rakhman Seidigali, Yermakhan Khamitov  
**Group:** SE-2411

Generate both reports from the stored experiment outputs:

```bash
python scripts/build_reports.py
```

- `reports/assignment1_foundation.md` and `.pdf`: Assignment 1, Simple CNN and MobileNetV2.
- `reports/assignment2_report.md` and `.pdf`: Assignment 2, including ConvNeXt-Tiny, augmentation ablation, Grad-CAM, and the application.

The reports include dataset preparation, model descriptions, experimental settings, measured tables, learning curves, confusion matrices, discussion, references, and reproducibility commands. Team details are loaded from `configs/submission.yaml`.
