# Run WeatherVision

```bash
git clone --branch technical/explainability-validation https://github.com/commedeschamps/WeatherVision.git
cd WeatherVision
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_checkpoints.py
python -m streamlit run app.py
```

On Windows activate `.venv\Scripts\activate` instead. Open the local URL printed by Streamlit. The default backend is CPU. Choose Upload, select a JPG/PNG/WebP, and press Classify image. View class scores and the single-forward/server prediction times. Open Timing and environment and click Measure average latency for the optional warmed benchmark. ConvNeXt supports Show Grad-CAM, an opacity slider, optional raw heatmap, final/earlier-stage radio controls, and PNG downloads. Compare on this image runs all three models.

Uploads and inference do not require the dataset. For test examples, run `python weather_cv.py prepare`, then restart/refresh the application. This restores the pinned dataset and verifies the existing manifest. For the fourth ablation model use `python scripts/download_checkpoints.py --all`. Existing files with the correct checksum are reused; a mismatched existing file is preserved and reported.

## Reproduce and validate

```bash
python -m pip install -r requirements-dev.txt
python scripts/download_checkpoints.py --all
python weather_cv.py prepare
python -m pytest -q
python scripts/verify_project.py
```

The notebook includes code and saved outputs. Its local execution reused completed training checkpoints and recalculated saved test metrics from predictions. To train fresh, use a separate copy, move its outputs directory to outputs_saved, disable USE_RELEASED_CHECKPOINTS in the notebook, and follow the training cells. Alternatively pass --outputs outputs/retrained to the CLI train/evaluate commands. To deliberately reevaluate on different hardware set REEVALUATE=True; the new latency describes that hardware. Full training commands and the seed-42 configurations are in README.

## Hosting

The verified delivery method is local execution. No public application URL is claimed. See DEPLOYMENT.md for the public-hosting preparation and account requirements.
