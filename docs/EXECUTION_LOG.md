# WeatherVision execution record

**Author:** Rakhman Seidigali  
**Group:** SE-2411

## Environment

Python 3.14.5, PyTorch 2.10.0, TorchVision 0.25.0, macOS ARM64. Full training and benchmark measurements used Apple MPS. CPU loading and prediction were verified for every trained model. The exact dependency snapshot is in `requirements-lock-local.txt`.

## Dataset and experiments

```bash
python weather_cv.py prepare
python scripts/smoke.py
python scripts/run_all.py --stages train --resume
python scripts/run_all.py --stages evaluate explain
```

The dataset audit counted 1,123 valid images, removed 26 exact duplicates, and retained 1,097 images. The final split is 768/164/165 training/validation/test samples. All full runs share this manifest.

The separate real-data smoke test completed one epoch per experiment, evaluation, and Grad-CAM. Full training completed 12 epochs for Simple CNN, eight for MobileNetV2, and six for each ConvNeXt configuration. Validation macro-F1 selected epochs 9, 8, 3, and 5 respectively.

Final metrics, prediction CSVs, confusion matrices, learning curves, and raw latency measurements are stored with each run. Six Grad-CAM examples were generated from the saved ConvNeXt test predictions.

## Validation

```bash
python -m pip check
python -m compileall -q app.py weather_cv.py src scripts tests
python -m pytest -q --junitxml=outputs/pytest.xml
python scripts/verify_project.py
```

The current test suite has 37 passing tests and covers mapping, splits, duplicate/corruption handling, leakage rejection, configuration validation, output shapes, checkpoint loading, CPU inference, metric calculations, Grad-CAM geometry, app startup, training resume, and portable dataset restoration.

The saved-output audit independently recalculates metrics from predictions, checks shared test membership and content hashes, loads all full checkpoints on CPU, validates notebook syntax, and checks the report page counts. Results are recorded in `outputs/verification.json` and `outputs/pytest.xml`.

## Application

```bash
python -m streamlit run app.py --server.headless true --server.address 127.0.0.1 --server.port 8501
python scripts/check_app.py
```

An isolated headless Chrome session checks the trained CPU prediction, Grad-CAM, comparison tab, 1440-wide desktop and 390-wide mobile layouts, and invalid-upload recovery. It also checks JSON/CSV/PNG downloads, preservation of predictions when toggling Grad-CAM, and classification of one photograph with all three models. The captures and result JSON are in `outputs/app/`.

## Reports and packaging

```bash
python scripts/build_notebook.py
python scripts/build_reports.py
python scripts/package_project.py
python scripts/package_project.py --results
```

The notebook has 32 cells, a valid schema, and executable Python code cells. It uses the same pipeline that produced the local experiment results.

The Assignment 1 PDF has five pages; Assignment 2 has eight. Both are generated from stored results and contain the project authors and group. PDF rendering checks cover text, tables, figures, and page flow.

The source/results archives preserve the project layout. Source extraction restores omitted raw images using the pinned dataset while keeping the manifest unchanged. The results archive additionally contains the trained best and resumable last checkpoints.
