# Final verification log

Date: 2026-11-03, Asia/Almaty.

## Initial inspection

- `git status --short`, `git remote -v`: project is not a Git repository.
- `gh auth status` outside network sandbox: authenticated as commedeschamps; repo/workflow scopes.
- Connected GitHub profile: Rakhman Seidigali / commedeschamps.
- Separate official assignment files not present; supplied final-task checklist used.
- Initial audit and immutable-artifact hashes saved before corrective project edits.

## Corrective changes

- Kept the seed-42 manifest, all four original best checkpoints, and per-run test metric JSON bytes unchanged; baseline SHA-256 values remain in FINAL_BASELINE.json.
- Aligned current author metadata with the final task: Rakhman Seidigali, SE-2411.
- Added current-image mean/median/p95 benchmarking (5 warm-ups, 30 synchronized forwards), backend/software/browser information and benchmark data in prediction JSON. Added saved mean timings to experiment tables.
- Added a scientifically accurate ConvNeXt stage/block diagram and explanation; enlarged chart labels and paired confusion matrices.
- Added concise ethics, limitations and 0% synthetic training-image disclosure. The recorded demo uses separately disclosed synthetic speech.
- Executed the notebook locally and retained genuine outputs. Completed training runs are reused; saved evaluation results are recalculated rather than overwritten by default. Fresh training instructions preserve released artifacts.
- Added checksum-validated release downloads, hosting entrypoint, local guide, demo script/recording workflow and download integrity tests.
- Browser checks now use accessible keyboard model selection, wait for completed model-specific results, and capture screenshots only after Streamlit reports notRunning. The PDF renderer now accepts citation brackets inside figure captions.

## Executed checks

| Command / action | Actual result |
|---|---|
| `.venv/bin/python -m pytest -q --junitxml=outputs/pytest_initial_final.xml` | Initial baseline: 37 passed, 18.03 s. |
| `.venv/bin/python -m pytest -q --junitxml=outputs/pytest_final.xml` | Final suite: 40 passed; three added release-download integrity/preservation tests. |
| `.venv/bin/python scripts/verify_project.py` | All dataset hashes, detected-duplicate grouping, split identity, class labels, four checkpoint hashes, recalculated metrics, real CPU prediction, notebook schema/outputs and 5/8 report pages passed. |
| `.venv/bin/python scripts/verify_saved_inference.py` | All 165 test images per model (660 total) reconstructed on CPU; every predicted label matches saved artifacts; 157/159/162/161 correct. `outputs/full_cpu_reconstruction.json`. |
| `.venv/bin/python scripts/execute_notebook.py` | Local nbclient/ipykernel: all 15 code cells executed, no errors; final run 42.90 s; pytest output 40 passed. Training cells logged “Skipping completed run” for all four models. No hosted Colab execution claimed. |
| `.venv/bin/python scripts/check_app.py --url http://127.0.0.1:8504` | 14 real Chrome checks passed, no page errors. Desktop 1440×1080, tablet 820×1180, mobile 390×844; `outputs/app/browser_checks.json`. |
| `.venv/bin/python scripts/check_camera.py` | Camera preview → Take Photo → CPU prediction passed with Chrome's simulated media device. Physical camera not accessed/tested. `outputs/app/camera_simulation.json`. |
| `.venv/bin/python -m pip check` | No broken requirements. |
| `gh repo create commedeschamps/WeatherVision --public ...` | Public dedicated repository created; source pushed to main. |
| `gh release create v1.0.0 ...` | Four original weight files uploaded separately from Git history. Release asset server digests match local SHA-256 values. |
| `git clone --depth 1 https://github.com/commedeschamps/WeatherVision.git tmp/fresh_clone` | Fresh public clone succeeded. |
| `.venv/bin/python tmp/fresh_clone/scripts/download_checkpoints.py --all` | All four public release assets downloaded and validated by size and SHA-256. |
| `.venv/bin/python tmp/fresh_clone/weather_cv.py prepare` | Pinned dataset restored; existing 768/164/165 manifest verified and reused. |
| `.venv/bin/python tmp/fresh_clone/scripts/verify_project.py` | Fresh checkout: four real CPU inferences and metrics/data/checkpoint validation passed. |
| `python -m venv tmp/fresh_environment`, then pip install of the cloned requirements.txt | Independent runtime environment installed successfully; pip check passed. Streamlit launched from the clone on local port 8505. |
| Streamlit hosting page inspection | Actual page title: “Sign in · Streamlit”. No authenticated session/deployment API available in connected tools. No public application URL claimed; local guide supplies application access. |

## PDF inspection

Rendered all 13 PDF pages at 1.5× using PyMuPDF and visually reviewed every page. Corrected the architecture image parser issue found on A2 page 3. Charts have readable labels, paired confusion matrices, captions, and data from saved histories/predictions. Page counts are A1=5 and A2=8. Final PDF checksums and rendering checks are recorded in outputs/final_pdf_review.json after the last export.

## Scope

Full model training was not repeated. Hosted Colab, public deployment, a physical camera and the student's oral defense are not represented as tested. The notebook was executed locally, camera integration was exercised with a simulated device, and the application was verified locally. The public dataset is instructor-approved. The demo's screen footage comes from the actual app; macOS Samantha supplies explicitly labeled synthetic English narration.

## Final artifact checks

- Final suite rerun: 40 passed in 17.31 s; pip check found no broken requirements.
- Fresh-runtime app: all 14 Chrome scenarios passed on local port 8505 with no JavaScript errors; outputs/fresh_app/browser_checks.json. The initial cold startup exceeded the original 30-second test wait; the check now allows 90 seconds. Independent dependency versions are recorded in outputs/fresh_clone_verification.json.
- Both final PDF downloads from the app matched the final on-disk PDFs byte for byte; outputs/final_report_downloads.json.
- Final PDF content remains within the page frame; A1 has 5 pages and A2 has 8. All pages were visually reviewed, including the final architecture, table/curve labels, real app screenshot, references and public repository/video links.
- Final demo: 248.2 s, 1440×1080, 25 fps, H.264/AAC. Full decode succeeded. Narration RMS is nonzero, English subtitles match the script, and nine opening/middle/final frames were reviewed. The final Grad-CAM shot shows both complete images. The public MP4 download SHA-256 matches the local file; outputs/demo_verification.json.
- Four public checkpoint release server digests and fresh downloads match configs/checkpoints.json. Weights remain outside Git history. No original checkpoint, manifest or test_metrics.json bytes changed.
- Final package ZIP members and SHA-256 values are checked by scripts/build_final_submission.py. Models/raw images/cache are excluded; the real public video URL is included instead of duplicating its MP4.

## Public release verification

`gh release create v1.1.0 ... --target fb1adc11bd62b6f9f263639d10e1244209b5ad6c`: final release published at https://github.com/commedeschamps/WeatherVision/releases/tag/v1.1.0. All five final asset sizes/server SHA-256 digests match local files. An anonymous download of WeatherVision_final_submission.zip matched its local hash and passed ZIP integrity checks. Anonymous raw downloads of README, both PDFs, the executed notebook and final audit matched local bytes. Evidence: outputs/public_release_verification.json. Stable checkpoint/video URLs in the reports remain available in v1.0.0.
