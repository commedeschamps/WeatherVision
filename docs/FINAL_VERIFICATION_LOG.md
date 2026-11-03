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

## Technical improvement verification

Baseline commit: `2f3de91eaafd0ee418535e1e78b7ccd169682dce`; clean working tree. Dedicated branch: `technical/explainability-validation`. Baseline hashes/results: `outputs/technical_audit/baseline.json` (93 protected artifacts).

| Executed command / check | Actual outcome |
|---|---|
| `.venv/bin/python -m pytest -q --junitxml=outputs/technical_audit/baseline_pytest.xml` | 40 passed in 18.38 s before source edits. |
| Controlled original helper reproductions | Mixed-mode Grad-CAM changed 197 module flags; boundary test tinted 852 unseen pixels on a 224-square all-positive map. Baseline scratch-CNN training-mode prediction changed nine BN buffers. A corrupted checkpoint raised UnpicklingError outside the app's handled tuple. Reproductions use separate temporary inputs/models. |
| `.venv/bin/python scripts/validate_explanations.py --device auto` | Actual MPS run: 22 selected test images, both stages, two replacements and 1,056 perturbations. Five of 49 cells deleted per mask, ten seeded random masks per image; fixed original target. Weights unchanged. Full scores/settings/maps in outputs/explainability_validation. |
| Six legacy-array and CPU/MPS comparisons | Maximum difference from preserved maps 2.39e-7; CPU/MPS difference 4.53e-6. outputs/technical_audit/gradcam_numerical_parity.json. |
| `MPLCONFIGDIR=.cache/matplotlib TORCH_HOME=.cache/torch .venv/bin/python scripts/audit_training_protocol.py` | Actual CPU reconstruction of 656 validation and 660 test predictions; all saved metrics/labels match. File/pixel/group checks, validation-selected epochs, deterministic head replay, two ablation disagreements and controlled rotation visualization passed. |
| Independent experiment identity check | All four checkpoint/config/manifest/test identity hashes matched evaluation metadata. Cached V1 weights match the vendor filename hash prefix. outputs/technical_audit/experiment_identity_check.json. |
| `.venv/bin/python scripts/profile_inference_policy.py` | Three paired loaded-model measurements per architecture/backend on CPU and MPS. ConvNeXt CPU median single path 265.12 ms versus 8,157.67 ms for replay of the previous automatic-benchmark policy; MPS 7.42 versus 232.19 ms. Raw timings/environment retained; excludes loading/browser time. |
| `.venv/bin/python -m pytest -q --junitxml=outputs/technical_audit/final_pytest.xml` | 63 passed in 23.86 s, including actual MPS parity and Streamlit corrupt-checkpoint recovery. |
| `.venv/bin/python scripts/check_app.py --url http://127.0.0.1:8507 --output-dir outputs/technical_audit/app` | All 14 existing Chrome scenarios passed after the improvements. New screenshots stored separately from protected figures. |
| `.venv/bin/python scripts/check_technical_app.py --url http://127.0.0.1:8507` | Seven technical browser scenarios passed: raw/opacity invariance, earlier stage, benchmark reuse, device/model state and responsive explanation. Actual automatic backend MPS; no page errors. Browser/server/forward timing scopes recorded separately. |
| `.venv/bin/python -m compileall -q src scripts tests`; `git diff --check`; `.venv/bin/python -m pip check` | Syntax checks passed, no whitespace errors, no broken dependencies. |
| Read-only `gh api` checks for v1.0.0/v1.1.0 | All preexisting release asset digests matched recorded published hashes. No release files changed. outputs/technical_audit/release_preservation.json. |

Final occlusion results are baseline-sensitive: final-stage high regions exceed random controls in 21/22 mean-fill cases and 18/22 blur cases. Cloud_267 has positive mean-fill but negative blur logit drops; this counterexample is included. Historical initial-head identity remains unverified because no original initial-state snapshot was saved; deterministic replay is described separately.

No official training was repeated. Test-suite training uses independent temporary fixtures. All 93 protected artifacts, including the 5/8-page reports and original figures, retain their baseline hashes. Additional scientific clarification is provided in the new technical documents; PDF regeneration was unnecessary.
