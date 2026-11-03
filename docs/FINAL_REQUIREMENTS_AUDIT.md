# Final requirements audit

Initial inspection: 2026-11-03 (Asia/Almaty). Final verification column is updated after corrections.

No separate official Assignment 1/2 requirement document was located in the workspace. This audit uses the 15 A1 and 23 A2 requirements supplied in the final task. The public dataset is instructor-approved.

Student: Rakhman Seidigali · SE-2411 · Astana IT University · Computer Vision · Nurym Kaiyrkhan.

| ID | Requirement | Initial status | Evidence / location | Exact initial problem | Corrective action | Final status and verification |
|---|---|---|---|---|---|---|
| A1-01 | Image classification problem definition | NOT VERIFIED | reports/assignment1_foundation.md §1 | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A1-02 | Dataset description and citation | NOT VERIFIED | data/statistics.json; docs/references.bib | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A1-03 | Dataset preprocessing | NOT VERIFIED | src/data.py; data/image_audit.csv | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A1-04 | Resizing and normalization | NOT VERIFIED | src/data.py: transform | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A1-05 | Data augmentation | NOT VERIFIED | configs/experiments.yaml; src/data.py | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A1-06 | Simple baseline model | NOT VERIFIED | src/models.py: SimpleCNN | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A1-07 | Improved transfer-learning model | NOT VERIFIED | src/models.py: build_model | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A1-08 | Training and evaluation | NOT VERIFIED | outputs/cnn_baseline/status.json; outputs/mobilenet_improved/status.json | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A1-09 | Accuracy, precision, recall and F1 | NOT VERIFIED | outputs/*/test_metrics.json; test_predictions.csv | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A1-10 | Confusion matrix | NOT VERIFIED | outputs/*/confusion_matrix.png | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A1-11 | Training curves | NOT VERIFIED | outputs/*/history.csv; training_curves.png | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A1-12 | Python notebook with code and results | PARTIAL | notebooks/WeatherVision_Colab.ipynb | Notebook has code but no execution outputs. | Implement missing material and verify. | NOT VERIFIED — pending final checks |
| A1-13 | Dataset download instructions | NOT VERIFIED | README.md; weather_cv.py prepare | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A1-14 | Report of 3–5 pages | NOT VERIFIED | reports/assignment1_foundation.pdf (5 pages) | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A1-15 | Reproducible experimental setup | NOT VERIFIED | configs/experiments.yaml; data/manifest.csv; README.md | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A2-01 | Continuation using the A1 dataset | NOT VERIFIED | data/manifest.csv; outputs/*/test_metrics.json | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A2-02 | Modern advanced CV method | NOT VERIFIED | src/models.py: convnext_tiny | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A2-03 | Comparison with A1 baseline | NOT VERIFIED | outputs/model_comparison.csv | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A2-04 | Working web application | PARTIAL | app.py; scripts/check_app.py | Previous browser evidence exists; independent rerun pending. | Implement missing material and verify. | NOT VERIFIED — pending final checks |
| A2-05 | Image upload or camera | PARTIAL | app.py: file_uploader/camera_input | Camera hardware not verified; upload rerun pending. | Implement missing material and verify. | NOT VERIFIED — pending final checks |
| A2-06 | Prediction | PARTIAL | src/engine.py: predict_image | Fresh actual-checkpoint and browser verification pending. | Implement missing material and verify. | NOT VERIFIED — pending final checks |
| A2-07 | Visual explanation | PARTIAL | src/explainability.py; outputs/convnext_aug/gradcam_* | Stored examples exist; browser overlay/state rerun pending. | Implement missing material and verify. | NOT VERIFIED — pending final checks |
| A2-08 | Accuracy and F1 evaluation | NOT VERIFIED | outputs/*/test_predictions.csv | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A2-09 | Training curves | NOT VERIFIED | outputs/*/history.csv | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A2-10 | Ablation study | NOT VERIFIED | outputs/ablation.csv; configs/experiments.yaml | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A2-11 | Average inference latency | PARTIAL | outputs/*/test_metrics.json: latency.mean_ms | Saved mean exists, but app and report emphasize median/single forward. | Implement missing material and verify. | NOT VERIFIED — pending final checks |
| A2-12 | Execution environment, device and browser | PARTIAL | src/utils.py: environment; app.py | App lacks environment and browser information. | Implement missing material and verify. | NOT VERIFIED — pending final checks |
| A2-13 | Dataset statistics, balance and augmentation | NOT VERIFIED | data/statistics.json; src/data.py | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A2-14 | Synthetic/generated media disclosure | PARTIAL | src/data.py; configs/experiments.yaml | Explicit 0% synthetic training share not documented. | Implement missing material and verify. | NOT VERIFIED — pending final checks |
| A2-15 | Ethics and limitations | PARTIAL | reports/assignment2_report.md | Concise dedicated ethics/limitations coverage missing. | Implement missing material and verify. | NOT VERIFIED — pending final checks |
| A2-16 | Reproducibility instructions | NOT VERIFIED | README.md; scripts/run_all.py | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A2-17 | Advanced model explanation and architecture diagram | PARTIAL | src/models.py; reports/assignment2_report.md §3 | Architecture diagram and approximately one-page explanation missing. | Implement missing material and verify. | NOT VERIFIED — pending final checks |
| A2-18 | References and applicable Neurohive source | PARTIAL | docs/SOURCE_VERIFICATION.md; docs/references.bib | Neurohive search and official implementation reference pending. | Implement missing material and verify. | NOT VERIFIED — pending final checks |
| A2-19 | Final report of 6–8 pages | NOT VERIFIED | reports/assignment2_report.pdf (8 pages) | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A2-20 | Working app link or local run guide | NOT VERIFIED | README.md: streamlit run app.py | No gap found in source/artifacts; independent verification pending. | Recalculate metrics / run checks / inspect rendered report as applicable. | NOT VERIFIED — pending final checks |
| A2-21 | Public GitHub repository | FAIL | No Git repository or remote in project | Repository not initialized or published. | Implement missing material and verify. | NOT VERIFIED — pending final checks |
| A2-22 | 3–5 minute demonstration video | FAIL | docs/DEMO_SCRIPT.md contains script only | No recorded video. | Implement missing material and verify. | NOT VERIFIED — pending final checks |
| A2-23 | Submission materials and defense preparation | PARTIAL | docs/DEFENSE_GUIDE.md; dist/WeatherVision_source.zip | Final audit, demo script, recording plan and Moodle package absent. | Implement missing material and verify. | NOT VERIFIED — pending final checks |

## Release and operational checks

| Check | Initial finding | Final verification |
|---|---|---|
| Checkpoint preservation | Four best.pt files present; capture baseline SHA-256 below. | Pending |
| Git/authentication | No .git, no remote; gh auth succeeds outside network sandbox as commedeschamps. Connector profile also confirms account. | Pending |
| Repository hygiene | .venv, raw images, caches, checkpoints, secrets and archives ignored. No tracked files yet. | Pending |
| Fresh setup and checkpoint distribution | No release downloader. | Pending |
| Public deployment | No public URL. Local-run alternative exists. | Pending investigation |
| Hosted Colab | No hosted execution evidence. | Pending local validation; hosted execution not claimed |
| PDF visual review | Existing PDFs are 5/8 pages; several chart labels are small. | Pending page-by-page review |
| Author metadata | Earlier project includes two names; final task identifies Rakhman Seidigali. | Align to final task |

## Preserved experiment baseline

Hashes and sizes before corrective changes: `docs/FINAL_BASELINE.json`. Model checkpoints, test metric JSON files and dataset manifest must retain these hashes.

## Changes and final verification

See `docs/FINAL_VERIFICATION_LOG.md` for commands, actual outcomes and scope.
