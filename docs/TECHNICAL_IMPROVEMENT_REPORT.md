# WeatherVision technical improvement report

Rakhman Seidigali · SE-2411 · Astana IT University · Computer Vision

Baseline: `2f3de91eaafd0ee418535e1e78b7ccd169682dce`. The working tree was clean. The improvement branch is `technical/explainability-validation`. The baseline contains hashes for 93 protected artifacts and the four official results. The project is technically ready for an undergraduate defense with the methodological limits below.

## A. Original issues

### Confirmed defects and reliability gaps

1. Grad-CAM restored only the top-level training flag. A mixed-mode call changed 197 submodule flags, enabling a previously frozen eval-mode extractor. This affects the helper's state contract; the actual app loaded eval-mode checkpoints.
2. Bilinear inverse mapping could tint a narrow boundary outside the geometric crop. The controlled all-positive 224-square test tinted 852 unseen pixels. Exact pixel preservation is now enforced at original-pixel centers.
3. The original prediction helper accepted a training-mode CNN. An isolated call changed nine BatchNorm buffers. Inference now rejects training-mode modules before a forward.
4. A corrupted checkpoint raised UnpicklingError outside the UI's recovery exceptions. Decoding and malformed metadata now become specific ValueErrors; weights-only loading remains enabled.
5. Every prediction performed five warm-ups and thirty extra forward passes. This increased interactive waiting even when the user only wanted a label. Average benchmarking is now requested separately and cached.
6. The upload decoder accepted any Pillow-readable image even though the interface advertised JPG/PNG/WebP. Content formats are now checked, including disguised unsupported files.
7. A newly computed benchmark could reset the timing expander as dynamic content moved. An explicit expander key preserves its state. Two-choice stage/device controls expose their options directly through radio inputs, avoiding editable-dropdown state races during repeated reruns.

### Methodological limitations

The final stage is spatially coarse; independently normalized maps do not provide calibrated or causal importance. Occlusion changes the input distribution. Rotation adds black corners. The CNN validation/test gap is real for this split. Historical initial classifier heads were not saved. One seed, 165 test images and heuristic duplicate grouping limit generalization claims.

### Visualization improvements and optional enhancement

The explanation now includes original/overlay comparison, an optional raw map, an Inferno legend, opacity, target-class and native-resolution labels. Earlier-stage 14×14 attribution is optional and experimental. The final 7×7 stage remains the default. No attribution values were edited to resemble human cloud masks.

## B. Changes implemented

| Files | Purpose |
| --- | --- |
| src/explainability.py | State-safe Grad-CAM, layer validation, numerical checks, earlier stage, exact crop masking and raw rendering. |
| src/occlusion.py | Deterministic, equal-area high/low/random occlusion of a fixed class logit. |
| src/engine.py | Eval-mode inference guard and targeted checkpoint decoding/metadata errors. Training mathematics unchanged. |
| src/app_utils.py | Supported-content validation and optional benchmark/server/environment exports. |
| app.py; .streamlit/style.css | Explicit benchmarking, keyed cache reuse, explanation controls, functional legend and direct binary choices. |
| scripts/validate_explanations.py | Seeded diagnostic set, two layers/two baselines, actual score exports and figures. |
| scripts/audit_training_protocol.py | File/pixel/group checks, validation/test reconstruction, group bootstrap, ablation replay and rotation examples. |
| scripts/profile_inference_policy.py | Measured loaded-model single-path versus previous benchmark policy. |
| scripts/check_app.py; scripts/check_technical_app.py | Existing flows plus raw-map/opacity, stage, backend and benchmark-state browser checks. |
| tests/test_explainability_validation.py; tests/test_technical_app.py | Focused mathematical, geometry, state, MPS, occlusion, export and failure-recovery tests. |
| README; LOCAL_RUN_GUIDE; DEFENSE_GUIDE; FINAL_REQUIREMENTS_AUDIT; FINAL_VERIFICATION_LOG | Current behavior, reproducibility instructions and accurate defense evidence. |
| EXPLAINABILITY_VALIDATION.md; VALIDATION_TEST_ANALYSIS.md | Methods, observed findings, individual counterexamples and limits. |
| outputs/explainability_validation; outputs/technical_audit | Separate new analyses; no official result files overwritten. |

## C. Grad-CAM validation

The code implements channel-wise spatial averaging of the selected **logit** gradient, weighted activation summation, ReLU, bilinear upsampling and within-map min/max display scaling. `features[7]` produces 1×768×7×7 and `features[5]` produces 1×384×14×14. Hooks are removed on success/failure; every submodule flag is restored; parameters and gradients are preserved. A direct independent gradient-formula test matches the helper. Constant/non-finite/invalid-target cases are covered.

All six preserved final-stage maps were reproduced within maximum absolute difference **2.39×10⁻⁷**; maximum CPU/MPS difference was **4.53×10⁻⁶**. The interior hotspots originate in the computed activation/gradient maps. In cloud_101, an upper-left numerical peak coexists with a secondary right-hand response; image contrast can make the latter look more prominent in an overlay. In cloud_267, the selected target is the erroneous Sunrise class. The small boundary-rendering defect does not explain an interior horizon response.

The equal-area test uses 22 held-out examples, including all three saved errors and all six existing explanation examples. Five of 49 cells (10.20% area) are replaced, with ten fixed-seed random controls per image. Targets remain fixed after perturbation. There are 1,056 actual perturbation records. Positive Δlogit means the selected score falls.

| Layer | Replacement | High / low / random mean Δlogit | High > random |
| --- | --- | --- | --- |
| earlier | blur | 0.323 / 0.034 / 0.156 | 16/22 |
| earlier | mean | 0.320 / -0.074 / -0.017 | 22/22 |
| final | blur | 0.346 / 0.085 / 0.156 | 18/22 |
| final | mean | 0.381 / -0.037 / -0.017 | 21/22 |

For cloud_267, mean replacement lowers the Sunrise logit by 0.248 in high-ranked cells; blur instead raises it (Δlogit −0.118). The results are supportive on many selected cases and baseline-dependent on others. They establish neither causal understanding nor pixel-level segmentation. The earlier stage does not consistently improve the check, so the default is unchanged. Full protocol, figures, sample IDs and counterexamples are in [EXPLAINABILITY_VALIDATION.md](EXPLAINABILITY_VALIDATION.md).

## D. Dataset and evaluation integrity

All 1,097 retained file and decoded-pixel hashes were recomputed. Class mapping, 768/164/165 membership, checkpoint/manifest matching, zero exact/pixel cross-split collisions and zero detected-group cross-split collisions passed. Validation/test augmentation is disabled. All four selected epochs match the maximum validation macro-F1, with earlier tie selection.

All **656 validation** and **660 test** predictions were reconstructed on CPU. Official test labels and metrics match exactly. New validation CSVs are stored separately. CNN has 141/164 correct validation images and 157/165 correct test images; Cloudy and Shine contribute most of its F1 gap. Its predictive entropy is higher on validation. A conditional group bootstrap leaves a positive CNN gap range (+0.0261 to +0.1602); it does not establish a cause. The source-scene coverage of perceptual grouping remains limited.

ConvNeXt frozen backbones are byte-equal and match V1 weights. Deterministic construction replay produces equal heads, but no historical initial-head tensors/checksums exist. Exactly two predicted labels differ: shine_1 remains wrong (Cloudy→Sunrise), and shine_123 becomes correct (Cloudy→Shine). Net gain: one correct image. No statistical-significance claim is made.

Controlled rotation at 8° fills 6.07% of the visible input with black. This is documented as a possible future training-recipe comparison, not silently corrected. See [VALIDATION_TEST_ANALYSIS.md](VALIDATION_TEST_ANALYSIS.md).

## E. Model results

| Model | Accuracy | Macro-F1 |
| --- | --- | --- |
| Simple CNN | 95.15% | 0.9477 |
| MobileNetV2 | 96.36% | 0.9620 |
| ConvNeXt + augmentation | 98.18% | 0.9827 |
| ConvNeXt without augmentation | 97.58% | 0.9759 |

All original best/last checkpoints, experiment configurations, manifest, predictions, metric JSONs, histories, A1/A2 reports and existing figures retain their baseline hashes. No official training run was repeated. The unchanged 5/8-page PDFs retain accurate descriptions of the main experiment; the new technical documents provide the additional diagnostic evidence. Published GitHub release assets are preserved.

## F. Verification

| Command | Actual outcome |
|---|---|
| `python -m pytest -q --junitxml=outputs/technical_audit/baseline_pytest.xml` | 40 passed before edits. |
| `python -m pytest -q --junitxml=outputs/technical_audit/final_pytest.xml` | 63 passed, no failures/errors/skips on this Mac, including actual MPS parity. |
| `python scripts/validate_explanations.py --device auto` | 22 images, 44 maps, 1,056 equal-budget perturbations on MPS; weights unchanged. |
| `MPLCONFIGDIR=.cache/matplotlib TORCH_HOME=.cache/torch python scripts/audit_training_protocol.py` | All data checks, 656 validation/660 test reconstructions, ablation replay and preservation passed. |
| `python scripts/profile_inference_policy.py` | Three paired measurements per model/backend, raw measurements retained. |
| `python scripts/check_app.py --url http://127.0.0.1:8507 --output-dir outputs/technical_audit/app` | 14 existing real Chrome scenarios passed; final run result in browser_checks.json. |
| `python scripts/check_technical_app.py --url http://127.0.0.1:8507` | 7 additional scenarios passed; actual automatic backend MPS, no page errors. |
| `python -m pip check`; `git diff --check` | No broken dependencies or whitespace errors. |

| Model | Backend | Single prediction ms | Previous automatic-benchmark policy ms |
| --- | --- | --- | --- |
| cnn_baseline | cpu | 2.12 | 97.95 |
| mobilenet_improved | cpu | 27.70 | 1043.01 |
| convnext_aug | cpu | 265.12 | 8157.67 |
| cnn_baseline | mps | 2.74 | 31.42 |
| mobilenet_improved | mps | 6.98 | 208.24 |
| convnext_aug | mps | 7.42 | 232.19 |

These are medians of three alternating paired **loaded-model** API timings in milliseconds, measured locally with four CPU threads. The previous policy is replayed using current code: one prediction plus five warm-ups and thirty forwards. It is not a historical measurement. Model loading, network and browser rendering are excluded. Local load affects wall time.

The browser measured **1193 ms** from click to rendered ready state, **271.8 ms** inside the server prediction path and **249.6 ms** for its single forward. Requesting the optional warmed benchmark took **9639 ms**, with a measured CPU mean forward of **252.28 ms**. These scopes are labeled separately; browser identity is not an inference device.

All new screenshots and machine-readable records are under outputs/technical_audit. The test suite's tiny training fixtures are isolated in temporary directories and do not modify the four completed experiments.

## G. Remaining limitations

A larger external set, multiple training seeds and source-scene metadata would better assess generalization and duplicate coverage. Historical initial-head identity cannot be recovered from the saved trained heads alone. A revised rotation fill requires matched retraining. Occlusion tests depend on replacement, grid and deletion budget; gradient attribution omits suppressive evidence after ReLU. The classifier has four labels and no trained unknown-class rejection or probability calibration. Hosted deployment and hosted Colab were outside this technical update.

## H. Defense notes

1. **Why background?** Context can influence a whole-image score, and the final grid is coarse.
2. **Does it prove cloud understanding?** It shows class-related feature attribution; it does not prove human-like understanding.
3. **Why the final spatial layer?** It provides the deeper feature representation while retaining a 7×7 grid. An earlier 14×14 view is complementary.
4. **Grad-CAM versus segmentation?** Segmentation predicts pixel labels; Grad-CAM localizes relative contributions to one image-class score.
5. **How test important regions?** Replace equal-area high/low/random cells and measure the original class logit with a fixed target.
6. **Why one-image augmentation gain?** Two labels change, only one becomes correct, and the pretrained representation is already strong. One run is limited evidence.
7. **Why validation/test differ?** The reconstructed validation subset has more errors and greater model uncertainty, especially Cloudy/Shine. One causal explanation is not established.
8. **Why frozen features?** They reuse ImageNet representations while limiting the number of parameters updated on a small dataset.
9. **What about a new image?** It is assigned one of the four classes and may be wrong on unfamiliar scenes.
10. **Largest limits?** Small sample, one seed, ambiguous labels, heuristic grouping, new environments and approximate explanations.
