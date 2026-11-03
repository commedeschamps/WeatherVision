# WeatherVision defense guide

Rakhman Seidigali · SE-2411 · Astana IT University

## Project in one minute

WeatherVision assigns one of four visual labels to an outdoor photograph: Cloudy, Rain, Shine or Sunrise. Assignment 1 compares a small CNN trained from scratch with MobileNetV2. Assignment 2 adds ConvNeXt-Tiny, a controlled augmentation ablation and Grad-CAM explanations. MobileNetV2 and ConvNeXt use frozen ImageNet features; only their new four-class heads are trained.

The fixed manifest contains 1,097 photographs: 768 training, 164 validation and 165 test. Duplicate detection runs before splitting, and detected groups remain in one partition. Validation macro-F1 selects the checkpoint; test results do not select it.

| Experiment | Correct / 165 | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| Simple CNN | 157 | 95.15% | 0.9477 |
| MobileNetV2 | 159 | 96.36% | 0.9620 |
| ConvNeXt + augmentation | 162 | 98.18% | 0.9827 |
| ConvNeXt without augmentation | 161 | 97.58% | 0.9759 |

## Demonstration

Launch `python -m streamlit run app.py` after installing dependencies and checkpoints. Use the [timed demo script](DEMO_SCRIPT.md) for the existing 4:08 recording. During a live defense:

1. Choose a photograph, classify it with ConvNeXt and explain the four softmax scores.
2. Show device information and request the optional average-latency measurement.
3. Compare predictions from the three model families on the same image.
4. Show Grad-CAM, adjust opacity and inspect the raw map. Explain a correct prediction and an error.
5. Open Experiments for learning curves, confusion matrices and the augmentation ablation.
6. Open Project for dataset attribution and both report downloads.

Keep the checkpoints installed locally. Uploads work without the dataset; dataset examples require the pinned source images.

## Questions to prepare

**What does transfer learning mean here?** ImageNet-trained features are reused with a new weather classifier. We train 5,124 head parameters for MobileNetV2 and 3,076 for ConvNeXt; the backbones remain frozen. This is not full-backbone fine-tuning.

**Why ConvNeXt?** The 2022 CNN combines a hierarchical feature representation, large depthwise kernels, LayerNorm and residual blocks. It is the advanced architecture actually evaluated here; no claim is made that it is the latest architecture worldwide.

**Why macro-F1?** It averages the four class F1 scores equally despite unequal class sizes. Accuracy counts all correct predictions; precision measures correctness among predicted labels, and recall measures recovery of true labels. A confusion matrix shows true classes in rows and predicted classes in columns.

**What does the ablation establish?** The ConvNeXt configurations change only training augmentation. Accuracy rises by one correct prediction and macro-F1 by 0.0068 in this single-seed comparison. It does not establish a reliable population improvement across datasets or seeds. Original initial-head tensors were not saved, so configuration controls should not be described as direct historical tensor evidence.

**Why are pretrained extractors kept in evaluation mode?** Frozen weights alone do not freeze batch-normalization statistics or disable stochastic depth. Evaluation mode preserves their feature behavior.

**Why can test accuracy exceed validation accuracy?** They are different small samples with different images. The best CNN validation macro-F1 is 0.8544, versus test macro-F1 0.9477. This gap does not by itself establish leakage or its absence; membership, image hashes and group separation are checked independently.

**How is latency measured?** The reported MPS comparison uses batch one, five warm-ups and 30 synchronized forwards, excluding preprocessing, transfer, loading and softmax. The app's ordinary prediction includes softmax and has a separate timer. Browser identity describes the client; PyTorch runs on the server's CPU or selected accelerator.

**Why might a heatmap highlight a horizon or background?** Grad-CAM attributes spatial features to the selected class score, not to a cloud mask. A Cloudy image misclassified as Sunrise is explained with respect to Sunrise. The final map is coarse, and scene context can affect classification. The earlier stage is a complementary experimental view, not an automatically better explanation.

**How were explanations checked?** A separate 22-image diagnostic removes equal-area regions ranked high or low by Grad-CAM and compares them with random masks. Final-stage high regions beat random controls in 21 mean-fill cases and 18 blur cases. Artificial perturbations and individual counterexamples limit the conclusion; see [the method and saved data](EXPLAINABILITY_VALIDATION.md).

**What are the main limitations?** One seed, 165 test images, ambiguous visual categories, heuristic duplicate grouping and limited coverage of new cameras and locations. Softmax scores are uncalibrated, and the classifier has no rejection class for unrelated images.

**What was executed and recorded?** The notebook was executed locally, not on hosted Colab. The published demo records the actual app and uses disclosed synthetic narration. Training uses dataset photographs, not generated images. The README links the code, reports, video, source dataset and released weights.
