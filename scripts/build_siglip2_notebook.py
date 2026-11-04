"""Build the complete notebook from visible canonical implementations and real artifacts."""

from pathlib import Path
from textwrap import dedent
import ast, json, re, hashlib
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def extract(path, *names):
    text = (ROOT / path).read_text()
    lines = text.splitlines()
    chunks = []
    for name in names:
        node = next(
            n
            for n in ast.parse(text).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
        )
        start = min([node.lineno] + [d.lineno for d in node.decorator_list])
        removed = {
            i
            for n in ast.walk(node)
            if isinstance(n, ast.ImportFrom)
            and (n.level or (n.module or "").startswith("src"))
            for i in range(n.lineno, n.end_lineno + 1)
        }
        chunks.append(
            "\n".join(
                line
                for i, line in enumerate(lines[start - 1 : node.end_lineno], start)
                if i not in removed
            )
        )
    return "\n\n\n".join(chunks)


def build_notebook():
    historical = nbformat.read(
        ROOT / "notebooks/history/WeatherVision_ConvNeXt_A1_A2.ipynb", 4
    )
    stop = next(i for i, c in enumerate(historical.cells) if c.id == "part-10")
    cells = historical.cells[:stop]
    for cell in cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        if cell.id == "title":
            cell.source = """# WeatherVision - Complete A1 + SigLIP 2 Assignment 2

Rakhman Seidigali · SE-2411 · Computer Vision, Astana IT University  
Instructor: Nurym Kaiyrkhan

Assignment 1 retains Simple CNN and MobileNetV2. The revised Assignment 2 implements Google SigLIP 2 (2025), zero-shot image-text matching, supervised linear probing, a controlled three-seed augmentation ablation and class-conditioned image-region occlusion. ConvNeXt results remain historical comparisons.

All custom model, data, training, evaluation and explanation implementations are visible in this notebook. No executable cell imports project algorithms from `src`. Default demonstration mode reads completed artifacts and runs real CPU predictions; it does not retrain or replace results. Full reproduction uses a separate output directory. The application and CLI are included as complete inspection appendices.

Run with the provided repository artifacts and Python 3.11+ dependencies. Data and pretrained binaries remain external assets. The original ConvNeXt notebook and report are archived in `notebooks/history` and `reports/history`. Local execution is recorded; hosted Google Colab execution is not claimed."""
        if cell.id == "package-installation":
            cell.source += '\nSIGLIP_PACKAGES = ["transformers==4.57.6", "huggingface-hub==0.36.2", "safetensors==0.8.0"]\nif INSTALL_PACKAGES:\n    subprocess.run([sys.executable, "-m", "pip", "install", *SIGLIP_PACKAGES], check=True)'
        if cell.id == "imports-and-paths":
            cell.source = cell.source.replace(
                "RUN_FULL_REPRODUCTION = False",
                "RUN_FULL_REPRODUCTION = False\nREPRODUCE_A1 = False\nINSTALL_SIGLIP_ASSETS = False",
            )
        if cell.id == "artifact-bootstrap":
            tree = ast.parse(cell.source)
            lines = cell.source.splitlines()
            for node in sorted(tree.body, key=lambda n: n.lineno, reverse=True):
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "SOURCE_SHA256"
                    for t in node.targets
                ):
                    values = {
                        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in sorted((ROOT / "src").glob("*.py"))
                    }
                    lines[node.lineno - 1 : node.end_lineno] = (
                        "SOURCE_SHA256 = " + json.dumps(values, indent=4)
                    ).splitlines()
                if isinstance(node, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "ASSET_SHA256"
                    for t in node.targets
                ):
                    values = ast.literal_eval(node.value)
                    values.pop("configs/submission.yaml", None)
                    lines[node.lineno - 1 : node.end_lineno] = (
                        "ASSET_SHA256 = " + json.dumps(values, indent=4)
                    ).splitlines()
            cell.source = "\n".join(lines)
        if cell.id == "part-4":
            cell.source += "\n\nThese are the preserved A1/historical CNN transforms. SigLIP 2 uses its own direct resize and RGB mean/std 0.5; its complete processor and augmentation path appear in Part 10."
        if cell.id == "dataloader-example":
            cell.source = cell.source.replace(
                'configs["convnext_aug"]', 'configs["mobilenet_improved"]'
            )
        if cell.id == "checkpoint-save-and-load":
            cell.source = extract(
                "src/engine.py", "atomic_checkpoint", "load_checkpoint"
            )
        if cell.id == "reproduction-controls":
            cell.source = cell.source.replace(
                "if not RUN_FULL_REPRODUCTION:\n        return",
                "if not RUN_FULL_REPRODUCTION or not REPRODUCE_A1:\n        return",
            )

    def md(id, text):
        cells.append(nbformat.v4.new_markdown_cell(dedent(text).strip(), id=id))

    def code(id, text):
        cells.append(nbformat.v4.new_code_cell(dedent(text).strip(), id=id))

    def part(n, title, text):
        md(f"part-{n}", f'## Part {n} - {title}\n\n<a id="part-{n}"></a>\n\n' + text)

    part(
        10,
        "SigLIP 2 architecture, assets and preprocessing",
        """SigLIP 2 extends vision-language pretraining with representation-learning techniques [1]. The pinned `google/siglip2-base-patch16-224` is the fixed-resolution 2025 checkpoint, Apache-2.0. Its official `model_type=siglip` uses the compatible `SiglipVisionModel` API; original SigLIP weights are not substituted. A real forward pass yields 196 patch tokens and a 768-dimensional pretrained attention-pooled vector. No CLS/register-token average is invented.

The image processor uses full-image bilinear resize to 224×224, 1/255 rescaling and mean/std 0.5. The visual encoder is frozen. A new four-class head trains; the optional text branch is used to precompute frozen prompt vectors. Install pretrained assets once using the visible installer below, or `python scripts/install_siglip2.py` before running the notebook. Default execution requires the installed assets and repository-trained heads; no missing classifier is replaced by another model.""",
    )
    code(
        "siglip-identities",
        """MODEL_ID = "google/siglip2-base-patch16-224"
REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
ASSET_DIR = ROOT / "models/siglip2"
SIGLIP_OUTPUTS = ROOT / "outputs/siglip2"
SIGLIP_CONFIG_PATH = ROOT / "configs/siglip2.json"
if not SIGLIP_CONFIG_PATH.is_file():
    raise FileNotFoundError("Install the provided SigLIP 2 configurations and experiment artifacts first.")
siglip_config = json.loads(SIGLIP_CONFIG_PATH.read_text())""",
    )
    code(
        "siglip-asset-installation",
        extract("scripts/install_siglip2.py", "install_siglip_assets")
        + "\n\nif INSTALL_SIGLIP_ASSETS:\n    install_siglip_assets()",
    )
    code("siglip-encoder-loader", extract("src/siglip2.py", "load_encoder"))
    code(
        "siglip-weather-model",
        extract("src/siglip2.py", "SiglipWeather", "load_siglip_checkpoint"),
    )
    code(
        "siglip-dataset-preprocessing",
        extract("src/siglip2.py", "SiglipDataset", "siglip_input"),
    )
    code(
        "siglip-loaded-architecture",
        """siglip_encoder, siglip_processor, siglip_provenance = load_encoder()
siglip_bundle = (siglip_encoder, siglip_processor, siglip_provenance)
siglip_model, siglip_checkpoint = load_siglip_checkpoint(OFFICIAL_OUTPUTS / "siglip2_aug/best.pt", device, encoder_bundle=siglip_bundle)
assert siglip_checkpoint["feature_dim"] == siglip_encoder.config.hidden_size
assert not any(parameter.requires_grad for parameter in siglip_model.encoder.parameters())
display(pd.DataFrame([{"Model":MODEL_ID,"Feature dimension":siglip_checkpoint["feature_dim"], **parameter_counts(siglip_model)}]).style.hide(axis="index"))
display(IPImage(filename=str(SIGLIP_OUTPUTS / "architecture.png"),width=900,alt="SigLIP 2 Transformer, pretrained pooling and classification branches"))""",
    )
    part(
        11,
        "Zero-shot classification and supervised training",
        """The complete frozen-feature training and zero-shot selection implementations are below. Three prespecified prompt sets are evaluated on validation macro-F1 only; the selected short prompts are cloudy weather, rainy weather, sunny weather and sunrise. Normalized image/text dot products receive the pretrained scale and bias. Independent Sigmoid matching scores do not sum to one and are not calibrated class probabilities.

The linear probe uses cross-entropy and AdamW, learning rate 0.001, weight decay 0.0001, batch size 32, a 25-epoch cap and patience five. Backbone inference runs on MPS when available; the small head is optimized on CPU. Every augmented seed/epoch receives newly generated image views. Deterministic features may be cached. Both treatments use identical head initialization within each seed. All six selections and prompt selection were locked before test access.""",
    )
    code(
        "siglip-feature-cache",
        extract(
            "src/siglip_experiments.py",
            "experiment_name",
            "feature_cache",
            "checkpoint_metadata",
        ),
    )
    code("siglip-probe-training", extract("src/siglip_experiments.py", "train_probe"))
    code(
        "siglip-zero-shot-selection",
        extract("src/siglip_experiments.py", "select_zero_shot"),
    )
    code("siglip-benchmark-definition", extract("src/evaluation.py", "benchmark"))
    code(
        "siglip-evaluation-definition",
        extract(
            "src/siglip_experiments.py", "evaluate_siglip_run", "run_siglip_experiments"
        ),
    )
    code(
        "siglip-reproduction-switch",
        """if RUN_FULL_REPRODUCTION:
    if REPRODUCTION_OUTPUTS.resolve() == OFFICIAL_OUTPUTS.resolve():
        raise ValueError("Reproduction must use a separate output directory")
    run_siglip_experiments(SIGLIP_CONFIG_PATH, device_name="auto", output_root=REPRODUCTION_OUTPUTS)

display(pd.DataFrame([{"Setting":key,"Value":str(siglip_config[key])} for key in ["seeds","epochs","patience","batch_size","feature_batch_size","lr","weight_decay","feature"]]).style.hide(axis="index"))
prompt_validation=pd.read_csv(OFFICIAL_OUTPUTS / "siglip2_zero_shot/prompt_validation.csv")
display(prompt_validation.style.hide(axis="index").format({"validation_accuracy":"{:.2%}","validation_f1_macro":"{:.4f}"}))""",
    )
    code(
        "siglip-training-histories",
        """for run in ["siglip2_aug","siglip2_noaug"]:
    directory=OFFICIAL_OUTPUTS/run
    history=pd.read_csv(directory/"history.csv")
    display(Markdown(f"**{run} / seed 42**"))
    display(history.style.hide(axis="index").format({key:"{:.4f}" for key in history.columns if key!="epoch"}))
    display(IPImage(filename=str(directory/"training_curves.png"),width=860,alt=f"Actual training and validation curves for {run}"))""",
    )
    part(
        12,
        "Three-seed SigLIP 2 augmentation ablation",
        """Seeds 42, 43 and 44 each have augmented and non-augmented runs. All settings except treatment are shared. Checkpoints are selected independently using validation macro-F1; the main application uses the predeclared seed-42 augmented run. Means and sample standard deviations describe these three runs; no significance claim is made. In seed 42 the two treatments have identical test accuracy and macro-F1. All augmented seeds produce the same test decisions, while one non-augmented seed loses one extra image.""",
    )
    code(
        "siglip-ablation-data",
        """per_seed=pd.read_csv(SIGLIP_OUTPUTS/"per_seed.csv")
seed_summary=pd.read_csv(SIGLIP_OUTPUTS/"ablation_summary.csv")
assert len(per_seed)==6 and set(per_seed.seed)=={42,43,44}
for seed in [42,43,44]:
    statuses=[json.loads((OFFICIAL_OUTPUTS/experiment_name(augment,seed)/"status.json").read_text()) for augment in [True,False]]
    assert statuses[0]["initial_head_sha256"]==statuses[1]["initial_head_sha256"]
    assert statuses[0]["protocol"]==statuses[1]["protocol"]
display(per_seed.style.hide(axis="index").format({"accuracy":"{:.2%}","f1_macro":"{:.4f}","training_seconds":"{:.1f}"}))
display(seed_summary.style.hide(axis="index").format({"accuracy_mean":"{:.2%}","accuracy_std":"{:.2%}","f1_mean":"{:.4f}","f1_std":"{:.4f}"}))
display(IPImage(filename=str(SIGLIP_OUTPUTS/"ablation.png"),width=760,alt="Three-seed augmentation comparison with mean, sample SD and individual runs"))""",
    )
    code(
        "siglip-classwise-ablation",
        """class_results=pd.read_csv(SIGLIP_OUTPUTS/"per_class_seeds.csv")
class_summary=class_results.groupby(["augment","class"])[["precision","recall","f1-score"]].agg(["mean","std"])
display(class_summary.style.format("{:.4f}"))""",
    )
    part(
        13,
        "Evaluation and historical comparisons",
        """Every completed model uses the unchanged 165-image test membership. The following code reconstructs accuracy, macro precision, macro recall, macro-F1 and confusion matrices from each saved prediction file, verifies the checkpoint hashes and shows class-wise reports. The complete evaluation/export implementation is already visible in Part 11. Historical A1/ConvNeXt results remain separate from new SigLIP 2 measurements.""",
    )
    code(
        "comparison-figure-implementation",
        extract("src/plots.py", "confusion_figure", "comparison_figure"),
    )
    code(
        "metric-reconstruction",
        """metrics_by_run={};predictions_by_run={}
new_runs=[experiment_name(a,s) for s in siglip_config["seeds"] for a in [True,False]]+["siglip2_zero_shot"]
for run in list(EXPERIMENT_LABELS)+new_runs:
    directory=OFFICIAL_OUTPUTS/run
    metrics=json.loads((directory/"test_metrics.json").read_text())
    predictions=pd.read_csv(directory/"test_predictions.csv")
    reconstructed=calculate_metrics(predictions.class_id.tolist(),predictions.predicted_class_id.tolist())
    for key in ["accuracy","precision_macro","recall_macro","f1_macro","confusion_matrix"]:
        np.testing.assert_allclose(reconstructed[key],metrics[key],rtol=0,atol=1e-12)
    assert predictions.path.tolist()==manifest[manifest.split=="test"].path.tolist()
    assert metrics["checkpoint_sha256"]==sha256_file(directory/"best.pt")
    metrics_by_run[run]=metrics;predictions_by_run[run]=predictions
for run in ["siglip2_zero_shot","siglip2_aug","siglip2_noaug"]:
    display(Markdown(f"**{run}**"))
    report=pd.DataFrame(metrics_by_run[run]["per_class"]).T.loc[list(CLASSES),["precision","recall","f1-score","support"]]
    display(report.style.format({"precision":"{:.4f}","recall":"{:.4f}","f1-score":"{:.4f}","support":"{:.0f}"}))
    display(IPImage(filename=str(OFFICIAL_OUTPUTS/run/"confusion_matrix.png"),width=460,alt=f"Actual confusion matrix for {run}"))""",
    )
    code(
        "optional-readonly-test-inference",
        """if RUN_FULL_TEST_INFERENCE:
    test_data=SiglipDataset(manifest,siglip_processor,"test")
    features=[]
    with torch.inference_mode():
        for images,_ in DataLoader(test_data,batch_size=16,shuffle=False):
            features.append(siglip_model.features(images.to(device)).cpu())
    test_features=torch.cat(features)
    for run in new_runs:
        current,checkpoint=load_siglip_checkpoint(OFFICIAL_OUTPUTS/run/"best.pt",device,encoder_bundle=siglip_bundle)
        with torch.inference_mode():
            if current.mode=="linear_probe":logits=current.head(test_features.to(device)).cpu()
            else:logits=F.normalize(test_features,dim=-1)@checkpoint["text_embeddings"].T*checkpoint["logit_scale"].exp()+checkpoint["logit_bias"]
        assert logits.argmax(1).tolist()==predictions_by_run[run].predicted_class_id.tolist()""",
    )
    part(
        14,
        "Transformer-compatible explanation implementation",
        """Class-conditioned image-region occlusion measures the change in the original predicted class logit when each 32×32 region is replaced with normalized zero (RGB mean 0.5). The 7×7 grid aligns with 2×2 native ViT patches. Signed drops are saved; positive evidence is normalized for display and mapped through the full-image resize geometry. It is not Grad-CAM, attention or causal proof. Historical ConvNeXt Grad-CAM remains visible below and continues to work in the application.""",
    )
    code(
        "siglip-patch-occlusion",
        extract("src/siglip2.py", "patch_occlusion", "siglip_overlay"),
    )
    code(
        "siglip-explanation-figures",
        """explanation_examples=json.loads((SIGLIP_OUTPUTS/"occlusion_examples.json").read_text())["examples"]
for example in explanation_examples:
    display(IPImage(filename=str(SIGLIP_OUTPUTS/example["figure"]),width=900,alt=f"Original, occlusion overlay and importance for {example['true']} predicted {example['predicted']}"))
    display(Markdown(f"*True: {example['true'].title()}; predicted: {example['predicted'].title()}; {'correct' if example['correct'] else 'incorrect'}.*"))""",
    )
    md(
        "historical-gradcam",
        "### Historical CNN Grad-CAM implementation\n\nThese definitions belong to ConvNeXt and are not applied to SigLIP 2.",
    )
    code(
        "historical-gradcam-core",
        'TARGET_STAGES={"final":7,"earlier":5}\n\n'
        + extract(
            "src/explainability.py",
            "gradcam_convnext",
            "map_heatmap_to_original",
            "heatmap_image",
            "overlay_image",
            "generate_examples",
        ),
    )
    part(
        15,
        "Explanation validation and limitations",
        """The numerical check below independently recomputes one masked-region logit difference and repeats the complete deterministic map. It checks implementation correctness, not causal faithfulness. Mean replacement creates an artificial distribution shift; region interactions, coarse resolution and positive-only display limit interpretation. The old 22-image Grad-CAM/occlusion study remains a historical CNN analysis, not a new SigLIP 2 result.""",
    )
    code(
        "live-occlusion-validation",
        """example=explanation_examples[0]
image=open_rgb(ROOT/example["path"])
inputs=siglip_input(siglip_model,image).to(device)
target=CLASS_TO_IDX[example["predicted"]]
heatmap,details=patch_occlusion(siglip_model,inputs,target)
repeated,repeated_details=patch_occlusion(siglip_model,inputs,target)
np.testing.assert_array_equal(heatmap,repeated)
changed=inputs.clone();changed[:,:,:32,:32]=0
with torch.inference_mode():
    actual_drop=float((siglip_model(inputs)[0,target]-siglip_model(changed)[0,target]).cpu())
assert abs(actual_drop-details["signed_logit_drops"][0][0])<1e-4
display(pd.DataFrame([{"Target":example["predicted"],"Region pixels":details["region_pixels"],"Independent logit drop":actual_drop,"Batched logit drop":details["signed_logit_drops"][0][0]}]).style.hide(axis="index").format({"Independent logit drop":"{:.6f}","Batched logit drop":"{:.6f}"}))""",
    )
    code(
        "historical-occlusion-implementation",
        extract("src/occlusion.py", "region_selections", "occlusion_check"),
    )
    part(
        16,
        "Inference, scores and latency",
        """Supervised prediction uses four-class Softmax. Zero-shot uses four independent Sigmoid matching scores; their sum need not be one. The current-device demonstration is distinct from historical MPS timing. The benchmark performs five warm-ups and thirty synchronized, batch-one forwards, excluding preprocessing, transfer, loading and score conversion. Zero-shot prompt vectors are cached; text encoding is not part of image-only latency.""",
    )
    code("siglip-image-inference", extract("src/siglip2.py", "predict_siglip"))
    code("shared-inference-dispatch", extract("src/engine.py", "predict_image"))
    code(
        "live-inference",
        """photograph=open_rgb(ROOT/manifest.loc[manifest.split=="test","path"].iloc[0])
rows=[]
for run in ["siglip2_aug","siglip2_zero_shot"]:
    model,checkpoint=load_siglip_checkpoint(OFFICIAL_OUTPUTS/run/"best.pt",device,encoder_bundle=siglip_bundle)
    prediction=predict_siglip(model,photograph,device)
    assert prediction["class_id"]==int(predictions_by_run[run].iloc[0].predicted_class_id)
    rows.append({"Model":run,"Prediction":prediction["label"],"Top score":prediction["confidence"],"Score type":prediction["score_type"],"Device":str(device),"Single forward ms":prediction["latency_ms"]})
display(pd.DataFrame(rows).style.hide(axis="index").format({"Top score":"{:.4f}","Single forward ms":"{:.2f}"}))
live_benchmark=benchmark(siglip_model,siglip_input(siglip_model,photograph),device)
display(pd.DataFrame([{"Device":str(device),"Mean forward ms":live_benchmark["mean_ms"],"Median forward ms":live_benchmark["median_ms"],"Timed repetitions":30}]).style.hide(axis="index").format({"Mean forward ms":"{:.2f}","Median forward ms":"{:.2f}"}))""",
    )
    part(
        17,
        "Complete model comparison",
        """A1 remains 95.15% / 0.9477 for Simple CNN and 96.36% / 0.9620 for MobileNetV2. Historical augmented ConvNeXt remains 98.18% / 0.9827. The main SigLIP 2 probe achieves 98.18% / 0.9839: equal accuracy and slightly different class balance. Zero-shot obtains 93.33% / 0.9366. The three-seed augmentation difference is descriptive, with a neutral seed-42 pair. Latency below comes from a new common-device benchmark; original timing files were not replaced.""",
    )
    code(
        "final-model-table",
        """comparison=pd.read_csv(SIGLIP_OUTPUTS/"model_comparison.csv")
display(comparison[["run","accuracy","precision_macro","recall_macro","f1_macro","trainable_parameters","training_seconds","latency_mean_ms","device"]].style.hide(axis="index").format({"accuracy":"{:.2%}","precision_macro":"{:.4f}","recall_macro":"{:.4f}","f1_macro":"{:.4f}","trainable_parameters":"{:,}","training_seconds":"{:.1f}","latency_mean_ms":"{:.2f}"}))
display(IPImage(filename=str(SIGLIP_OUTPUTS/"comparison.png"),width=900,alt="A1, historical ConvNeXt, zero-shot and supervised SigLIP 2 results"))""",
    )
    part(
        18,
        "Application integration and complete source",
        """The actual Streamlit application defaults to the trained SigLIP 2 linear probe. It preserves upload, camera, dataset examples, baseline models and historical ConvNeXt. Supervised and matching scores have distinct labels; SigLIP uses occlusion while ConvNeXt retains Grad-CAM. Model/processor caching enables offline reruns after installation. Missing assets show an error rather than a relabeled fallback.

Run `python -m streamlit run app.py`. The complete application source is included below for inspection and is not executed inside Jupyter. Its repository imports refer to the same functions shown in this notebook.""",
    )
    app_utils = (ROOT / "src/app_utils.py").read_text()
    tree = ast.parse(app_utils)
    assignments = [
        ast.get_source_segment(app_utils, node)
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.For))
    ]
    names = [node.name for node in tree.body if isinstance(node, ast.FunctionDef)]
    code(
        "app-helpers",
        "\n".join(assignments) + "\n\n" + extract("src/app_utils.py", *names),
    )
    if (ROOT / "outputs/siglip2/app.png").exists():
        code(
            "actual-app-screenshot",
            'display(IPImage(filename=str(SIGLIP_OUTPUTS/"app.png"),width=960,alt="Working SigLIP 2 Streamlit application"))',
        )
    md(
        "app-source-appendix",
        "### Complete `app.py` source\n\n```python\n"
        + (ROOT / "app.py").read_text()
        + "\n```",
    )
    md(
        "cli-source-appendix",
        "### Training and evaluation entrypoint\n\n```python\n"
        + (ROOT / "scripts/run_siglip2.py").read_text()
        + "\n```\n\n### Historical command-line entrypoint\n\n```python\n"
        + (ROOT / "weather_cv.py").read_text()
        + "\n```",
    )
    md(
        "style-source-appendix",
        "### Application styling\n\n```css\n"
        + (ROOT / ".streamlit/style.css").read_text()
        + "\n```",
    )
    part(
        19,
        "Conclusion, ethics and references",
        """SigLIP 2 provides genuine 2025 visual-semantic features for both zero-shot matching and supervised adaptation. The frozen linear probe matches historical ConvNeXt accuracy on this small test set and improves on A1 baselines; it does not establish universal superiority. Its slightly higher macro-F1 reflects different errors, not more correct images. Across three seeds, augmentation mean accuracy is 98.18% (SD 0.00%) versus 97.98% (SD 0.35%); seed-42 results are neutral.

Limitations include 165 test images, only three training seeds, ambiguous labels, heuristic duplicate grouping, frozen features, aspect-ratio distortion, uncalibrated scores and imperfect occlusion explanations. No significance or causal claim is made. Data are credited under CC BY 4.0 and pretrained assets under Apache-2.0. No generated photographs enter training. The previous ConvNeXt video is historical; revised media use actual SigLIP 2 app footage with disclosed synthetic narration.

All displayed code cells were executed locally in demonstration mode; full reproduction and hosted Colab execution are not claimed. The default does not overwrite official checkpoints, metrics or the dataset split. Original A1 and ConvNeXt artifacts remain available.

### References

1. Tschannen et al. (2025). SigLIP 2. [arXiv:2502.14786](https://arxiv.org/abs/2502.14786).
2. Google. [Official SigLIP 2 checkpoint](https://huggingface.co/google/siglip2-base-patch16-224), pinned revision 75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2. [Official implementation](https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md).
3. Hugging Face. [SigLIP 2 fixed-resolution API](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/siglip).
4. Ajayi (2018). Multi-class Weather Dataset. [DOI:10.17632/4drtyfjtfy.1](https://doi.org/10.17632/4drtyfjtfy.1). [Pinned mirror](https://github.com/PerceptiLabs/Weather-Analysis).
5. Sandler et al. (2018), MobileNetV2, arXiv:1801.04381; Liu et al. (2022), ConvNeXt, arXiv:2201.03545; Selvaraju et al. (2017), Grad-CAM, arXiv:1610.02391.
6. Dosovitskiy et al. (2020/2021). An Image is Worth 16x16 Words. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929).""",
    )
    for cell in cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
        if cell.cell_type == "markdown":
            cell.source = cell.source.replace(
                "The implementation source hashes identify the canonical source revision from which the visible definitions were copied.",
                "Artifacts are verified locally by SHA-256 hash. No files are fetched from a specific historical Git commit.",
            )
            cell.source = cell.source.replace(
                "SOURCE_REVISION pins historical artifact download URLs. SOURCE_SHA256 identifies the local canonical source files embedded when this notebook was built.",
                "Artifacts are verified locally by SHA-256 hash. No files are fetched from a specific historical Git commit.",
            )

    # Add an honest local-artifact requirement instead of requesting unpublished results from an old commit.
    cells[
        0
    ].source += "\n\nNew SigLIP 2 heads, configurations and results must accompany the notebook. Pretrained assets can be installed by the visible code; no custom source modules are required."
    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "title": "WeatherVision - SigLIP 2 Complete A1 + A2",
            "authors": [{"name": "Rakhman Seidigali"}],
        },
    )
    nbformat.validate(notebook)
    target = ROOT / "notebooks/WeatherVision_Complete_A1_A2.ipynb"
    nbformat.write(notebook, target)
    print(
        "Built",
        len(cells),
        "cells;",
        sum(c.cell_type == "code" for c in cells),
        "code cells",
    )


if __name__ == "__main__":
    build_notebook()
