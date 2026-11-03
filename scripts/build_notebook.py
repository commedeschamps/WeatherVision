"""Generate a Colab notebook from the current CLI; executable workflow and locally validated outputs."""
from pathlib import Path
import nbformat as nb

ROOT = Path(__file__).resolve().parents[1]
n = nb.v4.new_notebook()
cells = []
def md(text): cells.append(nb.v4.new_markdown_cell(text))
def code(text): cells.append(nb.v4.new_code_cell(text))

md('''# WeatherVision: Computer Vision Assignments 1 and 2

**Author:** Rakhman Seidigali  
**Group:** SE-2411  
**Course:** Computer Vision, Astana IT University  
**Instructor:** Nurym Kaiyrkhan

This notebook runs the same source code as the local app: Simple CNN, MobileNetV2, ConvNeXt-Tiny, and an augmentation ablation. It classifies photographs as **Cloudy, Rain, Shine, or Sunrise**.

The experiments use the Multi-class Weather Dataset for Image Classification by Gbeminiyi Ajayi. Model metrics and figures are generated from the saved experiment outputs.

Choose **Runtime > Change runtime type > T4 GPU**, when available. A GPU is recommended. CPU fallback works but may take tens of minutes or longer. You can first run the isolated smoke test.''')
md('''## 1. Load the project and install dependencies

Create the source archive locally with `python scripts/package_project.py`. Upload **WeatherVision_source.zip** in the next cell. The upload is extracted with path, symlink, member-size, and total-size checks. Existing project files are reused. You can also clone [the repository](https://github.com/commedeschamps/WeatherVision) and run this notebook from its project root.''')
code('''from pathlib import Path, PurePosixPath
import os, stat, sys, subprocess, zipfile

PROJECT = Path('/content/WeatherVision') if Path('/content').exists() else Path.cwd()
if not (PROJECT / 'weather_cv.py').exists():
    from google.colab import files
    uploaded = files.upload()
    archives = [Path(name) for name in uploaded if name.endswith('.zip')]
    if len(archives) != 1:
        raise ValueError('Upload exactly one WeatherVision_source.zip archive.')
    with zipfile.ZipFile(archives[0]) as archive:
        total = 0
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            total += member.file_size
            if (name.is_absolute() or '..' in name.parts or chr(92) in member.filename
                or stat.S_ISLNK(member.external_attr >> 16)
                or member.file_size > 100_000_000 or total > 2_000_000_000):
                raise ValueError(f'Unsafe archive member: {member.filename}')
        archive.extractall(PROJECT)
os.chdir(PROJECT)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-dev.txt'], check=True)
os.environ['MPLCONFIGDIR'] = str(PROJECT / '.cache/matplotlib')
os.environ['TORCH_HOME'] = str(PROJECT / '.cache/torch')
print('Project:', PROJECT)''')
md('## 2. Check GPU and environment')
code('''import torch, torchvision
from src.utils import environment, select_device
print(environment())
print('Selected device:', select_device())
if not torch.cuda.is_available():
    print('CUDA unavailable. CPU/MPS fallback is supported; expect slower training.')

def cli(*args):
    subprocess.run([sys.executable, 'weather_cv.py', *map(str, args)], check=True)''')
md('''## 3. Download, verify, deduplicate, and split the dataset

The project downloads the 224-square photographs from the pinned PerceptiLabs mirror. The preparation step validates images, removes exact duplicates, groups near duplicates, and assigns whole groups to approximately 70/15/15 splits using seed 42. Existing verified data are reused. All models share this manifest.

Dataset: Ajayi, G. (2018), *Multi-class Weather Dataset for Image Classification*, Mendeley Data, V1, [DOI 10.17632/4drtyfjtfy.1](https://doi.org/10.17632/4drtyfjtfy.1), CC BY 4.0. Mirror: [PerceptiLabs](https://github.com/PerceptiLabs/Weather-Analysis).

If automatic downloading fails, upload a dataset ZIP and set `LOCAL_DATASET_ZIP` to its path. A manual original archive may have different counts; train fresh runs with its manifest.''')
code('''LOCAL_DATASET_ZIP = None  # e.g. '/content/Weather-Analysis-main.zip'
cli('prepare', *(['--local-zip', LOCAL_DATASET_ZIP] if LOCAL_DATASET_ZIP else []))
import json, pandas as pd
manifest = pd.read_csv('data/manifest.csv')
display(pd.crosstab(manifest.label, manifest.split, margins=True))
print(json.dumps(json.loads(Path('data/statistics.json').read_text()), indent=2))''')
md('''## 4. Inspect data and preprocessing

Inspection figures use training photographs. Review `data/duplicate_pairs.csv` before experimentation. Detected components are assigned to a single split, and the candidate pairs are saved for inspection. ImageNet normalization is shared; inference uses official V1 resize sizes (MobileNet 256, ConvNeXt 236) and a 224 center crop. Training-only augmentation adds horizontal flips, rotation up to 8 degrees, and mild color jitter. Validation/test transforms are deterministic.''')
code('''from IPython.display import display, Image as IPImage
for name in ['inspection.png', 'class_distribution.png']:
    display(IPImage(filename=f'data/{name}', width=760))
display(pd.read_csv('data/duplicate_pairs.csv').head(10))
from src.data import validate_manifest, verify_content
verify_content(manifest)
print('Manifest and file-content checks passed.')''')
md('''## 5. Configurations and model architectures

The Simple CNN uses three Conv-BatchNorm-ReLU-Pool blocks (16, 32, 64 channels), global average pooling, dropout, and a four-logit head. MobileNetV2 uses inverted residuals and depthwise convolution. ConvNeXt-Tiny is a modern CNN using large depthwise kernels and hierarchical features. Both transfer models start from ImageNet-1K V1 weights and train only the new linear head. Frozen feature extractors remain in evaluation mode, preserving BatchNorm and stochastic-depth behavior.

AdamW and cross-entropy are used. Validation macro-F1 selects the best checkpoint; patience controls early stopping. Test images do not enter training/model selection. For a fresh training run, work in a separate copy, move `outputs/` to `outputs_saved/`, and set `USE_RELEASED_CHECKPOINTS=False`. This preserves existing results while the training cells create new ones. Edit hyperparameters **before** training; existing runs with different settings are rejected. ConvNeXt ablation settings must differ only in `augment`. Full checkpoints avoid any pretrained download during inference.''')
code('''import yaml
CONFIG = Path('configs/experiments.yaml')
settings = yaml.safe_load(CONFIG.read_text())
# Optional: edit settings['experiments'][...], then CONFIG.write_text(yaml.safe_dump(settings, sort_keys=False)).
# Always keep all ConvNeXt settings equal except augment. Default seed is 42.
print(CONFIG.read_text())
# Download the released checkpoints for a demonstration; omit this for fresh training.
USE_RELEASED_CHECKPOINTS = True
if USE_RELEASED_CHECKPOINTS:
    subprocess.run([sys.executable, 'scripts/download_checkpoints.py', '--all'], check=True)
from src.config import load_config, validate_ablation
configs = load_config(CONFIG)
validate_ablation(configs['convnext_aug'], configs['convnext_noaug'])
RESUME = True

def train_run(name):
    args = ['train', '--config', CONFIG, '--run', name, '--skip-complete']
    if RESUME and Path(f'outputs/{name}/status.json').exists():
        args.append('--resume')
    cli(*args)''')
md('''## 6. Automated tests and optional isolated smoke test

Tests are offline and use temporary synthetic fixtures. The smoke test uses four real images per class per split and one epoch. Its outputs remain in `outputs/smoke`; these are not full experiment results.''')
code('''subprocess.run([sys.executable, '-m', 'pytest', '-q'], check=True)
RUN_SMOKE = False
if RUN_SMOKE:
    subprocess.run([sys.executable, 'scripts/smoke.py'], check=True)''')
md('## 7. A1 baseline: train Simple CNN from scratch')
code("train_run('cnn_baseline')")
md('## 8. A1 improved model: train MobileNetV2 classification head')
code("train_run('mobilenet_improved')")
md('## 9. A2 advanced model: train ConvNeXt-Tiny with augmentation')
code("train_run('convnext_aug')")
md('''## 10. Controlled ablation: ConvNeXt-Tiny without augmentation

Same architecture, initialization protocol, pretrained weights, manifest, shuffle seeds, optimizer, learning rate, batch size, epoch limit, and early stopping; only augmentation changes. Each run independently selects its best epoch by validation macro-F1. Do not tune configurations based on the forthcoming test results.''')
code("train_run('convnext_noaug')")
md('''## 11. Final held-out test evaluation and comparison

Run after all configurations are finalized. Accuracy, macro precision/recall/F1, per-class scores, predictions, and confusion matrices are generated from actual test images. Batch-one latency uses 5 warm-up runs and 30 synchronized repeated forwards of the same resident test tensor. Preprocessing, transfer, I/O, model loading, and softmax are excluded from the benchmark. Hardware and raw timings are saved. Measurements on T4 and local CPU/MPS are not interchangeable.''')
code('''REEVALUATE = False  # True deliberately replaces test outputs and latency for this machine.
if REEVALUATE or any(not Path(f'outputs/{run}/test_metrics.json').exists() for run in configs):
    cli('evaluate', '--runs', *configs)
else:
    from src.engine import calculate_metrics
    import numpy as np
    for run in configs:
        saved = json.loads(Path(f'outputs/{run}/test_metrics.json').read_text())
        pred = pd.read_csv(f'outputs/{run}/test_predictions.csv')
        actual = calculate_metrics(pred.class_id.tolist(), pred.predicted_class_id.tolist())
        for metric in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'confusion_matrix']:
            assert np.allclose(saved[metric], actual[metric])
    print('Recalculated all four saved test results from predictions; existing timing environment retained.')
    from src.evaluation import compare_runs
    compare_runs(list(configs), Path('outputs'))
comparison = pd.read_csv('outputs/model_comparison.csv')
display(comparison)
display(pd.read_csv('outputs/ablation.csv'))
print(Path('outputs/ablation_analysis.json').read_text())''')
md('## 12. Learning curves and confusion matrices')
code('''cli('figures')
subprocess.run([sys.executable, 'scripts/final_figures.py'], check=True)
display(IPImage(filename='outputs/model_comparison.png', width=850))
for run in configs:
    print(run)
    display(IPImage(filename=f'outputs/{run}/training_curves.png', width=900))
    display(IPImage(filename=f'outputs/{run}/confusion_matrix.png', width=420))''')
md('''## 13. Grad-CAM on real, correctly and incorrectly classified test examples

Gradients of a selected class logit weight the final spatial feature maps at `features[-1]`. Frozen models still require an input graph. ReLU and normalization form the heatmap. The overlay inverts the resize/center-crop geometry onto the original photograph and preserves unseen borders. The heatmap gives an approximate spatial attribution for the selected class score. Errors are selected when available, after evaluation.''')
code('''if not Path('outputs/convnext_aug/gradcam_metadata.json').exists():
    cli('explain', '--run', 'convnext_aug', '--count', 6)
for figure in sorted(Path('outputs/convnext_aug').glob('gradcam_*.png')):
    display(IPImage(filename=str(figure), width=800))
print(Path('outputs/convnext_aug/gradcam_metadata.json').read_text())''')
md('''## 14. Generate reports from real stored outputs

Produces an eight-page Assignment 2 report and five-page Assignment 1 report using the saved metrics, figures, and author details from `configs/submission.yaml`. Rebuild reports after changing experiments.''')
code("subprocess.run([sys.executable, 'scripts/build_reports.py'], check=True)")
md('''## 15. Export checkpoints, results, and reports

Keep `best.pt` for deployment and `last.pt` to resume. Full checkpoints include the frozen backbone. The export also contains the manifest and audit, and source references. Data images and cached pretrained weights are not included. Save the archive to Drive for persistence before a Colab runtime resets.''')
code('''import shutil
EXPORT = Path('/content/WeatherVision_results.zip') if Path('/content').exists() else PROJECT / 'dist/WeatherVision_results.zip'
EXPORT.parent.mkdir(parents=True, exist_ok=True)
BUILD_RESULTS_ARCHIVE = Path('/content').exists()
if BUILD_RESULTS_ARCHIVE:
    subprocess.run([sys.executable, 'scripts/package_project.py', '--results'], check=True)
    if EXPORT.resolve() != Path('dist/WeatherVision_results.zip').resolve():
        shutil.copy2('dist/WeatherVision_results.zip', EXPORT)
    print(EXPORT)
else:
    print('Local execution: checkpoints and results retained in outputs/. Archive creation is optional.')
SAVE_TO_DRIVE = False
if SAVE_TO_DRIVE and BUILD_RESULTS_ARCHIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    destination = Path('/content/drive/MyDrive/WeatherVision')
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(EXPORT, destination / EXPORT.name)
DOWNLOAD = True
if DOWNLOAD and BUILD_RESULTS_ARCHIVE:
    from google.colab import files
    files.download(str(EXPORT))''')
md('''## 16. Run the local application and prepare submission

Extract the results ZIP inside the local project, preserving `outputs/` and `data/`. Prepare the pinned dataset locally if real examples are desired. Then:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The application reads trained checkpoints; it never trains at launch. Use `docs/DEFENSE_GUIDE.md` and `docs/DEMO_SCRIPT.md` to prepare a 3-5 minute presentation. Deadline: **November 7, 2026, 23:59 Asia/Almaty**.''')
n['cells'] = cells
n['metadata'] = {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                 'language_info': {'name': 'python'}, 'colab': {'name': 'WeatherVision_Colab.ipynb'}, 'accelerator': 'GPU'}
nb.validate(n)
nb.write(n, ROOT / 'notebooks/WeatherVision_Colab.ipynb')
print('Generated', len(cells), 'cells')
