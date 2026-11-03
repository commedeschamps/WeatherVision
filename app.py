"""WeatherVision local application: trained models, predictions, and explanations."""
from __future__ import annotations

import html
import json
import threading
from pathlib import Path

import pandas as pd
import streamlit as st
import torch
from PIL import Image, UnidentifiedImageError

from src.app_utils import (MODEL_SPECS, RUN_LABELS, decode_upload, example_catalog,
                           export_prediction, result_key, sync_image_state)
from src.data import transform
from src.engine import load_checkpoint, predict_image
from src.evaluation import benchmark
from src.explainability import gradcam_convnext, overlay_image
from src.utils import LABELS, ROOT, select_device

st.set_page_config(page_title="WeatherVision", page_icon=":material/partly_cloudy_day:", layout="wide")
torch.set_num_threads(4)
st.markdown(f"<style>{(ROOT / '.streamlit/style.css').read_text() if (ROOT / '.streamlit/style.css').exists() else ''}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False, max_entries=6)
def cached_model(path: str, modified: int, device_name: str):
    model, checkpoint = load_checkpoint(path, device_name)
    return model, checkpoint, threading.RLock()


@st.cache_data(show_spinner=False)
def cached_catalog(manifest_path: str, modified: int, root: str):
    return example_catalog(Path(manifest_path), Path(root))


@st.cache_data(show_spinner=False)
def cached_file(path: str, modified: int):
    return Path(path).read_bytes()


def read_file(path: Path):
    return cached_file(str(path), path.stat().st_mtime_ns)


def clear_image():
    st.session_state['upload_generation'] = st.session_state.get('upload_generation', 0) + 1
    st.session_state['image_source'] = 'Upload'
    sync_image_state(st.session_state, None)


def classify_one(name, image, device, key):
    path = ROOT / 'outputs' / MODEL_SPECS[name]['run'] / 'best.pt'
    model, checkpoint, lock = cached_model(str(path), path.stat().st_mtime_ns, str(device))
    architecture = MODEL_SPECS[name]['architecture']
    if checkpoint['config']['model'] != architecture:
        raise ValueError('Checkpoint does not match the selected model.')
    with lock:
        prediction = predict_image(model, image, architecture, device)
        sample = transform(architecture=architecture)(image).unsqueeze(0)
        prediction['benchmark'] = benchmark(model, sample, device)
    prediction['browser_user_agent'] = st.context.headers.get('User-Agent', 'Unavailable')
    st.session_state.image_predictions[key] = prediction
    st.session_state.image_heatmaps.pop(key, None)
    return prediction


def prediction_html(prediction):
    name = html.escape(prediction['label'].title())
    rows = []
    for i, (label, value) in enumerate(zip(LABELS, prediction['probabilities'], strict=True)):
        selected = ' is-selected' if i == prediction['class_id'] else ''
        percent = max(0.0, min(100.0, value * 100))
        rows.append(f'<div class="score-row{selected}"><span>{label}</span>'
                    f'<div class="score-track"><div class="score-fill" style="width:{percent:.3f}%"></div></div>'
                    f'<span class="score-number">{value:.1%}</span></div>')
    return (f'<div class="prediction-summary"><span class="prediction-class">{name}</span>'
            f'<span class="prediction-score">{prediction["confidence"]:.1%}</span></div>'
            '<div class="result-caption">Softmax confidence</div>'
            f'<div class="score-list">{"".join(rows)}</div>')


def show_failure(message, error):
    st.error(message)
    with st.expander('Error details'):
        st.code(str(error), language=None)


ready = [name for name, spec in MODEL_SPECS.items() if (ROOT / 'outputs' / spec['run'] / 'best.pt').is_file()]
manifest_path = ROOT / 'data/manifest.csv'
try:
    catalog = cached_catalog(str(manifest_path), manifest_path.stat().st_mtime_ns, str(ROOT)) if manifest_path.exists() else {}
except (OSError, ValueError, pd.errors.ParserError):
    catalog = {}
has_examples = any(catalog.values())

brand, availability = st.columns([3, 1], vertical_alignment='center')
with brand:
    st.title('WeatherVision')
    st.markdown('<div class="brand-description">Weather image classification</div>', unsafe_allow_html=True)
with availability:
    status_text = f'{len(ready)} models ready' if ready else 'Models unavailable'
    status_class = '' if ready else ' unavailable'
    st.markdown(f'<div class="header-status"><span class="availability{status_class}"><span class="availability-dot"></span>{status_text}</span></div>', unsafe_allow_html=True)

classify_tab, experiments_tab, project_tab = st.tabs(['Classify', 'Experiments', 'Project'])

with classify_tab:
    with st.container(key='classify_workspace'):
        workspace, result_column = st.columns([1.18, 1], gap='large')
        with workspace:
            st.subheader('Photograph')
            source = st.segmented_control('Image source', ['Examples', 'Upload', 'Camera'],
                                          default='Examples' if has_examples else 'Upload',
                                          key='image_source', required=True, label_visibility='collapsed', width='stretch')
            content, filename, original = None, 'photograph.jpg', None
            if source == 'Examples':
                categories = [label for label in LABELS if catalog.get(label)]
                if categories:
                    category = st.pills('Example category', categories, default=categories[0],
                                        key='sample_category', required=True, label_visibility='collapsed')
                    index = st.session_state.get(f'example_index_{category}', 0) % len(catalog[category])
                    path = Path(catalog[category][index])
                    try:
                        content = read_file(path)
                        filename = path.name
                    except OSError as error:
                        show_failure('Example is unavailable. Upload a photograph to continue.', error)
                else:
                    st.info('Upload a photograph to begin. Dataset examples are unavailable.')
            else:
                generation = st.session_state.get('upload_generation', 0)
                upload = (st.camera_input('Take a photograph', key=f'camera_{generation}') if source == 'Camera'
                          else st.file_uploader('Upload a photograph', type=['jpg', 'jpeg', 'png', 'webp'],
                                                key=f'upload_{generation}', help='JPG, PNG, or WebP. Up to 12 MB and 25 megapixels.'))
                if upload is not None:
                    content, filename = upload.getvalue(), upload.name
            if content is not None:
                try:
                    original = decode_upload(content)
                    with st.container(key='image_preview'):
                        st.image(original, width='stretch')
                    st.caption(f'{original.width} × {original.height} px  ·  {source}')
                except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
                    st.error('Cannot read this image. Choose a different JPG, PNG, or WebP file.')
                    with st.expander('Image details'):
                        st.write(str(error))
            else:
                st.markdown('''<div class="photo-empty"><svg width="42" height="42" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="3"/><circle cx="8" cy="8" r="1.5"/><path d="m3 17 5-5 4 4 3-3 6 6"/></svg><strong>Add a photograph</strong><span>Upload an image, choose an example,<br>or take a photo with your camera.</span></div>''', unsafe_allow_html=True)
            if content is not None:
                with st.container(key='image_actions'):
                    actions = st.columns(2)
                    if source == 'Examples' and categories:
                        if actions[0].button('Another example', icon=':material/refresh:', width='stretch'):
                            st.session_state[f'example_index_{category}'] = index + 1
                            st.rerun()
                    actions[1].button('Clear image', icon=':material/close:', on_click=clear_image, width='stretch')
        digest = sync_image_state(st.session_state, content if original is not None else None)
        with result_column:
            st.subheader('Classifier')
            selected = st.selectbox('Model', list(MODEL_SPECS), index=2, key='selected_model')
            spec = MODEL_SPECS[selected]
            st.caption(spec['description'])
            with st.expander('Prediction settings'):
                selected_device = st.selectbox('Inference device', ['CPU', 'Automatic'], key='inference_device')
                st.caption('Automatic selects CUDA or Apple MPS when available.')
                st.caption('Forward time includes softmax; image preparation and model loading are excluded.')
            device = select_device('cpu' if st.session_state.get('inference_device', 'CPU') == 'CPU' else 'auto')
            checkpoint_path = ROOT / 'outputs' / spec['run'] / 'best.pt'
            available = checkpoint_path.is_file()
            modified = checkpoint_path.stat().st_mtime_ns if available else 0
            key = result_key(digest, spec['run'], str(device), modified) if digest else None
            if not available:
                st.warning(f'{selected} has no trained checkpoint. Download the project weights to use this model.')
                with st.expander('Checkpoint setup'):
                    st.code('python scripts/download_checkpoints.py', language='bash')
            if st.button('Classify image', type='primary', icon=':material/arrow_forward:',
                         disabled=not available or original is None, width='stretch'):
                try:
                    with st.spinner('Classifying photograph…'):
                        classify_one(selected, original, device, key)
                except (OSError, ValueError, RuntimeError, KeyError) as error:
                    st.session_state.image_predictions.pop(key, None)
                    show_failure('Could not classify this image. Try another model or CPU inference.', error)
            prediction = st.session_state.image_predictions.get(key)
            if prediction:
                st.markdown(prediction_html(prediction), unsafe_allow_html=True)
                st.markdown(f'<div class="prediction-meta"><span>Forward pass <strong>{prediction["latency_ms"]:.1f} ms</strong></span><span>Device <strong>{str(device).upper()}</strong></span></div>', unsafe_allow_html=True)
                timing = prediction['benchmark']
                st.caption(f'{selected} · Average forward: {timing["mean_ms"]:.2f} ms · 30 measurements · {str(device).upper()}')
                with st.expander('Timing and environment'):
                    st.write(f'Mean {timing["mean_ms"]:.2f} ms · median {timing["median_ms"]:.2f} ms · p95 {timing["p95_ms"]:.2f} ms')
                    st.caption('Batch 1; five warm-up forwards followed by 30 measurements on this image. Accelerator calls are synchronized. Excludes softmax, image preparation, transfer, model loading, and Grad-CAM. The single forward above includes softmax.')
                    env = timing['environment']
                    st.write(f'Inference backend: {env["device"].upper()} · Python {env["python"]} · PyTorch {env["torch"]} · TorchVision {env["torchvision"]} · {env["threads"]} CPU threads')
                    st.caption(env['platform'])
                    st.text('Browser (reported User-Agent): ' + prediction['browser_user_agent'])
                st.download_button('Download result', export_prediction(prediction, selected, str(device), filename, original.size),
                                   'weathervision_prediction.json', 'application/json', icon=':material/download:',
                                   on_click='ignore', width='stretch')
            else:
                st.markdown('<div class="result-empty"><strong>Ready to classify</strong><span>Run the selected model to see the category and four class scores.</span></div>', unsafe_allow_html=True)
            if len(ready) >= 2:
                if st.button('Compare on this image', icon=':material/compare_arrows:',
                             disabled=original is None, width='stretch'):
                    st.session_state.comparison_shown = False
                    try:
                        with st.spinner('Comparing the models…'):
                            for name in ready:
                                run = MODEL_SPECS[name]['run']
                                path = ROOT / 'outputs' / run / 'best.pt'
                                comparison_key = result_key(digest, run, str(device), path.stat().st_mtime_ns)
                                if comparison_key not in st.session_state.image_predictions:
                                    classify_one(name, original, device, comparison_key)
                        st.session_state.comparison_shown = True
                        st.session_state.comparison_device = str(device)
                        st.rerun()
                    except (OSError, ValueError, RuntimeError, KeyError) as error:
                        show_failure('Model comparison could not be completed.', error)
            show_cam = st.checkbox('Show Grad-CAM', value=False, key='show_gradcam') if prediction and spec['architecture'] == 'convnext' else False

    if st.session_state.comparison_shown and st.session_state.get('comparison_device') == str(device) and original is not None:
        st.subheader('Predictions on this photograph')
        comparison_rows = []
        for name in ready:
            run = MODEL_SPECS[name]['run']
            path = ROOT / 'outputs' / run / 'best.pt'
            comparison_key = result_key(digest, run, str(device), path.stat().st_mtime_ns)
            item = st.session_state.image_predictions.get(comparison_key)
            if item:
                comparison_rows.append({'Model': name, 'Prediction': item['label'].title(),
                                        'Confidence': f'{item["confidence"]:.1%}', 'Forward ms': round(item['latency_ms'], 1)})
        comparison_frame = pd.DataFrame(comparison_rows)
        st.dataframe(comparison_frame, hide_index=True, width='stretch')
        st.download_button('Download image comparison', comparison_frame.to_csv(index=False),
                           'weathervision_image_comparison.csv', 'text/csv', icon=':material/download:', on_click='ignore')
    if show_cam:
        try:
            if key not in st.session_state.image_heatmaps:
                with st.spinner('Generating Grad-CAM…'):
                    model, _, lock = cached_model(str(checkpoint_path), modified, str(device))
                    with lock:
                        x = transform(architecture='convnext')(original).unsqueeze(0).to(device)
                        heatmap = gradcam_convnext(model, x, prediction['class_id'])
                        st.session_state.image_heatmaps[key] = overlay_image(original, heatmap)
            st.subheader('Grad-CAM explanation')
            st.caption('Brighter regions show spatial features associated with the predicted class.')
            with st.container(key='explanation_view'):
                left, right = st.columns(2)
                left.image(original, caption='Original photograph', width='stretch')
                right.image(st.session_state.image_heatmaps[key], caption=f'Grad-CAM · {prediction["label"].title()}', width='stretch')
            import io
            buffer = io.BytesIO()
            st.session_state.image_heatmaps[key].save(buffer, format='PNG')
            st.download_button('Download heatmap', buffer.getvalue(), 'weathervision_gradcam.png', 'image/png',
                               icon=':material/download:', on_click='ignore')
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            show_failure('Could not generate the heatmap. Try CPU inference.', error)

with experiments_tab:
    st.subheader('Test-set results')
    comparison_path = ROOT / 'outputs/model_comparison.csv'
    if comparison_path.exists():
        try:
            comparison = pd.read_csv(comparison_path)
            st.caption(f'{int(comparison.test_samples.iloc[0])} test images · {len(comparison)} experiments · checkpoint selected by validation macro-F1')
            display = pd.DataFrame({'Model': comparison.run.map(RUN_LABELS).fillna(comparison.run),
                                    'Accuracy': comparison.accuracy.map(lambda x: f'{x:.2%}'),
                                    'Macro F1': comparison.f1_macro.round(4),
                                    'Precision': comparison.precision_macro.round(4),
                                    'Recall': comparison.recall_macro.round(4),
                                    'Mean ms': comparison.latency_mean_ms.round(3),
                                    'Median ms': comparison.latency_median_ms.round(3)})
            st.dataframe(display, hide_index=True, width='stretch')
            chart = pd.DataFrame({'Model': display.Model, 'Accuracy': comparison.accuracy, 'Macro F1': comparison.f1_macro})
            chart['Model'] = comparison.run.map({'cnn_baseline': 'Simple CNN', 'mobilenet_improved': 'MobileNetV2',
                                                 'convnext_aug': 'ConvNeXt + aug.', 'convnext_noaug': 'ConvNeXt no aug.'}).fillna(chart.Model)
            with st.container(key='experiment_chart'):
                st.bar_chart(chart, x='Model', y=['Accuracy', 'Macro F1'], x_label='Model', y_label='Score',
                             horizontal=True, stack=False, sort=False, color=['#21635D', '#6E967F'], height=280)
            st.download_button('Download comparison CSV', read_file(comparison_path), 'weathervision_comparison.csv', 'text/csv',
                               icon=':material/download:', on_click='ignore')
            run_choice = st.selectbox('Inspect experiment', comparison.run.tolist(), format_func=lambda run: RUN_LABELS.get(run, run))
            curve, matrix = st.columns([1.6, 1], gap='large')
            for column, filename, caption in [(curve, 'training_curves.png', 'Training and validation'), (matrix, 'confusion_matrix.png', 'Test confusion matrix')]:
                path = ROOT / 'outputs' / run_choice / filename
                if path.exists():
                    column.image(str(path), caption=caption, width='stretch')
            with st.expander('Benchmark protocol'):
                st.write('Batch size 1, five warm-up runs, and 30 synchronized forward measurements on Apple MPS. The table reports the arithmetic mean and median. Image preparation, transfer, softmax, and model loading are excluded.')
                st.caption('The Classify tab shows a single forward including softmax; these experimental timings use warmed network forwards.')
            ablation_path = ROOT / 'outputs/ablation_analysis.json'
            if ablation_path.exists():
                st.markdown('**Augmentation ablation**')
                st.write(json.loads(read_file(ablation_path))['interpretation'])
        except (OSError, ValueError, KeyError) as error:
            show_failure('Experiment results are unavailable. Regenerate the evaluation outputs.', error)
    else:
        st.info('Train and evaluate the models to populate this comparison.')
        st.code('python scripts/run_all.py --resume', language='bash')

with project_tab:
    st.subheader('WeatherVision')
    st.write('Four-class image classification with a CNN baseline, transfer learning, and Grad-CAM explanations.')
    st.markdown('**Rakhman Seidigali** · SE-2411  \nComputer Vision, Astana IT University  \nInstructor: Nurym Kaiyrkhan')
    st.markdown('**Dataset**')
    st.markdown('[Gbeminiyi Ajayi (2018), Multi-class Weather Dataset for Image Classification](https://doi.org/10.17632/4drtyfjtfy.1). Mendeley Data, V1, CC BY 4.0. [PerceptiLabs mirror](https://github.com/PerceptiLabs/Weather-Analysis).')
    stats_path = ROOT / 'data/statistics.json'
    if stats_path.exists():
        try:
            stats = json.loads(read_file(stats_path))
            st.write(f'{stats["retained_images"]:,} images after duplicate removal. Seed 42; approximately 70/15/15 train, validation, and test split. Augmentation is applied during training.')
        except (OSError, ValueError, KeyError):
            st.caption('Prepare the dataset to view its statistics.')
    st.markdown('**Models**')
    st.dataframe(pd.DataFrame([{'Model': name, 'Method': spec['description']} for name, spec in MODEL_SPECS.items()]),
                 hide_index=True, width='stretch')
    st.markdown('**Reports**')
    a1, a2 = st.columns(2)
    for column, stem, label in [(a1, 'assignment1_foundation', 'Assignment 1 report'), (a2, 'assignment2_report', 'Assignment 2 report')]:
        path = ROOT / 'reports' / f'{stem}.pdf'
        if path.exists():
            column.download_button(label, read_file(path), f'WeatherVision_{stem}.pdf', 'application/pdf',
                                   icon=':material/download:', on_click='ignore', width='stretch')
    st.caption('Uploaded images are processed in server memory and are not saved by the application.')

st.markdown('<div class="project-footer">Rakhman Seidigali &nbsp; / &nbsp; SE-2411 &nbsp; / &nbsp; Astana IT University</div>', unsafe_allow_html=True)
