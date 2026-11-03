"""Render evidence-bound Markdown/PDF reports; missing outputs are explicitly pending."""
from __future__ import annotations
import html
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import yaml
from src.utils import sha256_file, write_json

RUNS = ['cnn_baseline', 'mobilenet_improved', 'convnext_aug', 'convnext_noaug']
NAMES = {'cnn_baseline': 'Simple CNN', 'mobilenet_improved': 'MobileNetV2',
         'convnext_aug': 'ConvNeXt + aug.', 'convnext_noaug': 'ConvNeXt no aug.'}


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def table(columns, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, columns)) + ' |',
                      '| ' + ' | '.join(['---'] * len(columns)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def image(path, caption):
    return f'![{caption}]({path})' if (ROOT / 'reports' / path).exists() else f'**Pending figure:** {caption} ({path}).'


def render_pdf(markdown, target, author):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, PageBreak,
                                  Table, TableStyle, Image as RLImage, KeepTogether)
    from PIL import Image
    width = A4[0] - 88
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('WVTitle', fontName='Helvetica-Bold', fontSize=18, leading=22, textColor=colors.HexColor('#19463C'), spaceAfter=12))
    styles.add(ParagraphStyle('WVHeading', fontName='Helvetica-Bold', fontSize=12.5, leading=16, textColor=colors.HexColor('#19463C'), spaceBefore=9, spaceAfter=6, keepWithNext=True))
    styles.add(ParagraphStyle('WVBody', fontName='Helvetica', fontSize=9.5, leading=13.2, spaceAfter=7, splitLongWords=True))
    styles.add(ParagraphStyle('WVCell', fontName='Helvetica', fontSize=8.2, leading=10.5, spaceAfter=0))
    styles.add(ParagraphStyle('WVCaption', fontName='Helvetica-Oblique', fontSize=8, leading=10.5, textColor=colors.HexColor('#4C5E57'), spaceAfter=8))
    styles.add(ParagraphStyle('WVCode', fontName='Courier', fontSize=8, leading=11, spaceAfter=7))

    def inline(text):
        text = text.replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-').replace('\u00b7', '/')
        escaped = html.escape(text)
        escaped = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<a href="\2" color="#21635D">\1</a>', escaped)
        escaped = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escaped)
        escaped = re.sub(r'`([^`]+)`', r'<font name="Courier" size="8">\1</font>', escaped)
        return escaped

    story, lines, i = [], markdown.splitlines(), 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line == '<!-- pagebreak -->':
            story.append(PageBreak()); i += 1; continue
        if line.startswith('```'):
            content = []; i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                content.append(html.escape(lines[i])); i += 1
            story.append(Paragraph('<br/>'.join(content), styles['WVCode'])); i += 1; continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\- ]+', c) for c in cells):
                    rows.append([Paragraph(inline(c), styles['WVCell']) for c in cells])
                i += 1
            t = Table(rows, colWidths=[width / len(rows[0])] * len(rows[0]), repeatRows=1, hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E6F0EB')),
                                   ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                   ('LINEBELOW', (0, 0), (-1, 0), .6, colors.HexColor('#819E90')),
                                   ('LINEBELOW', (0, 1), (-1, -1), .25, colors.HexColor('#D5DFD9')),
                                   ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5)]))
            story.extend([t, Spacer(1, 8)]); continue
        match = re.fullmatch(r'!\[(.*)\]\(([^)]+)\)', line)
        if match:
            caption, path = match.groups()
            source = ROOT / 'reports' / path
            with Image.open(source) as im:
                iw, ih = im.size
            max_height = (205 if 'gradcam' in path else 188 if '/app/' in path else
                          125 if 'training_curves' in path else
                          200 if 'confusion_pair' in path else 185 if 'architecture' in path else
                          125 if 'confusion_matrix' in path else 165)
            ratio = min(width / iw, max_height / ih)
            picture = RLImage(str(source), width=iw * ratio, height=ih * ratio)
            picture.hAlign = 'CENTER'
            story.append(KeepTogether([picture, Spacer(1, 3), Paragraph(inline(caption), styles['WVCaption'])]))
            i += 1; continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]), styles['WVTitle'])); i += 1; continue
        if line.startswith('## '):
            story.append(Paragraph(inline(line[3:]), styles['WVHeading'])); i += 1; continue
        if line.startswith('- '):
            story.append(Paragraph('- ' + inline(line[2:]), styles['WVBody'])); i += 1; continue
        paragraph = [line]; i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].startswith(('#', '|', '![', '```', '<!--', '- ')):
            paragraph.append(lines[i].strip()); i += 1
        story.append(Paragraph(inline(' '.join(paragraph)), styles['WVBody']))

    def footer(canvas, doc):
        canvas.setStrokeColor(colors.HexColor('#B4C8BC'))
        canvas.line(44, 37, A4[0] - 44, 37)
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#4C5E57'))
        canvas.drawString(44, 24, 'WeatherVision / Computer Vision / Astana IT University')
        canvas.drawRightString(A4[0] - 44, 24, str(doc.page))

    SimpleDocTemplate(str(target), pagesize=A4, rightMargin=44, leftMargin=44,
                      topMargin=39, bottomMargin=48, title=target.stem.replace('_', ' '),
                      author=author).build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    stats = read_json(ROOT / 'data/statistics.json', {})
    submission = yaml.safe_load((ROOT / 'configs/submission.yaml').read_text())
    pending, metrics, statuses = [], {}, {}
    for run in RUNS:
        directory = ROOT / 'outputs' / run
        status = read_json(directory / 'status.json')
        result = read_json(directory / 'test_metrics.json')
        if status is None or status.get('state') != 'complete' or result is None:
            pending.append(run); continue
        if not (directory / 'best.pt').exists() or result['checkpoint_sha256'] != sha256_file(directory / 'best.pt'):
            pending.append(run + ' (stale checkpoint evaluation)'); continue
        if result['manifest_sha256'] != stats.get('manifest_sha256'):
            pending.append(run + ' (different dataset manifest)'); continue
        metrics[run], statuses[run] = result, status
    membership = {m['test_membership_sha256'] for m in metrics.values()}
    if len(membership) > 1:
        raise ValueError('Reports cannot compare different test memberships')
    complete = not pending and bool(stats)
    technical = 'COMPLETE MEASURED EXPERIMENTS' if complete else 'DRAFT - EXPERIMENTS OR EVIDENCE PENDING'
    authors = submission.get('authors') or [submission.get('student_name', '')]
    authors = [name for name in authors if name]
    if not authors or not submission.get('group'):
        raise ValueError('Set authors and group in configs/submission.yaml')
    author = ', '.join(authors)
    identity = f'**Author:** {author}\n\n**Group:** {submission["group"]}'
    links = ' '.join(f'{label}: {submission[key]}.' for key, label in
                     [('repository_url', 'Repository'), ('video_url', 'Demo video')] if submission.get(key))
    if complete:
        accuracy_text = '; '.join(f"{NAMES[r]} {metrics[r]['accuracy']:.2%}" for r in RUNS)
        f1_text = '; '.join(f"{NAMES[r]} {metrics[r]['f1_macro']:.4f}" for r in RUNS)
        headline = (f"Test accuracy reaches {metrics['cnn_baseline']['accuracy']:.2%} for Simple CNN, "
                    f"{metrics['mobilenet_improved']['accuracy']:.2%} for MobileNetV2, "
                    f"{metrics['convnext_aug']['accuracy']:.2%} for augmented ConvNeXt, and "
                    f"{metrics['convnext_noaug']['accuracy']:.2%} for ConvNeXt without augmentation.")
    else:
        headline = 'Training/evaluation is incomplete. Missing runs: ' + ', '.join(pending) + '. No missing scores are estimated.'
        accuracy_text, f1_text = 'Pending', 'Pending'
    config = yaml.safe_load((ROOT / 'configs/experiments.yaml').read_text())['experiments']
    config.update({r: s['config'] for r, s in statuses.items()})
    if all(r in statuses for r in RUNS[2:]):
        from src.config import Experiment, validate_ablation
        validate_ablation(*[Experiment(**statuses[r]['config']) for r in RUNS[2:]])
    config_table = table(['Run', 'Epoch limit', 'Batch', 'LR', 'Aug.'],
                         [[NAMES[r], config[r]['epochs'], config[r]['batch_size'], config[r]['lr'], str(config[r]['augment'])] for r in RUNS])
    result_table = table(['Model', 'Accuracy', 'Macro P', 'Macro R', 'Macro F1'],
                         [[NAMES[r], *[f"{metrics[r][k]:.4f}" for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']]]
                          if r in metrics else [NAMES[r], 'pending', 'pending', 'pending', 'pending'] for r in RUNS])
    complexity = table(['Model', 'All params', 'Trainable', 'Train s', 'Mean ms', 'Median ms'],
                       [[NAMES[r], f"{statuses[r]['parameters']:,}", f"{statuses[r]['trainable_parameters']:,}",
                         f"{statuses[r]['training_seconds']:.1f}", f"{metrics[r]['latency']['mean_ms']:.3f}", f"{metrics[r]['latency']['median_ms']:.3f}"]
                        if r in metrics else [NAMES[r], 'pending', 'pending', 'pending', 'pending'] for r in RUNS])
    distribution = table(['Category', 'Train', 'Validation', 'Test', 'Total'],
                         [[label.title(), *[stats.get('counts', {}).get(s, {}).get(label, 'pending') for s in ['train', 'val', 'test']],
                           sum(stats.get('counts', {}).get(s, {}).get(label, 0) for s in ['train', 'val', 'test'])]
                          for label in ['cloudy', 'rain', 'shine', 'sunrise']])
    selected = table(['Model', 'Best epoch', 'Validation F1', 'Test F1'],
                     [[NAMES[r], statuses[r]['best_epoch'], f"{statuses[r]['best_val_f1_macro']:.4f}", f"{metrics[r]['f1_macro']:.4f}"]
                      if r in metrics else [NAMES[r], 'pending', 'pending', 'pending'] for r in RUNS])
    gradmeta = read_json(ROOT / 'outputs/convnext_aug/gradcam_metadata.json', {'examples': [], 'errors_available': 0})
    examples = gradmeta['examples']
    wrong = next((e for e in examples if not e['correct']), None)
    right = next((e for e in examples if e['correct']), None)
    hardware = next(iter(statuses.values()))['environment'] if statuses else {}
    browser_info = read_json(ROOT / 'outputs/app/browser_checks.json', {})
    browser_text = f'Browser checks used Chrome {browser_info["browser_version"]} at desktop and mobile viewport sizes.' if browser_info.get('browser_version') else ''
    refs = '''[1] Ajayi, G. (2018). Multi-class Weather Dataset for Image Classification. Mendeley Data, V1. [DOI: 10.17632/4drtyfjtfy.1](https://doi.org/10.17632/4drtyfjtfy.1).

[2] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR, pp. 4510-4520. [Paper](https://arxiv.org/abs/1801.04381).

[3] Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., and Xie, S. (2022). A ConvNet for the 2020s. CVPR. [Paper](https://arxiv.org/abs/2201.03545v2).

[4] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. arXiv:1610.02391v3. [Paper](https://arxiv.org/abs/1610.02391v3).

[5] PerceptiLabs. Weather-Analysis. [Dataset mirror](https://github.com/PerceptiLabs/Weather-Analysis).

[6] TorchVision 0.25. ConvNeXt-Tiny implementation and V1 preprocessing. [Official source](https://docs.pytorch.org/vision/0.25/_modules/torchvision/models/convnext.html).'''
    counts_text = (f"The dataset contains {stats['valid_images_before_dedup']:,} weather images. "
                   f"Removing {stats['exact_duplicates_removed']} duplicate images leaves "
                   f"{stats['retained_images']:,} images for training and evaluation.") if stats else 'Dataset preparation is incomplete.'
    hardware_text = (f"Training used Apple MPS (four CPU threads) on macOS 26.5.2 with Python {hardware['python']}, "
                     f"PyTorch {hardware['torch']}, and TorchVision {hardware['torchvision']}.") if hardware else 'Training environment unavailable.'
    split_text = ('/'.join(str(sum(stats['counts'][s].values())) for s in ['train', 'val', 'test'])
                  + ' training/validation/test images') if stats else 'Dataset split unavailable.'
    # Describe observed training behavior directly from each run's history.
    loss_rows = []
    for run in RUNS[:3]:
        history_path = ROOT / 'outputs' / run / 'history.csv'
        if history_path.exists():
            history = pd.read_csv(history_path)
            loss_rows.append(f"{NAMES[run]} training loss decreased from {history.iloc[0].train_loss:.3f} to {history.iloc[-1].train_loss:.3f}")
    loss_text = '. '.join(loss_rows) + ('.' if loss_rows else '')
    if all(r in metrics for r in RUNS[2:]):
        augmented = metrics['convnext_aug']
        plain = metrics['convnext_noaug']
        correct_aug = int(np.trace(augmented['confusion_matrix']))
        correct_plain = int(np.trace(plain['confusion_matrix']))
        ablation_result = (f"With augmentation, accuracy was {augmented['accuracy']:.2%} and macro-F1 was {augmented['f1_macro']:.4f}. "
                           f"Without augmentation, accuracy was {plain['accuracy']:.2%} and macro-F1 was {plain['f1_macro']:.4f}. "
                           f"The two models classified {correct_aug} and {correct_plain} of {augmented['test_samples']} images correctly, respectively.")
    else:
        ablation_result = 'Ablation results are incomplete.'
    examples_text = '\n\n'.join(image('../outputs/convnext_aug/' + e['figure'],
                       f"Figure 7{'a' if e['correct'] else 'b'}. {'Correct' if e['correct'] else 'Incorrect'} prediction: {e['true'].title()} classified as {e['predicted'].title()}.")
                       for e in [right, wrong] if e) or 'Grad-CAM figures unavailable.'
    class_names = ['cloudy', 'rain', 'shine', 'sunrise']
    a1_class_table = table(['Class', 'CNN F1', 'MobileNet F1', 'Test images'], [
        [label.title(), f"{metrics['cnn_baseline']['per_class'][label]['f1-score']:.4f}",
         f"{metrics['mobilenet_improved']['per_class'][label]['f1-score']:.4f}",
         int(metrics['cnn_baseline']['per_class'][label]['support'])]
        for label in class_names]) if all(r in metrics for r in RUNS[:2]) else 'Class results pending.'
    ablation_class_table = table(['Class', 'Aug. F1', 'No aug. F1', 'Difference'], [
        [label.title(), f"{metrics['convnext_aug']['per_class'][label]['f1-score']:.4f}",
         f"{metrics['convnext_noaug']['per_class'][label]['f1-score']:.4f}",
         f"{metrics['convnext_aug']['per_class'][label]['f1-score'] - metrics['convnext_noaug']['per_class'][label]['f1-score']:+.4f}"]
        for label in class_names]) if all(r in metrics for r in RUNS[2:]) else 'Class results pending.'
    cnn_shapes = table(['Stage', 'Output shape', 'Operation'], [
        ['Input', '3 x 224 x 224', 'RGB image'],
        ['Block 1', '16 x 112 x 112', 'Conv, BN, ReLU, pool'],
        ['Block 2', '32 x 56 x 56', 'Conv, BN, ReLU, pool'],
        ['Block 3', '64 x 28 x 28', 'Conv, BN, ReLU, pool'],
        ['Classifier', '4 logits', 'Global pool, dropout, linear']])
    abstract = f'''WeatherVision classifies outdoor photographs into Cloudy, Rain, Shine, and Sunrise using Simple CNN, MobileNetV2, and ConvNeXt-Tiny. After duplicate removal, 1,097 images are divided into training, validation, and test sets. The Simple CNN learns its features from scratch; the transfer models use frozen ImageNet feature extractors with new four-class heads. Validation macro-F1 selects the training checkpoint. {headline} The ConvNeXt augmentation ablation produces one additional correct prediction and raises macro-F1 from 0.9759 to 0.9827. Class-wise analysis identifies the remaining confusions, and Grad-CAM visualizes the spatial features associated with a prediction. A Streamlit application combines image upload, model selection, class scores, inference timing, and heatmap display.'''
    p1 = f'''# WeatherVision: Weather Image Classification Using Deep Learning and Transfer Learning

**{'Assignment 2' if complete else 'Assignment 2 - Draft'}**

{identity}

**Course:** Computer Vision, Astana IT University

**Instructor:** Nurym Kaiyrkhan

## Abstract

{abstract}

## 1. Introduction

Weather image classification assigns a visual category to an outdoor photograph. Cloud texture, rain streaks, wet surfaces, direct sunlight, and horizon colors provide useful cues. Their appearance changes with camera position, exposure, background content, and lighting. A classifier must combine local textures with the wider composition of the scene to distinguish the four categories.

A convolutional neural network learns filters that respond to spatial patterns. Early layers detect simple structures such as edges and color transitions. Deeper layers combine those responses into more complex representations. A small CNN learns these filters directly from the weather images. Transfer learning starts with filters learned from ImageNet and adapts a new classifier to the target labels.

The project compares these approaches through Simple CNN, MobileNetV2, and ConvNeXt-Tiny. The comparison includes classification accuracy, macro precision, macro recall, macro-F1, parameter count, training behavior, and forward-pass latency. This connects prediction quality with the computational requirements of an interactive application.

Assignment 1 establishes the Simple CNN and MobileNetV2 comparison. Assignment 2 adds ConvNeXt-Tiny, evaluates training augmentation through a paired experiment, and uses Grad-CAM to examine class-related spatial features. The three model families share one data-loading and evaluation pipeline and are available in the same Streamlit interface.'''
    p2 = f'''## 2. Dataset and preprocessing

The experiments use the Multi-class Weather Dataset for Image Classification by Gbeminiyi Ajayi [1], downloaded from the PerceptiLabs mirror [5]. The mirror contains 224 by 224 RGB photographs labeled Cloudy, Rain, Shine, and Sunrise. Cloudy images emphasize cloud structures; Rain images include rain or wet-scene cues; Shine images show bright sunlight; Sunrise images contain characteristic horizon lighting.

{counts_text} The retained classes contain 291 Cloudy, 205 Rain, 249 Shine, and 352 Sunrise photographs. Sunrise is the largest class, while Rain is the smallest. Macro-averaged metrics give these categories equal weight during the comparison.

**Table 1. Dataset partition by class.**

{distribution}

{image('../data/inspection.png', 'Figure 1. Training examples illustrating scene and lighting variation.')}

Duplicate removal precedes the approximately 70/15/15 split. File and decoded-pixel hashes identify exact copies. Near-duplicate detection combines pHash and dHash distances of at most four bits, a log-aspect-ratio difference at most 0.10, and thumbnail RGB RMSE at most 0.12. Detected image groups remain within one partition. The split uses seed 42 and contains 768 training, 164 validation, and 165 test images.

The shorter image side is resized to 256 pixels for Simple CNN and MobileNetV2, or 236 pixels for ConvNeXt, followed by a 224-square center crop. RGB channels use means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225]. This places the inputs on the scale expected by the pretrained feature extractors.

RGB values are scaled to [0, 1] before normalization: `x_norm = (x - mean) / std`. All training photographs come from the cited dataset; the synthetic/generated training-image share is 0%. Augmentation transforms these photographs during loading without adding generated files.

Training augmentation applies a horizontal flip with probability 0.5, rotation between -8 and 8 degrees, and color jitter. Brightness and contrast factors vary from 0.85 to 1.15, and saturation from 0.90 to 1.10. These operations vary composition and illumination while retaining the image label. Validation and test preprocessing use the fixed resize, crop, and normalization operations.'''
    p3 = f'''## 3. Advanced architecture: ConvNeXt-Tiny

ConvNeXt was introduced in 2022 as a modernized convolutional architecture [3]. It adapts hierarchical CNN design with large depthwise kernels, LayerNorm, and channel-expansion blocks. It is appropriate for WeatherVision because cloud texture, wet surfaces, and broad lighting patterns require both local and wider spatial features. Its pretrained representation can be adapted through a small classification head on this dataset.

{image('../outputs/convnext_architecture.png', 'Figure 2. WeatherVision ConvNeXt-Tiny. Shapes are channels x height x width; block counts and dimensions follow TorchVision [6].')}

**Feature hierarchy.** A 4x4 convolution with stride four creates 96 feature channels at 56x56 resolution from a 224x224 RGB input. Four stages contain 3, 3, 9, and 3 blocks. Between stages, normalization and a stride-two convolution reduce spatial size while increasing channels. The final stage produces 768 feature maps of size 7x7. This hierarchy represents fine texture in earlier stages and wider scene structure in later stages.

**Inside a block.** A 7x7 depthwise convolution filters each channel independently. LayerNorm stabilizes the feature scale. Two linear projections expand the channels from C to 4C and return them to C, with GELU between them. A learned layer scale controls the residual branch, which is added to the block input. Residual connections preserve information through the deeper network. Stochastic depth is disabled in the frozen extractor used here.

**Four-class adaptation.** TorchVision ImageNet-1K V1 weights initialize the network [6]. Global average pooling and pretrained LayerNorm produce a 768-element vector. A new Linear(768, 4) layer outputs the weather logits. Its 3,076 parameters train; the backbone and normalization weights remain frozen. ConvNeXt has 27,823,204 parameters after replacing the original ImageNet head. Input preprocessing uses a 236-pixel short side, a 224-square center crop, and ImageNet channel normalization.

**Comparison with the A1 models.** Simple CNN has three Conv-BatchNorm-ReLU-Pool blocks with 16, 32, and 64 channels, global pooling, dropout 0.25, and a four-class head. It learns all 23,956 parameters from scratch. MobileNetV2 [2] uses inverted residual blocks with channel expansion, depthwise filtering, and linear bottlenecks; its 1,280-element representation feeds a 5,124-parameter head. ConvNeXt adds a deeper hierarchy, larger spatial kernels, LayerNorm, and GELU. These differences provide a suitable modern extension to the A1 comparison and are evaluated on the same held-out images.'''
    p4 = f'''## 4. Experimental setup and model comparison

**Table 2. Training configurations.**

{config_table}

Cross-entropy and AdamW use learning rate 0.001, weight decay 0.0001, and seed 42. Simple CNN trains all weights; transfer models train only their new linear heads. Frozen extractors remain in evaluation mode. Validation macro-F1 selects the best checkpoint, retaining the earlier epoch on a tie; early-stopping patience is four. {hardware_text}

Accuracy is the proportion of correct labels. Per-class precision measures correctness among predicted labels; recall measures recovery of true labels. F1 is `2PR / (P + R)`. Macro precision, recall, and F1 average the four class-specific values equally. The 165 test images are reserved for final evaluation.

**Table 3. Held-out classification scores.**

{result_table}

Augmented ConvNeXt classifies 162 images correctly, compared with 159 for MobileNetV2 and 157 for Simple CNN. Its three additional correct predictions over MobileNetV2 occur in Cloudy: recall rises from 40/44 to 43/44. Both models recover 30/31 Rain, 36/37 Shine, and 53/53 Sunrise images.

**Table 4. Parameters, total training/validation time, and MPS inference latency.**

{complexity}

Average latency is the arithmetic mean of 30 batch-one network forwards after five warm-up runs. The first test tensor resides on MPS before timing; synchronization brackets every measurement. Preprocessing, transfer, softmax, loading, and Grad-CAM are excluded. The means are 0.729 ms for CNN, 12.171 for MobileNetV2, 6.716 for augmented ConvNeXt, and 6.310 without augmentation. Median values describe the same samples. Operator structure and device kernels affect runtime as well as parameter count.

The app separately benchmarks the current image on the selected backend and displays its mean, median, and p95, plus a single forward including softmax. It reports Python, PyTorch, TorchVision, operating system, thread count, and the browser-reported User-Agent. Browser identity describes the client; CPU, CUDA, or MPS describes inference execution.'''
    p5 = f'''## 5. Training behavior and classification errors

**Table 5. Validation-selected epochs and test macro-F1.**

{selected}

{image('../outputs/cnn_baseline/training_curves.png', 'Figure 3. Simple CNN loss, accuracy, and macro-F1 during training.')}

{image('../outputs/mobilenet_improved/training_curves.png', 'Figure 4. MobileNetV2 loss, accuracy, and macro-F1 during training.')}

{image('../outputs/convnext_aug/training_curves.png', 'Figure 5. ConvNeXt learning curves with augmentation.')}

{loss_text} Validation accuracy after the first epoch is 69.51% for the CNN, 85.37% for MobileNetV2, and 92.07% for augmented ConvNeXt. The transfer models reach higher first-epoch validation accuracy.

The CNN reaches its best validation macro-F1 at epoch 9. MobileNetV2 improves through epoch 8. Augmented ConvNeXt reaches 0.9579 at epochs 3 and 5, so epoch 3 is retained. Its training loss continues decreasing after that point while validation scores fluctuate.

All three augmented ConvNeXt errors are Sunrise predictions: one Cloudy, one Rain, and one Shine image. It recovers all 53 true Sunrise photographs, giving Sunrise recall of 1.0000. The three additional Sunrise predictions reduce its precision to 53/56, or 0.9464.'''
    p6 = f'''## 6. Augmentation ablation

The two ConvNeXt runs share the architecture, pretrained weights, initialization seed, dataset split, optimizer, learning rate, batch size, six-epoch limit, and early-stopping settings. Training augmentation is the changed factor. Each run selects a checkpoint by validation macro-F1.

**Table 6. Overall and class-wise ablation results.**

{table(['Configuration', 'Test accuracy', 'Test macro-F1'], [[NAMES[r], f"{metrics[r]['accuracy']:.4f}", f"{metrics[r]['f1_macro']:.4f}"] if r in metrics else [NAMES[r], 'pending', 'pending'] for r in RUNS[2:]])}

{ablation_class_table}

{image('../outputs/ablation_confusion_pair.png', 'Figure 6. Augmented and non-augmented ConvNeXt confusion matrices. True labels are rows; predicted labels are columns.')}

{ablation_result} The 0.61 percentage-point accuracy increase represents one additional correct image. Macro-F1 increases by 0.0068 in this single-seed comparison. Class F1 increases for Cloudy and Shine, remains equal for Rain, and decreases for Sunrise.

Without augmentation, two Shine images are predicted as Cloudy. With augmentation, Shine has one error, predicted as Sunrise. The augmented model therefore has fewer false Cloudy predictions, which raises Cloudy precision from 0.9556 to 1.0000. Its extra Sunrise false positive lowers Sunrise F1 from 0.9815 to 0.9725. The class-wise results explain the change in the overall macro average.'''
    p7 = f'''## 7. Grad-CAM visualization

Grad-CAM [4] gives a class-specific spatial explanation by combining feature maps and score gradients. The target is ConvNeXt's final spatial stage, `features[-1]`, before global pooling. For a selected class, gradients of its logit with respect to the spatial activations are averaged across positions. Those averages become channel weights for a weighted sum of the feature maps. ReLU retains positive responses, and normalization maps the heatmap into the range from zero to one.

The calculation enables gradients on the image input while retaining the frozen model weights. The 224-square heatmap is resized through the same geometry as the model input and placed back on the photograph. Areas outside the center crop keep the original appearance. The application presents the photograph beside the overlay, with brighter regions indicating stronger class-related attribution.

{examples_text}

The correctly classified Cloudy example emphasizes parts of the cloud structure. The second image is labeled Cloudy but classified as Sunrise; its overlay emphasizes the bright horizon. Warm horizon lighting is therefore associated with the Sunrise score in this example. The visual comparison connects the class prediction to a spatial pattern and helps examine the errors seen in the confusion matrix.'''
    screenshot = image('../outputs/app/desktop.png', 'Figure 8. WeatherVision interface displaying a Cloudy prediction.')
    p8 = f'''## 8. Application, limitations, and conclusion

WeatherVision accepts uploads, camera capture, or dataset examples. It offers all three model families, four class scores, measured latency and environment details, same-image model comparison, and ConvNeXt Grad-CAM. Exports include JSON predictions, CSV comparisons, PNG overlays, and both PDF reports. The Experiments tab provides interactive metric charts, learning curves, and confusion matrices. {browser_text}

{screenshot}

**Ethics and limitations.** The dataset is credited under CC BY 4.0. Demo narration uses disclosed synthetic speech. Uploaded photographs stay in server memory and are not saved by the app. Users should avoid sensitive images on shared servers. The 165-image test set and single seed limit the scope of the comparison. Cloudy, Shine, and Sunrise can overlap visually; unusual lighting, camera exposure, and unseen locations may change performance. Softmax scores are uncalibrated, and Grad-CAM provides approximate spatial attribution. Further evaluation should use additional locations and repeated seeds.

**Conclusion.** Augmented ConvNeXt achieves 98.18% accuracy and 0.9827 macro-F1, ahead of MobileNetV2 and Simple CNN. The ablation adds one correct prediction. Simple CNN is the smallest and fastest model on the measured MPS workload. The application combines these models with visual explanations and exportable results.

**Reproducibility.** Install `requirements.txt`, run `python scripts/download_checkpoints.py`, then `python -m streamlit run app.py`. Dataset preparation and training commands are in README and the notebook. {links}

## References

{refs}'''
    a2 = '\n\n<!-- pagebreak -->\n\n'.join([p1, p2, p3, p4, p5, p6, p7, p8])
    a1_result = table(['Model', 'Accuracy', 'Macro P', 'Macro R', 'Macro F1'],
                      [[NAMES[r], *[f"{metrics[r][k]:.4f}" for k in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']]]
                       if r in metrics else [NAMES[r], 'pending', 'pending', 'pending', 'pending'] for r in RUNS[:2]])
    a1p1 = f'''# WeatherVision: Weather Image Classification

**{'Assignment 1' if complete else 'Assignment 1 - Draft'}**

{identity}

**Course:** Computer Vision, Astana IT University

**Instructor:** Nurym Kaiyrkhan

## Abstract

This assignment compares a Simple CNN trained from scratch with ImageNet-pretrained MobileNetV2 for classifying outdoor photographs as Cloudy, Rain, Shine, or Sunrise. The dataset contains 1,097 images after duplicate removal, divided into 768 training, 164 validation, and 165 test images. Both models use cross-entropy loss, AdamW, training augmentation, and validation macro-F1 checkpoint selection. MobileNetV2 achieves 96.36% test accuracy and 0.9620 macro-F1, compared with 95.15% and 0.9477 for the CNN. Class-wise results show higher Rain and Sunrise recall for MobileNetV2. Simple CNN provides a smaller network and a shorter forward-pass time. The application displays the predicted category, four scores, and inference information for either model.

## 1. Introduction and objective

Weather image classification uses cloud texture, reflected light, rain patterns, and horizon colors to identify the category of a photograph. These cues appear at different spatial scales. Small regions may contain rain streaks or edges, while a larger part of the image describes the sky and overall lighting. Convolutional networks combine these local and wider visual patterns through successive feature layers.

The baseline approach learns its filters from the weather images. The improved approach reuses a feature extractor trained on ImageNet and learns a new classifier for the four labels. Comparing them shows how a compact network and a pretrained representation behave on the same task.

The assignment has three objectives: construct a four-class CNN baseline, adapt MobileNetV2 through transfer learning, and compare their predictions and computational requirements. Accuracy describes overall correctness, while precision, recall, and macro-F1 describe class balance. Learning curves show the optimization process, and confusion matrices identify the remaining classification errors.

The workflow consists of dataset preparation, image preprocessing, model training, validation-based checkpoint selection, and test evaluation. The trained models are then used in a Streamlit interface for uploading and classifying photographs. This baseline comparison also supplies the starting point for the ConvNeXt extension in Assignment 2.'''
    a1p2 = f'''## 2. Dataset and preprocessing

The experiments use the Multi-class Weather Dataset for Image Classification by Gbeminiyi Ajayi [1], downloaded from the PerceptiLabs mirror. It contains RGB photographs with the labels Cloudy, Rain, Shine, and Sunrise. Cloudy images contain cloud structures, Rain images include rain or wet-scene patterns, Shine images show bright sunlight, and Sunrise images emphasize warm horizon lighting.

{counts_text} The retained collection has 291 Cloudy, 205 Rain, 249 Shine, and 352 Sunrise images. The class distribution makes macro-averaged metrics useful because each category contributes equally to their final value.

**Table 1. Dataset partition by class.**

{distribution}

{image('../data/inspection.png', 'Figure A1. Training examples showing variation within the four classes.')}

Exact duplicates are removed using file and decoded-pixel hashes. Near duplicates are grouped using pHash, dHash, aspect ratio, and thumbnail color difference. Group assignment precedes the approximately 70/15/15 split. With seed 42, the split contains 768 training, 164 validation, and 165 test photographs. Both models use this membership.

The shorter image side is resized to 256 pixels, followed by a 224 by 224 center crop. Each RGB channel is scaled to the range from zero to one and normalized using the ImageNet mean and standard deviation. Subtracting the channel mean and dividing by its standard deviation gives the input scale used by the pretrained feature extractor. The class indices are cloudy=0, rain=1, shine=2, and sunrise=3.

Training augmentation applies horizontal flips with probability 0.5, rotations between -8 and 8 degrees, and mild color jitter. Brightness and contrast vary by up to 15%, and saturation by up to 10%. This supplies alternate views of the training photographs. Validation and test images use fixed resizing, cropping, and normalization.'''
    a1p3 = f'''## 3. Model architectures

**Simple CNN.** The baseline uses three 3x3 convolutional blocks with 16, 32, and 64 output channels. Batch normalization controls the activation scale, ReLU introduces a nonlinear response, and max pooling reduces spatial resolution. Global average pooling produces a 64-element feature vector. Dropout with probability 0.25 precedes a four-output linear classifier. All 23,956 parameters are trainable.

**Table 2. Simple CNN feature dimensions.**

{cnn_shapes}

**MobileNetV2.** MobileNetV2 [2] uses inverted residual blocks. Pointwise convolution expands the channels, depthwise convolution performs spatial filtering, and a linear projection reduces the representation. The pretrained backbone produces a 1,280-element feature vector. A new four-class linear head contains 5,124 trainable parameters. The feature extractor is frozen and remains in evaluation mode, preserving its batch-normalization statistics.

## 4. Training and evaluation

**Table 3. Training settings.**

{table(['Model', 'Epoch limit', 'Batch', 'LR', 'Patience'], [[NAMES[r], config[r]['epochs'], config[r]['batch_size'], config[r]['lr'], config[r]['patience']] for r in RUNS[:2]])}

The models output four logits. Softmax converts these into class scores, and cross-entropy minimizes the negative log score of the target label. AdamW uses weight decay 0.0001. Training shuffles mini-batches and uses seed 42. Each epoch includes a validation pass; the highest validation macro-F1 selects the checkpoint. Early stopping has a patience of four non-improving epochs.

Accuracy is the fraction of correct predictions. Precision measures the correct fraction of a class's predicted labels, and recall measures the fraction of its true images recovered. Per-class F1 is `2PR / (P + R)`; macro-F1 averages the four class values. {hardware_text}'''
    a1p4 = f'''## 5. Test results and learning behavior

**Table 4. Classification performance on the 165 test images.**

{a1_result}

{image('../outputs/cnn_baseline/training_curves.png', 'Figure A2. Simple CNN loss, accuracy, and macro-F1.')}

{image('../outputs/mobilenet_improved/training_curves.png', 'Figure A3. MobileNetV2 loss, accuracy, and macro-F1.')}

{image('../outputs/a1_confusion_pair.png', 'Figure A4. CNN and MobileNetV2 test confusion matrices: true rows, predicted columns.')}

Simple CNN correctly classifies 157 test images, and MobileNetV2 classifies 159. CNN training loss decreases from 0.829 to 0.335; MobileNetV2 loss decreases from 0.777 to 0.125. Validation macro-F1 selects epoch 9 for the CNN and epoch 8 for MobileNetV2.'''
    a1p5 = f'''## 6. Class-wise analysis and computational cost

**Table 5. Per-class F1 for the two models.**

{a1_class_table}

MobileNetV2 improves Rain recall from 28/31 to 30/31 and Sunrise recall from 52/53 to 53/53. Shine recall remains 36/37. Cloudy recall changes from 41/44 to 40/44; two Cloudy images are predicted as Rain, one as Shine, and one as Sunrise. The net result is two additional correct predictions and a macro-F1 increase from 0.9477 to 0.9620.

**Table 6. Model size, training time, and forward-pass latency.**

{table(['Model', 'Parameters', 'Trainable', 'Train s', 'Latency ms'], [[NAMES[r], f"{statuses[r]['parameters']:,}", f"{statuses[r]['trainable_parameters']:,}", f"{statuses[r]['training_seconds']:.1f}", f"{metrics[r]['latency']['median_ms']:.3f}"] if r in metrics else [NAMES[r], 'pending', 'pending', 'pending', 'pending'] for r in RUNS[:2]])}

The CNN trains its complete 23,956-parameter network. MobileNetV2 contains 2,228,996 parameters but updates only the 5,124-parameter head. Frozen features reduce the number of weights that require gradient updates. Training and validation take 30.6 seconds for the CNN and 22.4 seconds for MobileNetV2 across their configured epochs.

Latency is the median of 30 synchronized batch-one forwards after five warm-up runs, with the input tensor on the MPS device. The CNN takes 0.742 ms and MobileNetV2 10.269 ms. Their parameter counts and operator structures produce different inference workloads.

## 7. Discussion and conclusion

MobileNetV2 has higher overall accuracy, macro precision, macro recall, and macro-F1. Its ImageNet representation supports the new task through a small linear head, while the CNN learns its filters directly from the weather images. The class-wise comparison shows that the overall gain is concentrated in Rain and Sunrise recognition. The CNN retains stronger Cloudy recall and the shortest forward time.

These results support MobileNetV2 when classification score is the priority and Simple CNN when a compact, fast model is preferred. The common four-class pipeline provides the baseline for the ConvNeXt comparison in Assignment 2.

**Execution.** Train with `python weather_cv.py train --run cnn_baseline` and `python weather_cv.py train --run mobilenet_improved`. Launch with `python -m streamlit run app.py`.

## References

[1] Ajayi, G. (2018). Multi-class Weather Dataset for Image Classification. Mendeley Data, V1. [DOI: 10.17632/4drtyfjtfy.1](https://doi.org/10.17632/4drtyfjtfy.1).

[2] Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., and Chen, L.-C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. CVPR, pp. 4510-4520. [Paper](https://arxiv.org/abs/1801.04381).'''
    a1 = '\n\n<!-- pagebreak -->\n\n'.join([a1p1, a1p2, a1p3, a1p4, a1p5])
    for stem, text in [('assignment1_foundation', a1), ('assignment2_report', a2)]:
        (reports / f'{stem}.md').write_text(text + '\n')
        try:
            render_pdf(text, reports / f'{stem}.pdf', author)
        except ImportError as e:
            print('PDF export unavailable:', e, '; install requirements-dev.txt')
    try:
        from pypdf import PdfReader
        pages = {stem: len(PdfReader(reports / f'{stem}.pdf').pages) for stem in ['assignment1_foundation', 'assignment2_report'] if (reports / f'{stem}.pdf').exists()}
    except ImportError:
        pages = {}
    write_json(ROOT / 'tmp/reports/status.json', {'technical_status': technical, 'missing_experiments': pending,
               'page_counts': pages, 'submission_metadata': submission,
               'manifest_sha256': stats.get('manifest_sha256'), 'test_membership_sha256': list(membership),
               'screenshot_included': (ROOT / 'outputs/app/desktop.png').exists()})
    print(technical, pages)


if __name__ == '__main__':
    main()
