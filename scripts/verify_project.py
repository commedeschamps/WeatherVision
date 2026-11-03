"""Verify saved data, checkpoints, metrics, notebook and reports without retraining."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbformat
import numpy as np
import pandas as pd
import torch
from pypdf import PdfReader
from torch.utils.data import DataLoader

from src.config import Experiment, load_config, validate_ablation
from src.data import WeatherDataset, open_rgb, verify_content
from src.engine import calculate_metrics, load_checkpoint, predict_image
from src.utils import CLASSES, seed_everything, sha256_file, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-inference', action='store_true', help='Compare all test predictions with actual CPU inference')
    parser.add_argument('--output', type=Path, default=ROOT / 'tmp/verification.json')
    args = parser.parse_args()
    seed_everything(42)
    manifest_path = ROOT / "data/manifest.csv"
    frame = pd.read_csv(manifest_path)
    verify_content(frame)
    stats = json.loads((ROOT / "data/statistics.json").read_text())
    assert stats['manifest_sha256'] == sha256_file(manifest_path)
    assert stats['retained_images'] == len(frame)
    audit = pd.read_csv(ROOT / 'data/image_audit.csv')
    assert stats['valid_images_before_dedup'] == len(audit)
    assert stats['exact_duplicates_removed'] == int((audit.status == 'exact_duplicate_removed').sum())
    assert stats['corrupt_images'] == len(pd.read_csv(ROOT / 'data/corrupted_images.csv'))
    groups = audit.set_index('path').group.to_dict()
    for pair in pd.read_csv(ROOT / 'data/duplicate_pairs.csv').itertuples():
        assert groups[pair.path_a] == groups[pair.path_b]
    configs = load_config(ROOT / 'configs/experiments.yaml')
    validate_ablation(configs['convnext_aug'], configs['convnext_noaug'])
    test = frame[frame.split == 'test'].reset_index(drop=True)
    test_hash = hashlib.sha256('\n'.join(test.path).encode()).hexdigest()
    model_results = {}
    saved_configs = {}
    for run, config in configs.items():
        directory = ROOT / 'outputs' / run
        metrics = json.loads((directory / 'test_metrics.json').read_text())
        status = json.loads((directory / 'status.json').read_text())
        predictions = pd.read_csv(directory / 'test_predictions.csv')
        assert status['state'] == 'complete'
        assert status['config'] == config.to_dict()
        assert metrics['manifest_sha256'] == stats['manifest_sha256']
        assert metrics['test_membership_sha256'] == test_hash
        assert metrics['checkpoint_sha256'] == sha256_file(directory / 'best.pt')
        assert predictions.path.tolist() == test.path.tolist()
        assert predictions.class_id.tolist() == test.class_id.tolist()
        calculated = calculate_metrics(predictions.class_id.tolist(), predictions.predicted_class_id.tolist())
        for key in ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'confusion_matrix']:
            assert np.allclose(calculated[key], metrics[key])
        scores = predictions[[f'score_{label}' for label in CLASSES]].to_numpy()
        assert np.allclose(scores.sum(1), 1, atol=1e-6)
        assert np.array_equal(scores.argmax(1), predictions.predicted_class_id)
        model, checkpoint = load_checkpoint(directory / 'best.pt', 'cpu')
        saved_configs[run] = Experiment(**checkpoint['config'])
        result = predict_image(model, open_rgb(ROOT / test.iloc[0].path), config.model, torch.device('cpu'))
        assert len(result['probabilities']) == 4
        assert abs(sum(result['probabilities']) - 1) < 1e-6
        if args.full_inference:
            dataset = WeatherDataset(frame, ROOT, 'test', False, config.model)
            predicted = []
            with torch.inference_mode():
                for inputs, _ in DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0):
                    predicted.extend(model(inputs).argmax(1).tolist())
            assert predicted == predictions.predicted_class_id.tolist(), f'CPU prediction mismatch: {run}'
        model_results[run] = {'actual_trained_cpu_inference': 'passed', 'test_metrics_recalculated': 'passed',
                              'cpu_inference_images': len(test) if args.full_inference else 1,
                              'best_epoch': status['best_epoch'], 'test_accuracy': metrics['accuracy']}
    validate_ablation(saved_configs['convnext_aug'], saved_configs['convnext_noaug'])
    notebook = nbformat.read(ROOT / 'notebooks/WeatherVision_Colab.ipynb', as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            compile(cell.source, '<notebook cell>', 'exec')
    assert any(cell.get('outputs') for cell in notebook.cells if cell.cell_type == 'code'), 'Notebook requires executed results'
    assert not any(o.get('output_type') == 'error' for c in notebook.cells for o in c.get('outputs', []))
    page_counts = {stem: len(PdfReader(ROOT / 'reports' / f'{stem}.pdf').pages)
                   for stem in ['assignment1_foundation', 'assignment2_report']}
    assert 3 <= page_counts['assignment1_foundation'] <= 5
    assert 6 <= page_counts['assignment2_report'] <= 8
    gradcam = json.loads((ROOT / 'outputs/convnext_aug/gradcam_metadata.json').read_text())
    assert any(e['correct'] for e in gradcam['examples'])
    if gradcam['errors_available']:
        assert any(not e['correct'] for e in gradcam['examples'])
    for example in gradcam['examples']:
        heatmap = np.load(ROOT / 'outputs/convnext_aug' / example['figure'].replace('.png', '.npy'))
        assert heatmap.shape == (224, 224) and np.isfinite(heatmap).all()
        assert heatmap.min() >= 0 and heatmap.max() <= 1
    result = {'status': 'passed', 'dataset_images': len(frame), 'test_images': len(test),
              'manifest_sha256': stats['manifest_sha256'], 'test_membership_sha256': test_hash,
              'all_detected_duplicate_pairs_grouped': True, 'models': model_results,
              'notebook': 'schema valid; all code cells compile; executed outputs present, no cell errors; local kernel, not hosted Colab',
              'report_page_counts': page_counts, 'gradcam_examples': len(gradcam['examples'])}
    write_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
