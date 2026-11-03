import io
import json

import pandas as pd
import pytest
from PIL import Image, UnidentifiedImageError

from src.app_utils import decode_upload, example_catalog, export_prediction, result_key, sync_image_state


def test_upload_validation_and_exif_orientation():
    image = Image.new('RGB', (40, 20), '#b1c7d0')
    exif = image.getexif()
    exif[274] = 6
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', exif=exif)
    assert decode_upload(buffer.getvalue()).size == (20, 40)
    with pytest.raises(ValueError, match='empty'):
        decode_upload(b'')
    with pytest.raises(UnidentifiedImageError):
        decode_upload(b'not an image')
    with pytest.raises(ValueError, match='12 MB'):
        decode_upload(b'x' * (12 * 1024 * 1024 + 1))


def test_example_catalog_uses_only_safe_existing_test_files(tmp_path):
    image = tmp_path / 'cloud.jpg'
    image.write_bytes(b'image placeholder')
    outside = tmp_path.parent / f'{tmp_path.name}-outside.jpg'
    outside.write_bytes(b'existing file outside the project')
    frame = pd.DataFrame([
        {'path': image.name, 'label': 'cloudy', 'split': 'test'},
        {'path': image.name, 'label': 'rain', 'split': 'train'},
        {'path': f'../{outside.name}', 'label': 'shine', 'split': 'test'},
        {'path': 'missing.jpg', 'label': 'sunrise', 'split': 'test'},
    ])
    path = tmp_path / 'manifest.csv'
    frame.to_csv(path, index=False)
    result = example_catalog(path, tmp_path)
    assert result['Cloudy'] == [str(image)]
    assert result['Rain'] == result['Shine'] == result['Sunrise'] == []


def test_display_changes_preserve_results_and_input_changes_clear_them():
    state = {}
    digest = sync_image_state(state, b'first image')
    key = result_key(digest, 'convnext_aug', 'cpu', 1)
    state['image_predictions'][key] = {'label': 'cloudy'}
    state['image_heatmaps'][key] = 'overlay'
    state['comparison_shown'] = True
    state['show_gradcam'] = True
    assert sync_image_state(state, b'first image') == digest
    assert state['image_predictions'][key]['label'] == 'cloudy'
    assert state['comparison_shown']
    assert result_key(digest, 'mobilenet_improved', 'cpu', 1) != key
    assert result_key(digest, 'convnext_aug', 'cpu', 2) != key
    sync_image_state(state, b'new image')
    assert state['image_predictions'] == state['image_heatmaps'] == {}
    assert not state['comparison_shown']


def test_result_export_keeps_scores_and_image_metadata():
    prediction = {'label': 'shine', 'confidence': .7, 'probabilities': [.1, .1, .7, .1], 'latency_ms': 12.5}
    result = json.loads(export_prediction(prediction, 'MobileNetV2', 'cpu', '/tmp/scene.jpg', (320, 240)))
    assert result['image'] == {'name': 'scene.jpg', 'width': 320, 'height': 240}
    assert result['predicted_class'] == 'Shine'
    assert result['class_scores']['Shine'] == .7
    assert sum(result['class_scores'].values()) == pytest.approx(1)
    assert result['forward_ms'] == 12.5
