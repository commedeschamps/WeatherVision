"""Real-browser checks for the local app, using an isolated Chrome profile.

Start Streamlit first. Requires requirements-ui.txt and Chrome or Playwright Chromium.
"""
import argparse
import io
import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from playwright.sync_api import sync_playwright
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8501')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'tmp/browser')
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    downloads = output / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    checks, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True,
                                    args=['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
                                    **({'executable_path': str(chrome)} if chrome.exists() else {}))
        page = browser.new_page(viewport={'width': 1440, 'height': 1080}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        def settle():
            page.wait_for_timeout(250)
            page.locator('[data-testid="stApp"][data-test-script-state="notRunning"]').wait_for(timeout=60000)
        def snapshot(**kwargs):
            settle()
            return page.screenshot(**kwargs)
        page.goto(args.url)
        page.get_by_role('heading', name='WeatherVision', exact=True).wait_for(timeout=90000)
        def siglip_download(label, filename):
            with page.expect_download() as event:
                page.get_by_role('button', name=label, exact=False).click()
            target = downloads / filename
            event.value.save_as(target)
            return target

        selector = page.get_by_role('combobox', name='Model', exact=True)
        assert selector.input_value() == 'SigLIP 2 Linear Probe'
        page.get_by_role('button', name='Classify image', exact=False).click()
        page.locator('.prediction-summary').wait_for(timeout=90000)
        settle()
        probe = json.loads(siglip_download('Download result', 'siglip2_probe.json').read_text())
        assert probe['score_type'] == 'softmax'
        assert abs(sum(probe['class_scores'].values()) - 1) < 1e-5
        snapshot(path=str(output / 'siglip2.png'), full_page=True)
        page.get_by_text('Show patch occlusion', exact=True).click()
        page.get_by_role('heading', name='Class-conditioned patch occlusion', exact=True).wait_for(timeout=90000)
        settle()
        explanation = json.loads(siglip_download('Download signed importance', 'siglip2_occlusion.json').read_text())
        assert explanation['grid'] == 7 and explanation['region_pixels'] == 32
        assert len(explanation['signed_logit_drops']) == 7
        siglip_download('Download occlusion overlay', 'siglip2_occlusion.png')
        before = probe['class_scores']
        page.get_by_test_id('stSlider').get_by_role('slider').focus()
        page.keyboard.press('ArrowRight')
        settle()
        assert json.loads(siglip_download('Download result', 'siglip2_after_opacity.json').read_text())['class_scores'] == before
        page.get_by_text('Show raw importance map', exact=True).click()
        settle()
        snapshot(path=str(output / 'siglip2_occlusion.png'), full_page=True)
        selector.fill('SigLIP 2 Zero-Shot'); selector.press('ArrowDown'); selector.press('Enter')
        settle()
        page.get_by_role('button', name='Classify image', exact=False).click()
        page.locator('.prediction-summary').wait_for(timeout=90000)
        settle()
        zero = json.loads(siglip_download('Download result', 'siglip2_zero_shot.json').read_text())
        assert zero['score_type'] == 'sigmoid_matching' and 'softmax_confidence' not in zero
        assert 'do not sum to one' in page.locator('body').inner_text()
        checks.append('actual SigLIP 2 default probe, zero-shot score semantics, occlusion and downloads; opacity preserves scores')
        selector.fill('ConvNeXt-Tiny'); selector.press('ArrowDown'); selector.press('Enter')
        settle()
        button = page.get_by_role('button', name='Classify image', exact=False)
        button.click()
        page.locator('.prediction-summary').wait_for(timeout=60000)
        initial_result = page.locator('.prediction-summary').inner_text()
        assert 'Cloudy' in initial_result
        assert 'Softmax confidence' in page.locator('body').inner_text()
        assert page.get_by_test_id('stException').count() == 0
        checks.append('trained ConvNeXt CPU prediction on a dataset example')

        def download(label, filename):
            with page.expect_download() as event:
                page.get_by_role('button', name=label, exact=False).click()
            target = downloads / filename
            event.value.save_as(target)
            return target

        record = json.loads(download('Download result', 'prediction.json').read_text())
        assert record['model'] == 'ConvNeXt-Tiny' and record['device'] == 'CPU'
        assert 'benchmark' not in record and record['server_prediction_ms'] > 0
        assert record['environment']['device'] == 'cpu'
        assert 'Chrome/' in record['browser_user_agent']
        page.get_by_text('Timing and environment', exact=True).click()
        page.get_by_role('button', name='Measure average latency', exact=True).click()
        page.get_by_text('Average forward:', exact=False).wait_for(timeout=60000)
        settle()
        record = json.loads(download('Download result', 'prediction.json').read_text())
        assert record['benchmark']['mean_ms'] > 0 and len(record['benchmark']['measurements_ms']) == 30
        assert record['benchmark']['environment']['device'] == 'cpu'
        assert 'Inference backend: CPU' in page.locator('body').inner_text()
        page.get_by_text('Timing and environment', exact=True).click()
        checks.append('optional average latency, valid protocol, real backend/environment and browser User-Agent')
        assert len(record['class_scores']) == 4 and abs(sum(record['class_scores'].values()) - 1) < 1e-6
        checks.append('prediction JSON download')
        page.get_by_text('Show Grad-CAM', exact=True).click()
        assert page.get_by_role('checkbox', name='Show Grad-CAM', exact=True).is_checked()
        page.get_by_role('heading', name='Grad-CAM explanation', exact=True).wait_for(timeout=60000)
        assert page.locator('.prediction-summary').inner_text() == initial_result
        heatmap = download('Download heatmap', 'downloaded_gradcam.png')
        assert Image.open(heatmap).size == (224, 224)
        checks.append('Grad-CAM toggle preserves prediction and downloads an overlay')
        page.get_by_text('Show Grad-CAM', exact=True).click()
        assert not page.get_by_role('checkbox', name='Show Grad-CAM', exact=True).is_checked()
        assert page.locator('.prediction-summary').inner_text() == initial_result
        page.get_by_role('button', name='Compare on this image', exact=False).click()
        page.get_by_role('heading', name='Predictions on this photograph', exact=True).wait_for(timeout=60000)
        table = pd.read_csv(download('Download image comparison', 'image_comparison.csv'))
        from src.app_utils import MODEL_SPECS
        assert set(table.Model) == set(MODEL_SPECS)
        assert len(table) == len(MODEL_SPECS)
        checks.append('same-image comparison using all five trained model choices')
        for name in ['Simple CNN', 'MobileNetV2', 'ConvNeXt-Tiny']:
            selector = page.get_by_role('combobox', name='Model', exact=True)
            selector.fill(name)
            selector.press('ArrowDown')
            selector.press('Enter')
            page.get_by_text(f'{name} · Server prediction:', exact=False).wait_for()
            settle()
            page.locator('.prediction-summary').wait_for()
            item = json.loads(download('Download result', 'switched_prediction.json').read_text())
            assert item['model'] == name and item['device'] == 'CPU', (name, item['model'], item['device'])
        page.get_by_role('button', name='Classify image', exact=False).click()
        page.get_by_text('Classifying photograph…', exact=False).wait_for(state='hidden', timeout=60000)
        page.locator('.prediction-summary').wait_for()
        repeated = json.loads(download('Download result', 'repeated_prediction.json').read_text())
        assert repeated['class_scores'] == record['class_scores']
        checks.append('switch all three models, preserve cached predictions, and repeat CPU inference')
        page.evaluate('document.querySelectorAll("*").forEach(e => { if(e.scrollTop) e.scrollTop=0; })')
        snapshot(path=str(output / 'desktop.png'), full_page=True, animations='disabled')
        page.get_by_role('tab', name='Experiments', exact=True).click()
        page.get_by_role('heading', name='Test-set results', exact=True).wait_for()
        page.get_by_test_id('stDataFrame').filter(visible=True).locator('canvas').first.wait_for(state='visible')
        plot = page.get_by_test_id('stVegaLiteChart').filter(visible=True)
        plot.locator('svg').wait_for(state='visible')
        page.wait_for_function("""Array.from(document.querySelectorAll('[data-testid="stDataFrame"] canvas')).some(e=>e.getBoundingClientRect().width>500)""")
        assert 'streamlit-generated' not in plot.inner_text()
        snapshot(path=str(output / 'comparison.png'), full_page=True, animations='disabled')
        assert page.get_by_test_id('stException').count() == 0
        checks.append('experiment table and interactive chart')
        page.get_by_role('tab', name='Project', exact=True).click()
        report = download('Assignment 2 report', 'downloaded_report.pdf')
        assert report.read_bytes() == (ROOT/'reports/assignment2_report.pdf').read_bytes()
        a1 = download('Assignment 1 report', 'downloaded_a1.pdf')
        assert a1.read_bytes() == (ROOT/'reports/assignment1_foundation.pdf').read_bytes()
        checks.append('report download and team details')
        page.get_by_role('tab', name='Classify', exact=True).click()
        # A different example must clear predictions for the previous image.
        page.get_by_text('Rain', exact=True).first.click()
        page.get_by_text('Ready to classify', exact=True).wait_for()
        assert page.locator('.prediction-summary').count() == 0
        checks.append('changing an example clears stale predictions')
        page.get_by_role('button', name='Classify image', exact=False).click()
        page.locator('.prediction-summary').wait_for(timeout=60000)
        for width, height, filename in [(820, 1180, 'tablet.png'), (390, 844, 'mobile.png')]:
            page.set_viewport_size({'width': width, 'height': height})
            page.evaluate('document.querySelectorAll("*").forEach(e => { if(e.scrollTop) e.scrollTop=0; })')
            snapshot(path=str(output / filename), full_page=True, animations='disabled')
            assert not page.evaluate('document.documentElement.scrollWidth > window.innerWidth'), f'Horizontal overflow at {width}px'
        page.locator('.prediction-summary').scroll_into_view_if_needed()
        snapshot(path=str(output / 'mobile_result.png'), full_page=True, animations='disabled')
        checks.append('desktop, tablet, and mobile layouts without horizontal overflow')
        page.get_by_role('button', name='Clear image', exact=False).click()
        page.get_by_text('Add a photograph', exact=True).wait_for()
        assert page.get_by_role('button', name='Classify image', exact=False).is_disabled()
        assert page.locator('.prediction-summary').count() == 0
        page.get_by_test_id('stFileUploader').locator('input[type=file]').set_input_files(
            {'name': 'invalid.png', 'mimeType': 'image/png', 'buffer': b'not an image'})
        page.get_by_text('Cannot read this image.', exact=False).wait_for(timeout=15000)
        assert page.get_by_role('button', name='Classify image', exact=False).is_disabled()
        checks.append('clear/reset and invalid image recovery')
        # A valid upload recovers from the rejected file and produces a new result.
        image = Image.new('RGB', (360, 240), (90, 150, 200))
        buffer = io.BytesIO(); image.save(buffer, format='PNG')
        page.get_by_test_id('stFileUploader').locator('input[type=file]').set_input_files(
            {'name': 'outdoor.png', 'mimeType': 'image/png', 'buffer': buffer.getvalue()})
        page.get_by_role('button', name='Classify image', exact=False).click()
        page.locator('.prediction-summary').wait_for(timeout=60000)
        checks.append('valid-upload recovery after a rejected image')
        page.get_by_test_id('stFileUploader').locator('input[type=file]').set_input_files(
            {'name': 'unsupported.txt', 'mimeType': 'text/plain', 'buffer': b'unsupported'})
        page.get_by_text('files are not allowed', exact=False).wait_for(timeout=15000)
        checks.append('unsupported file extension rejected by upload control')
        page.set_viewport_size({'width': 1440, 'height': 1080})
        page.get_by_text('Camera', exact=True).click()
        page.get_by_test_id('stCameraInput').wait_for()
        page.wait_for_function("document.querySelector('[data-testid=stCameraInput] video')?.videoWidth > 0")
        page.get_by_role('button', name='Take Photo', exact=True).click()
        page.get_by_role('button', name='Clear image', exact=False).wait_for()
        page.get_by_role('button', name='Classify image', exact=False).click()
        page.locator('.prediction-summary').wait_for(timeout=60000)
        snapshot(path=str(output/'camera.png'), full_page=True)
        checks.append('camera capture and prediction using a Chrome simulated media device; no physical camera tested')
        assert page.get_by_test_id('stException').count() == 0
        assert not errors, errors
        result = {'status': 'passed', 'url': args.url, 'desktop': [1440, 1080], 'tablet': [820, 1180], 'mobile': [390, 844],
                  'checks': checks, 'page_errors': errors, 'browser_version': browser.version}
        (output / 'browser_checks.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        browser.close()


if __name__ == '__main__':
    main()
