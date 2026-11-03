"""Real-browser checks for the local app, using an isolated Chrome profile.

Start Streamlit first. Requires requirements-ui.txt and Chrome or Playwright Chromium.
"""
import argparse
import io
import json
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def main():
    from playwright.sync_api import sync_playwright
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8501')
    args = parser.parse_args()
    output = ROOT / 'outputs/app'
    output.mkdir(parents=True, exist_ok=True)
    downloads = ROOT / 'tmp/app_downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    chrome = Path('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
    checks, errors = [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, **({'executable_path': str(chrome)} if chrome.exists() else {}))
        page = browser.new_page(viewport={'width': 1440, 'height': 1080}, device_scale_factor=1)
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.goto(args.url)
        page.get_by_role('heading', name='WeatherVision', exact=True).wait_for()
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
        assert set(table.Model) == {'Simple CNN', 'MobileNetV2', 'ConvNeXt-Tiny'}
        assert len(table) == 3
        checks.append('same-image comparison using all three trained checkpoints')
        page.evaluate('document.querySelectorAll("*").forEach(e => { if(e.scrollTop) e.scrollTop=0; })')
        page.screenshot(path=str(output / 'desktop.png'), full_page=True, animations='disabled')
        page.get_by_role('tab', name='Experiments', exact=True).click()
        page.get_by_role('heading', name='Test-set results', exact=True).wait_for()
        page.get_by_test_id('stDataFrame').filter(visible=True).locator('canvas').first.wait_for(state='visible')
        plot = page.get_by_test_id('stVegaLiteChart').filter(visible=True)
        plot.locator('svg').wait_for(state='visible')
        page.wait_for_function("""Array.from(document.querySelectorAll('[data-testid="stDataFrame"] canvas')).some(e=>e.getBoundingClientRect().width>500)""")
        assert 'streamlit-generated' not in plot.inner_text()
        page.screenshot(path=str(output / 'comparison.png'), full_page=True, animations='disabled')
        assert page.get_by_test_id('stException').count() == 0
        checks.append('experiment table and interactive chart')
        page.get_by_role('tab', name='Project', exact=True).click()
        report = download('Assignment 2 report', 'downloaded_report.pdf')
        assert report.read_bytes().startswith(b'%PDF-')
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
            page.screenshot(path=str(output / filename), full_page=True, animations='disabled')
            assert not page.evaluate('document.documentElement.scrollWidth > window.innerWidth'), f'Horizontal overflow at {width}px'
        page.locator('.prediction-summary').scroll_into_view_if_needed()
        page.screenshot(path=str(output / 'mobile_result.png'), full_page=True, animations='disabled')
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
        assert page.get_by_test_id('stException').count() == 0
        assert not errors, errors
        result = {'status': 'passed', 'desktop': [1440, 1080], 'tablet': [820, 1180], 'mobile': [390, 844],
                  'checks': checks, 'page_errors': errors, 'browser_version': browser.version}
        (output / 'browser_checks.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result, indent=2))
        browser.close()


if __name__ == '__main__':
    main()
