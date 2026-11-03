"""Real-browser checks for explanation controls, caching, devices and responsiveness."""
import argparse,json,time
from pathlib import Path
from playwright.sync_api import sync_playwright, expect
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--url',default='http://127.0.0.1:8507');args=parser.parse_args()
out=ROOT/'outputs/technical_audit/app';out.mkdir(exist_ok=True,parents=True);downloads=ROOT/'tmp/technical_downloads';downloads.mkdir(exist_ok=True,parents=True)
checks=[];errors=[]
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome');page=browser.new_page(viewport={'width':1440,'height':1080});page.on('pageerror',lambda e:errors.append(str(e)))
 def settle():
  page.wait_for_timeout(250);page.locator('[data-testid="stApp"][data-test-script-state="notRunning"]').wait_for(timeout=60000)
 def download(label,name):
  with page.expect_download() as e:page.get_by_role('button',name=label,exact=True).click()
  target=downloads/name;e.value.save_as(target);return target.read_bytes()
 def select(label,value):
  if label in ['Inference device','Explanation stage']:
   group=page.get_by_test_id('stRadio').filter(has=page.get_by_text(label,exact=True))
   group.get_by_text(value,exact=True).click();settle()
   assert group.get_by_role('radio',name=value,exact=True).is_checked()
  else:
   selector=page.get_by_role('combobox',name=label,exact=True);selector.fill(value);selector.press('ArrowDown');selector.press('Enter');settle()
   assert selector.input_value()==value

 page.goto(args.url);page.get_by_role('heading',name='WeatherVision',exact=True).wait_for(timeout=90000)
 start=time.perf_counter();page.get_by_role('button',name='Classify image',exact=False).click();page.locator('.prediction-summary').wait_for(timeout=60000);settle();click_to_ready_ms=(time.perf_counter()-start)*1000
 initial=json.loads(download('Download result','initial.json'));assert 'benchmark' not in initial and initial['device']=='CPU'
 checks.append('prediction returns without automatic 35-forward benchmark; CPU/backend metadata correct')
 page.get_by_text('Show Grad-CAM',exact=True).click();page.get_by_role('heading',name='Grad-CAM explanation',exact=True).wait_for();settle()
 page.get_by_text('Show raw heatmap',exact=True).click();settle()
 assert page.locator('.cam-legend').is_visible()
 original_raw=download('Download raw heatmap','raw0.png');original_overlay=download('Download heatmap','overlay0.png')
 slider=page.get_by_test_id('stSlider').get_by_role('slider');slider.press('End');settle()
 assert download('Download raw heatmap','raw1.png')==original_raw
 assert download('Download heatmap','overlay1.png')!=original_overlay
 assert json.loads(download('Download result','opacity.json'))['class_scores']==initial['class_scores']
 checks.append('opacity changes only rendered overlay; raw map and prediction exports unchanged')
 select('Explanation stage','Earlier stage · 14 × 14 (experimental)')
 page.get_by_text('native spatial grid 14 × 14',exact=False).wait_for();earlier_raw=download('Download raw heatmap','earlier.png');assert earlier_raw!=original_raw
 assert json.loads(download('Download result','earlier.json'))['class_scores']==initial['class_scores']
 checks.append('earlier 14x14 stage is functional, labeled experimental, and preserves the predicted class')
 select('Explanation stage','Final stage · 7 × 7');assert download('Download raw heatmap','final_again.png')==original_raw
 page.get_by_role('heading',name='Grad-CAM explanation',exact=True).evaluate("e=>e.scrollIntoView({block:'start'})");settle();page.screenshot(path=str(out/'explanation_desktop.png'),full_page=True)
 for width,height in [(820,1180),(390,844)]:
  page.set_viewport_size({'width':width,'height':height});settle()
  page.get_by_role('heading',name='Grad-CAM explanation',exact=True).evaluate("e=>e.scrollIntoView({block:'start'})");settle()
  assert not page.evaluate('document.documentElement.scrollWidth > innerWidth')
  page.screenshot(path=str(out/f'explanation_{width}.png'),full_page=True)
 page.set_viewport_size({'width':1440,'height':1080})
 checks.append('explanation controls/legend/raw map have no horizontal overflow at desktop, tablet, mobile sizes')
 page.get_by_text('Timing and environment',exact=True).click();start=time.perf_counter();page.get_by_role('button',name='Measure average latency',exact=True).click();page.get_by_text('Average forward:',exact=False).wait_for(timeout=60000);settle();benchmark_click_ms=(time.perf_counter()-start)*1000
 measured=json.loads(download('Download result','measured.json'));assert measured['benchmark']['repeats']==30
 timestamp=measured['benchmark']['measured_at_utc'];page.get_by_role('button',name='Classify image',exact=False).click();settle();repeated=json.loads(download('Download result','repeated.json'))
 assert repeated['benchmark']['measured_at_utc']==timestamp and repeated['class_scores']==initial['class_scores']
 checks.append('explicit average benchmark is cached across repeat prediction on the same image/model/device/checkpoint')
 page.get_by_text('Timing and environment',exact=True).click()
 page.get_by_text('Prediction settings',exact=True).click();select('Inference device','Automatic')
 import torch
 expected_auto='CUDA' if torch.cuda.is_available() else 'MPS' if torch.backends.mps.is_available() else 'CPU'
 if expected_auto != 'CPU':
  page.get_by_text('Ready to classify',exact=True).wait_for();assert page.locator('.prediction-summary').count()==0 and page.get_by_role('heading',name='Grad-CAM explanation',exact=True).count()==0
 page.get_by_role('button',name='Classify image',exact=False).click();page.locator('.prediction-summary').wait_for(timeout=60000);settle()
 automatic=json.loads(download('Download result','automatic.json'));assert automatic['device']==expected_auto
 if expected_auto!='CPU':assert 'benchmark' not in automatic
 assert max(abs(a-b) for a,b in zip(automatic['class_scores'].values(),initial['class_scores'].values()))<1e-3
 select('Inference device','CPU');settle();cpu_again=json.loads(download('Download result','cpu_again.json'));assert cpu_again['benchmark']['measured_at_utc']==timestamp
 checks.append('device switch separates caches, hides stale explanation, predicts on actual automatic backend, and restores matching CPU cache')
 select('Model','MobileNetV2');page.get_by_text('Ready to classify',exact=True).wait_for();assert page.get_by_role('heading',name='Grad-CAM explanation',exact=True).count()==0
 page.get_by_role('button',name='Classify image',exact=False).click();page.locator('.prediction-summary').wait_for(timeout=60000);settle();assert json.loads(download('Download result','mobile.json'))['model']=='MobileNetV2'
 checks.append('model switch hides ConvNeXt heatmap and displays corresponding MobileNet prediction')
 assert not errors and page.get_by_test_id('stException').count()==0
 result={'status':'passed','url':args.url,'checks':checks,'page_errors':errors,'browser_version':browser.version,'automatic_backend':automatic['device'],'click_to_ready_ms':click_to_ready_ms,'optional_benchmark_click_ms':benchmark_click_ms,'server_prediction_ms':initial['server_prediction_ms'],'single_forward_ms':initial['forward_ms'],'warmed_cpu_mean_ms':measured['benchmark']['mean_ms']}
 (out/'technical_browser_checks.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));browser.close()
