"""Exercise camera capture with Chrome's simulated media device (no physical camera)."""
import json
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',args=['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream'])
    context=browser.new_context(permissions=['camera'],viewport={'width':1440,'height':1080})
    page=context.new_page();page.goto('http://127.0.0.1:8504')
    page.get_by_role('heading',name='Classifier',exact=True).wait_for();page.get_by_text('Camera',exact=True).click()
    page.get_by_test_id('stCameraInput').wait_for()
    page.wait_for_function("document.querySelector('[data-testid=stCameraInput] video')?.videoWidth > 0")
    page.get_by_role('button',name='Take Photo',exact=True).click()
    page.get_by_role('button',name='Clear image',exact=False).wait_for()
    page.get_by_role('button',name='Classify image',exact=False).click()
    page.locator('.prediction-summary').wait_for(timeout=60000)
    page.locator('[data-testid="stApp"][data-test-script-state="notRunning"]').wait_for(timeout=60000)
    assert page.get_by_test_id('stException').count()==0
    page.screenshot(path=str(ROOT/'outputs/app/camera_simulation.png'),full_page=True)
    result={'status':'passed','capture':'Chrome simulated media device','physical_camera_tested':False,'flows':['permission granted','video preview','Take Photo','CPU prediction on captured frame'],'browser_version':browser.version}
    (ROOT/'outputs/app/camera_simulation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));browser.close()
