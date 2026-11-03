"""Capture genuine WeatherVision browser interactions for the narrated demo."""
import argparse,json,time
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--url',default='http://127.0.0.1:8504');args=parser.parse_args()
    segments=json.loads((ROOT/'docs/demo_segments.json').read_text())
    starts=[];total=0
    for s in segments: starts.append(total);total+=s['seconds']
    output=ROOT/'tmp/demo_recording';output.mkdir(parents=True,exist_ok=True)
    frame=__import__('pandas').read_csv(ROOT/'data/manifest.csv')
    photo=ROOT/frame[(frame.split=='test')&(frame.label=='cloudy')].iloc[0].path
    with sync_playwright() as p:
        browser=p.chromium.launch(headless=True,executable_path='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome')
        context=browser.new_context(viewport={'width':1440,'height':1080},record_video_dir=str(output),record_video_size={'width':1440,'height':1080},accept_downloads=True)
        page=context.new_page();capture_start=time.monotonic();page.goto(args.url)
        page.get_by_role('heading',name='WeatherVision',exact=True).wait_for()
        start=time.monotonic();lead=start-capture_start
        def wait_to(seconds):
            while (remaining:=start+seconds-time.monotonic())>0:
                page.wait_for_timeout(min(remaining*1000,5000))
        def top():page.evaluate('document.querySelectorAll("*").forEach(e=>{if(e.scrollTop)e.scrollTop=0})')
        def download(label,filename):
            with page.expect_download() as event:page.get_by_role('button',name=label,exact=False).click()
            event.value.save_as(output/filename)
        wait_to(starts[1]);page.get_by_role('tab',name='Project',exact=True).click();top();print('Dataset and models',flush=True)
        wait_to(starts[2]);page.get_by_role('tab',name='Classify',exact=True).click();page.get_by_text('Upload',exact=True).click()
        page.get_by_test_id('stFileUploader').locator('input[type=file]').set_input_files(str(photo));top();print('Upload real photograph',flush=True)
        wait_to(starts[2]+4);page.get_by_role('button',name='Classify image',exact=False).click();page.locator('.prediction-summary').wait_for(timeout=60000);top()
        wait_to(starts[2]+25);page.get_by_text('Timing and environment',exact=True).click();page.get_by_text('Inference backend: CPU',exact=False).scroll_into_view_if_needed()
        wait_to(starts[2]+40);page.get_by_text('Timing and environment',exact=True).click();page.get_by_role('button',name='Compare on this image',exact=False).click()
        page.get_by_role('heading',name='Predictions on this photograph',exact=True).wait_for(timeout=60000);page.get_by_role('heading',name='Predictions on this photograph',exact=True).evaluate("e => e.scrollIntoView({block: 'start'})");print('Comparison',flush=True)
        wait_to(starts[2]+52);page.get_by_text('Show Grad-CAM',exact=True).click();page.get_by_role('heading',name='Grad-CAM explanation',exact=True).wait_for(timeout=60000);page.get_by_role('heading',name='Grad-CAM explanation',exact=True).evaluate("e => e.scrollIntoView({block: 'start'})");print('Grad-CAM',flush=True)
        wait_to(starts[2]+68);download('Download heatmap','demo_heatmap.png');download('Download image comparison','demo_comparison.csv');download('Download result','demo_result.json')
        top();selector=page.get_by_role('combobox',name='Model',exact=True);selector.click();selector.press('Home');selector.press('ArrowDown');selector.press('Enter')
        wait_to(starts[3]);page.get_by_role('tab',name='Experiments',exact=True).click();top();print('Experiments',flush=True)
        wait_to(starts[3]+25);page.get_by_role('combobox',name='Inspect experiment',exact=True).scroll_into_view_if_needed();page.get_by_text('Augmentation ablation',exact=True).scroll_into_view_if_needed()
        wait_to(starts[4]);page.get_by_role('tab',name='Project',exact=True).click();top();print('Conclusion and report downloads',flush=True)
        wait_to(total);video=page.video;context.close();path=Path(video.path());browser.close()
    (output/'capture.json').write_text(json.dumps({'raw_video':str(path.relative_to(ROOT)),'lead_seconds':lead,'duration_seconds':total,'viewport':[1440,1080],'source':'actual running Streamlit application','narration':'macOS Samantha text-to-speech, synthetic','physical_camera_recorded':False},indent=2)+'\n')
    print(path,flush=True)

if __name__=='__main__':main()
