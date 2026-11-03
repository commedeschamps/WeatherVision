"""Mux actual app capture, disclosed synthetic narration, and English subtitles."""
from pathlib import Path
import json,re,subprocess,wave
import numpy as np
import imageio_ffmpeg
ROOT=Path(__file__).resolve().parents[1]
ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
parts=json.loads((ROOT/'docs/demo_segments.json').read_text())
capture=json.loads((ROOT/'tmp/demo_recording/capture.json').read_text())

def stamp(t):
    ms=round(t*1000);return f'{ms//3600000:02}:{ms//60000%60:02}:{ms//1000%60:02},{ms%1000:03}'

captions=[];start=0
for part in parts:
    sentences=re.split(r'(?<=[.!?])\s+',part['narration']);chunks=[]
    for sentence in sentences:
        words=sentence.split();chunk=[]
        for word in words:
            if sum(len(w)+1 for w in chunk)+len(word)>90:
                chunks.append(' '.join(chunk));chunk=[]
            chunk.append(word)
        if chunk:chunks.append(' '.join(chunk))
    count=sum(len(c.split()) for c in chunks);t=start
    for chunk in chunks:
        duration=part['audio_seconds']*len(chunk.split())/count
        captions.append(f'{len(captions)+1}\n{stamp(t)} --> {stamp(t+duration)}\n{chunk}\n');t+=duration
    start+=part['seconds']
srt=ROOT/'docs/WeatherVision_demo.srt';srt.write_text('\n'.join(captions))
output=ROOT/'dist/WeatherVision_demo.mp4';output.parent.mkdir(exist_ok=True)
label=ROOT/'tmp/demo_recording/narration_label.txt';label.write_text('Actual app capture | Synthetic English narration')
filters=f"subtitles={srt}:force_style='Fontname=Arial,Fontsize=12,MarginV=18,Outline=1,Shadow=0',drawtext=textfile={label}:fontfile=/System/Library/Fonts/Helvetica.ttc:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=7:x=14:y=14"
subprocess.run([ffmpeg,'-y','-v','warning','-ss',str(capture['lead_seconds']),'-i',str(ROOT/capture['raw_video']),'-i',str(ROOT/'tmp/demo_audio/narration.wav'),'-t',str(capture['duration_seconds']),'-vf',filters,'-c:v','libx264','-preset','veryfast','-crf','21','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k','-movflags','+faststart',str(output)],check=True)
# Decode the complete deliverable to verify stream integrity.
decoded=subprocess.run([ffmpeg,'-v','error','-i',str(output),'-f','null','-'],capture_output=True,text=True,check=True)
assert not decoded.stderr.strip(),decoded.stderr
reader=imageio_ffmpeg.read_frames(str(output),pix_fmt='rgb24');metadata=next(reader);reader.close()
assert 180<=metadata['duration']<=300
assert tuple(metadata['size'])==(1440,1080)
with wave.open(str(ROOT/'tmp/demo_audio/narration.wav')) as w:
    audio=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(float)
rms=float(np.sqrt(np.mean(audio**2)));assert rms>200
for i,t in enumerate([5,45,95,115,140,155,185,225,240]):
    subprocess.run([ffmpeg,'-y','-v','error','-ss',str(t),'-i',str(output),'-frames:v','1',str(ROOT/f'tmp/demo_recording/review_{i}.png')],check=True)
result={'status':'passed','duration_seconds':metadata['duration'],'resolution':list(metadata['size']),'fps':metadata['fps'],'video_codec':metadata['codec'],'audio_codec':'AAC','audio_rms_pcm16':rms,'full_decode':'passed','narration':'macOS Samantha synthetic text-to-speech; not student voice','subtitles':'English, embedded and docs/WeatherVision_demo.srt','bytes':output.stat().st_size,'screen_capture':'genuine running Streamlit app','review_frames':[5,45,95,115,140,155,185,225,240]}
(ROOT/'outputs/demo_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
