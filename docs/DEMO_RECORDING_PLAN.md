# Demo recording plan

Record the actual Streamlit app at 1440×1080, using an isolated Chrome profile. Keep the image, score bars, timing panel, comparison, Grad-CAM, charts, and PDF download controls readable.

| Time | Actual screen action |
|---|---|
| 0:00–0:30 | Open app title and Classify workspace; introduce classes. |
| 0:30–1:15 | Show Project dataset/model descriptions. |
| 1:15–2:45 | Upload a real test photo, classify on CPU, show environment and mean latency, compare models, show Grad-CAM, download JSON/CSV/PNG, switch models. |
| 2:45–3:30 | Show Experiments, ablation, training curves and confusion matrix. |
| 3:30–4:00 | Show Project reports and attribution; conclude. |

The delivered MP4 combines genuine browser screen capture with macOS Samantha text-to-speech, explicitly labeled as synthetic narration. No synthetic images are used as training data. Timed English subtitles accompany the recording. The raw screen capture is retained locally under tmp/demo_recording for traceability.

Verify: MP4 duration between 180 and 300 seconds, H.264 video, AAC audio, 1440×1080, successful full decode, non-silent narration, opening/middle/final frames. Review the script alongside the video. A personal oral defense is performed by the student during practical lessons. To substitute the student’s own voice, read FINAL_DEMO_SCRIPT.md while recording the same actions and replace the synthetic audio track.
