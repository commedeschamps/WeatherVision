"""Assemble the compact Moodle package from verified final deliverables."""
from pathlib import Path
import hashlib,json,shutil,zipfile
ROOT=Path(__file__).resolve().parents[1]
TARGET=ROOT/'dist/final_submission'
TARGET.mkdir(parents=True,exist_ok=True)
submission=__import__('yaml').safe_load((ROOT/'configs/submission.yaml').read_text())
video=json.loads((ROOT/'outputs/demo_verification.json').read_text())
assert video['status']=='passed' and video['public_download_verified']
assert submission['repository_url'] and submission['video_url']
files={
'reports/assignment2_report.pdf':'assignment2_report.pdf',
'reports/assignment1_foundation.pdf':'assignment1_foundation.pdf',
'notebooks/WeatherVision_Colab.ipynb':'WeatherVision_Colab.ipynb',
'README.md':'README.md',
'docs/LOCAL_RUN_GUIDE.md':'REPRODUCIBILITY.md',
'docs/FINAL_DEMO_SCRIPT.md':'FINAL_DEMO_SCRIPT.md',
'docs/DEMO_RECORDING_PLAN.md':'DEMO_RECORDING_PLAN.md',
'docs/DEFENSE_GUIDE.md':'DEFENSE_GUIDE.md',
'docs/FINAL_REQUIREMENTS_AUDIT.md':'FINAL_REQUIREMENTS_AUDIT.md',
'docs/FINAL_VERIFICATION_LOG.md':'FINAL_VERIFICATION_LOG.md',
'docs/WeatherVision_demo.srt':'WeatherVision_demo.srt',
'outputs/model_comparison.csv':'results/model_comparison.csv',
'outputs/ablation.csv':'results/ablation.csv',
'outputs/ablation_analysis.json':'results/ablation_analysis.json',
'outputs/full_cpu_reconstruction.json':'verification/full_cpu_reconstruction.json',
'outputs/verification.json':'verification/project_verification.json',
'outputs/fresh_clone_verification.json':'verification/fresh_clone_verification.json',
'outputs/fresh_app/browser_checks.json':'verification/fresh_app_browser_checks.json',
'outputs/app/browser_checks.json':'verification/browser_checks.json',
'outputs/app/camera_simulation.json':'verification/camera_simulation.json',
'outputs/pytest_final.xml':'verification/pytest_final.xml',
'outputs/final_pdf_review.json':'verification/final_pdf_review.json',
'outputs/demo_verification.json':'verification/demo_verification.json',
'outputs/notebook_execution.json':'verification/notebook_execution.json',
'outputs/final_report_downloads.json':'verification/final_report_downloads.json',
'outputs/final_requirement_status.json':'verification/final_requirement_status.json',
'configs/checkpoints.json':'verification/checkpoints.json',
'outputs/app/desktop.png':'screenshots/desktop.png',
'outputs/app/mobile_result.png':'screenshots/mobile_result.png',
'outputs/app/comparison.png':'screenshots/experiments.png',
'outputs/app/project.png':'screenshots/project.png',
'outputs/convnext_architecture.png':'results/convnext_architecture.png',
}
for source,target in files.items():
 path=TARGET/target;path.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/source,path)
# The actual local video is delivered separately; the compact ZIP contains its public link.
(TARGET/'DEMO_VIDEO.txt').write_text(f"WeatherVision demonstration (4:08)\n{submission['video_url']}\n\nLocal file: ../WeatherVision_demo.mp4\nActual app screen capture with disclosed synthetic English narration and subtitles.\n")
(TARGET/'SUBMISSION_CHECKLIST.md').write_text(f'''# Submission checklist

Rakhman Seidigali · SE-2411 · Astana IT University · Computer Vision

- [x] Assignment 2 report: assignment2_report.pdf — 8 pages, visually reviewed.
- [x] Assignment 1 foundation: assignment1_foundation.pdf — 5 pages, visually reviewed.
- [x] Executed notebook: WeatherVision_Colab.ipynb — all 15 code cells executed locally, no errors.
- [x] Public repository: {submission['repository_url']}
- [x] Four model weights distributed through GitHub Releases; checksum downloader tested in a fresh clone.
- [x] Application access: tested local guide in REPRODUCIBILITY.md.
- [x] Demo: {submission['video_url']} — 4:08, actual app footage, disclosed synthetic English narration, subtitles.
- [x] Results, ablation, architecture, screenshots, audit and verification evidence included.
- [x] Defense guide: 48 short questions and answers.
- [ ] PENDING — student uploads the materials to Moodle and confirms the final submission.
- [ ] PENDING — student attends the practical defense.

Public Streamlit hosting and hosted Colab execution are not claimed. The local application and notebook were verified. Physical camera hardware was not tested; the camera workflow passed with a simulated device.

The compact ZIP excludes model weights and the MP4 to avoid duplicate downloads. The video is available at the public link above and as dist/WeatherVision_demo.mp4. Source is in the public repository and dist/WeatherVision_source.zip.

Deadline: November 7, 2026, 23:59, Asia/Almaty.
''')
(TARGET/'MOODLE_SUBMISSION.txt').write_text(f'''WeatherVision: Weather Image Classification Using Deep Learning and Transfer Learning
Rakhman Seidigali — SE-2411
Astana IT University — Computer Vision — Nurym Kaiyrkhan

Repository: {submission['repository_url']}
Demo video (4:08): {submission['video_url']}
Model weights: https://github.com/commedeschamps/WeatherVision/releases/tag/v1.0.0
Application local run guide: https://github.com/commedeschamps/WeatherVision/blob/main/docs/LOCAL_RUN_GUIDE.md

Files to submit:
1. assignment2_report.pdf (8 pages)
2. WeatherVision_final_submission.zip (contains A1 foundation, executed notebook, reproducibility guide, results, audit and screenshots)
Optional separate attachments if requested: assignment1_foundation.pdf, WeatherVision_Colab.ipynb.

Local app: install requirements.txt, run python scripts/download_checkpoints.py, then python -m streamlit run app.py.
Video: genuine app recording with synthetic English narration and subtitles.

Deadline: November 7, 2026, 23:59 Asia/Almaty.
After uploading, save and confirm the final submission in Moodle. Attend the practical defense.
''')
manifest={str(p.relative_to(TARGET)):{'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()} for p in sorted(TARGET.rglob('*')) if p.is_file() and p.name!='FILES_SHA256.json'}
(TARGET/'FILES_SHA256.json').write_text(json.dumps(manifest,indent=2)+'\n')
archive=ROOT/'dist/WeatherVision_final_submission.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(TARGET.rglob('*')):
  if p.is_file():z.write(p,Path('WeatherVision_final_submission')/p.relative_to(TARGET))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None
 assert not any(n.endswith(('.pt','.pth')) or '/.venv/' in n or '/raw/' in n for n in z.namelist())
 for name,info in manifest.items():
  assert hashlib.sha256(z.read('WeatherVision_final_submission/'+name)).hexdigest()==info['sha256']
print(f'{len(manifest)} verified files; {archive.stat().st_size/1024**2:.2f} MiB; no checkpoints/raw data/cache.')
(ROOT/'outputs/final_package_verification.json').write_text(json.dumps({'status':'passed','files':len(manifest),'archive':str(archive.relative_to(ROOT)),'bytes':archive.stat().st_size,'sha256':hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),'all_members_checksums_verified':True,'weights_in_archive':False,'video_distributed_by_public_url':submission['video_url']},indent=2)+'\n')
