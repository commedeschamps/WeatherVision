"""Measure loaded-model single prediction versus the previous automatic benchmark policy."""
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from src.data import open_rgb,transform
from src.engine import load_checkpoint,predict_image
from src.evaluation import benchmark
from src.utils import seed_everything,environment,synchronize,write_json
seed_everything(42);frame=pd.read_csv(ROOT/'data/manifest.csv');image=open_rgb(ROOT/frame[frame.split=='test'].iloc[0].path);rows=[]
for backend in ['cpu']+(['mps'] if torch.backends.mps.is_available() else []):
 device=torch.device(backend)
 for run,architecture in [('cnn_baseline','cnn'),('mobilenet_improved','mobilenet'),('convnext_aug','convnext')]:
  model,_=load_checkpoint(ROOT/'outputs'/run/'best.pt',device);sample=transform(architecture=architecture)(image).unsqueeze(0).to(device)
  with torch.inference_mode():
   for _ in range(5):model(sample)
  values={'single_prediction':[],'previous_auto_benchmark_policy':[]}
  for repeat in range(3):
   for policy in (list(values) if repeat%2==0 else list(values)[::-1]):
    synchronize(device);start=time.perf_counter();prediction=predict_image(model,image,architecture,device)
    if policy=='previous_auto_benchmark_policy':timing=benchmark(model,sample,device)
    synchronize(device);values[policy].append((time.perf_counter()-start)*1000)
  rows.append({'run':run,'backend':backend,'single_prediction_median_ms':float(np.median(values['single_prediction'])),'previous_policy_median_ms':float(np.median(values['previous_auto_benchmark_policy'])),'measurements_ms':values,'optional_benchmark_forward_mean_ms':timing['mean_ms'],'environment':environment(device)})
  print(run,backend,rows[-1]['single_prediction_median_ms'],rows[-1]['previous_policy_median_ms'],flush=True)
write_json(ROOT/'outputs/technical_audit/inference_policy_benchmark.json',{'protocol':'Current code, loaded model, same first test photograph; five warm-ups; three paired timed repetitions with alternating order. Single path includes transforms, forward and softmax; previous policy replays that path plus five warm-ups and thirty additional forwards. Excludes model loading and browser overhead. Not a historical wall-time measurement.','results':rows})
