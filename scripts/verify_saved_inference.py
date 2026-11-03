"""Reconstruct all held-out predictions on CPU without changing experiment artifacts."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
from src.engine import load_checkpoint,calculate_metrics
from src.data import WeatherDataset
from src.utils import seed_everything,write_json

def main():
    seed_everything(42);frame=pd.read_csv(ROOT/'data/manifest.csv');result={}
    for run in ['cnn_baseline','mobilenet_improved','convnext_aug','convnext_noaug']:
        directory=ROOT/'outputs'/run;model,checkpoint=load_checkpoint(directory/'best.pt','cpu')
        dataset=WeatherDataset(frame,ROOT,'test',False,checkpoint['config']['model'])
        predicted=[];targets=[]
        with torch.inference_mode():
            for x,y in DataLoader(dataset,batch_size=8,shuffle=False,num_workers=0):
                predicted.extend(model(x).argmax(1).tolist());targets.extend(y.tolist())
        saved=pd.read_csv(directory/'test_predictions.csv');metrics=json.loads((directory/'test_metrics.json').read_text())
        assert saved.path.tolist()==dataset.frame.path.tolist()
        assert predicted==saved.predicted_class_id.tolist(),f'CPU prediction mismatch: {run}'
        recalculated=calculate_metrics(targets,predicted)
        for key in ['accuracy','precision_macro','recall_macro','f1_macro','confusion_matrix']:
            assert np.allclose(recalculated[key],metrics[key]),(run,key)
        result[run]={'device':'cpu','images':len(predicted),'predictions_equal_saved':True,'correct':sum(a==b for a,b in zip(targets,predicted)),'accuracy':recalculated['accuracy'],'macro_f1':recalculated['f1_macro']}
        print(run,result[run],flush=True)
    write_json(ROOT/'outputs/full_cpu_reconstruction.json',{'status':'passed','models':result})

if __name__=='__main__':main()
