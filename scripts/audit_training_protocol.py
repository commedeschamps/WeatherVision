"""Read-only reconstruction of validation/test results, data integrity and ablation."""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from src.data import WeatherDataset,open_rgb,transform,verify_content,resize_size
from src.config import load_config,validate_ablation
from src.engine import load_checkpoint,calculate_metrics
from src.utils import seed_everything,sha256_file,write_json


def bootstrap_macro(frame,seed=42,repeats=2000):
    # Resample whole detected groups within each class, not individual duplicates.
    rng=np.random.default_rng(seed);labels=[];predictions=[]
    groups=[]
    for _,sub in frame.groupby('class_id',sort=True):
        groups.append([g[['class_id','predicted_class_id']].to_numpy() for _,g in sub.groupby('group',sort=True)])
    result=[]
    for _ in range(repeats):
        values=np.concatenate([np.concatenate([g[i] for i in rng.integers(0,len(g),len(g))]) for g in groups])
        matrix=np.bincount(values[:,0]*4+values[:,1],minlength=16).reshape(4,4)
        denominator=matrix.sum(0)+matrix.sum(1);f1=np.divide(2*np.diag(matrix),denominator,out=np.zeros(4),where=denominator!=0)
        result.append(float(f1.mean()))
    return np.array(result)


def main():
    seed_everything(42);out=ROOT/'outputs/technical_audit';out.mkdir(exist_ok=True);(out/'validation').mkdir(exist_ok=True)
    frame=pd.read_csv(ROOT/'data/manifest.csv');verify_content(frame)
    for row in frame.itertuples():
        image=open_rgb(ROOT/row.path);pixel=hashlib.sha256(f'{image.width}x{image.height}:'.encode()+image.tobytes()).hexdigest();assert pixel==row.pixel_sha256
    configs=load_config(ROOT/'configs/experiments.yaml');validate_ablation(configs['convnext_aug'],configs['convnext_noaug'])
    baseline=json.loads((out/'baseline.json').read_text());results={};all_class_rows=[];frozen=[]
    for run,config in configs.items():
        directory=ROOT/'outputs'/run;model,checkpoint=load_checkpoint(directory/'best.pt','cpu');assert not any(m.training for m in model.modules())
        assert checkpoint['manifest_sha256']==sha256_file(ROOT/'data/manifest.csv')
        assert checkpoint['config']==config.to_dict()
        selected=pd.read_csv(directory/'history.csv').sort_values('epoch').sort_values('val_f1_macro',ascending=False,kind='stable').iloc[0]
        assert int(selected.epoch)==checkpoint['best_epoch']
        split_frames={};split_metrics={}
        for split in ['val','test']:
            data=WeatherDataset(frame,ROOT,split,False,config.model);scores=[]
            with torch.inference_mode():
                for x,_ in DataLoader(data,batch_size=8,shuffle=False):scores.extend(model(x).softmax(1).tolist())
            probabilities=np.asarray(scores);predicted=probabilities.argmax(1);metrics=calculate_metrics(data.frame.class_id.tolist(),predicted.tolist())
            current=data.frame[['path','label','class_id','group']].copy();current['predicted_class_id']=predicted;current['correct']=current.class_id==predicted;current['confidence']=probabilities.max(1);current['entropy']=-(probabilities*np.log(np.maximum(probabilities,1e-12))).sum(1)
            for j,label in enumerate(['cloudy','rain','shine','sunrise']):current['score_'+label]=probabilities[:,j]
            if split=='val':
                current.to_csv(out/'validation'/f'{run}_predictions.csv',index=False)
                assert abs(metrics['f1_macro']-float(selected.val_f1_macro))<1e-10
            else:
                saved=pd.read_csv(directory/'test_predictions.csv');assert saved.path.tolist()==current.path.tolist();assert saved.class_id.tolist()==current.class_id.tolist();assert saved.predicted_class_id.tolist()==predicted.tolist()
                official=json.loads((directory/'test_metrics.json').read_text())
                assert official['manifest_sha256']==checkpoint['manifest_sha256']
                assert official['checkpoint_sha256']==sha256_file(directory/'best.pt')
                assert official['test_membership_sha256']==hashlib.sha256('\n'.join(data.frame.path).encode()).hexdigest()
                for key in ['accuracy','precision_macro','recall_macro','f1_macro','confusion_matrix']:assert np.allclose(metrics[key],official[key])
            split_frames[split]=current;split_metrics[split]=metrics
            for label,data_row in metrics['per_class'].items():
                if label in ['cloudy','rain','shine','sunrise']:
                    all_class_rows.append({'run':run,'split':split,'class':label,'support':int(data_row['support']),'f1':data_row['f1-score'],'recall':data_row['recall'],'precision':data_row['precision']})
        boot_val=bootstrap_macro(split_frames['val'],42);boot_test=bootstrap_macro(split_frames['test'],43);gap=boot_test-boot_val
        results[run]={'best_epoch':int(selected.epoch),'validation':split_metrics['val'],'test':split_metrics['test'],
          'val_correct':int(split_frames['val'].correct.sum()),'test_correct':int(split_frames['test'].correct.sum()),
          'mean_confidence':{s:float(f.confidence.mean()) for s,f in split_frames.items()},'mean_entropy':{s:float(f.entropy.mean()) for s,f in split_frames.items()},
          'macro_f1_gap_test_minus_validation':split_metrics['test']['f1_macro']-split_metrics['val']['f1_macro'],
          'descriptive_group_bootstrap_95':{'validation':np.quantile(boot_val,[.025,.975]).tolist(),'test':np.quantile(boot_test,[.025,.975]).tolist(),'gap':np.quantile(gap,[.025,.975]).tolist(),'repeats':2000,'method':'Whole detected groups resampled within each class in each split; independent split resamples. Descriptive, conditional on this dataset; not a causal or significance claim.'},
          'initial_state_keys_present':[k for k in checkpoint if 'initial' in k],
          'historical_source_sha_matches_current':{name:sha==sha256_file(ROOT/'src'/name) for name,sha in checkpoint['metadata']['source_sha256'].items() if (ROOT/'src'/name).exists()}}
        print(run,'val/test F1',split_metrics['val']['f1_macro'],split_metrics['test']['f1_macro'],flush=True)
        if run.startswith('convnext'):
            frozen.append({k:v.clone() for k,v in checkpoint['model_state'].items() if not k.startswith('classifier.2.')})
    difficulty=[]
    for split in ['val','test']:
        combined=None
        for run in configs:
            source=out/'validation'/f'{run}_predictions.csv' if split=='val' else ROOT/'outputs'/run/'test_predictions.csv'
            f=pd.read_csv(source);f['correct_'+run]=f.class_id==f.predicted_class_id;f=f[['path','correct_'+run]]
            combined=f if combined is None else combined.merge(f,on='path',validate='one_to_one')
        combined['models_correct']=combined.filter(like='correct_').sum(axis=1);combined['split']=split;difficulty.append(combined)
    pd.concat(difficulty).to_csv(out/'sample_difficulty.csv',index=False)
    assert all(torch.equal(frozen[0][k],frozen[1][k]) for k in frozen[0])
    pd.DataFrame(all_class_rows).to_csv(out/'validation/class_metrics.csv',index=False)
    augmented=pd.read_csv(ROOT/'outputs/convnext_aug/test_predictions.csv');plain=pd.read_csv(ROOT/'outputs/convnext_noaug/test_predictions.csv')
    changed=augmented[augmented.predicted_class_id!=plain.predicted_class_id].copy();changed['noaug_predicted_label']=plain.loc[changed.index,'predicted_label'];changed['noaug_correct']=plain.loc[changed.index,'correct'];changed.to_csv(out/'ablation_disagreements.csv',index=False)
    # Replay the archived model-construction procedure in the recorded environment.
    from src.models import build_model
    __import__('os').environ['TORCH_HOME']=str(ROOT/'.cache/torch')
    initial_heads=[];initial_backbone_matches=[]
    for run in ['convnext_aug','convnext_noaug']:
        seed_everything(configs[run].seed);m=build_model('convnext',True,True)
        initial_heads.append({k:v.detach().clone() for k,v in m.classifier[2].state_dict().items()})
        initial_backbone_matches.append(all(torch.equal(v,frozen[0][k]) for k,v in m.state_dict().items() if not k.startswith('classifier.2.')))
    assert all(torch.equal(initial_heads[0][k],initial_heads[1][k]) for k in initial_heads[0])
    head_hash=hashlib.sha256(b''.join(initial_heads[0][k].numpy().tobytes() for k in sorted(initial_heads[0]))).hexdigest()
    write_json(out/'protocol_analysis.json',{'reconstruction_backend':'cpu','dataset':{'images':len(frame),'counts':pd.crosstab(frame.label,frame.split).to_dict(),'all_file_and_decoded_pixel_hashes_verified':True,'exact_and_pixel_hash_split_collisions':0,'detected_group_split_collisions':0},
       'results':results,'ablation':{'disagreements':len(changed),'aug_correct':int(augmented.correct.sum()),'noaug_correct':int(plain.correct.sum()),'frozen_backbones_byte_equal':True,'backbone_matches_official_v1_in_replay':initial_backbone_matches,
       'historical_initial_head_state_saved':False,'deterministic_replay_initial_heads_identical':True,'replayed_initial_head_sha256':head_hash,'interpretation':'The archived code seeds construction before augmentation and reproduces equal initial heads. Historical head tensors/hashes were not saved, so replay is protocol evidence, not direct evidence of the past initial state.'}})
    # Controlled rotation evidence; no training transform is changed.
    mask=np.asarray(TF.rotate(Image.new('L',(224,224),255),8,fill=0));black_fraction=float((mask==0).mean())
    fig,axes=plt.subplots(4,4,figsize=(12,10),layout='constrained');rotation_rows=[]
    for axes_row,label in zip(axes,['cloudy','rain','shine','sunrise']):
        row=frame[(frame.label==label)&(frame.split=='train')].iloc[0];image=open_rgb(ROOT/row.path);crop=TF.center_crop(TF.resize(image,236),224)
        seed_everything(42);aug=transform('train',True,'convnext')(image);mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None];aug_image=TF.to_pil_image((aug*std+mean).clamp(0,1))
        for ax,im,title in zip(axes_row,[crop,TF.rotate(crop,8,fill=0),TF.rotate(crop,-8,fill=0),aug_image],[label.title()+' / center crop','Rotation +8° / black fill','Rotation −8° / black fill','Full training view / seed42']):ax.imshow(im);ax.set_title(title,fontsize=10);ax.axis('off')
        rotation_rows.append({'path':row.path,'class':label})
    fig.savefig(out/'rotation_examples.png',dpi=220,bbox_inches='tight');plt.close(fig)
    write_json(out/'rotation_analysis.json',{'rotation_degrees':[-8,8],'interpolation':'NEAREST (TorchVision RandomRotation default)','fill_rgb':[0,0,0],'fraction_black_fill_at_8_degrees':black_fraction,'input_size':[224,224],'examples':rotation_rows,'training_recipe_changed':False,'future_experiment':'Compare reflection padding followed by rotation/crop, or alternative fill, under fresh training; current artifacts retain the original augmentation.'})
    unchanged=[]
    for name,info in baseline['files'].items():
        assert sha256_file(ROOT/name)==info['sha256'],name;unchanged.append(name)
    write_json(out/'preservation_check.json',{'status':'passed','files_unchanged':len(unchanged),'baseline_commit':baseline['commit'],'protected_paths':unchanged})

if __name__=='__main__':main()
