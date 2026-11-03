"""Validate both ConvNeXt stages on seeded correct examples and every saved error."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd,torch
from src.data import open_rgb,transform
from src.engine import load_checkpoint
from src.explainability import gradcam_convnext,overlay_image
from src.occlusion import occlusion_check
from src.utils import seed_everything,select_device,environment,sha256_file,write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--device',default='auto');args=parser.parse_args()
    seed_everything(42);device=select_device(args.device);directory=ROOT/'outputs/explainability_validation';directory.mkdir(exist_ok=True)
    predictions=pd.read_csv(ROOT/'outputs/convnext_aug/test_predictions.csv')
    rng=np.random.default_rng(42);chosen=[]
    for label in ['cloudy','rain','shine','sunrise']:
        candidates=predictions[(predictions.label==label)&(predictions.class_id==predictions.predicted_class_id)]
        chosen.extend(candidates.iloc[rng.choice(len(candidates),4,replace=False)].path.tolist())
    chosen.extend(predictions[predictions.class_id!=predictions.predicted_class_id].path)
    chosen.extend(pd.read_csv(ROOT/'outputs/convnext_aug/gradcam_examples.csv').path)
    chosen=list(dict.fromkeys(chosen));samples=predictions.set_index('path').loc[chosen].reset_index()
    samples.to_csv(directory/'samples.csv',index=False)
    model,_=load_checkpoint(ROOT/'outputs/convnext_aug/best.pt',device)
    before={name:p.detach().cpu().clone() for name,p in model.state_dict().items()};records=[];maps=[];start=time.perf_counter()
    for i,row in enumerate(samples.itertuples()):
        image=open_rgb(ROOT/row.path);x=transform(architecture='convnext')(image).unsqueeze(0).to(device)
        with torch.inference_mode(): assert int(model(x).argmax())==row.predicted_class_id
        for layer in ['final','earlier']:
            cam,details=gradcam_convnext(model,x,row.predicted_class_id,layer,True)
            native=details.pop('native_heatmap');name=f'{i:02d}_{Path(row.path).stem}_{layer}'
            np.savez_compressed(directory/f'{name}.npz',heatmap=cam,native=native)
            maps.append({'path':row.path,'correct':bool(row.correct),'label':row.label,'predicted':row.predicted_label,
                         'file':name+'.npz',**details,'peak_yx_224':list(map(int,np.unravel_index(cam.argmax(),cam.shape))),
                         'bottom_third_mass_fraction':float(cam[149:].sum()/cam.sum()) if cam.sum() else 0})
            for baseline in ['mean','blur']:
                result=occlusion_check(model,x,cam,row.predicted_class_id,baseline=baseline,seed=42+i)
                for r in result['rows']:
                    records.append({'path':row.path,'label':row.label,'predicted':row.predicted_label,'correct':bool(row.correct),
                                    'layer':layer,'baseline':baseline,'seed':42+i,**r})
        print(f'{i+1}/{len(samples)} {Path(row.path).name}',flush=True)
    assert all(torch.equal(p.detach().cpu(),before[name]) for name,p in model.state_dict().items())
    assert not model.training and all(p.grad is None for p in model.parameters())
    frame=pd.DataFrame(records);frame.to_json(directory/'scores.json',orient='records',indent=2);csv=frame.copy();csv['regions']=csv.regions.map(json.dumps);csv.to_csv(directory/'scores.csv',index=False)
    # Each image has equal weight; first average its 10 random controls.
    paired=frame.groupby(['path','layer','baseline','control'],sort=False).logit_drop.mean().unstack('control').reset_index()
    paired.to_csv(directory/'paired_scores.csv',index=False);summary=[]
    for (layer,baseline),group in paired.groupby(['layer','baseline']):
        summary.append({'layer':layer,'baseline':baseline,'samples':len(group),
                        **{control+'_mean_logit_drop':float(group[control].mean()) for control in ['high','low','random']},
                        'high_gt_low_count':int((group.high>group.low).sum()),'high_gt_random_count':int((group.high>group.random).sum()),
                        'high_minus_low_median':float((group.high-group.low).median())})
    pd.DataFrame(summary).to_csv(directory/'summary.csv',index=False)
    write_json(directory/'settings.json',{'checkpoint_sha256':sha256_file(ROOT/'outputs/convnext_aug/best.pt'),'manifest_sha256':sha256_file(ROOT/'data/manifest.csv'),
       'environment':environment(device),'selection':'Seed 42: four saved-correct test images per class, all three saved errors, plus all six existing Grad-CAM examples; duplicates removed. Enriched diagnostic sample, not a population estimate.',
       'sample_count':len(samples),'layers':{'final':'features[7]: 768x7x7','earlier':'features[5]: 384x14x14'},'occlusion':{'grid':[7,7],'cell_pixels':[32,32],'cells_removed':5,'fraction':5/49,'random_repeats':10,'seed':'42 + fixed sample row index','target':'original predicted class, held fixed after perturbation','mean_baseline':'normalized zeros = ImageNet RGB mean [0.485,0.456,0.406]','blur_baseline':'Gaussian blur of the same normalized input; kernel31, sigma10; replace only selected cells'},
       'map_settings':maps,'parameter_mutation':False,'seconds':time.perf_counter()-start})
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,baseline in zip(axes,['mean','blur']):
        for j,layer in enumerate(['final','earlier']):
            part=paired[(paired.baseline==baseline)&(paired.layer==layer)]
            for k,control in enumerate(['high','low','random']):
                x=j*4+k;values=part[control].to_numpy();ax.bar(x,values.mean(),color=['#21635D','#9E7560','#5B7395'][k],width=.7)
                ax.scatter(np.full(len(values),x)+np.linspace(-.2,.2,len(values)),values,s=11,c='#263D35',alpha=.65,zorder=3)
        ax.set(xticks=[0,1,2,4,5,6],xticklabels=['High','Low','Random']*2,ylabel='Original target logit − occluded logit',title=f'{baseline.title()} replacement; 5 of 49 equal cells')
        ax.axhline(0,c='#7A8F84',lw=.7);ax.text(.23,-.17,'Final stage',transform=ax.transAxes,ha='center');ax.text(.77,-.17,'Earlier stage',transform=ax.transAxes,ha='center')
    fig.savefig(directory/'occlusion_comparison.png',dpi=250,bbox_inches='tight');plt.close(fig)
    focus=samples[samples.path.str.endswith(('/cloud_101.jpeg','/cloud_108.jpeg','/cloud_112.jpeg','/cloud_267.jpeg','/rain_209.jpeg','/shine_1.jpeg'))]
    fig,axes=plt.subplots(len(focus),3,figsize=(10,3.1*len(focus)),layout='constrained')
    for axes_row,row in zip(axes,focus.itertuples()):
        im=open_rgb(ROOT/row.path);axes_row[0].imshow(im);axes_row[0].set_title(f'{Path(row.path).name}: {row.label} → {row.predicted_label}',fontsize=11)
        idx=chosen.index(row.path)
        for ax,layer in zip(axes_row[1:],['final','earlier']):
            cam=np.load(directory/f'{idx:02d}_{Path(row.path).stem}_{layer}.npz')['heatmap'];ax.imshow(overlay_image(im,cam,alpha=.65));ax.set_title('Final 7×7' if layer=='final' else 'Earlier 14×14 (experimental)')
        for ax in axes_row:ax.axis('off')
    fig.savefig(directory/'stage_comparison.png',dpi=220,bbox_inches='tight');plt.close(fig)
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
