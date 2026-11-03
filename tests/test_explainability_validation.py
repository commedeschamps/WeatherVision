import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision.transforms import functional as TF
from src.explainability import gradcam_convnext, overlay_image, map_heatmap_to_original
from src.models import build_model,set_training_mode
from src.occlusion import occlusion_check,region_selections

@pytest.fixture(scope='module')
def convnext():
    return build_model('convnext',False,True).eval()

@pytest.mark.parametrize('layer,shape',[('final',[1,768,7,7]),('earlier',[1,384,14,14])])
def test_gradcam_details_determinism_and_no_mutation(convnext,layer,shape):
    model=convnext;model.eval();x=torch.rand(1,3,224,224,requires_grad=True);x.grad=torch.ones_like(x)
    before={name:p.clone() for name,p in model.state_dict().items()};hooks=len(model.features[7 if layer=='final' else 5]._forward_hooks)
    cam,details=gradcam_convnext(model,x,2,layer,True)
    repeated=gradcam_convnext(model,x,2,layer)
    assert cam.shape==(224,224) and np.isfinite(cam).all() and 0<=cam.min()<=cam.max()<=1
    np.testing.assert_array_equal(cam,repeated)
    assert details['target_class']==2 and details['activation_shape']==details['gradient_shape']==shape
    assert len(model.features[7 if layer=='final' else 5]._forward_hooks)==hooks
    assert all(torch.equal(p,before[n]) for n,p in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters()) and torch.all(x.grad==1)


def test_gradcam_matches_direct_logit_gradient_formula(convnext):
    model=convnext.eval();captured=[];handle=model.features[-1].register_forward_hook(lambda m,i,o:captured.append(o))
    x=torch.rand(1,3,224,224,requires_grad=True)
    try:
        logits=model(x);g=torch.autograd.grad(logits[0,1],captured[0])[0]
        native=torch.relu((g.mean((2,3),keepdim=True)*captured[0]).sum(1,keepdim=True))
        expected=torch.nn.functional.interpolate(native,size=(224,224),mode='bilinear',align_corners=False)[0,0]
        expected=expected-expected.min()
        if expected.max()>0:expected=expected/expected.max()
        actual=gradcam_convnext(model,x,1)
        np.testing.assert_allclose(actual,expected.detach().numpy(),atol=1e-7,rtol=1e-6)
    finally:handle.remove()


def test_gradcam_preserves_mixed_modes_on_success_and_failure(convnext,monkeypatch):
    model=convnext;set_training_mode(model,'convnext',True);before=[m.training for m in model.modules()]
    gradcam_convnext(model,torch.rand(1,3,224,224),0)
    assert [m.training for m in model.modules()]==before
    hooks=len(model.features[-1]._forward_hooks)
    def fail(*args):raise RuntimeError('injected classifier failure')
    with monkeypatch.context() as patch:
        patch.setattr(model.classifier[2],'forward',fail)
        with pytest.raises(RuntimeError,match='injected'):gradcam_convnext(model,torch.zeros(1,3,224,224),0)
    assert [m.training for m in model.modules()]==before and len(model.features[-1]._forward_hooks)==hooks
    model.eval()


def test_zero_gradcam_and_outer_inference_mode(convnext):
    model=convnext.eval();handle=model.classifier.register_forward_hook(lambda m,i,o:o*0)
    try:
        with torch.inference_mode():
            cam,details=gradcam_convnext(model,torch.ones(1,3,224,224),0,return_details=True)
        assert np.count_nonzero(cam)==0 and details['spatially_constant']
    finally:handle.remove()

@pytest.mark.parametrize('target',[-1,4,.9,True])
def test_gradcam_rejects_invalid_targets_without_leaking_hooks(convnext,target):
    hooks=len(convnext.features[-1]._forward_hooks)
    with pytest.raises(ValueError,match='Target'):gradcam_convnext(convnext,torch.zeros(1,3,224,224),target)
    assert len(convnext.features[-1]._forward_hooks)==hooks


def test_gradcam_rejects_invalid_layer_and_nonfinite_input(convnext):
    with pytest.raises(ValueError,match='final or earlier'):gradcam_convnext(convnext,torch.zeros(1,3,224,224),0,'arbitrary')
    with pytest.raises(ValueError,match='finite'):gradcam_convnext(convnext,torch.full((1,3,224,224),float('nan')),0)

@pytest.mark.parametrize('size',[(224,224),(400,240),(237,401),(500,500)])
def test_overlay_preserves_exact_unseen_pixels_and_zero_opacity(size):
    original=Image.fromarray(np.random.default_rng(42).integers(0,256,(size[1],size[0],3),dtype=np.uint8));cam=np.ones((224,224),dtype=np.float32)
    w,h=TF.resize(original,236).size;l,t=round((w-224)/2),round((h-224)/2);y,x=np.mgrid[:size[1],:size[0]]
    visible=((x+.5)*w/size[0]>=l)&((x+.5)*w/size[0]<l+224)&((y+.5)*h/size[1]>=t)&((y+.5)*h/size[1]<t+224)
    before=np.asarray(original);after=np.asarray(overlay_image(original,cam,alpha=.8))
    assert np.array_equal(before[~visible],after[~visible])
    assert np.array_equal(before,np.asarray(overlay_image(original,cam,alpha=0)))
    assert np.array_equal(before,np.asarray(overlay_image(original,np.zeros_like(cam))))
    assert np.count_nonzero(map_heatmap_to_original(original,cam)[~visible])==0


def test_overlay_validation():
    image=Image.new('RGB',(224,224))
    for alpha in [-.1,1.1,float('nan')]:
        with pytest.raises(ValueError):overlay_image(image,np.zeros((224,224)),alpha=alpha)
    with pytest.raises(ValueError):overlay_image(image,np.full((224,224),float('nan')))

class CornerModel(nn.Module):
    def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(1.))
    def forward(self,x):
        score=x[:,0,:32,:32].mean((1,2))*self.scale
        return torch.stack([score,-score,torch.full_like(score,.5),score*0],dim=1)


def test_occlusion_equal_budget_deterministic_and_fixed_target(tmp_path):
    import json
    model=CornerModel().train();model.scale.grad=torch.tensor(7.);before=model.scale.clone()
    x=torch.ones(1,3,224,224);cam=np.zeros((224,224),dtype=np.float32);cam[:32,:32]=1
    a=occlusion_check(model,x,cam,0,random_repeats=3);b=occlusion_check(model,x,cam,0,random_repeats=3)
    assert a==b and model.training and torch.equal(before,model.scale) and model.scale.grad==7
    high,low=a['rows'][:2];assert high['logit_drop']==1 and low['logit_drop']==0
    assert high['occluded_predicted_class']==2 and high['target_class']==0
    assert {r['occluded_pixels'] for r in a['rows']}=={5*32*32}
    assert all(len(r['regions'])==len(set(r['regions']))==5 for r in a['rows'])
    path=tmp_path/'scores.json';path.write_text(json.dumps(a,allow_nan=False));assert json.loads(path.read_text())==a
    negative=occlusion_check(model,x,cam,1,random_repeats=1)
    assert negative['rows'][0]['logit_drop']==-1
    blur=occlusion_check(model,x,cam,0,baseline='blur',random_repeats=1)
    assert abs(blur['rows'][0]['logit_drop'])<1e-5


def test_occlusion_selection_validation_and_ties():
    cam=np.zeros((224,224));a=region_selections(cam);b=region_selections(cam)
    assert a[1]==b[1] and len(set(a[1][0][1])&set(a[1][1][1]))==0
    for kwargs in [{'grid':6},{'count':30},{'random_repeats':0}]:
        with pytest.raises(ValueError):region_selections(cam,**kwargs)
    with pytest.raises(ValueError):occlusion_check(CornerModel(),torch.ones(1,3,224,224),cam,4)

@pytest.mark.skipif(not torch.backends.mps.is_available(),reason='Apple MPS unavailable in this environment')
def test_trained_convnext_mps_gradcam_matches_cpu():
    from src.engine import load_checkpoint
    from src.data import open_rgb,transform
    from src.utils import ROOT
    import pandas as pd
    path=ROOT/'outputs/convnext_aug/best.pt'
    if not path.exists():pytest.skip('Released checkpoint required')
    model,_=load_checkpoint(path,'cpu');frame=pd.read_csv(ROOT/'data/manifest.csv');image=open_rgb(ROOT/frame[frame.split=='test'].iloc[0].path);x=transform(architecture='convnext')(image).unsqueeze(0)
    for layer in ['final','earlier']:
        model.cpu();cpu=gradcam_convnext(model,x,0,layer);model.to('mps');mps=gradcam_convnext(model,x.to('mps'),0,layer)
        np.testing.assert_allclose(cpu,mps,atol=.003,rtol=.01)
    assert all(p.grad is None for p in model.parameters())


def test_block_overlay_matches_full_array_rendering():
    import matplotlib.pyplot as plt
    rng=np.random.default_rng(42);image=Image.fromarray(rng.integers(0,256,(180,300,3),dtype=np.uint8));cam=rng.uniform(0,1,(224,224)).astype(np.float32)
    mapped=map_heatmap_to_original(image,cam);pixels=np.asarray(image);opacity=(.65*mapped)[...,None]
    expected=pixels.copy();blended=pixels*(1-opacity)+plt.get_cmap('inferno')(mapped)[...,:3]*255*opacity;active=opacity[...,0]>0
    expected[active]=np.rint(blended[active]).clip(0,255).astype(np.uint8)
    np.testing.assert_array_equal(np.asarray(overlay_image(image,cam,alpha=.65)),expected)
