import io,json
import numpy as np
import pytest,torch
from PIL import Image
from streamlit.testing.v1 import AppTest
from src.app_utils import decode_upload,export_prediction
from src.utils import ROOT
from src.engine import load_checkpoint,predict_image
from src.models import build_model


def button(app,label):return next(b for b in app.button if b.label==label)
def checkbox(app,label):return next(c for c in app.checkbox if c.label==label)


def test_opacity_stage_and_benchmark_preserve_valid_predictions(monkeypatch):
    if not (ROOT/'outputs/convnext_aug/best.pt').exists():pytest.skip('Released model required')
    calls=[]
    import src.evaluation as evaluation
    real=evaluation.benchmark
    def counted(*args,**kwargs):calls.append(1);return real(*args,**kwargs)
    monkeypatch.setattr(evaluation,'benchmark',counted)
    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=40)
    if not app.session_state['active_image']:pytest.skip('Test examples required')
    next(s for s in app.selectbox if s.key=='selected_model').select('ConvNeXt-Tiny').run(timeout=40)
    button(app,'Classify image').click().run(timeout=40)
    assert not app.exception and not calls
    key=next(iter(app.session_state['image_predictions']));prediction=dict(app.session_state['image_predictions'][key]);assert 'benchmark' not in prediction
    checkbox(app,'Show Grad-CAM').check().run(timeout=40)
    cam=app.session_state['image_heatmaps'][key]['final']['heatmap'].copy()
    next(s for s in app.slider if s.key=='cam_opacity').set_value(.8).run(timeout=40)
    checkbox(app,'Show raw heatmap').check().run(timeout=40)
    assert app.session_state['image_predictions'][key]==prediction and not calls
    np.testing.assert_array_equal(cam,app.session_state['image_heatmaps'][key]['final']['heatmap'])
    next(s for s in app.radio if s.key=='cam_stage').set_value('Earlier stage · 14 × 14 (experimental)').run(timeout=40)
    assert set(app.session_state['image_heatmaps'][key])=={'final','earlier'} and not calls
    assert app.session_state['image_predictions'][key]==prediction
    button(app,'Measure average latency').click().run(timeout=40)
    assert len(calls)==1 and app.session_state['image_predictions'][key]['benchmark']['repeats']==30
    benchmark=app.session_state['image_predictions'][key]['benchmark']
    next(s for s in app.selectbox if s.key=='selected_model').select('ConvNeXt-Tiny').run(timeout=40)
    button(app,'Classify image').click().run(timeout=40)
    assert len(calls)==1 and app.session_state['image_predictions'][key]['benchmark']==benchmark
    next(s for s in app.selectbox if s.key=='selected_model').select('MobileNetV2').run(timeout=40)
    assert not app.exception and not any(c.label=='Show Grad-CAM' for c in app.checkbox)
    next(s for s in app.selectbox if s.key=='selected_model').select('ConvNeXt-Tiny').run(timeout=40)
    assert app.session_state['image_predictions'][key]['class_id']==prediction['class_id']


def test_upload_rejects_large_pixels_and_disguised_unsupported_format():
    out=io.BytesIO();Image.new('1',(5001,5001)).save(out,format='PNG')
    with pytest.raises(ValueError,match='megapixels'):decode_upload(out.getvalue())
    out=io.BytesIO();Image.new('RGB',(10,10)).save(out,format='GIF')
    with pytest.raises(ValueError,match='JPG, PNG, or WebP'):decode_upload(out.getvalue())


def test_corrupt_checkpoint_and_train_mode_inference_are_rejected(tmp_path):
    path=tmp_path/'bad.pt';path.write_bytes(b'invalid checkpoint')
    with pytest.raises(ValueError,match='decode checkpoint'):load_checkpoint(path)
    model=build_model('cnn',False,False)
    with pytest.raises(ValueError,match='model.eval'):predict_image(model,Image.new('RGB',(224,224)),'cnn',torch.device('cpu'))


def test_prediction_export_without_optional_benchmark():
    prediction={'label':'cloudy','class_id':0,'confidence':.7,'probabilities':[.7,.1,.1,.1],'latency_ms':2.,'server_prediction_ms':10.,'environment':{'device':'cpu'},'browser_user_agent':'test fixture'}
    data=json.loads(export_prediction(prediction,'Simple CNN','cpu','input.png',(224,224)))
    assert 'benchmark' not in data and data['environment']['device']=='cpu'
    assert data['server_prediction_ms']==10. and data['browser_user_agent']=='test fixture'


def test_corrupt_checkpoint_is_recoverable_in_streamlit(tmp_path,monkeypatch):
    import pandas as pd
    import src.utils as utils
    project=utils.ROOT;monkeypatch.setattr(utils,'ROOT',tmp_path)
    checkpoint=tmp_path/'outputs/convnext_aug/best.pt';checkpoint.parent.mkdir(parents=True);checkpoint.write_bytes(b'invalid checkpoint')
    image=tmp_path/'data/example.png';image.parent.mkdir(parents=True);Image.new('RGB',(224,224),(100,130,170)).save(image)
    pd.DataFrame([{'path':'data/example.png','label':'cloudy','split':'test'}]).to_csv(tmp_path/'data/manifest.csv',index=False)
    app=AppTest.from_file(str(project/'app.py')).run(timeout=40)
    next(s for s in app.selectbox if s.key=='selected_model').select('ConvNeXt-Tiny').run(timeout=40)
    button(app,'Classify image').click().run(timeout=40)
    assert not app.exception and any('Could not classify' in e.value for e in app.error)
    assert app.session_state['image_predictions']=={}


def test_extreme_aspect_ratio_is_rejected_before_resize():
    from src.explainability import overlay_image
    image=Image.new('RGB',(3000,1),(100,140,180));buffer=io.BytesIO();image.save(buffer,format='PNG')
    with pytest.raises(ValueError,match='aspect ratio'):decode_upload(buffer.getvalue())
    with pytest.raises(ValueError,match='aspect ratio'):predict_image(build_model('cnn',False,False).eval(),image,'cnn',torch.device('cpu'))
    with pytest.raises(ValueError,match='aspect ratio'):overlay_image(image,np.ones((224,224),dtype=np.float32))
