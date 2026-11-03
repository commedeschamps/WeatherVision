def test_missing_checkpoints_are_graceful(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    import src.utils as utils
    project = utils.ROOT
    monkeypatch.setattr(utils, 'ROOT', tmp_path)
    app = AppTest.from_file(str(project / 'app.py')).run(timeout=30)
    assert not list(app.exception)
    assert app.title[0].value == 'WeatherVision'
    assert any('no trained checkpoint' in w.value for w in app.warning)
    assert next(b for b in app.button if b.label == 'Classify image').disabled


def test_explanation_toggle_preserves_prediction():
    from streamlit.testing.v1 import AppTest
    from src.utils import ROOT
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=40)
    if not (ROOT / 'outputs/convnext_aug/best.pt').exists() or not app.session_state['active_image']:
        import pytest
        pytest.skip('Requires a trained checkpoint and dataset examples')
    next(b for b in app.button if b.label == 'Classify image').click().run(timeout=40)
    assert not list(app.exception)
    before = dict(app.session_state['image_predictions'])
    app.checkbox[0].check().run(timeout=40)
    assert not list(app.exception)
    assert app.session_state['image_predictions'] == before
    assert len(app.session_state['image_heatmaps']) == 1
    app.checkbox[0].uncheck().run(timeout=40)
    assert app.session_state['image_predictions'] == before


def test_clear_image_resets_results():
    from streamlit.testing.v1 import AppTest
    from src.utils import ROOT
    app = AppTest.from_file(str(ROOT / 'app.py')).run(timeout=40)
    if not app.session_state['active_image']:
        import pytest
        pytest.skip('Requires dataset examples')
    next(b for b in app.button if b.label == 'Clear image').click().run(timeout=40)
    assert not list(app.exception)
    assert app.session_state['active_image'] is None
    assert app.session_state['image_predictions'] == {}
    assert next(b for b in app.button if b.label == 'Classify image').disabled
