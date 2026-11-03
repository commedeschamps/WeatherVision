"""Optional hosting entrypoint: verify released weights, then start the app."""
import runpy
from pathlib import Path
import streamlit as st

@st.cache_resource(show_spinner="Preparing WeatherVision models…")
def prepare_models():
    import json
    from scripts.download_checkpoints import download_asset, ROOT
    for asset in json.loads((ROOT/'configs/checkpoints.json').read_text())['assets']:
        if asset['run'] != 'convnext_noaug':
            download_asset(asset, ROOT/'outputs'/asset['run']/'best.pt')
    return True

prepare_models()
runpy.run_path(str(Path(__file__).with_name('app.py')), run_name='__main__')
