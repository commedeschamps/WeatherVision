"""Optional hosting entrypoint: install verified historical and SigLIP 2 assets."""
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
    from scripts.install_siglip2 import install_siglip_assets
    from src.siglip2 import ASSET_DIR
    from src.utils import sha256_file
    if not (ASSET_DIR / 'provenance.json').is_file():
        install_siglip_assets()
    for item in json.loads((ROOT/'configs/siglip2_artifacts.json').read_text())['heads']:
        path = ROOT/item['path']
        if not path.is_file() or sha256_file(path) != item['sha256']:
            raise ValueError('Missing or changed repository-trained SigLIP 2 head: ' + item['path'])
    return True

prepare_models()
runpy.run_path(str(Path(__file__).with_name('app.py')), run_name='__main__')
