import hashlib
import io
import pytest
from scripts.download_checkpoints import download_asset

URL='https://github.com/commedeschamps/WeatherVision/releases/download/v1.0.0/test.pt'

def asset(data):
    return {'url':URL,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}

def test_download_validates_and_reuses(tmp_path,monkeypatch):
    data=b'checkpoint download test fixture'
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:io.BytesIO(data))
    target=tmp_path/'best.pt'
    assert download_asset(asset(data),target)=='downloaded and verified'
    assert target.read_bytes()==data
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:pytest.fail('Existing correct file must not download'))
    assert download_asset(asset(data),target)=='verified existing'

def test_mismatch_preserves_existing_file(tmp_path):
    target=tmp_path/'best.pt';target.write_bytes(b'original')
    with pytest.raises(ValueError,match='different checksum'):
        download_asset(asset(b'new weights'),target)
    assert target.read_bytes()==b'original'

def test_corrupt_download_is_not_installed(tmp_path,monkeypatch):
    monkeypatch.setattr('urllib.request.urlopen',lambda *a,**kw:io.BytesIO(b'bad'))
    with pytest.raises(ValueError,match='validation'):
        download_asset(asset(b'good'),tmp_path/'best.pt')
    assert list(tmp_path.iterdir())==[]
