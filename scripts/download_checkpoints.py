"""Download released WeatherVision weights with SHA-256 validation (no training)."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download_asset(asset, destination):
    destination = Path(destination)
    if destination.is_file() and sha256(destination) == asset['sha256']:
        return 'verified existing'
    if destination.exists():
        raise ValueError(f'Existing file has a different checksum: {destination}. Move it aside before downloading.')
    if not asset['url'].startswith('https://github.com/commedeschamps/WeatherVision/releases/download/'):
        raise ValueError('Checkpoint URL must identify the WeatherVision release.')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = urllib.request.Request(asset['url'], headers={'User-Agent': 'WeatherVision-checkpoint-setup/1'})
        with urllib.request.urlopen(request, timeout=120) as response:
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.download', delete=False) as target:
                temporary = Path(target.name)
                total = 0
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > asset['bytes']:
                        raise ValueError('Checkpoint exceeds expected size')
                    target.write(block)
        if temporary.stat().st_size != asset['bytes'] or sha256(temporary) != asset['sha256']:
            raise ValueError('Downloaded checkpoint failed size/SHA-256 validation')
        os.replace(temporary, destination)
        return 'downloaded and verified'
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--siglip2', action='store_true', help='Install the pinned SigLIP 2 encoder and verify the repository-trained heads')
    parser.add_argument('--offline', action='store_true', help='Use only cached Hugging Face assets for SigLIP 2 installation')
    parser.add_argument('--all', action='store_true', help='Also download the non-augmented ablation model')
    parser.add_argument('--root', type=Path, default=ROOT, help='Project root to receive outputs/<run>/best.pt')
    args = parser.parse_args()
    if args.siglip2:
        if args.root.resolve() != ROOT.resolve():
            raise ValueError('SigLIP 2 installation uses the current repository root')
        from install_siglip2 import install_siglip_assets
        install_siglip_assets(args.offline)
        manifest = json.loads((ROOT/'configs/siglip2_artifacts.json').read_text())
        for item in manifest['heads']:
            path = ROOT/item['path']
            if not path.is_file() or sha256(path) != item['sha256']:
                raise ValueError('Missing or changed trained SigLIP 2 head: ' + item['path'])
        print('SigLIP 2 pretrained assets and trained heads verified')
        return
    assets = json.loads((ROOT/'configs/checkpoints.json').read_text())['assets']
    for asset in assets:
        if asset['run'] == 'convnext_noaug' and not args.all:
            continue
        print(asset['run'] + ': ' + download_asset(asset, args.root/'outputs'/asset['run']/'best.pt'), flush=True)

if __name__ == '__main__':
    main()
