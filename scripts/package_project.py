"""Build portable, auditable source/results archives without private or cached files."""
import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = ['src', 'scripts', 'configs', 'tests', 'notebooks', 'docs', 'reports', '.streamlit']
SOURCE_FILES = ['deploy_app.py', 'app.py', 'weather_cv.py', 'README.md', 'requirements.txt', 'requirements-dev.txt',
                'requirements-lock-local.txt', '.gitignore', 'pytest.ini', 'PRODUCT.md', 'DESIGN.md',
                'requirements-ui.txt',
                'REPORT_OUTLINE.md', 'DEFENSE_GUIDE.md']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', action='store_true', help='Include full best/last checkpoints and actual result artifacts')
    args = parser.parse_args()
    target = ROOT / 'dist' / ('WeatherVision_results.zip' if args.results else 'WeatherVision_source.zip')
    target.parent.mkdir(parents=True, exist_ok=True)
    paths = [ROOT / name for name in SOURCE_FILES]
    for name in SOURCE_DIRECTORIES + ['data'] + ['outputs']:
        paths.extend((ROOT / name).rglob('*'))
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=4) as archive:
        for path in sorted(set(paths)):
            if not path.is_file():
                continue
            parts = path.relative_to(ROOT).parts
            if any(part in {'raw', 'downloads', '__pycache__', '.pytest_cache', 'smoke'} for part in parts):
                continue
            if not args.results and path.suffix in {'.pt', '.pth', '.mp4', '.webm'}:
                continue
            if path.name.endswith('_original.ipynb'):
                continue
            archive.write(path, path.relative_to(ROOT))
    print(target, f'({target.stat().st_size / 1024**2:.1f} MB)')


if __name__ == '__main__':
    main()
