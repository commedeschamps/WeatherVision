"""Build source or results archives without raw data, caches or secrets."""

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORIES = [
    "src",
    "scripts",
    "configs",
    "tests",
    "notebooks",
    "docs",
    "reports",
    ".streamlit",
]
SOURCE_FILES = [
    "deploy_app.py",
    "app.py",
    "weather_cv.py",
    "README.md",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-lock-local.txt",
    "requirements-ui.txt",
    ".gitignore",
    "pytest.ini",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        action="store_true",
        help="Include full best/last checkpoints and actual result artifacts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination archive (default: dist/WeatherVision_source.zip or WeatherVision_results.zip)",
    )
    args = parser.parse_args()
    target = args.output or ROOT / "dist" / (
        "WeatherVision_results.zip" if args.results else "WeatherVision_source.zip"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    small_heads = {
        item["path"]
        for item in json.loads((ROOT / "configs/siglip2_artifacts.json").read_text())[
            "heads"
        ]
    }
    paths = [ROOT / name for name in SOURCE_FILES + [".gitattributes"]]
    for name in SOURCE_DIRECTORIES + ["data"] + ["outputs"]:
        paths.extend((ROOT / name).rglob("*"))
    with zipfile.ZipFile(
        target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=4
    ) as archive:
        for path in sorted(set(paths)):
            if not path.is_file():
                continue
            parts = path.relative_to(ROOT).parts
            if any(
                part
                in {
                    "raw",
                    "downloads",
                    "__pycache__",
                    ".pytest_cache",
                    ".cache",
                    ".ipynb_checkpoints",
                    ".git",
                    "tmp",
                    "dist",
                    "smoke",
                }
                for part in parts
            ):
                continue
            if (
                path.name == ".DS_Store"
                or path.name.startswith(".env")
                or parts == (".streamlit", "secrets.toml")
                or path.suffix == ".log"
            ):
                continue
            if (
                not args.results
                and path.suffix in {".pt", ".pth", ".mp4", ".webm"}
                and path.relative_to(ROOT).as_posix() not in small_heads
            ):
                continue
            if path.name.endswith("_original.ipynb"):
                continue
            archive.write(path, path.relative_to(ROOT))
    print(target, f"({target.stat().st_size / 1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
