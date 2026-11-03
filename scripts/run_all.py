"""Run the registered protocol in order. Completed runs are safely skipped."""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import load_config, validate_ablation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--local-zip", type=Path)
    parser.add_argument("--stages", nargs="+", choices=["prepare", "train", "evaluate", "explain", "report"],
                        default=["prepare", "train", "evaluate", "explain", "report"])
    args = parser.parse_args()
    configs = load_config(args.config)
    validate_ablation(configs["convnext_aug"], configs["convnext_noaug"])

    def run(*arguments):
        subprocess.run([sys.executable, str(ROOT / "weather_cv.py"), *map(str, arguments)], cwd=ROOT, check=True)

    if "prepare" in args.stages:
        run("prepare", *(["--local-zip", args.local_zip] if args.local_zip else []))
    if "train" in args.stages:
        for name in configs:
            continuing = args.resume and (ROOT / "outputs" / name / "status.json").exists()
            run("train", "--config", args.config, "--run", name, "--skip-complete", *(["--resume"] if continuing else []))
    if "evaluate" in args.stages:
        run("evaluate", "--runs", *configs)
    if "explain" in args.stages:
        run("explain")
    if "report" in args.stages:
        subprocess.run([sys.executable, str(ROOT / "scripts/build_reports.py")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
