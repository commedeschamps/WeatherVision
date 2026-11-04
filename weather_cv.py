"""WeatherVision command line.

Run `python weather_cv.py --help`; the app and notebook use these same modules.
"""
import argparse
import logging
from pathlib import Path

from src.config import load_config
from src.data import obtain_dataset, prepare_dataset, verify_content
from src.engine import train
from src.evaluation import compare_runs, evaluate_run, regenerate_figures
from src.explainability import generate_examples
from src.utils import ROOT, environment


def parser():
    p = argparse.ArgumentParser(description="WeatherVision: reproducible four-class scene classification")
    p.add_argument("--root", type=Path, default=ROOT, help="Project root (default: directory containing this script)")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Download, audit, deduplicate, and create one split")
    prep.add_argument("--local-zip", type=Path)
    prep.add_argument("--source-dir", type=Path)
    prep.add_argument("--data-dir", type=Path)
    prep.add_argument("--seed", type=int, default=42)
    prep.add_argument("--force", action="store_true", help="Rebuild manifest; existing checkpoints will reject changed splits")
    training = sub.add_parser("train", help="Train one named configuration; never reads test images")
    training.add_argument("--config", type=Path, default=ROOT / "configs/experiments.yaml")
    training.add_argument("--run", required=True)
    training.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.csv")
    training.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    training.add_argument("--resume", action="store_true")
    training.add_argument("--skip-complete", action="store_true")
    training.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"])
    evaluation = sub.add_parser("evaluate", help="Evaluate finalized configurations on the shared test split")
    evaluation.add_argument("--runs", nargs="+", default=["cnn_baseline", "mobilenet_improved", "convnext_aug", "convnext_noaug"])
    evaluation.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.csv")
    evaluation.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    evaluation.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    evaluation.add_argument("--warmup", type=int, default=5)
    evaluation.add_argument("--repeats", type=int, default=30)
    explain = sub.add_parser("explain", help="Save real ConvNeXt test-set Grad-CAM examples after evaluation")
    explain.add_argument("--run", default="convnext_aug")
    explain.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.csv")
    explain.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    explain.add_argument("--count", type=int, default=6)
    explain.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    plots = sub.add_parser("figures", help="Rebuild charts from stored CSV/JSON without training")
    plots.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    sub.add_parser("doctor", help="Print dependency versions, hardware, and checkpoint availability")
    return p


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser().parse_args()
    root = args.root.resolve()
    try:
        if args.command == "prepare":
            import pandas as pd
            directory = args.data_dir or root / "data"
            manifest = directory / "manifest.csv"
            if manifest.exists() and not args.force:
                import json
                statistics = json.loads((directory / "statistics.json").read_text())
                if statistics["seed"] != args.seed:
                    raise ValueError("Existing manifest uses a different seed; use a new data directory")
                frame = pd.read_csv(manifest)
                # Portable archives preserve the split while omitting raw images.
                if not all((root / value).is_file() for value in frame.path):
                    obtain_dataset(directory, args.local_zip, args.source_dir)
                verify_content(frame, root)
                print("Reusing verified manifest; --force explicitly rebuilds it.")
            else:
                source = obtain_dataset(directory, args.local_zip, args.source_dir)
                frame = prepare_dataset(source, directory, root, args.seed)
            print(pd.crosstab(frame.label, frame.split))
        elif args.command == "train":
            from dataclasses import replace
            configs = load_config(args.config)
            if args.run not in configs:
                raise ValueError(f"Unknown run {args.run}; choose one of {list(configs)}")
            config = configs[args.run]
            if args.device:
                config = replace(config, device=args.device)
            train(args.run, config, args.manifest, args.outputs / args.run, root, args.resume, args.skip_complete)
        elif args.command == "evaluate":
            # Validate every run before touching test data, preventing partial comparisons.
            import json
            for run in args.runs:
                status = json.loads((args.outputs / run / "status.json").read_text())
                if status["state"] != "complete":
                    raise ValueError(f"Finalize training first: {run}")
            if {"convnext_aug", "convnext_noaug"}.issubset(args.runs):
                from src.config import Experiment, validate_ablation
                validate_ablation(*[Experiment(**json.loads((args.outputs / run / "status.json").read_text())["config"])
                                    for run in ("convnext_aug", "convnext_noaug")])
            for run in args.runs:
                metrics = evaluate_run(run, args.manifest, args.outputs, root, args.device, args.warmup, args.repeats)
                print(f"{run}: test accuracy={metrics['accuracy']:.4f}, macro-F1={metrics['f1_macro']:.4f}")
            print(compare_runs(args.runs, args.outputs).to_string(index=False))
        elif args.command == "explain":
            print(generate_examples(args.outputs / args.run, args.manifest, root, args.count, args.device))
        elif args.command == "figures":
            regenerate_figures(args.outputs)
        elif args.command == "doctor":
            import json
            info = environment()
            info["checkpoints"] = {run: (root / "outputs" / run / "best.pt").is_file()
                                   for run in load_config(root / "configs/experiments.yaml")}
            print(json.dumps(info, indent=2))
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        raise SystemExit(f"WeatherVision: {e}") from e


if __name__ == "__main__":
    main()
