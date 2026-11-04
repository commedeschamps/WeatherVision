"""One real-data epoch per run on a tiny isolated subset; never submission metrics."""
import argparse
import logging
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd
from src.config import load_config
from src.data import validate_manifest
from src.engine import train
from src.evaluation import evaluate_run, compare_runs
from src.explainability import generate_examples
from src.utils import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto")
    p.add_argument("--random-weights", action="store_true", help="Offline structural smoke test; not transfer-learning evidence")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    directory = ROOT / "outputs/smoke"
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(ROOT / "data/manifest.csv")
    subset = frame.groupby(["label", "split"], group_keys=False).head(4)
    validate_manifest(subset)
    manifest = directory / "manifest.csv"
    subset.to_csv(manifest, index=False)
    configs = load_config(ROOT / "configs/experiments.yaml")
    for name, config in configs.items():
        config = replace(config, epochs=1, batch_size=4, device=args.device,
                         pretrained=config.pretrained and not args.random_weights)
        train(name, config, manifest, directory / name, resume=True if (directory / name / "status.json").exists() else False,
              skip_complete=True)
    for name in configs:
        evaluate_run(name, manifest, directory, device_name=args.device, warmup=1, repeats=3)
    compare_runs(list(configs), directory)
    generate_examples(directory / "convnext_aug", manifest, count=2, device_name=args.device)
    write_json(directory / "SMOKE_ONLY.json", {"purpose": "Pipeline validation, not a complete experiment",
               "images_per_class_per_split": 4, "epochs": 1, "pretrained": not args.random_weights})
    print("SMOKE TEST COMPLETE. Results are isolated under outputs/smoke and must not be cited as full experiments.")


if __name__ == "__main__":
    main()
