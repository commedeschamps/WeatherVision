"""Run the locked SigLIP 2 zero-shot and three-seed supervised protocol."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.siglip_experiments import run_siglip_experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, default=ROOT / "outputs")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cpu", "mps", "cuda"]
    )
    args = parser.parse_args()
    run_siglip_experiments(device_name=args.device, output_root=args.outputs)
