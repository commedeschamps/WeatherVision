import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data import prepare_dataset
from src.utils import CLASSES, seed_everything


@pytest.fixture(autouse=True)
def reproducible():
    seed_everything(42)
    torch.set_num_threads(2)


@pytest.fixture
def synthetic_data(tmp_path):
    source = tmp_path / "data/raw"
    rng = np.random.default_rng(42)
    for label in CLASSES:
        folder = source / label
        folder.mkdir(parents=True)
        for i in range(8):
            Image.fromarray(rng.integers(0, 255, (32, 48, 3), dtype=np.uint8)).save(folder / f"{label}_{i}.png")
    frame = prepare_dataset(source, tmp_path / "data", root=tmp_path)
    return tmp_path, frame
