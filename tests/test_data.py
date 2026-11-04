import shutil
import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data import (assign_splits, infer_label, prepare_dataset, safe_extract,
                      transform, validate_manifest, verify_content)
from src.utils import CLASSES, CLASS_TO_IDX, LABELS


def test_class_mapping():
    assert CLASSES == ("cloudy", "rain", "shine", "sunrise")
    assert CLASS_TO_IDX == {"cloudy": 0, "rain": 1, "shine": 2, "sunrise": 3}
    assert LABELS == ("Cloudy", "Rain", "Shine", "Sunrise")


def test_label_inference(tmp_path):
    assert infer_label(tmp_path / "archive/cloud/a.jpg", tmp_path) == "cloudy"
    assert infer_label(tmp_path / "MWD/cloudy123.jpg", tmp_path) == "cloudy"
    assert infer_label(tmp_path / "pl_logo.png", tmp_path) is None
    with pytest.raises(ValueError):
        infer_label(tmp_path / "cloud/rain/a.jpg", tmp_path)


def test_split_reproducible_and_grouped(synthetic_data):
    root, frame = synthetic_data
    repeated = pd.concat([frame, frame.iloc[[0]].assign(path="copy.png")], ignore_index=True)
    a = assign_splits(repeated, 42)
    b = assign_splits(repeated, 42)
    assert a.split.equals(b.split)
    assert a.groupby("group").split.nunique().max() == 1
    assert not set(a[a.split == "train"].group) & set(a[a.split == "test"].group)
    assert pd.crosstab(a.label, a.split).min().min() > 0
    assert not assign_splits(repeated, 43).split.equals(a.split)


def test_leakage_rejected(synthetic_data):
    _, frame = synthetic_data
    copied = frame.iloc[[0]].copy()
    copied["split"] = "test" if copied.iloc[0].split != "test" else "train"
    copied["path"] = "different_path.png"
    with pytest.raises(ValueError, match="Leakage"):
        validate_manifest(pd.concat([frame, copied]), check_files=False)


def test_content_integrity(synthetic_data):
    root, frame = synthetic_data
    verify_content(frame, root)
    (root / frame.iloc[0].path).write_bytes(b"modified")
    with pytest.raises(ValueError, match="content changed"):
        verify_content(frame, root)


def test_dedup_and_corrupt(synthetic_data):
    root, frame = synthetic_data
    first = root / frame.iloc[0].path
    shutil.copy(first, first.parent / "copy.png")
    # One real near duplicate: change one pixel without changing aspect ratio.
    image = np.asarray(Image.open(first)).copy()
    image[0, 0] ^= 1
    Image.fromarray(image).save(first.parent / "near.png")
    (first.parent / "corrupt.jpg").write_bytes(b"not an image")
    result = prepare_dataset(root / "data/raw", root / "data", root)
    audit = pd.read_csv(root / "data/image_audit.csv")
    assert (audit.status == "exact_duplicate_removed").sum() == 1
    assert len(pd.read_csv(root / "data/corrupted_images.csv")) == 1
    relevant = result[result.path.str.endswith((first.name, "near.png"))]
    assert relevant.group.nunique() == 1
    assert relevant.split.nunique() == 1


def test_cross_label_conflict_quarantined(synthetic_data):
    root, frame = synthetic_data
    first = root / frame[frame.label == "cloudy"].iloc[0].path
    shutil.copy(first, root / "data/raw/rain/conflict.png")
    result = prepare_dataset(root / "data/raw", root / "data", root)
    assert first.relative_to(root).as_posix() not in set(result.path)
    audit = pd.read_csv(root / "data/image_audit.csv")
    assert (audit.status == "quarantined_label_conflict").sum() == 2


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt", "back\\slash.txt"])
def test_safe_zip_rejects_traversal(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        safe_extract(archive, tmp_path / "dest")


def test_zip_extract(tmp_path):
    archive = tmp_path / "ok.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("folder/ok.txt", "ok")
    safe_extract(archive, tmp_path / "dest")
    assert (tmp_path / "dest/folder/ok.txt").read_text() == "ok"


def test_eval_not_augmented():
    image = Image.new("RGB", (310, 200), (70, 90, 110))
    a = transform("test", True, "convnext")(image)
    b = transform("test", False, "convnext")(image)
    assert a.shape == (3, 224, 224)
    assert (a == b).all()


def test_portable_manifest_restores_raw_images(synthetic_data, monkeypatch):
    import sys
    import weather_cv
    from src.utils import sha256_file
    root, frame = synthetic_data
    archive = root / "dataset.zip"
    raw = root / "data/raw"
    original_hash = sha256_file(root / "data/manifest.csv")
    with zipfile.ZipFile(archive, "w") as z:
        for path in raw.rglob("*.png"):
            z.write(path, path.relative_to(raw))
    shutil.rmtree(raw)
    monkeypatch.setattr(sys, "argv", ["weather_cv.py", "--root", str(root), "prepare", "--local-zip", str(archive)])
    weather_cv.main()
    verify_content(frame, root)
    assert sha256_file(root / "data/manifest.csv") == original_hash
