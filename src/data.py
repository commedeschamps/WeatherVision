"""Download, inspect, deduplicate, group, and split *before* any training."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import stat
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from .utils import CLASSES, CLASS_TO_IDX, ROOT, sha256_file, write_json

ALIASES = {"cloud": "cloudy", "cloudy": "cloudy", "rain": "rain",
           "shine": "shine", "sunrise": "sunrise"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
MIRROR_REPO = "PerceptiLabs/Weather-Analysis"
# Resolved once and pinned in provenance; future runs reuse the cached archive.
MIRROR_COMMIT = "ad78ac8fcf1b1aae229ca6e5310aa75b06783787"


def safe_extract(archive: Path, destination: Path) -> None:
    """Reject traversal, symlinks, oversized members, and ZIP bombs before writing."""
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as z:
        total = 0
        for member in z.infolist():
            name = PurePosixPath(member.filename)
            mode = member.external_attr >> 16
            total += member.file_size
            if (name.is_absolute() or ".." in name.parts or "\\" in member.filename
                    or stat.S_ISLNK(mode) or member.file_size > 100_000_000
                    or total > 2_000_000_000):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError("ZIP path escapes destination")
        destination.mkdir(parents=True, exist_ok=True)
        z.extractall(destination)


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "WeatherVision/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, part.open("wb") as f:
            shutil.copyfileobj(response, f)
        part.replace(path)
    finally:
        part.unlink(missing_ok=True)


def obtain_dataset(data_dir: Path, local_zip: Path | None = None,
                   source_dir: Path | None = None) -> Path:
    data_dir = Path(data_dir)
    raw = data_dir / "raw"
    provenance_path = data_dir / "source.json"
    if source_dir is not None:
        if not Path(source_dir).is_dir():
            raise ValueError(f"Dataset directory does not exist: {source_dir}")
        write_json(provenance_path, {"type": "local_directory", "source": str(Path(source_dir).resolve()),
                                    "original_doi": "10.17632/4drtyfjtfy.1"})
        return Path(source_dir).resolve()
    if raw.exists() and any(raw.rglob("*")):
        if local_zip:
            raise ValueError("Raw data already exists; use a new --data-dir for a different archive")
        return raw
    archive = data_dir / "downloads" / "weather.zip"
    if local_zip:
        archive = Path(local_zip).resolve()
        provenance = {"type": "manual_zip", "archive_name": archive.name}
    else:
        if archive.exists() and provenance_path.exists():
            provenance = json.loads(provenance_path.read_text())
            if sha256_file(archive) != provenance.get("archive_sha256"):
                raise ValueError("Cached archive hash differs from source.json")
        else:
            try:
                commit = MIRROR_COMMIT
                url = f"https://codeload.github.com/{MIRROR_REPO}/zip/{commit}"
                download(url, archive)
                provenance = {"type": "public_mirror", "repository": MIRROR_REPO,
                              "commit": commit, "url": url}
            except Exception as e:
                raise RuntimeError("Automatic download failed. Download the Mendeley or GitHub ZIP, "
                                   "then run prepare --local-zip /path/weather.zip. " + str(e)) from e
    provenance.update(archive_sha256=sha256_file(archive), original_doi="10.17632/4drtyfjtfy.1",
                      attribution="Gbeminiyi Ajayi (2018); CC BY 4.0",
                      mirror_note="PerceptiLabs mirror images are resized; not an asserted byte-identical original")
    # Failed extraction never leaves a partially populated raw/ cache.
    data_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=data_dir) as staging:
        safe_extract(archive, Path(staging) / "raw")
        shutil.move(str(Path(staging) / "raw"), raw)
    write_json(provenance_path, provenance)
    return raw


def infer_label(path: Path, source: Path) -> str | None:
    relative = path.relative_to(source)
    labels = {ALIASES[p.lower()] for p in relative.parts[:-1] if p.lower() in ALIASES}
    if len(labels) > 1:
        raise ValueError(f"Conflicting class directories: {relative}")
    if labels:
        return next(iter(labels))
    # Original archives can contain flat names such as cloudy001.jpg.
    stem = path.stem.lower()
    for prefix in sorted(ALIASES, key=len, reverse=True):
        suffix = stem[len(prefix):].lstrip("_- ")
        if stem.startswith(prefix) and suffix.isdigit():
            return ALIASES[prefix]
    return None


def open_rgb(path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def assign_splits(frame: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Approximate class-stratified 70/15/15 allocation of indivisible groups."""
    frame = frame.copy()
    if frame.groupby("group").label.nunique().max() > 1:
        raise ValueError("Conflicting labels within a duplicate group must be quarantined")
    rng = np.random.default_rng(seed)
    frame["split"] = "train"
    for label in CLASSES:
        sizes = frame[frame.label == label].groupby("group", sort=True).size()
        if len(sizes) < 3:
            raise ValueError(f"{label}: need at least 3 independent groups to split")
        groups = list(rng.permutation(sizes.index.to_numpy()))
        target = max(1, round(sizes.sum() * 0.15))
        for split, reserve in (("val", 2), ("test", 1)):
            chosen, count = [], 0
            # Whole groups are selected until closest to the image-count target.
            while len(groups) > reserve:
                group = groups[0]
                if chosen and abs(count - target) <= abs(count + sizes[group] - target):
                    break
                chosen.append(groups.pop(0))
                count += int(sizes[group])
            frame.loc[frame.group.isin(chosen), "split"] = split
    validate_manifest(frame, check_files=False)
    return frame


def validate_manifest(frame: pd.DataFrame, root: Path = ROOT, check_files=True) -> None:
    required = {"path", "label", "class_id", "split", "group", "sha256", "pixel_sha256"}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Manifest is empty or missing required columns")
    if frame[list(required)].isna().any().any():
        raise ValueError("Manifest contains missing identifiers")
    if set(frame.label) != set(CLASSES) or set(frame.split) != {"train", "val", "test"}:
        raise ValueError("All four classes and all three splits are required")
    if not (frame.label.map(CLASS_TO_IDX) == frame.class_id).all():
        raise ValueError("Class index mapping is inconsistent")
    if frame.path.duplicated().any():
        raise ValueError("Duplicate paths in manifest")
    for key in ("group", "sha256", "pixel_sha256"):
        if (frame.groupby(key).split.nunique() > 1).any():
            raise ValueError(f"Leakage across splits: {key}")
    if (frame.groupby("group").label.nunique() > 1).any():
        raise ValueError("Duplicate group contains conflicting labels")
    counts = pd.crosstab(frame.label, frame.split).reindex(index=CLASSES, columns=["train", "val", "test"], fill_value=0)
    if (counts == 0).any().any():
        raise ValueError("Every class must appear in each split")
    if check_files:
        for value in frame.path:
            path = (root / value).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f"Missing or unsafe manifest path: {value}")


def verify_content(frame: pd.DataFrame, root: Path = ROOT) -> None:
    validate_manifest(frame, root)
    for row in frame.itertuples():
        if sha256_file(root / row.path) != row.sha256:
            raise ValueError(f"Image content changed since preparation: {row.path}")


def prepare_dataset(source: Path, data_dir: Path, root: Path = ROOT, seed: int = 42) -> pd.DataFrame:
    from .plots import dataset_figures
    source, data_dir, root = Path(source).resolve(), Path(data_dir), Path(root).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    rows, corrupt, ignored, thumbnails = [], [], [], []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
            continue
        label = infer_label(path, source)
        if label is None:
            ignored.append(str(path.relative_to(source)))
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            image = open_rgb(path)
            width, height = image.size
            if min(width, height) < 16:
                raise ValueError("Image shorter side is below 16 pixels")
            pixels = hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest()
            rows.append({"path": path.relative_to(root).as_posix(), "label": label,
                         "class_id": CLASS_TO_IDX[label], "width": width, "height": height,
                         "sha256": sha256_file(path), "pixel_sha256": pixels,
                         "phash": str(imagehash.phash(image)), "dhash": str(imagehash.dhash(image))})
            thumbnails.append(np.asarray(image.resize((16, 16)), dtype=np.float32) / 255)
        except (OSError, ValueError, Image.DecompressionBombError) as e:
            corrupt.append({"path": str(path), "error": str(e)})
    if not rows:
        raise ValueError("No recognized, valid weather images found")
    frame = pd.DataFrame(rows)
    parent = list(range(len(frame)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(j)] = find(i)

    pairs = []
    for i in range(len(rows)):
        for j in range(i):
            a, b = rows[i], rows[j]
            exact = a["sha256"] == b["sha256"] or a["pixel_sha256"] == b["pixel_sha256"]
            p = (int(a["phash"], 16) ^ int(b["phash"], 16)).bit_count()
            d = (int(a["dhash"], 16) ^ int(b["dhash"], 16)).bit_count()
            aspect = abs(math.log((a["width"] / a["height"]) / (b["width"] / b["height"])))
            rmse = float(np.sqrt(np.mean((thumbnails[i] - thumbnails[j]) ** 2)))
            near = p <= 4 and d <= 4 and aspect <= 0.10 and rmse <= 0.12
            if exact or near:
                union(i, j)
                pairs.append({"path_a": a["path"], "path_b": b["path"], "label_a": a["label"],
                              "label_b": b["label"], "kind": "exact" if exact else "near",
                              "phash_distance": p, "dhash_distance": d, "rgb_rmse": rmse})
    group_ids = {}
    for i in range(len(rows)):
        group_ids.setdefault(find(i), []).append(rows[i]["pixel_sha256"])
    frame["group"] = [min(group_ids[find(i)])[:20] for i in range(len(rows))]
    conflicts = set(frame.groupby("group").filter(lambda x: x.label.nunique() > 1).group)
    frame["status"] = "retained"
    frame.loc[frame.group.isin(conflicts), "status"] = "quarantined_label_conflict"
    eligible = frame.status == "retained"
    repeated = frame.loc[eligible].duplicated("pixel_sha256")
    frame.loc[repeated[repeated].index, "status"] = "exact_duplicate_removed"
    manifest = assign_splits(frame[frame.status == "retained"].drop(columns="status"), seed)
    validate_manifest(manifest, root)
    frame.to_csv(data_dir / "image_audit.csv", index=False)
    pd.DataFrame(corrupt, columns=["path", "error"]).to_csv(data_dir / "corrupted_images.csv", index=False)
    pd.DataFrame(pairs, columns=["path_a", "path_b", "label_a", "label_b", "kind", "phash_distance", "dhash_distance", "rgb_rmse"]).to_csv(data_dir / "duplicate_pairs.csv", index=False)
    manifest.to_csv(data_dir / "manifest.csv", index=False)
    stats = {"seed": seed, "valid_images_before_dedup": len(frame), "corrupt_images": len(corrupt),
             "ignored_unrecognized_images": ignored, "retained_images": len(manifest),
             "exact_duplicates_removed": int((frame.status == "exact_duplicate_removed").sum()),
             "quarantined_conflicts": int((frame.status == "quarantined_label_conflict").sum()),
             "duplicate_pairs": len(pairs), "near_duplicate_pairs": sum(p["kind"] == "near" for p in pairs),
             "independent_groups": int(manifest.group.nunique()),
             "counts": pd.crosstab(manifest.label, manifest.split).to_dict(),
             "manifest_sha256": sha256_file(data_dir / "manifest.csv"),
             "near_duplicate_rule": "pHash<=4 AND dHash<=4 AND abs(log(aspect ratio ratio))<=0.10 AND RGB thumbnail RMSE<=0.12",
             "limitations": "Detected duplicate components cannot cross splits. Conservative hashes can miss transformed copies; manually review duplicate_pairs.csv and inspection figures. Mirror may differ from original dataset."}
    # Convert pandas/numpy scalars to portable JSON scalars.
    write_json(data_dir / "statistics.json", json.loads(json.dumps(stats, default=int)))
    dataset_figures(manifest, root, data_dir)
    return manifest


def resize_size(architecture: str) -> int:
    return 236 if architecture == "convnext" else 256


def transform(split="test", augment=False, architecture="cnn"):
    operations = [T.Resize(resize_size(architecture), interpolation=InterpolationMode.BILINEAR),
                  T.CenterCrop(224)]
    if split == "train" and augment:
        operations.extend([T.RandomHorizontalFlip(), T.RandomRotation(8),
                           T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10)])
    operations.extend([T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return T.Compose(operations)


class WeatherDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root=ROOT, split="test", augment=False, architecture="cnn"):
        self.frame = frame[frame.split == split].reset_index(drop=True)
        self.root = Path(root)
        self.transform = transform(split, augment, architecture)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        return self.transform(open_rgb(self.root / row.path)), int(row.class_id)
