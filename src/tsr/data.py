"""Download and cache the German Traffic Sign Recognition Benchmark (GTSRB).

43 classes, ~39k training and ~12.6k test images. Images are resized to a
fixed size and cached as a single .npz so later runs load in seconds.

If the download fails (network, mirror moved), download the three zip files
manually from the GTSRB benchmark site or Kaggle and put them in data/raw/.
"""

from __future__ import annotations

import csv
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

BASE_URL = "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"
FILES = {
    "train": "GTSRB-Training_fixed.zip",
    "test_images": "GTSRB_Final_Test_Images.zip",
    "test_labels": "GTSRB_Final_Test_GT.zip",
}
NUM_CLASSES = 43


def _download(raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES.values():
        target = raw_dir / name
        if target.exists():
            continue
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(BASE_URL + name, target)
    for name in FILES.values():
        with zipfile.ZipFile(raw_dir / name) as zf:
            zf.extractall(raw_dir)


def _load_image(path: Path, size: int) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)


def _load_train(raw_dir: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for class_dir in sorted(p for p in raw_dir.rglob("Training/*") if p.is_dir()):
        label = int(class_dir.name)
        for img_path in sorted(class_dir.glob("*.ppm")):
            xs.append(_load_image(img_path, size))
            ys.append(label)
    return np.stack(xs), np.array(ys, dtype=np.int64)


def _load_test(raw_dir: Path, size: int) -> tuple[np.ndarray, np.ndarray]:
    gt_file = next(raw_dir.rglob("GT-final_test.csv"))
    image_dir = next(raw_dir.rglob("Final_Test/Images"))
    xs, ys = [], []
    with open(gt_file, newline="") as f:
        for row in csv.DictReader(f, delimiter=";"):
            xs.append(_load_image(image_dir / row["Filename"], size))
            ys.append(int(row["ClassId"]))
    return np.stack(xs), np.array(ys, dtype=np.int64)


def load_gtsrb(data_dir: str | Path = "data", size: int = 32) -> dict[str, np.ndarray]:
    """Return {"x_train", "y_train", "x_test", "y_test"}; images uint8 (N, size, size, 3)."""
    data_dir = Path(data_dir)
    cache = data_dir / "processed" / f"gtsrb_{size}.npz"
    if cache.exists():
        with np.load(cache) as d:
            return {k: d[k] for k in d.files}

    raw_dir = data_dir / "raw"
    _download(raw_dir)
    x_train, y_train = _load_train(raw_dir, size)
    x_test, y_test = _load_test(raw_dir, size)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)
    print(f"Cached {len(x_train)} train / {len(x_test)} test images to {cache}")
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


def to_float(images: np.ndarray) -> np.ndarray:
    return images.astype(np.float32) / 255.0


def stratified_subsample(
    x: np.ndarray, y: np.ndarray, n: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Keep ~n examples with the original class balance (used by --quick)."""
    if n >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        k = max(1, round(n * len(idx) / len(y)))
        keep.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    keep = np.sort(np.concatenate(keep))
    return x[keep], y[keep]
