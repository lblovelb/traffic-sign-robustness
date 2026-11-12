"""Save example images so you can SEE the data and the corruptions.

    python -m tsr.preview

Writes two images to results/:
* examples_corruptions.png - 8 random test signs: clean + every corruption (severity 3)
* examples_severity.png    - one sign at severities 1-5 for every corruption
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

# "Agg" draws straight to image files without opening a window.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tsr import corruptions, data  # noqa: E402
from tsr.corruptions import CORRUPTIONS, SEVERITIES  # noqa: E402


def _show(ax, img: np.ndarray, title: str | None = None) -> None:
    # interpolation="nearest" keeps the 32x32 pixels sharp instead of smoothing them
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title, fontsize=8)


def corruption_grid(images: np.ndarray, labels: np.ndarray, out: Path, severity: int = 3) -> None:
    names = ["clean", *CORRUPTIONS]
    fig, axes = plt.subplots(len(names), len(images), figsize=(1.3 * len(images), 1.4 * len(names)))
    for row, name in enumerate(names):
        # seed=0 so the example images look the same every time the script runs.
        imgs = images if name == "clean" else corruptions.apply(images, name, severity, seed=0)
        for col in range(len(images)):
            _show(axes[row, col], imgs[col], f"class {labels[col]}" if row == 0 else None)
        axes[row, 0].set_ylabel(name, fontsize=8)
    fig.suptitle(f"Test images: clean and corrupted (severity {severity})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def severity_grid(image: np.ndarray, out: Path) -> None:
    batch = image[None]  # corruptions expect a batch: (1, H, W, 3)
    fig, axes = plt.subplots(
        len(CORRUPTIONS), len(SEVERITIES) + 1, figsize=(8, 1.5 * len(CORRUPTIONS))
    )
    for row, name in enumerate(CORRUPTIONS):
        _show(axes[row, 0], image, "clean" if row == 0 else None)
        axes[row, 0].set_ylabel(name, fontsize=8)
        for col, sev in enumerate(SEVERITIES, start=1):
            img = corruptions.apply(batch, name, sev, seed=0)[0]
            _show(axes[row, col], img, f"severity {sev}" if row == 0 else None)
    fig.suptitle("One sign, increasing severity", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--n", type=int, default=8, help="number of example signs")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    d = data.load_gtsrb(args.data_dir)
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(d["y_test"]), size=args.n, replace=False)
    images, labels = data.to_float(d["x_test"][idx]), d["y_test"][idx]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corruption_grid(images, labels, out_dir / "examples_corruptions.png")
    severity_grid(images[0], out_dir / "examples_severity.png")
    print(f"Wrote {out_dir / 'examples_corruptions.png'} and {out_dir / 'examples_severity.png'}")


if __name__ == "__main__":
    main()
