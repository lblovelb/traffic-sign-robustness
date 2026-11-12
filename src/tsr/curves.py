"""Plot CNN learning curves (accuracy per epoch) from the runs logged in MLflow.

    python -m tsr.curves                     # full experiment
    python -m tsr.curves --quick             # quick experiment

Reads the per-epoch metrics that experiment.py logs (epoch_accuracy and
epoch_val_accuracy), so no re-training is needed. Writes
results/learning_curves.png (or results/quick/learning_curves.png).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# One curve per (training data, split). Seeds are averaged into mean ± std.
Histories = dict[str, dict[str, list[list[float]]]]


def fetch_histories(experiment_name: str) -> Histories:
    """Return {train_kind: {"train": [[acc per epoch], ...], "val": [...]}} for CNN runs."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise SystemExit(
            f"No MLflow experiment named {experiment_name!r}. Run the experiment first."
        )

    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="params.model = 'cnn'",
        order_by=["attributes.start_time DESC"],
    )

    histories: Histories = {}
    seen: set[tuple[str, str]] = set()
    for run in runs:  # newest first, so we keep only the latest run per (train, seed)
        key = (run.data.params["train"], run.data.params["seed"])
        if key in seen:
            continue
        seen.add(key)
        curves = histories.setdefault(key[0], {"train": [], "val": []})
        for split, metric in [("train", "epoch_accuracy"), ("val", "epoch_val_accuracy")]:
            points = sorted(
                client.get_metric_history(run.info.run_id, metric), key=lambda m: m.step
            )
            if points:
                curves[split].append([m.value for m in points])
    return histories


def plot_curves(histories: Histories, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = sorted(histories)
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.5 * len(kinds), 3.4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, kind in zip(axes, kinds, strict=True):
        for split, label in [("train", "training"), ("val", "validation")]:
            runs = histories[kind][split]
            if not runs:
                continue
            n = min(len(r) for r in runs)  # in case runs have different lengths
            values = np.array([r[:n] for r in runs])
            mean, std = values.mean(axis=0), values.std(axis=0)
            epochs = np.arange(1, n + 1)
            ax.plot(epochs, mean, marker="o", label=label)
            ax.fill_between(epochs, mean - std, mean + std, alpha=0.2)
        ax.set_title(f"CNN trained on {kind} images (n={len(histories[kind]['train'])} seeds)")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("accuracy")
    axes[-1].legend(loc="lower right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="use the quick-run experiment")
    args = parser.parse_args()

    name = "traffic-sign-robustness" + ("-quick" if args.quick else "")
    out = Path("results") / ("quick" if args.quick else "") / "learning_curves.png"
    plot_curves(fetch_histories(name), out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
