"""Run the full robustness experiment.

    python -m tsr.experiment                 # full run (configs/experiment.yaml)
    python -m tsr.experiment --quick         # small smoke test, a few minutes

Experimental conditions
-----------------------
* baseline   - trained on clean images, tested on clean images
* shift      - trained on clean images, tested on corrupted images
* mitigation - trained on partly corrupted images, tested on corrupted images;
               split into corruptions *seen* and *unseen* during training

Every (model, training data, seed) combination becomes one MLflow run, and
every evaluation becomes one row in results/results.csv.
"""

from __future__ import annotations

import argparse
import csv
import zlib
from pathlib import Path

import numpy as np
import yaml

from tsr import corruptions, data, metrics
from tsr.data import NUM_CLASSES

FIELDS = [
    "seed",
    "model",
    "train",
    "corruption",
    "severity",
    "seen_in_training",
    "accuracy",
    "macro_f1",
    "latency_ms",
    "size_mb",
]


def load_config(path: str | Path, quick: bool) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if quick:
        q = cfg["quick"]
        cfg["seeds"] = q["seeds"]
        cfg["cnn"]["epochs"] = q["cnn_epochs"]
        cfg["evaluation"]["severities"] = q["severities"]
        cfg["train_samples"] = q["train_samples"]
        cfg["test_samples"] = q["test_samples"]
        cfg["mlflow"]["experiment_name"] += "-quick"
        cfg["results_dir"] = str(Path(cfg["results_dir"]) / "quick")
    return cfg


def corruption_seed(base: int, name: str, severity: int) -> int:
    """Stable per-(corruption, severity) seed, identical for every model."""
    return base + zlib.crc32(f"{name}-{severity}".encode()) % 100_000


def _row(base: dict, corruption: str, severity: int, seen: bool, m: dict) -> dict:
    return {
        **base,
        "corruption": corruption,
        "severity": severity,
        "seen_in_training": seen,
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
    }


def make_model(name: str, cfg: dict, seed: int, input_shape: tuple[int, int, int]):
    from tsr.models import CNNClassifier, HogSvmClassifier

    if name == "cnn":
        return CNNClassifier(
            NUM_CLASSES,
            input_shape,
            seed,
            epochs=cfg["cnn"]["epochs"],
            batch_size=cfg["cnn"]["batch_size"],
        )
    if name == "hog_svm":
        return HogSvmClassifier(seed, C=cfg["hog_svm"]["C"])
    raise ValueError(f"unknown model {name!r}")


def run(cfg: dict) -> Path:
    import mlflow

    results_dir = Path(cfg["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "results.csv"

    d = data.load_gtsrb(cfg["data_dir"], cfg["image_size"])
    x_train, y_train = d["x_train"], d["y_train"]
    x_test, y_test = d["x_test"], d["y_test"]
    if "train_samples" in cfg:
        x_train, y_train = data.stratified_subsample(x_train, y_train, cfg["train_samples"], 0)
        x_test, y_test = data.stratified_subsample(x_test, y_test, cfg["test_samples"], 0)
    x_train, x_test = data.to_float(x_train), data.to_float(x_test)
    # GTSRB is stored sorted by class, and Keras' validation_split takes the
    # LAST 10% of the data. Without shuffling, the last classes end up only in
    # validation and are never trained on. Shuffle once with a fixed seed.
    order = np.random.default_rng(0).permutation(len(y_train))
    x_train, y_train = x_train[order], y_train[order]

    aug = cfg["augmentation"]
    ev = cfg["evaluation"]
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        for seed in cfg["seeds"]:
            x_aug = None
            for spec in cfg["runs"]:
                model_name, train_kind = spec["model"], spec["train"]
                if train_kind == "augmented":
                    if x_aug is None:
                        x_aug = corruptions.random_augment(
                            x_train, aug["corruptions"], aug["max_severity"], aug["prob"], seed
                        )
                    x_fit = x_aug
                else:
                    x_fit = x_train

                run_name = f"{model_name}-{train_kind}-seed{seed}"
                print(f"\n=== {run_name} ===")
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_params(
                        {
                            "model": model_name,
                            "train": train_kind,
                            "seed": seed,
                            "n_train": len(x_fit),
                            "image_size": cfg["image_size"],
                        }
                    )
                    if model_name == "cnn":
                        mlflow.log_params(
                            {"epochs": cfg["cnn"]["epochs"], "batch_size": cfg["cnn"]["batch_size"]}
                        )
                    if train_kind == "augmented":
                        mlflow.log_params(
                            {
                                "aug_corruptions": ",".join(aug["corruptions"]),
                                "aug_max_severity": aug["max_severity"],
                                "aug_prob": aug["prob"],
                            }
                        )

                    model = make_model(model_name, cfg, seed, x_fit.shape[1:]).fit(x_fit, y_train)

                    # Log Keras training history per epoch so MLflow can plot learning curves.
                    # HOG+SVM has no epochs, so it has no history and logs nothing here.
                    for metric_name, values in getattr(model, "history", {}).items():
                        for epoch, value in enumerate(values, start=1):
                            mlflow.log_metric(f"epoch_{metric_name}", value, step=epoch)
                    latency = metrics.latency_ms_per_image(model.predict, x_test)
                    size = model.size_mb()
                    mlflow.log_metrics({"latency_ms": latency, "size_mb": size})

                    base_row = {
                        "seed": seed,
                        "model": model_name,
                        "train": train_kind,
                        "latency_ms": latency,
                        "size_mb": size,
                    }
                    seen = set(aug["corruptions"]) if train_kind == "augmented" else set()

                    m = metrics.classification_metrics(y_test, model.predict(x_test))
                    writer.writerow(_row(base_row, "clean", 0, False, m))
                    mlflow.log_metrics(
                        {"clean_accuracy": m["accuracy"], "clean_macro_f1": m["macro_f1"]}
                    )
                    print(f"clean accuracy {m['accuracy']:.4f}")

                    corrupted_accs = []
                    for name in ev["corruptions"]:
                        for sev in ev["severities"]:
                            c_seed = corruption_seed(ev["corruption_seed"], name, sev)
                            x_c = corruptions.apply(x_test, name, sev, c_seed)
                            m = metrics.classification_metrics(y_test, model.predict(x_c))
                            writer.writerow(_row(base_row, name, sev, name in seen, m))
                            f.flush()
                            corrupted_accs.append(m["accuracy"])
                            mlflow.log_metric(f"{name}_s{sev}_accuracy", m["accuracy"])
                            print(f"{name:>15} s{sev}: {m['accuracy']:.4f}")
                    mlflow.log_metric("mean_corrupted_accuracy", float(np.mean(corrupted_accs)))

    print(f"\nWrote {out_csv}")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/experiment.yaml")
    parser.add_argument("--quick", action="store_true", help="small smoke-test run")
    args = parser.parse_args()
    cfg = load_config(args.config, args.quick)
    out_csv = run(cfg)

    from tsr.curves import fetch_histories, plot_curves
    from tsr.report import write_report

    write_report(out_csv)
    plot_curves(
        fetch_histories(cfg["mlflow"]["experiment_name"]),
        Path(cfg["results_dir"]) / "learning_curves.png",
    )


if __name__ == "__main__":
    main()
