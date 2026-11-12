import numpy as np
import pandas as pd

from tsr import metrics
from tsr.curves import plot_curves
from tsr.experiment import FIELDS, corruption_seed, load_config
from tsr.report import summarize, write_report


def test_classification_metrics_perfect_and_wrong():
    y = np.array([0, 1, 2, 2])
    assert metrics.classification_metrics(y, y) == {"accuracy": 1.0, "macro_f1": 1.0}
    assert metrics.classification_metrics(y, (y + 1) % 3)["accuracy"] == 0.0


def test_relative_drop():
    assert metrics.relative_drop(0.9, 0.45) == 0.5
    assert metrics.relative_drop(0.0, 0.0) == 0.0


def test_corruption_seed_is_stable_and_distinct():
    assert corruption_seed(1234, "fog", 3) == corruption_seed(1234, "fog", 3)
    assert corruption_seed(1234, "fog", 3) != corruption_seed(1234, "fog", 4)
    assert corruption_seed(1234, "fog", 3) != corruption_seed(1234, "rain", 3)


def test_quick_config_overrides():
    cfg = load_config("configs/experiment.yaml", quick=True)
    assert cfg["seeds"] == cfg["quick"]["seeds"]
    assert cfg["cnn"]["epochs"] == cfg["quick"]["cnn_epochs"]
    assert cfg["results_dir"].replace("\\", "/").endswith("quick")
    # Unseen corruptions must exist, otherwise the seen/unseen split is meaningless.
    unseen = set(cfg["evaluation"]["corruptions"]) - set(cfg["augmentation"]["corruptions"])
    assert unseen


def _fake_results() -> pd.DataFrame:
    rows = []
    for seed in (0, 1):
        for model, train in [("cnn", "clean"), ("cnn", "augmented"), ("hog_svm", "clean")]:
            base = 0.97 if model == "cnn" else 0.9
            rows.append(
                dict(
                    seed=seed,
                    model=model,
                    train=train,
                    corruption="clean",
                    severity=0,
                    seen_in_training=False,
                    accuracy=base,
                    macro_f1=base,
                    latency_ms=0.1,
                    size_mb=1.0,
                )
            )
            for name in ("fog", "blur"):
                for sev in (1, 3, 5):
                    seen = train == "augmented" and name == "fog"
                    acc = base - 0.1 * sev * (0.3 if seen else 1.0) + 0.01 * seed
                    rows.append(
                        dict(
                            seed=seed,
                            model=model,
                            train=train,
                            corruption=name,
                            severity=sev,
                            seen_in_training=seen,
                            accuracy=acc,
                            macro_f1=acc,
                            latency_ms=0.1,
                            size_mb=1.0,
                        )
                    )
    df = pd.DataFrame(rows)
    assert list(df.columns) == FIELDS
    return df


def test_summarize_groups_by_model_and_training():
    table = summarize(_fake_results())
    assert len(table) == 3
    assert {"clean", "all_corrupted", "seen", "unseen"} <= set(table.columns)


def test_write_report_creates_files(tmp_path):
    csv = tmp_path / "results.csv"
    _fake_results().to_csv(csv, index=False)
    out = write_report(csv)
    assert out.exists() and (tmp_path / "robustness.png").exists()
    assert "Results summary" in out.read_text(encoding="utf-8")


def test_plot_curves_writes_png(tmp_path):
    histories = {
        "clean": {"train": [[0.5, 0.8, 0.9], [0.4, 0.7, 0.9]], "val": [[0.6, 0.8, 0.85]] * 2},
        "augmented": {"train": [[0.4, 0.7, 0.85]], "val": [[0.5, 0.7, 0.8]]},
    }
    out = tmp_path / "learning_curves.png"
    plot_curves(histories, out)
    assert out.exists() and out.stat().st_size > 0
